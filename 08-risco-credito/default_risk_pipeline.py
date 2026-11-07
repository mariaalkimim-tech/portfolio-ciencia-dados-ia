#!/usr/bin/env python3
"""Benchmark de risco de inadimplência UCI com validação e auditoria por grupos.

O modelo prevê Y (default no mês seguinte). X2, sexo, é removida da lista de
features e mantida apenas para uma auditoria descritiva pós-teste. Não é score
regulatório nem modelo pronto para decisão de crédito.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, precision_recall_curve, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

LOGGER = logging.getLogger("default_risk")
SEED = 42


def load_data() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    from ucimlrepo import fetch_ucirepo
    dataset = fetch_ucirepo(id=350)
    X = dataset.data.features.copy()
    target = dataset.data.targets
    if target is None or "Y" not in target:
        raise ValueError("UCI target Y não encontrado.")
    y = pd.to_numeric(target["Y"], errors="raise").astype(int)
    if "X2" not in X:
        raise ValueError("Coluna X2 para auditoria descritiva não encontrada.")
    audit_group = X["X2"].copy().reset_index(drop=True)
    X = X.drop(columns=["X2"]).reset_index(drop=True)
    y = y.reset_index(drop=True)
    if set(y.unique()) != {0, 1}:
        raise ValueError("Esperado target binário {0,1}.")
    return X, y, audit_group


def split_data(X: pd.DataFrame, y: pd.Series, audit: pd.Series) -> dict[str, tuple[pd.DataFrame, pd.Series, pd.Series]]:
    x_dev, x_test, y_dev, y_test, a_dev, a_test = train_test_split(
        X, y, audit, test_size=0.25, random_state=SEED, stratify=y
    )
    x_train, x_val, y_train, y_val, a_train, a_val = train_test_split(
        x_dev, y_dev, a_dev, test_size=1 / 3, random_state=SEED, stratify=y_dev
    )
    return {"train": (x_train, y_train, a_train), "validation": (x_val, y_val, a_val), "test": (x_test, y_test, a_test)}


def make_pipeline(X: pd.DataFrame, estimator: Any) -> Pipeline:
    categorical = [c for c in ["X3", "X4"] if c in X]
    numeric = [c for c in X.columns if c not in categorical]
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical))
    return Pipeline([("preprocess", ColumnTransformer(transformers)), ("model", estimator)])


def capacities(y: pd.Series, score: np.ndarray) -> list[dict[str, float | int]]:
    yv = np.asarray(y, dtype=int)
    order = np.argsort(-score, kind="stable")
    result = []
    for fraction in (0.05, 0.10, 0.20):
        k = int(np.ceil(len(yv) * fraction))
        tp = int(yv[order[:k]].sum())
        result.append({"review_fraction": fraction, "n_reviewed": k, "defaults_in_top_k": tp, "precision_at_k": tp / k, "recall_at_k": tp / max(int(yv.sum()), 1)})
    return result


def group_audit(y: pd.Series, predictions: np.ndarray, groups: pd.Series) -> dict[str, Any]:
    out: dict[str, Any] = {}
    yv, pred, gv = np.asarray(y), np.asarray(predictions), np.asarray(groups)
    for group in sorted(pd.unique(gv), key=str):
        mask = gv == group
        positives = yv[mask] == 1
        negatives = ~positives
        out[str(group)] = {
            "rows": int(mask.sum()),
            "default_rate": float(yv[mask].mean()),
            "recall_at_threshold_0_5_descriptive": float(((pred[mask] == 1) & positives).sum() / max(int(positives.sum()), 1)),
            "false_positive_rate_at_threshold_0_5_descriptive": float(((pred[mask] == 1) & negatives).sum() / max(int(negatives.sum()), 1)),
        }
    return out


def score_metrics(y: pd.Series, scores: np.ndarray) -> dict[str, Any]:
    yv = np.asarray(y, dtype=int)
    return {
        "n": int(len(yv)), "defaults": int(yv.sum()), "default_rate": float(yv.mean()),
        "average_precision": float(average_precision_score(yv, scores)),
        "roc_auc_secondary": float(roc_auc_score(yv, scores)),
        "brier_score": float(brier_score_loss(yv, scores)),
        "log_loss": float(log_loss(yv, scores, labels=[0, 1])),
        "capacity": capacities(y, scores),
    }


def save_plots(y: pd.Series, scores: np.ndarray, out: Path) -> None:
    precision, recall, _ = precision_recall_curve(y, scores)
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(recall, precision, color="#174F76", lw=2)
    ax.axhline(float(y.mean()), color="#777", ls="--", label=f"Prevalência ({y.mean():.1%})")
    ax.set(title="Risco de default — Precisão–Recall no teste", xlabel="Recall", ylabel="Precisão", xlim=(0, 1), ylim=(0, 1)); ax.grid(alpha=.2); ax.legend(frameon=False)
    fig.savefig(out / "precision_recall_test.png", dpi=170); plt.close(fig)
    from sklearn.calibration import CalibrationDisplay
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    CalibrationDisplay.from_predictions(y, scores, n_bins=10, strategy="quantile", ax=ax, name="Modelo")
    ax.set_title("Risco de default — calibração no teste")
    fig.savefig(out / "calibration_test.png", dpi=170); plt.close(fig)


def run(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    X, y, audit = load_data()
    splits = split_data(X, y, audit)
    candidates = {
        "logistic_regression": make_pipeline(X, LogisticRegression(max_iter=2500, random_state=SEED)),
        "hist_gradient_boosting": make_pipeline(X, HistGradientBoostingClassifier(max_iter=160, max_leaf_nodes=15, l2_regularization=1.0, random_state=SEED)),
    }
    val_scores = {}
    val_metrics = {}
    for name, model in candidates.items():
        model.fit(splits["train"][0], splits["train"][1])
        score = model.predict_proba(splits["validation"][0])[:, 1]
        val_scores[name] = score
        val_metrics[name] = score_metrics(splits["validation"][1], score)
    selected_name = max(val_metrics, key=lambda name: val_metrics[name]["average_precision"])
    devX = pd.concat([splits["train"][0], splits["validation"][0]])
    devy = pd.concat([splits["train"][1], splits["validation"][1]])
    selected = make_pipeline(X, candidates[selected_name].named_steps["model"])
    selected.fit(devX, devy)
    y_test, groups_test = splits["test"][1], splits["test"][2]
    scores = selected.predict_proba(splits["test"][0])[:, 1]
    test_metrics = score_metrics(y_test, scores)
    # Descriptive only: threshold is not an approved credit policy.
    predictions = (scores >= 0.5).astype(int)
    audit_result = group_audit(y_test, predictions, groups_test)
    save_plots(y_test, scores, output_dir)
    joblib.dump(selected, output_dir / "default_risk_model.joblib")
    report = {
        "dataset": "UCI Default of Credit Card Clients (id 350)",
        "source_population": "Taiwan credit-card clients; payment history from April-September 2005; target next-month default",
        "target": "Y", "sensitive_audit_field_excluded_from_features": "X2",
        "features": list(X.columns), "split": "stratified 50/25/25 train-validation-test; fixed seed 42",
        "validation_metrics": val_metrics, "selected_model": selected_name,
        "test_metrics": test_metrics, "sex_code_group_audit_descriptive": audit_result,
        "limitations": [
            "Historical Taiwan credit-card portfolio; not representative of current or other-country portfolios.",
            "Threshold 0.5 is only descriptive and not a credit decision rule.",
            "Sex is excluded from predictors but reported for descriptive group audit; this is not a fairness or legal-compliance certification.",
            "Dataset lacks loss given default, recovery, collection and funding costs; expected loss and business ROI are not estimated.",
            "Public retrospective test results do not establish production or regulatory suitability.",
        ],
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = run(args.output_dir)
    print(json.dumps({"selected_model": result["selected_model"], "test_metrics": result["test_metrics"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
