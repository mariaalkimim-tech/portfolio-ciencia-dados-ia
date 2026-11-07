#!/usr/bin/env python3
"""Pipeline de classificação de falha para benchmark industrial UCI AI4I 2020.

O dataset é sintético. A ordem original dos registros é preservada para separar
60% desenvolvimento, 20% validação e 20% teste; a ordem não é usada como feature.
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

LOGGER = logging.getLogger("ai4i_failure")
SEED = 42
TARGET = "Machine failure"


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Fetch AI4I 2020 from UCI; do not use failure-mode labels as predictors."""
    from ucimlrepo import fetch_ucirepo

    dataset = fetch_ucirepo(id=601)
    features = dataset.data.features.copy()
    targets = dataset.data.targets.copy()
    if TARGET not in targets:
        raise ValueError(f"Alvo {TARGET!r} ausente: {list(targets.columns)}")
    y = pd.to_numeric(targets[TARGET], errors="raise").astype(int)
    if not set(y.unique()).issubset({0, 1}) or len(y.unique()) != 2:
        raise ValueError("O alvo deve conter as duas classes binárias 0 e 1.")
    if len(features) != len(y):
        raise ValueError("Features e target têm quantidades diferentes de linhas.")
    # UCI AI4I labels also include TWF/HDF/PWF/OSF/RNF; they are intentionally
    # not predictors because they directly compose the Machine failure target.
    features = features.drop(columns=[c for c in ["UDI", "Product ID"] if c in features], errors="ignore")
    return features.reset_index(drop=True), y.reset_index(drop=True)


def chronological_indices(n_rows: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return order-preserving 60/20/20 positional splits."""
    first, second = int(0.60 * n_rows), int(0.80 * n_rows)
    return np.arange(first), np.arange(first, second), np.arange(second, n_rows)


def make_pipeline(x: pd.DataFrame, estimator: Any) -> Pipeline:
    numeric = x.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in x.columns if c not in numeric]
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric:
        transformers.append(("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical))
    preprocess = ColumnTransformer(transformers=transformers, remainder="drop")
    return Pipeline([("preprocess", preprocess), ("model", estimator)])


def capacity_metrics(y_true: pd.Series, scores: np.ndarray, fractions: tuple[float, ...] = (0.01, 0.05, 0.10)) -> list[dict[str, float | int]]:
    y = np.asarray(y_true, dtype=int)
    order = np.argsort(-scores, kind="stable")
    output = []
    for fraction in fractions:
        k = max(1, int(np.ceil(len(y) * fraction)))
        tp = int(y[order[:k]].sum())
        output.append({
            "review_capacity_fraction": fraction,
            "alerts": k,
            "failures_captured": tp,
            "precision_at_capacity": tp / k,
            "recall_at_capacity": tp / max(int(y.sum()), 1),
        })
    return output


def metrics(y: pd.Series, scores: np.ndarray) -> dict[str, Any]:
    y_array = np.asarray(y, dtype=int)
    return {
        "rows": int(len(y_array)),
        "failures": int(y_array.sum()),
        "failure_rate": float(y_array.mean()),
        "average_precision": float(average_precision_score(y_array, scores)),
        "roc_auc_secondary": float(roc_auc_score(y_array, scores)),
        "brier_score": float(brier_score_loss(y_array, scores)),
        "capacity": capacity_metrics(y, scores),
    }


def save_figures(y: pd.Series, scores: np.ndarray, out: Path) -> None:
    precision, recall, _ = precision_recall_curve(y, scores)
    prevalence = float(y.mean())
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(recall, precision, lw=2.2, color="#185A7D", label="Modelo selecionado")
    ax.axhline(prevalence, color="#777777", ls="--", label=f"Prevalência ({prevalence:.2%})")
    ax.set(title="AI4I — Precisão–Recall no teste cronológico", xlabel="Recall", ylabel="Precisão", xlim=(0, 1), ylim=(0, 1))
    ax.grid(alpha=.2); ax.legend(frameon=False)
    fig.savefig(out / "precision_recall_test.png", dpi=170); plt.close(fig)

    caps = capacity_metrics(y, scores, tuple(np.linspace(.01, .20, 40)))
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot([100 * x["review_capacity_fraction"] for x in caps], [100 * x["recall_at_capacity"] for x in caps], color="#0B7A65", lw=2.2)
    ax.set(title="AI4I — falhas capturadas versus capacidade de inspeção", xlabel="Produção encaminhada para revisão (%)", ylabel="Falhas capturadas (%)")
    ax.grid(alpha=.2)
    fig.savefig(out / "capture_vs_capacity.png", dpi=170); plt.close(fig)


def run(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    X, y = load_dataset()
    train_idx, val_idx, test_idx = chronological_indices(len(X))
    for label, idx in [("train", train_idx), ("validation", val_idx), ("test", test_idx)]:
        if y.iloc[idx].nunique() < 2:
            raise ValueError(f"Split {label} não contém ambas as classes; não será embaralhado silenciosamente.")
    x_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    x_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    candidates = {
        "logistic_regression_balanced": make_pipeline(
            X, LogisticRegression(class_weight="balanced", max_iter=2500, random_state=SEED)
        ),
        "random_forest_balanced": make_pipeline(
            X, RandomForestClassifier(n_estimators=350, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1, random_state=SEED)
        ),
    }
    val_results: dict[str, dict[str, Any]] = {}
    for name, model in candidates.items():
        LOGGER.info("Ajustando %s", name)
        model.fit(x_train, y_train)
        val_results[name] = metrics(y_val, model.predict_proba(x_val)[:, 1])
    selected_name = max(val_results, key=lambda name: val_results[name]["average_precision"])

    dev_idx = np.concatenate([train_idx, val_idx])
    selected = candidates[selected_name]
    selected.fit(X.iloc[dev_idx], y.iloc[dev_idx])
    y_test = y.iloc[test_idx]
    test_scores = selected.predict_proba(X.iloc[test_idx])[:, 1]
    test_results = metrics(y_test, test_scores)
    save_figures(y_test, test_scores, output_dir)
    joblib.dump(selected, output_dir / "failure_model.joblib")
    report = {
        "dataset": "UCI AI4I 2020 Predictive Maintenance Dataset (id 601)",
        "dataset_nature": "synthetic benchmark; not field evidence",
        "target": TARGET,
        "features": list(X.columns),
        "excluded_leakage_columns": ["TWF", "HDF", "PWF", "OSF", "RNF", "UDI", "Product ID"],
        "split": "original row order, 60% train / 20% validation / 20% test; order is not a predictor",
        "split_counts": {name: {"rows": int(len(idx)), "failures": int(y.iloc[idx].sum())} for name, idx in [("train", train_idx), ("validation", val_idx), ("test", test_idx)]},
        "validation_metrics": val_results,
        "selected_model": selected_name,
        "test_metrics": test_results,
        "limitations": [
            "Synthetic observations and failure mechanisms; performance cannot be generalized to real factories.",
            "A row is not necessarily a full asset lifecycle, and no costs, maintenance actions or repair outcomes are supplied.",
            "Capacity metrics measure ranking only; they do not establish saved downtime or economic benefit.",
        ],
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")
    result = run(args.output_dir)
    print(json.dumps({"selected_model": result["selected_model"], "test_metrics": result["test_metrics"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
