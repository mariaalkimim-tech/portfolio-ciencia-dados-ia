#!/usr/bin/env python3
"""Benchmark de resposta a campanha bancária sem vazamento pós-contato.

A saída é propensão observacional à subscrição. Não estima efeito incremental
causal nem retorno de campanha. 'duration' e 'campaign' são excluídas porque
não estão seguramente disponíveis antes do contato atual.
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
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

LOGGER = logging.getLogger("bank_campaign_response")
SEED = 42


def prepare_features(X: pd.DataFrame) -> pd.DataFrame:
    """Keep only predictors safely available before the current contact."""
    return X.drop(columns=[c for c in ["duration", "campaign"] if c in X], errors="ignore").copy()


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    from ucimlrepo import fetch_ucirepo
    dataset = fetch_ucirepo(id=222)
    X = dataset.data.features.copy()
    y_df = dataset.data.targets
    if y_df is None or "y" not in y_df:
        raise ValueError("Target y não encontrado no dataset UCI.")
    target = y_df["y"].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
    if target.isna().any():
        raise ValueError("Target tem valores além de yes/no.")
    X = prepare_features(X)
    return X.reset_index(drop=True), target.astype(int).reset_index(drop=True)


def split_data(X: pd.DataFrame, y: pd.Series) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    x_dev, x_test, y_dev, y_test = train_test_split(X, y, test_size=.25, stratify=y, random_state=SEED)
    x_train, x_val, y_train, y_val = train_test_split(x_dev, y_dev, test_size=1/3, stratify=y_dev, random_state=SEED)
    return {"train": (x_train, y_train), "validation": (x_val, y_val), "test": (x_test, y_test)}


def make_pipeline(X: pd.DataFrame, estimator: Any) -> Pipeline:
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    pre = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical),
    ])
    return Pipeline([("preprocess", pre), ("model", estimator)])


def evaluate(y: pd.Series, score: np.ndarray) -> dict[str, Any]:
    yv = np.asarray(y, dtype=int)
    order = np.argsort(-score, kind="stable")
    capacity = []
    for fraction in (.01, .05, .10, .20):
        k = int(np.ceil(len(yv) * fraction))
        yes = int(yv[order[:k]].sum())
        capacity.append({"contact_capacity_fraction": fraction, "contacts": k, "observed_subscriptions_in_top_k": yes, "response_rate_at_k": yes / k, "share_of_all_subscriptions_in_top_k": yes / max(int(yv.sum()), 1)})
    return {
        "n": int(len(yv)), "subscriptions": int(yv.sum()), "observed_response_rate": float(yv.mean()),
        "average_precision": float(average_precision_score(yv, score)),
        "roc_auc_secondary": float(roc_auc_score(yv, score)),
        "brier_score": float(brier_score_loss(yv, score)), "capacity": capacity,
    }


def save_figures(y: pd.Series, score: np.ndarray, output_dir: Path) -> None:
    p, r, _ = precision_recall_curve(y, score)
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(r, p, color="#184F79", lw=2)
    ax.axhline(float(y.mean()), color="#777", ls="--", label=f"Response rate base ({y.mean():.1%})")
    ax.set(title="Bank Marketing — Precisão–Recall no teste", xlabel="Recall", ylabel="Precisão", xlim=(0, 1), ylim=(0, 1)); ax.grid(alpha=.2); ax.legend(frameon=False)
    fig.savefig(output_dir / "precision_recall_test.png", dpi=170); plt.close(fig)
    order = np.argsort(-score, kind="stable")
    fractions = np.linspace(.01, .5, 50)
    ks = np.maximum(1, np.ceil(len(y) * fractions).astype(int))
    rates = [np.asarray(y)[order[:k]].mean() for k in ks]
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(fractions * 100, np.asarray(rates) * 100, color="#0B7A65", lw=2)
    ax.axhline(float(y.mean()) * 100, color="#777", ls="--", label="Response rate base")
    ax.set(title="Resposta observada por capacidade de contato", xlabel="Clientes priorizados (%)", ylabel="Subscrição observada no grupo (%)"); ax.grid(alpha=.2); ax.legend(frameon=False)
    fig.savefig(output_dir / "response_by_capacity.png", dpi=170); plt.close(fig)


def run(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    X, y = load_data()
    splits = split_data(X, y)
    candidates = {
        "logistic_regression": make_pipeline(X, LogisticRegression(max_iter=2500, random_state=SEED)),
        "hist_gradient_boosting": make_pipeline(X, HistGradientBoostingClassifier(max_iter=160, max_leaf_nodes=20, l2_regularization=1.0, random_state=SEED)),
    }
    validation = {}
    for name, model in candidates.items():
        model.fit(*splits["train"])
        validation[name] = evaluate(splits["validation"][1], model.predict_proba(splits["validation"][0])[:, 1])
    selected_name = max(validation, key=lambda name: validation[name]["average_precision"])
    development_x = pd.concat([splits["train"][0], splits["validation"][0]])
    development_y = pd.concat([splits["train"][1], splits["validation"][1]])
    selected = make_pipeline(X, candidates[selected_name].named_steps["model"])
    selected.fit(development_x, development_y)
    y_test = splits["test"][1]
    scores = selected.predict_proba(splits["test"][0])[:, 1]
    test_metrics = evaluate(y_test, scores)
    save_figures(y_test, scores, output_dir)
    joblib.dump(selected, output_dir / "campaign_response_model.joblib")
    report = {
        "dataset": "UCI Bank Marketing (id 222)",
        "scope": "propensity to subscribe to a term deposit after a telephone campaign",
        "target": "y", "excluded_post_contact_fields": ["duration", "campaign"],
        "features": list(X.columns), "split": "stratified 50/25/25 train-validation-test; seed 42",
        "validation_metrics": validation, "selected_model": selected_name, "test_metrics": test_metrics,
        "limitations": [
            "This is an observational campaign dataset; response propensity is not incremental treatment effect.",
            "No campaign contact cost or net present value is supplied; ROI is not estimated.",
            "The data represent a Portuguese bank campaign from 2008-2010 and should not be generalized to current customers without validation.",
            "Duration and campaign were omitted to avoid features not reliably available before the current contact.",
        ],
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    report = run(args.output_dir)
    print(json.dumps({"selected_model": report["selected_model"], "test_metrics": report["test_metrics"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
