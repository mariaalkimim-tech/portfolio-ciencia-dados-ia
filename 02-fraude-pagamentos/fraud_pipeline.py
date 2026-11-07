#!/usr/bin/env python3
"""Pipeline reproduzível para benchmark público de fraude em cartões.

Uso:
  python fraud_pipeline.py
  python fraud_pipeline.py --csv data/creditcard.csv

Sem --csv, tenta carregar o dataset OpenML 42175 (requer internet na primeira execução).
As partições são cronológicas: 60% treino, 20% validação, 20% teste.
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
from sklearn.compose import make_column_transformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

LOGGER = logging.getLogger("fraud_pipeline")
TARGET = "Class"
TIME_COLUMN = "Time"
SEED = 42


def load_data(csv_path: Path | None) -> pd.DataFrame:
    """Load a local CSV or fetch the published OpenML dataset."""
    if csv_path is not None:
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV não encontrado: {csv_path}")
        frame = pd.read_csv(csv_path)
    else:
        try:
            from sklearn.datasets import fetch_openml
        except ImportError as exc:
            raise RuntimeError("Instale scikit-learn para baixar o dataset OpenML.") from exc
        LOGGER.info("Baixando dataset OpenML 42175; isso requer internet na primeira execução.")
        bunch = fetch_openml(data_id=42175, as_frame=True, parser="auto")
        frame = bunch.frame.copy()
        if TARGET not in frame.columns and bunch.target is not None:
            frame[TARGET] = bunch.target
    frame.columns = [str(c).strip() for c in frame.columns]
    if TARGET not in frame.columns:
        raise ValueError(f"Coluna-alvo '{TARGET}' ausente. Colunas: {list(frame.columns)}")
    if TIME_COLUMN not in frame.columns:
        raise ValueError(f"Coluna '{TIME_COLUMN}' necessária para o split temporal.")
    return frame


def prepare_data(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Validate schema and return numeric features and binary labels."""
    data = frame.copy()
    data[TARGET] = pd.to_numeric(data[TARGET], errors="coerce")
    data[TIME_COLUMN] = pd.to_numeric(data[TIME_COLUMN], errors="coerce")
    data = data.dropna(subset=[TARGET, TIME_COLUMN])
    labels = data[TARGET].astype(int)
    unique = set(labels.unique())
    if not unique.issubset({0, 1}) or unique != {0, 1}:
        raise ValueError(f"Esperado alvo binário 0/1 com ambas classes; encontrado {unique}")
    X = data.drop(columns=[TARGET]).copy()
    # This benchmark has numeric PCA components, transaction amount and elapsed time.
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    if X.select_dtypes(exclude=[np.number]).shape[1]:
        raise ValueError("Há atributos não numéricos após conversão; revise o esquema da base.")
    if X.isna().all().any():
        bad = X.columns[X.isna().all()].tolist()
        raise ValueError(f"Atributos inteiramente ausentes: {bad}")
    # Stable ordering preserves input order for tied timestamps.
    ordered = data.assign(_row_order=np.arange(len(data))).sort_values(
        [TIME_COLUMN, "_row_order"], kind="mergesort"
    )
    X = X.loc[ordered.index]
    labels = labels.loc[ordered.index]
    return X.reset_index(drop=True), labels.reset_index(drop=True)


def chronological_split(X: pd.DataFrame, y: pd.Series) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    """Split ordered observations without shuffling or stratification."""
    n = len(X)
    i_train, i_val = int(n * 0.60), int(n * 0.80)
    partitions = {
        "train": (X.iloc[:i_train], y.iloc[:i_train]),
        "validation": (X.iloc[i_train:i_val], y.iloc[i_train:i_val]),
        "test": (X.iloc[i_val:], y.iloc[i_val:]),
    }
    for name, (_, target) in partitions.items():
        positives = int(target.sum())
        if positives == 0 or positives == len(target):
            raise ValueError(
                f"Partição {name} não contém as duas classes ({positives} positivos). "
                "Revise as fronteiras temporais e documente uma alternativa sem embaralhar o teste."
            )
    return partitions


def build_candidates(feature_names: list[str]) -> dict[str, Pipeline]:
    """Return a transparent linear baseline and a nonlinear tree baseline."""
    linear = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    solver="lbfgs",
                    max_iter=2000,
                    random_state=SEED,
                ),
            ),
        ]
    )
    tree = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingClassifier(
                    max_iter=150,
                    learning_rate=0.08,
                    max_leaf_nodes=15,
                    l2_regularization=1.0,
                    random_state=SEED,
                ),
            ),
        ]
    )
    return {"logistic_regression_balanced": linear, "hist_gradient_boosting": tree}


def top_fraction_metrics(y_true: pd.Series, scores: np.ndarray, fractions: list[float]) -> list[dict[str, float]]:
    """Measure alert quality when only a fraction of transactions can be reviewed."""
    y = np.asarray(y_true, dtype=int)
    n_alerts_total = len(y)
    order = np.argsort(-scores, kind="stable")
    results = []
    for fraction in fractions:
        k = max(1, int(np.ceil(n_alerts_total * fraction)))
        selected = order[:k]
        tp = int(y[selected].sum())
        results.append(
            {
                "capacity_fraction": float(fraction),
                "alerts": int(k),
                "true_positives": tp,
                "precision_at_capacity": float(tp / k),
                "recall_at_capacity": float(tp / max(int(y.sum()), 1)),
            }
        )
    return results


def evaluate(y_true: pd.Series, scores: np.ndarray) -> dict[str, Any]:
    """Calculate ranking and calibration measures; 0.5 threshold is descriptive only."""
    y = np.asarray(y_true, dtype=int)
    pred = (scores >= 0.5).astype(int)
    return {
        "n": int(len(y)),
        "positive_count": int(y.sum()),
        "prevalence": float(y.mean()),
        "average_precision_pr_auc": float(average_precision_score(y, scores)),
        "roc_auc_secondary": float(roc_auc_score(y, scores)),
        "brier_score": float(brier_score_loss(y, scores)),
        "precision_at_threshold_0_5_descriptive": float(precision_score(y, pred, zero_division=0)),
        "recall_at_threshold_0_5_descriptive": float(recall_score(y, pred, zero_division=0)),
        "confusion_matrix_threshold_0_5": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        "capacity_metrics": top_fraction_metrics(pd.Series(y), scores, [0.001, 0.005, 0.01]),
    }


def save_plots(y_true: pd.Series, scores: np.ndarray, output_dir: Path, model_name: str) -> None:
    """Save a PR curve and capacity-versus-captured-fraud chart from held-out test."""
    y = np.asarray(y_true, dtype=int)
    precision, recall, _ = precision_recall_curve(y, scores)
    prevalence = float(y.mean())
    fig, ax = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    ax.plot(recall, precision, color="#175A8A", linewidth=2.4, label="Modelo")
    ax.axhline(prevalence, color="#888888", linestyle="--", linewidth=1.4, label=f"Baseline prevalência ({prevalence:.3%})")
    ax.set(title="Curva Precisão–Revocação — teste temporal", xlabel="Recall", ylabel="Precisão", xlim=(0, 1), ylim=(0, 1))
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.savefig(output_dir / "precision_recall_test.png", dpi=180)
    plt.close(fig)

    capacities = np.linspace(0.001, 0.05, 50)
    metrics = top_fraction_metrics(pd.Series(y), scores, capacities.tolist())
    fig, ax = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    ax.plot([m["capacity_fraction"] * 100 for m in metrics], [m["recall_at_capacity"] * 100 for m in metrics], color="#0A7A63", linewidth=2.4)
    ax.set(title="Fraudes capturadas por capacidade de revisão — teste temporal", xlabel="Transações encaminhadas para revisão (%)", ylabel="Fraudes capturadas (%)")
    ax.grid(alpha=0.2)
    fig.savefig(output_dir / "capture_vs_review_capacity.png", dpi=180)
    plt.close(fig)


def run(csv_path: Path | None, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = load_data(csv_path)
    X, y = prepare_data(raw)
    splits = chronological_split(X, y)
    x_train, y_train = splits["train"]
    x_val, y_val = splits["validation"]
    x_test, y_test = splits["test"]
    split_summary = {
        name: {
            "rows": int(len(part[0])),
            "frauds": int(part[1].sum()),
            "fraud_rate": float(part[1].mean()),
            "time_min": float(part[0][TIME_COLUMN].min()),
            "time_max": float(part[0][TIME_COLUMN].max()),
        }
        for name, part in splits.items()
    }
    candidates = build_candidates(list(X.columns))
    validation_scores: dict[str, np.ndarray] = {}
    validation_metrics: dict[str, dict[str, Any]] = {}
    for name, pipeline in candidates.items():
        LOGGER.info("Treinando %s", name)
        pipeline.fit(x_train, y_train)
        scores = pipeline.predict_proba(x_val)[:, 1]
        validation_scores[name] = scores
        validation_metrics[name] = evaluate(y_val, scores)
    selected_name = max(validation_metrics, key=lambda name: validation_metrics[name]["average_precision_pr_auc"])
    LOGGER.info("Selecionado pelo PR-AUC de validação: %s", selected_name)

    # Refit the selected pipeline on train+validation only after model choice.
    x_development = pd.concat([x_train, x_val], ignore_index=True)
    y_development = pd.concat([y_train, y_val], ignore_index=True)
    selected = build_candidates(list(X.columns))[selected_name]
    selected.fit(x_development, y_development)
    test_scores = selected.predict_proba(x_test)[:, 1]
    test_metrics = evaluate(y_test, test_scores)
    save_plots(y_test, test_scores, output_dir, selected_name)
    joblib.dump(selected, output_dir / "fraud_model.joblib")

    report = {
        "dataset": "Credit Card Fraud Detection, OpenML data_id=42175 / ULB-Worldline source",
        "target": TARGET,
        "ordering": f"ascending {TIME_COLUMN}; stable order for ties",
        "split_policy": "chronological 60/20/20; no shuffle and no stratification",
        "features": list(X.columns),
        "rows_after_validation": int(len(X)),
        "overall_frauds": int(y.sum()),
        "overall_prevalence": float(y.mean()),
        "split_summary": split_summary,
        "validation_metrics_for_model_selection": validation_metrics,
        "selected_model": selected_name,
        "test_metrics_once_after_selection": test_metrics,
        "notes": [
            "Average Precision (PR-AUC) is primary because the positive class is rare; ROC-AUC is secondary.",
            "Threshold 0.5 is descriptive only, not a recommended operational threshold.",
            "Capacity metrics reflect ranking on the held-out test set; use validation to set a deployable threshold.",
            "The dataset covers only two days and is anonymized; results do not establish production performance.",
            "No business costs were supplied, so no monetary savings or ROI is estimated.",
        ],
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("Relatório: %s", output_dir / "metrics.json")
    LOGGER.info("Gráficos: %s", output_dir)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=None, help="CSV local com Time, Class e atributos numéricos")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"), help="Pasta para métricas, gráficos e modelo")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")
    report = run(args.csv, args.output_dir)
    print(json.dumps({"selected_model": report["selected_model"], "test_metrics": report["test_metrics_once_after_selection"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
