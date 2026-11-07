#!/usr/bin/env python3
"""Classificação de defeitos de placas de aço com validação sem leakage.

Usa UCI Steel Plates Faults. A fonte fornece sete colunas binárias; o script
confere que cada observação tem exatamente um rótulo ativo e deriva a classe.
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
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

LOGGER = logging.getLogger("steel_quality")
SEED = 42


def load_dataset() -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Fetch UCI dataset 198 and validate the one-hot target contract."""
    from ucimlrepo import fetch_ucirepo

    dataset = fetch_ucirepo(id=198)
    X = dataset.data.features.copy()
    target_matrix = dataset.data.targets.copy()
    if target_matrix.empty or target_matrix.isna().any().any():
        raise ValueError("Target matrix está vazia ou contém ausências.")
    target_matrix = target_matrix.apply(pd.to_numeric, errors="raise").astype(int)
    if not set(np.unique(target_matrix.to_numpy())).issubset({0, 1}):
        raise ValueError("Esperados rótulos binários one-hot 0/1.")
    row_sums = target_matrix.sum(axis=1)
    if not row_sums.eq(1).all():
        counts = row_sums.value_counts().sort_index().to_dict()
        raise ValueError(f"Não é codificação de classe exclusiva; contagens por soma: {counts}")
    classes = list(target_matrix.columns)
    y = pd.Series(np.array(classes, dtype=object)[target_matrix.to_numpy().argmax(axis=1)], name="fault_type")
    return X.reset_index(drop=True), y.reset_index(drop=True), classes


def make_pipeline(estimator: Any) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", estimator),
    ])


def evaluate(y_true: pd.Series, prediction: np.ndarray, probabilities: np.ndarray, classes: list[str]) -> dict[str, Any]:
    y_array = np.asarray(y_true)
    top2_indices = np.argsort(-probabilities, axis=1, kind="stable")[:, :2]
    top2_labels = np.asarray(classes, dtype=object)[top2_indices]
    top2_accuracy = float(np.mean(np.any(top2_labels == y_array[:, None], axis=1)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_array, prediction, labels=classes, zero_division=0
    )
    class_ap = average_precision_score(
        pd.get_dummies(pd.Categorical(y_array, categories=classes)).to_numpy(),
        probabilities,
        average=None,
    )
    return {
        "rows": int(len(y_array)),
        "class_counts": {str(label): int((y_array == label).sum()) for label in classes},
        "accuracy": float(accuracy_score(y_array, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_array, prediction)),
        "macro_f1": float(f1_score(y_array, prediction, labels=classes, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_array, prediction, labels=classes, average="weighted", zero_division=0)),
        "macro_average_precision_ovr": float(average_precision_score(
            pd.get_dummies(pd.Categorical(y_array, categories=classes)).to_numpy(), probabilities, average="macro"
        )),
        "top_2_accuracy": top2_accuracy,
        "per_class": {
            str(label): {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
                "average_precision_ovr": float(class_ap[i]),
            }
            for i, label in enumerate(classes)
        },
        "confusion_matrix_labels_order": classes,
        "confusion_matrix": confusion_matrix(y_array, prediction, labels=classes).tolist(),
        "classification_report": classification_report(y_array, prediction, labels=classes, output_dict=True, zero_division=0),
    }


def align_probabilities(model: Pipeline, X: pd.DataFrame, classes: list[str]) -> np.ndarray:
    raw = model.predict_proba(X)
    model_classes = list(model.named_steps["model"].classes_)
    return raw[:, [model_classes.index(label) for label in classes]]


def save_figures(metrics: dict[str, Any], output_dir: Path) -> None:
    classes = metrics["confusion_matrix_labels_order"]
    matrix = np.asarray(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    image = ax.imshow(matrix, cmap="Blues")
    ax.set(title="Steel Plates Faults — matriz de confusão (teste)", xlabel="Classe prevista", ylabel="Classe real", xticks=np.arange(len(classes)), yticks=np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=35, ha="right")
    ax.set_yticklabels(classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="#111111", fontsize=8)
    fig.colorbar(image, ax=ax, shrink=.8)
    fig.savefig(output_dir / "confusion_matrix_test.png", dpi=170)
    plt.close(fig)

    recalls = [metrics["per_class"][label]["recall"] for label in classes]
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.bar(classes, recalls, color="#176B87")
    ax.set(title="Recall por tipo de defeito — teste", ylabel="Recall", ylim=(0, 1), xlabel="Classe de defeito")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=.2)
    fig.savefig(output_dir / "recall_by_fault_test.png", dpi=170)
    plt.close(fig)


def run(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    X, y, classes = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=SEED, stratify=y
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    candidates: dict[str, tuple[Pipeline, dict[str, list[Any]]]] = {
        "logistic_regression_balanced": (
            make_pipeline(LogisticRegression(class_weight="balanced", max_iter=3500, random_state=SEED)),
            {"model__C": [0.1, 1.0, 10.0]},
        ),
        "random_forest_balanced": (
            make_pipeline(RandomForestClassifier(class_weight="balanced_subsample", random_state=SEED, n_jobs=1)),
            {"model__n_estimators": [250], "model__max_depth": [None, 15], "model__min_samples_leaf": [1, 3]},
        ),
    }
    searches: dict[str, GridSearchCV] = {}
    cv_summary: dict[str, Any] = {}
    for name, (pipeline, params) in candidates.items():
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=params,
            scoring="f1_macro",
            cv=cv,
            n_jobs=-1,
            refit=True,
            error_score="raise",
            return_train_score=False,
        )
        LOGGER.info("Busca de hiperparâmetros %s", name)
        search.fit(X_train, y_train)
        searches[name] = search
        cv_summary[name] = {
            "best_cv_macro_f1": float(search.best_score_),
            "best_params": search.best_params_,
        }
    selected_name = max(cv_summary, key=lambda name: cv_summary[name]["best_cv_macro_f1"])
    selected = searches[selected_name].best_estimator_
    predictions = selected.predict(X_test)
    probabilities = align_probabilities(selected, X_test, classes)
    test_metrics = evaluate(y_test, predictions, probabilities, classes)

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    baseline_prediction = dummy.predict(X_test)
    baseline = {
        "accuracy": float(accuracy_score(y_test, baseline_prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, baseline_prediction)),
        "macro_f1": float(f1_score(y_test, baseline_prediction, average="macro", zero_division=0)),
    }
    save_figures(test_metrics, output_dir)
    joblib.dump(selected, output_dir / "steel_fault_model.joblib")
    report = {
        "dataset": "UCI Steel Plates Faults (id 198)",
        "dataset_nature": "historical public manufacturing benchmark",
        "target_formulation": "seven one-hot target columns verified to have exactly one active class per row; converted to multiclass label",
        "features": list(X.columns),
        "classes": classes,
        "rows": int(len(X)),
        "split": "stratified 75/25; hyperparameters selected by 5-fold StratifiedKFold macro-F1 on training partition",
        "random_seed": SEED,
        "cross_validation": cv_summary,
        "selected_model": selected_name,
        "test_metrics": test_metrics,
        "most_frequent_baseline_test": baseline,
        "limitations": [
            "Small dataset; minority classes have few test cases and per-class scores are uncertain.",
            "Random stratified split is appropriate to this non-temporal benchmark but is not a test of future production drift.",
            "No inspection costs, downstream scrap/rework costs or factory-specific quality definitions are provided.",
            "A class prediction is decision support for inspection prioritization, not authorization to auto-reject product.",
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
    report = run(args.output_dir)
    result = report["test_metrics"]
    print(json.dumps({"selected_model": report["selected_model"], "macro_f1": result["macro_f1"], "balanced_accuracy": result["balanced_accuracy"], "macro_average_precision_ovr": result["macro_average_precision_ovr"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
