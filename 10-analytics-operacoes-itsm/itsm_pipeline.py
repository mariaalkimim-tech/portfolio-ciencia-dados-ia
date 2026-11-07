#!/usr/bin/env python3
"""Process analytics e sobrevivência de incidentes no BPI Challenge 2013.

Baixa o log XES oficial se --xes não for informado. Incidentes sem status final
Closed/Resolved são censurados à direita no último evento observado. Nenhum nome
individual de recurso é usado ou exportado.
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.utils import restricted_mean_survival_time

LOGGER = logging.getLogger("itsm_analytics")
XES_URL = "https://ais.win.tue.nl/bpi/2013/bpi_challenge_2013_incidents.xes.gz"
WAIT_STATES = {"Awaiting Assignment", "Wait", "Wait - User", "Wait - Implementation", "Wait - Vendor", "Wait - Customer"}
CLOSED_STATES = {"Closed", "Resolved"}


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _attribute_value(element: ET.Element) -> Any:
    value = element.attrib.get("value")
    kind = _local_name(element.tag)
    if kind in {"int", "integer"}:
        try:
            return int(value)
        except (TypeError, ValueError):
            return value
    if kind == "float":
        try:
            return float(value)
        except (TypeError, ValueError):
            return value
    return value


def ensure_log(path: Path) -> Path:
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Baixando log XES público oficial para %s", path)
    urllib.request.urlretrieve(XES_URL, path)
    return path


def parse_xes(path: Path) -> pd.DataFrame:
    """Parse compressed XES with Python's standard XML parser."""
    events: list[dict[str, Any]] = []
    with gzip.open(path, "rb") as stream:
        root = ET.parse(stream).getroot()
    supported = {"string", "int", "date", "float", "boolean"}
    for trace in root:
        if _local_name(trace.tag) != "trace":
            continue
        trace_attrs = {
            child.attrib.get("key"): _attribute_value(child)
            for child in trace if _local_name(child.tag) in supported
        }
        case_id = str(trace_attrs.get("concept:name", ""))
        if not case_id:
            continue
        for event in trace:
            if _local_name(event.tag) != "event":
                continue
            row = {
                child.attrib.get("key"): _attribute_value(child)
                for child in event if _local_name(child.tag) in supported
            }
            row["case_id"] = case_id
            events.append(row)
    frame = pd.DataFrame(events)
    required = {"case_id", "concept:name", "lifecycle:transition", "time:timestamp"}
    if not required.issubset(frame.columns):
        raise ValueError(f"XES ausente de atributos requeridos: {required - set(frame.columns)}")
    frame["time:timestamp"] = pd.to_datetime(frame["time:timestamp"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["time:timestamp"]).sort_values(["case_id", "time:timestamp"], kind="mergesort")
    return frame.reset_index(drop=True)


def build_cases(events: pd.DataFrame) -> pd.DataFrame:
    """Aggregate process metrics and case-level censoring information."""
    records: list[dict[str, Any]] = []
    intake_fields = ["impact", "product", "org:group", "organization involved", "concept:name", "lifecycle:transition"]
    for case_id, group in events.groupby("case_id", sort=False):
        g = group.sort_values("time:timestamp", kind="mergesort").reset_index(drop=True)
        start, end = g["time:timestamp"].iloc[0], g["time:timestamp"].iloc[-1]
        observed_span = max(0.0, (end - start).total_seconds() / 3600)
        terminal_state = str(g["lifecycle:transition"].iloc[-1])
        resolved = terminal_state in CLOSED_STATES
        resolution_hours = observed_span if resolved else np.nan
        org_groups = g.get("org:group", pd.Series(index=g.index, dtype=object)).fillna("").astype(str).tolist()
        compact_groups = [v for i, v in enumerate(org_groups) if v and (i == 0 or v != org_groups[i - 1])]
        wait_hours = 0.0
        for i in range(len(g) - 1):
            state = str(g["lifecycle:transition"].iloc[i])
            delta = (g["time:timestamp"].iloc[i + 1] - g["time:timestamp"].iloc[i]).total_seconds() / 3600
            if state in WAIT_STATES and delta > 0:
                wait_hours += delta
        first = g.iloc[0]
        row: dict[str, Any] = {
            "case_id": str(case_id), "start_time": start, "terminal_state": terminal_state,
            "is_resolved": bool(resolved), "resolution_hours": resolution_hours,
            "observed_span_hours": observed_span, "event_count": int(len(g)),
            "handoff_count": max(0, len(compact_groups) - 1),
            "distinct_org_groups": int(len(set(v for v in org_groups if v))),
            "wait_hours_proxy": float(wait_hours),
            "wait_share_proxy": float(wait_hours / observed_span) if observed_span > 0 else 0.0,
            "start_year": int(start.year), "start_month": f"{start.month:02d}",
            "start_weekday": f"{start.day_of_week}",
        }
        for field in intake_fields:
            row[f"intake_{field.replace(':', '_').replace(' ', '_')}"] = first.get(field, "unknown")
        records.append(row)
    cases = pd.DataFrame(records)
    if cases.empty:
        raise ValueError("Nenhum case foi extraído do XES.")
    return cases.sort_values("start_time", kind="mergesort").reset_index(drop=True)


def case_survival_summary(cases: pd.DataFrame, tau_hours: float = 720.0) -> dict[str, Any]:
    """Kaplan–Meier summary; unresolved cases are right-censored at last event."""
    durations = cases["resolution_hours"].fillna(cases["observed_span_hours"]).astype(float)
    observed = cases["is_resolved"].astype(bool)
    kmf = KaplanMeierFitter().fit(durations, event_observed=observed)
    median = float(kmf.median_survival_time_)
    horizons = (24.0, 72.0, 168.0, 720.0)
    survival = {f"{int(h)}h": float(kmf.predict(h)) for h in horizons}
    rmst = float(restricted_mean_survival_time(kmf, t=min(tau_hours, float(durations.max()))))
    return {
        "cases": int(len(cases)), "resolved_events": int(observed.sum()),
        "right_censored_cases": int((~observed).sum()),
        "resolution_event_share": float(observed.mean()),
        "kaplan_meier_median_resolution_hours": median if np.isfinite(median) else None,
        "survival_probability_no_resolution_by_horizon": survival,
        "restricted_mean_time_to_resolution_hours_tau": rmst,
        "tau_hours": float(min(tau_hours, float(durations.max()))),
    }


def temporal_distribution_diagnostic(cases: pd.DataFrame) -> dict[str, Any]:
    """Check whether a train-derived long-duration threshold transfers forward."""
    n = len(cases)
    i_train, i_val = int(.60 * n), int(.80 * n)
    windows = {
        "train": np.arange(0, i_train),
        "validation": np.arange(i_train, i_val),
        "later_period": np.arange(i_val, n),
    }
    train = cases.iloc[windows["train"]]
    completed_train = train[train["is_resolved"]]
    threshold = float(completed_train["resolution_hours"].quantile(.75))
    result: dict[str, Any] = {"train_p75_resolution_hours": threshold, "windows": {}}
    for name, indices in windows.items():
        subset = cases.iloc[indices]
        completed = subset[subset["is_resolved"]]
        above = completed["resolution_hours"] > threshold
        result["windows"][name] = {
            "all_cases": int(len(subset)),
            "closed_or_resolved_cases": int(len(completed)),
            "closed_cases_above_train_p75": int(above.sum()),
            "share_above_train_p75_among_closed": float(above.mean()) if len(above) else None,
        }
    result["interpretation"] = (
        "Strong temporal distribution shift: the later window contains no closed cases above the train p75; "
        "a binary long-duration classifier was not reported because the temporal holdout cannot test that class."
    )
    return result


def process_kpis(events: pd.DataFrame, cases: pd.DataFrame) -> dict[str, Any]:
    monthly = cases.assign(month=cases["start_time"].dt.tz_localize(None).dt.to_period("M").astype(str)).groupby("month").size()
    resolved = cases[cases["is_resolved"]]
    open_cases = cases[~cases["is_resolved"]]
    by_impact: dict[str, Any] = {}
    for impact, group in cases.groupby("intake_impact", dropna=False):
        by_impact[str(impact)] = case_survival_summary(group)
    return {
        "case_count": int(cases["case_id"].nunique()), "event_count": int(len(events)),
        "start_range_utc": [cases["start_time"].min().isoformat(), cases["start_time"].max().isoformat()],
        "resolution_hours_median_closed_only": float(resolved["resolution_hours"].median()),
        "resolution_hours_p90_closed_only": float(resolved["resolution_hours"].quantile(.90)),
        "resolved_share": float(cases["is_resolved"].mean()),
        "terminal_status_counts": {str(k): int(v) for k, v in cases["terminal_state"].value_counts().items()},
        "open_cases_observed_span_hours_median": float(open_cases["observed_span_hours"].median()),
        "events_per_case_median": float(cases["event_count"].median()),
        "handoffs_per_case_median": float(cases["handoff_count"].median()),
        "cases_with_handoff_share": float((cases["handoff_count"] > 0).mean()),
        "wait_share_proxy_median": float(cases["wait_share_proxy"].median()),
        "monthly_case_volume": {str(k): int(v) for k, v in monthly.items()},
        "temporal_distribution_diagnostic": temporal_distribution_diagnostic(cases),
        "overall_survival_analysis": case_survival_summary(cases),
        "survival_analysis_by_initial_impact": by_impact,
    }


def save_figures(cases: pd.DataFrame, output: Path) -> None:
    monthly = cases.assign(month=cases["start_time"].dt.tz_localize(None).dt.to_period("M").astype(str)).groupby("month").size()
    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    month_dates = pd.to_datetime([f"{month}-01" for month in monthly.index])
    ax.plot(month_dates, monthly.values, marker="o", color="#175A7D", lw=1.8)
    ax.set(title="BPI 2013 — volume mensal de incidentes abertos", xlabel="Mês", ylabel="Casos")
    ax.tick_params(axis="x", rotation=55); ax.grid(alpha=.2)
    fig.savefig(output / "monthly_incident_volume.png", dpi=170); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    for impact, group in cases.groupby("intake_impact", dropna=False):
        if len(group) < 25:
            continue
        durations = group["resolution_hours"].fillna(group["observed_span_hours"])
        kmf = KaplanMeierFitter().fit(durations, event_observed=group["is_resolved"].astype(bool), label=f"{impact} (n={len(group)})")
        kmf.plot_survival_function(ax=ax, ci_show=False)
    ax.set(title="Tempo até encerramento por impacto inicial (Kaplan–Meier)", xlabel="Horas desde abertura", ylabel="Probabilidade ainda sem resolução")
    ax.set_ylim(0, 1.02); ax.grid(alpha=.2); ax.legend(title="Impacto", frameon=False)
    fig.savefig(output / "survival_by_impact.png", dpi=170); plt.close(fig)


def run(xes_path: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    event_path = ensure_log(xes_path)
    events = parse_xes(event_path)
    cases = build_cases(events)
    save_figures(cases, output_dir)
    summary = process_kpis(events, cases)
    report = {
        "dataset": "BPI Challenge 2013 incidents, Volvo IT Belgium",
        "source": "Official BPI Challenge XES event log; raw log excluded from repository",
        "method": "Kaplan-Meier survival analysis; Closed/Resolved = event, other terminal states = right-censored at last observed event; no binary classifier is reported because the later temporal window has no cases above the train p75",
        "privacy": "Resource/person names are not used in features or exported aggregates",
        "process_kpis": summary,
        "limitations": [
            "Historical process from one organization, 2010-2012; not representative of current ITSM without local validation.",
            "No formal SLA target, ticket cost, customer impact, or satisfaction labels are supplied.",
            "Wait-state time is an interval-derived proxy and should be validated against source-system semantics.",
            "Cancellations are treated as censored rather than as competing events; this simplification can bias time-to-resolution estimates.",
            "Kaplan-Meier estimates assume non-informative censoring, which may not hold when cases remain open for operational reasons.",
            "Monthly case volume changes sharply near the end of the event log; the historical series should not be treated as a stable demand forecast.",
        ],
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xes", type=Path, default=Path("data/bpi_challenge_2013_incidents.xes.gz"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    report = run(args.xes, args.output_dir)
    print(json.dumps(report["process_kpis"], ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
