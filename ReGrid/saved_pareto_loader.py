from __future__ import annotations

import csv
import json
import math
import os
from typing import Dict, Optional, Sequence

from reuse_genetic_evaluator import CandidateEvaluation


def _safe_float(value, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


_SAVED_OBJECTIVE_COLUMNS = (
    "mass",
    "cost",
    "reuse_ratio",
    "reuse_efficiency",
    "reuse_ratio_members",
    "dcr_max",
    "ec",
    "lscrap",
    "n_cuts",
    "max_abs_uz",
    "lambda_ref",
    "pi_mass",
    "pi_ec",
    "pi_mr",
)


def _restore_saved_objectives(
    objectives: Dict[str, float],
    meta: Dict[str, object],
    source: Dict[str, object],
) -> None:
    for key in _SAVED_OBJECTIVE_COLUMNS:
        raw_value = source.get(key, None)
        if raw_value in {None, ""}:
            continue
        value = _safe_float(raw_value)
        if not math.isfinite(value):
            continue
        if key == "lambda_ref":
            meta.setdefault(key, value)
        elif key in {"dcr_max", "ec", "reuse_ratio_members", "lscrap", "n_cuts", "max_abs_uz"}:
            if key == "n_cuts" and value == int(value):
                meta.setdefault(key, int(value))
            else:
                meta.setdefault(key, value)
        else:
            objectives.setdefault(key, value)


def _candidate_from_pareto_record(record: Dict[str, object]) -> CandidateEvaluation:
    objectives = {str(k): float(v) for k, v in dict(record.get("objectives", {}) or {}).items()}
    constraints = {str(k): float(v) for k, v in dict(record.get("constraints", {}) or {}).items()}
    meta = dict(record.get("meta", {}) or {})
    _restore_saved_objectives(objectives, meta, meta)
    candidate_id = record.get("candidate_id", meta.get("candidate_id", None))
    if candidate_id is not None:
        try:
            meta.setdefault("candidate_id", int(candidate_id))
        except Exception:
            meta.setdefault("candidate_id", candidate_id)
    decision_vector = tuple(float(v) for v in list(record.get("decision_vector", []) or []))
    lambda_ref = record.get("lambda_ref", None)
    lambda_value = None if lambda_ref is None else float(lambda_ref)
    return CandidateEvaluation(
        x=decision_vector,
        feasible=bool(record.get("feasible", False)),
        objectives=objectives,
        constraints=constraints,
        moved_G=None,
        moved_M=None,
        opt_result=None,
        reuse_result=None,
        final_state=None,
        section_by_member=None,
        lambda_ref=lambda_value,
        meta=meta,
    )


def _candidate_from_metrics_row(row: Dict[str, str], objective_names: Sequence[str]) -> CandidateEvaluation:
    objectives: Dict[str, float] = {}
    meta: Dict[str, object] = {}
    for name in objective_names:
        if name == "lambda_ref":
            continue
        if name in {"dcr_max", "max_abs_uz", "reuse_ratio_members", "ec"}:
            meta[name] = _safe_float(row.get(name, None))
        else:
            objectives[name] = _safe_float(row.get(name, None))
    _restore_saved_objectives(objectives, meta, row)
    for key in (
        "cost",
        "reuse_ratio_members",
        "ec",
        "dcr_max",
        "dcr_mean",
        "max_abs_uz",
        "merchant_rankine",
        "reused_mass",
        "new_mass",
        "new_reference_mass",
        "new_reference_mass_tonnes",
        "mscrap",
        "lscrap",
        "n_cuts",
        "new_reference_ec",
        "new_reference_cost",
        "new_reference_dcr_max",
        "new_reference_dcr_mean",
        "new_reference_max_abs_uz",
        "new_reference_lambda_ref",
        "new_reference_merchant_rankine",
        "new_reference_merchant_rankine_lambda",
        "comparison_reference_mass",
        "comparison_reference_mass_tonnes",
        "comparison_reference_ec",
        "comparison_reference_cost",
        "comparison_reference_dcr_max",
        "comparison_reference_dcr_mean",
        "comparison_reference_max_abs_uz",
        "comparison_reference_lambda_ref",
        "comparison_reference_merchant_rankine",
        "comparison_reference_merchant_rankine_lambda",
        "pi_mass",
        "pi_ec",
        "pi_mr",
    ):
        if key in row and row.get(key, "") != "":
            value = _safe_float(row.get(key))
            meta[key] = int(value) if key == "n_cuts" and value == int(value) else value
    for key in ("comparison_reference_mode", "comparison_reference_label"):
        raw_value = row.get(key, "")
        if raw_value != "":
            meta[key] = raw_value
    candidate_id = row.get("candidate_id", "")
    if candidate_id != "":
        try:
            meta["candidate_id"] = int(float(candidate_id))
        except Exception:
            meta["candidate_id"] = candidate_id
    generation = row.get("generation", "")
    if generation != "":
        try:
            meta["generation"] = int(float(generation))
        except Exception:
            meta["generation"] = generation
    decision_vector_raw = row.get("decision_vector", "")
    try:
        decision_vector = tuple(float(v) for v in json.loads(decision_vector_raw)) if decision_vector_raw else tuple()
    except Exception:
        decision_vector = tuple()
    constraints = {
        "g_dcr": _safe_float(row.get("g_dcr", None)),
        "g_disp": _safe_float(row.get("g_disp", None)),
        "g_buckling": _safe_float(row.get("g_buckling", None)),
        "g_merchant_rankine": _safe_float(row.get("g_merchant_rankine", None)),
        "g_analysis": _safe_float(row.get("g_analysis", None)),
    }
    lambda_ref_raw = row.get("lambda_ref", "")
    lambda_ref = None if lambda_ref_raw == "" else _safe_float(lambda_ref_raw)
    return CandidateEvaluation(
        x=decision_vector,
        feasible=_safe_bool(row.get("feasible", "")),
        objectives=objectives,
        constraints=constraints,
        moved_G=None,
        moved_M=None,
        opt_result=None,
        reuse_result=None,
        final_state=None,
        section_by_member=None,
        lambda_ref=lambda_ref,
        meta=meta,
    )


def _candidate_key(evaluation: CandidateEvaluation):
    candidate_id = dict(evaluation.meta or {}).get("candidate_id", None)
    if candidate_id is not None and str(candidate_id).strip() != "":
        try:
            return ("candidate_id", int(float(candidate_id)))
        except Exception:
            return ("candidate_id", str(candidate_id))
    vector = tuple(round(float(v), 12) for v in (getattr(evaluation, "x", None) or ()))
    if vector:
        return ("decision_vector", vector)
    return ("object", id(evaluation))


def _load_metrics_evaluations(path: Optional[str], objective_names: Sequence[str]) -> list[CandidateEvaluation]:
    if not path or not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8", newline="") as fobj:
        return [_candidate_from_metrics_row(row, objective_names) for row in csv.DictReader(fobj)]


def _merge_evaluation_sources(*sources: Sequence[CandidateEvaluation]) -> list[CandidateEvaluation]:
    merged: list[CandidateEvaluation] = []
    seen = set()
    for source in sources:
        for evaluation in source or []:
            key = _candidate_key(evaluation)
            if key in seen:
                continue
            seen.add(key)
            merged.append(evaluation)
    return merged


def load_saved_pareto_front(
    output_dir: Optional[str] = None,
    *,
    pareto_json_path: Optional[str] = None,
    metrics_csv_path: Optional[str] = None,
) -> tuple[list[CandidateEvaluation], list[CandidateEvaluation], tuple[str, ...]]:
    if pareto_json_path is None:
        if not output_dir:
            raise ValueError("Either output_dir or pareto_json_path must be provided.")
        global_pareto_json_path = os.path.join(output_dir, "reuse_genetic_pareto_front_global.json")
        if os.path.exists(global_pareto_json_path):
            pareto_json_path = global_pareto_json_path
        else:
            pareto_json_path = os.path.join(output_dir, "reuse_genetic_pareto_front.json")
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(pareto_json_path))

    with open(pareto_json_path, "r", encoding="utf-8") as fobj:
        raw_records = json.load(fobj)
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError(f"Saved Pareto file is empty or invalid: {pareto_json_path}")

    first_objectives = dict(raw_records[0].get("objectives", {}) or {})
    objective_names = tuple(str(name) for name in first_objectives.keys())
    if len(objective_names) not in {2, 3}:
        raise ValueError(
            f"Saved Pareto GUI supports 2 or 3 objectives, found {len(objective_names)} in {pareto_json_path}."
        )

    pareto_front = [_candidate_from_pareto_record(dict(record)) for record in raw_records]
    metric_paths = []
    if metrics_csv_path is None:
        completed_metrics_path = os.path.join(output_dir, "genetic_search", "genetic_candidate_metrics_completed.csv")
        live_metrics_path = os.path.join(output_dir, "genetic_search", "genetic_candidate_metrics.csv")
        metric_paths.append(completed_metrics_path if os.path.exists(completed_metrics_path) else live_metrics_path)
        metric_paths.append(os.path.join(output_dir, "pareto_candidate_metrics.csv"))
    else:
        metric_paths.append(metrics_csv_path)
    metric_evaluations = []
    for path in metric_paths:
        metric_evaluations.extend(_load_metrics_evaluations(path, objective_names))

    all_evaluations = _merge_evaluation_sources(metric_evaluations, pareto_front)
    if not all_evaluations:
        all_evaluations = list(pareto_front)
    by_candidate_id = {}
    for evaluation in all_evaluations:
        candidate_id = evaluation.meta.get("candidate_id", None)
        if candidate_id is not None:
            try:
                by_candidate_id[int(candidate_id)] = evaluation
            except Exception:
                pass
    enriched_pareto = []
    for evaluation in pareto_front:
        candidate_id = evaluation.meta.get("candidate_id", None)
        replacement = None
        if candidate_id is not None:
            try:
                replacement = by_candidate_id.get(int(candidate_id), None)
            except Exception:
                replacement = None
        enriched_pareto.append(replacement if replacement is not None else evaluation)
    pareto_front = enriched_pareto
    return pareto_front, all_evaluations, objective_names
