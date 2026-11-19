from __future__ import annotations

import concurrent.futures
import contextlib
import csv
import importlib.util
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from pareto_plot_style import save_pareto_figure
from transparent_plots import configure_transparent_matplotlib_saves

configure_transparent_matplotlib_saves()

from reuse_genetic_evaluator import (
    CandidateEvaluation,
    ReuseGeneticEvaluationContext,
    build_decision_bounds,
    canonicalize_decision_vector,
    decision_variable_labels,
    evaluate_candidate_design,
)


_HAS_PYMOO = importlib.util.find_spec("pymoo") is not None
if _HAS_PYMOO:
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.algorithms.soo.nonconvex.ga import GA
    from pymoo.core.problem import Problem
    from pymoo.optimize import minimize
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rnd import FloatRandomSampling
    try:
        from pymoo.operators.sampling.lhs import LHS
    except Exception:  # pragma: no cover - optional import across pymoo versions
        LHS = None


@dataclass
class ReuseGeneticAlgorithmSettings:
    population_size: int = 12
    n_generations: int = 8
    elite_fraction: float = 0.20
    crossover_prob: float = 0.90
    mutation_prob: float = 0.20
    mutation_scale: float = 0.10
    seed: Optional[int] = 42
    multi_objective: bool = False
    engine: str = "auto"
    verbose: int = 1
    n_offsprings: Optional[int] = None
    sampling: str = "real_random"
    eliminate_duplicates: bool = True
    crossover_eta: float = 15.0
    mutation_eta: float = 20.0
    n_workers: int = 1
    initial_population: Optional[Sequence[Sequence[float]]] = None
    initial_cache: Optional[Sequence[CandidateEvaluation]] = None
    candidate_id_start: int = 1
    generation_offset: int = 0
    history_seed: Optional[Sequence[Mapping[str, float]]] = None
    progress_heartbeat_s: float = 30.0
    candidate_stall_timeout_s: Optional[float] = None
    suppress_evaluation_output: bool = True
    terminal_progress_every: int = 50


@dataclass
class ReuseGeneticOptimizationResult:
    best: Optional[CandidateEvaluation]
    history: List[Dict[str, float]] = field(default_factory=list)
    method: str = "builtin-ga"
    n_evaluations: int = 0
    cache_hits: int = 0
    pareto_front: List[CandidateEvaluation] = field(default_factory=list)
    pareto_front_global: List[CandidateEvaluation] = field(default_factory=list)
    objective_names: Tuple[str, ...] = field(default_factory=tuple)
    all_evaluations: List[CandidateEvaluation] = field(default_factory=list)


def _generation_value(evaluation: CandidateEvaluation) -> Optional[int]:
    try:
        generation = evaluation.meta.get("generation", None)
        if generation is None:
            return None
        return int(generation)
    except Exception:
        return None


def _candidate_id_value(evaluation: CandidateEvaluation) -> Optional[int]:
    try:
        candidate_id = evaluation.meta.get("candidate_id", None)
        if candidate_id is None:
            return None
        return int(candidate_id)
    except Exception:
        return None


def resume_population_from_evaluations(
    evaluations: Sequence[CandidateEvaluation],
    *,
    expected_size: Optional[int] = None,
) -> List[List[float]]:
    valid_rows = [ev for ev in evaluations if getattr(ev, "x", None)]
    if not valid_rows:
        return []
    generations = [gen for gen in (_generation_value(ev) for ev in valid_rows) if gen is not None]
    if not generations:
        rows = valid_rows
    else:
        last_generation = max(generations)
        rows = [ev for ev in valid_rows if _generation_value(ev) == last_generation]
        if not rows:
            rows = valid_rows
    rows = sorted(
        rows,
        key=lambda ev: (
            _candidate_id_value(ev) is None,
            _candidate_id_value(ev) if _candidate_id_value(ev) is not None else float("inf"),
        ),
    )
    population = [list(float(v) for v in ev.x) for ev in rows if getattr(ev, "x", None)]
    if expected_size is not None and expected_size > 0 and len(population) > expected_size:
        population = population[:expected_size]
    return population


def build_resume_settings(
    previous_result: ReuseGeneticOptimizationResult,
    base_settings: ReuseGeneticAlgorithmSettings,
    extra_generations: int,
) -> ReuseGeneticAlgorithmSettings:
    extra_generations = max(1, int(extra_generations))
    all_evaluations = list(previous_result.all_evaluations or [])
    last_generation = 0
    candidate_id_start = 1
    for evaluation in all_evaluations:
        generation = _generation_value(evaluation)
        if generation is not None:
            last_generation = max(last_generation, generation)
        candidate_id = _candidate_id_value(evaluation)
        if candidate_id is not None:
            candidate_id_start = max(candidate_id_start, candidate_id + 1)
    initial_population = resume_population_from_evaluations(
        all_evaluations,
        expected_size=max(1, int(base_settings.population_size)),
    )
    return ReuseGeneticAlgorithmSettings(
        population_size=base_settings.population_size,
        n_generations=extra_generations + 1,
        elite_fraction=base_settings.elite_fraction,
        crossover_prob=base_settings.crossover_prob,
        mutation_prob=base_settings.mutation_prob,
        mutation_scale=base_settings.mutation_scale,
        seed=base_settings.seed,
        multi_objective=base_settings.multi_objective,
        engine=base_settings.engine,
        verbose=base_settings.verbose,
        n_offsprings=base_settings.n_offsprings,
        sampling=base_settings.sampling,
        eliminate_duplicates=base_settings.eliminate_duplicates,
        crossover_eta=base_settings.crossover_eta,
        mutation_eta=base_settings.mutation_eta,
        n_workers=base_settings.n_workers,
        initial_population=initial_population,
        initial_cache=all_evaluations,
        candidate_id_start=candidate_id_start,
        generation_offset=max(0, int(last_generation) - 1),
        history_seed=list(previous_result.history or []),
        progress_heartbeat_s=base_settings.progress_heartbeat_s,
        candidate_stall_timeout_s=base_settings.candidate_stall_timeout_s,
        suppress_evaluation_output=base_settings.suppress_evaluation_output,
        terminal_progress_every=base_settings.terminal_progress_every,
    )



def _coerce_csv_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default
    text = str(value).strip()
    if text == "" or text.lower() in {"none", "nan", "null"}:
        return default
    try:
        return float(text)
    except Exception:
        return default


def _coerce_csv_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    number = _coerce_csv_float(value, None)
    if number is None:
        return default
    try:
        return int(number)
    except Exception:
        return default


def load_candidate_metrics_csv(path: str) -> List[CandidateEvaluation]:
    """Load previous GA evaluations from genetic_candidate_metrics.csv.

    This is intentionally lightweight: it restores decision vectors, objectives,
    constraints and metadata so the next headless run can seed/cache the GA.
    Heavy objects such as moved geometry and section assignments are not stored
    in this CSV and are therefore set to None.
    """
    if not path or not os.path.exists(path):
        return []
    evaluations: List[CandidateEvaluation] = []
    objective_columns = [
        "mass", "cost", "ec", "pi_mass", "pi_ec", "pi_mr", "pi_cost",
        "reuse_ratio", "reuse_efficiency", "reuse_ratio_members", "n_cuts", "lscrap",
        "dcr_max", "max_abs_uz", "lambda_ref",
    ]
    constraint_columns = ["g_dcr", "g_disp", "g_buckling", "g_merchant_rankine", "g_analysis"]
    with open(path, "r", newline="", encoding="utf-8") as fobj:
        reader = csv.DictReader(fobj)
        for row in reader:
            raw_vector = row.get("decision_vector", "")
            try:
                vector_data = json.loads(raw_vector) if raw_vector else []
                vector = tuple(float(v) for v in vector_data)
            except Exception:
                continue
            if not vector:
                continue
            objectives: Dict[str, float] = {}
            for name in objective_columns:
                value = _coerce_csv_float(row.get(name), None)
                if value is not None and math.isfinite(float(value)):
                    objectives[name] = float(value)
            constraints: Dict[str, float] = {}
            for name in constraint_columns:
                value = _coerce_csv_float(row.get(name), 0.0)
                constraints[name] = float(value or 0.0)
            meta: Dict[str, Any] = {}
            for key, value in row.items():
                if key in {"decision_vector", "decision_labels"}:
                    continue
                parsed = _coerce_csv_float(value, None)
                meta[key] = parsed if parsed is not None else value
            candidate_id = _coerce_csv_int(row.get("candidate_id"), None)
            generation = _coerce_csv_int(row.get("generation"), None)
            if candidate_id is not None:
                meta["candidate_id"] = candidate_id
            if generation is not None:
                meta["generation"] = generation
            error_text = str(row.get("error", "") or "")
            if error_text:
                meta["error"] = error_text
            feasible_raw = str(row.get("feasible", "0") or "0").strip().lower()
            feasible = feasible_raw in {"1", "true", "yes", "y"}
            evaluations.append(
                CandidateEvaluation(
                    x=vector,
                    feasible=bool(feasible),
                    objectives=objectives,
                    constraints=constraints,
                    moved_G=None,
                    moved_M=None,
                    opt_result=None,
                    reuse_result=None,
                    final_state=None,
                    section_by_member=None,
                    lambda_ref=_coerce_csv_float(row.get("lambda_ref"), None),
                    meta=meta,
                )
            )
    return evaluations

def _seed_cache_from_settings(
    ctx: ReuseGeneticEvaluationContext,
    settings: ReuseGeneticAlgorithmSettings,
) -> Dict[Tuple[float, ...], CandidateEvaluation]:
    cache: Dict[Tuple[float, ...], CandidateEvaluation] = {}
    for evaluation in list(settings.initial_cache or []):
        if not getattr(evaluation, "x", None):
            continue
        cache[_cache_key(ctx, evaluation.x)] = evaluation
    return cache


def _build_initial_population(
    rng: random.Random,
    settings: ReuseGeneticAlgorithmSettings,
    n_var: int,
    xl: Sequence[float],
    xu: Sequence[float],
) -> List[List[float]]:
    n_pop = max(4, int(settings.population_size))
    seeded: List[List[float]] = []
    for raw_vector in list(settings.initial_population or []):
        if raw_vector is None:
            continue
        vector = list(float(v) for v in raw_vector)
        if len(vector) != n_var:
            continue
        clipped = [
            max(float(xl[idx]), min(float(xu[idx]), float(value)))
            for idx, value in enumerate(vector)
        ]
        seeded.append(clipped)
        if len(seeded) >= n_pop:
            break
    if len(seeded) < n_pop:
        seeded.extend(_random_population(rng, n_pop - len(seeded), xl, xu))
    return seeded[:n_pop]


_OBJECTIVE_ALIASES = {
    "mass": "mass",
    "cost": "cost",
    "price": "cost",
    "ec": "ec",
    "reuse": "reuse_ratio",
    "reuse_ratio": "reuse_ratio",
    "max_reuse_ratio": "reuse_ratio",
    "reuse_efficiency": "reuse_efficiency",
    "reuse_eff": "reuse_efficiency",
    "reuse_ratio_members": "reuse_ratio_members",
    "dcr": "dcr_max",
    "dcr_max": "dcr_max",
    "lscrap": "lscrap",
    "scrap_length": "lscrap",
    "ncut": "n_cuts",
    "ncuts": "n_cuts",
    "n_cut": "n_cuts",
    "n_cuts": "n_cuts",
    "disp": "max_abs_uz",
    "displacement": "max_abs_uz",
    "max_abs_uz": "max_abs_uz",
    "uz": "max_abs_uz",
    "buckling": "lambda_ref",
    "lambda": "lambda_ref",
    "lambda_ref": "lambda_ref",
    "pi_mass": "pi_mass",
    "pi_m": "pi_mass",
    "pi_ec": "pi_ec",
    "pi_mr": "pi_mr",
    "pi_cost": "pi_cost",
    "pi_c": "pi_cost",
}

_OBJECTIVE_DIRECTIONS = {
    "mass": "min",
    "cost": "min",
    "ec": "min",
    "reuse_ratio": "max",
    "reuse_efficiency": "max",
    "reuse_ratio_members": "max",
    "dcr_max": "min",
    "lscrap": "min",
    "n_cuts": "min",
    "max_abs_uz": "min",
    "lambda_ref": "max",
    "pi_mass": "max",
    "pi_ec": "max",
    "pi_mr": "max",
    "pi_cost": "max",
}

_CANDIDATE_METRICS_FIELDNAMES = [
    "candidate_id",
    "generation",
    "feasible",
    "error",
    "total_violation",
    "normalized_violation",
    "mass",
    "cost",
    "ec",
    "reuse_ratio",
    "reuse_efficiency",
    "reuse_ratio_members",
    "dcr_max",
    "dcr_mean",
    "max_abs_uz",
    "merchant_rankine",
    "pi_mass",
    "pi_ec",
    "pi_mr",
    "pi_cost",
    "reused_mass",
    "new_mass",
    "new_reference_mass",
    "new_reference_mass_tonnes",
    "mscrap",
    "lscrap",
    "n_cuts",
    "diagonal_topology_enabled",
    "diagonal_topology_pattern",
    "diagonal_topology_group_count",
    "diagonal_candidate_count",
    "diagonal_active_count",
    "new_reference_ec",
    "new_reference_cost",
    "new_reference_dcr_max",
    "new_reference_dcr_mean",
    "new_reference_max_abs_uz",
    "new_reference_lambda_ref",
    "new_reference_merchant_rankine",
    "ga_dcr_limit_used",
    "ga_disp_limit_used",
    "ga_buckling_lambda_min_used",
    "ga_merchant_rankine_min_used",
    "opt_dcr_limit_used",
    "opt_dlim_used",
    "opt_buckling_lambda_min_used",
    "opt_merchant_rankine_min_used",
    "g_dcr",
    "g_disp",
    "g_buckling",
    "g_merchant_rankine",
    "g_analysis",
    "delta_z_max",
    "lambda_ref",
    "time_geom_s",
    "time_mech_s",
    "time_mech_inner_total_s",
    "n_mech_state_calls",
    "time_mech_state_total_s",
    "n_mech_base_calls",
    "time_mech_base_s",
    "n_mech_buckling_calls",
    "time_mech_buckling_s",
    "time_match_s",
    "time_match_single_search_s",
    "time_match_group_search_s",
    "time_match_stock_update_s",
    "n_match_single_checked",
    "n_match_group_checked",
    "time_recheck_new_s",
    "time_recheck_s",
    "time_total_s",
    "decision_labels",
    "decision_vector",
]


_PARALLEL_EVAL_CTX: Optional[ReuseGeneticEvaluationContext] = None
_PARALLEL_SUPPRESS_EVAL_OUTPUT: bool = True


@contextlib.contextmanager
def _suppress_terminal_output(enabled: bool = True):
    """Temporarily silence stdout/stderr, including noisy prints from worker code."""
    if not enabled:
        yield
        return
    sys.stdout.flush()
    sys.stderr.flush()
    saved_stdout_fd = os.dup(1)
    saved_stderr_fd = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_stdout_fd, 1)
        os.dup2(saved_stderr_fd, 2)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)
        os.close(devnull_fd)


def _evaluation_output_is_suppressed(settings: Optional[ReuseGeneticAlgorithmSettings]) -> bool:
    if settings is None:
        return True
    return bool(getattr(settings, "suppress_evaluation_output", True))


def _parallel_worker_init(ctx: ReuseGeneticEvaluationContext, suppress_output: bool = True) -> None:
    global _PARALLEL_EVAL_CTX, _PARALLEL_SUPPRESS_EVAL_OUTPUT
    _PARALLEL_EVAL_CTX = ctx
    _PARALLEL_SUPPRESS_EVAL_OUTPUT = bool(suppress_output)


def _evaluate_candidate_in_worker(vector: Sequence[float]) -> CandidateEvaluation:
    if _PARALLEL_EVAL_CTX is None:
        raise RuntimeError("Parallel GA worker context is not initialized.")
    with _suppress_terminal_output(_PARALLEL_SUPPRESS_EVAL_OUTPUT):
        return evaluate_candidate_design(_PARALLEL_EVAL_CTX, vector)


def _cache_key(ctx: ReuseGeneticEvaluationContext, vector: Sequence[float]) -> Tuple[float, ...]:
    canonical = canonicalize_decision_vector(ctx, vector)
    return tuple(round(float(v), 6) for v in canonical)


def _normalize_objective_name(name: object) -> str:
    key = str(name).strip().lower()
    normalized = _OBJECTIVE_ALIASES.get(key, None)
    if normalized is None:
        raise ValueError(
            "Unsupported objective "
            f"'{name}'. Supported objectives: {', '.join(sorted(_OBJECTIVE_DIRECTIONS))}."
        )
    return normalized


def _normalize_objective_names(objective_names: Sequence[str]) -> Tuple[str, ...]:
    out: List[str] = []
    for name in objective_names:
        normalized = _normalize_objective_name(name)
        if normalized not in out:
            out.append(normalized)
    if not out:
        raise ValueError("At least one objective must be provided.")
    return tuple(out)


def _objective_direction(name: str) -> str:
    normalized = _normalize_objective_name(name)
    return str(_OBJECTIVE_DIRECTIONS[normalized])


def _objective_value(evaluation: CandidateEvaluation, name: str) -> float:
    objective = _normalize_objective_name(name)
    if objective == "lambda_ref":
        value = evaluation.lambda_ref
    elif objective == "dcr_max":
        value = evaluation.meta.get("dcr_max", None)
    elif objective == "lscrap":
        value = evaluation.meta.get("lscrap", None)
    elif objective == "n_cuts":
        value = evaluation.meta.get("n_cuts", None)
    elif objective == "max_abs_uz":
        value = evaluation.meta.get("max_abs_uz", None)
    elif objective == "reuse_ratio_members":
        value = evaluation.meta.get("reuse_ratio_members", None)
    elif objective == "reuse_efficiency":
        value = evaluation.meta.get("reuse_efficiency", evaluation.objectives.get("reuse_efficiency", None))
    elif objective == "ec":
        value = evaluation.meta.get("ec", None)
    else:
        value = evaluation.objectives.get(objective, None)

    try:
        numeric = float(value)
    except Exception:
        numeric = None

    if numeric is None or not math.isfinite(numeric):
        return float("-inf") if _objective_direction(objective) == "max" else float("inf")
    return float(numeric)


def _objective_display_value(evaluation: CandidateEvaluation, objective_names: Sequence[str]) -> Dict[str, float]:
    return {str(name): float(_objective_value(evaluation, name)) for name in objective_names}


def _objective_optimization_value(evaluation: CandidateEvaluation, name: str) -> float:
    value = float(_objective_value(evaluation, name))
    if _objective_direction(name) == "max":
        return -value
    return value


def _objective_score(evaluation: CandidateEvaluation, objective_names: Sequence[str]) -> float:
    if len(objective_names) != 1:
        raise ValueError("Built-in GA supports exactly one active objective. Use pymoo for multi-objective runs.")

    violation = evaluation.normalized_violation
    penalty = 1.0e9 * max(0.0, violation)
    base = _objective_optimization_value(evaluation, objective_names[0])
    return float(base + penalty)


def _objective_vector(evaluation: CandidateEvaluation, objective_names: Sequence[str]) -> List[float]:
    return [float(_objective_optimization_value(evaluation, name)) for name in objective_names]


def _constraint_vector(evaluation: CandidateEvaluation) -> List[float]:
    return [
        float(evaluation.constraints.get("g_dcr", float("inf"))),
        float(evaluation.constraints.get("g_disp", float("inf"))),
        float(evaluation.constraints.get("g_buckling", float("inf"))),
        float(evaluation.constraints.get("g_merchant_rankine", float("inf"))),
        float(evaluation.constraints.get("g_analysis", float("inf"))),
    ]


def _history_entry(
    generation: int,
    evaluations: Sequence[CandidateEvaluation],
    objective_names: Sequence[str],
    *,
    single_objective: bool,
) -> Dict[str, float]:
    feasible = [ev for ev in evaluations if ev.feasible]
    pool = feasible or list(evaluations)
    if single_objective:
        best = min(pool, key=lambda ev: _objective_score(ev, objective_names))
        entry = {
            "generation": float(generation),
            "best_score": float(_objective_score(best, objective_names)),
            "best_total_violation": float(best.total_violation),
            "best_normalized_violation": float(best.normalized_violation),
            "best_feasible": 1.0 if best.feasible else 0.0,
            "n_feasible": float(len(feasible)),
        }
        for name in objective_names:
            entry[f"best_{name}"] = float(_objective_value(best, name))
    else:
        entry = {
            "generation": float(generation),
            "n_feasible": float(len(feasible)),
            "min_total_violation": float(min(ev.total_violation for ev in evaluations)),
            "min_normalized_violation": float(min(ev.normalized_violation for ev in evaluations)),
        }
        for name in objective_names:
            if _objective_direction(name) == "max":
                entry[f"max_{name}"] = float(max(_objective_value(ev, name) for ev in pool))
            else:
                entry[f"min_{name}"] = float(min(_objective_value(ev, name) for ev in pool))
    return entry


def _fmt_float(value: object, default: str = "nan") -> str:
    try:
        v = float(value)
        if math.isfinite(v):
            return f"{v:.6f}"
    except Exception:
        pass
    return default


def _fmt_percent(value: object, default: str = "nan") -> str:
    try:
        v = float(value)
        if math.isfinite(v):
            return f"{100.0 * v:.2f}%"
    except Exception:
        pass
    return default


def _build_pymoo_sampling(settings: ReuseGeneticAlgorithmSettings):
    sampling_name = str(settings.sampling or "real_random").strip().lower()
    if sampling_name in {"real_random", "random", "float_random"}:
        return FloatRandomSampling()
    if sampling_name in {"lhs", "latin_hypercube", "latin-hypercube"}:
        if LHS is None:
            raise ValueError("pymoo LHS sampling is not available in this environment.")
        return LHS()
    raise ValueError(
        "Unsupported pymoo sampling "
        f"'{settings.sampling}'. Supported values: real_random, lhs."
    )


def _print_run_header(method: str, settings: ReuseGeneticAlgorithmSettings, objective_names: Sequence[str], n_var: int) -> None:
    if int(settings.verbose) < 1:
        return
    print(
        f"# GENETIC | start | method={method} | pop={int(settings.population_size)} | "
        f"gen={int(settings.n_generations)} | obj={','.join(str(v) for v in objective_names)} | "
        f"n_var={int(n_var)} | seed={settings.seed} | "
        f"workers={max(1, int(settings.n_workers or 1))} | "
        f"offsprings={settings.n_offsprings} | sampling={settings.sampling} | "
        f"cx_prob={float(settings.crossover_prob):.3f} | cx_eta={float(settings.crossover_eta):.3f} | "
        f"mut_prob={float(settings.mutation_prob):.3f} | mut_eta={float(settings.mutation_eta):.3f} | "
        f"dedup={bool(settings.eliminate_duplicates)}",
        flush=True,
    )


def _print_candidate_summary(
    settings: ReuseGeneticAlgorithmSettings,
    evaluation: CandidateEvaluation,
    *,
    candidate_id: int,
) -> None:
    if int(settings.verbose) < 2:
        return
    print(
        f"# GENETIC | candidate={int(candidate_id):04d} | "
        f"feasible={bool(evaluation.feasible)} | "
        f"mass={_fmt_float(evaluation.objectives.get('mass', None), 'inf')} | "
        f"cost={_fmt_float(evaluation.objectives.get('cost', None), 'inf')} | "
        f"EC={_fmt_float(evaluation.meta.get('ec', None), 'inf')} | "
        f"PI_M={_fmt_float(evaluation.objectives.get('pi_mass', None), 'nan')} | "
        f"PI_EC={_fmt_float(evaluation.objectives.get('pi_ec', None), 'nan')} | "
        f"PI_MR={_fmt_float(evaluation.objectives.get('pi_mr', None), 'nan')} | "
        f"PI_C={_fmt_float(evaluation.objectives.get('pi_cost', None), 'nan')} | "
        f"reuse={_fmt_percent(evaluation.objectives.get('reuse_ratio', None))} | "
        f"reuse_eff={_fmt_percent(evaluation.objectives.get('reuse_efficiency', None))} | "
        f"dcr={_fmt_float(evaluation.meta.get('dcr_max', None), 'inf')} | "
        f"dcr_lim={_fmt_float(evaluation.meta.get('ga_dcr_limit_used', None), 'nan')} | "
        f"mr={_fmt_float(evaluation.meta.get('merchant_rankine', None), 'nan')} | "
        f"mr_lim={_fmt_float(evaluation.meta.get('ga_merchant_rankine_min_used', None), 'nan')} | "
        f"uz={_fmt_float(evaluation.meta.get('max_abs_uz', None), 'inf')} | "
        f"uz_lim={_fmt_float(evaluation.meta.get('ga_disp_limit_used', None), 'nan')} | "
        f"lambda_ref={_fmt_float(evaluation.lambda_ref, 'nan')} | "
        f"lambda_lim={_fmt_float(evaluation.meta.get('ga_buckling_lambda_min_used', None), 'nan')} | "
        f"Lscrap={_fmt_float(evaluation.meta.get('lscrap', None), '0')} | "
        f"ncut={int(evaluation.meta.get('n_cuts', 0) or 0)} | "
        f"viol={_fmt_float(evaluation.total_violation, 'inf')} | "
        f"viol_norm={_fmt_float(evaluation.normalized_violation, 'inf')} | "
        f"t_geom={_fmt_float(evaluation.meta.get('time_geom_s', None))}s | "
        f"t_mech={_fmt_float(evaluation.meta.get('time_mech_s', None))}s | "
        f"mech_calls={int(evaluation.meta.get('n_mech_state_calls', 0))} | "
        f"mech_base={int(evaluation.meta.get('n_mech_base_calls', 0))}/{_fmt_float(evaluation.meta.get('time_mech_base_s', None))}s | "
        f"mech_buck={int(evaluation.meta.get('n_mech_buckling_calls', 0))}/{_fmt_float(evaluation.meta.get('time_mech_buckling_s', None))}s | "
        f"t_match={_fmt_float(evaluation.meta.get('time_match_s', None))}s | "
        f"t_match_1L={_fmt_float(evaluation.meta.get('time_match_single_search_s', None))}s | "
        f"t_match_grp={_fmt_float(evaluation.meta.get('time_match_group_search_s', None))}s | "
        f"t_match_upd={_fmt_float(evaluation.meta.get('time_match_stock_update_s', None))}s | "
        f"t_recheck={_fmt_float(evaluation.meta.get('time_recheck_s', None))}s | "
        f"t_total={_fmt_float(evaluation.meta.get('time_total_s', None))}s | "
        f"error={evaluation.meta.get('error', '')}",
        flush=True,
    )


def _print_generation_summary(
    settings: ReuseGeneticAlgorithmSettings,
    *,
    method: str,
    entry: Mapping[str, float],
) -> None:
    if int(settings.verbose) < 2:
        return
    if "best_score" in entry:
        parts = [
            f"# GENETIC | generation={int(entry['generation'])} | method={method}",
            f"best_score={_fmt_float(entry.get('best_score', None), 'inf')}",
        ]
        for key, value in entry.items():
            if key.startswith("best_") and key not in {"best_score", "best_total_violation", "best_normalized_violation", "best_feasible"}:
                formatter = _fmt_percent if "reuse_ratio" in key else _fmt_float
                parts.append(f"{key}={formatter(value)}")
        parts.extend(
            [
                f"viol={_fmt_float(entry.get('best_total_violation', None), 'inf')}",
                f"viol_norm={_fmt_float(entry.get('best_normalized_violation', None), 'inf')}",
                f"feasible={bool(int(entry.get('best_feasible', 0.0)))}",
                f"n_feasible={int(entry.get('n_feasible', 0.0)) if 'n_feasible' in entry else 0}",
            ]
        )
        print(" | ".join(parts), flush=True)
    else:
        parts = [f"# GENETIC | generation={int(entry['generation'])} | method={method}"]
        for key, value in entry.items():
            if key.startswith("min_") or key.startswith("max_"):
                formatter = _fmt_percent if "reuse_ratio" in key else _fmt_float
                parts.append(f"{key}={formatter(value)}")
        parts.extend(
            [
                f"min_viol={_fmt_float(entry.get('min_total_violation', None), 'inf')}",
                f"min_viol_norm={_fmt_float(entry.get('min_normalized_violation', None), 'inf')}",
                f"n_feasible={int(entry.get('n_feasible', 0.0))}",
            ]
        )
        print(" | ".join(parts), flush=True)


def _select_representative_best(
    evaluations: Sequence[CandidateEvaluation],
    objective_names: Sequence[str],
) -> Optional[CandidateEvaluation]:
    if not evaluations:
        return None
    feasible = [ev for ev in evaluations if ev.feasible]
    pool = feasible or list(evaluations)
    return min(
        pool,
        key=lambda ev: (
            float(ev.normalized_violation),
            float(ev.total_violation),
            *tuple(_objective_vector(ev, objective_names)),
        ),
    )


def _dominates(a: CandidateEvaluation, b: CandidateEvaluation, objective_names: Sequence[str]) -> bool:
    a_vec = _objective_vector(a, objective_names)
    b_vec = _objective_vector(b, objective_names)
    return all(float(xa) <= float(xb) for xa, xb in zip(a_vec, b_vec)) and any(
        float(xa) < float(xb) for xa, xb in zip(a_vec, b_vec)
    )


def _extract_pareto_front(
    evaluations: Sequence[CandidateEvaluation],
    objective_names: Sequence[str],
) -> List[CandidateEvaluation]:
    if not evaluations:
        return []
    feasible = [ev for ev in evaluations if ev.feasible]
    if feasible:
        pool = feasible
    else:
        min_violation = min(float(ev.normalized_violation) for ev in evaluations)
        pool = [ev for ev in evaluations if abs(float(ev.normalized_violation) - min_violation) <= 1e-12]

    front: List[CandidateEvaluation] = []
    for candidate in pool:
        dominated = False
        for other in pool:
            if other is candidate:
                continue
            if _dominates(other, candidate, objective_names):
                dominated = True
                break
        if not dominated:
            front.append(candidate)

    front.sort(
        key=lambda ev: (
            float(ev.normalized_violation),
            float(ev.total_violation),
            *tuple(_objective_vector(ev, objective_names)),
        )
    )
    return front


def _candidate_record(evaluation: CandidateEvaluation, objective_names: Sequence[str]) -> Dict[str, Any]:
    return {
        "candidate_id": evaluation.meta.get("candidate_id", None),
        "feasible": bool(evaluation.feasible),
        "total_violation": float(evaluation.total_violation),
        "normalized_violation": float(evaluation.normalized_violation),
        "objectives": _objective_display_value(evaluation, objective_names),
        "constraints": {str(k): float(v) for k, v in evaluation.constraints.items()},
        "lambda_ref": None if evaluation.lambda_ref is None else float(evaluation.lambda_ref),
        "meta": dict(evaluation.meta),
        "decision_vector": [float(v) for v in evaluation.x],
    }


def objective_display_values(evaluation: CandidateEvaluation, objective_names: Sequence[str]) -> Dict[str, float]:
    return _objective_display_value(evaluation, objective_names)


def objective_coordinate_matrix(
    evaluations: Sequence[CandidateEvaluation],
    objective_names: Sequence[str],
) -> np.ndarray:
    if not evaluations:
        return np.empty((0, len(objective_names)), dtype=float)
    return np.asarray(
        [list(_objective_display_value(ev, objective_names).values()) for ev in evaluations],
        dtype=float,
    )


def pareto_proximity_scores(
    evaluations: Sequence[CandidateEvaluation],
    pareto_front: Sequence[CandidateEvaluation],
    objective_names: Sequence[str],
) -> Dict[str, np.ndarray]:
    coords = objective_coordinate_matrix(evaluations, objective_names)
    pareto_coords = objective_coordinate_matrix(pareto_front, objective_names)
    closeness = np.zeros(len(coords), dtype=float)
    distances = np.full(len(coords), float("nan"), dtype=float)
    if len(coords) == 0 or len(pareto_coords) == 0:
        return {
            "coords": coords,
            "pareto_coords": pareto_coords,
            "distances": distances,
            "closeness": closeness,
        }

    combined = np.vstack([coords, pareto_coords])
    finite_combined = combined[np.all(np.isfinite(combined), axis=1)]
    if len(finite_combined) == 0:
        return {
            "coords": coords,
            "pareto_coords": pareto_coords,
            "distances": distances,
            "closeness": closeness,
        }
    mins = np.min(finite_combined, axis=0)
    spans = np.max(finite_combined, axis=0) - mins
    spans = np.where(spans > 1.0e-12, spans, 1.0)

    scaled_coords = (coords - mins) / spans
    scaled_pareto = (pareto_coords - mins) / spans
    finite_coords = np.all(np.isfinite(scaled_coords), axis=1)
    finite_pareto = np.all(np.isfinite(scaled_pareto), axis=1)
    if np.any(finite_coords) and np.any(finite_pareto):
        delta = scaled_coords[finite_coords, None, :] - scaled_pareto[finite_pareto][None, :, :]
        distances[finite_coords] = np.sqrt(np.sum(delta * delta, axis=2)).min(axis=1)

    finite = distances[np.isfinite(distances)]
    if len(finite) > 0:
        ref_distance = float(np.percentile(finite, 90.0))
        if ref_distance <= 1.0e-12:
            ref_distance = max(float(np.max(finite)), 1.0)
        closeness = 1.0 - np.clip(distances / ref_distance, 0.0, 1.0)
        closeness = np.power(closeness, 0.72)

    return {
        "coords": coords,
        "pareto_coords": pareto_coords,
        "distances": distances,
        "closeness": closeness,
    }


def candidate_metrics_fieldnames() -> List[str]:
    return list(_CANDIDATE_METRICS_FIELDNAMES)


def candidate_metrics_row(
    ctx: ReuseGeneticEvaluationContext,
    evaluation: CandidateEvaluation,
    candidate_id: Optional[int] = None,
) -> Dict[str, Any]:
    resolved_candidate_id = evaluation.meta.get("candidate_id", None) if candidate_id is None else candidate_id
    if resolved_candidate_id is not None:
        resolved_candidate_id = int(resolved_candidate_id)
    return {
        "candidate_id": resolved_candidate_id,
        "generation": evaluation.meta.get("generation", None),
        "feasible": int(bool(evaluation.feasible)),
        "error": evaluation.meta.get("error", ""),
        "total_violation": float(evaluation.total_violation),
        "normalized_violation": float(evaluation.normalized_violation),
        "mass": _objective_value(evaluation, "mass"),
        "cost": _objective_value(evaluation, "cost"),
        "ec": evaluation.meta.get("ec", None),
        "pi_mass": _objective_value(evaluation, "pi_mass"),
        "pi_ec": _objective_value(evaluation, "pi_ec"),
        "pi_mr": _objective_value(evaluation, "pi_mr"),
        "pi_cost": _objective_value(evaluation, "pi_cost"),
        "reuse_ratio": _objective_value(evaluation, "reuse_ratio"),
        "reuse_efficiency": _objective_value(evaluation, "reuse_efficiency"),
        "reuse_ratio_members": _objective_value(evaluation, "reuse_ratio_members"),
        "dcr_max": evaluation.meta.get("dcr_max", None),
        "dcr_mean": evaluation.meta.get("dcr_mean", None),
        "max_abs_uz": evaluation.meta.get("max_abs_uz", None),
        "merchant_rankine": evaluation.meta.get("merchant_rankine", None),
        "reused_mass": evaluation.meta.get("reused_mass", None),
        "new_mass": evaluation.meta.get("new_mass", None),
        "new_reference_mass": evaluation.meta.get("new_reference_mass", None),
        "new_reference_mass_tonnes": evaluation.meta.get("new_reference_mass_tonnes", None),
        "mscrap": evaluation.meta.get("mscrap", None),
        "lscrap": evaluation.meta.get("lscrap", None),
        "n_cuts": evaluation.meta.get("n_cuts", None),
        "diagonal_topology_enabled": int(bool(evaluation.meta.get("diagonal_topology_enabled", False))),
        "diagonal_topology_pattern": evaluation.meta.get("diagonal_topology_pattern", ""),
        "diagonal_topology_group_count": evaluation.meta.get("diagonal_topology_group_count", None),
        "diagonal_candidate_count": evaluation.meta.get("diagonal_candidate_count", None),
        "diagonal_active_count": evaluation.meta.get("diagonal_active_count", None),
        "new_reference_ec": evaluation.meta.get("new_reference_ec", None),
        "new_reference_cost": evaluation.meta.get("new_reference_cost", None),
        "new_reference_dcr_max": evaluation.meta.get("new_reference_dcr_max", None),
        "new_reference_dcr_mean": evaluation.meta.get("new_reference_dcr_mean", None),
        "new_reference_max_abs_uz": evaluation.meta.get("new_reference_max_abs_uz", None),
        "new_reference_lambda_ref": evaluation.meta.get("new_reference_lambda_ref", None),
        "new_reference_merchant_rankine": evaluation.meta.get("new_reference_merchant_rankine", None),
        "ga_dcr_limit_used": evaluation.meta.get("ga_dcr_limit_used", None),
        "ga_disp_limit_used": evaluation.meta.get("ga_disp_limit_used", None),
        "ga_buckling_lambda_min_used": evaluation.meta.get("ga_buckling_lambda_min_used", None),
        "ga_merchant_rankine_min_used": evaluation.meta.get("ga_merchant_rankine_min_used", None),
        "opt_dcr_limit_used": evaluation.meta.get("opt_dcr_limit_used", None),
        "opt_dlim_used": evaluation.meta.get("opt_dlim_used", None),
        "opt_buckling_lambda_min_used": evaluation.meta.get("opt_buckling_lambda_min_used", None),
        "opt_merchant_rankine_min_used": evaluation.meta.get("opt_merchant_rankine_min_used", None),
        "g_dcr": evaluation.constraints.get("g_dcr", None),
        "g_disp": evaluation.constraints.get("g_disp", None),
        "g_buckling": evaluation.constraints.get("g_buckling", None),
        "g_merchant_rankine": evaluation.constraints.get("g_merchant_rankine", None),
        "g_analysis": evaluation.constraints.get("g_analysis", None),
        "delta_z_max": evaluation.meta.get("max_abs_uz", None),
        "lambda_ref": evaluation.lambda_ref,
        "time_geom_s": evaluation.meta.get("time_geom_s", None),
        "time_mech_s": evaluation.meta.get("time_mech_s", None),
        "time_mech_inner_total_s": evaluation.meta.get("time_mech_inner_total_s", None),
        "n_mech_state_calls": evaluation.meta.get("n_mech_state_calls", None),
        "time_mech_state_total_s": evaluation.meta.get("time_mech_state_total_s", None),
        "n_mech_base_calls": evaluation.meta.get("n_mech_base_calls", None),
        "time_mech_base_s": evaluation.meta.get("time_mech_base_s", None),
        "n_mech_buckling_calls": evaluation.meta.get("n_mech_buckling_calls", None),
        "time_mech_buckling_s": evaluation.meta.get("time_mech_buckling_s", None),
        "time_match_s": evaluation.meta.get("time_match_s", None),
        "time_match_single_search_s": evaluation.meta.get("time_match_single_search_s", None),
        "time_match_group_search_s": evaluation.meta.get("time_match_group_search_s", None),
        "time_match_stock_update_s": evaluation.meta.get("time_match_stock_update_s", None),
        "n_match_single_checked": evaluation.meta.get("n_match_single_checked", None),
        "n_match_group_checked": evaluation.meta.get("n_match_group_checked", None),
        "time_recheck_new_s": evaluation.meta.get("time_recheck_new_s", None),
        "time_recheck_s": evaluation.meta.get("time_recheck_s", None),
        "time_total_s": evaluation.meta.get("time_total_s", None),
        "decision_labels": json.dumps(decision_variable_labels(ctx)),
        "decision_vector": json.dumps([float(v) for v in evaluation.x]),
    }


def export_candidate_metrics_csv(
    path: str,
    ctx: ReuseGeneticEvaluationContext,
    evaluations: Sequence[CandidateEvaluation],
) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=candidate_metrics_fieldnames())
        writer.writeheader()
        for evaluation in evaluations:
            writer.writerow(candidate_metrics_row(ctx, evaluation))


def export_pareto_front_json(
    path: str,
    evaluations: Sequence[CandidateEvaluation],
    objective_names: Sequence[str],
) -> None:
    with open(path, "w", encoding="utf-8") as fobj:
        json_ready = [_candidate_record(ev, objective_names) for ev in evaluations]
        json.dump(json_ready, fobj, indent=2)


def export_pareto_front_csv(
    path: str,
    evaluations: Sequence[CandidateEvaluation],
    objective_names: Sequence[str],
) -> None:
    fieldnames = [
        "rank",
        "candidate_id",
        "feasible",
        "total_violation",
        "normalized_violation",
        *list(objective_names),
    ]
    for extra_name in (
        "cost",
        "ec",
        "pi_mass",
        "pi_ec",
        "pi_mr",
        "pi_cost",
        "dcr_mean",
        "reused_mass",
        "new_mass",
        "new_reference_mass",
        "new_reference_ec",
        "new_reference_merchant_rankine",
        "mscrap",
        "lscrap",
        "n_cuts",
        "g_dcr",
        "g_disp",
        "g_buckling",
        "g_merchant_rankine",
        "g_analysis",
        "lambda_ref",
        "decision_vector",
    ):
        if extra_name not in fieldnames:
            fieldnames.append(extra_name)
    with open(path, "w", newline="", encoding="utf-8") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=fieldnames)
        writer.writeheader()
        for idx, evaluation in enumerate(evaluations, start=1):
            row = {
                "rank": int(idx),
                "candidate_id": evaluation.meta.get("candidate_id", None),
                "feasible": int(bool(evaluation.feasible)),
                "total_violation": float(evaluation.total_violation),
                "normalized_violation": float(evaluation.normalized_violation),
                "cost": evaluation.objectives.get("cost", None),
                "ec": evaluation.meta.get("ec", None),
                "pi_mass": evaluation.objectives.get("pi_mass", None),
                "pi_ec": evaluation.objectives.get("pi_ec", None),
                "pi_mr": evaluation.objectives.get("pi_mr", None),
                "pi_cost": evaluation.objectives.get("pi_cost", None),
                "dcr_mean": evaluation.meta.get("dcr_mean", None),
                "reused_mass": evaluation.meta.get("reused_mass", None),
                "new_mass": evaluation.meta.get("new_mass", None),
                "new_reference_mass": evaluation.meta.get("new_reference_mass", None),
                "new_reference_ec": evaluation.meta.get("new_reference_ec", None),
                "new_reference_merchant_rankine": evaluation.meta.get("new_reference_merchant_rankine", None),
                "mscrap": evaluation.meta.get("mscrap", None),
                "lscrap": evaluation.meta.get("lscrap", None),
                "n_cuts": evaluation.meta.get("n_cuts", None),
                "lambda_ref": None if evaluation.lambda_ref is None else float(evaluation.lambda_ref),
                "decision_vector": " ".join(f"{float(v):.6f}" for v in evaluation.x),
            }
            row.update(_objective_display_value(evaluation, objective_names))
            row.update({str(k): float(v) for k, v in evaluation.constraints.items()})
            writer.writerow(row)


def _genetic_output_root(ctx: ReuseGeneticEvaluationContext) -> Optional[str]:
    if not ctx.save_candidate_artifacts or not ctx.output_dir:
        return None
    genetic_dir = os.path.abspath(str(ctx.output_dir))
    if os.path.basename(genetic_dir).lower() == "genetic_search":
        return os.path.dirname(genetic_dir)
    return genetic_dir


def _write_generation_checkpoint(
    ctx: ReuseGeneticEvaluationContext,
    settings: ReuseGeneticAlgorithmSettings,
    *,
    method: str,
    generation: int,
    objective_names: Sequence[str],
    history: Sequence[Mapping[str, float]],
    cache: Mapping[Tuple[float, ...], CandidateEvaluation],
    n_evaluations: Optional[int] = None,
    cache_hits: Optional[int] = None,
) -> None:
    output_root = _genetic_output_root(ctx)
    if not output_root:
        return

    all_evaluations = list(cache.values())
    if not all_evaluations:
        return

    os.makedirs(output_root, exist_ok=True)
    genetic_dir = os.path.abspath(str(ctx.output_dir))
    os.makedirs(genetic_dir, exist_ok=True)

    global_pareto = _extract_pareto_front(all_evaluations, objective_names)
    displayed_pareto = list(global_pareto)

    history_path = os.path.join(output_root, "reuse_genetic_history.json")
    pareto_json_path = os.path.join(output_root, "reuse_genetic_pareto_front.json")
    pareto_csv_path = os.path.join(output_root, "reuse_genetic_pareto_front.csv")
    pareto_global_json_path = os.path.join(output_root, "reuse_genetic_pareto_front_global.json")
    pareto_global_csv_path = os.path.join(output_root, "reuse_genetic_pareto_front_global.csv")
    pareto_metrics_csv_path = os.path.join(output_root, "pareto_candidate_metrics.csv")
    completed_metrics_csv_path = os.path.join(genetic_dir, "genetic_candidate_metrics_completed.csv")
    summary_path = os.path.join(output_root, "reuse_genetic_checkpoint_summary.json")
    pareto_plot_path = os.path.join(output_root, "reuse_genetic_pareto_front.png")
    parallel_coords_plot_path = os.path.join(output_root, "reuse_genetic_parallel_coords.png")
    pareto_metrics_parallel_plot_path = os.path.join(output_root, "reuse_genetic_parallel_coords_pareto_metrics.png")
    fixed_parallel_plot_path = os.path.join(output_root, "reuse_genetic_parallel_coords_pareto_fixed_base.png")

    with open(history_path, "w", encoding="utf-8") as fobj:
        json.dump(list(history or []), fobj, indent=2)
    export_pareto_front_json(pareto_json_path, displayed_pareto, objective_names)
    export_pareto_front_csv(pareto_csv_path, displayed_pareto, objective_names)
    export_pareto_front_json(pareto_global_json_path, global_pareto, objective_names)
    export_pareto_front_csv(pareto_global_csv_path, global_pareto, objective_names)
    export_candidate_metrics_csv(pareto_metrics_csv_path, ctx, displayed_pareto)
    export_candidate_metrics_csv(completed_metrics_csv_path, ctx, all_evaluations)

    pareto_plot_written = plot_pareto_front(
        pareto_plot_path,
        displayed_pareto,
        objective_names,
        all_evaluations=all_evaluations,
    )
    parallel_coords_written = None
    pareto_metrics_parallel_written = None
    fixed_parallel_written = None
    try:
        parallel_coords_written = _checkpoint_save_candidate_parallel_plot(parallel_coords_plot_path, all_evaluations)
        pareto_metrics_parallel_written = _checkpoint_save_pareto_metrics_parallel_plot(
            pareto_metrics_parallel_plot_path,
            displayed_pareto,
        )
        fixed_parallel_written = _checkpoint_save_fixed_parallel_plot(fixed_parallel_plot_path, displayed_pareto)
    except Exception as exc:
        print(
            "# GENETIC | warning | generation checkpoint plots failed | "
            f"generation={int(generation)} | error={type(exc).__name__}: {exc}",
            flush=True,
        )

    with open(summary_path, "w", encoding="utf-8") as fobj:
        json.dump(
            {
                "method": method,
                "checkpoint_generation": int(generation),
                "objective_names": list(objective_names),
                "n_evaluations": None if n_evaluations is None else int(n_evaluations),
                "cache_hits": None if cache_hits is None else int(cache_hits),
                "all_evaluations_completed": len(all_evaluations),
                "pareto_front_global_size": len(global_pareto),
                "candidate_metrics_completed_csv": completed_metrics_csv_path,
                "pareto_plot": pareto_plot_path if pareto_plot_written else None,
                "parallel_coords_plot": parallel_coords_written,
                "pareto_metrics_parallel_coords_plot": pareto_metrics_parallel_written,
                "fixed_parallel_coords_plot": fixed_parallel_written,
            },
            fobj,
            indent=2,
        )


def _safe_generation_checkpoint(
    ctx: ReuseGeneticEvaluationContext,
    settings: ReuseGeneticAlgorithmSettings,
    *,
    method: str,
    generation: int,
    objective_names: Sequence[str],
    history: Sequence[Mapping[str, float]],
    cache: Mapping[Tuple[float, ...], CandidateEvaluation],
    n_evaluations: Optional[int] = None,
    cache_hits: Optional[int] = None,
) -> None:
    try:
        _write_generation_checkpoint(
            ctx,
            settings,
            method=method,
            generation=generation,
            objective_names=objective_names,
            history=history,
            cache=cache,
            n_evaluations=n_evaluations,
            cache_hits=cache_hits,
        )
    except Exception as exc:
        print(
            "# GENETIC | warning | generation checkpoint failed | "
            f"generation={int(generation)} | error={type(exc).__name__}: {exc}",
            flush=True,
        )


def _checkpoint_metric_value(evaluation: CandidateEvaluation, *keys: str) -> float:
    for key in keys:
        raw_value = evaluation.lambda_ref if key == "lambda_ref" else None
        if raw_value in {None, ""}:
            raw_value = evaluation.objectives.get(key, None)
        if raw_value in {None, ""}:
            raw_value = evaluation.meta.get(key, None)
        try:
            value = float(raw_value)
            if math.isfinite(value):
                return value
        except Exception:
            pass
        if key == "mass":
            try:
                reused_mass = float(evaluation.meta.get("reused_mass", float("nan")))
                new_mass = float(evaluation.meta.get("new_mass", float("nan")))
                if math.isfinite(reused_mass) and math.isfinite(new_mass):
                    return reused_mass + new_mass
            except Exception:
                pass
    return float("nan")


def _checkpoint_fixed_value(fixed_reference: Mapping[str, object], *keys: str) -> float:
    summary = fixed_reference.get("summary", {}) if isinstance(fixed_reference, Mapping) else {}
    summary = dict(summary or {}) if isinstance(summary, Mapping) else {}
    for key in keys:
        raw_value = fixed_reference.get(key, None) if isinstance(fixed_reference, Mapping) else None
        if raw_value in {None, ""}:
            raw_value = summary.get(key, None)
        try:
            value = float(raw_value)
            if math.isfinite(value):
                return value
        except Exception:
            pass
    return float("nan")


def _checkpoint_format_float(value: float) -> str:
    abs_value = abs(float(value))
    if abs_value >= 1000.0 or (abs_value > 0.0 and abs_value < 0.1):
        return f"{value:.2e}"
    if abs_value >= 100.0:
        return f"{value:.0f}"
    if abs_value >= 10.0:
        return f"{value:.1f}"
    if abs_value >= 1.0:
        return f"{value:.2f}"
    return f"{value:.3f}"


def _checkpoint_render_parallel(ax, labels: Sequence[str], value_rows: Sequence[Sequence[float]], *, fixed_row=None, fixed_label="Fixed base") -> bool:
    raw = np.asarray(value_rows, dtype=float) if value_rows else np.empty((0, len(labels)), dtype=float)
    if fixed_row is not None:
        fixed = np.asarray(fixed_row, dtype=float)
        raw_for_ranges = np.vstack([raw, fixed.reshape(1, -1)]) if raw.size else fixed.reshape(1, -1)
    else:
        fixed = None
        raw_for_ranges = raw
    if not raw_for_ranges.size:
        ax.text(0.5, 0.5, "No finite metrics available.", transform=ax.transAxes, ha="center", va="center")
        ax.set_axis_off()
        return False

    mins = raw_for_ranges.min(axis=0)
    maxs = raw_for_ranges.max(axis=0)
    spans = np.where((maxs - mins) > 1e-12, (maxs - mins), 1.0)
    x = np.arange(len(labels), dtype=float)
    tick_levels = np.linspace(0.0, 1.0, 5)

    if raw.size:
        normalized = (raw - mins) / spans
        for row in normalized:
            ax.plot(x, row, color="#4f5d6a", alpha=0.30, linewidth=1.0, zorder=1)
    if fixed is not None:
        fixed_norm = (fixed - mins) / spans
        ax.plot(x, fixed_norm, color="#c62828", alpha=0.95, linewidth=2.2, zorder=5, label=fixed_label)

    for idx, label in enumerate(labels):
        x_pos = x[idx]
        ax.axvline(x_pos, color="black", linewidth=1.1, alpha=1.0, zorder=2)
        axis_tick_values = mins[idx] + tick_levels * (maxs[idx] - mins[idx])
        if abs(maxs[idx] - mins[idx]) <= 1e-12:
            axis_tick_values = np.full_like(tick_levels, mins[idx], dtype=float)
        for level, tick_value in zip(tick_levels, axis_tick_values):
            ax.plot([x_pos - 0.028, x_pos], [level, level], color="black", linewidth=0.9, zorder=3)
            ax.text(
                x_pos - 0.08,
                level,
                _checkpoint_format_float(float(tick_value)),
                ha="right",
                va="center",
                fontsize=8,
                color="black",
                zorder=4,
            )
        ax.text(x_pos, -0.12, label, ha="center", va="top", fontsize=10, fontweight="bold", color="black", zorder=4)

    if fixed is not None:
        ax.legend(loc="upper right", frameon=False)
    ax.set_xlim(-0.4, len(labels) - 1 + 0.35)
    ax.set_ylim(-0.18, 1.08)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor("white")
    return True


def _checkpoint_save_parallel_plot(path: str, labels: Sequence[str], value_rows: Sequence[Sequence[float]], *, fixed_row=None, fixed_label="Fixed base") -> Optional[str]:
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        from matplotlib.figure import Figure
    except Exception:
        return None
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    fig = Figure(figsize=(10.8, 3.55), dpi=160)
    ax = fig.add_subplot(111)
    written = _checkpoint_render_parallel(ax, labels, value_rows, fixed_row=fixed_row, fixed_label=fixed_label)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.86, bottom=0.26)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    return path if written else None


def _checkpoint_save_candidate_parallel_plot(path: str, evaluations: Sequence[CandidateEvaluation]) -> Optional[str]:
    metric_defs = (
        ("candidate_id", ("candidate_id",)),
        ("%Riuso", ("reuse_ratio", "reuse_ratio_length", "reuse_ratio_members")),
        ("ReuseEff", ("reuse_efficiency",)),
        ("Massa", ("mass",)),
        ("Cost", ("cost",)),
        ("EC", ("ec",)),
        ("PI_M", ("pi_mass",)),
        ("PI_EC", ("pi_ec",)),
        ("PI_MR", ("pi_mr",)),
        ("DCRmax", ("dcr_max",)),
        ("DCRmean", ("dcr_mean",)),
        ("Lambda", ("lambda_ref",)),
        ("MR", ("merchant_rankine",)),
        ("Mscrap", ("mscrap",)),
        ("dmax", ("max_abs_uz",)),
        ("Lscrap", ("lscrap",)),
        ("n_cut", ("n_cuts", "ncut", "n_cut")),
    )
    labels = [label for label, _ in metric_defs]
    rows = []
    for evaluation in evaluations:
        values = []
        for label, keys in metric_defs:
            value = _checkpoint_metric_value(evaluation, *keys)
            if label in {"%Riuso", "ReuseEff"} and math.isfinite(value) and abs(value) <= 1.0:
                value *= 100.0
            values.append(value)
        if all(math.isfinite(v) for v in values):
            rows.append(values)
    return _checkpoint_save_parallel_plot(path, labels, rows)


def _checkpoint_save_pareto_metrics_parallel_plot(path: str, pareto_front: Sequence[CandidateEvaluation]) -> Optional[str]:
    metric_defs = (
        ("Massa", ("mass",)),
        ("EC", ("ec",)),
        ("PI_Mass", ("pi_mass",)),
        ("PI_Cost", ("pi_cost",)),
        ("PI_EC", ("pi_ec",)),
        ("%Riuso", ("reuse_ratio", "reuse_ratio_length", "reuse_ratio_members")),
        ("Lscrap", ("lscrap",)),
        ("ncut", ("n_cuts", "ncut", "n_cut")),
    )
    labels = [label for label, _ in metric_defs]
    rows = []
    for evaluation in pareto_front:
        values = []
        for label, keys in metric_defs:
            value = _checkpoint_metric_value(evaluation, *keys)
            if label == "%Riuso" and math.isfinite(value) and abs(value) <= 1.0:
                value *= 100.0
            values.append(value)
        if all(math.isfinite(v) for v in values):
            rows.append(values)
    return _checkpoint_save_parallel_plot(path, labels, rows)


def _checkpoint_fixed_reference_from_pareto(pareto_front: Sequence[CandidateEvaluation]) -> Dict[str, object]:
    for evaluation in pareto_front:
        meta = dict(evaluation.meta or {})
        return {
            "label": meta.get("comparison_reference_label", "fixed_base_new"),
            "mass": meta.get("comparison_reference_mass", meta.get("new_reference_mass", None)),
            "ec": meta.get("comparison_reference_ec", meta.get("new_reference_ec", None)),
            "cost": meta.get("comparison_reference_cost", meta.get("new_reference_cost", None)),
            "summary": {
                "dcr_max": meta.get("comparison_reference_dcr_max", meta.get("new_reference_dcr_max", None)),
                "dcr_mean": meta.get("comparison_reference_dcr_mean", meta.get("new_reference_dcr_mean", None)),
                "max_abs_uz": meta.get("comparison_reference_max_abs_uz", meta.get("new_reference_max_abs_uz", None)),
                "lambda_ref": meta.get("comparison_reference_lambda_ref", meta.get("new_reference_lambda_ref", None)),
                "merchant_rankine": meta.get("comparison_reference_merchant_rankine", meta.get("new_reference_merchant_rankine", None)),
            },
        }
    return {}


def _checkpoint_save_fixed_parallel_plot(path: str, pareto_front: Sequence[CandidateEvaluation]) -> Optional[str]:
    fixed_reference = _checkpoint_fixed_reference_from_pareto(pareto_front)
    if not fixed_reference:
        return None
    metric_defs = (
        ("Massa", ("mass",), ("mass",)),
        ("EC", ("ec",), ("ec",)),
        ("Cost", ("cost",), ("cost",)),
        ("DCRmax", ("dcr_max",), ("dcr_max",)),
        ("DCRmean", ("dcr_mean",), ("dcr_mean",)),
        ("dmax", ("max_abs_uz",), ("max_abs_uz",)),
        ("LambdaMin", ("lambda_ref",), ("lambda_ref",)),
        ("MR", ("merchant_rankine",), ("merchant_rankine",)),
    )
    labels = [label for label, _, _ in metric_defs]
    rows = []
    for evaluation in pareto_front:
        values = [_checkpoint_metric_value(evaluation, *candidate_keys) for _, candidate_keys, _ in metric_defs]
        if all(math.isfinite(v) for v in values):
            rows.append(values)
    fixed_values = [_checkpoint_fixed_value(fixed_reference, *fixed_keys) for _, _, fixed_keys in metric_defs]
    if not all(math.isfinite(v) for v in fixed_values):
        fixed_values = None
    fixed_label = str(fixed_reference.get("label", "Fixed base") or "Fixed base")
    return _checkpoint_save_parallel_plot(path, labels, rows, fixed_row=fixed_values, fixed_label=fixed_label)


def plot_pareto_front(
    path: str,
    evaluations: Sequence[CandidateEvaluation],
    objective_names: Sequence[str],
    *,
    all_evaluations: Optional[Sequence[CandidateEvaluation]] = None,
) -> bool:
    if len(objective_names) not in {2, 3} or not evaluations:
        return False
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib import colors as mcolors
    except Exception:
        return False

    pareto_coords = objective_coordinate_matrix(evaluations, objective_names)
    if pareto_coords.size == 0 or not np.all(np.isfinite(pareto_coords)):
        return False

    all_points = list(all_evaluations) if all_evaluations else list(evaluations)
    feasible = [ev for ev in all_points if ev.feasible]
    proximity = pareto_proximity_scores(feasible, evaluations, objective_names)
    feasible_coords = proximity["coords"]
    feasible_closeness = proximity["closeness"]
    display_coords = np.vstack([feasible_coords, pareto_coords]) if len(feasible_coords) > 0 else pareto_coords

    labels = list(objective_names)
    pareto_color = "#c62828"
    pareto_edge = "#7f0000"
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "pareto_proximity",
        ["#7d92ab", "#bcc7d3", "#ecd08a", "#ee8b63"],
    )
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

    def _set_axis_limits_2d(ax, coords: np.ndarray, ix: int, iy: int) -> None:
        x_vals = coords[:, ix]
        y_vals = coords[:, iy]
        x_min = float(np.min(x_vals))
        x_max = float(np.max(x_vals))
        y_min = float(np.min(y_vals))
        y_max = float(np.max(y_vals))
        if abs(x_max - x_min) <= 1e-12:
            pad = max(1.0, abs(x_min) * 0.05)
        else:
            pad = max(abs(x_max - x_min) * 0.06, 1e-9)
        ax.set_xlim(x_min - pad, x_max + pad)
        if abs(y_max - y_min) <= 1e-12:
            pad = max(1.0, abs(y_min) * 0.05)
        else:
            pad = max(abs(y_max - y_min) * 0.06, 1e-9)
        ax.set_ylim(y_min - pad, y_max + pad)

    def _style_2d_axis(ax, ix: int, iy: int, title: str) -> None:
        ax.set_title(title, fontname="Arial")
        ax.set_xlabel(labels[ix], fontname="Arial")
        ax.set_ylabel(labels[iy], fontname="Arial")
        ax.margins(x=0.04, y=0.04)
        _set_axis_limits_2d(ax, display_coords, ix, iy)
        ax.set_facecolor("#fbfbfd")
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
            tick_label.set_fontname("Arial")

    def _scatter_projection(ax, ix: int, iy: int, title: str, *, show_legend: bool = False) -> None:
        if len(feasible_coords) > 0:
            ax.scatter(
                feasible_coords[:, ix],
                feasible_coords[:, iy],
                c=feasible_closeness,
                cmap=cmap,
                norm=norm,
                s=32,
                alpha=0.88,
                edgecolors="white",
                linewidths=0.25,
                label="Feasible",
            )
        ax.scatter(
            pareto_coords[:, ix],
            pareto_coords[:, iy],
            c=pareto_color,
            s=60,
            edgecolors=pareto_edge,
            linewidths=0.8,
            label="Pareto front",
        )
        _style_2d_axis(ax, ix, iy, title)
        if show_legend:
            ax.legend(loc="best")

    def _path_with_suffix(base_path: str, suffix: str) -> str:
        root, ext = os.path.splitext(base_path)
        return f"{root}{suffix}{ext or '.png'}"

    def _label_slug(label: str) -> str:
        safe = "".join(ch if ch.isalnum() else "_" for ch in str(label))
        safe = "_".join(part for part in safe.split("_") if part)
        return safe.lower() or "obj"

    if len(objective_names) == 2:
        fig, ax = plt.subplots(figsize=(7, 5))
        if len(feasible_coords) > 0:
            feasible_artist = ax.scatter(
                feasible_coords[:, 0],
                feasible_coords[:, 1],
                c=feasible_closeness,
                cmap=cmap,
                norm=norm,
                s=34,
                alpha=0.88,
                edgecolors="white",
                linewidths=0.28,
                label="Feasible",
            )
        ax.scatter(
            pareto_coords[:, 0],
            pareto_coords[:, 1],
            c=pareto_color,
            s=66,
            edgecolors=pareto_edge,
            linewidths=0.85,
            label="Pareto front",
        )
        _style_2d_axis(ax, 0, 1, "Pareto Front and Evaluated Solutions")
        if len(feasible_coords) > 0:
            cbar = fig.colorbar(feasible_artist, ax=ax, fraction=0.05, pad=0.035)
            cbar.set_label("Closer to Pareto", fontname="Arial")
    else:
        fig = plt.figure(figsize=(9.2, 6.8))
        ax3d = fig.add_subplot(111, projection="3d")
        if len(feasible_coords) > 0:
            feasible_artist = ax3d.scatter(
                feasible_coords[:, 0],
                feasible_coords[:, 1],
                feasible_coords[:, 2],
                c=feasible_closeness,
                cmap=cmap,
                norm=norm,
                s=32,
                alpha=0.86,
                edgecolors="white",
                linewidths=0.22,
                depthshade=True,
                label="Feasible",
            )
        ax3d.scatter(
            pareto_coords[:, 0],
            pareto_coords[:, 1],
            pareto_coords[:, 2],
            c=pareto_color,
            s=66,
            depthshade=True,
            edgecolors=pareto_edge,
            linewidths=0.75,
            label="Pareto front",
        )
        ax3d.set_title("Pareto Front and Evaluated Solutions", fontname="Arial")
        ax3d.set_xlabel(labels[0], fontname="Arial")
        ax3d.set_ylabel(labels[1], fontname="Arial")
        ax3d.set_zlabel(labels[2], fontname="Arial")
        for tick_label in ax3d.get_xticklabels() + ax3d.get_yticklabels() + ax3d.get_zticklabels():
            tick_label.set_fontname("Arial")
        ax3d.grid(False)
        try:
            ax3d.xaxis.set_pane_color((0.98, 0.985, 0.99, 1.0))
            ax3d.yaxis.set_pane_color((0.98, 0.985, 0.99, 1.0))
            ax3d.zaxis.set_pane_color((0.98, 0.985, 0.99, 1.0))
        except Exception:
            pass
        ax3d.legend(loc="best")
        if len(feasible_coords) > 0:
            cbar = fig.colorbar(feasible_artist, ax=ax3d, fraction=0.045, pad=0.08)
            cbar.set_label("Closer to Pareto", fontname="Arial")
        fig.subplots_adjust(left=0.04, right=0.88, bottom=0.06, top=0.94)
        save_pareto_figure(fig, path, is_3d=len(objective_names) == 3)
        plt.close(fig)

        for ix, iy in ((0, 1), (0, 2), (1, 2)):
            projection_path = _path_with_suffix(
                path,
                f"_{_label_slug(labels[ix])}_vs_{_label_slug(labels[iy])}",
            )
            fig2, ax2 = plt.subplots(figsize=(7, 5))
            _scatter_projection(ax2, ix, iy, f"{labels[ix]} vs {labels[iy]}", show_legend=True)
            save_pareto_figure(fig2, projection_path)
            plt.close(fig2)
        return True

    ax.legend(loc="best")
    save_pareto_figure(fig, path, is_3d=len(objective_names) == 3)
    plt.close(fig)
    return True


def _tournament_pick(rng: random.Random, scores: Sequence[float], k: int = 3) -> int:
    choices = [rng.randrange(len(scores)) for _ in range(max(1, int(k)))]
    return min(choices, key=lambda idx: scores[idx])


def _clip_vector(vector: Sequence[float], xl: Sequence[float], xu: Sequence[float]) -> List[float]:
    return [min(max(float(v), float(lo)), float(hi)) for v, lo, hi in zip(vector, xl, xu)]


def _mutate_vector(
    rng: random.Random,
    vector: Sequence[float],
    xl: Sequence[float],
    xu: Sequence[float],
    mutation_prob: float,
    mutation_scale: float,
) -> List[float]:
    out = list(vector)
    for idx, (lo, hi) in enumerate(zip(xl, xu)):
        if rng.random() <= float(mutation_prob):
            span = float(hi) - float(lo)
            sigma = max(1e-12, float(mutation_scale) * span)
            out[idx] = float(out[idx]) + rng.gauss(0.0, sigma)
    return _clip_vector(out, xl, xu)


def _blend_crossover(rng: random.Random, p1: Sequence[float], p2: Sequence[float], xl: Sequence[float], xu: Sequence[float]) -> Tuple[List[float], List[float]]:
    c1: List[float] = []
    c2: List[float] = []
    for a, b, lo, hi in zip(p1, p2, xl, xu):
        alpha = rng.random()
        v1 = alpha * float(a) + (1.0 - alpha) * float(b)
        v2 = alpha * float(b) + (1.0 - alpha) * float(a)
        c1.append(min(max(v1, float(lo)), float(hi)))
        c2.append(min(max(v2, float(lo)), float(hi)))
    return c1, c2


def _random_population(rng: random.Random, n_pop: int, xl: Sequence[float], xu: Sequence[float]) -> List[List[float]]:
    out: List[List[float]] = []
    for _ in range(int(n_pop)):
        out.append([rng.uniform(float(lo), float(hi)) for lo, hi in zip(xl, xu)])
    return out


def _resolved_worker_count(settings: Optional[ReuseGeneticAlgorithmSettings]) -> int:
    if settings is None:
        return 2
    try:
        workers = int(settings.n_workers)
    except Exception:
        workers = 2
    return max(2, workers)


def _progress_heartbeat_s(settings: Optional[ReuseGeneticAlgorithmSettings]) -> float:
    if settings is None:
        return 30.0
    try:
        value = float(settings.progress_heartbeat_s)
    except Exception:
        value = 30.0
    return max(5.0, value)


def _candidate_stall_timeout_s(settings: Optional[ReuseGeneticAlgorithmSettings]) -> Optional[float]:
    if settings is None:
        return None
    raw_value = settings.candidate_stall_timeout_s
    if raw_value in {None, 0, 0.0, ""}:
        return None
    try:
        value = float(raw_value)
    except Exception:
        return None
    return value if value > 0.0 else None


def _ga_parallel_is_disabled(ctx: ReuseGeneticEvaluationContext) -> bool:
    return False


def _disable_ga_parallel(ctx: ReuseGeneticEvaluationContext, *, reason: str, settings: Optional[ReuseGeneticAlgorithmSettings] = None) -> None:
    if settings is None or int(getattr(settings, "verbose", 1) or 0) >= 1:
        print(
            f"# GENETIC | warning | parallel issue | reason={reason} | skipped_pending=1 | keep_configured_workers=true",
            flush=True,
        )


def _make_timeout_candidate_evaluation(
    ctx: ReuseGeneticEvaluationContext,
    vector: Sequence[float],
    *,
    generation: Optional[int],
    reason: str,
    no_progress_s: Optional[float] = None,
) -> CandidateEvaluation:
    """Return a penalized evaluation for a candidate that could not be evaluated.

    This is an emergency-safe fallback: a timeout/stalled worker must not abort
    the whole GA run. The candidate is marked infeasible and receives very large
    constraint violations, so pymoo should discard it naturally.
    """
    penalty = 1.0e9
    meta: Dict[str, Any] = {
        "error": str(reason),
        "failed": True,
        "timeout": True,
        "skipped_by_ga_timeout": True,
    }
    if generation is not None:
        meta["generation"] = int(generation)
    if no_progress_s is not None:
        meta["no_progress_s"] = float(no_progress_s)
    try:
        meta["decision_vector_length"] = int(len(vector))
    except Exception:
        pass

    return CandidateEvaluation(
        x=tuple(float(v) for v in vector),
        feasible=False,
        objectives={},
        constraints={
            "g_dcr": penalty,
            "g_disp": penalty,
            "g_buckling": penalty,
            "g_merchant_rankine": penalty,
            "g_analysis": penalty,
        },
        moved_G=None,
        moved_M=None,
        opt_result=None,
        reuse_result=None,
        final_state=None,
        section_by_member=None,
        lambda_ref=None,
        meta=meta,
    )



def _terminal_progress_every(settings: Optional[ReuseGeneticAlgorithmSettings]) -> int:
    if settings is None:
        return 50
    try:
        value = int(getattr(settings, "terminal_progress_every", 50))
    except Exception:
        return 50
    return max(1, value)


def _should_print_terminal_candidate_progress(
    completed: int,
    total: int,
    settings: Optional[ReuseGeneticAlgorithmSettings],
) -> bool:
    if settings is None or int(settings.verbose) < 1:
        return False
    every = _terminal_progress_every(settings)
    return completed == 1 or completed == total or completed % every == 0

def _write_candidate_progress(
    ctx: ReuseGeneticEvaluationContext,
    *,
    generation: Optional[int],
    total: int,
    completed: int,
    workers: int,
    status: str,
    pending: Optional[int] = None,
    last_candidate_id: Optional[int] = None,
    no_progress_s: Optional[float] = None,
    message: Optional[str] = None,
) -> None:
    if not ctx.save_candidate_artifacts or not ctx.output_dir:
        return

    now = time.time()
    payload: Dict[str, Any] = {
        "status": str(status),
        "generation": None if generation is None else int(generation),
        "completed": int(completed),
        "total": int(total),
        "pending": None if pending is None else int(pending),
        "workers": int(workers),
        "last_candidate_id": last_candidate_id,
        "no_progress_s": None if no_progress_s is None else float(no_progress_s),
        "updated_at_unix": float(now),
        "updated_at_local": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
        "pid": os.getpid(),
    }
    if message:
        payload["message"] = str(message)

    output_dir = str(ctx.output_dir)
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception:
        return

    path = os.path.join(output_dir, "genetic_progress.json")
    tmp_path = f"{path}.{os.getpid()}.{time.time_ns()}.tmp"

    # Progress logging must never stop the GA. On Windows, os.replace can fail
    # with WinError 5 if the destination file is momentarily locked by VS Code,
    # Explorer preview, antivirus, sync tools, or another Python process. Retry a
    # few times and then skip this progress update instead of crashing pymoo.
    try:
        with open(tmp_path, "w", encoding="utf-8") as fobj:
            json.dump(payload, fobj, indent=2)
        last_error: Optional[BaseException] = None
        for attempt in range(8):
            try:
                os.replace(tmp_path, path)
                return
            except PermissionError as exc:
                last_error = exc
                time.sleep(0.05 * (attempt + 1))
            except OSError as exc:
                last_error = exc
                time.sleep(0.05 * (attempt + 1))
        # If Windows still keeps the file locked, leave the optimization running.
        if int(getattr(ctx, "verbose", 0) or 0) >= 3:
            print(f"# GENETIC | warning | progress file skipped | {last_error}", flush=True)
    except Exception as exc:
        if int(getattr(ctx, "verbose", 0) or 0) >= 3:
            print(f"# GENETIC | warning | progress file skipped | {exc}", flush=True)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def _finalize_candidate_evaluation(
    ctx: ReuseGeneticEvaluationContext,
    settings: Optional[ReuseGeneticAlgorithmSettings],
    evaluation: CandidateEvaluation,
    *,
    generation: Optional[int],
    artifact_state: Optional[Dict[str, int]],
) -> Optional[int]:
    if generation is not None:
        evaluation.meta["generation"] = int(generation)
    if artifact_state is None:
        return _candidate_id_value(evaluation)

    candidate_id = _candidate_id_value(evaluation)
    if candidate_id is None:
        artifact_state["next_id"] = int(artifact_state.get("next_id", 1))
        candidate_id = int(artifact_state["next_id"])
        artifact_state["next_id"] = candidate_id + 1
        evaluation.meta["candidate_id"] = candidate_id
    _persist_candidate_evaluation(ctx, evaluation, candidate_id)
    if settings is not None:
        _print_candidate_summary(settings, evaluation, candidate_id=candidate_id)
    return int(candidate_id)


def _evaluate_pending_vectors(
    ctx: ReuseGeneticEvaluationContext,
    pending_vectors: Sequence[Sequence[float]],
    *,
    settings: Optional[ReuseGeneticAlgorithmSettings] = None,
    generation: Optional[int] = None,
    executor: Optional[concurrent.futures.ProcessPoolExecutor] = None,
    on_complete: Optional[Callable[[int, CandidateEvaluation], Optional[int]]] = None,
) -> List[CandidateEvaluation]:
    if not pending_vectors:
        return []
    total = len(pending_vectors)
    workers = _resolved_worker_count(settings)
    heartbeat_s = _progress_heartbeat_s(settings)
    stall_timeout_s = _candidate_stall_timeout_s(settings)
    _write_candidate_progress(
        ctx,
        generation=generation,
        total=total,
        completed=0,
        workers=workers,
        status="running",
        pending=total,
    )
    if executor is None or len(pending_vectors) <= 1:
        results: List[CandidateEvaluation] = []
        last_candidate_id = None
        last_progress_write = time.time()
        for idx, vector in enumerate(pending_vectors):
            gen_label = "?" if generation is None else int(generation)
            evaluated = idx + 1
            if _should_print_terminal_candidate_progress(evaluated, total, settings):
                print(
                    f"# GENETIC | generation={gen_label} | evaluate candidate {evaluated}/{total} | "
                    f"workers={workers}",
                    flush=True,
                )
            with _suppress_terminal_output(_evaluation_output_is_suppressed(settings)):
                evaluation = evaluate_candidate_design(ctx, vector)
            if on_complete is not None:
                last_candidate_id = on_complete(idx, evaluation)
            results.append(evaluation)
            completed = len(results)
            now = time.time()
            if completed == total or now - last_progress_write >= heartbeat_s:
                _write_candidate_progress(
                    ctx,
                    generation=generation,
                    total=total,
                    completed=completed,
                    workers=workers,
                    status="running" if completed < total else "completed",
                    pending=total - completed,
                    last_candidate_id=last_candidate_id,
                    no_progress_s=0.0,
                )
                last_progress_write = now
        return results

    results: List[Optional[CandidateEvaluation]] = [None] * total
    progress_every = max(1, total // 10)
    completed = 0
    future_to_idx = {
        executor.submit(_evaluate_candidate_in_worker, vector): idx
        for idx, vector in enumerate(pending_vectors)
    }
    pending_futures = set(future_to_idx)
    last_completion_at = time.time()
    last_candidate_id = None
    while pending_futures:
        done, pending_futures = concurrent.futures.wait(
            pending_futures,
            timeout=heartbeat_s,
            return_when=concurrent.futures.FIRST_COMPLETED,
        )
        if not done:
            no_progress_s = time.time() - last_completion_at
            _write_candidate_progress(
                ctx,
                generation=generation,
                total=total,
                completed=completed,
                workers=workers,
                status="waiting",
                pending=len(pending_futures),
                last_candidate_id=last_candidate_id,
                no_progress_s=no_progress_s,
            )
            if stall_timeout_s is not None and no_progress_s >= stall_timeout_s:
                gen_label = "?" if generation is None else int(generation)
                skipped = len(pending_futures)
                if settings is None or int(getattr(settings, "verbose", 1) or 0) >= 1:
                    print(
                        f"# GENETIC | generation={gen_label} | stall detected | "
                        f"completed={completed}/{total} | skipped={skipped} | "
                        f"no_progress_s={no_progress_s:.1f} | continuing",
                        flush=True,
                    )

                # Do not abort the GA. Mark all still-pending candidates as
                # infeasible/penalized so pymoo can continue with the next
                # generation. This may skip more than one candidate, but it
                # protects long conference runs from losing all progress.
                for future in list(pending_futures):
                    idx = future_to_idx[future]
                    try:
                        future.cancel()
                    except Exception:
                        pass
                    evaluation = _make_timeout_candidate_evaluation(
                        ctx,
                        pending_vectors[idx],
                        generation=generation,
                        reason="candidate_stall_timeout",
                        no_progress_s=no_progress_s,
                    )
                    results[idx] = evaluation
                    if on_complete is not None:
                        last_candidate_id = on_complete(idx, evaluation)

                _write_candidate_progress(
                    ctx,
                    generation=generation,
                    total=total,
                    completed=total,
                    workers=workers,
                    status="completed_with_skips",
                    pending=0,
                    last_candidate_id=last_candidate_id,
                    no_progress_s=no_progress_s,
                    message=f"stall timeout: skipped {skipped} pending candidates",
                )

                # If a worker is stuck inside OpenSees/native code, skip the
                # pending candidates for this generation. The next generation
                # still uses the configured parallel worker count.
                _disable_ga_parallel(ctx, reason="candidate_stall_timeout", settings=settings)
                try:
                    if executor is not None:
                        executor.shutdown(wait=False, cancel_futures=True)
                except TypeError:
                    try:
                        if executor is not None:
                            executor.shutdown(wait=False)
                    except Exception:
                        pass
                except Exception:
                    pass
                pending_futures.clear()
                break
            continue

        broken_pool_detected = False
        for future in done:
            idx = future_to_idx[future]
            try:
                evaluation = future.result()
            except BaseException as exc:
                exc_name = exc.__class__.__name__
                gen_label = "?" if generation is None else int(generation)

                if exc_name == "BrokenProcessPool":
                    # A worker process died abruptly, usually because OpenSees or
                    # native code crashed/was killed. This must not abort the
                    # whole GA: mark every candidate that has not produced a
                    # valid result as infeasible and let pymoo continue.
                    remaining_indices = set()
                    remaining_indices.add(idx)
                    for pending_future in pending_futures:
                        remaining_indices.add(future_to_idx[pending_future])
                    for other_future in done:
                        other_idx = future_to_idx[other_future]
                        if results[other_idx] is None:
                            remaining_indices.add(other_idx)

                    skipped = len(remaining_indices)
                    if settings is None or int(getattr(settings, "verbose", 1) or 0) >= 1:
                        print(
                            f"# GENETIC | generation={gen_label} | broken process pool | "
                            f"completed={completed}/{total} | skipped={skipped} | continuing",
                            flush=True,
                        )

                    for fail_idx in sorted(remaining_indices):
                        if results[fail_idx] is not None:
                            continue
                        failed_eval = _make_timeout_candidate_evaluation(
                            ctx,
                            pending_vectors[fail_idx],
                            generation=generation,
                            reason="broken_process_pool",
                            no_progress_s=time.time() - last_completion_at,
                        )
                        results[fail_idx] = failed_eval
                        if on_complete is not None:
                            last_candidate_id = on_complete(fail_idx, failed_eval)

                    _write_candidate_progress(
                        ctx,
                        generation=generation,
                        total=total,
                        completed=total,
                        workers=workers,
                        status="completed_with_broken_pool_skips",
                        pending=0,
                        last_candidate_id=last_candidate_id,
                        no_progress_s=time.time() - last_completion_at,
                        message=f"broken process pool: skipped {skipped} candidates",
                    )

                    _disable_ga_parallel(ctx, reason="broken_process_pool", settings=settings)
                    try:
                        if executor is not None:
                            executor.shutdown(wait=False, cancel_futures=True)
                    except TypeError:
                        try:
                            if executor is not None:
                                executor.shutdown(wait=False)
                        except Exception:
                            pass
                    except Exception:
                        pass

                    pending_futures.clear()
                    broken_pool_detected = True
                    break

                # Non-fatal Python exception from a single candidate. Penalize
                # only that candidate and keep collecting the rest.
                if settings is None or int(getattr(settings, "verbose", 1) or 0) >= 2:
                    print(
                        f"# GENETIC | generation={gen_label} | candidate failed | "
                        f"index={idx + 1}/{total} | error={exc_name} | continuing",
                        flush=True,
                    )
                evaluation = _make_timeout_candidate_evaluation(
                    ctx,
                    pending_vectors[idx],
                    generation=generation,
                    reason=f"candidate_exception:{exc_name}",
                    no_progress_s=None,
                )

            results[idx] = evaluation
            if on_complete is not None:
                last_candidate_id = on_complete(idx, evaluation)
            completed += 1
            last_completion_at = time.time()
            if _should_print_terminal_candidate_progress(completed, total, settings):
                gen_label = "?" if generation is None else int(generation)
                print(
                    f"# GENETIC | generation={gen_label} | evaluate candidate {completed}/{total} | "
                    f"workers={workers}",
                    flush=True,
                )
            if _should_print_terminal_candidate_progress(completed, total, settings):
                _write_candidate_progress(
                    ctx,
                    generation=generation,
                    total=total,
                    completed=completed,
                    workers=workers,
                    status="running" if completed < total else "completed",
                    pending=len(pending_futures),
                    last_candidate_id=last_candidate_id,
                    no_progress_s=0.0,
                )
        if broken_pool_detected:
            break
    return [evaluation for evaluation in results if evaluation is not None]


def _evaluate_with_cache(
    ctx: ReuseGeneticEvaluationContext,
    population: Sequence[Sequence[float]],
    cache: Dict[Tuple[float, ...], CandidateEvaluation],
    generation: Optional[int] = None,
    artifact_state: Optional[Dict[str, int]] = None,
    settings: Optional[ReuseGeneticAlgorithmSettings] = None,
    executor: Optional[concurrent.futures.ProcessPoolExecutor] = None,
) -> Tuple[List[CandidateEvaluation], int]:
    out: List[Optional[CandidateEvaluation]] = [None] * len(population)
    hits = 0
    pending_positions: List[int] = []
    pending_vectors: List[Sequence[float]] = []
    pending_keys: List[Tuple[float, ...]] = []
    for idx, vector in enumerate(population):
        key = _cache_key(ctx, vector)
        if key in cache:
            hits += 1
            out[idx] = cache[key]
            continue
        pending_positions.append(idx)
        pending_vectors.append(vector)
        pending_keys.append(key)

    persisted_pending_indices = set()

    def _on_pending_complete(pending_idx: int, evaluation: CandidateEvaluation) -> Optional[int]:
        candidate_id = _finalize_candidate_evaluation(
            ctx,
            settings,
            evaluation,
            generation=generation,
            artifact_state=artifact_state,
        )
        persisted_pending_indices.add(int(pending_idx))
        return candidate_id

    pending_evaluations = _evaluate_pending_vectors(
        ctx,
        pending_vectors,
        settings=settings,
        generation=generation,
        executor=executor,
        on_complete=_on_pending_complete if artifact_state is not None else None,
    )

    for pending_idx, (idx, key, evaluation) in enumerate(zip(pending_positions, pending_keys, pending_evaluations)):
        if pending_idx not in persisted_pending_indices:
            _finalize_candidate_evaluation(
                ctx,
                settings,
                evaluation,
                generation=generation,
                artifact_state=artifact_state,
            )
        cache[key] = evaluation
        out[idx] = evaluation
    return [ev for ev in out if ev is not None], hits


def _persist_candidate_evaluation(
    ctx: ReuseGeneticEvaluationContext,
    evaluation: CandidateEvaluation,
    candidate_id: int,
) -> None:
    if not ctx.save_candidate_artifacts or not ctx.output_dir:
        return

    output_dir = str(ctx.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "genetic_candidate_metrics.csv")
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=candidate_metrics_fieldnames())
        if write_header:
            writer.writeheader()
        writer.writerow(candidate_metrics_row(ctx, evaluation, candidate_id=candidate_id))


def _run_builtin_ga(
    ctx: ReuseGeneticEvaluationContext,
    settings: ReuseGeneticAlgorithmSettings,
    objective_names: Sequence[str],
) -> ReuseGeneticOptimizationResult:
    if len(objective_names) != 1:
        raise ValueError("Built-in GA supports exactly one objective for now.")

    xl, xu = build_decision_bounds(ctx)
    n_var = len(xl)
    if n_var <= 0:
        raise ValueError("Move spec has no decision variables.")

    rng = random.Random(settings.seed)
    n_pop = max(4, int(settings.population_size))
    n_gen = max(1, int(settings.n_generations))
    elite_n = max(1, min(n_pop, int(math.ceil(float(settings.elite_fraction) * n_pop))))

    population = _build_initial_population(rng, settings, n_var, xl, xu)
    cache = _seed_cache_from_settings(ctx, settings)
    artifact_state = {"next_id": max(1, int(settings.candidate_id_start or 1))}
    history: List[Dict[str, float]] = list(settings.history_seed or [])
    best: Optional[CandidateEvaluation] = None
    n_evaluations = 0
    cache_hits_total = 0
    _print_run_header("builtin-ga", settings, objective_names, n_var)
    max_workers = _resolved_worker_count(settings)
    try:
        for gen in range(1, n_gen + 1):
            actual_generation = int(settings.generation_offset or 0) + gen
            executor: Optional[concurrent.futures.ProcessPoolExecutor] = None
            try:
                if max_workers > 1 and len(population) > 1:
                    executor = concurrent.futures.ProcessPoolExecutor(
                        max_workers=max_workers,
                        initializer=_parallel_worker_init,
                        initargs=(ctx, _evaluation_output_is_suppressed(settings)),
                    )
                evaluations, cache_hits = _evaluate_with_cache(
                    ctx,
                    population,
                    cache,
                    generation=actual_generation,
                    artifact_state=artifact_state,
                    settings=settings,
                    executor=executor,
                )
            finally:
                if executor is not None:
                    try:
                        executor.shutdown(wait=False, cancel_futures=True)
                    except TypeError:
                        executor.shutdown(wait=False)
                    except Exception:
                        pass
            cache_hits_total += cache_hits
            n_evaluations += len(population) - cache_hits
            scores = [_objective_score(ev, objective_names) for ev in evaluations]

            ranked = sorted(zip(scores, evaluations, population), key=lambda item: item[0])
            gen_best = ranked[0][1]
            if best is None or _objective_score(gen_best, objective_names) < _objective_score(best, objective_names):
                best = gen_best

            entry = _history_entry(actual_generation, evaluations, objective_names, single_objective=True)
            history.append(entry)
            _print_generation_summary(settings, method="builtin-ga", entry=entry)
            _safe_generation_checkpoint(
                ctx,
                settings,
                method="builtin-ga",
                generation=actual_generation,
                objective_names=objective_names,
                history=history,
                cache=cache,
                n_evaluations=n_evaluations,
                cache_hits=cache_hits_total,
            )

            elites = [list(item[2]) for item in ranked[:elite_n]]
            next_population: List[List[float]] = list(elites)

            while len(next_population) < n_pop:
                i1 = _tournament_pick(rng, scores)
                i2 = _tournament_pick(rng, scores)
                p1 = population[i1]
                p2 = population[i2]

                if rng.random() <= float(settings.crossover_prob):
                    c1, c2 = _blend_crossover(rng, p1, p2, xl, xu)
                else:
                    c1, c2 = list(p1), list(p2)

                c1 = _mutate_vector(rng, c1, xl, xu, settings.mutation_prob, settings.mutation_scale)
                c2 = _mutate_vector(rng, c2, xl, xu, settings.mutation_prob, settings.mutation_scale)

                next_population.append(c1)
                if len(next_population) < n_pop:
                    next_population.append(c2)

            population = next_population[:n_pop]
    finally:
        pass

    all_evaluations = list(cache.values())
    global_pareto_front = _extract_pareto_front(all_evaluations, objective_names)

    return ReuseGeneticOptimizationResult(
        best=best,
        history=history,
        method="builtin-ga",
        n_evaluations=n_evaluations,
        cache_hits=cache_hits_total,
        pareto_front=[] if best is None else [best],
        pareto_front_global=global_pareto_front,
        objective_names=tuple(objective_names),
        all_evaluations=all_evaluations,
    )


def _run_pymoo_optimization(
    ctx: ReuseGeneticEvaluationContext,
    settings: ReuseGeneticAlgorithmSettings,
    objective_names: Sequence[str],
) -> ReuseGeneticOptimizationResult:
    if not _HAS_PYMOO:
        raise ImportError("pymoo is not installed in this environment.")

    xl, xu = build_decision_bounds(ctx)
    n_var = len(xl)
    if n_var <= 0:
        raise ValueError("Move spec has no decision variables.")

    cache = _seed_cache_from_settings(ctx, settings)
    artifact_state = {"next_id": max(1, int(settings.candidate_id_start or 1))}
    history: List[Dict[str, float]] = list(settings.history_seed or [])
    stats = {"eval_requests": 0, "cache_hits": 0}
    max_workers = _resolved_worker_count(settings)
    requested_generations = max(1, int(settings.n_generations))
    generation_offset = int(settings.generation_offset or 0)
    last_requested_generation = generation_offset + requested_generations
    executor: Optional[concurrent.futures.ProcessPoolExecutor] = None

    class _ReuseProblem(Problem):
        def __init__(self):
            super().__init__(
                n_var=n_var,
                n_obj=len(objective_names),
                n_ieq_constr=5,
                xl=np.asarray(xl, dtype=float),
                xu=np.asarray(xu, dtype=float),
            )
            self._current_generation = max(1, generation_offset + 1)

        def _evaluate(self, x, out, *args, **kwargs):
            population = np.atleast_2d(np.asarray(x, dtype=float))
            generation_label = min(int(getattr(self, "_current_generation", 1)), last_requested_generation)
            local_executor: Optional[concurrent.futures.ProcessPoolExecutor] = None
            try:
                if max_workers > 1 and len(population) > 1:
                    local_executor = concurrent.futures.ProcessPoolExecutor(
                        max_workers=max_workers,
                        initializer=_parallel_worker_init,
                        initargs=(ctx, _evaluation_output_is_suppressed(settings)),
                    )
                evaluations, cache_hits = _evaluate_with_cache(
                    ctx,
                    population,
                    cache,
                    generation=generation_label,
                    artifact_state=artifact_state,
                    settings=settings,
                    executor=local_executor,
                )
            finally:
                if local_executor is not None:
                    try:
                        local_executor.shutdown(wait=False, cancel_futures=True)
                    except TypeError:
                        local_executor.shutdown(wait=False)
                    except Exception:
                        pass
            stats["eval_requests"] += len(population)
            stats["cache_hits"] += cache_hits
            out["F"] = np.asarray(
                [_objective_vector(evaluation, objective_names) for evaluation in evaluations],
                dtype=float,
            )
            out["G"] = np.asarray(
                [_constraint_vector(evaluation) for evaluation in evaluations],
                dtype=float,
            )

    problem = _ReuseProblem()

    def _callback(algorithm):
        completed_generation = min(generation_offset + int(algorithm.n_gen), last_requested_generation)
        # pymoo calls _evaluate before the callback for the next generation.
        # Keep the callback summary on the completed generation, but advance
        # the progress label used by the next _evaluate() batch.
        problem._current_generation = min(completed_generation + 1, last_requested_generation)
        X = algorithm.pop.get("X")
        X = np.atleast_2d(X)
        evaluations = [cache[_cache_key(ctx, row)] for row in X]
        for evaluation in evaluations:
            evaluation.meta["generation"] = completed_generation
        entry = _history_entry(
            completed_generation,
            evaluations,
            objective_names,
            single_objective=(len(objective_names) == 1),
        )
        history.append(entry)
        _print_generation_summary(settings, method=method, entry=entry)
        _safe_generation_checkpoint(
            ctx,
            settings,
            method=method,
            generation=completed_generation,
            objective_names=objective_names,
            history=history,
            cache=cache,
            n_evaluations=int(stats["eval_requests"] - stats["cache_hits"]),
            cache_hits=int(stats["cache_hits"]),
        )

    n_pop = max(4, int(settings.population_size))
    n_offsprings = None if settings.n_offsprings in {None, 0} else max(1, int(settings.n_offsprings))
    if settings.initial_population:
        resume_population = _build_initial_population(
            random.Random(settings.seed),
            settings,
            n_var,
            xl,
            xu,
        )
        sampling = np.asarray(resume_population, dtype=float)
    else:
        sampling = _build_pymoo_sampling(settings)
    crossover = SBX(
        prob=float(settings.crossover_prob),
        eta=float(settings.crossover_eta),
    )
    mutation = PM(
        prob=float(settings.mutation_prob),
        eta=float(settings.mutation_eta),
    )
    if len(objective_names) == 1:
        algorithm = GA(
            pop_size=n_pop,
            n_offsprings=n_offsprings,
            sampling=sampling,
            crossover=crossover,
            mutation=mutation,
            eliminate_duplicates=bool(settings.eliminate_duplicates),
        )
        method = "pymoo-ga"
    else:
        algorithm = NSGA2(
            pop_size=n_pop,
            n_offsprings=n_offsprings,
            sampling=sampling,
            crossover=crossover,
            mutation=mutation,
            eliminate_duplicates=bool(settings.eliminate_duplicates),
        )
        method = "pymoo-nsga2"
    _print_run_header(method, settings, objective_names, n_var)

    res = minimize(
        problem,
        algorithm,
        ("n_gen", max(1, requested_generations - 1)),
        seed=settings.seed,
        verbose=False,
        callback=_callback,
    )

    all_evaluations = list(cache.values())
    result_vectors = []
    if res.X is not None:
        result_vectors = np.atleast_2d(res.X)
    evaluations = [cache[_cache_key(ctx, row)] for row in result_vectors] if len(result_vectors) > 0 else list(all_evaluations)
    best = _select_representative_best(evaluations, objective_names)
    pareto_front = _extract_pareto_front(evaluations, objective_names)
    global_pareto_front = _extract_pareto_front(all_evaluations, objective_names)

    return ReuseGeneticOptimizationResult(
        best=best,
        history=history,
        method=method,
        n_evaluations=int(stats["eval_requests"] - stats["cache_hits"]),
        cache_hits=int(stats["cache_hits"]),
        pareto_front=pareto_front,
        pareto_front_global=global_pareto_front,
        objective_names=tuple(objective_names),
        all_evaluations=all_evaluations,
    )


def run_reuse_genetic_optimization(
    ctx: ReuseGeneticEvaluationContext,
    settings: ReuseGeneticAlgorithmSettings,
) -> ReuseGeneticOptimizationResult:
    objective_names = _normalize_objective_names(ctx.objectives.names)
    multi_objective = bool(settings.multi_objective) or bool(ctx.objectives.multi_objective) or len(objective_names) > 1
    engine = str(settings.engine or "auto").strip().lower()
    if engine not in {"auto", "builtin", "pymoo"}:
        raise ValueError("settings.engine must be one of: 'auto', 'builtin', 'pymoo'.")

    if multi_objective and engine == "builtin":
        raise ValueError("Built-in GA does not support multi-objective optimization. Use engine='pymoo' or 'auto'.")

    use_pymoo = False
    if engine == "pymoo":
        use_pymoo = True
    elif engine == "auto":
        use_pymoo = _HAS_PYMOO

    if use_pymoo:
        return _run_pymoo_optimization(ctx, settings, objective_names)
    return _run_builtin_ga(ctx, settings, objective_names)
