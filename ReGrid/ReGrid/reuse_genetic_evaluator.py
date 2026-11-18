from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from OS_LB_fast import get_tangent_stiffness, linear_buckling_sparse
from OS_opt_sections_ec3 import _build_elastic_state, optimize_sections_ec3
from OS_self_weigth import apply_selfweight
from Merchant_Rankine import merchant_rankine_factor
from genetic_fast_mech_solver import solve_genetic_fast_mech
from genetic_grid_helpers import apply_quarter_mirror_moves
from mesh import define_mesh
from reuse_matching import match_reused_members


MemberKey = Tuple[int, int]


def _member_key(a: int, b: int) -> MemberKey:
    a = int(a)
    b = int(b)
    return (a, b) if a < b else (b, a)


def _member_lengths(G: Mapping[str, Any], M: Mapping[str, Any]) -> Dict[MemberKey, float]:
    nodes = G["nodes"]
    out: Dict[MemberKey, float] = {}
    for na, nb in M.get("members", []):
        xa, ya, za = nodes[int(na)]
        xb, yb, zb = nodes[int(nb)]
        L = math.sqrt((xb - xa) ** 2 + (yb - ya) ** 2 + (zb - za) ** 2)
        out[_member_key(na, nb)] = float(L)
    return out


def _section_weight_per_length(sec: Mapping[str, Any], gamma_steel: float) -> float:
    try:
        value = sec.get("w_kN_per_m", None)
        if value is not None:
            return float(value)
    except Exception:
        pass
    try:
        return float(sec.get("A", 0.0)) * float(gamma_steel)
    except Exception:
        return 0.0


def _total_weight(G: Mapping[str, Any], M: Mapping[str, Any], section_by_member: Mapping[MemberKey, Mapping[str, Any]], gamma_steel: float) -> float:
    lengths = _member_lengths(G, M)
    total = 0.0
    for mk, sec in section_by_member.items():
        total += float(lengths.get(_member_key(*mk), 0.0)) * _section_weight_per_length(sec, gamma_steel)
    return float(total)


def _reused_and_new_weight(
    G: Mapping[str, Any],
    M: Mapping[str, Any],
    section_by_member: Mapping[MemberKey, Mapping[str, Any]],
    reuse_result: Optional[Mapping[str, Any]],
    gamma_steel: float,
) -> Tuple[float, float]:
    if not section_by_member:
        return 0.0, 0.0
    lengths = _member_lengths(G, M)
    reused_records = [] if not reuse_result else list(reuse_result.get("reused_records", []) or [])
    reused_members = {
        _member_key(int(rec["member"][0]), int(rec["member"][1]))
        for rec in reused_records
        if isinstance(rec, Mapping) and rec.get("member", None) is not None
    }
    reused_mass = 0.0
    total_mass = 0.0
    for mk, sec in section_by_member.items():
        mass = float(lengths.get(_member_key(*mk), 0.0)) * _section_weight_per_length(sec, gamma_steel)
        total_mass += mass
        if _member_key(*mk) in reused_members:
            reused_mass += mass
    new_mass = total_mass - reused_mass
    return float(reused_mass), float(new_mass)


def _reuse_ratio_metrics(M: Mapping[str, Any], reuse_result: Optional[Mapping[str, Any]]) -> Dict[str, float]:
    members = M.get("members", []) or []
    n_members = max(1, len(members))
    if not reuse_result:
        return {
            "reuse_ratio_members": 0.0,
        }

    reused_records = reuse_result.get("reused_records", []) or []
    return {
        "reuse_ratio_members": float(len(reused_records)) / float(n_members),
    }


def _full_reuse_ratio_length(G: Mapping[str, Any], M: Mapping[str, Any], reuse_result: Optional[Mapping[str, Any]]) -> float:
    if not reuse_result:
        return 0.0
    lengths = _member_lengths(G, M)
    total_length = sum(lengths.values())
    if total_length <= 0.0:
        return 0.0
    reused_records = reuse_result.get("reused_records", []) or []
    reused_length = sum(float(rec.get("member_length", 0.0)) for rec in reused_records)
    return float(reused_length) / float(total_length)


def _max_abs_uz(res: Optional[Mapping[str, Any]]) -> float:
    if not isinstance(res, Mapping):
        return float("inf")
    try:
        max_abs_nd = res.get("max_vertical", {}).get("max_abs_nd", None)
        if isinstance(max_abs_nd, Mapping):
            val = max_abs_nd.get("Uz", max_abs_nd.get("value", None))
            if val is not None:
                return abs(float(val))
    except Exception:
        pass
    try:
        uz_min = float(res.get("max_vertical", {}).get("min", {}).get("Uz", 0.0))
        uz_max = float(res.get("max_vertical", {}).get("max", {}).get("Uz", 0.0))
        return max(abs(uz_min), abs(uz_max))
    except Exception:
        return float("inf")


def _mean_finite_dcr(ec3_results: Optional[Mapping[Any, Mapping[str, Any]]]) -> float:
    if not isinstance(ec3_results, Mapping):
        return float("inf")
    values: List[float] = []
    for record in ec3_results.values():
        if not isinstance(record, Mapping):
            continue
        try:
            dcr = float(record.get("dcr", float("nan")))
        except Exception:
            dcr = float("nan")
        if math.isfinite(dcr):
            values.append(dcr)
    if not values:
        return float("inf")
    return float(sum(values) / float(len(values)))


def _weight_kn_to_metric_tonnes(weight_kn: object) -> float:
    try:
        numeric = float(weight_kn)
    except Exception:
        return float("inf")
    if not math.isfinite(numeric):
        return float("inf")
    return float(numeric) / 9.80665


def _safe_ratio(numerator: object, denominator: object) -> float:
    try:
        num = float(numerator)
        den = float(denominator)
    except Exception:
        return float("nan")
    if not math.isfinite(num) or not math.isfinite(den) or abs(den) <= 1.0e-12:
        return float("nan")
    return float(num) / float(den)


def _reuse_efficiency_score(
    reused_length: float,
    scrap_length: float,
    reused_count: int,
    n_cuts: int,
) -> float:
    if float(reused_length) <= 0.0 or int(reused_count) <= 0:
        return 0.0
    material_fit = float(reused_length) / float(max(1.0e-12, float(reused_length) + max(0.0, float(scrap_length))))
    cut_fit = float(reused_count) / float(max(1, int(reused_count) + max(0, int(n_cuts))))
    # Equal weights keep the score interpretable as a 0..1 fit indicator.
    return float(max(0.0, min(1.0, 0.5 * (material_fit + cut_fit))))


def _constraint_vector(
    *,
    dcr_max: float,
    disp_ratio: float,
    lambda_ref: Optional[float],
    merchant_rankine_value: Optional[float],
    dcr_limit: float,
    lambda_min: float,
    merchant_rankine_enabled: bool,
    merchant_rankine_min: float,
) -> Dict[str, float]:
    lambda_val = float(lambda_ref) if lambda_ref is not None else -float("inf")
    return {
        "g_dcr": float(dcr_max) - float(dcr_limit),
        "g_disp": float(disp_ratio) - 1.0,
        "g_buckling": float(lambda_min) - float(lambda_val),
        "g_merchant_rankine": (
            float(merchant_rankine_min) - float(merchant_rankine_value)
            if bool(merchant_rankine_enabled) and merchant_rankine_value is not None
            else (float("inf") if bool(merchant_rankine_enabled) else 0.0)
        ),
    }


def _failure_constraints() -> Dict[str, float]:
    return {
        "g_dcr": float("inf"),
        "g_disp": float("inf"),
        "g_buckling": float("inf"),
        "g_merchant_rankine": float("inf"),
        "g_analysis": float("inf"),
    }


def _failure_evaluation(
    *,
    x_tuple: Tuple[float, ...],
    error: str,
    moved_G: Optional[Mapping[str, Any]] = None,
    moved_M: Optional[Mapping[str, Any]] = None,
    opt_result: Optional[Mapping[str, Any]] = None,
    reuse_result: Optional[Mapping[str, Any]] = None,
    final_state: Optional[Mapping[str, Any]] = None,
    section_by_member: Optional[Mapping[MemberKey, Mapping[str, Any]]] = None,
    lambda_ref: Optional[float] = None,
    ) -> "CandidateEvaluation":
    return CandidateEvaluation(
        x=x_tuple,
        feasible=False,
        objectives={
            "mass": float("inf"),
            "cost": float("inf"),
            "reuse_ratio": 0.0,
            "reuse_efficiency": 0.0,
            "reuse_ratio_members": 0.0,
            "dcr_max": float("inf"),
            "max_abs_uz": float("inf"),
            "lambda_ref": float("-inf"),
            "merchant_rankine": float("-inf"),
            "ec": float("inf"),
            "pi_mass": float("-inf"),
            "pi_ec": float("-inf"),
            "pi_mr": float("-inf"),
            "pi_cost": float("-inf"),
        },
        constraints=_failure_constraints(),
        moved_G=moved_G,
        moved_M=moved_M,
        opt_result=opt_result,
        reuse_result=reuse_result,
        final_state=final_state,
        section_by_member=section_by_member,
        lambda_ref=lambda_ref,
        meta={
            "error": error,
            "dcr_mean": float("inf"),
        },
    )


def _normalized_positive_violation(value: object, scale: object = 1.0, *, nonfinite_penalty: float = 1.0) -> float:
    try:
        numeric_value = float(value)
    except Exception:
        return float(nonfinite_penalty)
    if not math.isfinite(numeric_value):
        return float(nonfinite_penalty) if (math.isnan(numeric_value) or numeric_value > 0.0) else 0.0
    if numeric_value <= 0.0:
        return 0.0
    try:
        numeric_scale = abs(float(scale))
    except Exception:
        numeric_scale = 1.0
    if not math.isfinite(numeric_scale) or numeric_scale <= 1.0e-12:
        numeric_scale = 1.0
    return float(numeric_value / numeric_scale)


def _clamp_numeric(value: float, lower: Optional[float], upper: Optional[float]) -> float:
    out = float(value)
    if lower is not None:
        out = max(out, float(lower))
    if upper is not None:
        out = min(out, float(upper))
    return float(out)


def _compute_buckling_modes(
    ctx: ReuseGeneticEvaluationContext,
    G: Mapping[str, Any],
    M: Mapping[str, Any],
    section_by_member: Mapping[MemberKey, Mapping[str, Any]],
    n_modes: int,
) -> List[float]:
    tangent_kwargs = dict(system="FullGeneral", numberer="RCM", constraints="Plain", algorithm="Newton")

    ctx.init_model(ctx.ndm, ctx.ndf)
    ctx.define_nodes(G, ctx.ndf, set_mass=False)
    ctx.apply_costr(G, ctx.ndf, ctx.c1, ctx.c2, ctx.c3, ctx.c4, ctx.c5)

    _, ele_data = ctx.el_ele(
        G=G,
        M=M,
        material=ctx.material,
        section=None,
        section_by_member=section_by_member,
        section_rotation_opts=ctx.section_rotation_opts,
        transf_type="PDelta",
        subdivide=ctx.subdivide,
        first_new_node_tag=ctx.first_new_node_tag,
        update_G=ctx.update_G,
        print_ops=False,
        print_summary=False,
    )

    apply_selfweight(
        G=G,
        ele_data=ele_data,
        mode="eleload",
        section=None,
        section_by_member=section_by_member,
        catalog=ctx.catalog,
        units_in=ctx.units_in,
        units_out=ctx.units_out,
        pattern_tag=21,
        ts_tag=21,
        print_summary=False,
    )

    K0, _ = get_tangent_stiffness(**tangent_kwargs, print_ops=False)

    ctx.apply_loads(
        G,
        M=M,
        q=ctx.q_load,
        direction="-Z",
        pattern_tag=2,
        ts_tag=2,
        ndf=ctx.ndf,
        area_mode=ctx.area_mode,
        print_ops=False,
        print_summary=False,
        plot_loads=False,
    )

    out = ctx.run_static(
        load_control=1.0,
        analyze_steps=1,
        set_load_const=False,
        print_ops=False,
        print_summary=False,
    )
    analysis_ok = False
    if isinstance(out, (tuple, list)) and len(out) >= 2:
        analysis_ok = int(out[1]) == 0
    elif isinstance(out, bool):
        analysis_ok = bool(out)
    elif isinstance(out, int):
        analysis_ok = int(out) == 0
    if not analysis_ok:
        return []

    n_modes = max(1, int(n_modes))
    lam, _ = linear_buckling_sparse(
        K0,
        n_modes=n_modes,
        drop_tol=1e-12,
        eig_tol=1e-12,
        tangent_kwargs=tangent_kwargs,
        print_ops=False,
        print_summary=False,
    )
    if lam is None:
        return []
    out_lam: List[float] = []
    for value in list(lam)[:n_modes]:
        try:
            out_lam.append(float(value))
        except Exception:
            continue
    return out_lam


def _evaluate_structural_summary(
    ctx: "ReuseGeneticEvaluationContext",
    G: Mapping[str, Any],
    M: Mapping[str, Any],
    section_by_member: Mapping[MemberKey, Mapping[str, Any]],
    *,
    dlim: float,
    buckling_mode_index: int,
    buckling_lambda_min: float,
    merchant_rankine_mode_index: Optional[int],
) -> Dict[str, Any]:
    mode_index = max(1, int(buckling_mode_index))
    mr_mode_index = int(merchant_rankine_mode_index or mode_index)
    mr_mode_index = max(1, mr_mode_index)

    final_state = _build_elastic_state(
        G,
        M,
        section_by_member=section_by_member,
        material=ctx.material,
        ndm=ctx.ndm,
        ndf=ctx.ndf,
        c1=ctx.c1,
        c2=ctx.c2,
        c3=ctx.c3,
        c4=ctx.c4,
        c5=ctx.c5,
        q_load=ctx.q_load,
        catalog=ctx.catalog,
        units_in=ctx.units_in,
        units_out=ctx.units_out,
        init_model=ctx.init_model,
        define_nodes=ctx.define_nodes,
        apply_costr=ctx.apply_costr,
        el_ele=ctx.el_ele,
        apply_loads=ctx.apply_loads,
        run_static=ctx.run_static,
        section_rotation_opts=ctx.section_rotation_opts,
        subdivide=ctx.subdivide,
        first_new_node_tag=ctx.first_new_node_tag,
        update_G=ctx.update_G,
        area_mode=ctx.area_mode,
        gamma_M0=ctx.gamma_M0,
        gamma_M1=ctx.gamma_M1,
        gamma_M2=ctx.gamma_M2,
        beta=ctx.beta,
        alpha_LT=ctx.alpha_LT,
        compression_sign=ctx.compression_sign,
        dlim=dlim,
        compute_buckling=(mode_index == 1),
        buckling_lambda_min=buckling_lambda_min,
        internal_quiet=True,
    )

    analysis_ok = bool(final_state.get("analysis_ok", False))
    buckling_lambdas: List[float] = []
    max_requested_mode = max(mode_index, mr_mode_index)
    if analysis_ok:
        if max_requested_mode == 1:
            first_lambda = final_state.get("lb_lambda_1", None)
            if first_lambda is not None:
                try:
                    buckling_lambdas = [float(first_lambda)]
                except Exception:
                    buckling_lambdas = []
        else:
            buckling_lambdas = _compute_buckling_modes(
                ctx,
                G,
                M,
                section_by_member,
                max_requested_mode,
            )

    lambda_ref = None
    if len(buckling_lambdas) >= mode_index:
        lambda_ref = float(buckling_lambdas[mode_index - 1])
    elif mode_index == 1:
        lambda_ref = final_state.get("lb_lambda_1", None)

    mr_lambda = None
    if len(buckling_lambdas) >= mr_mode_index:
        mr_lambda = float(buckling_lambdas[mr_mode_index - 1])
    elif mr_mode_index == 1:
        mr_lambda = final_state.get("lb_lambda_1", None)

    dcr_max = float(final_state.get("ec3_dcr_max", float("inf")))
    dcr_mean = _mean_finite_dcr(final_state.get("ec3_results", None))
    disp_ratio = float(final_state.get("disp_ratio", float("inf")))
    max_abs_uz = _max_abs_uz(final_state.get("res", None))

    merchant_rankine_value = None
    if mr_lambda is not None and math.isfinite(float(dcr_max)) and math.isfinite(float(mr_lambda)):
        try:
            merchant_rankine_value = float(merchant_rankine_factor(dcr_max=dcr_max, linear_buckling_factor=mr_lambda))
        except Exception:
            merchant_rankine_value = None

    return {
        "final_state": final_state,
        "analysis_ok": analysis_ok,
        "lambda_ref": (None if lambda_ref is None else float(lambda_ref)),
        "merchant_rankine_lambda": (None if mr_lambda is None else float(mr_lambda)),
        "merchant_rankine_mode_index": int(mr_mode_index),
        "merchant_rankine": merchant_rankine_value,
        "dcr_max": dcr_max,
        "dcr_mean": dcr_mean,
        "disp_ratio": disp_ratio,
        "max_abs_uz": max_abs_uz,
    }


def _run_mechanical_optimization(
    ctx: "ReuseGeneticEvaluationContext",
    G: Mapping[str, Any],
    M: Mapping[str, Any],
    *,
    opt_dcr_limit: float,
    opt_dlim: float,
    opt_buckling_lambda_min: float,
    opt_merchant_rankine_min: float,
) -> Mapping[str, Any]:
    if str(ctx.mech_solver).strip().lower() == "fast":
        return solve_genetic_fast_mech(
            G=G,
            M=M,
            mecc_opt=ctx.mecc_opt,
            material=ctx.material,
            catalog=ctx.catalog,
            candidate_sections=ctx.candidate_sections or (),
            units_in=ctx.units_in,
            units_out=ctx.units_out,
            c1=ctx.c1,
            c2=ctx.c2,
            c3=ctx.c3,
            c4=ctx.c4,
            c5=ctx.c5,
            q_load=ctx.q_load,
            ndm=ctx.ndm,
            ndf=ctx.ndf,
            init_model=ctx.init_model,
            define_nodes=ctx.define_nodes,
            apply_costr=ctx.apply_costr,
            el_ele=ctx.el_ele,
            apply_loads=ctx.apply_loads,
            run_static=ctx.run_static,
            initial_section_name=ctx.initial_section_name,
            start_from_first_section=ctx.start_from_first_section,
            start_section_row=ctx.start_section_row,
            dcr_limit=opt_dcr_limit,
            dlim=opt_dlim,
            check_disp=ctx.opt_check_disp,
            check_buckling=ctx.opt_check_buckling,
            buckling_lambda_min=opt_buckling_lambda_min,
            buckling_mode_index=ctx.opt_buckling_mode_index,
            check_merchant_rankine=ctx.opt_check_merchant_rankine,
            merchant_rankine_min=opt_merchant_rankine_min,
            merchant_rankine_mode_index=ctx.opt_merchant_rankine_mode_index,
            gamma_M0=ctx.gamma_M0,
            gamma_M1=ctx.gamma_M1,
            gamma_M2=ctx.gamma_M2,
            beta=ctx.beta,
            alpha_LT=ctx.alpha_LT,
            compression_sign=ctx.compression_sign,
            max_iter=ctx.opt_max_iter,
            subdivide=ctx.subdivide,
            first_new_node_tag=ctx.first_new_node_tag,
            update_G=ctx.update_G,
            area_mode=ctx.area_mode,
        )
    return optimize_sections_ec3(
        G=G,
        M=M,
        mecc_opt=ctx.mecc_opt,
        material=ctx.material,
        catalog=ctx.catalog,
        units_in=ctx.units_in,
        units_out=ctx.units_out,
        c1=ctx.c1,
        c2=ctx.c2,
        c3=ctx.c3,
        c4=ctx.c4,
        c5=ctx.c5,
        q_load=ctx.q_load,
        ndm=ctx.ndm,
        ndf=ctx.ndf,
        init_model=ctx.init_model,
        define_nodes=ctx.define_nodes,
        apply_costr=ctx.apply_costr,
        el_ele=ctx.el_ele,
        apply_loads=ctx.apply_loads,
        run_static=ctx.run_static,
        candidate_sections=ctx.candidate_sections,
        initial_section_name=ctx.initial_section_name,
        start_from_first_section=ctx.start_from_first_section,
        start_section_row=ctx.start_section_row,
        dcr_limit=opt_dcr_limit,
        dlim=opt_dlim,
        check_disp=ctx.opt_check_disp,
        check_buckling=ctx.opt_check_buckling,
        buckling_lambda_min=opt_buckling_lambda_min,
        buckling_mode_index=ctx.opt_buckling_mode_index,
        check_merchant_rankine=ctx.opt_check_merchant_rankine,
        merchant_rankine_min=opt_merchant_rankine_min,
        merchant_rankine_mode_index=ctx.opt_merchant_rankine_mode_index,
        buckling_max_retries=ctx.opt_buckling_retries,
        buckling_group_limit=ctx.opt_buckling_group_limit,
        gamma_M0=ctx.gamma_M0,
        gamma_M1=ctx.gamma_M1,
        gamma_M2=ctx.gamma_M2,
        beta=ctx.beta,
        alpha_LT=ctx.alpha_LT,
        compression_sign=ctx.compression_sign,
        max_iter=ctx.opt_max_iter,
        subdivide=ctx.subdivide,
        first_new_node_tag=ctx.first_new_node_tag,
        update_G=ctx.update_G,
        area_mode=ctx.area_mode,
        opt_verbose=0,
        group_report_path=None,
    )


def compute_fixed_reference_design(
    ctx: "ReuseGeneticEvaluationContext",
) -> Dict[str, Any]:
    base_M = ctx.base_M if ctx.base_M is not None else ctx.mesh_builder(ctx.base_G, ctx.grid)
    opt_result = _run_mechanical_optimization(
        ctx,
        ctx.base_G,
        base_M,
        opt_dcr_limit=float(ctx.opt_dcr_limit),
        opt_dlim=float(ctx.opt_dlim),
        opt_buckling_lambda_min=float(ctx.opt_buckling_lambda_min),
        opt_merchant_rankine_min=float(ctx.opt_merchant_rankine_min),
    )
    section_by_member = opt_result.get("section_by_member", None) if isinstance(opt_result, Mapping) else None
    if not section_by_member:
        raise RuntimeError("fixed base reference optimization returned no section assignment.")
    summary = _evaluate_structural_summary(
        ctx,
        ctx.base_G,
        base_M,
        section_by_member,
        dlim=float(ctx.constraints.disp_limit),
        buckling_mode_index=int(ctx.constraints.buckling_mode_index),
        buckling_lambda_min=float(ctx.constraints.buckling_lambda_min),
        merchant_rankine_mode_index=ctx.constraints.merchant_rankine_mode_index,
    )
    total_weight = _total_weight(ctx.base_G, base_M, section_by_member, ctx.gamma_steel)
    mass_tonnes = _weight_kn_to_metric_tonnes(total_weight)
    return {
        "label": "fixed_base_new",
        "G": ctx.base_G,
        "M": base_M,
        "opt_result": opt_result,
        "summary": summary,
        "disp_limit": float(ctx.constraints.disp_limit),
        "opt_dlim": float(ctx.opt_dlim),
        "mass": float(total_weight),
        "mass_tonnes": float(mass_tonnes),
        "ec": float(ctx.ecf_new) * float(mass_tonnes),
        "cost": float(ctx.cost_new) * float(mass_tonnes),
    }


@dataclass
class ReuseGeneticConstraints:
    dcr_limit: float = 1.0
    disp_limit: float = 0.15
    buckling_lambda_min: float = 1.0
    buckling_mode_index: int = 1
    buckling_n_modes: Optional[int] = None
    merchant_rankine_enabled: bool = False
    merchant_rankine_min: float = 1.0
    merchant_rankine_mode_index: Optional[int] = None


@dataclass
class ReuseGeneticVariableControl:
    enabled: bool = False
    lower: Optional[float] = None
    upper: Optional[float] = None


@dataclass
class ReuseGeneticDecisionControls:
    ga_dcr_limit: ReuseGeneticVariableControl = field(default_factory=ReuseGeneticVariableControl)
    ga_disp_limit: ReuseGeneticVariableControl = field(default_factory=ReuseGeneticVariableControl)
    ga_buckling_lambda_min: ReuseGeneticVariableControl = field(default_factory=ReuseGeneticVariableControl)
    ga_merchant_rankine_min: ReuseGeneticVariableControl = field(default_factory=ReuseGeneticVariableControl)
    reuse_max_cut_length: ReuseGeneticVariableControl = field(default_factory=ReuseGeneticVariableControl)
    opt_dcr_limit: ReuseGeneticVariableControl = field(default_factory=ReuseGeneticVariableControl)
    opt_dlim: ReuseGeneticVariableControl = field(default_factory=ReuseGeneticVariableControl)
    opt_buckling_lambda_min: ReuseGeneticVariableControl = field(default_factory=ReuseGeneticVariableControl)
    opt_merchant_rankine_min: ReuseGeneticVariableControl = field(default_factory=ReuseGeneticVariableControl)


@dataclass
class ReuseGeneticObjectives:
    names: Tuple[str, ...] = ("mass",)
    multi_objective: bool = False


@dataclass
class ReuseGeneticEvaluationContext:
    base_G: Mapping[str, Any]
    move_spec: Mapping[str, Any]
    grid: int
    material: Any
    catalog: Mapping[str, Any]
    units_in: str
    units_out: str
    c1: Sequence[int]
    c2: Sequence[int]
    c3: Sequence[int]
    c4: Sequence[int]
    c5: Sequence[int]
    q_load: Sequence[Any]
    ndm: int
    ndf: int
    init_model: Callable[..., Any]
    define_nodes: Callable[..., Any]
    apply_costr: Callable[..., Any]
    el_ele: Callable[..., Any]
    apply_loads: Callable[..., Any]
    run_static: Callable[..., Any]
    section_rotation_opts: Any = None
    base_M: Optional[Mapping[str, Any]] = None
    mesh_builder: Callable[..., Any] = define_mesh
    reuse_base_mesh_topology: bool = False
    candidate_sections: Optional[Sequence[Mapping[str, Any]]] = None
    mech_solver: str = "ec3"
    mecc_opt: int = 1
    initial_section_name: Optional[str] = None
    start_from_first_section: bool = False
    start_section_row: Optional[int] = None
    opt_max_iter: int = 30
    opt_dcr_limit: float = 1.0
    opt_dlim: float = 0.15
    opt_check_disp: bool = True
    opt_check_buckling: bool = True
    opt_buckling_lambda_min: float = 1.0
    opt_buckling_mode_index: int = 1
    opt_check_merchant_rankine: bool = False
    opt_merchant_rankine_min: float = 1.0
    opt_merchant_rankine_mode_index: Optional[int] = None
    opt_buckling_retries: int = 4
    opt_buckling_group_limit: Optional[int] = None
    opt_section_group_limit: Optional[int] = None
    opt_section_symmetry: Optional[str] = None
    opt_demand_grouping: bool = True
    opt_demand_group_rel_tol: float = 0.12
    gamma_M0: float = 1.0
    gamma_M1: float = 1.0
    gamma_M2: float = 1.25
    beta: float = 1.0
    alpha_LT: Optional[float] = None
    compression_sign: str = "negative"
    subdivide: int = 2
    first_new_node_tag: int = 2000
    update_G: bool = False
    area_mode: str = "faces"
    gamma_steel: float = 78.5
    ecf_new: float = 0.0
    ecf_reused: float = 0.0
    ecf_work: float = 0.0
    ecf_scrap: float = 0.0
    cost_new: float = 0.0
    cost_reused: float = 0.0
    cost_work: float = 0.0
    cost_scrap: float = 0.0
    reuse_enabled: bool = True
    reuse_csv_path: Optional[str] = None
    reuse_length_units_in: str = "m"
    reuse_max_cut_length: Optional[float] = None
    reuse_min_remainder_length: float = 0.0
    reuse_group_sizes: Sequence[int] = (1, 2, 4)
    reuse_member_mask_enabled: bool = False
    reuse_member_mask_threshold: float = 0.5
    reuse_member_mask_mode: str = "member"
    reuse_member_mask_groups: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    output_dir: Optional[str] = None
    save_candidate_artifacts: bool = False
    reference_mode: str = "candidate_new"
    fixed_reference: Optional[Mapping[str, Any]] = None
    constraints: ReuseGeneticConstraints = field(default_factory=ReuseGeneticConstraints)
    objectives: ReuseGeneticObjectives = field(default_factory=ReuseGeneticObjectives)
    decision_controls: ReuseGeneticDecisionControls = field(default_factory=ReuseGeneticDecisionControls)


@dataclass
class CandidateEvaluation:
    x: Tuple[float, ...]
    feasible: bool
    objectives: Dict[str, float]
    constraints: Dict[str, float]
    moved_G: Optional[Mapping[str, Any]]
    moved_M: Optional[Mapping[str, Any]]
    opt_result: Optional[Mapping[str, Any]]
    reuse_result: Optional[Mapping[str, Any]]
    final_state: Optional[Mapping[str, Any]]
    section_by_member: Optional[Mapping[MemberKey, Mapping[str, Any]]]
    lambda_ref: Optional[float]
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_violation(self) -> float:
        return float(sum(max(0.0, float(v)) for v in self.constraints.values()))

    @property
    def normalized_constraint_violations(self) -> Dict[str, float]:
        return {
            "g_dcr": _normalized_positive_violation(
                self.constraints.get("g_dcr", 0.0),
                self.meta.get("ga_dcr_limit_used", 1.0),
            ),
            "g_disp": _normalized_positive_violation(
                self.constraints.get("g_disp", 0.0),
                1.0,
            ),
            "g_buckling": _normalized_positive_violation(
                self.constraints.get("g_buckling", 0.0),
                self.meta.get("ga_buckling_lambda_min_used", 1.0),
            ),
            "g_merchant_rankine": _normalized_positive_violation(
                self.constraints.get("g_merchant_rankine", 0.0),
                self.meta.get("ga_merchant_rankine_min_used", 1.0),
            ),
            "g_analysis": _normalized_positive_violation(
                self.constraints.get("g_analysis", 0.0),
                1.0,
            ),
        }

    @property
    def normalized_violation(self) -> float:
        return float(sum(float(v) for v in self.normalized_constraint_violations.values()))


def _active_control_specs(
    ctx: ReuseGeneticEvaluationContext,
) -> List[Tuple[str, ReuseGeneticVariableControl, float]]:
    controls = ctx.decision_controls
    specs = [
        ("ga_dcr_limit", controls.ga_dcr_limit, float(ctx.constraints.dcr_limit)),
        ("ga_disp_limit", controls.ga_disp_limit, float(ctx.constraints.disp_limit)),
        ("ga_buckling_lambda_min", controls.ga_buckling_lambda_min, float(ctx.constraints.buckling_lambda_min)),
        ("ga_merchant_rankine_min", controls.ga_merchant_rankine_min, float(ctx.constraints.merchant_rankine_min)),
        (
            "reuse_max_cut_length",
            controls.reuse_max_cut_length,
            (
                float(ctx.reuse_max_cut_length)
                if ctx.reuse_max_cut_length is not None
                else 0.0
            ),
        ),
        ("opt_dcr_limit", controls.opt_dcr_limit, float(ctx.opt_dcr_limit)),
        ("opt_dlim", controls.opt_dlim, float(ctx.opt_dlim)),
        ("opt_buckling_lambda_min", controls.opt_buckling_lambda_min, float(ctx.opt_buckling_lambda_min)),
        ("opt_merchant_rankine_min", controls.opt_merchant_rankine_min, float(ctx.opt_merchant_rankine_min)),
    ]
    out: List[Tuple[str, ReuseGeneticVariableControl, float]] = []
    for name, control, default in specs:
        if name == "ga_merchant_rankine_min" and not bool(ctx.constraints.merchant_rankine_enabled):
            continue
        if name == "opt_merchant_rankine_min" and not bool(ctx.opt_check_merchant_rankine):
            continue
        if name == "reuse_max_cut_length" and not (bool(ctx.reuse_enabled) and bool(ctx.reuse_csv_path)):
            continue
        if not bool(control.enabled):
            continue
        if control.lower is None or control.upper is None:
            raise ValueError(f"Decision control '{name}' is enabled but lower/upper bounds are missing.")
        lo = float(control.lower)
        hi = float(control.upper)
        if hi < lo:
            raise ValueError(f"Decision control '{name}' has upper < lower ({hi} < {lo}).")
        out.append((name, control, default))
    return out


def build_decision_bounds(ctx: ReuseGeneticEvaluationContext) -> Tuple[List[float], List[float]]:
    groups = ctx.move_spec.get("movable_groups", []) or []
    x_min, x_max = ctx.move_spec["x_move_range"]
    y_min, y_max = ctx.move_spec["y_move_range"]
    xl: List[float] = []
    xu: List[float] = []
    for _ in groups:
        xl.extend([float(x_min), float(y_min)])
        xu.extend([float(x_max), float(y_max)])
    for _, control, _ in _active_control_specs(ctx):
        xl.append(float(control.lower))
        xu.append(float(control.upper))
    if bool(ctx.reuse_member_mask_enabled):
        for _ in ctx.reuse_member_mask_groups:
            xl.append(0.0)
            xu.append(1.0)
    return xl, xu


def decision_variable_labels(ctx: ReuseGeneticEvaluationContext) -> List[str]:
    labels: List[str] = []
    n_geom = int(ctx.move_spec.get("n_var", 0))
    for idx in range(n_geom // 2):
        labels.extend([f"geom_dx_{idx + 1}", f"geom_dy_{idx + 1}"])
    for name, _, _ in _active_control_specs(ctx):
        labels.append(name)
    if bool(ctx.reuse_member_mask_enabled):
        for idx, group in enumerate(ctx.reuse_member_mask_groups, start=1):
            label = str(group.get("label", "")).strip()
            labels.append(label or f"reuse_mask_group_{idx}")
    return labels


def split_decision_vector(
    ctx: ReuseGeneticEvaluationContext,
    decision_vector: Sequence[float],
) -> Tuple[Tuple[float, ...], Dict[str, float], Dict[MemberKey, int]]:
    values = tuple(float(v) for v in decision_vector)
    n_geom = int(ctx.move_spec.get("n_var", 0))
    if len(values) < n_geom:
        raise ValueError(f"Decision vector has {len(values)} values but geometry needs at least {n_geom}.")
    geom_vector = values[:n_geom]
    tail = values[n_geom:]
    active_specs = _active_control_specs(ctx)
    n_mask = len(ctx.reuse_member_mask_groups) if bool(ctx.reuse_member_mask_enabled) else 0
    if len(tail) != len(active_specs) + n_mask:
        raise ValueError(
            f"Decision vector has {len(values)} values but expected {n_geom + len(active_specs) + n_mask} "
            f"({n_geom} geometry + {len(active_specs)} controls + {n_mask} reuse flags)."
        )
    control_values: Dict[str, float] = {}
    for raw_value, (name, _, _) in zip(tail[: len(active_specs)], active_specs):
        control_values[name] = float(raw_value)
    reuse_mask_values: Dict[MemberKey, int] = {}
    if bool(ctx.reuse_member_mask_enabled):
        threshold = float(ctx.reuse_member_mask_threshold)
        mask_tail = tail[len(active_specs):]
        for raw_value, group in zip(mask_tail, ctx.reuse_member_mask_groups):
            flag = 1 if float(raw_value) >= threshold else 0
            for mk in group.get("members", ()):
                reuse_mask_values[_member_key(int(mk[0]), int(mk[1]))] = flag
    return geom_vector, control_values, reuse_mask_values


def canonicalize_decision_vector(
    ctx: ReuseGeneticEvaluationContext,
    decision_vector: Sequence[float],
) -> Tuple[float, ...]:
    values = [float(v) for v in decision_vector]
    n_geom = int(ctx.move_spec.get("n_var", 0))
    active_specs = _active_control_specs(ctx)
    n_mask = len(ctx.reuse_member_mask_groups) if bool(ctx.reuse_member_mask_enabled) else 0
    expected = n_geom + len(active_specs) + n_mask
    if len(values) != expected:
        raise ValueError(
            f"Decision vector has {len(values)} values but expected {expected} "
            f"({n_geom} geometry + {len(active_specs)} controls + {n_mask} reuse flags)."
        )
    if n_mask > 0:
        threshold = float(ctx.reuse_member_mask_threshold)
        start = n_geom + len(active_specs)
        for idx in range(start, len(values)):
            values[idx] = 1.0 if float(values[idx]) >= threshold else 0.0
    return tuple(values)


def evaluate_candidate_design(
    ctx: ReuseGeneticEvaluationContext,
    decision_vector: Sequence[float],
) -> CandidateEvaluation:
    x_tuple = canonicalize_decision_vector(ctx, decision_vector)
    geom_vector, ga_control_values, reuse_mask_values = split_decision_vector(ctx, x_tuple)
    ga_dcr_limit = float(ga_control_values.get("ga_dcr_limit", ctx.constraints.dcr_limit))
    ga_disp_limit = float(ga_control_values.get("ga_disp_limit", ctx.constraints.disp_limit))
    ga_buckling_lambda_min = float(ga_control_values.get("ga_buckling_lambda_min", ctx.constraints.buckling_lambda_min))
    ga_merchant_rankine_min = float(ga_control_values.get("ga_merchant_rankine_min", ctx.constraints.merchant_rankine_min))
    reuse_max_cut_length = ga_control_values.get("reuse_max_cut_length", ctx.reuse_max_cut_length)
    if reuse_max_cut_length is not None:
        reuse_max_cut_length = float(reuse_max_cut_length)
        reuse_cut_control = ctx.decision_controls.reuse_max_cut_length
        if bool(reuse_cut_control.enabled):
            reuse_max_cut_length = float(
                int(
                    round(
                        _clamp_numeric(
                            reuse_max_cut_length,
                            reuse_cut_control.lower,
                            reuse_cut_control.upper,
                        )
                    )
                )
            )
    opt_dcr_limit = float(ga_control_values.get("opt_dcr_limit", ctx.opt_dcr_limit))
    opt_dlim = float(ga_control_values.get("opt_dlim", ctx.opt_dlim))
    opt_buckling_lambda_min = float(ga_control_values.get("opt_buckling_lambda_min", ctx.opt_buckling_lambda_min))
    opt_merchant_rankine_min = float(ga_control_values.get("opt_merchant_rankine_min", ctx.opt_merchant_rankine_min))
    mode_index = max(1, int(ctx.constraints.buckling_mode_index))
    mr_enabled = bool(ctx.constraints.merchant_rankine_enabled)
    mr_mode_index = int(ctx.constraints.merchant_rankine_mode_index or mode_index)
    mr_mode_index = max(1, mr_mode_index)
    t_start = time.perf_counter()
    t_geom = 0.0
    t_mech = 0.0
    t_match = 0.0
    t_recheck = 0.0

    try:
        t0 = time.perf_counter()
        moved_G = apply_quarter_mirror_moves(ctx.base_G, ctx.move_spec, geom_vector)
        if bool(ctx.reuse_base_mesh_topology) and ctx.base_M is not None:
            moved_M = copy.deepcopy(ctx.base_M)
            if isinstance(moved_M, Mapping) and "meta" in moved_M:
                moved_M["meta"] = dict(moved_M.get("meta", {}))
                moved_M["meta"]["topology_reused_from_base"] = True
        else:
            moved_M = ctx.mesh_builder(moved_G, ctx.grid)
        t_geom = time.perf_counter() - t0
    except Exception as exc:
        return _failure_evaluation(
            x_tuple=x_tuple,
            error=f"geometry_move_failed: {exc}",
        )

    try:
        t0 = time.perf_counter()
        opt_result = _run_mechanical_optimization(
            ctx,
            moved_G,
            moved_M,
            opt_dcr_limit=opt_dcr_limit,
            opt_dlim=opt_dlim,
            opt_buckling_lambda_min=opt_buckling_lambda_min,
            opt_merchant_rankine_min=opt_merchant_rankine_min,
        )
        t_mech = time.perf_counter() - t0
    except Exception as exc:
        return _failure_evaluation(
            x_tuple=x_tuple,
            error=f"mechanical_optimization_failed: {exc}",
            moved_G=moved_G,
            moved_M=moved_M,
        )

    section_by_member = opt_result.get("section_by_member", None) if isinstance(opt_result, Mapping) else None
    if not section_by_member:
        return _failure_evaluation(
            x_tuple=x_tuple,
            error="mechanical_optimization_returned_no_sections",
            moved_G=moved_G,
            moved_M=moved_M,
            opt_result=opt_result,
        )

    reuse_result: Optional[Mapping[str, Any]] = None
    final_section_by_member = dict(section_by_member)
    eligible_reuse_members = None
    if bool(ctx.reuse_member_mask_enabled):
        eligible_reuse_members = [mk for mk, flag in reuse_mask_values.items() if int(flag) == 1]
    if ctx.reuse_enabled and ctx.reuse_csv_path:
        try:
            if eligible_reuse_members is not None and len(eligible_reuse_members) == 0:
                reuse_result = {
                    "section_by_member": dict(section_by_member),
                    "section_name_by_member": {mk: str(sec.get("name", "")) for mk, sec in section_by_member.items()},
                    "reused_records": [],
                    "scrap_records": [],
                    "unmatched_members": [],
                    "skipped_members": list(section_by_member.keys()),
                    "total_cut_length": 0.0,
                    "total_scrap_mass": 0.0,
                    "n_cuts": 0,
                    "remaining_stock_count": 0,
                    "timings": {
                        "time_total_s": 0.0,
                        "time_single_search_s": 0.0,
                        "time_group_search_s": 0.0,
                        "time_stock_update_s": 0.0,
                        "n_single_checked": 0.0,
                        "n_group_checked": 0.0,
                    },
                }
            else:
                t0 = time.perf_counter()
                reuse_result = match_reused_members(
                    G=moved_G,
                    M=moved_M,
                    section_by_member=section_by_member,
                    eligible_members=eligible_reuse_members,
                    reuse_csv_path=str(ctx.reuse_csv_path),
                    section_units_in=ctx.units_in,
                    units_out=ctx.units_out,
                    length_units_in=ctx.reuse_length_units_in,
                    max_cut_length=reuse_max_cut_length,
                    min_remainder_length=ctx.reuse_min_remainder_length,
                    group_sizes=ctx.reuse_group_sizes,
                    remaining_csv_path=None,
                    report_txt_path=None,
                    verbose=False,
                )
                t_match = time.perf_counter() - t0
            final_section_by_member = dict(reuse_result["section_by_member"])
        except Exception as exc:
            return _failure_evaluation(
                x_tuple=x_tuple,
                error=f"reuse_matching_failed: {exc}",
                moved_G=moved_G,
                moved_M=moved_M,
                opt_result=opt_result,
            )

    t0 = time.perf_counter()
    new_reference_summary = _evaluate_structural_summary(
        ctx,
        moved_G,
        moved_M,
        section_by_member,
        dlim=ga_disp_limit,
        buckling_mode_index=mode_index,
        buckling_lambda_min=ga_buckling_lambda_min,
        merchant_rankine_mode_index=mr_mode_index,
    )
    t_recheck_new = time.perf_counter() - t0

    t0 = time.perf_counter()
    final_summary = _evaluate_structural_summary(
        ctx,
        moved_G,
        moved_M,
        final_section_by_member,
        dlim=ga_disp_limit,
        buckling_mode_index=mode_index,
        buckling_lambda_min=ga_buckling_lambda_min,
        merchant_rankine_mode_index=mr_mode_index,
    )
    t_recheck = time.perf_counter() - t0

    final_state = final_summary["final_state"]
    analysis_ok = bool(final_summary["analysis_ok"])
    lambda_ref = final_summary["lambda_ref"]
    mr_lambda = final_summary["merchant_rankine_lambda"]
    dcr_max = float(final_summary["dcr_max"])
    dcr_mean = float(final_summary["dcr_mean"])
    disp_ratio = float(final_summary["disp_ratio"])
    max_abs_uz = float(final_summary["max_abs_uz"])
    merchant_rankine_value = final_summary["merchant_rankine"]
    constraints = _constraint_vector(
        dcr_max=dcr_max,
        disp_ratio=disp_ratio,
        lambda_ref=lambda_ref,
        merchant_rankine_value=merchant_rankine_value,
        dcr_limit=ga_dcr_limit,
        lambda_min=ga_buckling_lambda_min,
        merchant_rankine_enabled=mr_enabled,
        merchant_rankine_min=ga_merchant_rankine_min,
    )
    constraints["g_analysis"] = 0.0 if analysis_ok else float("inf")

    total_weight = _total_weight(moved_G, moved_M, final_section_by_member, ctx.gamma_steel)
    new_reference_total_weight = _total_weight(moved_G, moved_M, section_by_member, ctx.gamma_steel)
    reused_mass, new_mass = _reused_and_new_weight(moved_G, moved_M, final_section_by_member, reuse_result, ctx.gamma_steel)
    scrap_mass = 0.0 if not reuse_result else float(reuse_result.get("total_scrap_mass", 0.0) or 0.0)
    total_mass_tonnes = _weight_kn_to_metric_tonnes(total_weight)
    new_reference_mass_tonnes = _weight_kn_to_metric_tonnes(new_reference_total_weight)
    reused_mass_tonnes = _weight_kn_to_metric_tonnes(reused_mass)
    new_mass_tonnes = _weight_kn_to_metric_tonnes(new_mass)
    scrap_mass_tonnes = _weight_kn_to_metric_tonnes(scrap_mass)
    n_cuts = 0 if not reuse_result else int(reuse_result.get("n_cuts", 0) or 0)
    sustainability_ec = (
        (float(ctx.ecf_new) * float(new_mass_tonnes))
        + (float(ctx.ecf_reused) * float(reused_mass_tonnes))
        + (float(ctx.ecf_work) * float(n_cuts))
        - (float(ctx.ecf_scrap) * float(scrap_mass_tonnes))
    )
    new_reference_ec = float(ctx.ecf_new) * float(new_reference_mass_tonnes)
    total_cost = (
        (float(ctx.cost_new) * float(new_mass_tonnes))
        + (float(ctx.cost_reused) * float(reused_mass_tonnes))
        + (float(ctx.cost_work) * float(n_cuts))
        - (float(ctx.cost_scrap) * float(scrap_mass_tonnes))
    )
    new_reference_cost = float(ctx.cost_new) * float(new_reference_mass_tonnes)
    reference_mode = str(getattr(ctx, "reference_mode", "candidate_new") or "candidate_new").strip().lower()
    reference_label = "candidate_new"
    reference_mass = float(new_reference_total_weight)
    reference_mass_tonnes = float(new_reference_mass_tonnes)
    reference_ec = float(new_reference_ec)
    reference_cost = float(new_reference_cost)
    reference_summary = new_reference_summary
    fixed_reference = getattr(ctx, "fixed_reference", None)
    if reference_mode == "fixed_base" and isinstance(fixed_reference, Mapping):
        fixed_summary = fixed_reference.get("summary", None)
        fixed_mass = fixed_reference.get("mass", None)
        fixed_ec = fixed_reference.get("ec", None)
        fixed_cost = fixed_reference.get("cost", None)
        if isinstance(fixed_summary, Mapping) and fixed_mass is not None and fixed_ec is not None and fixed_cost is not None:
            reference_label = str(fixed_reference.get("label", "fixed_base_new") or "fixed_base_new")
            reference_mass = float(fixed_mass)
            reference_mass_tonnes = float(fixed_reference.get("mass_tonnes", _weight_kn_to_metric_tonnes(fixed_mass)))
            reference_ec = float(fixed_ec)
            reference_cost = float(fixed_cost)
            reference_summary = fixed_summary
        else:
            reference_mode = "candidate_new"
    reuse_ratio_length = _full_reuse_ratio_length(moved_G, moved_M, reuse_result)
    reuse_metrics = _reuse_ratio_metrics(moved_M, reuse_result)
    reused_length = 0.0 if not reuse_result else sum(
        float(rec.get("member_length", 0.0) or 0.0)
        for rec in (reuse_result.get("reused_records", []) or [])
        if isinstance(rec, Mapping)
    )
    scrap_length = 0.0 if not reuse_result else float(reuse_result.get("total_cut_length", 0.0) or 0.0)
    reused_count = 0 if not reuse_result else len(reuse_result.get("reused_records", []) or [])
    reuse_efficiency = _reuse_efficiency_score(
        reused_length=reused_length,
        scrap_length=scrap_length,
        reused_count=reused_count,
        n_cuts=n_cuts,
    )
    pi_mass = _safe_ratio(reference_mass, total_weight)
    pi_ec = _safe_ratio(reference_ec, sustainability_ec)
    pi_mr = _safe_ratio(merchant_rankine_value, reference_summary["merchant_rankine"])
    pi_cost = _safe_ratio(reference_cost, total_cost)

    feasible = bool(final_state.get("analysis_ok", False)) and all(float(v) <= 0.0 for v in constraints.values())
    reuse_timings = {}
    if isinstance(reuse_result, Mapping):
        reuse_timings = dict(reuse_result.get("timings", {}) or {})
    mech_timings = {}
    if isinstance(opt_result, Mapping):
        mech_timings = dict(opt_result.get("optimization_timings", {}) or {})
    t_total = time.perf_counter() - t_start
    return CandidateEvaluation(
        x=x_tuple,
        feasible=feasible,
        objectives={
            "mass": total_weight,
            "cost": float(total_cost),
            "reuse_ratio": reuse_ratio_length,
            "reuse_efficiency": float(reuse_efficiency),
            "reuse_ratio_members": reuse_metrics["reuse_ratio_members"],
            "dcr_max": dcr_max,
            "max_abs_uz": max_abs_uz,
            "lambda_ref": float(lambda_ref) if lambda_ref is not None else float("-inf"),
            "merchant_rankine": (
                float(merchant_rankine_value) if merchant_rankine_value is not None else float("-inf")
            ),
            "ec": float(sustainability_ec),
            "pi_mass": float(pi_mass) if math.isfinite(float(pi_mass)) else float("-inf"),
            "pi_ec": float(pi_ec) if math.isfinite(float(pi_ec)) else float("-inf"),
            "pi_mr": float(pi_mr) if math.isfinite(float(pi_mr)) else float("-inf"),
            "pi_cost": float(pi_cost) if math.isfinite(float(pi_cost)) else float("-inf"),
        },
        constraints=constraints,
        moved_G=moved_G,
        moved_M=moved_M,
        opt_result=opt_result,
        reuse_result=reuse_result,
        final_state=final_state,
        section_by_member=final_section_by_member,
        lambda_ref=None if lambda_ref is None else float(lambda_ref),
        meta={
            "reuse_ratio_members": reuse_metrics["reuse_ratio_members"],
            "reuse_ratio_length": reuse_ratio_length,
            "reused_count": int(reused_count),
            "reuse_efficiency": float(reuse_efficiency),
            "unmatched_count": 0 if not reuse_result else len(reuse_result.get("unmatched_members", []) or []),
            "skipped_reuse_count": 0 if not reuse_result else len(reuse_result.get("skipped_members", []) or []),
            "reused_length": float(reused_length),
            "reused_mass": float(reused_mass),
            "new_mass": float(new_mass),
            "new_reference_mass": float(new_reference_total_weight),
            "new_reference_mass_tonnes": float(new_reference_mass_tonnes),
            "comparison_reference_mode": str(reference_mode),
            "comparison_reference_label": str(reference_label),
            "comparison_reference_mass": float(reference_mass),
            "comparison_reference_mass_tonnes": float(reference_mass_tonnes),
            "mscrap": float(scrap_mass_tonnes),
            "scrap_mass": float(scrap_mass),
            "mass_tonnes": float(total_mass_tonnes),
            "reused_mass_tonnes": float(reused_mass_tonnes),
            "new_mass_tonnes": float(new_mass_tonnes),
            "scrap_mass_tonnes": float(scrap_mass_tonnes),
            "ec": float(sustainability_ec),
            "new_reference_ec": float(new_reference_ec),
            "comparison_reference_ec": float(reference_ec),
            "cost": float(total_cost),
            "new_reference_cost": float(new_reference_cost),
            "comparison_reference_cost": float(reference_cost),
            "ecf_new": float(ctx.ecf_new),
            "ecf_reused": float(ctx.ecf_reused),
            "ecf_work": float(ctx.ecf_work),
            "ecf_scrap": float(ctx.ecf_scrap),
            "cost_new": float(ctx.cost_new),
            "cost_reused": float(ctx.cost_reused),
            "cost_work": float(ctx.cost_work),
            "cost_scrap": float(ctx.cost_scrap),
            "lscrap": float(scrap_length),
            "n_cuts": int(n_cuts),
            "reuse_member_mask_enabled": bool(ctx.reuse_member_mask_enabled),
            "reuse_member_mask_mode": str(ctx.reuse_member_mask_mode),
            "reuse_member_mask_group_count": int(len(ctx.reuse_member_mask_groups)),
            "reuse_selected_count": (len(eligible_reuse_members) if eligible_reuse_members is not None else len(section_by_member)),
            "reuse_member_mask": {f"{int(mk[0])}-{int(mk[1])}": int(flag) for mk, flag in reuse_mask_values.items()},
            "reuse_eligible_members": ([] if eligible_reuse_members is None else [list(mk) for mk in eligible_reuse_members]),
            "analysis_ok": bool(final_state.get("analysis_ok", False)),
            "dcr_max": dcr_max,
            "dcr_mean": dcr_mean,
            "max_abs_uz": max_abs_uz,
            "merchant_rankine": merchant_rankine_value,
            "merchant_rankine_mode_index": int(final_summary["merchant_rankine_mode_index"]),
            "merchant_rankine_lambda": mr_lambda,
            "new_reference_dcr_max": float(new_reference_summary["dcr_max"]),
            "new_reference_dcr_mean": float(new_reference_summary["dcr_mean"]),
            "new_reference_max_abs_uz": float(new_reference_summary["max_abs_uz"]),
            "new_reference_lambda_ref": (
                None if new_reference_summary["lambda_ref"] is None else float(new_reference_summary["lambda_ref"])
            ),
            "new_reference_merchant_rankine": new_reference_summary["merchant_rankine"],
            "new_reference_merchant_rankine_lambda": new_reference_summary["merchant_rankine_lambda"],
            "comparison_reference_dcr_max": float(reference_summary["dcr_max"]),
            "comparison_reference_dcr_mean": float(reference_summary["dcr_mean"]),
            "comparison_reference_max_abs_uz": float(reference_summary["max_abs_uz"]),
            "comparison_reference_lambda_ref": (
                None if reference_summary["lambda_ref"] is None else float(reference_summary["lambda_ref"])
            ),
            "comparison_reference_merchant_rankine": reference_summary["merchant_rankine"],
            "comparison_reference_merchant_rankine_lambda": reference_summary["merchant_rankine_lambda"],
            "pi_mass": float(pi_mass) if math.isfinite(float(pi_mass)) else float("nan"),
            "pi_ec": float(pi_ec) if math.isfinite(float(pi_ec)) else float("nan"),
            "pi_mr": float(pi_mr) if math.isfinite(float(pi_mr)) else float("nan"),
            "pi_cost": float(pi_cost) if math.isfinite(float(pi_cost)) else float("nan"),
            "ga_control_values": dict(ga_control_values),
            "ga_dcr_limit_used": ga_dcr_limit,
            "ga_disp_limit_used": ga_disp_limit,
            "ga_buckling_lambda_min_used": ga_buckling_lambda_min,
            "ga_merchant_rankine_min_used": ga_merchant_rankine_min,
            "reuse_max_cut_length_used": (None if reuse_max_cut_length is None else float(reuse_max_cut_length)),
            "opt_dcr_limit_used": opt_dcr_limit,
            "opt_dlim_used": opt_dlim,
            "opt_buckling_lambda_min_used": opt_buckling_lambda_min,
            "opt_merchant_rankine_min_used": opt_merchant_rankine_min,
            "time_geom_s": t_geom,
            "time_mech_s": t_mech,
            "time_match_s": t_match,
            "time_recheck_new_s": t_recheck_new,
            "time_recheck_s": t_recheck,
            "time_total_s": t_total,
            "time_mech_inner_total_s": mech_timings.get("time_total_s", 0.0),
            "n_mech_state_calls": mech_timings.get("state_calls_total", 0),
            "time_mech_state_total_s": mech_timings.get("state_time_total_s", 0.0),
            "n_mech_base_calls": mech_timings.get("state_calls_base", 0),
            "time_mech_base_s": mech_timings.get("state_time_base_s", 0.0),
            "n_mech_buckling_calls": mech_timings.get("state_calls_buckling", 0),
            "time_mech_buckling_s": mech_timings.get("state_time_buckling_s", 0.0),
            "time_match_single_search_s": reuse_timings.get("time_single_search_s", 0.0),
            "time_match_group_search_s": reuse_timings.get("time_group_search_s", 0.0),
            "time_match_stock_update_s": reuse_timings.get("time_stock_update_s", 0.0),
            "n_match_single_checked": reuse_timings.get("n_single_checked", 0.0),
            "n_match_group_checked": reuse_timings.get("n_group_checked", 0.0),
        },
    )
