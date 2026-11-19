from __future__ import annotations

import math
import time
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from OS_opt_sections_ec3 import (
    _build_elastic_state,
    _build_similarity_groups,
    _disp_uplift_jump,
    _group_indices_to_member_indices,
    _group_section_mapping,
    _is_feasible_state,
    _member_envelopes,
    _member_local_dcrs,
    _member_key,
    _penalty_score,
    _resistance_upgrade_plan,
    _section_name,
)


MemberKey = Tuple[int, int]
SectionDict = Dict[str, Any]


def _make_section_by_member(M: Mapping[str, Any], sec: SectionDict) -> Dict[MemberKey, SectionDict]:
    return {_member_key(a, b): dict(sec) for (a, b) in M.get("members", [])}


def _clamp_index(idx: int, n_sections: int) -> int:
    return max(0, min(int(n_sections) - 1, int(idx)))


def _global_uplift(group_to_idx: Mapping[str, int], n_sections: int, jump: int) -> Tuple[Dict[str, int], int]:
    upgraded = dict(group_to_idx)
    changed = 0
    for gname, idx in group_to_idx.items():
        new_idx = _clamp_index(int(idx) + int(jump), n_sections)
        if new_idx > int(idx):
            upgraded[gname] = new_idx
            changed += 1
    return upgraded, changed


def _state_timings_template() -> Dict[str, float]:
    return {
        "time_total_s": 0.0,
        "state_calls_total": 0,
        "state_time_total_s": 0.0,
        "state_calls_base": 0,
        "state_time_base_s": 0.0,
        "state_calls_buckling": 0,
        "state_time_buckling_s": 0.0,
    }


def solve_genetic_fast_mech(
    G: Mapping[str, Any],
    M: Mapping[str, Any],
    *,
    mecc_opt: int,
    material: Any,
    catalog: Mapping[str, Any],
    candidate_sections: Sequence[SectionDict],
    units_in: str,
    units_out: str,
    c1: Sequence[int],
    c2: Sequence[int],
    c3: Sequence[int],
    c4: Sequence[int],
    c5: Sequence[int],
    q_load: Sequence[Any],
    ndm: int,
    ndf: int,
    init_model,
    define_nodes,
    apply_costr,
    el_ele,
    apply_loads,
    run_static,
    initial_section_name: Optional[str] = None,
    start_from_first_section: bool = False,
    start_section_row: Optional[int] = None,
    dcr_limit: float = 1.0,
    dlim: Optional[float] = None,
    check_disp: bool = True,
    check_buckling: bool = False,
    buckling_lambda_min: float = 1.0,
    buckling_mode_index: int = 1,
    check_merchant_rankine: bool = False,
    merchant_rankine_min: float = 1.0,
    merchant_rankine_mode_index: Optional[int] = None,
    gamma_M0: float = 1.0,
    gamma_M1: float = 1.0,
    gamma_M2: float = 1.25,
    beta: float = 1.0,
    alpha_LT: Optional[float] = None,
    compression_sign: str = "negative",
    max_iter: int = 6,
    subdivide: int = 1,
    first_new_node_tag: int = 2000,
    update_G: bool = False,
    area_mode: str = "faces",
) -> Dict[str, Any]:
    if not candidate_sections:
        raise ValueError("candidate_sections is empty.")

    t_start = time.perf_counter()
    timings = _state_timings_template()
    members = [_member_key(a, b) for (a, b) in M.get("members", [])]
    if not members:
        raise ValueError("M['members'] is empty.")

    name_to_index = {_section_name(sec): idx for idx, sec in enumerate(candidate_sections)}
    start_idx = 0
    if start_section_row is not None:
        start_idx = _clamp_index(int(start_section_row) - 1, len(candidate_sections))
    elif not bool(start_from_first_section) and initial_section_name and initial_section_name in name_to_index:
        start_idx = int(name_to_index[initial_section_name])

    def build_state(section_by_member: Mapping[MemberKey, SectionDict]) -> Dict[str, Any]:
        t0 = time.perf_counter()
        state = _build_elastic_state(
            G,
            M,
            section_by_member=section_by_member,
            material=material,
            ndm=ndm,
            ndf=ndf,
            c1=c1,
            c2=c2,
            c3=c3,
            c4=c4,
            c5=c5,
            q_load=q_load,
            catalog=catalog,
            units_in=units_in,
            units_out=units_out,
            init_model=init_model,
            define_nodes=define_nodes,
            apply_costr=apply_costr,
            el_ele=el_ele,
            apply_loads=apply_loads,
            run_static=run_static,
            subdivide=subdivide,
            first_new_node_tag=first_new_node_tag,
            update_G=update_G,
            area_mode=area_mode,
            gamma_M0=gamma_M0,
            gamma_M1=gamma_M1,
            gamma_M2=gamma_M2,
            beta=beta,
            alpha_LT=alpha_LT,
            compression_sign=compression_sign,
            dlim=dlim,
            compute_buckling=False,
            buckling_lambda_min=buckling_lambda_min,
            buckling_mode_index=buckling_mode_index,
            merchant_rankine_enabled=bool(check_merchant_rankine),
            merchant_rankine_min=merchant_rankine_min,
            merchant_rankine_mode_index=merchant_rankine_mode_index,
            internal_quiet=True,
        )
        dt = time.perf_counter() - t0
        timings["state_calls_total"] += 1
        timings["state_time_total_s"] += dt
        timings["state_calls_base"] += 1
        timings["state_time_base_s"] += dt
        return state

    def build_advanced_state(section_by_member: Mapping[MemberKey, SectionDict]) -> Dict[str, Any]:
        if not bool(check_buckling) and not bool(check_merchant_rankine):
            return build_state(section_by_member)
        t0 = time.perf_counter()
        state = _build_elastic_state(
            G,
            M,
            section_by_member=section_by_member,
            material=material,
            ndm=ndm,
            ndf=ndf,
            c1=c1,
            c2=c2,
            c3=c3,
            c4=c4,
            c5=c5,
            q_load=q_load,
            catalog=catalog,
            units_in=units_in,
            units_out=units_out,
            init_model=init_model,
            define_nodes=define_nodes,
            apply_costr=apply_costr,
            el_ele=el_ele,
            apply_loads=apply_loads,
            run_static=run_static,
            subdivide=subdivide,
            first_new_node_tag=first_new_node_tag,
            update_G=update_G,
            area_mode=area_mode,
            gamma_M0=gamma_M0,
            gamma_M1=gamma_M1,
            gamma_M2=gamma_M2,
            beta=beta,
            alpha_LT=alpha_LT,
            compression_sign=compression_sign,
            dlim=dlim,
            compute_buckling=True,
            buckling_lambda_min=buckling_lambda_min,
            buckling_mode_index=buckling_mode_index,
            merchant_rankine_enabled=bool(check_merchant_rankine),
            merchant_rankine_min=merchant_rankine_min,
            merchant_rankine_mode_index=merchant_rankine_mode_index,
            internal_quiet=True,
        )
        dt = time.perf_counter() - t0
        timings["state_calls_total"] += 1
        timings["state_time_total_s"] += dt
        timings["state_calls_buckling"] += 1
        timings["state_time_buckling_s"] += dt
        return state

    best_state: Optional[Dict[str, Any]] = None
    best_penalty = float("inf")
    history = []

    if int(mecc_opt) == 1:
        jump = 2
        idx = start_idx
        while idx < len(candidate_sections):
            sec = candidate_sections[idx]
            section_by_member = _make_section_by_member(M, sec)
            state = build_state(section_by_member)
            dcr_max = float(state.get("ec3_dcr_max", float("inf")))
            disp_ratio = float(state.get("disp_ratio", float("inf")))
            feasible = _is_feasible_state(
                state,
                dcr_limit,
                check_disp,
                check_buckling=False,
                buckling_lambda_min=buckling_lambda_min,
                buckling_mode_index=buckling_mode_index,
                check_merchant_rankine=False,
                merchant_rankine_min=merchant_rankine_min,
            )
            if feasible and (bool(check_buckling) or bool(check_merchant_rankine)):
                state = build_advanced_state(section_by_member)
                dcr_max = float(state.get("ec3_dcr_max", float("inf")))
                disp_ratio = float(state.get("disp_ratio", float("inf")))
                feasible = _is_feasible_state(
                    state,
                    dcr_limit,
                    check_disp,
                    check_buckling=bool(check_buckling),
                    buckling_lambda_min=buckling_lambda_min,
                    buckling_mode_index=buckling_mode_index,
                    check_merchant_rankine=bool(check_merchant_rankine),
                    merchant_rankine_min=merchant_rankine_min,
                )
            penalty = _penalty_score(dcr_max, disp_ratio, dcr_limit)
            history.append({"idx": idx, "dcr_max": dcr_max, "disp_ratio": disp_ratio, "feasible": feasible})
            if penalty < best_penalty:
                best_penalty = penalty
                best_state = {
                    **state,
                    "props_global": dict(sec),
                    "section_by_member": section_by_member,
                    "section_name_by_member": {mk: _section_name(sec) for mk in members},
                    "selected_section": dict(sec),
                    "mecc_opt": 1,
                    "history": history,
                }
            if feasible:
                best_state = {
                    **state,
                    "props_global": dict(sec),
                    "section_by_member": section_by_member,
                    "section_name_by_member": {mk: _section_name(sec) for mk in members},
                    "selected_section": dict(sec),
                    "mecc_opt": 1,
                    "history": history,
                }
                break
            if not bool(state.get("analysis_ok", False)):
                idx += max(2, jump)
            elif check_disp and disp_ratio > 1.0:
                idx += _disp_uplift_jump(disp_ratio)
            else:
                idx += max(1, jump)

        result = best_state or {
            "analysis_ok": False,
            "props_global": None,
            "section_by_member": None,
            "section_name_by_member": None,
            "history": history,
            "mecc_opt": 1,
        }
        timings["time_total_s"] = time.perf_counter() - t_start
        return {**result, "optimization_timings": dict(timings)}

    group_to_members = _build_similarity_groups(G, M, members)
    group_to_idx = {gname: int(start_idx) for gname in group_to_members}

    for it in range(1, max(1, int(max_iter)) + 1):
        current_map = _group_section_mapping(members, group_to_members, group_to_idx, candidate_sections)
        state = build_state(current_map)
        dcr_max = float(state.get("ec3_dcr_max", float("inf")))
        disp_ratio = float(state.get("disp_ratio", float("inf")))
        feasible = _is_feasible_state(
            state,
            dcr_limit,
            check_disp,
            check_buckling=False,
            buckling_lambda_min=buckling_lambda_min,
            buckling_mode_index=buckling_mode_index,
            check_merchant_rankine=False,
            merchant_rankine_min=merchant_rankine_min,
        )
        if feasible and (bool(check_buckling) or bool(check_merchant_rankine)):
            state = build_advanced_state(current_map)
            dcr_max = float(state.get("ec3_dcr_max", float("inf")))
            disp_ratio = float(state.get("disp_ratio", float("inf")))
            feasible = _is_feasible_state(
                state,
                dcr_limit,
                check_disp,
                check_buckling=bool(check_buckling),
                buckling_lambda_min=buckling_lambda_min,
                buckling_mode_index=buckling_mode_index,
                check_merchant_rankine=bool(check_merchant_rankine),
                merchant_rankine_min=merchant_rankine_min,
            )
        penalty = _penalty_score(dcr_max, disp_ratio, dcr_limit)
        history.append({
            "iter": it,
            "dcr_max": dcr_max,
            "disp_ratio": disp_ratio,
            "feasible": feasible,
            "analysis_ok": bool(state.get("analysis_ok", False)),
        })

        if penalty < best_penalty:
            best_penalty = penalty
            best_state = {
                **state,
                "props_global": dict(candidate_sections[next(iter(group_to_idx.values()))]) if len(set(group_to_idx.values())) == 1 else None,
                "section_by_member": current_map,
                "section_name_by_member": {mk: _section_name(current_map[mk]) for mk in members},
                "mecc_opt": 2,
                "history": history,
                "group_to_members": group_to_members,
                "group_to_idx": dict(group_to_idx),
            }

        if feasible:
            best_state = {
                **state,
                "props_global": dict(candidate_sections[next(iter(group_to_idx.values()))]) if len(set(group_to_idx.values())) == 1 else None,
                "section_by_member": current_map,
                "section_name_by_member": {mk: _section_name(current_map[mk]) for mk in members},
                "mecc_opt": 2,
                "history": history,
                "group_to_members": group_to_members,
                "group_to_idx": dict(group_to_idx),
            }
            break

        if not bool(state.get("analysis_ok", False)):
            group_to_idx, changed = _global_uplift(group_to_idx, len(candidate_sections), 3)
            if changed <= 0:
                break
            continue

        if (bool(check_buckling) or bool(check_merchant_rankine)) and not feasible:
            group_to_idx, changed = _global_uplift(group_to_idx, len(candidate_sections), 1)
            if changed <= 0:
                break
            continue

        if check_disp and disp_ratio > 1.0:
            jump = _disp_uplift_jump(disp_ratio)
            group_to_idx, changed = _global_uplift(group_to_idx, len(candidate_sections), jump)
            if changed <= 0:
                break
            continue

        envelopes = _member_envelopes(state["res"]["ele_forces"], state["ele_data"], compression_sign)
        local_dcrs = _member_local_dcrs(state.get("ec3_results", {}), members)
        group_local_dcrs = {
            gname: max(float(local_dcrs.get(mk, 0.0)) for mk in gmembers)
            for gname, gmembers in group_to_members.items()
        }
        group_to_idx_new, changed = _resistance_upgrade_plan(
            group_to_idx=group_to_idx,
            group_local_dcrs=group_local_dcrs,
            candidate_sections=candidate_sections,
            dcr_max=dcr_max,
            dcr_limit=dcr_limit,
            remaining_iters=max(1, int(max_iter) - int(it)),
            max_iter=max_iter,
        )
        if changed <= 0:
            group_to_idx_new, changed = _global_uplift(group_to_idx, len(candidate_sections), 1)
        group_to_idx = group_to_idx_new
        if changed <= 0:
            break

    result = best_state or {
        "analysis_ok": False,
        "props_global": None,
        "section_by_member": None,
        "section_name_by_member": None,
        "history": history,
        "mecc_opt": 2,
        "group_to_members": group_to_members,
        "group_to_idx": dict(group_to_idx),
    }
    timings["time_total_s"] = time.perf_counter() - t_start
    return {**result, "optimization_timings": dict(timings)}
