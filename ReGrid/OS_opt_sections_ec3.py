from __future__ import annotations

import math
import contextlib
import io
import os
import sys
import time
from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import openseespy.opensees as ops
from import_section import get_section_properties
from Merchant_Rankine import merchant_rankine_factor
from OS_LB_fast import get_tangent_stiffness, linear_buckling_first_mode_sparse, linear_buckling_sparse
from OS_self_weigth import apply_selfweight
from OS_results import extract_results
from OS_EC3_structuralcheck import (
    run_ec3_check,
    _actions_from_record,
    _compute_one_check,
    _build_member_lengths,
)


MemberKey = Tuple[int, int]
SectionDict = Dict[str, Any]
MaterialLike = Union[Mapping[str, Any], Sequence[Any]]


def _member_key(a: int, b: int) -> MemberKey:
    a = int(a)
    b = int(b)
    return (a, b) if a < b else (b, a)


def _parse_material(material: MaterialLike) -> Dict[str, Any]:
    if isinstance(material, dict):
        out = dict(material)
        if "type" in out:
            out["type"] = str(out["type"]).strip()
        return out

    if isinstance(material, (tuple, list)):
        if len(material) < 4:
            raise ValueError("material tuple must be at least ('elastic', fyk, E, G)")
        mtype = str(material[0]).strip()
        out: Dict[str, Any] = {
            "type": mtype,
            "fyk": float(material[1]),
            "E": float(material[2]),
            "G": float(material[3]),
        }
        if mtype.lower() in {"steel01", "steel02"} and len(material) >= 5:
            out["b"] = float(material[4])
        elif mtype.lower() in {"ppmat", "elasticpp"} and len(material) >= 5:
            out["epsyP"] = float(material[4])
            out["epsyN"] = float(material[4])
            out["eps0"] = 0.0
        elif mtype.lower() == "hardening" and len(material) >= 5:
            out = {
                "type": mtype,
                "E": float(material[1]),
                "fyk": float(material[2]),
                "H_iso": float(material[3]),
                "H_kin": float(material[4]),
                "G": 0.0,
            }
        return out

    raise TypeError("material must be a dict or tuple/list.")


def _all_same_section(section_by_member: Mapping[MemberKey, SectionDict]) -> bool:
    names = {str(sec.get("name", "")).strip() for sec in section_by_member.values()}
    return len(names) <= 1


def _section_name(sec: Mapping[str, Any]) -> str:
    return str(sec.get("name", "")).strip()


def _normalize_section_token(name: Any) -> str:
    s = str(name or "").strip().lower()
    s = s.replace("ã˜", "ø").replace("Ų", "ø").replace("ø", "o")
    s = s.replace(",", ".")
    s = "".join(ch for ch in s if ch.isalnum() or ch in {".", "_"})
    return s


def _make_section_by_member(M: Mapping[str, Any], sec: SectionDict) -> Dict[MemberKey, SectionDict]:
    return {_member_key(a, b): dict(sec) for (a, b) in M.get("members", [])}


def _safe_get_disp_ratio(res: Mapping[str, Any], dlim: Optional[float]) -> float:
    if dlim is None or float(dlim) <= 0.0:
        return 0.0
    dlim = float(dlim)
    try:
        max_abs_nd = res.get("max_vertical", {}).get("max_abs_nd", None)
        if isinstance(max_abs_nd, Mapping):
            val = max_abs_nd.get("Uz", max_abs_nd.get("value", None))
            if val is not None:
                return abs(float(val)) / dlim
    except Exception:
        pass
    try:
        uz_min = float(res.get("max_vertical", {}).get("min", {}).get("Uz", 0.0))
        uz_max = float(res.get("max_vertical", {}).get("max", {}).get("Uz", 0.0))
        return max(abs(uz_min), abs(uz_max)) / dlim
    except Exception:
        return float("inf")


def _safe_merchant_rankine(dcr_max: Any, lambda_value: Any) -> Optional[float]:
    try:
        dcr = float(dcr_max)
        lam = float(lambda_value)
        if not math.isfinite(dcr) or not math.isfinite(lam) or dcr <= 0.0 or lam <= 0.0:
            return None
        return float(merchant_rankine_factor(dcr_max=dcr, linear_buckling_factor=lam))
    except Exception:
        return None


def _candidate_sort_value(sec: Mapping[str, Any]) -> Tuple[float, float, str]:
    w = sec.get("w_kN_per_m", None)
    A = sec.get("A", None)
    try:
        wv = float(w) if w is not None else float("inf")
    except Exception:
        wv = float("inf")
    try:
        av = float(A) if A is not None else float("inf")
    except Exception:
        av = float("inf")
    return (wv, av, _section_name(sec))


@contextlib.contextmanager
def _suppress_native_output(enabled: bool):
    if not enabled:
        yield
        return
    devnull_fd = None
    saved_stdout_fd = None
    saved_stderr_fd = None
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    try:
        try:
            ops.logFile("NUL", "-noEcho")
        except Exception:
            pass
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        saved_stdout_fd = os.dup(1)
        saved_stderr_fd = os.dup(2)
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        yield
    finally:
        try:
            if saved_stdout_fd is not None:
                os.dup2(saved_stdout_fd, 1)
            if saved_stderr_fd is not None:
                os.dup2(saved_stderr_fd, 2)
        finally:
            try:
                ops.logFile("stdout")
            except Exception:
                pass
            if saved_stdout_fd is not None:
                os.close(saved_stdout_fd)
            if saved_stderr_fd is not None:
                os.close(saved_stderr_fd)
            if devnull_fd is not None:
                os.close(devnull_fd)


def _call_quiet(fn, *args, quiet: bool = False, **kwargs):
    if not quiet:
        return fn(*args, **kwargs)
    buf_out = io.StringIO()
    buf_err = io.StringIO()
    with _suppress_native_output(True):
        with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
            return fn(*args, **kwargs)


def _sections_label_opt2(member_to_idx: Mapping[MemberKey, int], candidate_sections: Sequence[SectionDict]) -> str:
    names = [_section_name(candidate_sections[int(idx)]) for idx in member_to_idx.values()]
    if not names:
        return "-"
    cnt = Counter(names)
    uniq = sorted(cnt.keys())
    if len(uniq) == 1:
        return uniq[0]
    top_name, top_n = cnt.most_common(1)[0]
    return f"{len(uniq)}sec|top={top_name}({top_n})"


def load_candidate_sections_ec3(
    catalog: Mapping[str, Any],
    units_in: str = "cm",
    units_out: str = "m",
    section_names: Optional[Iterable[str]] = None,
) -> List[SectionDict]:
    if section_names is None:
        sections = get_section_properties(
            catalog=catalog,
            units_in=units_in,
            units_out=units_out,
            print_props=False,
            print_all=False,
        )
    else:
        sections = [
            get_section_properties(
                catalog=catalog,
                name=name,
                units_in=units_in,
                units_out=units_out,
                print_props=False,
            )
            for name in section_names
        ]
    return sorted([dict(s) for s in sections], key=_candidate_sort_value)


def _build_elastic_state(
    G: Mapping[str, Any],
    M: Mapping[str, Any],
    *,
    section_by_member: Mapping[MemberKey, SectionDict],
    material: MaterialLike,
    ndm: int,
    ndf: int,
    c1: Sequence[int],
    c2: Sequence[int],
    c3: Sequence[int],
    c4: Sequence[int],
    c5: Sequence[int],
    q_load: Sequence[Any],
    catalog: Optional[Mapping[str, Any]],
    units_in: str,
    units_out: str,
    init_model,
    define_nodes,
    apply_costr,
    el_ele,
    apply_loads,
    run_static,
    section_rotation_opts=None,
    subdivide: int = 1,
    first_new_node_tag: int = 2000,
    update_G: bool = False,
    transf_type: str = "PDelta",
    area_mode: str = "faces",
    gamma_M0: float = 1.0,
    gamma_M1: float = 1.0,
    gamma_M2: float = 1.25,
    beta: float = 1.0,
    alpha_LT: Optional[float] = None,
    compression_sign: str = "negative",
    dlim: Optional[float] = None,
    compute_buckling: bool = False,
    buckling_lambda_min: float = 1.0,
    buckling_mode_index: int = 1,
    merchant_rankine_enabled: bool = False,
    merchant_rankine_min: float = 1.0,
    merchant_rankine_mode_index: Optional[int] = None,
    internal_quiet: bool = True,
) -> Dict[str, Any]:
    material_d = _parse_material(material)
    tangent_kwargs = dict(system="FullGeneral", numberer="RCM", constraints="Plain", algorithm="Newton")

    init_model(ndm, ndf)
    define_nodes(G, ndf, set_mass=False)
    apply_costr(G, ndf, c1, c2, c3, c4, c5)

    py_ele, ele_data = el_ele(
        G=G,
        M=M,
        material=material_d,
        section=None,
        section_by_member=section_by_member,
        section_rotation_opts=section_rotation_opts,
        transf_type=transf_type,
        subdivide=subdivide,
        first_new_node_tag=first_new_node_tag,
        update_G=update_G,
        print_ops=False,
        print_summary=False,
    )
    _ = py_ele

    _call_quiet(
        apply_selfweight,
        G=G,
        ele_data=ele_data,
        mode="eleload",
        section=None,
        section_by_member=section_by_member,
        catalog=catalog,
        units_in=units_in,
        units_out=units_out,
        pattern_tag=21,
        ts_tag=21,
        print_summary=False,
        quiet=internal_quiet,
    )

    K0 = None
    if bool(compute_buckling):
        K0, _ = get_tangent_stiffness(**tangent_kwargs, print_ops=False)

    loads, areas, py_loads = _call_quiet(
        apply_loads,
        G,
        M=M,
        q=q_load,
        direction="-Z",
        pattern_tag=2,
        ts_tag=2,
        ndf=ndf,
        area_mode=area_mode,
        print_ops=False,
        print_summary=False,
        plot_loads=False,
        quiet=internal_quiet,
    )
    _ = loads, areas, py_loads

    out = _call_quiet(
        run_static,
        load_control=1.0,
        analyze_steps=1,
        set_load_const=False,
        print_ops=False,
        print_summary=False,
        quiet=internal_quiet,
    )
    static_ok = False
    if isinstance(out, (tuple, list)) and len(out) >= 2:
        static_ok = int(out[1]) == 0
    elif isinstance(out, bool):
        static_ok = out
    elif isinstance(out, int):
        static_ok = int(out) == 0
    else:
        static_ok = bool(out)

    state: Dict[str, Any] = {
        "analysis_ok": bool(static_ok),
        "ele_data": ele_data,
        "loads": loads,
        "section_by_member": dict(section_by_member),
        "material": material_d,
        "lb_lambda_1": None,
        "lb_lambda_ref": None,
        "lb_lambdas": [],
        "lb_ok": None,
        "merchant_rankine": None,
        "merchant_rankine_ok": None,
        "merchant_rankine_lambda": None,
        "merchant_rankine_mode_index": None,
    }
    if not static_ok:
        state.update({
            "res": None,
            "ec3_results": None,
            "ec3_dcr_max": float("inf"),
            "disp_ratio": float("inf"),
        })
        return state

    res = extract_results(
        G,
        M,
        ele_data=ele_data,
        ndf=ndf,
        uz_dof=3,
        uz_ref=1.0,
    )
    ec3_results, ec3_dcr_max, sections_by_key, lengths_by_key = run_ec3_check(
        G=G,
        ele_data=ele_data,
        ele_forces=res["ele_forces"],
        material=material_d,
        section=None,
        section_by_member=section_by_member,
        gamma_M0=gamma_M0,
        gamma_M1=gamma_M1,
        gamma_M2=gamma_M2,
        beta=beta,
        alpha_LT=alpha_LT,
        compression_sign=compression_sign,
        aggregate_by_member=True,
        print_summary=False,
        plot_dcr=False,
        save_txt=None,
    )
    disp_ratio = _safe_get_disp_ratio(res, dlim)
    state.update({
        "res": res,
        "ec3_results": ec3_results,
        "ec3_dcr_max": float(ec3_dcr_max),
        "sections_by_key": sections_by_key,
        "lengths_by_key": lengths_by_key,
        "disp_ratio": disp_ratio,
    })

    if bool(compute_buckling) and K0 is not None:
        target_buckling_mode = max(1, int(buckling_mode_index))
        target_mr_mode = max(1, int(merchant_rankine_mode_index or target_buckling_mode))
        n_modes = max(target_buckling_mode, target_mr_mode if merchant_rankine_enabled else 1)

        lambdas: List[float] = []
        if n_modes <= 1:
            lam, _ = linear_buckling_first_mode_sparse(
                K0,
                tangent_kwargs=tangent_kwargs,
                print_summary=False,
            )
            if lam is not None and len(lam) > 0:
                try:
                    lambdas = [float(lam[0])]
                except Exception:
                    lambdas = []
        else:
            lam, _ = linear_buckling_sparse(
                K0,
                n_modes=n_modes,
                drop_tol=1e-12,
                eig_tol=1e-12,
                tangent_kwargs=tangent_kwargs,
                print_ops=False,
                print_summary=False,
            )
            if lam is not None:
                for value in list(lam)[:n_modes]:
                    try:
                        lambdas.append(float(value))
                    except Exception:
                        continue

        lb_lambda_1 = lambdas[0] if lambdas else None
        lb_lambda_ref = lambdas[target_buckling_mode - 1] if len(lambdas) >= target_buckling_mode else None
        state["lb_lambda_1"] = lb_lambda_1
        state["lb_lambda_ref"] = lb_lambda_ref
        state["lb_lambdas"] = list(lambdas)
        state["lb_mode_index"] = int(target_buckling_mode)
        state["lb_ok"] = bool(lb_lambda_ref is not None and math.isfinite(lb_lambda_ref) and lb_lambda_ref >= float(buckling_lambda_min))

        if bool(merchant_rankine_enabled):
            mr_lambda = lambdas[target_mr_mode - 1] if len(lambdas) >= target_mr_mode else None
            mr_value = _safe_merchant_rankine(state.get("ec3_dcr_max", None), mr_lambda)
            state["merchant_rankine"] = mr_value
            state["merchant_rankine_lambda"] = mr_lambda
            state["merchant_rankine_mode_index"] = int(target_mr_mode)
            state["merchant_rankine_ok"] = bool(
                mr_value is not None and math.isfinite(float(mr_value)) and float(mr_value) >= float(merchant_rankine_min)
            )

    return state


def _member_envelopes(ele_forces: Mapping[Any, Any], ele_data: Mapping[str, Any], compression_sign: str):
    from OS_results import build_member_force_envelopes

    return build_member_force_envelopes(
        ele_forces=ele_forces,
        ele_data=ele_data,
        compression_sign=compression_sign,
    )


def _pick_first_feasible_section_for_member(
    *,
    actions: Mapping[str, float],
    length: float,
    candidates: Sequence[SectionDict],
    material: Mapping[str, Any],
    gamma_M0: float,
    gamma_M1: float,
    gamma_M2: float,
    beta: float,
    alpha_LT: Optional[float],
    dcr_limit: float,
    current_index: int,
) -> Tuple[int, Dict[str, Any]]:
    last_idx = current_index
    last_info: Optional[Dict[str, Any]] = None
    for idx in range(max(0, int(current_index)), len(candidates)):
        sec = candidates[idx]
        info = _compute_one_check(
            sec=sec,
            actions=actions,
            l_member=float(length),
            material=material,
            gamma_M0=float(gamma_M0),
            gamma_M1=float(gamma_M1),
            gamma_M2=float(gamma_M2),
            kc=1.0,
            beta=float(beta),
            alpha_LT_user=alpha_LT,
        )
        last_idx = idx
        last_info = info
        dcr = float(info.get("dcr", float("inf")))
        if math.isfinite(dcr) and dcr <= float(dcr_limit):
            return idx, info
    if last_info is None:
        last_info = {"dcr": float("inf")}
    return last_idx, last_info


def _sec_float(sec: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    try:
        val = sec.get(key, default)
        if val is None:
            return float(default)
        return float(val)
    except Exception:
        return float(default)


def _member_compliance_proxy(
    *,
    actions: Mapping[str, float],
    length: float,
    sec: Mapping[str, Any],
    material: Mapping[str, Any],
) -> float:
    E = max(float(material.get("E", 0.0) or 0.0), 1e-12)
    G = max(float(material.get("G", 0.0) or 0.0), 1e-12)
    L = max(float(length), 1e-12)

    A = max(_sec_float(sec, "A", 0.0), 1e-12)
    Iy = max(_sec_float(sec, "Iy", 0.0), 1e-12)
    Iz = max(_sec_float(sec, "Iz", 0.0), 1e-12)
    J = max(_sec_float(sec, "J", 0.0), 1e-12)

    N = abs(float(actions.get("N_abs", 0.0)))
    My = abs(float(actions.get("My", actions.get("My_ED_Max", 0.0))))
    Mz = abs(float(actions.get("Mz", actions.get("Mz_ED_Max", 0.0))))
    T = abs(float(actions.get("T", 0.0)))

    axial = (N * N) / (E * A)
    bend_y = (My * My) / (E * Iy)
    bend_z = (Mz * Mz) / (E * Iz)
    tors = (T * T) / (G * J)
    return float(L * (axial + bend_y + bend_z + tors))


def _displacement_upgrade_plan(
    *,
    members: Sequence[MemberKey],
    member_to_idx: Mapping[MemberKey, int],
    current_map: Mapping[MemberKey, SectionDict],
    candidate_sections: Sequence[SectionDict],
    envelopes: Mapping[MemberKey, Mapping[str, Any]],
    lengths: Mapping[MemberKey, float],
    material: Mapping[str, Any],
    compression_sign: str,
    disp_ratio: float,
    local_dcrs: Optional[Mapping[MemberKey, float]] = None,
) -> Tuple[Dict[MemberKey, int], int, Dict[MemberKey, float]]:
    local_dcrs = dict(local_dcrs or {})
    max_idx = len(candidate_sections) - 1
    severity = max(0.0, float(disp_ratio) - 1.0)
    step_jump = 1
    if severity >= 1.0:
        step_jump = 3
    elif severity >= 0.35:
        step_jump = 2

    ranked: List[Tuple[float, MemberKey, int]] = []
    score_map: Dict[MemberKey, float] = {}

    for mk in members:
        cur_idx = int(member_to_idx[mk])
        if cur_idx >= max_idx or mk not in envelopes:
            continue

        next_idx = min(max_idx, cur_idx + step_jump)
        cur_sec = current_map[mk]
        next_sec = candidate_sections[next_idx]
        actions = _actions_from_record(envelopes[mk], compression_sign=compression_sign)
        L = float(lengths.get(mk, 0.0))

        c_cur = _member_compliance_proxy(actions=actions, length=L, sec=cur_sec, material=material)
        c_next = _member_compliance_proxy(actions=actions, length=L, sec=next_sec, material=material)
        benefit = max(0.0, c_cur - c_next)

        w_cur = max(_sec_float(cur_sec, "w_kN_per_m", _sec_float(cur_sec, "A", 0.0)), 0.0)
        w_next = max(_sec_float(next_sec, "w_kN_per_m", _sec_float(next_sec, "A", 0.0)), w_cur)
        dw = max(1e-12, w_next - w_cur)
        dcr_bias = 1.0 + max(0.0, float(local_dcrs.get(mk, 0.0)) - 0.75)
        score = (benefit / dw) * dcr_bias

        if score > 0.0:
            ranked.append((score, mk, next_idx))
            score_map[mk] = float(score)

    if not ranked:
        return dict(member_to_idx), 0, score_map

    ranked.sort(key=lambda x: (-x[0], x[1][0], x[1][1]))

    frac = min(0.35, 0.08 + 0.18 * severity)
    quota = max(1, min(len(ranked), int(math.ceil(len(ranked) * frac))))

    total_score = sum(score for score, _, _ in ranked)
    cumulative = 0.0
    selected: List[Tuple[float, MemberKey, int]] = []
    target_share = 0.6 if severity < 0.5 else 0.75
    for item in ranked:
        selected.append(item)
        cumulative += item[0]
        if len(selected) >= quota and (total_score <= 0.0 or cumulative / total_score >= target_share):
            break

    upgraded = dict(member_to_idx)
    changed = 0
    for _, mk, next_idx in selected:
        if int(next_idx) > int(upgraded[mk]):
            upgraded[mk] = int(next_idx)
            changed += 1

    return upgraded, changed, score_map


def _dcr_uplift_jump(dcr_max: float, dcr_limit: float, *, remaining_iters: Optional[int] = None, max_iter: Optional[int] = None) -> int:
    if float(dcr_limit) <= 0.0:
        return 1
    ratio = float(dcr_max) / float(dcr_limit)
    if not math.isfinite(ratio) or ratio <= 1.0:
        return 1
    if ratio >= 8.0:
        jump = 6
    elif ratio >= 4.0:
        jump = 4
    elif ratio >= 2.0:
        jump = 2
    else:
        jump = 1

    if remaining_iters is not None and max_iter is not None and int(max_iter) > 0:
        rem = max(1, int(remaining_iters))
        total = max(1, int(max_iter))
        used_frac = 1.0 - (float(rem) / float(total))
        if rem <= max(2, int(math.ceil(0.15 * total))):
            jump += 2
        elif rem <= max(3, int(math.ceil(0.30 * total))):
            jump += 1
        if used_frac >= 0.75:
            jump += 1
    return max(1, int(jump))


def _resistance_upgrade_plan(
    *,
    group_to_idx: Mapping[str, int],
    group_local_dcrs: Mapping[str, float],
    candidate_sections: Sequence[SectionDict],
    dcr_max: float,
    dcr_limit: float,
    remaining_iters: Optional[int] = None,
    max_iter: Optional[int] = None,
) -> Tuple[Dict[str, int], int]:
    max_idx = len(candidate_sections) - 1
    jump = _dcr_uplift_jump(dcr_max, dcr_limit, remaining_iters=remaining_iters, max_iter=max_iter)
    ratio = float(dcr_max) / max(float(dcr_limit), 1e-12)

    ranked = [
        (float(group_local_dcrs.get(gname, 0.0)), gname)
        for gname in group_to_idx
        if int(group_to_idx[gname]) < max_idx
    ]
    ranked = [item for item in ranked if item[0] > float(dcr_limit)]
    ranked.sort(key=lambda x: (-x[0], x[1]))
    if not ranked:
        return dict(group_to_idx), 0

    if ratio >= 4.0:
        quota = max(1, int(math.ceil(0.65 * len(ranked))))
    elif ratio >= 2.0:
        quota = max(1, int(math.ceil(0.50 * len(ranked))))
    else:
        quota = max(1, int(math.ceil(0.30 * len(ranked))))

    if remaining_iters is not None and max_iter is not None and int(max_iter) > 0:
        rem = max(1, int(remaining_iters))
        total = max(1, int(max_iter))
        if rem <= max(2, int(math.ceil(0.15 * total))):
            quota = len(ranked)
        elif rem <= max(3, int(math.ceil(0.30 * total))):
            quota = max(quota, int(math.ceil(0.75 * len(ranked))))

    quota = min(quota, len(ranked))

    upgraded = dict(group_to_idx)
    changed = 0
    for _, gname in ranked[:quota]:
        new_idx = min(max_idx, int(upgraded[gname]) + int(jump))
        if new_idx > int(upgraded[gname]):
            upgraded[gname] = int(new_idx)
            changed += 1
    return upgraded, changed


def _terminal_force_group_feasibility_plan(
    *,
    group_to_idx: Mapping[str, int],
    group_to_members: Mapping[str, Sequence[MemberKey]],
    envelopes: Mapping[MemberKey, Mapping[str, Any]],
    lengths: Mapping[MemberKey, float],
    candidates: Sequence[SectionDict],
    material: Mapping[str, Any],
    gamma_M0: float,
    gamma_M1: float,
    gamma_M2: float,
    beta: float,
    alpha_LT: Optional[float],
    dcr_limit: float,
    compression_sign: str,
) -> Tuple[Dict[str, int], int, float]:
    upgraded_idx = dict(group_to_idx)
    changed = 0
    local_dcr_max = 0.0

    for gname, gmembers in group_to_members.items():
        idx_new, info = _pick_first_feasible_section_for_group(
            group_members=gmembers,
            envelopes=envelopes,
            lengths=lengths,
            candidates=candidates,
            material=material,
            gamma_M0=gamma_M0,
            gamma_M1=gamma_M1,
            gamma_M2=gamma_M2,
            beta=beta,
            alpha_LT=alpha_LT,
            dcr_limit=dcr_limit,
            current_index=int(group_to_idx[gname]),
            compression_sign=compression_sign,
        )
        local_dcr_max = max(local_dcr_max, float(info.get("group_dcr_max", info.get("dcr", float("inf")))))
        if int(idx_new) > int(upgraded_idx[gname]):
            upgraded_idx[gname] = int(idx_new)
            changed += 1

    return upgraded_idx, changed, float(local_dcr_max)


def _all_member_indices_equal(member_to_idx: Mapping[MemberKey, int]) -> bool:
    return len(set(int(v) for v in member_to_idx.values())) <= 1


def _current_section_mapping(members: Iterable[MemberKey], indices: Mapping[MemberKey, int], candidates: Sequence[SectionDict]) -> Dict[MemberKey, SectionDict]:
    return {mk: dict(candidates[int(indices[mk])]) for mk in members}


def _line_members(node_ids: Sequence[int]) -> List[MemberKey]:
    out: List[MemberKey] = []
    ids = [int(n) for n in node_ids]
    for a, b in zip(ids[:-1], ids[1:]):
        out.append(_member_key(a, b))
    return out


def _normalize_group_label(prefix: str, level: Any) -> str:
    try:
        lvl = abs(float(level))
        return f"{prefix}|{lvl:.6f}"
    except Exception:
        return f"{prefix}|{level}"


def _build_similarity_groups(G: Mapping[str, Any], M: Mapping[str, Any], members: Sequence[MemberKey]) -> Dict[str, List[MemberKey]]:
    members_set = {_member_key(*mk) for mk in members}
    groups: Dict[str, List[MemberKey]] = {}

    raw_groups = M.get("member_groups", {}) or {}
    for name, group_members in raw_groups.items():
        g_members = []
        for mk in group_members:
            try:
                key = _member_key(*mk)
            except Exception:
                continue
            if key in members_set:
                g_members.append(key)
        if g_members:
            groups[f"predef:{name}"] = sorted(set(g_members))

    lines = M.get("lines", {}) or {}
    x_lines = lines.get("x", {}) or {}
    y_lines = lines.get("y", {}) or {}
    for level, ids in x_lines.items():
        gm = [mk for mk in _line_members(ids) if mk in members_set]
        if gm:
            groups.setdefault(_normalize_group_label("x", level), [])
            groups[_normalize_group_label("x", level)].extend(gm)
    for level, ids in y_lines.items():
        gm = [mk for mk in _line_members(ids) if mk in members_set]
        if gm:
            groups.setdefault(_normalize_group_label("y", level), [])
            groups[_normalize_group_label("y", level)].extend(gm)

    clean_groups: Dict[str, List[MemberKey]] = {}
    assigned: Dict[MemberKey, str] = {}
    for gname in sorted(groups.keys()):
        uniq = []
        seen = set()
        for mk in groups[gname]:
            if mk in seen:
                continue
            seen.add(mk)
            uniq.append(mk)
        if not uniq:
            continue
        clean_groups[gname] = sorted(uniq)
        for mk in uniq:
            assigned.setdefault(mk, gname)

    for mk in members:
        if mk not in assigned:
            clean_groups[f"member:{mk[0]}-{mk[1]}"] = [mk]

    return clean_groups


def _group_indices_to_member_indices(
    members: Sequence[MemberKey],
    group_to_members: Mapping[str, Sequence[MemberKey]],
    group_to_idx: Mapping[str, int],
) -> Dict[MemberKey, int]:
    out: Dict[MemberKey, int] = {}
    for gname, gmembers in group_to_members.items():
        idx = int(group_to_idx[gname])
        for mk in gmembers:
            out[_member_key(*mk)] = idx
    for mk in members:
        out.setdefault(_member_key(*mk), 0)
    return out


def _group_section_mapping(
    members: Sequence[MemberKey],
    group_to_members: Mapping[str, Sequence[MemberKey]],
    group_to_idx: Mapping[str, int],
    candidates: Sequence[SectionDict],
) -> Dict[MemberKey, SectionDict]:
    member_to_idx = _group_indices_to_member_indices(members, group_to_members, group_to_idx)
    return _current_section_mapping(members, member_to_idx, candidates)


def _demand_score_from_actions(actions: Mapping[str, float], length: float, local_dcr: float) -> float:
    N = abs(float(actions.get("N_abs", 0.0)))
    My = abs(float(actions.get("My", 0.0)))
    Mz = abs(float(actions.get("Mz", 0.0)))
    T = abs(float(actions.get("T", 0.0)))
    L = max(float(length), 1e-9)
    dcr = max(0.0, float(local_dcr))
    return float(dcr + 0.15 * L + 0.35 * N + 0.30 * My + 0.30 * Mz + 0.05 * T)


def _refine_groups_by_demand(
    *,
    group_to_members: Mapping[str, Sequence[MemberKey]],
    envelopes: Mapping[MemberKey, Mapping[str, Any]],
    lengths: Mapping[MemberKey, float],
    local_dcrs: Mapping[MemberKey, float],
    compression_sign: str,
    min_group_size: int = 4,
    min_cv: float = 0.35,
    max_splits_per_group: int = 2,
) -> Dict[str, List[MemberKey]]:
    refined: Dict[str, List[MemberKey]] = {}

    for gname, gmembers in group_to_members.items():
        members = [_member_key(*mk) for mk in gmembers]
        if len(members) < int(min_group_size):
            refined[gname] = members
            continue

        scored: List[Tuple[float, MemberKey]] = []
        for mk in members:
            env = envelopes.get(mk)
            if env is None:
                continue
            actions = _actions_from_record(env, compression_sign=compression_sign)
            score = _demand_score_from_actions(
                actions,
                float(lengths.get(mk, 0.0)),
                float(local_dcrs.get(mk, 0.0)),
            )
            scored.append((score, mk))

        if len(scored) < int(min_group_size):
            refined[gname] = members
            continue

        vals = [s for s, _ in scored]
        mean = sum(vals) / len(vals)
        if mean <= 1e-12:
            refined[gname] = members
            continue
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        cv = math.sqrt(var) / mean
        if cv < float(min_cv):
            refined[gname] = members
            continue

        scored.sort(key=lambda x: x[0])
        n_parts = 2 if len(scored) < 10 else min(int(max_splits_per_group) + 1, 3)
        chunk = int(math.ceil(len(scored) / n_parts))
        parts: List[List[MemberKey]] = []
        for i in range(0, len(scored), chunk):
            part = [mk for _, mk in scored[i:i + chunk]]
            if part:
                parts.append(part)

        if len(parts) <= 1:
            refined[gname] = members
            continue

        for idx, part in enumerate(parts, start=1):
            refined[f"{gname}#d{idx}"] = part

    return refined


def _sections_label_opt2_groups(group_to_idx: Mapping[str, int], candidate_sections: Sequence[SectionDict]) -> str:
    names = [_section_name(candidate_sections[int(idx)]) for idx in group_to_idx.values()]
    if not names:
        return "-"
    cnt = Counter(names)
    uniq = sorted(cnt.keys())
    if len(uniq) == 1:
        return f"{uniq[0]}|groups={len(group_to_idx)}"
    top_name, top_n = cnt.most_common(1)[0]
    return f"{len(uniq)}sec|groups={len(group_to_idx)}|top={top_name}({top_n})"


def _write_opt2_group_report(
    *,
    filepath: Optional[str],
    group_to_members: Mapping[str, Sequence[MemberKey]],
    group_to_idx: Mapping[str, int],
    candidate_sections: Sequence[SectionDict],
    lengths_by_key: Optional[Mapping[MemberKey, float]],
    ec3_results: Optional[Mapping[Any, Mapping[str, Any]]],
    feasible: bool,
    dcr_max: float,
    disp_ratio: float,
    history: Sequence[Mapping[str, Any]],
) -> None:
    if not filepath:
        return
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    ec3_results = ec3_results or {}
    lengths_by_key = lengths_by_key or {}

    rows: List[Dict[str, Any]] = []
    for gname, gmembers in group_to_members.items():
        idx = int(group_to_idx.get(gname, 0))
        sec = candidate_sections[idx]
        sec_name = _section_name(sec)
        member_list = [_member_key(*mk) for mk in gmembers]
        total_length = sum(float(lengths_by_key.get(mk, 0.0)) for mk in member_list)

        worst_dcr = -float("inf")
        worst_member = None
        for mk in member_list:
            rec = ec3_results.get(mk, {}) if isinstance(ec3_results, Mapping) else {}
            try:
                dcr = float(rec.get("dcr", float("nan")))
            except Exception:
                dcr = float("nan")
            if math.isfinite(dcr) and dcr > worst_dcr:
                worst_dcr = dcr
                worst_member = mk

        if not math.isfinite(worst_dcr):
            worst_dcr = float("nan")

        rows.append({
            "group": gname,
            "index": idx,
            "section": sec_name,
            "n_members": len(member_list),
            "total_length": total_length,
            "worst_dcr": worst_dcr,
            "worst_member": worst_member,
        })

    rows.sort(key=lambda r: (
        -(r["worst_dcr"] if math.isfinite(r["worst_dcr"]) else -1.0),
        -r["total_length"],
        r["group"],
    ))

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("OPT_MECH_2 FINAL GROUP REPORT\n")
        f.write("============================\n\n")
        f.write(f"feasible = {bool(feasible)}\n")
        f.write(f"DCR_max = {float(dcr_max):.6f}\n")
        f.write(f"d_over_dlim_max = {float(disp_ratio):.6f}\n")
        f.write(f"n_groups = {len(group_to_members)}\n")
        f.write(f"n_history_steps = {len(history)}\n\n")
        f.write(
            f"{'group':34s} {'idx':>5s} {'n_mem':>6s} {'L_tot[m]':>12s} "
            f"{'worst_DCR':>12s} {'worst_member':>18s} {'section':>24s}\n"
        )
        f.write("-" * 130 + "\n")
        for r in rows:
            worst_member = "-" if r["worst_member"] is None else str(r["worst_member"])
            worst_dcr = "nan" if not math.isfinite(r["worst_dcr"]) else f"{r['worst_dcr']:.6f}"
            f.write(
                f"{r['group'][:34]:34s} "
                f"{int(r['index']):5d} "
                f"{int(r['n_members']):6d} "
                f"{float(r['total_length']):12.3f} "
                f"{worst_dcr:>12s} "
                f"{worst_member:>18s} "
                f"{str(r['section'])[:24]:>24s}\n"
            )

        f.write("\nDETAIL - GROUP MEMBERS\n")
        f.write("----------------------\n")
        for r in rows:
            f.write(f"\n[{r['group']}]\n")
            f.write(f"section = {r['section']}\n")
            f.write(f"catalog_index = {int(r['index'])}\n")
            f.write(f"n_members = {int(r['n_members'])}\n")
            f.write(f"total_length = {float(r['total_length']):.6f}\n")
            if math.isfinite(r["worst_dcr"]):
                f.write(f"worst_dcr = {float(r['worst_dcr']):.6f}\n")
            else:
                f.write("worst_dcr = nan\n")
            f.write(f"worst_member = {r['worst_member']}\n")
            f.write("members = " + ", ".join(str(_member_key(*mk)) for mk in group_to_members[r["group"]]) + "\n")


def _pick_first_feasible_section_for_group(
    *,
    group_members: Sequence[MemberKey],
    envelopes: Mapping[MemberKey, Mapping[str, Any]],
    lengths: Mapping[MemberKey, float],
    candidates: Sequence[SectionDict],
    material: Mapping[str, Any],
    gamma_M0: float,
    gamma_M1: float,
    gamma_M2: float,
    beta: float,
    alpha_LT: Optional[float],
    dcr_limit: float,
    current_index: int,
    compression_sign: str,
) -> Tuple[int, Dict[str, Any]]:
    last_idx = int(current_index)
    last_info: Dict[str, Any] = {"dcr": float("inf")}
    start_idx = max(0, int(current_index))

    for idx in range(start_idx, len(candidates)):
        sec = candidates[idx]
        group_dcr_max = -float("inf")
        worst_member = None
        worst_info: Optional[Dict[str, Any]] = None

        for mk in group_members:
            env = envelopes.get(mk)
            if env is None:
                continue
            actions = _actions_from_record(env, compression_sign=compression_sign)
            info = _compute_one_check(
                sec=sec,
                actions=actions,
                l_member=float(lengths.get(mk, 0.0)),
                material=material,
                gamma_M0=float(gamma_M0),
                gamma_M1=float(gamma_M1),
                gamma_M2=float(gamma_M2),
                kc=1.0,
                beta=float(beta),
                alpha_LT_user=alpha_LT,
            )
            dcr = float(info.get("dcr", float("inf")))
            if dcr > group_dcr_max:
                group_dcr_max = dcr
                worst_member = mk
                worst_info = info

        if worst_info is None:
            worst_info = {"dcr": float("inf")}
            group_dcr_max = float("inf")

        last_idx = idx
        last_info = dict(worst_info)
        last_info["group_dcr_max"] = float(group_dcr_max)
        last_info["worst_member"] = worst_member
        if math.isfinite(group_dcr_max) and group_dcr_max <= float(dcr_limit):
            return idx, last_info

    return last_idx, last_info


def _objective_util(dcr_max: float, disp_ratio: float, dcr_limit: float) -> float:
    return float(dcr_max) / max(float(dcr_limit), 1e-18)


def _penalty_score(dcr_max: float, disp_ratio: float, dcr_limit: float) -> float:
    return max(0.0, float(dcr_max) - float(dcr_limit)) * 100.0 + max(0.0, float(disp_ratio) - 1.0) * 10.0

def _disp_uplift_jump(disp_ratio: float) -> int:
    r = float(disp_ratio)
    if not math.isfinite(r) or r <= 1.0:
        return 1
    if r >= 3.0:
        return 4
    if r >= 2.0:
        return 3
    if r >= 1.25:
        return 2
    return 1

def _is_feasible_state(
    state: Mapping[str, Any],
    dcr_limit: float,
    check_disp: bool,
    *,
    check_buckling: bool = False,
    buckling_lambda_min: float = 1.0,
    buckling_mode_index: int = 1,
    check_merchant_rankine: bool = False,
    merchant_rankine_min: float = 1.0,
) -> bool:
    dcr = float(state.get("ec3_dcr_max", float("inf")))
    disp = float(state.get("disp_ratio", float("inf")))
    lb_ok = True
    if bool(check_buckling):
        lb = state.get("lb_lambda_ref", state.get("lb_lambda_1", None))
        try:
            lb_ok = lb is not None and math.isfinite(float(lb)) and float(lb) >= float(buckling_lambda_min)
        except Exception:
            lb_ok = False
    mr_ok = True
    if bool(check_merchant_rankine):
        mr_value = state.get("merchant_rankine", None)
        try:
            mr_ok = mr_value is not None and math.isfinite(float(mr_value)) and float(mr_value) >= float(merchant_rankine_min)
        except Exception:
            mr_ok = False
    return bool(state.get("analysis_ok", False)) and (dcr <= float(dcr_limit)) and ((disp <= 1.0) if check_disp else True) and lb_ok and mr_ok

def _member_local_dcrs(ec3_results: Optional[Mapping[Any, Mapping[str, Any]]], members: Sequence[MemberKey]) -> Dict[MemberKey, float]:
    out: Dict[MemberKey, float] = {}
    results = ec3_results or {}
    for mk in members:
        rec = results.get(mk, {}) if isinstance(results, Mapping) else {}
        try:
            out[mk] = float(rec.get("dcr", 0.0))
        except Exception:
            out[mk] = 0.0
    return out


def _buckling_uplift_jump(current_lambda_1: Optional[float], target_lambda_1: float) -> int:
    try:
        lam = max(float(current_lambda_1), 1e-9)
    except Exception:
        lam = 1e-9
    target = max(float(target_lambda_1), 1e-9)
    ratio = target / lam
    if ratio >= 8.0:
        return 8
    if ratio >= 5.0:
        return 6
    if ratio >= 3.0:
        return 4
    if ratio >= 2.0:
        return 3
    if ratio >= 1.25:
        return 2
    return 1


def _group_buckling_vulnerability(
    *,
    group_members: Sequence[MemberKey],
    envelopes: Mapping[MemberKey, Mapping[str, Any]],
    lengths: Mapping[MemberKey, float],
    compression_sign: str,
) -> float:
    vuln = 0.0
    for mk in group_members:
        env = envelopes.get(mk)
        if env is None:
            continue
        actions = _actions_from_record(env, compression_sign=compression_sign)
        n_comp = max(0.0, float(actions.get("N_compression", 0.0)))
        L = max(float(lengths.get(mk, 0.0)), 1e-12)
        vuln += n_comp * (L ** 2)
    return float(vuln)


def _buckling_group_upgrade_plan(
    *,
    group_to_idx: Mapping[str, int],
    group_to_members: Mapping[str, Sequence[MemberKey]],
    candidate_sections: Sequence[SectionDict],
    envelopes: Mapping[MemberKey, Mapping[str, Any]],
    lengths: Mapping[MemberKey, float],
    compression_sign: str,
    current_lambda_1: Optional[float],
    target_lambda_1: float,
    max_groups: Optional[int] = None,
) -> Tuple[Dict[str, int], int, Dict[str, float]]:
    max_idx = len(candidate_sections) - 1
    ranked: List[Tuple[float, str, int]] = []
    score_map: Dict[str, float] = {}
    jump = _buckling_uplift_jump(current_lambda_1, target_lambda_1)

    for gname, gmembers in group_to_members.items():
        cur_idx = int(group_to_idx.get(gname, 0))
        if cur_idx >= max_idx:
            continue

        cur_sec = candidate_sections[cur_idx]
        next_idx = min(max_idx, cur_idx + int(jump))
        next_sec = candidate_sections[next_idx]

        iy_cur = max(_sec_float(cur_sec, "Iy", 0.0), 0.0)
        iz_cur = max(_sec_float(cur_sec, "Iz", 0.0), 0.0)
        imin_cur = max(min(iy_cur, iz_cur), 1e-18)
        iavg_cur = max(0.5 * (iy_cur + iz_cur), 1e-18)
        a_cur = max(_sec_float(cur_sec, "A", 0.0), 1e-18)

        iy_next = max(_sec_float(next_sec, "Iy", 0.0), iy_cur)
        iz_next = max(_sec_float(next_sec, "Iz", 0.0), iz_cur)
        imin_next = max(min(iy_next, iz_next), imin_cur)
        iavg_next = max(0.5 * (iy_next + iz_next), iavg_cur)
        a_next = max(_sec_float(next_sec, "A", 0.0), a_cur)

        vuln = _group_buckling_vulnerability(
            group_members=gmembers,
            envelopes=envelopes,
            lengths=lengths,
            compression_sign=compression_sign,
        )
        if vuln <= 0.0:
            continue

        rel_imin_gain = max(0.0, (imin_next - imin_cur) / imin_cur)
        rel_iavg_gain = max(0.0, (iavg_next - iavg_cur) / iavg_cur)
        rel_a_gain = max(0.0, (a_next - a_cur) / a_cur)
        stiff_gain = 0.65 * rel_imin_gain + 0.25 * rel_iavg_gain + 0.10 * rel_a_gain
        if stiff_gain <= 0.0:
            continue

        score = vuln * stiff_gain
        if score <= 0.0 or not math.isfinite(score):
            continue

        ranked.append((float(score), gname, next_idx))
        score_map[gname] = float(score)

    if not ranked:
        return dict(group_to_idx), 0, score_map

    ranked.sort(key=lambda x: (-x[0], x[1]))
    if max_groups is None:
        lam = max(float(current_lambda_1 or 0.0), 0.0)
        target = max(float(target_lambda_1), 1e-9)
        gap = max(0.0, (target - lam) / target)
        frac = min(0.80, 0.25 + 0.45 * gap)
        quota = max(1, min(len(ranked), int(math.ceil(frac * len(ranked)))))
    else:
        quota = max(1, min(len(ranked), int(max_groups)))

    upgraded = dict(group_to_idx)
    changed = 0
    for _, gname, next_idx in ranked[:quota]:
        if int(next_idx) > int(upgraded[gname]):
            upgraded[gname] = int(next_idx)
            changed += 1

    return upgraded, changed, score_map


def _buckling_global_upgrade_plan(
    *,
    group_to_idx: Mapping[str, int],
    group_to_members: Mapping[str, Sequence[MemberKey]],
    candidate_sections: Sequence[SectionDict],
    envelopes: Mapping[MemberKey, Mapping[str, Any]],
    lengths: Mapping[MemberKey, float],
    compression_sign: str,
    current_lambda_1: Optional[float],
    target_lambda_1: float,
    max_groups: Optional[int] = None,
) -> Tuple[Dict[str, int], int, Dict[str, float]]:
    max_idx = len(candidate_sections) - 1
    jump = max(1, _buckling_uplift_jump(current_lambda_1, target_lambda_1))
    ranked: List[Tuple[float, str, int]] = []
    score_map: Dict[str, float] = {}

    for gname, gmembers in group_to_members.items():
        cur_idx = int(group_to_idx.get(gname, 0))
        if cur_idx >= max_idx:
            continue
        vuln = _group_buckling_vulnerability(
            group_members=gmembers,
            envelopes=envelopes,
            lengths=lengths,
            compression_sign=compression_sign,
        )
        if vuln <= 0.0:
            continue
        next_idx = min(max_idx, cur_idx + jump)
        ranked.append((float(vuln), gname, next_idx))
        score_map[gname] = float(vuln)

    if not ranked:
        return dict(group_to_idx), 0, score_map

    ranked.sort(key=lambda x: (-x[0], x[1]))
    if max_groups is None:
        quota = max(1, min(len(ranked), int(math.ceil(0.60 * len(ranked)))))
    else:
        quota = max(1, min(len(ranked), int(max_groups)))

    upgraded = dict(group_to_idx)
    changed = 0
    for _, gname, next_idx in ranked[:quota]:
        if int(next_idx) > int(upgraded[gname]):
            upgraded[gname] = int(next_idx)
            changed += 1

    return upgraded, changed, score_map



def optimize_sections_ec3(
    G: Mapping[str, Any],
    M: Mapping[str, Any],
    *,
    mecc_opt: int,
    material: MaterialLike,
    catalog: Mapping[str, Any],
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
    candidate_sections: Optional[Sequence[SectionDict]] = None,
    section_names: Optional[Iterable[str]] = None,
    initial_section_name: Optional[str] = None,
    start_from_first_section: bool = True,
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
    buckling_max_retries: int = 4,
    buckling_group_limit: Optional[int] = None,
    gamma_M0: float = 1.0,
    gamma_M1: float = 1.0,
    gamma_M2: float = 1.25,
    beta: float = 1.0,
    alpha_LT: Optional[float] = None,
    compression_sign: str = "negative",
    max_iter: int = 15,
    subdivide: int = 2,
    first_new_node_tag: int = 2000,
    update_G: bool = False,
    area_mode: str = "faces",
    verbose: Union[bool, int] = True,
    opt_verbose: Optional[int] = None,
    group_report_path: Optional[str] = "outputs/opt_mech_2_group_report.txt",
) -> Dict[str, Any]:
    verbosity = int(opt_verbose) if opt_verbose is not None else (1 if bool(verbose) else 0)
    if candidate_sections is None:
        candidate_sections = load_candidate_sections_ec3(
            catalog=catalog,
            units_in=units_in,
            units_out=units_out,
            section_names=section_names,
        )
    else:
        candidate_sections = sorted([dict(s) for s in candidate_sections], key=_candidate_sort_value)

    if not candidate_sections:
        raise ValueError("No candidate sections available for optimization.")

    material_d = _parse_material(material)
    members = [_member_key(a, b) for (a, b) in M.get("members", [])]
    if not members:
        raise ValueError("M['members'] is empty.")

    name_to_index = {_section_name(sec): idx for idx, sec in enumerate(candidate_sections)}
    norm_name_to_index = {_normalize_section_token(_section_name(sec)): idx for idx, sec in enumerate(candidate_sections)}
    if start_section_row is not None:
        try:
            start_idx = max(0, min(len(candidate_sections) - 1, int(start_section_row) - 1))
        except Exception:
            start_idx = 0
    elif bool(start_from_first_section):
        start_idx = 0
    elif initial_section_name:
        raw_name = str(initial_section_name).strip()
        norm_name = _normalize_section_token(raw_name)
        if raw_name in name_to_index:
            start_idx = int(name_to_index[raw_name])
        elif norm_name in norm_name_to_index:
            start_idx = int(norm_name_to_index[norm_name])
        else:
            start_idx = 0
    else:
        start_idx = 0

    history: List[Dict[str, Any]] = []
    best_state: Optional[Dict[str, Any]] = None
    best_penalty = float("inf")
    mech_timing = {
        "time_total_s": 0.0,
        "state_calls_total": 0,
        "state_time_total_s": 0.0,
        "state_calls_base": 0,
        "state_time_base_s": 0.0,
        "state_calls_buckling": 0,
        "state_time_buckling_s": 0.0,
    }
    t_mech_start = time.perf_counter()

    def _record_state(tag: str, state: Mapping[str, Any], extra: Optional[Mapping[str, Any]] = None):
        entry = {
            "step": len(history) + 1,
            "tag": tag,
            "analysis_ok": bool(state.get("analysis_ok", False)),
            "ec3_dcr_max": float(state.get("ec3_dcr_max", float("inf"))),
            "disp_ratio": float(state.get("disp_ratio", float("inf"))),
            "lb_lambda_1": state.get("lb_lambda_1", None),
        }
        if extra:
            entry.update(dict(extra))
        history.append(entry)

    def _print_step_opt1(step: int, sec_name: str, dcr_max: float, disp_ratio: float, feasible: bool, lb_lambda_1: Optional[float] = None):
        if verbosity < 1:
            return
        msg = (
            f"# SUMMARY | OPT_1 | step={step:02d} | "
            f"sec={sec_name} | "
            f"DCR_max={float(dcr_max):.4f} | "
            f"d/dlim_max={float(disp_ratio):.4f} | "
            f"feasible={bool(feasible)}"
        )
        if lb_lambda_1 is not None and math.isfinite(float(lb_lambda_1)):
            msg += f" | lambda_1={float(lb_lambda_1):.4f}"
        print(msg)

    def _build_state_timed(*args, phase: str = "base", **kwargs):
        t0 = time.perf_counter()
        state = _build_elastic_state(*args, **kwargs)
        dt = time.perf_counter() - t0
        mech_timing["state_calls_total"] += 1
        mech_timing["state_time_total_s"] += dt
        if phase == "buckling":
            mech_timing["state_calls_buckling"] += 1
            mech_timing["state_time_buckling_s"] += dt
        else:
            mech_timing["state_calls_base"] += 1
            mech_timing["state_time_base_s"] += dt
        return state

    def _finalize_mech_timing(state: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if state is None:
            return None
        mech_timing["time_total_s"] = time.perf_counter() - t_mech_start
        return {
            **state,
            "optimization_timings": dict(mech_timing),
        }

    def _print_step_opt2(step: int, label: str, dcr_max: float, disp_ratio: float, *, changed: int = 0, uplift: int = 0, feasible: Optional[bool] = None, analysis_failed: bool = False, lb_lambda_1: Optional[float] = None):
        if verbosity < 1:
            return
        msg = (
            f"# SUMMARY | OPT_2 | step={step:02d} | "
            f"sections={label} | "
            f"changed={int(changed):3d} | "
            f"DCR_max={float(dcr_max):.4f} | "
            f"d/dlim_max={float(disp_ratio):.4f}"
        )
        if uplift > 0:
            msg += f" | uplift={int(uplift):3d}"
        if analysis_failed:
            msg += " | analysis_failed=True"
        if feasible is not None:
            msg += f" | feasible={bool(feasible)}"
        if lb_lambda_1 is not None and math.isfinite(float(lb_lambda_1)):
            msg += f" | lambda_1={float(lb_lambda_1):.4f}"
        print(msg)

    def _print_final_opt1(state: Optional[Mapping[str, Any]], feasible: bool):
        if verbosity < 1:
            return
        if state:
            sec_name = _section_name(state.get("selected_section", state.get("props_global", {})))
            dcr_max = float(state.get("ec3_dcr_max", float("inf")))
            disp_ratio = float(state.get("disp_ratio", float("inf")))
            lb_lambda_1 = state.get("lb_lambda_1", None)
        else:
            sec_name = "-"
            dcr_max = float("inf")
            disp_ratio = float("inf")
            lb_lambda_1 = None
        msg = (
            f"# SUMMARY | OPT_1 FINAL | "
            f"sec={sec_name} | "
            f"feasible={bool(feasible)} | "
            f"steps={len(history)} | "
            f"DCR_max={dcr_max:.4f} | "
            f"d/dlim_max={disp_ratio:.4f}"
        )
        if lb_lambda_1 is not None and math.isfinite(float(lb_lambda_1)):
            msg += f" | lambda_1={float(lb_lambda_1):.4f}"
        print(msg)

    def _print_final_opt2(state: Optional[Mapping[str, Any]], feasible: bool, member_to_idx: Optional[Mapping[MemberKey, int]] = None):
        if verbosity < 1:
            return
        dcr_max = float(state.get("ec3_dcr_max", float("inf"))) if state else float("inf")
        disp_ratio = float(state.get("disp_ratio", float("inf"))) if state else float("inf")
        lb_lambda_1 = state.get("lb_lambda_1", None) if state else None
        if state and isinstance(state.get("group_to_idx", None), Mapping):
            label = _sections_label_opt2_groups(state["group_to_idx"], candidate_sections)
        else:
            label = _sections_label_opt2(member_to_idx, candidate_sections) if member_to_idx else "-"
        msg = (
            f"# SUMMARY | OPT_2 FINAL | "
            f"sections={label} | "
            f"feasible={bool(feasible)} | "
            f"steps={len(history)} | "
            f"DCR_max={dcr_max:.4f} | "
            f"d/dlim_max={disp_ratio:.4f}"
        )
        if lb_lambda_1 is not None and math.isfinite(float(lb_lambda_1)):
            msg += f" | lambda_1={float(lb_lambda_1):.4f}"
        print(msg)

    # ------------------------------------------------------------------
    # mecc_opt = 1 -> one section for all members (fast first feasible)
    # ------------------------------------------------------------------
    if int(mecc_opt) == 1:
        best_feasible: Optional[Dict[str, Any]] = None
        best_util = -float("inf")

        for idx, sec in enumerate(candidate_sections[start_idx:], start=start_idx + 1):
            section_by_member = _make_section_by_member(M, sec)
            state = _build_state_timed(
                G, M,
                section_by_member=section_by_member,
                material=material_d,
                ndm=ndm,
                ndf=ndf,
                c1=c1, c2=c2, c3=c3, c4=c4, c5=c5,
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
                buckling_mode_index=buckling_mode_index,
                merchant_rankine_enabled=bool(check_merchant_rankine),
                merchant_rankine_min=merchant_rankine_min,
                merchant_rankine_mode_index=merchant_rankine_mode_index,
                internal_quiet=(verbosity <= 1),
            )
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
                state = _build_state_timed(
                    G, M,
                    section_by_member=section_by_member,
                    material=material_d,
                    ndm=ndm,
                    ndf=ndf,
                    c1=c1, c2=c2, c3=c3, c4=c4, c5=c5,
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
                    internal_quiet=(verbosity <= 1),
                    phase="buckling",
                )
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

            util = _objective_util(dcr_max, disp_ratio, dcr_limit)

            _record_state("global_candidate", state, {
                "candidate_index": idx - 1,
                "section_name": _section_name(sec),
                "feasible": feasible,
                "util": util,
            })
            _print_step_opt1(idx, _section_name(sec), dcr_max, disp_ratio, feasible, state.get("lb_lambda_1", None))

            penalty = _penalty_score(dcr_max, disp_ratio, dcr_limit)
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
                if util > best_util:
                    best_util = util
                    best_feasible = {
                        **state,
                        "props_global": dict(sec),
                        "section_by_member": section_by_member,
                        "section_name_by_member": {mk: _section_name(sec) for mk in members},
                        "selected_section": dict(sec),
                        "mecc_opt": 1,
                        "history": history,
                    }
                best_feasible = _finalize_mech_timing(best_feasible)
                _print_final_opt1(best_feasible, True)
                return best_feasible

        best_state = _finalize_mech_timing(best_state)
        _print_final_opt1(best_state, False)
        return best_state or _finalize_mech_timing({
            "analysis_ok": False,
            "props_global": None,
            "section_by_member": None,
            "section_name_by_member": None,
            "history": history,
            "mecc_opt": 1,
        })

    # ------------------------------------------------------------------
    # mecc_opt = 2 -> different section per member
    # Phase A: feasibility search
    #   1) if displacement governs -> targeted stiffness upgrades
    #   2) else if resistance governs -> local EC3 resizing
    # Phase B: local re-lightening refinement starting from first feasible
    # ------------------------------------------------------------------
    group_to_members = _build_similarity_groups(G, M, members)
    group_to_idx = {gname: int(start_idx) for gname in group_to_members}
    groups_refined_by_demand = False
    last_state: Optional[Dict[str, Any]] = None
    feasible_state: Optional[Dict[str, Any]] = None
    feasible_indices: Optional[Dict[str, int]] = None

    for it in range(1, int(max_iter) + 1):
        remaining_iters = int(max_iter) - int(it) + 1
        member_to_idx = _group_indices_to_member_indices(members, group_to_members, group_to_idx)
        current_map = _group_section_mapping(members, group_to_members, group_to_idx, candidate_sections)
        state = _build_state_timed(
            G, M,
            section_by_member=current_map,
            material=material_d,
            ndm=ndm,
            ndf=ndf,
            c1=c1, c2=c2, c3=c3, c4=c4, c5=c5,
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
            buckling_mode_index=buckling_mode_index,
            merchant_rankine_enabled=bool(check_merchant_rankine),
            merchant_rankine_min=merchant_rankine_min,
            merchant_rankine_mode_index=merchant_rankine_mode_index,
            internal_quiet=(verbosity <= 1),
        )
        last_state = state

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

        if state.get("analysis_ok", False) and not groups_refined_by_demand:
            envelopes = _member_envelopes(state["res"]["ele_forces"], state["ele_data"], compression_sign)
            lengths = _build_member_lengths(G, state["ele_data"])
            local_dcrs = _member_local_dcrs(state.get("ec3_results", {}), members)
            refined_groups = _refine_groups_by_demand(
                group_to_members=group_to_members,
                envelopes=envelopes,
                lengths=lengths,
                local_dcrs=local_dcrs,
                compression_sign=compression_sign,
            )
            if sorted(refined_groups.keys()) != sorted(group_to_members.keys()):
                new_group_to_idx: Dict[str, int] = {}
                for new_gname, new_members in refined_groups.items():
                    inherited = start_idx
                    for old_gname, old_members in group_to_members.items():
                        if any(mk in old_members for mk in new_members):
                            inherited = int(group_to_idx.get(old_gname, start_idx))
                            break
                    new_group_to_idx[new_gname] = inherited
                group_to_members = refined_groups
                group_to_idx = new_group_to_idx
                groups_refined_by_demand = True
                _record_state("group_split_by_demand", state, {
                    "iter": it,
                    "n_groups": len(group_to_members),
                })
                if verbosity >= 1:
                    print(f"# SUMMARY | OPT_2 | demand-based group split -> groups={len(group_to_members)}")
                continue
            groups_refined_by_demand = True

        penalty = _penalty_score(dcr_max, disp_ratio, dcr_limit)
        if penalty < best_penalty:
            best_penalty = penalty
            best_state = {
                **state,
                "props_global": dict(candidate_sections[next(iter(group_to_idx.values()))]) if len(set(int(v) for v in group_to_idx.values())) <= 1 else None,
                "section_by_member": current_map,
                "section_name_by_member": {mk: _section_name(current_map[mk]) for mk in members},
                "mecc_opt": 2,
                "history": history,
                "group_to_members": group_to_members,
                "group_to_idx": dict(group_to_idx),
            }

        if feasible:
            if bool(check_buckling) or bool(check_merchant_rankine):
                buckling_state = _build_state_timed(
                    G, M,
                    section_by_member=current_map,
                    material=material_d,
                    ndm=ndm,
                    ndf=ndf,
                    c1=c1, c2=c2, c3=c3, c4=c4, c5=c5,
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
                    internal_quiet=(verbosity <= 1),
                    phase="buckling",
                )
                buckling_feasible = _is_feasible_state(
                    buckling_state,
                    dcr_limit,
                    check_disp,
                    check_buckling=bool(check_buckling),
                    buckling_lambda_min=buckling_lambda_min,
                    buckling_mode_index=buckling_mode_index,
                    check_merchant_rankine=bool(check_merchant_rankine),
                    merchant_rankine_min=merchant_rankine_min,
                )
                _record_state("buckling_check", buckling_state, {
                    "iter": it,
                    "feasible": buckling_feasible,
                    "changed": 0,
                })
                _print_step_opt2(
                    len(history),
                    _sections_label_opt2_groups(group_to_idx, candidate_sections),
                    float(buckling_state.get("ec3_dcr_max", float("inf"))),
                    float(buckling_state.get("disp_ratio", float("inf"))),
                    changed=0,
                    feasible=buckling_feasible,
                    lb_lambda_1=buckling_state.get("lb_lambda_1", None),
                )

                if not buckling_feasible:
                    buckling_indices = dict(group_to_idx)
                    buckling_state_current = buckling_state
                    buckling_progress = False
                    lambda_prev = buckling_state.get("lb_lambda_1", None)
                    stagnation_count = 0
                    force_global_next = False

                    for bretry in range(1, max(0, int(buckling_max_retries)) + 1):
                        envelopes = _member_envelopes(buckling_state_current["res"]["ele_forces"], buckling_state_current["ele_data"], compression_sign)
                        lengths = _build_member_lengths(G, buckling_state_current["ele_data"])
                        plan_kind = "global" if force_global_next else "targeted"
                        if plan_kind == "targeted":
                            upgraded_idx, changed, score_map = _buckling_group_upgrade_plan(
                                group_to_idx=buckling_indices,
                                group_to_members=group_to_members,
                                candidate_sections=candidate_sections,
                                envelopes=envelopes,
                                lengths=lengths,
                                compression_sign=compression_sign,
                                current_lambda_1=buckling_state_current.get("lb_lambda_1", None),
                                target_lambda_1=buckling_lambda_min,
                                max_groups=buckling_group_limit,
                            )
                        else:
                            upgraded_idx, changed, score_map = dict(buckling_indices), 0, {}

                        if changed <= 0:
                            plan_kind = "global"
                            upgraded_idx, changed, score_map = _buckling_global_upgrade_plan(
                                group_to_idx=buckling_indices,
                                group_to_members=group_to_members,
                                candidate_sections=candidate_sections,
                                envelopes=envelopes,
                                lengths=lengths,
                                compression_sign=compression_sign,
                                current_lambda_1=buckling_state_current.get("lb_lambda_1", None),
                                target_lambda_1=buckling_lambda_min,
                                max_groups=buckling_group_limit,
                            )
                            if changed <= 0:
                                break
                        force_global_next = False

                        buckling_progress = True
                        buckling_indices = upgraded_idx
                        trial_map = _group_section_mapping(members, group_to_members, buckling_indices, candidate_sections)
                        buckling_state_current = _build_state_timed(
                            G, M,
                            section_by_member=trial_map,
                            material=material_d,
                            ndm=ndm,
                            ndf=ndf,
                            c1=c1, c2=c2, c3=c3, c4=c4, c5=c5,
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
                            internal_quiet=(verbosity <= 1),
                            phase="buckling",
                        )
                        buckling_feasible = _is_feasible_state(
                            buckling_state_current,
                            dcr_limit,
                            check_disp,
                            check_buckling=bool(check_buckling),
                            buckling_lambda_min=buckling_lambda_min,
                            buckling_mode_index=buckling_mode_index,
                            check_merchant_rankine=bool(check_merchant_rankine),
                            merchant_rankine_min=merchant_rankine_min,
                        )
                        lambda_curr = buckling_state_current.get("lb_lambda_1", None)
                        lambda_prev_f = None
                        lambda_curr_f = None
                        try:
                            if lambda_prev is not None:
                                lambda_prev_f = float(lambda_prev)
                        except Exception:
                            lambda_prev_f = None
                        try:
                            if lambda_curr is not None:
                                lambda_curr_f = float(lambda_curr)
                        except Exception:
                            lambda_curr_f = None

                        improved = False
                        if lambda_prev_f is not None and lambda_curr_f is not None and math.isfinite(lambda_prev_f) and math.isfinite(lambda_curr_f):
                            improved = lambda_curr_f > (lambda_prev_f + max(1e-4, 0.02 * max(lambda_prev_f, 1e-6)))
                        stagnation_count = 0 if improved else (stagnation_count + 1)
                        lambda_prev = lambda_curr
                        force_global_next = stagnation_count >= 2

                        _record_state("buckling_retry", buckling_state_current, {
                            "iter": it,
                            "buckling_retry": bretry,
                            "buckling_plan": plan_kind,
                            "changed": changed,
                            "top_buckling_score": max(score_map.values()) if score_map else 0.0,
                            "buckling_stagnation": stagnation_count,
                            "feasible": buckling_feasible,
                        })
                        _print_step_opt2(
                            len(history),
                            _sections_label_opt2_groups(buckling_indices, candidate_sections),
                            float(buckling_state_current.get("ec3_dcr_max", float("inf"))),
                            float(buckling_state_current.get("disp_ratio", float("inf"))),
                            changed=changed,
                            feasible=buckling_feasible,
                            lb_lambda_1=buckling_state_current.get("lb_lambda_1", None),
                        )
                        if buckling_feasible:
                            group_to_idx = dict(buckling_indices)
                            current_map = _group_section_mapping(members, group_to_members, group_to_idx, candidate_sections)
                            state = buckling_state_current
                            feasible = True
                            break

                    if not buckling_feasible:
                        if buckling_progress:
                            group_to_idx = dict(buckling_indices)
                            last_state = buckling_state_current
                            continue
                        break

            feasible_state = {
                **state,
                "props_global": dict(candidate_sections[next(iter(group_to_idx.values()))]) if len(set(int(v) for v in group_to_idx.values())) <= 1 else None,
                "section_by_member": current_map,
                "section_name_by_member": {mk: _section_name(current_map[mk]) for mk in members},
                "mecc_opt": 2,
                "history": history,
                "group_to_members": group_to_members,
                "group_to_idx": dict(group_to_idx),
            }
            feasible_indices = dict(group_to_idx)
            _record_state("feasible_found", state, {"iter": it, "feasible": True, "changed": 0})
            _print_step_opt2(
                it,
                _sections_label_opt2_groups(group_to_idx, candidate_sections),
                float(state.get("ec3_dcr_max", float("inf"))),
                float(state.get("disp_ratio", float("inf"))),
                changed=0,
                feasible=True,
                lb_lambda_1=state.get("lb_lambda_1", None),
            )
            break

        if not state.get("analysis_ok", False):
            jump = _disp_uplift_jump(disp_ratio)
            changed = 0
            for gname in group_to_idx:
                new_idx = min(len(candidate_sections) - 1, int(group_to_idx[gname]) + int(jump))
                if new_idx > int(group_to_idx[gname]):
                    group_to_idx[gname] = int(new_idx)
                    changed += 1
            _record_state("analysis_failed_uplift", state, {"iter": it, "changed": changed, "jump": int(jump)})
            _print_step_opt2(it, _sections_label_opt2_groups(group_to_idx, candidate_sections), dcr_max, disp_ratio, changed=changed, analysis_failed=True, feasible=False)
            if changed == 0:
                break
            continue

        envelopes = _member_envelopes(state["res"]["ele_forces"], state["ele_data"], compression_sign)
        lengths = _build_member_lengths(G, state["ele_data"])
        local_dcrs = _member_local_dcrs(state.get("ec3_results", {}), members)
        group_local_dcrs = {
            gname: max(float(local_dcrs.get(mk, 0.0)) for mk in gmembers)
            for gname, gmembers in group_to_members.items()
        }

        if dcr_max > float(dcr_limit):
            terminal_force_phase = remaining_iters <= max(2, int(math.ceil(0.20 * int(max_iter))))
            if terminal_force_phase:
                forced_idx, forced_changed, forced_local_dcr_max = _terminal_force_group_feasibility_plan(
                    group_to_idx=group_to_idx,
                    group_to_members=group_to_members,
                    envelopes=envelopes,
                    lengths=lengths,
                    candidates=candidate_sections,
                    material=material_d,
                    gamma_M0=gamma_M0,
                    gamma_M1=gamma_M1,
                    gamma_M2=gamma_M2,
                    beta=beta,
                    alpha_LT=alpha_LT,
                    dcr_limit=dcr_limit,
                    compression_sign=compression_sign,
                )
                if forced_changed > 0:
                    group_to_idx = forced_idx
                    _record_state("terminal_force_feasibility", state, {
                        "iter": it,
                        "changed": forced_changed,
                        "ec3_dcr_max": dcr_max,
                        "local_dcr_max": forced_local_dcr_max,
                    })
                    _print_step_opt2(
                        it,
                        _sections_label_opt2_groups(group_to_idx, candidate_sections),
                        dcr_max,
                        disp_ratio,
                        changed=forced_changed,
                        feasible=False,
                    )
                    continue

            boosted_idx, boosted_changed = _resistance_upgrade_plan(
                group_to_idx=group_to_idx,
                group_local_dcrs=group_local_dcrs,
                candidate_sections=candidate_sections,
                dcr_max=dcr_max,
                dcr_limit=dcr_limit,
                remaining_iters=remaining_iters,
                max_iter=int(max_iter),
            )
            if boosted_changed > 0:
                group_to_idx = boosted_idx
                _record_state("global_dcr_uplift", state, {
                    "iter": it,
                    "changed": boosted_changed,
                    "dcr_max": dcr_max,
                    "remaining_iters": remaining_iters,
                })
                _print_step_opt2(
                    it,
                    _sections_label_opt2_groups(group_to_idx, candidate_sections),
                    dcr_max,
                    disp_ratio,
                    changed=boosted_changed,
                    feasible=False,
                )
                continue

        if check_disp and disp_ratio > 1.0:
            uplifted_idx, changed, score_map = _displacement_upgrade_plan(
                members=list(group_to_idx.keys()),
                member_to_idx=group_to_idx,
                current_map={gname: candidate_sections[int(group_to_idx[gname])] for gname in group_to_idx},
                candidate_sections=candidate_sections,
                envelopes={
                    gname: {"envelope": {
                        "N_abs_max": max(float(_actions_from_record(envelopes[mk], compression_sign=compression_sign).get("N_abs", 0.0)) for mk in gmembers if mk in envelopes),
                        "My_abs_max": max(float(_actions_from_record(envelopes[mk], compression_sign=compression_sign).get("My", 0.0)) for mk in gmembers if mk in envelopes),
                        "Mz_abs_max": max(float(_actions_from_record(envelopes[mk], compression_sign=compression_sign).get("Mz", 0.0)) for mk in gmembers if mk in envelopes),
                        "T_abs_max": max(float(_actions_from_record(envelopes[mk], compression_sign=compression_sign).get("T", 0.0)) for mk in gmembers if mk in envelopes),
                    }}
                    for gname, gmembers in group_to_members.items()
                    if any(mk in envelopes for mk in gmembers)
                },
                lengths={gname: sum(float(lengths.get(mk, 0.0)) for mk in gmembers) for gname, gmembers in group_to_members.items()},
                material=material_d,
                compression_sign=compression_sign,
                disp_ratio=disp_ratio,
                local_dcrs={
                    gname: max(float(local_dcrs.get(mk, 0.0)) for mk in gmembers)
                    for gname, gmembers in group_to_members.items()
                },
            )

            if changed > 0:
                top_score = max(score_map.values()) if score_map else 0.0
                group_to_idx = uplifted_idx
                _record_state("targeted_disp_uplift", state, {
                    "iter": it,
                    "changed": changed,
                    "disp_ratio": disp_ratio,
                    "top_disp_score": top_score,
                })
                _print_step_opt2(
                    it,
                    _sections_label_opt2_groups(group_to_idx, candidate_sections),
                    dcr_max,
                    disp_ratio,
                    changed=changed,
                    feasible=False,
                )
                continue

            jump = _disp_uplift_jump(disp_ratio)
            changed = 0
            uplifted_idx = dict(group_to_idx)
            for gname in group_to_idx:
                new_idx = min(len(candidate_sections) - 1, int(group_to_idx[gname]) + int(jump))
                if new_idx > int(group_to_idx[gname]):
                    uplifted_idx[gname] = int(new_idx)
                    changed += 1
            group_to_idx = uplifted_idx
            _record_state("global_disp_uplift_fallback", state, {"iter": it, "changed": changed, "disp_ratio": disp_ratio, "jump": int(jump)})
            _print_step_opt2(it, _sections_label_opt2_groups(group_to_idx, candidate_sections), dcr_max, disp_ratio, changed=0, uplift=changed, feasible=False)
            if changed == 0:
                break
            continue

        local_changed = 0
        local_dcr_max = 0.0
        upgraded_idx = dict(group_to_idx)
        for gname, gmembers in group_to_members.items():
            idx_new, info = _pick_first_feasible_section_for_group(
                group_members=gmembers,
                envelopes=envelopes,
                lengths=lengths,
                candidates=candidate_sections,
                material=material_d,
                gamma_M0=gamma_M0,
                gamma_M1=gamma_M1,
                gamma_M2=gamma_M2,
                beta=beta,
                alpha_LT=alpha_LT,
                dcr_limit=dcr_limit,
                current_index=int(group_to_idx[gname]),
                compression_sign=compression_sign,
            )
            local_dcr_max = max(local_dcr_max, float(info.get("group_dcr_max", info.get("dcr", float("inf")))))
            if int(idx_new) > int(group_to_idx[gname]):
                upgraded_idx[gname] = int(idx_new)
                local_changed += 1

        _record_state("local_resize", state, {
            "iter": it,
            "local_changed": local_changed,
            "ec3_dcr_max": dcr_max,
            "local_dcr_max": local_dcr_max,
            "disp_ratio": disp_ratio,
        })
        _print_step_opt2(
            it,
            _sections_label_opt2_groups(upgraded_idx if local_changed > 0 else group_to_idx, candidate_sections),
            dcr_max,
            disp_ratio,
            changed=local_changed,
            feasible=False,
        )

        if local_changed > 0:
            group_to_idx = upgraded_idx
            continue

        if dcr_max > float(dcr_limit):
            fallback_jump = _dcr_uplift_jump(
                dcr_max,
                dcr_limit,
                remaining_iters=remaining_iters,
                max_iter=int(max_iter),
            )
            fallback_idx = dict(group_to_idx)
            fallback_changed = 0
            for gname in group_to_idx:
                new_idx = min(len(candidate_sections) - 1, int(group_to_idx[gname]) + int(fallback_jump))
                if new_idx > int(group_to_idx[gname]):
                    fallback_idx[gname] = int(new_idx)
                    fallback_changed += 1
            if fallback_changed > 0:
                group_to_idx = fallback_idx
                _record_state("global_force_uplift", state, {
                    "iter": it,
                    "changed": fallback_changed,
                    "jump": int(fallback_jump),
                    "ec3_dcr_max": dcr_max,
                })
                _print_step_opt2(
                    it,
                    _sections_label_opt2_groups(group_to_idx, candidate_sections),
                    dcr_max,
                    disp_ratio,
                    changed=fallback_changed,
                    feasible=False,
                )
                continue

        break

    if feasible_state is not None and feasible_indices is not None:
        feasible_state = {
            **feasible_state,
            "group_to_members": group_to_members,
            "group_to_idx": dict(feasible_indices),
            "history": history,
        }
        _write_opt2_group_report(
            filepath=group_report_path,
            group_to_members=group_to_members,
            group_to_idx=feasible_indices,
            candidate_sections=candidate_sections,
            lengths_by_key=feasible_state.get("lengths_by_key", None),
            ec3_results=feasible_state.get("ec3_results", None),
            feasible=True,
            dcr_max=float(feasible_state.get("ec3_dcr_max", float("inf"))),
            disp_ratio=float(feasible_state.get("disp_ratio", float("inf"))),
            history=history,
        )
        feasible_state = _finalize_mech_timing(feasible_state)
        _print_final_opt2(feasible_state, True, _group_indices_to_member_indices(members, group_to_members, feasible_indices))
        return feasible_state


    if best_state is not None:
        best_dcr = float(best_state.get("ec3_dcr_max", float("inf")))
        best_disp = float(best_state.get("disp_ratio", float("inf")))
        best_feasible = _is_feasible_state(
            best_state,
            dcr_limit,
            check_disp,
            check_buckling=check_buckling,
            buckling_lambda_min=buckling_lambda_min,
            buckling_mode_index=buckling_mode_index,
            check_merchant_rankine=bool(check_merchant_rankine),
            merchant_rankine_min=merchant_rankine_min,
        )
        best_state = {
            **best_state,
            "group_to_members": group_to_members,
            "group_to_idx": dict(group_to_idx),
            "history": history,
        }
        _write_opt2_group_report(
            filepath=group_report_path,
            group_to_members=group_to_members,
            group_to_idx=group_to_idx,
            candidate_sections=candidate_sections,
            lengths_by_key=best_state.get("lengths_by_key", None),
            ec3_results=best_state.get("ec3_results", None),
            feasible=best_feasible,
            dcr_max=best_dcr,
            disp_ratio=best_disp,
            history=history,
        )
        best_state = _finalize_mech_timing(best_state)
        _print_final_opt2(best_state, best_feasible, _group_indices_to_member_indices(members, group_to_members, group_to_idx))
        return best_state

    final_map = _group_section_mapping(members, group_to_members, group_to_idx, candidate_sections)
    fallback = {
        **(last_state or {}),
        "props_global": dict(candidate_sections[next(iter(group_to_idx.values()))]) if len(set(int(v) for v in group_to_idx.values())) <= 1 else None,
        "section_by_member": final_map,
        "section_name_by_member": {mk: _section_name(final_map[mk]) for mk in members},
        "mecc_opt": 2,
        "history": history,
        "group_to_members": group_to_members,
        "group_to_idx": dict(group_to_idx),
    }
    fallback_dcr = float(fallback.get("ec3_dcr_max", float("inf")))
    fallback_disp = float(fallback.get("disp_ratio", float("inf")))
    fallback_feasible = _is_feasible_state(
        fallback,
        dcr_limit,
        check_disp,
        check_buckling=check_buckling,
        buckling_lambda_min=buckling_lambda_min,
        buckling_mode_index=buckling_mode_index,
        check_merchant_rankine=bool(check_merchant_rankine),
        merchant_rankine_min=merchant_rankine_min,
    )
    _write_opt2_group_report(
        filepath=group_report_path,
        group_to_members=group_to_members,
        group_to_idx=group_to_idx,
        candidate_sections=candidate_sections,
        lengths_by_key=fallback.get("lengths_by_key", None),
        ec3_results=fallback.get("ec3_results", None),
        feasible=fallback_feasible,
        dcr_max=fallback_dcr,
        disp_ratio=fallback_disp,
        history=history,
    )
    fallback = _finalize_mech_timing(fallback)
    _print_final_opt2(fallback, fallback_feasible, _group_indices_to_member_indices(members, group_to_members, group_to_idx))
    return fallback
