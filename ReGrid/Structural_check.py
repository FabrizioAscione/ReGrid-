from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Tuple


Number = float | int


def _as_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default
    try:
        return float(value)
    except Exception:
        return default


def _sec_get(sec: Mapping[str, Any], *names: str, default: Optional[float] = None) -> Optional[float]:
    for name in names:
        if name in sec and sec[name] is not None:
            return _as_float(sec[name])
    return default


def _txt_get(sec: Mapping[str, Any], *names: str, default: Optional[str] = None) -> Optional[str]:
    for name in names:
        if name in sec and sec[name] is not None:
            return str(sec[name]).strip()
    return default


def _safe_div(num: float, den: Optional[float]) -> float:
    if den is None or abs(den) <= 1e-18:
        return math.nan
    return float(num) / float(den)


def _member_actions(d: Mapping[str, Any], compression_sign: str = "negative") -> Dict[str, float]:
    di = d.get("i", {})
    dj = d.get("j", {})
    Ni = float(di.get("N", 0.0))
    Nj = float(dj.get("N", 0.0))
    Vy_i = float(di.get("Vy", 0.0))
    Vy_j = float(dj.get("Vy", 0.0))
    Vz_i = float(di.get("Vz", 0.0))
    Vz_j = float(dj.get("Vz", 0.0))
    T_i = float(di.get("T", 0.0))
    T_j = float(dj.get("T", 0.0))
    My_i = float(di.get("My", 0.0))
    My_j = float(dj.get("My", 0.0))
    Mz_i = float(di.get("Mz", 0.0))
    Mz_j = float(dj.get("Mz", 0.0))

    N_abs = max(abs(Ni), abs(Nj))
    sign = str(compression_sign).strip().lower()
    if sign == "negative":
        N_comp = max(0.0, -Ni, -Nj)
    elif sign == "positive":
        N_comp = max(0.0, Ni, Nj)
    else:
        raise ValueError("compression_sign must be 'negative' or 'positive'.")

    Vy_abs = max(abs(Vy_i), abs(Vy_j))
    Vz_abs = max(abs(Vz_i), abs(Vz_j))
    V_res = math.hypot(Vy_abs, Vz_abs)
    T_abs = max(abs(T_i), abs(T_j))
    My_abs = max(abs(My_i), abs(My_j))
    Mz_abs = max(abs(Mz_i), abs(Mz_j))
    M_res = math.hypot(My_abs, Mz_abs)

    return {
        "N_i": Ni,
        "N_j": Nj,
        "NEd_abs": N_abs,
        "NEd_comp_used": N_comp,
        "Vy_i": Vy_i,
        "Vy_j": Vy_j,
        "VyEd_abs": Vy_abs,
        "Vz_i": Vz_i,
        "Vz_j": Vz_j,
        "VzEd_abs": Vz_abs,
        "VEd_res_abs": V_res,
        "T_i": T_i,
        "T_j": T_j,
        "TEd_abs": T_abs,
        "My_i": My_i,
        "My_j": My_j,
        "MyEd_abs": My_abs,
        "Mz_i": Mz_i,
        "Mz_j": Mz_j,
        "MzEd_abs": Mz_abs,
        "MEd_res_abs": M_res,
    }


def _section_family(sec: Mapping[str, Any]) -> str:
    fam = _txt_get(sec, "section_family", "family", "type", default="")
    return fam.upper() if fam else ""


def _fabrication(sec: Mapping[str, Any]) -> str:
    fab = _txt_get(sec, "fabrication", default="")
    return fab.lower() if fab else ""


def _section_class(sec: Mapping[str, Any]) -> Optional[int]:
    v = sec.get("section_class")
    try:
        return int(round(float(v))) if v is not None else None
    except Exception:
        return None


def _compression_resistance(sec: Mapping[str, Any], fy: float, gamma_M0: float) -> Optional[float]:
    cls = _section_class(sec)
    if cls == 4:
        A = _sec_get(sec, "Aeff", "A")
    else:
        A = _sec_get(sec, "A")
    if A is None:
        return None
    return A * fy / gamma_M0


def _compression_characteristic_resistance(sec: Mapping[str, Any], fy: float) -> Optional[float]:
    cls = _section_class(sec)
    if cls == 4:
        A = _sec_get(sec, "Aeff", "A")
    else:
        A = _sec_get(sec, "A")
    if A is None:
        return None
    return A * fy


def _bending_resistance(sec: Mapping[str, Any], axis: str, fy: float, gamma_M0: float) -> Optional[float]:
    axis = axis.lower()
    cls = _section_class(sec)
    if cls in (1, 2):
        W = _sec_get(sec, f"Wpl_{axis}", f"Wpl{axis}")
        if W is None:
            W = _sec_get(sec, f"Wel_{axis}", f"W{axis}")
    elif cls == 3:
        W = _sec_get(sec, f"Wel_{axis}", f"W{axis}")
    elif cls == 4:
        W = _sec_get(sec, f"Weff_{axis}")
        if W is None:
            W = _sec_get(sec, f"Wel_{axis}", f"W{axis}")
    else:
        W = _sec_get(sec, f"Wpl_{axis}", f"Wel_{axis}", f"W{axis}")
    if W is None:
        return None
    return W * fy / gamma_M0


def _member_moment_characteristic_resistance(sec: Mapping[str, Any], axis: str, fy: float) -> Optional[float]:
    axis = axis.lower()
    cls = _section_class(sec)
    if cls in (1, 2):
        W = _sec_get(sec, f"Wpl_{axis}", f"Wpl{axis}")
        if W is None:
            W = _sec_get(sec, f"Wel_{axis}", f"W{axis}")
    elif cls == 3:
        W = _sec_get(sec, f"Wel_{axis}", f"W{axis}")
    elif cls == 4:
        W = _sec_get(sec, f"Weff_{axis}")
        if W is None:
            W = _sec_get(sec, f"Wel_{axis}", f"W{axis}")
    else:
        W = _sec_get(sec, f"Wpl_{axis}", f"Wel_{axis}", f"W{axis}")
    if W is None:
        return None
    return W * fy


def _shear_area(sec: Mapping[str, Any], axis: str) -> Optional[float]:
    axis = axis.lower()
    Av = _sec_get(sec, f"Av_{axis}")
    if Av is not None:
        return Av
    fam = _section_family(sec)
    A = _sec_get(sec, "A")
    if fam == "CHS" and A is not None:
        return 2.0 * A / math.pi
    return _sec_get(sec, "Av")


def _shear_resistance(sec: Mapping[str, Any], axis: str, fy: float, gamma_M0: float) -> Optional[float]:
    Av = _shear_area(sec, axis)
    if Av is None:
        return None
    return Av * fy / (math.sqrt(3.0) * gamma_M0)


def _default_alpha_for_axis(sec: Mapping[str, Any], axis: str) -> float:
    ax = axis.lower()
    given = _sec_get(sec, f"alpha_{ax}")
    if given is not None:
        return float(given)
    fam = _section_family(sec)
    fab = _fabrication(sec)
    if fam == "CHS":
        return 0.21 if fab != "cold_formed" else 0.49
    if fam in {"RHS", "SHS", "BOX"}:
        return 0.21 if fab in {"hot_finished", "rolled"} else 0.49
    return 0.34


def _buckling_reduction(alpha: float, lam_bar: float) -> float:
    lam_bar = max(0.0, float(lam_bar))
    phi = 0.5 * (1.0 + alpha * (lam_bar - 0.2) + lam_bar * lam_bar)
    root = phi * phi - lam_bar * lam_bar
    if root < 0.0:
        root = 0.0
    chi = 1.0 / (phi + math.sqrt(root)) if phi > 0.0 else 1.0
    return min(1.0, chi)


def _flexural_buckling(sec: Mapping[str, Any], axis: str, fy: float, E: float, gamma_M1: float, Lcr: Optional[float]) -> Tuple[Optional[float], Dict[str, float]]:
    axis = axis.lower()
    info: Dict[str, float] = {}
    A = _sec_get(sec, "A")
    I = _sec_get(sec, "Iy" if axis == "y" else "Iz")
    if A is None or I is None or Lcr is None or Lcr <= 0.0:
        return None, info
    Ncr = math.pi * math.pi * E * I / (Lcr * Lcr)
    if Ncr <= 0.0:
        return None, info
    lam_bar = math.sqrt(max(0.0, A * fy / Ncr))
    alpha = _default_alpha_for_axis(sec, axis)
    chi = _buckling_reduction(alpha, lam_bar)
    NbRd = chi * A * fy / gamma_M1
    info.update({
        f"Lcr_{axis}": float(Lcr),
        f"Ncr_{axis}": Ncr,
        f"lambda_bar_{axis}": lam_bar,
        f"alpha_{axis}": alpha,
        f"chi_{axis}": chi,
        f"NbRd_{axis}": NbRd,
    })
    return NbRd, info


def _mnrd_chs_eq632_approx(sec: Mapping[str, Any], fy: float, gamma_M0: float, NEd_comp: float, axis: str) -> Tuple[Optional[float], str]:
    Mpl = _bending_resistance(sec, axis, fy, gamma_M0)
    Npl = _compression_resistance(sec, fy, gamma_M0)
    if Mpl is None or Npl is None:
        return None, ""
    n = 0.0 if Npl <= 0.0 else max(0.0, min(1.0, NEd_comp / Npl))
    MNr = Mpl * max(0.0, 1.0 - n * n)
    return MNr, "CHS combined axial+bending uses Eq. (6.32) approximation requested by the user, closed with biaxial exponents alpha=beta=2."


def _section_checks_for_element(sec: Mapping[str, Any], acts: Mapping[str, float], fy: float, gamma_M0: float) -> Tuple[Dict[str, float], list[str], list[str]]:
    checks: Dict[str, float] = {}
    warnings: list[str] = []
    errors: list[str] = []
    fam = _section_family(sec)

    NplRd = _compression_resistance(sec, fy, gamma_M0)
    MyRd = _bending_resistance(sec, "y", fy, gamma_M0)
    MzRd = _bending_resistance(sec, "z", fy, gamma_M0)
    VyRd = _shear_resistance(sec, "y", fy, gamma_M0)
    VzRd = _shear_resistance(sec, "z", fy, gamma_M0)

    if NplRd is not None:
        checks["6.2.4_compression"] = _safe_div(acts["NEd_comp_used"], NplRd)
    if MyRd is not None:
        checks["6.2.5_bending_y"] = _safe_div(acts["MyEd_abs"], MyRd)
    if MzRd is not None:
        checks["6.2.5_bending_z"] = _safe_div(acts["MzEd_abs"], MzRd)
    if VyRd is not None:
        checks["6.2.6_shear_y"] = _safe_div(acts["VyEd_abs"], VyRd)
    if VzRd is not None:
        checks["6.2.6_shear_z"] = _safe_div(acts["VzEd_abs"], VzRd)

    cls = _section_class(sec)
    if cls in (1, 2):
        if fam == "CHS":
            MNyRd, warn = _mnrd_chs_eq632_approx(sec, fy, gamma_M0, acts["NEd_comp_used"], "y")
            MNzRd, _ = _mnrd_chs_eq632_approx(sec, fy, gamma_M0, acts["NEd_comp_used"], "z")
            if warn:
                warnings.append(warn)
            if MNyRd is not None:
                checks["6.2.9_bending_axial_y"] = _safe_div(acts["MyEd_abs"], MNyRd)
            if MNzRd is not None:
                checks["6.2.9_bending_axial_z"] = _safe_div(acts["MzEd_abs"], MNzRd)
            if MNyRd is not None and MNzRd is not None:
                denom_y = max(MNyRd, 1e-18)
                denom_z = max(MNzRd, 1e-18)
                checks["6.2.9_biaxial"] = (acts["MyEd_abs"] / denom_y) ** 2 + (acts["MzEd_abs"] / denom_z) ** 2
        else:
            if NplRd is not None and MyRd is not None:
                MNyRd = MyRd * max(0.0, 1.0 - min(1.0, acts["NEd_comp_used"] / NplRd) ** 2)
                checks["6.2.9_bending_axial_y"] = _safe_div(acts["MyEd_abs"], MNyRd)
            if NplRd is not None and MzRd is not None:
                MNzRd = MzRd * max(0.0, 1.0 - min(1.0, acts["NEd_comp_used"] / NplRd) ** 2)
                checks["6.2.9_bending_axial_z"] = _safe_div(acts["MzEd_abs"], MNzRd)
            if NplRd is not None and MyRd is not None and MzRd is not None:
                MNyRd = MyRd * max(0.0, 1.0 - min(1.0, acts["NEd_comp_used"] / NplRd) ** 2)
                MNzRd = MzRd * max(0.0, 1.0 - min(1.0, acts["NEd_comp_used"] / NplRd) ** 2)
                checks["6.2.9_biaxial"] = (acts["MyEd_abs"] / max(MNyRd, 1e-18)) + (acts["MzEd_abs"] / max(MNzRd, 1e-18))

    return checks, warnings, errors


def _member_delta_moments(sec: Mapping[str, Any], acts: Mapping[str, float]) -> Tuple[float, float]:
    cls = _section_class(sec)
    if cls != 4:
        return 0.0, 0.0
    eNy = _sec_get(sec, "eN_y", default=0.0) or 0.0
    eNz = _sec_get(sec, "eN_z", default=0.0) or 0.0
    N = acts["NEd_comp_used"]
    return N * eNy, N * eNz


def _member_chiLT(sec: Mapping[str, Any], info: Mapping[str, Any]) -> Tuple[float, list[str]]:
    warnings: list[str] = []
    fam = _section_family(sec)
    if fam in {"CHS", "SHS", "RHS", "BOX"}:
        return 1.0, warnings
    chiLT = _as_float(info.get("chiLT", info.get("chi_LT")))
    if chiLT is None:
        warnings.append("6.3.3: chiLT not provided; using chiLT=1.0. This is exact for members not susceptible to torsional deformation, e.g. CHS, but only an assumption for open sections.")
        chiLT = 1.0
    return float(chiLT), warnings


def _method2_kij_rhs_like(sec: Mapping[str, Any], acts: Mapping[str, float], fy: float, gamma_M1: float, info: Mapping[str, Any]) -> Tuple[Optional[Dict[str, float]], list[str]]:
    warnings: list[str] = []
    cls = _section_class(sec)
    if cls not in (1, 2, 3, 4):
        return None, warnings

    NRk = _compression_characteristic_resistance(sec, fy)
    chi_y = _as_float(info.get("chi_y"))
    chi_z = _as_float(info.get("chi_z"))
    lam_y = _as_float(info.get("lambda_bar_y"), 0.0) or 0.0
    lam_z = _as_float(info.get("lambda_bar_z"), 0.0) or 0.0
    if NRk is None or chi_y is None or chi_z is None:
        return None, warnings

    Cmy = _as_float(info.get("Cmy"), 1.0) or 1.0
    Cmz = _as_float(info.get("Cmz"), 1.0) or 1.0

    Ny_denom = chi_y * NRk / gamma_M1
    Nz_denom = chi_z * NRk / gamma_M1
    ny = 0.0 if Ny_denom <= 1e-18 else acts["NEd_comp_used"] / Ny_denom
    nz = 0.0 if Nz_denom <= 1e-18 else acts["NEd_comp_used"] / Nz_denom

    if cls in (1, 2):
        kyy = min(Cmy * (1.0 + max(0.0, lam_y - 0.2) * ny), Cmy * (1.0 + 0.8 * ny))
        kzz = min(Cmz * (1.0 + max(0.0, lam_z - 0.2) * nz), Cmz * (1.0 + 0.8 * nz))
    else:
        kyy = min(Cmy * (1.0 + lam_y * ny), Cmy * (1.0 + 0.6 * ny))
        kzz = min(Cmz * (1.0 + lam_z * nz), Cmz * (1.0 + 0.6 * nz))

    # RHS-like cross-coupling, also used here for CHS as requested in this workflow.
    kyz = 0.6 * kzz
    kzy = 0.6 * kyy

    fam = _section_family(sec)
    if fam == "CHS":
        warnings.append("6.3.3: kij for CHS are being evaluated with an RHS-like Method 2 approximation (kyz=0.6*kzz, kzy=0.6*kyy) because no CHS-specific kij set was provided.")

    return {"kyy": kyy, "kyz": kyz, "kzy": kzy, "kzz": kzz, "Cmy": Cmy, "Cmz": Cmz}, warnings


def _member_beam_column_633(sec: Mapping[str, Any], acts: Mapping[str, float], fy: float, gamma_M1: float, info: Mapping[str, Any]) -> Tuple[Dict[str, float], Dict[str, float], list[str], list[str]]:
    checks: Dict[str, float] = {}
    add_info: Dict[str, float] = {}
    warnings: list[str] = []
    errors: list[str] = []

    chi_y = _as_float(info.get("chi_y"))
    chi_z = _as_float(info.get("chi_z"))
    if chi_y is None or chi_z is None:
        return checks, add_info, warnings, errors

    NRk = _compression_characteristic_resistance(sec, fy)
    MyRk = _member_moment_characteristic_resistance(sec, "y", fy)
    MzRk = _member_moment_characteristic_resistance(sec, "z", fy)
    if NRk is None or MyRk is None or MzRk is None:
        return checks, add_info, warnings, errors

    dMy, dMz = _member_delta_moments(sec, acts)
    chiLT, wlt = _member_chiLT(sec, info)
    warnings.extend(wlt)

    # user-supplied kij first
    kyy = _as_float(info.get("kyy"))
    kyz = _as_float(info.get("kyz"))
    kzy = _as_float(info.get("kzy"))
    kzz = _as_float(info.get("kzz"))
    if None in (kyy, kyz, kzy, kzz):
        kij_auto, wk = _method2_kij_rhs_like(sec, acts, fy, gamma_M1, info)
        warnings.extend(wk)
        if kij_auto is not None:
            kyy = kij_auto["kyy"]
            kyz = kij_auto["kyz"]
            kzy = kij_auto["kzy"]
            kzz = kij_auto["kzz"]
            add_info.update(kij_auto)

    if None in (kyy, kyz, kzy, kzz):
        warnings.append("6.3.3 not evaluated because kij factors are not available. Provide kyy, kyz, kzy, kzz or use a supported automatic branch.")
        return checks, add_info, warnings, errors

    NbRd_y = chi_y * NRk / gamma_M1
    NbRd_z = chi_z * NRk / gamma_M1
    MybRd = chiLT * MyRk / gamma_M1
    MzbRd = MzRk / gamma_M1

    add_info.update({
        "DeltaMy_633": dMy,
        "DeltaMz_633": dMz,
        "chiLT_633": chiLT,
        "NRk_633": NRk,
        "MyRk_633": MyRk,
        "MzRk_633": MzRk,
        "NbRd_y_633": NbRd_y,
        "NbRd_z_633": NbRd_z,
        "MybRd_633": MybRd,
        "MzbRd_633": MzbRd,
        "kyy": float(kyy),
        "kyz": float(kyz),
        "kzy": float(kzy),
        "kzz": float(kzz),
    })

    termN_y = _safe_div(acts["NEd_comp_used"], NbRd_y)
    termN_z = _safe_div(acts["NEd_comp_used"], NbRd_z)
    termMy = _safe_div(acts["MyEd_abs"] + abs(dMy), MybRd)
    termMz = _safe_div(acts["MzEd_abs"] + abs(dMz), MzbRd)

    checks["6.3.3_eq_6.61"] = termN_y + float(kyy) * termMy + float(kyz) * termMz
    checks["6.3.3_eq_6.62"] = termN_z + float(kzy) * termMy + float(kzz) * termMz
    return checks, add_info, warnings, errors


def compute_DCR_elements(
    ele_forces: Mapping[int, Mapping[str, Any]],
    sections_by_element: Mapping[int, Mapping[str, Any]],
    material: Mapping[str, Any],
    member_data_by_element: Optional[Mapping[int, Mapping[str, Any]]] = None,
    gamma_M0: float = 1.0,
    gamma_M1: float = 1.0,
    gamma_M2: float = 1.25,
    compression_sign: str = "negative",
) -> Tuple[Dict[int, Dict[str, Any]], float]:
    fy = _as_float(material.get("fyk", material.get("fy")))
    E = _as_float(material.get("E"))
    if fy is None or E is None:
        raise ValueError("material must provide at least 'fyk' (or 'fy') and 'E'.")

    results: Dict[int, Dict[str, Any]] = {}
    dcr_max = 0.0
    member_data_by_element = member_data_by_element or {}

    for e, d in ele_forces.items():
        e = int(e)
        sec = sections_by_element[e]
        acts = _member_actions(d, compression_sign=compression_sign)
        info = dict(member_data_by_element.get(e, {}))
        checks: Dict[str, float] = {}
        warnings: list[str] = []
        errors: list[str] = []

        try:
            sec_checks, sec_warnings, sec_errors = _section_checks_for_element(sec, acts, fy, gamma_M0)
            checks.update(sec_checks)
            warnings.extend(sec_warnings)
            errors.extend(sec_errors)
        except Exception as exc:
            errors.append(f"section checks: {exc}")

        try:
            Lcr_y = _as_float(info.get("Lcr_y"))
            Lcr_z = _as_float(info.get("Lcr_z"))
            NbRd_y, infy = _flexural_buckling(sec, "y", fy, E, gamma_M1, Lcr_y)
            NbRd_z, infz = _flexural_buckling(sec, "z", fy, E, gamma_M1, Lcr_z)
            info.update(infy)
            info.update(infz)
            if NbRd_y is not None:
                checks["6.3.1_buckling_y"] = _safe_div(acts["NEd_comp_used"], NbRd_y)
            if NbRd_z is not None:
                checks["6.3.1_buckling_z"] = _safe_div(acts["NEd_comp_used"], NbRd_z)
        except Exception as exc:
            errors.append(f"buckling checks: {exc}")

        try:
            mchecks, minfo, mwarn, merr = _member_beam_column_633(sec, acts, fy, gamma_M1, info)
            checks.update(mchecks)
            info.update(minfo)
            warnings.extend(mwarn)
            errors.extend(merr)
        except Exception as exc:
            errors.append(f"6.3.3 beam-column checks: {exc}")

        valid_checks = {k: v for k, v in checks.items() if isinstance(v, (int, float)) and not math.isnan(v)}
        if valid_checks:
            gov = max(valid_checks, key=lambda k: valid_checks[k])
            dcr = float(valid_checks[gov])
        else:
            gov = None
            dcr = math.nan

        result = {
            "section": _txt_get(sec, "name", default=f"ele_{e}"),
            "dcr": dcr,
            "governing_check": gov,
            "checks": checks,
            "warnings": warnings,
            "errors": errors,
            "actions": acts,
            "member_data": {
                "L_member": _as_float(info.get("L_member")),
                "Lcr_y": _as_float(info.get("Lcr_y")),
                "Lcr_z": _as_float(info.get("Lcr_z")),
                "Mcr": _as_float(info.get("Mcr")),
                "lambda_bar_y": _as_float(info.get("lambda_bar_y")),
                "lambda_bar_z": _as_float(info.get("lambda_bar_z")),
                "chi_y": _as_float(info.get("chi_y")),
                "chi_z": _as_float(info.get("chi_z")),
                "NbRd_y": _as_float(info.get("NbRd_y")),
                "NbRd_z": _as_float(info.get("NbRd_z")),
                "DeltaMy_633": _as_float(info.get("DeltaMy_633")),
                "DeltaMz_633": _as_float(info.get("DeltaMz_633")),
                "chiLT_633": _as_float(info.get("chiLT_633")),
                "kyy": _as_float(info.get("kyy")),
                "kyz": _as_float(info.get("kyz")),
                "kzy": _as_float(info.get("kzy")),
                "kzz": _as_float(info.get("kzz")),
                "Cmy": _as_float(info.get("Cmy")),
                "Cmz": _as_float(info.get("Cmz")),
            },
        }
        results[e] = result
        if isinstance(dcr, (int, float)) and not math.isnan(dcr):
            dcr_max = max(dcr_max, float(dcr))

    return results, dcr_max
