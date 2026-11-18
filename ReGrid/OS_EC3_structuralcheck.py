import math
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import openseespy.opensees as ops
from transparent_plots import configure_transparent_matplotlib_saves

configure_transparent_matplotlib_saves()

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d.art3d import Line3DCollection


MemberKey = Tuple[int, int]
ScalarOrMap = Union[float, int, str, Mapping[Any, Any], None]


# -----------------------------------------------------------------------------
# basic helpers
# -----------------------------------------------------------------------------

def _as_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default
    try:
        return float(value)
    except Exception:
        return default


def _safe_div(num: float, den: Optional[float]) -> float:
    if den is None:
        return math.nan
    den = float(den)
    if abs(den) <= 1e-18:
        return math.nan
    return float(num) / den


def _radq(x: float) -> float:
    return math.sqrt(max(0.0, float(x)))


def _first_not_none(*values: Any) -> Any:
    for v in values:
        if v is not None:
            return v
    return None


def _member_key(a: int, b: int) -> MemberKey:
    a = int(a)
    b = int(b)
    return (a, b) if a < b else (b, a)


def _txt_get(sec: Mapping[str, Any], *names: str, default: Optional[str] = None) -> Optional[str]:
    for name in names:
        if name in sec and sec[name] is not None:
            return str(sec[name]).strip()
    return default


def _sec_get(sec: Mapping[str, Any], *names: str, default: Optional[float] = None) -> Optional[float]:
    for name in names:
        if name in sec and sec[name] is not None:
            return _as_float(sec[name])
    return default


def _section_name(sec: Mapping[str, Any]) -> str:
    return _txt_get(sec, "name", "section_name", "profile", default="") or ""


def _section_class(sec: Mapping[str, Any]) -> Optional[int]:
    for key in ("section_class", "class", "classe", "classe_sezione", "section class"):
        if key in sec and sec[key] is not None:
            try:
                return int(round(float(sec[key])))
            except Exception:
                return None
    return None


def _section_family_raw(sec: Mapping[str, Any]) -> str:
    fam = _txt_get(sec, "section_family", "family", "type", "section_type", default="") or ""
    if fam:
        return fam.upper().replace(" ", "")

    name = _section_name(sec).upper().replace(" ", "")
    if any(tok in name for tok in ("IPE", "HEA", "HEB", "HEM", "HE")):
        return "HE"
    if any(tok in name for tok in ("RHS", "SHS", "BOX")):
        return "RHS"
    if any(tok in name for tok in ("CHS", "Ø", "D=")):
        return "CHS"
    if name.startswith("2L") or "2L" in name:
        return "2L"
    if name.startswith("UPN") or name.startswith("C"):
        return "C"
    if name.startswith("L"):
        return "L"
    return "OTHER"


def _family_group(sec: Mapping[str, Any]) -> str:
    fam = _section_family_raw(sec)
    if fam.startswith("IPE"):
        return "IPE"
    if fam.startswith("HE"):
        return "HE"
    if fam in {"CHS"}:
        return "CHS"
    if fam in {"RHS", "SHS", "BOX"}:
        return "RHS"
    if fam in {"C", "UPN", "U"}:
        return "C"
    if fam in {"2L", "DOUBLEL"}:
        return "2L"
    if fam in {"L", "ANGLE"}:
        return "L"
    return "OTHER"


def _A(sec: Mapping[str, Any]) -> Optional[float]:
    return _sec_get(sec, "A")


def _Av(sec: Mapping[str, Any]) -> Optional[float]:
    Av = _sec_get(sec, "Av", "A_v", "Avz", "Avy")
    if Av is not None:
        return Av
    if _family_group(sec) == "CHS":
        return _A(sec)
    return None


def _Iy(sec: Mapping[str, Any]) -> Optional[float]:
    return _sec_get(sec, "Iy", "Iyy")


def _Iz(sec: Mapping[str, Any]) -> Optional[float]:
    return _sec_get(sec, "Iz", "Izz")


def _It(sec: Mapping[str, Any]) -> Optional[float]:
    return _sec_get(sec, "It", "J", "Iv")


def _Iw(sec: Mapping[str, Any]) -> Optional[float]:
    return _sec_get(sec, "Iw", "Cw")


def _b(sec: Mapping[str, Any]) -> Optional[float]:
    return _sec_get(sec, "b")


def _h(sec: Mapping[str, Any]) -> Optional[float]:
    return _sec_get(sec, "h")


def _tf(sec: Mapping[str, Any]) -> Optional[float]:
    return _sec_get(sec, "tf")


def _tw(sec: Mapping[str, Any]) -> Optional[float]:
    return _sec_get(sec, "tw")


def _Di(sec: Mapping[str, Any]) -> Optional[float]:
    return _sec_get(sec, "Di")


def _De(sec: Mapping[str, Any]) -> Optional[float]:
    return _sec_get(sec, "De")


def _t(sec: Mapping[str, Any]) -> Optional[float]:
    return _sec_get(sec, "t")


def _Wpl(sec: Mapping[str, Any], axis: str) -> Optional[float]:
    axis = axis.lower()
    if axis == "y":
        return _sec_get(sec, "Wpl_y", "Wply", "WplY")
    return _sec_get(sec, "Wpl_z", "Wplz", "WplZ")


def _Wel(sec: Mapping[str, Any], axis: str) -> Optional[float]:
    axis = axis.lower()
    if axis == "y":
        return _sec_get(sec, "Wel_y", "Wy", "Wely", "WelY")
    return _sec_get(sec, "Wel_z", "Wz", "Welz", "WelZ")


# -----------------------------------------------------------------------------
# geometry / mapping helpers
# -----------------------------------------------------------------------------

def _node_xyz(G: Dict[str, Any], nid: int):
    nid = int(nid)
    nodes = G.get("nodes", {})
    if nid in nodes:
        return nodes[nid]
    x, y, z = ops.nodeCoord(nid)
    return float(x), float(y), float(z)


def _distance(p1, p2) -> float:
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2 + (p2[2] - p1[2]) ** 2)


def _build_seg_to_member(ele_data: Dict[str, Any]) -> Dict[MemberKey, MemberKey]:
    chains = ele_data.get("member_nodechains", {}) or {}
    out: Dict[MemberKey, MemberKey] = {}
    for mkey, chain in chains.items():
        mk = _member_key(*mkey)
        ch = [int(n) for n in chain]
        for i in range(len(ch) - 1):
            out[_member_key(ch[i], ch[i + 1])] = mk
    return out


def _build_member_lengths(G: Dict[str, Any], ele_data: Dict[str, Any]) -> Dict[MemberKey, float]:
    chains = ele_data.get("member_nodechains", {}) or {}
    out: Dict[MemberKey, float] = {}
    for mkey, chain in chains.items():
        mk = _member_key(*mkey)
        ch = [int(n) for n in chain]
        L = 0.0
        for i in range(len(ch) - 1):
            p1 = _node_xyz(G, ch[i])
            p2 = _node_xyz(G, ch[i + 1])
            L += _distance(p1, p2)
        out[mk] = L
    return out


def _ele_to_member_map(ele_data: Dict[str, Any]) -> Dict[int, MemberKey]:
    seg_to_member = _build_seg_to_member(ele_data)
    out: Dict[int, MemberKey] = {}
    for e in ele_data.get("ele_tags", []):
        try:
            nds = ops.eleNodes(int(e))
        except Exception:
            nds = None
        if nds is None or len(nds) < 2:
            continue
        seg = _member_key(int(nds[0]), int(nds[1]))
        if seg in seg_to_member:
            out[int(e)] = seg_to_member[seg]
    return out


def build_sections_by_member(
    ele_data: Dict[str, Any],
    section: Optional[Dict[str, Any]] = None,
    section_by_member: Optional[Mapping[MemberKey, Dict[str, Any]]] = None,
) -> Dict[MemberKey, Dict[str, Any]]:
    chains = ele_data.get("member_nodechains", {}) or {}
    member_keys = [_member_key(*mkey) for mkey in chains.keys()]
    if section_by_member is None:
        if section is None:
            raise ValueError("Provide either section or section_by_member.")
        return {mk: section for mk in member_keys}

    out: Dict[MemberKey, Dict[str, Any]] = {}
    for mk in member_keys:
        if mk in section_by_member:
            out[mk] = section_by_member[mk]
        else:
            mk2 = _member_key(*mk)
            if mk2 not in section_by_member:
                raise KeyError(f"section_by_member has no entry for member {mk}.")
            out[mk] = section_by_member[mk2]
    return out


def build_sections_by_element(
    ele_data: Dict[str, Any],
    section: Optional[Dict[str, Any]] = None,
    section_by_member: Optional[Mapping[MemberKey, Dict[str, Any]]] = None,
) -> Dict[int, Dict[str, Any]]:
    ele_tags = [int(e) for e in ele_data.get("ele_tags", [])]
    if section_by_member is None:
        if section is None:
            raise ValueError("Provide either section or section_by_member.")
        return {e: section for e in ele_tags}

    e2m = _ele_to_member_map(ele_data)
    out: Dict[int, Dict[str, Any]] = {}
    for e in ele_tags:
        mk = e2m.get(e)
        if mk is None:
            raise KeyError(f"Cannot map element {e} to a parent member.")
        if mk not in section_by_member:
            raise KeyError(f"section_by_member has no entry for member {mk}.")
        out[e] = section_by_member[mk]
    return out


# -----------------------------------------------------------------------------
# actions
# -----------------------------------------------------------------------------

def _signed_MA_MB(m_i: float, m_j: float) -> Tuple[float, float]:
    """
    MA = end moment with larger absolute value (keeps sign)
    MB = the other end moment (keeps sign)
    """
    m_i = float(m_i)
    m_j = float(m_j)
    if abs(m_i) >= abs(m_j):
        return m_i, m_j
    return m_j, m_i


def _psi_ratio(MA: float, MB: float, default_if_zero: float = 0.0) -> float:
    MA = float(MA)
    MB = float(MB)
    if abs(MA) <= 1e-18:
        return float(default_if_zero)
    return MB / MA


def _actions_from_record(d: Mapping[str, Any], compression_sign: str = "negative") -> Dict[str, float]:
    """
    Simple actions for one item.
    If 'envelope' exists, use member envelope values.
    Otherwise use end-force maxima on the single element.
    """
    if isinstance(d, Mapping) and "envelope" in d:
        env = d.get("envelope", {}) or {}
        if str(compression_sign).strip().lower() == "negative":
            N_tension = float(env.get("N_tension_max", 0.0))
            N_compression = float(env.get("N_comp_max", 0.0))
        else:
            N_tension = float(env.get("N_comp_max", 0.0))
            N_compression = float(env.get("N_tension_max", 0.0))

        My_i = float(env.get("My_member_end_i", 0.0))
        My_j = float(env.get("My_member_end_j", 0.0))
        Mz_i = float(env.get("Mz_member_end_i", 0.0))
        Mz_j = float(env.get("Mz_member_end_j", 0.0))
        MAy, MBy = _signed_MA_MB(My_i, My_j)
        MAz, MBz = _signed_MA_MB(Mz_i, Mz_j)

        return {
            "N_tension": N_tension,
            "N_compression": N_compression,
            "N_abs": float(env.get("N_abs_max", 0.0)),
            "My": float(env.get("My_abs_max", 0.0)),
            "Mz": float(env.get("Mz_abs_max", 0.0)),
            "Vy": float(env.get("Vy_abs_max", 0.0)),
            "Vz": float(env.get("Vz_abs_max", 0.0)),
            "T": float(env.get("T_abs_max", 0.0)),
            "My_ED_Max": float(env.get("My_abs_max", 0.0)),
            "Mz_ED_Max": float(env.get("Mz_abs_max", 0.0)),
            "My_end_i": My_i,
            "My_end_j": My_j,
            "Mz_end_i": Mz_i,
            "Mz_end_j": Mz_j,
            "MA": MAy,
            "MB": MBy,
            "MAy": MAy,
            "MBy": MBy,
            "MAz": MAz,
            "MBz": MBz,
        }

    di = d.get("i", {})
    dj = d.get("j", {})
    Ni = float(di.get("N", 0.0))
    Nj = float(dj.get("N", 0.0))
    if str(compression_sign).strip().lower() == "negative":
        N_tension = max(0.0, Ni, Nj)
        N_compression = max(0.0, -Ni, -Nj)
    else:
        N_tension = max(0.0, -Ni, -Nj)
        N_compression = max(0.0, Ni, Nj)

    My_i = float(di.get("My", 0.0))
    My_j = float(dj.get("My", 0.0))
    Mz_i = float(di.get("Mz", 0.0))
    Mz_j = float(dj.get("Mz", 0.0))
    MAy, MBy = _signed_MA_MB(My_i, My_j)
    MAz, MBz = _signed_MA_MB(Mz_i, Mz_j)

    My = max(abs(My_i), abs(My_j))
    Mz = max(abs(Mz_i), abs(Mz_j))
    Vy = max(abs(float(di.get("Vy", 0.0))), abs(float(dj.get("Vy", 0.0))))
    Vz = max(abs(float(di.get("Vz", 0.0))), abs(float(dj.get("Vz", 0.0))))
    T = max(abs(float(di.get("T", 0.0))), abs(float(dj.get("T", 0.0))))
    return {
        "N_tension": N_tension,
        "N_compression": N_compression,
        "N_abs": max(abs(Ni), abs(Nj)),
        "My": My,
        "Mz": Mz,
        "Vy": Vy,
        "Vz": Vz,
        "T": T,
        "My_ED_Max": My,
        "Mz_ED_Max": Mz,
        "My_end_i": My_i,
        "My_end_j": My_j,
        "Mz_end_i": Mz_i,
        "Mz_end_j": Mz_j,
        "MA": MAy,
        "MB": MBy,
        "MAy": MAy,
        "MBy": MBy,
        "MAz": MAz,
        "MBz": MBz,
    }


# -----------------------------------------------------------------------------
# simple EC3 rules exactly in the requested architecture
# -----------------------------------------------------------------------------

def _alpha_buckling_yz(sec: Mapping[str, Any]) -> Tuple[float, float]:
    fam = _family_group(sec)
    h = _h(sec)
    b = _b(sec)
    tf = _tf(sec)
    hb = None if (h is None or b is None or abs(b) <= 1e-18) else h / b

    if fam in {"IPE", "HE"}:
        if hb is not None and hb > 1.2:
            if tf is not None and tf <= 40e-3:
                return 0.21, 0.34
            if tf is not None and tf <= 100e-3:
                return 0.34, 0.49
        if hb is not None and hb <= 1.2:
            if tf is not None and tf <= 100e-3:
                return 0.34, 0.49
            if tf is not None and tf > 100e-3:
                return 0.76, 0.76
        return 0.34, 0.49

    if fam in {"CHS", "RHS"}:
        return 0.21, 0.21
    if fam in {"C", "2L"}:
        return 0.49, 0.49
    if fam == "L":
        return 0.34, 0.34
    return 0.34, 0.34


def _buckling_reduction(alpha: float, lam_bar: float) -> Tuple[float, float]:
    phi = 0.50 * (1.0 + alpha * (lam_bar - 0.2) + lam_bar ** 2)
    chi = 1.0 / (phi + _radq(phi ** 2 - lam_bar ** 2)) if phi > 0.0 else 1.0
    chi = min(1.0, chi)
    return phi, chi


def _compute_one_check(
    sec: Mapping[str, Any],
    actions: Mapping[str, float],
    l_member: float,
    material: Mapping[str, Any],
    gamma_M0: float,
    gamma_M1: float,
    gamma_M2: float,
    kc: float,
    beta: float,
    alpha_LT_user: Optional[float],
) -> Dict[str, Any]:
    """
    Deliberately simple implementation that follows the user logic.
    """
    fyk = float(_first_not_none(material.get("fyk"), material.get("fy")))
    E = float(material.get("E"))
    G = float(_first_not_none(material.get("G"), E / (2.0 * (1.0 + 0.30))))

    fam = _family_group(sec)
    cls = _section_class(sec)
    A = _A(sec)
    Av = _Av(sec)
    Iy = _Iy(sec)
    Iz = _Iz(sec)
    Wpl_y = _Wpl(sec, "y")
    Wpl_z = _Wpl(sec, "z")
    Wel_y = _Wel(sec, "y")
    Wel_z = _Wel(sec, "z")
    b = _b(sec)
    h = _h(sec)
    tw = _tw(sec)
    tf = _tf(sec)
    Di = _Di(sec)
    De = _De(sec)
    t = _t(sec)
    It = _It(sec)
    Iw = _Iw(sec)

    out: Dict[str, Any] = {
        "section": _section_name(sec) or fam,
        "section_family": fam,
        "section_class": cls,
        "A": A,
        "Av": Av,
        "Iy": Iy,
        "Iz": Iz,
        "Wpl_y": Wpl_y,
        "Wpl_z": Wpl_z,
        "Wel_y": Wel_y,
        "Wel_z": Wel_z,
        "b": b,
        "h": h,
        "tw": tw,
        "tf": tf,
        "Di": Di,
        "De": De,
        "t": t,
        "It": It,
        "Iw": Iw,
        "fyk": fyk,
        "E": E,
        "G": G,
        "gamma_M0": gamma_M0,
        "gamma_M1": gamma_M1,
        "gamma_M2": gamma_M2,
        "kc": kc,
        "beta": beta,
        "alpha_LT_user": alpha_LT_user,
        "l_0": float(l_member),
        "L_0": float(l_member) * float(beta),
        "warnings": [],
        "checks": {},
    }

    warnings = out["warnings"]
    checks = out["checks"]

    N_t = float(actions.get("N_tension", 0.0))
    N_c = float(actions.get("N_compression", 0.0))
    My = float(actions.get("My", 0.0))
    Mz = float(actions.get("Mz", 0.0))
    Vy = float(actions.get("Vy", 0.0))
    Vz = float(actions.get("Vz", 0.0))
    V = max(Vy, Vz)

    My_ED_Max = float(actions.get("My_ED_Max", My))
    Mz_ED_Max = float(actions.get("Mz_ED_Max", Mz))

    MA = float(actions.get("MA", 0.0))
    MB = float(actions.get("MB", 0.0))
    MAy = float(actions.get("MAy", 0.0))
    MBy = float(actions.get("MBy", 0.0))
    MAz = float(actions.get("MAz", 0.0))
    MBz = float(actions.get("MBz", 0.0))

    out.update({
        "N_tension": N_t,
        "N_compression": N_c,
        "My": My,
        "Mz": Mz,
        "Vy": Vy,
        "Vz": Vz,
        "V": V,
        "My_ED_Max": My_ED_Max,
        "Mz_ED_Max": Mz_ED_Max,
        "My_end_i": float(actions.get("My_end_i", 0.0)),
        "My_end_j": float(actions.get("My_end_j", 0.0)),
        "Mz_end_i": float(actions.get("Mz_end_i", 0.0)),
        "Mz_end_j": float(actions.get("Mz_end_j", 0.0)),
        "MA": MA,
        "MB": MB,
        "MAy": MAy,
        "MBy": MBy,
        "MAz": MAz,
        "MBz": MBz,
        "MEDAy": MAy,
        "MEDBy": MBy,
        "MEDAz": MAz,
        "MEDBz": MBz,
    })

    if cls == 4:
        warnings.append("Class 4 not implemented.")

    # ------------------------------------------------------------------
    # TENSION
    # ------------------------------------------------------------------
    Nt_RD = A * fyk / gamma_M0 if A is not None else None
    out["Nt_RD"] = Nt_RD
    if N_t > 0.0 and Nt_RD is not None:
        checks["DCR_t"] = N_t / Nt_RD

    # ------------------------------------------------------------------
    # COMPRESSION
    # ------------------------------------------------------------------
    Nc_RD = A * fyk / gamma_M0 if A is not None else None
    out["Nc_RD"] = Nc_RD
    if N_c > 0.0 and Nc_RD is not None:
        checks["DCR_c"] = N_c / Nc_RD

    # stability in compression
    # kept with L_0 so beta still counts in the verification workflow
    L0 = out["L_0"]
    Ncr_y = ((math.pi ** 2) * E * Iy) / (L0 ** 2) if (Iy is not None and L0 > 0.0) else None
    Ncr_z = ((math.pi ** 2) * E * Iz) / (L0 ** 2) if (Iz is not None and L0 > 0.0) else None
    out["Ncr_y"] = Ncr_y
    out["Ncr_z"] = Ncr_z

    lambda_s_y = _radq((A * fyk) / Ncr_y) if (A is not None and Ncr_y is not None and Ncr_y > 0.0) else math.nan
    lambda_s_z = _radq((A * fyk) / Ncr_z) if (A is not None and Ncr_z is not None and Ncr_z > 0.0) else math.nan
    out["lambda_s_y"] = lambda_s_y
    out["lambda_s_z"] = lambda_s_z

    alpha_y, alpha_z = _alpha_buckling_yz(sec)
    out["alpha_y"] = alpha_y
    out["alpha_z"] = alpha_z

    phi_y, chi_y = _buckling_reduction(alpha_y, lambda_s_y) if not math.isnan(lambda_s_y) else (math.nan, math.nan)
    phi_z, chi_z = _buckling_reduction(alpha_z, lambda_s_z) if not math.isnan(lambda_s_z) else (math.nan, math.nan)
    out["phi_y"] = phi_y
    out["phi_z"] = phi_z
    out["chi_y"] = chi_y
    out["chi_z"] = chi_z

    Nb_RD_y = None
    Nb_RD_z = None
    DCR_c_b_y = math.nan
    DCR_c_b_z = math.nan
    if cls in (1, 2, 3) and A is not None and not math.isnan(chi_y) and not math.isnan(chi_z):
        Nb_RD_y = (chi_y * A * fyk) / gamma_M1
        Nb_RD_z = (chi_z * A * fyk) / gamma_M1
        out["Nb_RD_y"] = Nb_RD_y
        out["Nb_RD_z"] = Nb_RD_z
        if N_c > 0.0:
            DCR_c_b_y = N_c / Nb_RD_y if Nb_RD_y else math.nan
            DCR_c_b_z = N_c / Nb_RD_z if Nb_RD_z else math.nan
            checks["DCR_c_b_y"] = DCR_c_b_y
            checks["DCR_c_b_z"] = DCR_c_b_z
    elif cls == 4:
        warnings.append("Buckling for class 4 not implemented.")

    # ------------------------------------------------------------------
    # SHEAR
    # ------------------------------------------------------------------
    V_RD = (Av * (fyk / _radq(3.0))) / gamma_M0 if Av is not None else None
    out["V_RD"] = V_RD
    DCR_V = math.nan
    if V_RD is not None:
        DCR_V = V / V_RD
        checks["DCR_V"] = DCR_V
        if DCR_V < 0.50:
            warnings.append("DCR_V < 0.50")

    rho = 0.0
    fyk_mod = fyk
    if V_RD is not None and V_RD > 0.0 and DCR_V > 0.50:
        rho = (((2.0 * V) / V_RD) - 1.0) ** 2
        rho = max(0.0, min(1.0, rho))
        fyk_mod = (1.0 - rho) * fyk
    out["rho"] = rho
    out["fyk_mod"] = fyk_mod

    # ------------------------------------------------------------------
    # BENDING MOMENT
    # ------------------------------------------------------------------
    Mc_RD_y = None
    Mc_RD_z = None
    if cls in (1, 2):
        fy_use = fyk_mod if DCR_V > 0.50 else fyk
        if Wpl_y is not None:
            Mc_RD_y = (Wpl_y * fy_use) / gamma_M0
        if Wpl_z is not None:
            Mc_RD_z = (Wpl_z * fy_use) / gamma_M0
    elif cls == 3:
        fy_use = fyk_mod if DCR_V > 0.50 else fyk
        if Wel_y is not None:
            Mc_RD_y = (Wel_y * fy_use) / gamma_M0
        if Wel_z is not None:
            Mc_RD_z = (Wel_z * fy_use) / gamma_M0
    elif cls == 4:
        warnings.append("Bending for class 4 not implemented.")

    out["Mc_RD_y"] = Mc_RD_y
    out["Mc_RD_z"] = Mc_RD_z
    if Mc_RD_y is not None and Mc_RD_z is not None:
        checks["DCR_M"] = _safe_div(My, Mc_RD_y) + _safe_div(Mz, Mc_RD_z)

    # ------------------------------------------------------------------
    # INSTABILITY (LTB)
    # ------------------------------------------------------------------
    psi_LT_ratio = _psi_ratio(MA, MB, default_if_zero=0.0)
    psi_LT = 1.75 - 1.05 * psi_LT_ratio + 0.3 * (psi_LT_ratio ** 2)
    out["psi_LT_ratio"] = psi_LT_ratio
    out["psi_LT"] = psi_LT

    Mcr = None
    if Iy is not None and It is not None and Iw is not None and L0 > 0.0 and G > 0.0:
        Mcr = psi_LT * (math.pi / L0) * _radq(E * Iy * G * It) * _radq(1.0 + ((math.pi / L0) ** 2) * ((E * Iw) / (G * It)))
    out["Mcr"] = Mcr

    Wref_LT = Wpl_y if Wpl_y is not None else Wel_y
    if Wpl_y is None and Wel_y is None:
        warnings.append("Wpl_y/Wel_y missing for lambda_LT.")
    lambda_LT = _radq((Wref_LT * fyk) / Mcr) if (Wref_LT is not None and Mcr is not None and Mcr > 0.0) else math.nan
    out["lambda_LT"] = lambda_LT

    alpha_LT = float(alpha_LT_user) if alpha_LT_user is not None else 0.34
    out["alpha_LT"] = alpha_LT
    phi_LT, chi_LT = _buckling_reduction(alpha_LT, lambda_LT) if not math.isnan(lambda_LT) else (math.nan, math.nan)
    out["phi_LT"] = phi_LT
    out["chi_LT"] = chi_LT

    Mb_RD = None
    if cls in (1, 2):
        if Wpl_y is not None and not math.isnan(chi_LT):
            Mb_RD = chi_LT * ((Wpl_y * fyk) / gamma_M1)
    elif cls == 3:
        if Wel_y is not None and not math.isnan(chi_LT):
            Mb_RD = chi_LT * ((Wel_y * fyk) / gamma_M1)
    elif cls == 4:
        warnings.append("LTB for class 4 not implemented.")
    out["Mb_RD"] = Mb_RD
    if Mb_RD is not None:
        checks["DCR_LT"] = _safe_div(My, Mb_RD)

    # ------------------------------------------------------------------
    # BENDING MOMENT + AXIAL FORCE (TENSION or COMPRESSION) - RESISTANCE
    # ------------------------------------------------------------------
    DCR_MN = math.nan
    if cls in (1, 2) and Nc_RD is not None and Nc_RD > 0.0 and Mc_RD_y is not None and Mc_RD_z is not None:
        n = N_c / Nc_RD
        out["n_ratio"] = n
        if fam == "CHS":
            Mn_RD_y = Mc_RD_y * (1.0 - n ** 2)
            Mn_RD_z = Mc_RD_z * (1.0 - n ** 2)
            a_exp = 2.0
            b_exp = 2.0
            DCR_MN = _safe_div(My, Mn_RD_y) ** a_exp + _safe_div(Mz, Mn_RD_z) ** b_exp
            out["Mn_RD_y"] = Mn_RD_y
            out["Mn_RD_z"] = Mn_RD_z
            out["a_exp"] = a_exp
            out["b_exp"] = b_exp
        elif fam == "RHS":
            Mn_RD_y = Mc_RD_y * (1.0 - n ** 2)
            Mn_RD_z = Mc_RD_z * (1.0 - n ** 2)
            a_exp = 1.66 / max(1e-12, (1.0 - 1.13 * n ** 2))
            a_exp = min(6.0, a_exp)
            b_exp = a_exp
            DCR_MN = _safe_div(My, Mn_RD_y) ** a_exp + _safe_div(Mz, Mn_RD_z) ** b_exp
            out["Mn_RD_y"] = Mn_RD_y
            out["Mn_RD_z"] = Mn_RD_z
            out["a_exp"] = a_exp
            out["b_exp"] = b_exp
        elif fam in {"IPE", "HE"}:
            a_coeff = None
            if A is not None and b is not None and tf is not None and A > 0.0:
                a_coeff = min(0.50, (A - 2.0 * b * tf) / A)
            Mn_RD_y = Mc_RD_y * ((1.0 - n) / max(1e-12, (1.0 - 0.5 * a_coeff))) if a_coeff is not None else None
            Mn_RD_z = Mc_RD_z * ((1.0 - n) / max(1e-12, (1.0 - 0.5 * a_coeff))) if a_coeff is not None else None
            a_exp = 2.0
            b_exp = max(1.0, 5.0 * n)
            if Mn_RD_y is not None and Mn_RD_z is not None:
                DCR_MN = _safe_div(My, Mn_RD_y) ** a_exp + _safe_div(Mz, Mn_RD_z) ** b_exp
            out["a_coeff"] = a_coeff
            out["Mn_RD_y"] = Mn_RD_y
            out["Mn_RD_z"] = Mn_RD_z
            out["a_exp"] = a_exp
            out["b_exp"] = b_exp
        else:
            warnings.append("Family not explicitly defined for class 1/2 MN check: linear fallback used.")
            DCR_MN = _safe_div(N_c, Nc_RD) + _safe_div(My, Mc_RD_y) + _safe_div(Mz, Mc_RD_z)
    elif cls == 3 and Nc_RD is not None and Mc_RD_y is not None and Mc_RD_z is not None:
        DCR_MN = _safe_div(N_c, Nc_RD) + _safe_div(My, Mc_RD_y) + _safe_div(Mz, Mc_RD_z)
    elif cls == 4:
        warnings.append("MN resistance for class 4 not implemented.")

    if not math.isnan(DCR_MN):
        checks["DCR_MN"] = DCR_MN

    # ------------------------------------------------------------------
    # BENDING MOMENT + AXIAL FORCE (COMPRESSION) - INSTABILITY
    # ------------------------------------------------------------------
    psi_y = _psi_ratio(MAy, MBy, default_if_zero=0.0)
    psi_z = _psi_ratio(MAz, MBz, default_if_zero=0.0)
    cmy = max(0.4, 0.6 + 0.4 * psi_y)
    cmLT = cmy
    cmz = max(0.4, 0.6 + 0.4 * psi_z)
    out["psi_y"] = psi_y
    out["psi_z"] = psi_z
    out["cmy"] = cmy
    out["cmLT"] = cmLT
    out["cmz"] = cmz

    if (
        N_c > 0.0 and
        Nb_RD_y is not None and Nb_RD_z is not None and
        Wpl_y is not None and Wpl_z is not None and
        not math.isnan(DCR_c_b_y) and not math.isnan(DCR_c_b_z)
    ):
        kyy_base = cmy * (1.0 + (lambda_s_y - 0.2) * DCR_c_b_y)
        kyy_cap = cmy * (1.0 + 0.8 * DCR_c_b_y)
        kyy = min(kyy_base, kyy_cap)

        kzz_base = cmz * (1.0 + (lambda_s_z - 0.2) * DCR_c_b_z)
        kzz_cap = cmz * (1.0 + 0.8 * DCR_c_b_z)
        kzz = min(kzz_base, kzz_cap)

        kzy = 0.6 * kyy
        kyz = 0.6 * kzz

        out["kyy_base"] = kyy_base
        out["kyy_cap"] = kyy_cap
        out["kzz_base"] = kzz_base
        out["kzz_cap"] = kzz_cap
        out["kyy"] = kyy
        out["kzy"] = kzy
        out["kzz"] = kzz
        out["kyz"] = kyz

        term_y = _safe_div(My_ED_Max * gamma_M1, (chi_LT * fyk * Wpl_y)) if not math.isnan(chi_LT) else math.nan
        term_z = _safe_div(Mz_ED_Max * gamma_M1, (fyk * Wpl_z))
        out["term_y"] = term_y
        out["term_z"] = term_z

        DCR_MN_inst_1 = DCR_c_b_y + (kyy * term_y) + (kyz * term_z) if not any(math.isnan(v) for v in (DCR_c_b_y, kyy, kyz, term_y, term_z)) else math.nan
        DCR_MN_inst_2 = DCR_c_b_z + (kzy * term_y) + (kzz * term_z) if not any(math.isnan(v) for v in (DCR_c_b_z, kzy, kzz, term_y, term_z)) else math.nan

        C1 = N_c / min(Nb_RD_y, Nb_RD_z)
        C2 = _safe_div(My_ED_Max * gamma_M1, (chi_LT * fyk * Wpl_y) * (1.0 - _safe_div(N_c, Ncr_y))) if not math.isnan(chi_LT) else math.nan
        C3 = _safe_div(Mz_ED_Max * gamma_M1, (fyk * Wpl_z) * (1.0 - _safe_div(N_c, Ncr_z)))
        #DCR_MN_inst_3 = C1 + C2 + C3 if not any(math.isnan(v) for v in (C1, C2, C3)) else math.nan

        out["C1"] = C1
        out["C2"] = C2
        #out["C3"] = C3

        if not math.isnan(DCR_MN_inst_1):
            checks["DCR_MN_inst_1"] = DCR_MN_inst_1
        if not math.isnan(DCR_MN_inst_2):
            checks["DCR_MN_inst_2"] = DCR_MN_inst_2
        #if not math.isnan(DCR_MN_inst_3):
            #checks["DCR_MN_inst_3"] = DCR_MN_inst_3

    valid = {k: float(v) for k, v in checks.items() if isinstance(v, (int, float)) and not math.isnan(v)}
    if valid:
        gov = max(valid, key=valid.get)
        dcr = valid[gov]
    else:
        gov = None
        dcr = math.nan

    out["dcr"] = dcr
    out["governing_check"] = gov
    return out


# -----------------------------------------------------------------------------
# reporting helpers
# -----------------------------------------------------------------------------

def _safe_float_txt(x, ndigits: int = 6) -> str:
    if x is None:
        return "None"
    try:
        x = float(x)
    except Exception:
        return str(x)
    if math.isnan(x):
        return "nan"
    return f"{x:.{ndigits}f}"


def _id_to_str(key: Any) -> str:
    if isinstance(key, tuple) and len(key) == 2:
        return f"member {key}"
    try:
        return f"ele {int(key)}"
    except Exception:
        return str(key)


def _discrete_green_to_red(n_colors: int = 10):
    return plt.get_cmap("RdYlGn_r", int(n_colors))


def _dcr_color_scale(dcr_max: float, n_colors: int = 10, force_unit_cap: bool = True):
    dmax = float(dcr_max) if dcr_max is not None else 0.0
    if force_unit_cap:
        vmax = max(1.0, math.ceil(max(dmax, 1e-12) * 10.0) / 10.0)
    else:
        vmax = max(math.ceil(max(dmax, 1e-12) * 10.0) / 10.0, 1e-6)
    bounds = [i * vmax / int(n_colors) for i in range(int(n_colors) + 1)]
    cmap = _discrete_green_to_red(n_colors=n_colors)
    norm = colors.BoundaryNorm(bounds, cmap.N, clip=True)
    return cmap, norm, bounds, vmax


def _set_equal_axes_3d(ax, xs, ys, zs):
    if not xs:
        return
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    zmin, zmax = min(zs), max(zs)
    xr = xmax - xmin
    yr = ymax - ymin
    zr = zmax - zmin
    r = max(xr, yr, zr, 1e-9) / 2.0
    xc = (xmin + xmax) / 2.0
    yc = (ymin + ymax) / 2.0
    zc = (zmin + zmax) / 2.0
    ax.set_xlim(xc - r, xc + r)
    ax.set_ylim(yc - r, yc + r)
    ax.set_zlim(zc - r, zc + r)


def plot_ec3_dcr_map(
    G: Dict[str, Any],
    ele_data: Dict[str, Any],
    results: Dict[Any, Dict[str, Any]],
    mode: str = "members",
    n_colors: int = 10,
    show: bool = True,
    save_path: Optional[str] = None,
    figsize=(9, 7),
    linewidth: float = 2.4,
    title: Optional[str] = None,
    force_unit_cap: bool = True,
):
    mode = str(mode).strip().lower()
    if mode not in {"members", "subelements"}:
        raise ValueError("mode must be 'members' or 'subelements'.")

    valid_dcr = [float(r["dcr"]) for r in results.values() if isinstance(r.get("dcr"), (int, float)) and not math.isnan(r["dcr"])]
    dcr_max = max(valid_dcr) if valid_dcr else 0.0
    cmap, norm, bounds, vmax = _dcr_color_scale(dcr_max=dcr_max, n_colors=n_colors, force_unit_cap=force_unit_cap)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    segments = []
    values = []
    xs, ys, zs = [], [], []

    if mode == "members":
        chains = ele_data.get("member_nodechains", {}) or {}
        member_dcr: Dict[MemberKey, float] = {}

        for key, r in results.items():
            dcr = float(r.get("dcr", float("nan")))
            if math.isnan(dcr):
                continue
            if isinstance(key, tuple) and len(key) == 2:
                mk = _member_key(*key)
            else:
                e2m = _ele_to_member_map(ele_data)
                mk = e2m.get(int(key), None)
                if mk is None:
                    continue
            member_dcr[mk] = max(member_dcr.get(mk, float("-inf")), dcr)

        for mkey, chain in chains.items():
            mk = _member_key(*mkey)
            dcr = member_dcr.get(mk, 0.0)
            ch = [int(n) for n in chain]
            for i in range(len(ch) - 1):
                p1 = _node_xyz(G, ch[i])
                p2 = _node_xyz(G, ch[i + 1])
                segments.append([p1, p2])
                values.append(dcr)
                xs.extend([p1[0], p2[0]])
                ys.extend([p1[1], p2[1]])
                zs.extend([p1[2], p2[2]])
    else:
        for e in ele_data.get("ele_tags", []):
            e = int(e)
            dcr = float(results.get(e, {}).get("dcr", 0.0) or 0.0)
            try:
                nds = ops.eleNodes(e)
            except Exception:
                nds = None
            if nds is None or len(nds) < 2:
                continue
            p1 = _node_xyz(G, int(nds[0]))
            p2 = _node_xyz(G, int(nds[1]))
            segments.append([p1, p2])
            values.append(dcr)
            xs.extend([p1[0], p2[0]])
            ys.extend([p1[1], p2[1]])
            zs.extend([p1[2], p2[2]])

    if not segments:
        raise ValueError("No segments available to plot DCR map.")

    lc = Line3DCollection(segments, cmap=cmap, norm=norm, linewidths=linewidth)
    lc.set_array(values)
    ax.add_collection3d(lc)

    _set_equal_axes_3d(ax, xs, ys, zs)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.grid(True)
    ax.set_title(title or f"EC3 DCR map ({mode})")

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.08, boundaries=bounds, ticks=bounds)
    cbar.set_label("DCR")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"# File scritto: EC3 DCR plot written: {save_path}")
    if show:
        plt.show()
    return fig, ax


def print_ec3_check_summary(results: Dict[Any, Dict[str, Any]], dcr_max: float, top_n: int = 15) -> None:
    valid = [(k, r) for k, r in results.items() if isinstance(r.get("dcr"), (int, float)) and not math.isnan(r["dcr"])]
    valid.sort(key=lambda x: x[1]["dcr"], reverse=True)
    print(f"# EC3 SIMPLE CHECKS | n_items={len(results)} | dcr_max={dcr_max:.4f}")
    for key, r in valid[: int(top_n)]:
        print(f"{_id_to_str(key)} | sec={r.get('section')} | DCR={r['dcr']:.4f} | gov={r.get('governing_check')}")


def save_ec3_check_report(results: Dict[Any, Dict[str, Any]], dcr_max: float, filepath: str) -> None:
    valid = [(k, r) for k, r in results.items() if isinstance(r.get("dcr"), (int, float)) and not math.isnan(r["dcr"])]
    valid.sort(key=lambda x: x[1]["dcr"], reverse=True)

    scalar_groups = [
        ("ACTIONS", [
            "N_tension", "N_compression", "N_abs",
            "My", "Mz", "Vy", "Vz", "V",
            "My_ED_Max", "Mz_ED_Max",
            "My_end_i", "My_end_j", "Mz_end_i", "Mz_end_j",
            "MA", "MB", "MAy", "MBy", "MAz", "MBz",
            "MEDAy", "MEDBy", "MEDAz", "MEDBz",
            "l_0", "L_0",
        ]),
        ("SECTION / MATERIAL", [
            "section_family", "section_class",
            "A", "Av", "Iy", "Iz", "Wpl_y", "Wpl_z", "Wel_y", "Wel_z",
            "b", "h", "tw", "tf", "Di", "De", "t", "It", "Iw",
            "fyk", "E", "G",
            "gamma_M0", "gamma_M1", "gamma_M2", "beta", "alpha_LT_user",
        ]),
        ("COMPRESSION / BUCKLING", [
            "Nt_RD", "Nc_RD",
            "Ncr_y", "Ncr_z",
            "lambda_s_y", "lambda_s_z",
            "alpha_y", "alpha_z",
            "phi_y", "phi_z",
            "chi_y", "chi_z",
            "Nb_RD_y", "Nb_RD_z",
        ]),
        ("SHEAR / BENDING", [
            "V_RD", "rho", "fyk_mod",
            "Mc_RD_y", "Mc_RD_z",
            "n_ratio", "Mn_RD_y", "Mn_RD_z", "a_coeff", "a_exp", "b_exp",
        ]),
        ("LTB", [
            "psi_LT_ratio", "psi_LT",
            "Mcr", "lambda_LT", "alpha_LT", "phi_LT", "chi_LT", "Mb_RD",
        ]),
        ("COMPRESSION + BENDING INSTABILITY", [
            "psi_y", "psi_z", "cmy", "cmLT", "cmz",
            "kyy_base", "kyy_cap", "kyy",
            "kzz_base", "kzz_cap", "kzz",
            "kzy", "kyz",
            "term_y", "term_z",
            "C1", "C2", "C3",
        ]),
    ]

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"# EC3 SIMPLE CHECKS REPORT | n_items={len(results)} | dcr_max={dcr_max:.6f}\n")
        for key, r in valid:
            f.write(f"{_id_to_str(key)} | sec={r.get('section')} | DCR={r['dcr']:.4f} | gov={r.get('governing_check')}\n")
            for group_name, keys in scalar_groups:
                vals = []
                for k in keys:
                    if k in r:
                        vals.append(f"{k}={_safe_float_txt(r.get(k))}")
                if vals:
                    f.write(f"    {group_name}\n")
                    f.write("    " + " | ".join(vals) + "\n")
            if r.get("checks"):
                f.write("    CHECKS\n")
                for ck, cv in sorted((r.get("checks") or {}).items()):
                    f.write(f"    {ck}: {_safe_float_txt(cv)}\n")
            for msg in r.get("warnings", []):
                f.write(f"    WARNING: {msg}\n")


# -----------------------------------------------------------------------------
# main loop for all elements / members
# -----------------------------------------------------------------------------

def run_ec3_check(
    G: Dict[str, Any],
    ele_data: Dict[str, Any],
    ele_forces: Dict[Any, Dict[str, Any]],
    material: Dict[str, Any],
    section: Optional[Dict[str, Any]] = None,
    section_by_member: Optional[Mapping[MemberKey, Dict[str, Any]]] = None,
    gamma_M0: float = 1.0,
    gamma_M1: float = 1.0,
    gamma_M2: float = 1.25,
    kc: float = 1.0,
    beta: float = 1.0,
    alpha_LT: Optional[float] = None,
    compression_sign: str = "negative",
    aggregate_by_member: bool = True,
    print_summary: bool = True,
    top_n: int = 15,
    save_txt: Optional[str] = None,
    plot_dcr: bool = False,
    plot_mode: str = "members",
    plot_show: bool = True,
    plot_save_path: Optional[str] = None,
    plot_n_colors: int = 10,
):
    """
    Very simple EC3-like precheck, intentionally aligned with the user pseudo-code.
    Default behaviour: aggregate subelement forces to original member and loop on members.
    """
    if aggregate_by_member:
        from OS_results import build_member_force_envelopes

        force_items = build_member_force_envelopes(
            ele_forces=ele_forces,
            ele_data=ele_data,
            compression_sign=compression_sign,
        )
        sections_by_key = build_sections_by_member(
            ele_data=ele_data,
            section=section,
            section_by_member=section_by_member,
        )
        lengths_by_key = _build_member_lengths(G, ele_data)
    else:
        force_items = ele_forces
        sections_by_key = build_sections_by_element(
            ele_data=ele_data,
            section=section,
            section_by_member=section_by_member,
        )
        e2m = _ele_to_member_map(ele_data)
        member_lengths = _build_member_lengths(G, ele_data)
        lengths_by_key = {e: member_lengths[e2m[e]] for e in sections_by_key if e in e2m}

    results: Dict[Any, Dict[str, Any]] = {}
    dcr_max = 0.0

    for key, record in force_items.items():
        sec = sections_by_key[key]
        l_member = float(lengths_by_key[key])
        actions = _actions_from_record(record, compression_sign=compression_sign)
        item = _compute_one_check(
            sec=sec,
            actions=actions,
            l_member=l_member,
            material=material,
            gamma_M0=float(gamma_M0),
            gamma_M1=float(gamma_M1),
            gamma_M2=float(gamma_M2),
            kc=float(kc),
            beta=float(beta),
            alpha_LT_user=_as_float(alpha_LT),
        )
        results[key] = item
        if isinstance(item.get("dcr"), (int, float)) and not math.isnan(item["dcr"]):
            dcr_max = max(dcr_max, float(item["dcr"]))

    if print_summary:
        print_ec3_check_summary(results, dcr_max=dcr_max, top_n=top_n)
    if save_txt:
        save_ec3_check_report(results, dcr_max=dcr_max, filepath=save_txt)
    if plot_dcr:
        plot_ec3_dcr_map(
            G=G,
            ele_data=ele_data,
            results=results,
            mode=plot_mode,
            n_colors=plot_n_colors,
            show=plot_show,
            save_path=plot_save_path,
            title=f"EC3 DCR map ({plot_mode}) | dcr_max={dcr_max:.3f}",
        )

    return results, dcr_max, sections_by_key, lengths_by_key
