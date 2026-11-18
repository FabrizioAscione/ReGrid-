import math
from typing import Dict, Iterable, List, Mapping, Optional, Tuple, Union

import numpy as np
import openseespy.opensees as ops


SectionLike = Union[str, Mapping[str, float]]
MemberKey = Tuple[int, int]
SegmentKey = Tuple[int, int]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _member_key(a: int, b: int) -> MemberKey:
    a = int(a)
    b = int(b)
    return (a, b) if a < b else (b, a)


def _normalize(v: np.ndarray) -> Optional[np.ndarray]:
    n = float(np.linalg.norm(v))
    if n <= 1e-14:
        return None
    return v / n


def _get_w_from_any_section(
    sec: SectionLike,
    catalog=None,
    units_in: str = "cm",
    units_out: str = "m",
) -> float:
    """
    Restituisce il peso lineare w [kN/m] da:
      - dict con chiave 'w_kN_per_m'
      - stringa con nome sezione (richiede catalog)
    """
    if isinstance(sec, Mapping):
        w = sec.get("w_kN_per_m", None)
        if w is None:
            raise ValueError("Sezione dict senza chiave 'w_kN_per_m'.")
        return float(w)

    if isinstance(sec, str):
        if catalog is None:
            raise ValueError("Se sec è una stringa, devi passare anche 'catalog'.")
        from import_section import get_section_properties

        props = get_section_properties(
            catalog=catalog,
            name=sec,
            units_in=units_in,
            units_out=units_out,
            require=False,
        )
        w = props.get("w_kN_per_m", None)
        if w is None:
            raise ValueError(f"Nel catalogo, la sezione '{sec}' non ha 'w_kN_per_m'.")
        return float(w)

    raise TypeError(f"Tipo sezione non supportato: {type(sec)}. Usa dict o str.")


def _resolve_member_weight(
    mkey: MemberKey,
    section: Optional[SectionLike] = None,
    section_by_member: Optional[Mapping[MemberKey, SectionLike]] = None,
    catalog=None,
    units_in: str = "cm",
    units_out: str = "m",
) -> float:
    if section_by_member is not None:
        if mkey in section_by_member:
            sec = section_by_member[mkey]
        else:
            mk = _member_key(*mkey)
            if mk not in section_by_member:
                raise KeyError(f"section_by_member manca per member {mkey}.")
            sec = section_by_member[mk]
        return _get_w_from_any_section(sec, catalog=catalog, units_in=units_in, units_out=units_out)

    if section is None:
        raise ValueError("Devi passare 'section' oppure 'section_by_member'.")
    return _get_w_from_any_section(section, catalog=catalog, units_in=units_in, units_out=units_out)


def _node_xyz(G, nid: int) -> np.ndarray:
    nid = int(nid)
    nodes = G["nodes"]
    if nid in nodes:
        return np.array(nodes[nid], dtype=float)
    try:
        x, y, z = ops.nodeCoord(nid)
        return np.array([float(x), float(y), float(z)], dtype=float)
    except Exception as exc:
        raise KeyError(f"Node {nid} non trovato in G['nodes'] né nel dominio OpenSees.") from exc


def _iter_subsegments(chains: Mapping[MemberKey, Iterable[int]]):
    for mkey, chain in chains.items():
        mk = _member_key(*mkey)
        chain = [int(n) for n in chain]
        if len(chain) < 2:
            continue
        for i in range(len(chain) - 1):
            a = int(chain[i])
            b = int(chain[i + 1])
            yield mk, _member_key(a, b), a, b


def _pick_vecxz_for_element(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """
    Replica il criterio tipico che hai usato negli elementi frame:
    prova Z globale, poi Y, poi X come vettore di appoggio non parallelo all'asse locale x.
    """
    x_axis = _normalize(p2 - p1)
    if x_axis is None:
        return np.array([0.0, 0.0, 1.0], dtype=float)

    candidates = [
        np.array([0.0, 0.0, 1.0], dtype=float),
        np.array([0.0, 1.0, 0.0], dtype=float),
        np.array([1.0, 0.0, 0.0], dtype=float),
    ]
    for c in candidates:
        cr = np.cross(x_axis, c)
        if _normalize(cr) is not None:
            return c
    return np.array([0.0, 0.0, 1.0], dtype=float)


def _remove_existing_pattern_and_series(pattern_tag: int, ts_tag: int) -> None:
    try:
        ops.remove("loadPattern", int(pattern_tag))
    except Exception:
        pass
    try:
        ops.remove("timeSeries", int(ts_tag))
    except Exception:
        pass


def _direction_sign(direction: str) -> float:
    return -1.0 if str(direction).strip().upper() == "-Z" else 1.0


# -----------------------------------------------------------------------------
# 1) Self-weight as equivalent nodal loads
# -----------------------------------------------------------------------------

def selfweight_nodal_from_sections(
    G,
    ele_data,
    section: Optional[SectionLike] = None,
    section_by_member: Optional[Mapping[MemberKey, SectionLike]] = None,
    catalog=None,
    units_in: str = "cm",
    units_out: str = "m",
    direction: str = "-Z",
    pattern_tag: int = 20,
    ts_tag: int = 20,
    ndf: int = 6,
    print_summary: bool = True,
    return_only: bool = False,
):
    """
    Applica il peso proprio come carico nodale equivalente.

    Per ogni sotto-segmento (a,b):
        W = w * L
        Fz(a) += W/2
        Fz(b) += W/2

    con segno determinato da `direction`.

    Usa ele_data['member_nodechains'], quindi rispetta eventuali suddivisioni.

    Returns
    -------
    loads_sw : dict
        {nid: (Fx, Fy, Fz)}
    """
    ndf = int(ndf)
    if ndf < 3:
        raise ValueError("Need ndf >= 3.")

    chains = ele_data.get("member_nodechains", {})
    if not chains:
        raise ValueError("ele_data['member_nodechains'] vuoto.")

    sign_z = _direction_sign(direction)
    Fz: Dict[int, float] = {}
    Wtot = 0.0
    n_sub = 0

    for mkey, _, a, b in _iter_subsegments(chains):
        Fz.setdefault(a, 0.0)
        Fz.setdefault(b, 0.0)

        w = _resolve_member_weight(
            mkey,
            section=section,
            section_by_member=section_by_member,
            catalog=catalog,
            units_in=units_in,
            units_out=units_out,
        )

        pa = _node_xyz(G, a)
        pb = _node_xyz(G, b)
        L = float(np.linalg.norm(pb - pa))
        if L <= 1e-12:
            continue

        W = float(w) * L
        Wtot += W
        n_sub += 1
        Fz[a] += sign_z * W / 2.0
        Fz[b] += sign_z * W / 2.0

    loads_sw = {nid: (0.0, 0.0, float(fz)) for nid, fz in sorted(Fz.items())}

    py_lines = [
        f"ops.timeSeries('Linear', {int(ts_tag)})",
        f"ops.pattern('Plain', {int(pattern_tag)}, {int(ts_tag)})",
    ]

    if not return_only:
        _remove_existing_pattern_and_series(pattern_tag, ts_tag)
        ops.timeSeries("Linear", int(ts_tag))
        ops.pattern("Plain", int(pattern_tag), int(ts_tag))

        for nid, fz in sorted(Fz.items()):
            if abs(fz) < 1e-18:
                continue
            vec = [0.0] * ndf
            vec[2] = float(fz)
            ops.load(int(nid), *vec)
            py_lines.append(f"ops.load({int(nid)}, {', '.join(str(v) for v in vec)})")
    else:
        for nid, fz in sorted(Fz.items()):
            if abs(fz) < 1e-18:
                continue
            vec = [0.0] * ndf
            vec[2] = float(fz)
            py_lines.append(f"ops.load({int(nid)}, {', '.join(str(v) for v in vec)})")

    py_text = "\n".join(py_lines)

    if print_summary:
        print(
            f"# SELF-WEIGHT NODAL | n_sub={n_sub} | "
            f"Wtot={Wtot:.6f} kN | sumFz={sum(Fz.values()):.6f} kN"
        )

    return loads_sw, py_text


# -----------------------------------------------------------------------------
# 2) Self-weight as OpenSees eleLoad -beamUniform
# -----------------------------------------------------------------------------

def selfweight_eleload_uniform_from_sections(
    G,
    ele_data,
    section: Optional[SectionLike] = None,
    section_by_member: Optional[Mapping[MemberKey, SectionLike]] = None,
    catalog=None,
    units_in: str = "cm",
    units_out: str = "m",
    direction: str = "-Z",
    pattern_tag: int = 21,
    ts_tag: int = 21,
    print_summary: bool = True,
    return_only: bool = False,
    warn_angle_deg: float = 15.0,
    assume_beamUniform_order: str = "WyWzWx",
):
    """
    Applica il peso proprio come carico distribuito uniforme sugli elementi OpenSees.

    La gravità è definita nel sistema GLOBALE (±Z), poi proiettata sugli assi locali
    dell'elemento per costruire i termini di `-beamUniform`.

    Richiede che `ele_data` consenta di ricostruire il mapping segmento -> eTag:
      - preferibilmente tramite ele_data['seg_to_etag']
      - altrimenti interrogando gli elementi presenti nel dominio OpenSees

    Returns
    -------
    ele_q_local : dict
        eTag -> {
            'Wy','Wz','Wx','w','L','angle_ez_Zdeg','seg','member','vecxz_used'
        }
    """
    chains = ele_data.get("member_nodechains", {})
    if not chains:
        raise ValueError("ele_data['member_nodechains'] vuoto.")

    sign_z = _direction_sign(direction)
    Zup = np.array([0.0, 0.0, 1.0], dtype=float)

    seg_to_member: Dict[SegmentKey, MemberKey] = {}
    for mkey, seg_key, _, _ in _iter_subsegments(chains):
        seg_to_member[seg_key] = mkey

    seg_to_etag: Dict[SegmentKey, int] = {}
    if isinstance(ele_data.get("seg_to_etag", None), Mapping):
        for k, v in ele_data["seg_to_etag"].items():
            a, b = k
            seg_to_etag[_member_key(a, b)] = int(v)
    else:
        for eTag in ops.getEleTags():
            try:
                nds = ops.eleNodes(int(eTag))
                if nds is None or len(nds) < 2:
                    continue
                a = int(nds[0])
                b = int(nds[1])
                seg_to_etag[_member_key(a, b)] = int(eTag)
            except Exception:
                continue

    ele_q_local = {}
    Wtot = 0.0
    n_loaded = 0
    missing = 0
    max_abs_Wx = 0.0
    max_abs_Wy = 0.0
    max_abs_Wz = 0.0
    max_angle = 0.0

    for seg, mkey in seg_to_member.items():
        eTag = seg_to_etag.get(seg, None)
        if eTag is None:
            missing += 1
            continue

        a, b = seg
        pa = _node_xyz(G, a)
        pb = _node_xyz(G, b)
        dx = pb - pa
        L = float(np.linalg.norm(dx))
        if L <= 1e-12:
            continue

        w = _resolve_member_weight(
            mkey,
            section=section,
            section_by_member=section_by_member,
            catalog=catalog,
            units_in=units_in,
            units_out=units_out,
        )

        q_global = np.array([0.0, 0.0, sign_z * float(w)], dtype=float)
        ex = dx / L

        zh = _pick_vecxz_for_element(pa, pb)
        zh = zh / max(float(np.linalg.norm(zh)), 1e-12)

        zproj = zh - np.dot(zh, ex) * ex
        nz = float(np.linalg.norm(zproj))
        if nz < 1e-12:
            for alt in (
                np.array([0.0, 1.0, 0.0], dtype=float),
                np.array([1.0, 0.0, 0.0], dtype=float),
            ):
                zproj = alt - np.dot(alt, ex) * ex
                nz = float(np.linalg.norm(zproj))
                if nz >= 1e-12:
                    break
        ez = zproj / max(nz, 1e-12)
        ey = np.cross(ez, ex)
        ey = ey / max(float(np.linalg.norm(ey)), 1e-12)
        ez = np.cross(ex, ey)
        ez = ez / max(float(np.linalg.norm(ez)), 1e-12)

        Wx = float(np.dot(q_global, ex))
        Wy = float(np.dot(q_global, ey))
        Wz = float(np.dot(q_global, ez))

        cosang = float(np.clip(np.dot(ez, Zup), -1.0, 1.0))
        ang_deg = float(np.degrees(np.arccos(abs(cosang))))

        ele_q_local[int(eTag)] = {
            "Wy": Wy,
            "Wz": Wz,
            "Wx": Wx,
            "w": float(w),
            "L": L,
            "angle_ez_Zdeg": ang_deg,
            "seg": seg,
            "member": mkey,
            "vecxz_used": (float(zh[0]), float(zh[1]), float(zh[2])),
        }

        Wtot += float(w) * L
        n_loaded += 1
        max_abs_Wx = max(max_abs_Wx, abs(Wx))
        max_abs_Wy = max(max_abs_Wy, abs(Wy))
        max_abs_Wz = max(max_abs_Wz, abs(Wz))
        max_angle = max(max_angle, ang_deg)

    py_lines = [
        f"ops.timeSeries('Linear', {int(ts_tag)})",
        f"ops.pattern('Plain', {int(pattern_tag)}, {int(ts_tag)})",
    ]

    order = str(assume_beamUniform_order).upper()
    if order not in {"WYWZWX", "WZWYWX"}:
        raise ValueError("assume_beamUniform_order non riconosciuto. Usa 'WyWzWx' oppure 'WzWyWx'.")

    if not return_only:
        _remove_existing_pattern_and_series(pattern_tag, ts_tag)
        ops.timeSeries("Linear", int(ts_tag))
        ops.pattern("Plain", int(pattern_tag), int(ts_tag))

    for eTag, d in sorted(ele_q_local.items()):
        Wy, Wz, Wx = d["Wy"], d["Wz"], d["Wx"]
        if order == "WYWZWX":
            cmd = f"ops.eleLoad('-ele', {int(eTag)}, '-type', '-beamUniform', {float(Wy)}, {float(Wz)}, {float(Wx)})"
            if not return_only:
                ops.eleLoad("-ele", int(eTag), "-type", "-beamUniform", float(Wy), float(Wz), float(Wx))
        else:
            cmd = f"ops.eleLoad('-ele', {int(eTag)}, '-type', '-beamUniform', {float(Wz)}, {float(Wy)}, {float(Wx)})"
            if not return_only:
                ops.eleLoad("-ele", int(eTag), "-type", "-beamUniform", float(Wz), float(Wy), float(Wx))
        py_lines.append(cmd)

    py_text = "\n".join(py_lines)

    if print_summary:
        print(
            f"# SUMMARY | SELF-WEIGHT ELELOAD | nEleLoaded={n_loaded} | missingSeg={missing} | Wtot={Wtot:.6f} kN"
        )
        print(
            f"# SUMMARY | max|Wx|={max_abs_Wx:.6e}  max|Wy|={max_abs_Wy:.6e}  max|Wz|={max_abs_Wz:.6e}  (kN/m)"
        )
        print(f"# SUMMARY | worst angle between ez(local) and Z(global) = {max_angle:.2f} deg")
        if max_angle > float(warn_angle_deg):
            print(
                f"# SUMMARY | WARNING: ez locale è molto inclinato rispetto a Z globale in alcuni elementi (> {warn_angle_deg} deg)."
            )

    return ele_q_local, py_text


# -----------------------------------------------------------------------------
# 3) Wrapper
# -----------------------------------------------------------------------------

def apply_selfweight(
    G,
    ele_data,
    mode: str = "nodal",
    **kwargs,
):
    """
    Wrapper comodo.

    mode='nodal'   -> selfweight_nodal_from_sections(...)
    mode='eleload' -> selfweight_eleload_uniform_from_sections(...)
    """
    mode = str(mode).strip().lower()
    if mode == "nodal":
        return selfweight_nodal_from_sections(G=G, ele_data=ele_data, **kwargs)
    if mode == "eleload":
        return selfweight_eleload_uniform_from_sections(G=G, ele_data=ele_data, **kwargs)
    raise ValueError("mode deve essere 'nodal' oppure 'eleload'.")

