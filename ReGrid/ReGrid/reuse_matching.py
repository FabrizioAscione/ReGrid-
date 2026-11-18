import csv
import copy
import math
import os
import time
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


MemberKey = Tuple[int, int]
SectionDict = Dict[str, Any]
StockItem = Dict[str, Any]


_ALIASES = {
    "name": {"name", "designation", "designazione", "id", "stock_id", "nome", "nomesezione", "profilo", "sezione"},
    "section_family": {"section_family", "family", "shape", "tipo", "type", "sectiontype"},
    "fabrication": {"fabrication", "manufacturing", "process"},
    "section_class": {"section_class", "class", "classe", "ec3class", "class_ec3"},
    "w_kN_per_m": {"w_kn_per_m", "wknperm", "selfweight", "peso", "g"},
    "A": {"a", "area"},
    "Aeff": {"aeff", "effectivearea"},
    "Av": {"av", "sheararea"},
    "Av_y": {"av_y", "avy", "shearareay"},
    "Av_z": {"av_z", "avz", "shearareaz"},
    "Iy": {"iy", "iyy", "inertiay"},
    "Iz": {"iz", "izz", "inertiaz"},
    "Iuu": {"iuu"},
    "Ivv": {"ivv"},
    "It": {"it", "torsionalconstant", "stvenant", "j"},
    "Iw": {"iw", "warpingconstant", "cw"},
    "iy": {"ry", "radiusgyrationy"},
    "iz": {"rz", "radiusgyrationz"},
    "Wel_y": {"wel_y", "wely", "wy", "elasticmodulusy"},
    "Wel_z": {"wel_z", "welz", "wz", "elasticmodulusz"},
    "Wpl_y": {"wpl_y", "wply", "plasticmodulusy"},
    "Wpl_z": {"wpl_z", "wplz", "plasticmodulusz"},
    "Weff_y": {"weff_y", "weffy"},
    "Weff_z": {"weff_z", "weffz"},
    "t": {"t", "thickness", "spessore", "s"},
    "tf": {"tf", "flangethickness", "spessoreala", "t2", "tfcm", "tfmm", "tfm"},
    "b": {"b", "width", "larghezza", "l2", "bcm", "bmm", "bm"},
    "h": {"h", "depth", "altezza", "l1", "hcm", "hmm", "hm"},
    "hw": {"hw", "webdepth", "altezzaanima"},
    "r": {"r", "rootradius", "radius"},
    "dext": {"dext", "de", "diameterext", "diametroesterno"},
    "dint": {"dint", "di", "diameterint", "diametrointerno"},
    "tw": {"tw", "webthickness", "spessoreanima", "t1", "twcm", "twmm", "twm"},
    "L": {"l", "length", "lunghezza", "lm", "lengthm"},
}

_LENGTH_KEYS = {"t", "tw", "tf", "b", "h", "hw", "r", "dext", "dint", "iy", "iz", "L"}
_AREA_KEYS = {"A", "Aeff", "Av", "Av_y", "Av_z"}
_INERTIA_KEYS = {"Iy", "Iz", "Iuu", "Ivv", "It", "Iw"}
_MODULUS_KEYS = {"Wel_y", "Wel_z", "Wpl_y", "Wpl_z", "Weff_y", "Weff_z"}
_STRING_KEYS = {"name", "section_family", "fabrication"}
_INT_KEYS = {"section_class"}


def _norm(text: Any) -> str:
    return "".join(ch for ch in str(text).strip().lower() if ch.isalnum() or ch == "_")


def _canonical_header(header: str) -> str:
    norm = _norm(header)
    for key, aliases in _ALIASES.items():
        if norm in aliases:
            return key
    return norm


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() in {"ND", "NONE", "N/A", "NA", "-"}:
        return None
    text = text.replace(",", ".")
    try:
        return float(text)
    except Exception:
        return None


def _scale_factor(units_in: str, units_out: str) -> float:
    to_m = {"mm": 1e-3, "cm": 1e-2, "m": 1.0}
    u_in = str(units_in).strip().lower()
    u_out = str(units_out).strip().lower()
    if u_in not in to_m or u_out not in to_m:
        raise ValueError("units_in/units_out must be one of: 'mm', 'cm', 'm'")
    return to_m[u_in] / to_m[u_out]


def _member_key(a: int, b: int) -> MemberKey:
    a = int(a)
    b = int(b)
    return (a, b) if a < b else (b, a)


def _node_xyz(G: Mapping[str, Any], nid: int) -> Tuple[float, float, float]:
    x, y, z = G["nodes"][int(nid)]
    return float(x), float(y), float(z)


def _distance(p1: Sequence[float], p2: Sequence[float]) -> float:
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2 + (p2[2] - p1[2]) ** 2)


def _member_lengths(G: Mapping[str, Any], M: Mapping[str, Any]) -> Dict[MemberKey, float]:
    out: Dict[MemberKey, float] = {}
    for na, nb in M.get("members", []):
        key = _member_key(na, nb)
        out[key] = _distance(_node_xyz(G, na), _node_xyz(G, nb))
    return out


def _section_family(sec: Mapping[str, Any]) -> str:
    fam = str(sec.get("section_family", "") or "").strip().upper().replace(" ", "")
    if fam:
        return fam
    name = str(sec.get("name", "") or "").strip().upper().replace(" ", "")
    if name.startswith("2L") or "2L" in name:
        return "2L"
    if name.startswith("4L") or "4L" in name:
        return "4L"
    if name.startswith("L"):
        return "L"
    if "RHS" in name or "SHS" in name or "BOX" in name:
        return "RHS"
    if "CHS" in name or "PIPE" in name or "TUBE" in name or "O" == name[:1]:
        return "CHS"
    return "OTHER"


def _safe_float(sec: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = sec.get(key, default)
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _clone_section(sec: Mapping[str, Any]) -> SectionDict:
    return dict(sec)


def _weight_from_area(
    sec: Mapping[str, Any],
    *,
    units_out: str = "m",
    gamma_steel: float = 78.5,
) -> Optional[float]:
    area = _safe_float(sec, "A", default=0.0)
    if area <= 0.0:
        return None
    area_to_m = _scale_factor(units_out, "m")
    area_m2 = area * (area_to_m ** 2)
    return area_m2 * float(gamma_steel)


def _ensure_section_weight(
    sec: Mapping[str, Any],
    *,
    units_out: str = "m",
    gamma_steel: float = 78.5,
) -> SectionDict:
    out = dict(sec)
    if out.get("w_kN_per_m", None) is not None:
        return out
    w_kN_per_m = _weight_from_area(out, units_out=units_out, gamma_steel=gamma_steel)
    if w_kN_per_m is not None:
        out["w_kN_per_m"] = w_kN_per_m
    return out


def _build_single_l_props(h: float, b: float, tw: float, tf: float) -> SectionDict:
    A1 = h * tw
    A2 = max(0.0, b - tw) * tf
    A = A1 + A2

    if A <= 0.0:
        raise ValueError("Invalid L geometry: computed area <= 0.")

    av_z = A1
    av_y = A2

    y_g1 = tw / 2.0
    z_g1 = h / 2.0
    y_g2 = tw + max(0.0, b - tw) / 2.0
    z_g2 = tf / 2.0
    y_g = (A1 * y_g1 + A2 * y_g2) / A
    z_g = (A1 * z_g1 + A2 * z_g2) / A

    iy1 = (tw * (h ** 3)) / 12.0
    iy2 = (max(0.0, b - tw) * (tf ** 3)) / 12.0
    iy = iy1 + A1 * ((z_g - z_g1) ** 2) + iy2 + A2 * ((z_g - z_g2) ** 2)

    iz1 = (h * (tw ** 3)) / 12.0
    iz2 = (tf * (max(0.0, b - tw) ** 3)) / 12.0
    iz = iz1 + A1 * ((y_g - y_g1) ** 2) + iz2 + A2 * ((y_g - y_g2) ** 2)

    iyz = A1 * (y_g1 - y_g) * (z_g1 - z_g) + A2 * (y_g2 - y_g) * (z_g2 - z_g)
    i_avg = 0.5 * (iy + iz)
    i_rad = math.sqrt(max(0.0, ((0.5 * (iy - iz)) ** 2) + (iyz ** 2)))
    iuu = i_avg + i_rad
    ivv = i_avg - i_rad

    wy = iy / max(z_g, h - z_g, 1e-18)
    wz = iz / max(y_g, b - y_g, 1e-18)
    it = ((h * (tw ** 3)) + (b * (tf ** 3))) / 3.0
    iw = (((tw ** 3) * (h ** 3)) + ((tf ** 3) * (b ** 3))) / 36.0

    return {
        "section_family": "L",
        "fabrication": "hot_rolled",
        "section_class": 3,
        "A": A,
        "Av": min(av_y, av_z) if min(av_y, av_z) > 0.0 else max(av_y, av_z),
        "Av_y": av_y,
        "Av_z": av_z,
        "Iy": iy,
        "Iz": iz,
        "Iuu": iuu,
        "Ivv": ivv,
        "It": it,
        "Iw": iw,
        "J": it,
        "Wel_y": wy,
        "Wel_z": wz,
        "Wy": wy,
        "Wz": wz,
        "h": h,
        "b": b,
        "tw": tw,
        "tf": tf,
        "t": min(tw, tf),
    }


def _load_reuse_stock(
    csv_path: str,
    *,
    section_units_in: str,
    units_out: str,
    length_units_in: str,
) -> Tuple[List[StockItem], List[str]]:
    sec_factor = _scale_factor(section_units_in, units_out)
    len_factor = _scale_factor(length_units_in, units_out)
    area_factor = sec_factor ** 2
    inertia_factor = sec_factor ** 4
    modulus_factor = sec_factor ** 3

    items: List[StockItem] = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=";")
        if reader.fieldnames is None:
            raise ValueError(f"CSV '{csv_path}' has no headers.")
        headers = list(reader.fieldnames)
        for row_idx, row in enumerate(reader, start=2):
            if not row or not any(str(v).strip() for v in row.values()):
                continue
            parsed: SectionDict = {}
            for raw_key, raw_value in row.items():
                key = _canonical_header(raw_key)
                if key in _STRING_KEYS:
                    parsed[key] = None if raw_value is None else str(raw_value).strip()
                elif key in _INT_KEYS:
                    value = _to_float(raw_value)
                    parsed[key] = None if value is None else int(round(value))
                else:
                    value = _to_float(raw_value)
                    if value is None:
                        parsed[key] = None
                    elif key in _LENGTH_KEYS:
                        factor = len_factor if key == "L" else sec_factor
                        parsed[key] = value * factor
                    elif key in _AREA_KEYS:
                        parsed[key] = value * area_factor
                    elif key in _INERTIA_KEYS:
                        parsed[key] = value * inertia_factor
                    elif key in _MODULUS_KEYS:
                        parsed[key] = value * modulus_factor
                    else:
                        parsed[key] = value

            base_name = str(parsed.get("name") or f"stock_{row_idx}").strip()
            length = parsed.get("L", None)
            if length is None:
                raise ValueError(f"Reuse stock row {row_idx} has no valid length 'L'.")

            h = parsed.get("h", None)
            b = parsed.get("b", None)
            tw = parsed.get("tw", None)
            tf = parsed.get("tf", None)
            if None not in (h, b, tw, tf):
                l_props = _build_single_l_props(float(h), float(b), float(tw), float(tf))
                for key, value in l_props.items():
                    if parsed.get(key, None) is None:
                        parsed[key] = value
                if not parsed.get("section_family", None):
                    parsed["section_family"] = "L"

            if parsed.get("Av") is None:
                av_y = parsed.get("Av_y", None)
                av_z = parsed.get("Av_z", None)
                av_values = [float(v) for v in (av_y, av_z) if v is not None]
                parsed["Av"] = min(av_values) if av_values else None
            if parsed.get("J") is None and parsed.get("It") is not None:
                parsed["J"] = parsed.get("It")
            if parsed.get("Wy") is None and parsed.get("Wel_y") is not None:
                parsed["Wy"] = parsed.get("Wel_y")
            if parsed.get("Wz") is None and parsed.get("Wel_z") is not None:
                parsed["Wz"] = parsed.get("Wel_z")
            if parsed.get("Wply") is None and parsed.get("Wpl_y") is not None:
                parsed["Wply"] = parsed.get("Wpl_y")
            if parsed.get("Wplz") is None and parsed.get("Wpl_z") is not None:
                parsed["Wplz"] = parsed.get("Wpl_z")

            parsed = _ensure_section_weight(parsed, units_out=units_out)

            stock_id = f"{base_name}__{row_idx:04d}"
            parsed["name"] = base_name
            parsed["stock_id"] = stock_id

            items.append(
                {
                    "id": stock_id,
                    "length": float(length),
                    "section": parsed,
                    "family": _section_family(parsed),
                    "units_out": str(units_out).strip().lower(),
                }
            )
    return items, headers


@lru_cache(maxsize=8)
def _load_reuse_stock_cached(
    csv_path: str,
    section_units_in: str,
    units_out: str,
    length_units_in: str,
) -> Tuple[Tuple[StockItem, ...], Tuple[str, ...]]:
    items, headers = _load_reuse_stock(
        csv_path,
        section_units_in=section_units_in,
        units_out=units_out,
        length_units_in=length_units_in,
    )
    return tuple(copy.deepcopy(items)), tuple(headers)


def _profile_signature(item: StockItem) -> Tuple[Any, ...]:
    sec = item["section"]
    return (
        item["family"],
        round(_safe_float(sec, "h"), 12),
        round(_safe_float(sec, "b"), 12),
        round(_safe_float(sec, "tw"), 12),
        round(_safe_float(sec, "tf"), 12),
    )


def _l_geometry_from_stock(item: StockItem) -> Tuple[float, float, float, float]:
    sec = item["section"]
    h = _safe_float(sec, "h")
    b = _safe_float(sec, "b")
    tw = _safe_float(sec, "tw")
    tf = _safe_float(sec, "tf")
    if h <= 0.0 or b <= 0.0 or tw <= 0.0 or tf <= 0.0:
        raise ValueError(
            f"Stock item '{item['id']}' is missing valid L geometry (h/b/tw/tf) needed for 2L/4L assembly."
        )
    return h, b, tw, tf


def _build_double_l_props(h: float, b: float, tw: float, tf: float) -> SectionDict:
    A1 = (2.0 * h) * tw
    A2 = max(0.0, b - tw) * (2.0 * tf)
    A = A1 + A2

    av_z = A1
    av_y = A2

    z_g1 = h
    y_g1 = tw / 2.0
    z_g2 = h
    y_g2 = tw + max(0.0, b - tw) / 2.0

    z_g = (A1 * z_g1 + A2 * z_g2) / A
    y_g = (A1 * y_g1 + A2 * y_g2) / A

    iy1 = (tw * ((2.0 * h) ** 3)) / 12.0
    iy2 = (max(0.0, b - tw) * ((2.0 * tf) ** 3)) / 12.0
    iy = iy1 + A1 * (z_g - z_g1) ** 2 + iy2 + A2 * (z_g - z_g2) ** 2

    iz1 = ((2.0 * h) * (tw ** 3)) / 12.0
    iz2 = ((2.0 * tf) * (max(0.0, b - tw) ** 3)) / 12.0
    iz = iz1 + A1 * (y_g - y_g1) ** 2 + iz2 + A2 * (y_g - y_g2) ** 2

    wz = iz / max(b - y_g, 1e-18)
    wy = iy / max((2.0 * h) - z_g, 1e-18)

    it = (((2.0 * h) * (tw ** 3)) + ((b - tw / 2.0) * ((2.0 * tf) ** 3))) / 3.0
    iw = abs(-(((h ** 3) * (tw ** 3)) / 144.0) + ((((b - tw / 2.0) ** 3) * ((2.0 * tf) ** 3)) / 36.0))

    return {
        "section_family": "2L",
        "fabrication": "built_up",
        "section_class": 3,
        "A": A,
        "Av": min(av_y, av_z),
        "Av_y": av_y,
        "Av_z": av_z,
        "Iy": iy,
        "Iz": iz,
        "It": it,
        "Iw": iw,
        "J": it,
        "Wel_y": wy,
        "Wel_z": wz,
        "Wy": wy,
        "Wz": wz,
        "h": 2.0 * h,
        "b": b,
        "t": min(tw, tf),
        "tw": 2.0 * tf,
        "tf": tw,
    }


def _build_four_l_props(h: float, b: float, tw_single: float, tf_single: float) -> SectionDict:
    tf = tf_single
    tw = 2.0 * tw_single
    H = 2.0 * h
    B = 2.0 * b

    A = (B * tf) * 2.0 + ((H - 2.0 * tf) * tw)
    av_z = tf * B * 2.0
    av_y = tw * (H - 2.0 * tf)
    iz = (B * (H ** 3)) / 12.0 - 2.0 * (((b - tw / 2.0) * ((H - 2.0 * tf) ** 3)) / 12.0)
    iy = (((B ** 3) * tf) / 12.0) * 2.0 + (((tw ** 3) * (H - 2.0 * tf)) / 12.0)
    wz = iz / max(H / 2.0, 1e-18)
    wy = iy / max(B / 2.0, 1e-18)
    it = ((2.0 * B * (tf ** 3)) + ((H - tf) * (tw ** 3))) / 3.0
    iw = abs(-((((H - tf) ** 2) * (B ** 3) * tf) / 24.0))

    return {
        "section_family": "4L",
        "fabrication": "built_up",
        "section_class": 3,
        "A": A,
        "Av": min(av_y, av_z),
        "Av_y": av_y,
        "Av_z": av_z,
        "Iy": iy,
        "Iz": iz,
        "It": it,
        "Iw": iw,
        "J": it,
        "Wel_y": wy,
        "Wel_z": wz,
        "Wy": wy,
        "Wz": wz,
        "h": H,
        "b": B,
        "t": min(tw_single, tf_single),
        "tw": tw,
        "tf": tf,
    }


def _candidate_from_components(components: Sequence[StockItem], group_size: int) -> Dict[str, Any]:
    units_out = str(components[0].get("units_out", "m")).strip().lower()
    if group_size == 1:
        sec = _ensure_section_weight(_clone_section(components[0]["section"]), units_out=units_out)
        base_name = str(components[0]["section"].get("name", components[0]["id"]))
        sec["name"] = f"1L_{base_name}"
        return {
            "group_size": 1,
            "family": components[0]["family"],
            "components": list(components),
            "length": float(components[0]["length"]),
            "section": sec,
            "section_name": sec["name"],
            "stock_label": components[0]["id"],
        }

    h, b, tw, tf = _l_geometry_from_stock(components[0])
    base_name = str(components[0]["section"].get("name", components[0]["id"]))
    if group_size == 2:
        sec = _build_double_l_props(h, b, tw, tf)
    elif group_size == 4:
        sec = _build_four_l_props(h, b, tw, tf)
    else:
        raise ValueError(f"Unsupported group_size={group_size}.")
    sec = _ensure_section_weight(sec, units_out=units_out)
    sec["name"] = f"{group_size}L_{base_name}"
    return {
        "group_size": group_size,
        "family": sec["section_family"],
        "components": list(components),
        "length": min(float(comp["length"]) for comp in components),
        "section": sec,
        "section_name": sec["name"],
        "stock_label": ";".join(comp["id"] for comp in components),
    }


def _build_candidates(stock_items: Sequence[StockItem], group_sizes: Iterable[int]) -> List[Dict[str, Any]]:
    available = list(stock_items)
    candidates: List[Dict[str, Any]] = []

    if 1 in set(group_sizes):
        for item in available:
            candidates.append(_candidate_from_components([item], 1))

    l_items = [item for item in available if item["family"] == "L"]
    grouped: Dict[Tuple[Any, ...], List[StockItem]] = {}
    for item in l_items:
        grouped.setdefault(_profile_signature(item), []).append(item)

    for items in grouped.values():
        items_sorted = sorted(items, key=lambda x: float(x["length"]), reverse=True)
        for group_size in group_sizes:
            if group_size not in {2, 4}:
                continue
            for idx in range(0, len(items_sorted), group_size):
                group = items_sorted[idx:idx + group_size]
                if len(group) == group_size:
                    candidates.append(_candidate_from_components(group, group_size))

    return candidates


def _group_l_stock_by_signature(stock_items: Iterable[StockItem]) -> Dict[Tuple[Any, ...], List[StockItem]]:
    grouped: Dict[Tuple[Any, ...], List[StockItem]] = {}
    for item in stock_items:
        if item["family"] != "L":
            continue
        grouped.setdefault(_profile_signature(item), []).append(item)
    return grouped


def _add_item_to_l_groups(grouped: Dict[Tuple[Any, ...], List[StockItem]], item: StockItem) -> None:
    if item["family"] != "L":
        return
    grouped.setdefault(_profile_signature(item), []).append(item)


def _remove_item_from_l_groups(grouped: Dict[Tuple[Any, ...], List[StockItem]], item: StockItem) -> None:
    if item["family"] != "L":
        return
    sig = _profile_signature(item)
    items = grouped.get(sig, None)
    if not items:
        return
    item_id = item["id"]
    for idx, current in enumerate(items):
        if current["id"] == item_id:
            items.pop(idx)
            break
    if not items:
        grouped.pop(sig, None)


def _sorted_items_by_length_desc(items: Iterable[StockItem]) -> List[StockItem]:
    return sorted(items, key=lambda x: float(x["length"]), reverse=True)


def _rebuild_signature_group_candidates(
    grouped_l_stock: Mapping[Tuple[Any, ...], List[StockItem]],
    signature: Tuple[Any, ...],
    group_size: int,
) -> List[Dict[str, Any]]:
    items = grouped_l_stock.get(signature, [])
    if len(items) < int(group_size):
        return []
    items_sorted = _sorted_items_by_length_desc(items)
    candidates: List[Dict[str, Any]] = []
    for idx in range(0, len(items_sorted), int(group_size)):
        group = items_sorted[idx:idx + int(group_size)]
        if len(group) == int(group_size):
            candidates.append(_candidate_from_components(group, int(group_size)))
    return candidates


def _build_group_candidate_databases(
    grouped_l_stock: Mapping[Tuple[Any, ...], List[StockItem]],
    group_sizes: Sequence[int],
) -> Dict[int, Dict[Tuple[Any, ...], List[Dict[str, Any]]]]:
    out: Dict[int, Dict[Tuple[Any, ...], List[Dict[str, Any]]]] = {}
    for group_size in {int(v) for v in group_sizes if int(v) in {2, 4}}:
        out[group_size] = {}
        for signature in grouped_l_stock.keys():
            out[group_size][signature] = _rebuild_signature_group_candidates(grouped_l_stock, signature, group_size)
    return out


def _refresh_signature_group_candidates(
    candidate_databases: Dict[int, Dict[Tuple[Any, ...], List[Dict[str, Any]]]],
    grouped_l_stock: Mapping[Tuple[Any, ...], List[StockItem]],
    signature: Tuple[Any, ...],
    group_sizes: Sequence[int],
) -> None:
    for group_size in {int(v) for v in group_sizes if int(v) in {2, 4}}:
        db = candidate_databases.setdefault(group_size, {})
        db[signature] = _rebuild_signature_group_candidates(grouped_l_stock, signature, group_size)


def _single_candidate_from_item(item: StockItem) -> Dict[str, Any]:
    return _candidate_from_components([item], 1)


def _build_single_candidate_database(stock_items: Iterable[StockItem]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for item in stock_items:
        out[str(item["id"])] = _single_candidate_from_item(item)
    return out


def _best_candidate_for_member(
    *,
    single_candidate_database: Mapping[str, Dict[str, Any]],
    group_candidate_databases: Mapping[int, Mapping[Tuple[Any, ...], List[Dict[str, Any]]]],
    req_sec: Mapping[str, Any],
    req_length: float,
    max_cut_length: Optional[float],
    group_sizes: Sequence[int],
) -> Tuple[Optional[Dict[str, Any]], Dict[str, float]]:
    allowed_group_sizes = {int(v) for v in group_sizes}
    best_candidate: Optional[Dict[str, Any]] = None
    best_score = float("inf")
    n_single_checked = 0
    n_group_checked = 0
    t_single = 0.0
    t_group = 0.0

    if 1 in allowed_group_sizes:
        t0 = time.perf_counter()
        for cand in single_candidate_database.values():
            n_single_checked += 1
            if not _candidate_is_admissible(cand, req_sec, req_length, max_cut_length):
                continue
            score = _candidate_score(cand, req_sec, req_length)
            if score < best_score:
                best_score = score
                best_candidate = cand
        t_single = time.perf_counter() - t0

    for group_size in sorted(v for v in allowed_group_sizes if v in {2, 4}):
        t0 = time.perf_counter()
        for candidates in group_candidate_databases.get(group_size, {}).values():
            for cand in candidates:
                n_group_checked += 1
                if not _candidate_is_admissible(cand, req_sec, req_length, max_cut_length):
                    continue
                score = _candidate_score(cand, req_sec, req_length)
                if score < best_score:
                    best_score = score
                    best_candidate = cand
        t_group += time.perf_counter() - t0

    return best_candidate, {
        "time_single_search_s": t_single,
        "time_group_search_s": t_group,
        "n_single_checked": float(n_single_checked),
        "n_group_checked": float(n_group_checked),
    }


def _candidate_is_admissible(candidate: Mapping[str, Any], req_sec: Mapping[str, Any], req_length: float, max_cut_length: Optional[float]) -> bool:
    cand_length = float(candidate["length"])
    if cand_length < float(req_length):
        return False
    if max_cut_length is not None and (cand_length - float(req_length)) > float(max_cut_length):
        return False

    cand_sec = candidate["section"]
    req_A = _safe_float(req_sec, "A")
    req_Iy = _safe_float(req_sec, "Iy")
    req_Iz = _safe_float(req_sec, "Iz")

    if _safe_float(cand_sec, "A") < req_A:
        return False

    group_size = int(candidate["group_size"])
    family = str(candidate["family"]).upper()
    if group_size == 1 and family == "L":
        weakest_principal = min(_safe_float(cand_sec, "Iuu"), _safe_float(cand_sec, "Ivv"))
        strongest_required = max(req_Iy, req_Iz)
        return weakest_principal >= strongest_required

    return _safe_float(cand_sec, "Iy") >= req_Iy and _safe_float(cand_sec, "Iz") >= req_Iz


def _candidate_score(candidate: Mapping[str, Any], req_sec: Mapping[str, Any], req_length: float) -> float:
    cand_sec = candidate["section"]
    req_A = max(_safe_float(req_sec, "A"), 1e-18)
    req_Iy = max(_safe_float(req_sec, "Iy"), 1e-18)
    req_Iz = max(_safe_float(req_sec, "Iz"), 1e-18)
    req_major = max(req_Iy, req_Iz)

    len_gap = max(0.0, float(candidate["length"]) - float(req_length)) / max(float(req_length), 1e-18)
    area_gap = max(0.0, _safe_float(cand_sec, "A") - req_A) / req_A
    iy_gap = max(0.0, _safe_float(cand_sec, "Iy") - req_Iy) / req_Iy
    iz_gap = max(0.0, _safe_float(cand_sec, "Iz") - req_Iz) / req_Iz

    extra_gap = 0.0
    if int(candidate["group_size"]) == 1 and str(candidate["family"]).upper() == "L":
        weakest_principal = min(_safe_float(cand_sec, "Iuu"), _safe_float(cand_sec, "Ivv"))
        extra_gap = max(0.0, weakest_principal - req_major) / max(req_major, 1e-18)

    return (6.0 * len_gap) + area_gap + iy_gap + iz_gap + (0.5 * extra_gap)


def _member_priority(item: Tuple[MemberKey, SectionDict], lengths: Mapping[MemberKey, float]) -> Tuple[float, float, float]:
    key, sec = item
    return (
        max(_safe_float(sec, "Iy"), _safe_float(sec, "Iz")),
        _safe_float(sec, "A"),
        lengths.get(key, 0.0),
    )


def _remainder_item(component: StockItem, remainder_length: float, suffix: str) -> StockItem:
    units_out = str(component.get("units_out", "m")).strip().lower()
    sec = _ensure_section_weight(_clone_section(component["section"]), units_out=units_out)
    new_id = f"CUT_{component['id']}_{suffix}"
    sec["name"] = new_id
    return {
        "id": new_id,
        "length": float(remainder_length),
        "section": sec,
        "family": component["family"],
        "units_out": units_out,
    }


def _item_weight(item: StockItem, length: float) -> float:
    sec = item.get("section", {}) or {}
    try:
        w_kN_per_m = float(sec.get("w_kN_per_m", 0.0) or 0.0)
    except Exception:
        w_kN_per_m = 0.0
    return float(max(0.0, float(length))) * float(max(0.0, w_kN_per_m))


def _save_remaining_stock(items: Sequence[StockItem], csv_path: str) -> None:
    if not csv_path:
        return
    folder = os.path.dirname(csv_path)
    if folder:
        os.makedirs(folder, exist_ok=True)

    headers = [
        "name", "section_family", "fabrication", "section_class", "L",
        "w_kN_per_m", "A", "Aeff", "Av", "Av_y", "Av_z",
        "Iy", "Iz", "Iuu", "Ivv", "It", "Iw",
        "iy", "iz",
        "Wel_y", "Wel_z", "Wpl_y", "Wpl_z", "Weff_y", "Weff_z",
        "h", "b", "tf", "tw", "r", "dext", "t", "dint",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers, delimiter=";")
        writer.writeheader()
        for item in sorted(items, key=lambda x: float(x["length"]), reverse=True):
            sec = item["section"]
            row = {key: "" for key in headers}
            for key in headers:
                if key == "L":
                    row[key] = item["length"]
                elif key == "name":
                    row[key] = item["id"]
                elif key in sec and sec[key] is not None:
                    row[key] = sec[key]
            writer.writerow(row)


def _units_factor_from_m(units_out: str) -> float:
    u = str(units_out).strip().lower()
    if u == "m":
        return 100.0
    if u == "cm":
        return 1.0
    if u == "mm":
        return 0.1
    raise ValueError("units_out must be one of: 'm', 'cm', 'mm'")


def _csv_value(value: Any) -> Any:
    if value is None:
        return "ND"
    return value


def write_sections_used_catalog(
    section_by_member: Mapping[MemberKey, SectionDict],
    *,
    csv_path: str,
    gamma_steel: float,
    units_out: str = "m",
) -> str:
    factor = _units_factor_from_m(units_out)
    factor_area = factor ** 2
    factor_inertia = factor ** 4
    factor_modulus = factor ** 3

    unique_sections: Dict[str, SectionDict] = {}
    for sec in section_by_member.values():
        sec_name = str(sec.get("name", "")).strip()
        if not sec_name:
            continue
        if sec_name not in unique_sections:
            unique_sections[sec_name] = dict(sec)

    headers = [
        "name", "section_family", "fabrication", "section_class", "w_kN_per_m",
        "A", "Aeff", "Av_y", "Av_z", "Iy", "Iz", "It", "Iw", "iy", "iz",
        "Wel_y", "Wel_z", "Wpl_y", "Wpl_z", "Weff_y", "Weff_z",
        "h", "b", "tf", "tw", "r", "dext", "t", "dint", "Iuu", "Ivv",
    ]

    folder = os.path.dirname(csv_path)
    if folder:
        os.makedirs(folder, exist_ok=True)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers, delimiter=";")
        writer.writeheader()

        for sec_name, sec in sorted(unique_sections.items(), key=lambda x: x[0]):
            A = _safe_float(sec, "A", default=0.0)
            Iy = _safe_float(sec, "Iy", default=0.0)
            Iz = _safe_float(sec, "Iz", default=0.0)
            iy = sec.get("iy", None)
            if iy is None and A > 0.0 and Iy > 0.0:
                iy = math.sqrt(Iy / A)
            iz = sec.get("iz", None)
            if iz is None and A > 0.0 and Iz > 0.0:
                iz = math.sqrt(Iz / A)

            av_y = sec.get("Av_y", None)
            av_z = sec.get("Av_z", None)
            av = sec.get("Av", None)
            if av_y is None:
                av_y = av
            if av_z is None:
                av_z = av

            w_kN_per_m = A * float(gamma_steel)

            row = {
                "name": sec_name,
                "section_family": sec.get("section_family", "OTHER"),
                "fabrication": sec.get("fabrication", "built_up"),
                "section_class": sec.get("section_class", 3),
                "w_kN_per_m": _csv_value(w_kN_per_m),
                "A": _csv_value(A * factor_area if A is not None else None),
                "Aeff": _csv_value((_safe_float(sec, "Aeff", default=A) or A) * factor_area if A is not None else None),
                "Av_y": _csv_value(av_y * factor_area if av_y is not None else None),
                "Av_z": _csv_value(av_z * factor_area if av_z is not None else None),
                "Iy": _csv_value(Iy * factor_inertia if Iy is not None else None),
                "Iz": _csv_value(Iz * factor_inertia if Iz is not None else None),
                "It": _csv_value(_safe_float(sec, "It", default=_safe_float(sec, "J", default=0.0)) * factor_inertia),
                "Iw": _csv_value(_safe_float(sec, "Iw", default=0.0) * factor_inertia if sec.get("Iw", None) is not None else None),
                "iy": _csv_value(float(iy) * factor if iy is not None else None),
                "iz": _csv_value(float(iz) * factor if iz is not None else None),
                "Wel_y": _csv_value(_safe_float(sec, "Wel_y", default=_safe_float(sec, "Wy", default=0.0)) * factor_modulus if sec.get("Wel_y", sec.get("Wy", None)) is not None else None),
                "Wel_z": _csv_value(_safe_float(sec, "Wel_z", default=_safe_float(sec, "Wz", default=0.0)) * factor_modulus if sec.get("Wel_z", sec.get("Wz", None)) is not None else None),
                "Wpl_y": _csv_value(_safe_float(sec, "Wpl_y", default=_safe_float(sec, "Wply", default=0.0)) * factor_modulus if sec.get("Wpl_y", sec.get("Wply", None)) is not None else None),
                "Wpl_z": _csv_value(_safe_float(sec, "Wpl_z", default=_safe_float(sec, "Wplz", default=0.0)) * factor_modulus if sec.get("Wpl_z", sec.get("Wplz", None)) is not None else None),
                "Weff_y": _csv_value(_safe_float(sec, "Weff_y", default=0.0) * factor_modulus if sec.get("Weff_y", None) is not None else None),
                "Weff_z": _csv_value(_safe_float(sec, "Weff_z", default=0.0) * factor_modulus if sec.get("Weff_z", None) is not None else None),
                "h": _csv_value(_safe_float(sec, "h", default=0.0) * factor if sec.get("h", None) is not None else None),
                "b": _csv_value(_safe_float(sec, "b", default=0.0) * factor if sec.get("b", None) is not None else None),
                "tf": _csv_value(_safe_float(sec, "tf", default=0.0) * factor if sec.get("tf", None) is not None else None),
                "tw": _csv_value(_safe_float(sec, "tw", default=0.0) * factor if sec.get("tw", None) is not None else None),
                "r": _csv_value(_safe_float(sec, "r", default=0.0) * factor if sec.get("r", None) is not None else None),
                "dext": _csv_value(_safe_float(sec, "dext", default=_safe_float(sec, "Dext", default=0.0)) * factor if sec.get("dext", sec.get("Dext", None)) is not None else None),
                "t": _csv_value(_safe_float(sec, "t", default=0.0) * factor if sec.get("t", None) is not None else None),
                "dint": _csv_value(_safe_float(sec, "dint", default=_safe_float(sec, "Dint", default=0.0)) * factor if sec.get("dint", sec.get("Dint", None)) is not None else None),
                "Iuu": _csv_value(_safe_float(sec, "Iuu", default=0.0) * factor_inertia if sec.get("Iuu", None) is not None else None),
                "Ivv": _csv_value(_safe_float(sec, "Ivv", default=0.0) * factor_inertia if sec.get("Ivv", None) is not None else None),
            }
            writer.writerow(row)

    return csv_path


def _write_reuse_report(
    filepath: str,
    *,
    reused: Sequence[Mapping[str, Any]],
    unmatched: Sequence[MemberKey],
    total_cut_length: float,
    total_scrap_mass: float,
) -> None:
    folder = os.path.dirname(filepath)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("# REUSE MATCHING REPORT\n")
        f.write(f"matched_members = {len(reused)}\n")
        f.write(f"unmatched_members = {len(unmatched)}\n")
        f.write(f"total_small_cut_length = {float(total_cut_length):.6f}\n\n")
        f.write(f"total_scrap_mass = {float(total_scrap_mass):.6f}\n\n")
        f.write("## MATCHED\n")
        for item in reused:
            f.write(
                f"{item['member']}: {item['section_name']} | stock={item['stock_label']} | "
                f"group_size={item['group_size']} | "
                f"stock_length={float(item['stock_length']):.6f} | member_length={float(item['member_length']):.6f}\n"
            )
        f.write("\n## UNMATCHED\n")
        for mk in unmatched:
            f.write(f"{mk}\n")


def match_reused_members(
    G: Mapping[str, Any],
    M: Mapping[str, Any],
    *,
    section_by_member: Mapping[MemberKey, SectionDict],
    eligible_members: Optional[Iterable[MemberKey]] = None,
    reuse_csv_path: str,
    section_units_in: str = "cm",
    units_out: str = "m",
    length_units_in: str = "m",
    max_cut_length: Optional[float] = None,
    min_remainder_length: float = 0.0,
    group_sizes: Sequence[int] = (1, 2, 4),
    remaining_csv_path: Optional[str] = None,
    report_txt_path: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    t_match_start = time.perf_counter()
    stock_items_cached, _ = _load_reuse_stock_cached(
        reuse_csv_path,
        section_units_in,
        units_out,
        length_units_in,
    )
    stock_items = copy.deepcopy(list(stock_items_cached))
    lengths = _member_lengths(G, M)
    member_items = sorted(section_by_member.items(), key=lambda x: _member_priority(x, lengths), reverse=True)
    eligible_set = None if eligible_members is None else {
        ((int(mk[0]), int(mk[1])) if int(mk[0]) < int(mk[1]) else (int(mk[1]), int(mk[0])))
        for mk in eligible_members
    }

    new_section_by_member: Dict[MemberKey, SectionDict] = {mk: _clone_section(sec) for mk, sec in section_by_member.items()}
    section_name_by_member: Dict[MemberKey, str] = {mk: str(sec.get("name", "")) for mk, sec in new_section_by_member.items()}
    reused_records: List[Dict[str, Any]] = []
    unmatched_members: List[MemberKey] = []
    skipped_members: List[MemberKey] = []
    scrap_records: List[Dict[str, Any]] = []
    total_cut_length = 0.0
    total_scrap_mass = 0.0
    n_cuts = 0

    remaining_stock_by_id: Dict[str, StockItem] = {str(item["id"]): item for item in stock_items}
    single_candidate_database = _build_single_candidate_database(remaining_stock_by_id.values())
    grouped_l_stock = _group_l_stock_by_signature(remaining_stock_by_id.values())
    group_candidate_databases = _build_group_candidate_databases(grouped_l_stock, group_sizes)
    cut_counter = 0
    t_search_single_total = 0.0
    t_search_group_total = 0.0
    t_stock_update_total = 0.0
    n_single_checked_total = 0.0
    n_group_checked_total = 0.0

    for mk, req_sec in member_items:
        if eligible_set is not None and mk not in eligible_set:
            skipped_members.append(mk)
            continue
        req_length = float(lengths.get(mk, 0.0))
        best, search_stats = _best_candidate_for_member(
            single_candidate_database=single_candidate_database,
            group_candidate_databases=group_candidate_databases,
            req_sec=req_sec,
            req_length=req_length,
            max_cut_length=max_cut_length,
            group_sizes=group_sizes,
        )
        t_search_single_total += float(search_stats.get("time_single_search_s", 0.0))
        t_search_group_total += float(search_stats.get("time_group_search_s", 0.0))
        n_single_checked_total += float(search_stats.get("n_single_checked", 0.0))
        n_group_checked_total += float(search_stats.get("n_group_checked", 0.0))

        if best is None:
            unmatched_members.append(mk)
            continue

        best_sec = _clone_section(best["section"])
        best_sec["name"] = best["section_name"]
        new_section_by_member[mk] = best_sec
        section_name_by_member[mk] = best["section_name"]

        t0 = time.perf_counter()
        affected_signatures = set()
        for comp in best["components"]:
            comp_id = str(comp["id"])
            current_item = remaining_stock_by_id.pop(comp_id, None)
            if current_item is not None:
                single_candidate_database.pop(comp_id, None)
                if current_item["family"] == "L":
                    affected_signatures.add(_profile_signature(current_item))
                _remove_item_from_l_groups(grouped_l_stock, current_item)

        for comp in best["components"]:
            remainder = float(comp["length"]) - req_length
            if remainder > 0.0:
                n_cuts += 1
            if remainder > float(min_remainder_length):
                cut_counter += 1
                remainder_item = _remainder_item(comp, remainder, f"{cut_counter:04d}")
                remaining_stock_by_id[str(remainder_item["id"])] = remainder_item
                single_candidate_database[str(remainder_item["id"])] = _single_candidate_from_item(remainder_item)
                _add_item_to_l_groups(grouped_l_stock, remainder_item)
                if remainder_item["family"] == "L":
                    affected_signatures.add(_profile_signature(remainder_item))
            else:
                scrap_length = max(0.0, remainder)
                scrap_mass = _item_weight(comp, scrap_length)
                total_cut_length += scrap_length
                total_scrap_mass += scrap_mass
                if scrap_length > 0.0:
                    scrap_records.append(
                        {
                            "member": mk,
                            "section_name": str(comp.get("section", {}).get("name", comp.get("id", ""))),
                            "stock_label": str(comp.get("id", "")),
                            "group_size": int(best["group_size"]),
                            "stock_length": float(comp.get("length", 0.0)),
                            "used_length": req_length,
                            "scrap_length": float(scrap_length),
                            "scrap_mass": float(scrap_mass),
                            "section": _clone_section(comp.get("section", {}) or {}),
                        }
                    )
        for signature in affected_signatures:
            _refresh_signature_group_candidates(
                group_candidate_databases,
                grouped_l_stock,
                signature,
                group_sizes,
            )
        t_stock_update_total += time.perf_counter() - t0

        reused_records.append(
            {
                "member": mk,
                "section_name": best["section_name"],
                "stock_label": best["stock_label"],
                "group_size": int(best["group_size"]),
                "stock_length": float(best["length"]),
                "member_length": req_length,
            }
        )

        if verbose:
            print(
                f"# REUSE | member={mk} -> {best['section_name']} | "
                f"stock={best['stock_label']} | "
                f"group_size={int(best['group_size'])} | "
                f"Lstock={float(best['length']):.4f} | Lmember={req_length:.4f}"
            )

    if remaining_csv_path:
        _save_remaining_stock(list(remaining_stock_by_id.values()), remaining_csv_path)
    if report_txt_path:
        _write_reuse_report(
            report_txt_path,
            reused=reused_records,
            unmatched=unmatched_members,
            total_cut_length=total_cut_length,
            total_scrap_mass=total_scrap_mass,
        )

    t_total = time.perf_counter() - t_match_start
    return {
        "section_by_member": new_section_by_member,
        "section_name_by_member": section_name_by_member,
        "reused_records": reused_records,
        "scrap_records": scrap_records,
        "unmatched_members": unmatched_members,
        "skipped_members": skipped_members,
        "total_cut_length": total_cut_length,
        "total_scrap_mass": float(total_scrap_mass),
        "n_cuts": int(n_cuts),
        "remaining_stock_count": len(remaining_stock_by_id),
        "timings": {
            "time_total_s": t_total,
            "time_single_search_s": t_search_single_total,
            "time_group_search_s": t_search_group_total,
            "time_stock_update_s": t_stock_update_total,
            "n_single_checked": n_single_checked_total,
            "n_group_checked": n_group_checked_total,
        },
    }
