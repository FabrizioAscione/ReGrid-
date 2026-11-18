
import csv
import math
from typing import Any, Dict, List, Optional


def _to_float(value):
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.upper() in {"ND", "NONE", "N/A", "NA", "-"}:
        return None
    s = s.replace(",", ".")
    try:
        return float(s)
    except Exception:
        return None


def _to_int(value):
    f = _to_float(value)
    if f is None:
        return None
    try:
        return int(round(f))
    except Exception:
        return None


def _norm(s: Any) -> str:
    return "".join(ch for ch in str(s).strip().lower() if ch.isalnum() or ch == "_")


# exact/safer aliases: avoid collisions Iy vs iy and Iz vs iz
FIELD_ALIASES = {
    "name": ["name", "designation", "designazione", "section", "profilo", "sezione", "sigla"],
    "section_family": ["section_family", "family", "shape", "tipo", "type", "sectiontype"],
    "fabrication": ["fabrication", "manufacturing", "process"],
    "section_class": ["section_class", "class", "classe", "ec3class", "class_ec3"],
    "doubly_symmetric": ["doubly_symmetric", "doppiasimmetria", "double_symmetric"],

    "w_kN_per_m": ["w_kN_per_m", "wknperm", "selfweight", "peso", "g"],

    "A": ["A", "area"],
    "Aeff": ["Aeff", "effectivearea"],
    "Anet": ["Anet", "netarea"],
    "Af": ["Af", "flangearea"],
    "Aw": ["Aw", "webarea"],
    "Av": ["Av", "sheararea"],
    "Av_y": ["Av_y", "avy", "shearareay"],
    "Av_z": ["Av_z", "avz", "shearareaz"],

    "Iy": ["Iy", "iyy", "inertiay"],
    "Iz": ["Iz", "izz", "inertiaz"],
    "Iuu": ["Iuu", "iuu"],
    "Ivv": ["Ivv", "ivv"],
    "It": ["It", "torsionalconstant", "stvenant", "J"],
    "Iw": ["Iw", "warpingconstant"],

    "iy": ["iy", "ry", "radiusgyrationy"],
    "iz": ["iz", "rz", "radiusgyrationz"],

    "Wel_y": ["Wel_y", "wely", "Wy", "elasticmodulusy"],
    "Wel_z": ["Wel_z", "welz", "Wz", "elasticmodulusz"],
    "Wpl_y": ["Wpl_y", "wply", "plasticmodulusy"],
    "Wpl_z": ["Wpl_z", "wplz", "plasticmodulusz"],
    "Weff_y": ["Weff_y", "weffy"],
    "Weff_z": ["Weff_z", "weffz"],

    "t": ["t", "thickness", "spessore", "s"],
    "tw": ["tw", "webthickness", "spessoreanima"],
    "tf": ["tf", "flangethickness", "spessoreala"],
    "b": ["b", "width", "larghezza"],
    "h": ["h", "depth", "altezza"],
    "hw": ["hw", "webdepth", "altezzaanima"],
    "r": ["r", "rootradius", "radius"],

    "dext": ["dext", "De", "de", "diameterext", "diametroesterno"],
    "dint": ["dint", "Di", "di", "diameterint", "diametrointerno"],

    "eN_y": ["eN_y", "eny", "en_y", "shiftcentroidy"],
    "eN_z": ["eN_z", "enz", "en_z", "shiftcentroidz"],

    "fu": ["fu", "ultimatestrength"],
    "alpha_y": ["alpha_y", "alphay", "imperfectionfactory"],
    "alpha_z": ["alpha_z", "alphaz", "imperfectionfactorz"],
    "alpha_lt": ["alpha_lt", "alphalt", "imperfectionfactorlt"],
    "ltb_curve": ["ltb_curve", "ltbcurve", "lt_curve", "ltbucklingcurve"],
    "buckling_curve_y": ["buckling_curve_y", "bucklingcurvey", "curve_y", "bucklingy"],
    "buckling_curve_z": ["buckling_curve_z", "bucklingcurvez", "curve_z", "bucklingz"],
}


LEGACY_FIXED_COLUMNS = {
    0: "name",
    1: "dext",
    2: "t",
    3: "dint",
    4: "w_kN_per_m",
    5: "A",
    6: "Iy",
    7: "Iz",
    8: "Wel_y",
    9: "Wel_z",
    10: "Wpl_y",
    11: "Wpl_z",
    12: "i_legacy",
    13: "Av",
    14: "It",
}

LENGTH_KEYS = {"t", "tw", "tf", "b", "h", "hw", "r", "dext", "dint", "iy", "iz", "i_legacy", "eN_y", "eN_z"}
AREA_KEYS = {"A", "Aeff", "Anet", "Af", "Aw", "Av", "Av_y", "Av_z"}
INERTIA_KEYS = {"Iy", "Iz", "Iuu", "Ivv", "It", "Iw"}
MODULUS_KEYS = {"Wel_y", "Wel_z", "Wpl_y", "Wpl_z", "Weff_y", "Weff_z"}
NO_SCALE_KEYS = {
    "name", "section_family", "fabrication", "section_class", "doubly_symmetric",
    "alpha_y", "alpha_z", "alpha_lt", "ltb_curve", "buckling_curve_y", "buckling_curve_z",
    "w_kN_per_m", "fu", "raw",
}


def _factor(in_u: str, out_u: str) -> float:
    in_u = str(in_u).strip().lower()
    out_u = str(out_u).strip().lower()
    if in_u == out_u:
        return 1.0
    to_m = {"mm": 1e-3, "cm": 1e-2, "m": 1.0}
    if in_u not in to_m or out_u not in to_m:
        raise ValueError("units_in/units_out must be one of: 'mm', 'cm', 'm'")
    return to_m[in_u] / to_m[out_u]


def _header_looks_like_header(row: List[str]) -> bool:
    if not row:
        return False
    aliases = {_norm(a) for vals in FIELD_ALIASES.values() for a in vals}
    hits = sum(1 for cell in row if _norm(cell) in aliases)
    return hits >= 3


def _canonical_key(header: str) -> Optional[str]:
    h = _norm(header)
    for key, aliases in FIELD_ALIASES.items():
        norm_aliases = {_norm(a) for a in aliases}
        if h in norm_aliases:
            return key
    return None


def _coerce_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "y", "si", "sì"}:
        return True
    if s in {"0", "false", "no", "n"}:
        return False
    return None


def _is_units_row(row: List[str]) -> bool:
    if not row:
        return False
    count = 0
    for c in row:
        s = str(c).strip().lower()
        if any(tok in s for tok in ["cm", "mm", "m]", "kn/m", "cm2", "cm3", "cm4", "[cm", "(cm"]):
            count += 1
    return count >= max(2, len(row)//4)


def load_section_catalog(csv_path, sep=";", encoding="utf-8", name_col_index=0, skip_rows=0):
    with open(csv_path, newline="", encoding=encoding) as f:
        reader = csv.reader(f, delimiter=sep)
        rows = list(reader)

    if len(rows) <= skip_rows:
        raise ValueError("CSV troppo corto o formattato male.")

    data_rows = rows[skip_rows:]
    data_rows = [r for r in data_rows if r and any(str(c).strip() for c in r)]
    if not data_rows:
        raise ValueError("CSV vuoto dopo skip_rows.")

    catalog: Dict[str, Any] = {}

    # Legacy fixed-format tube CSV: title/header row + unit row + data rows
    if len(data_rows) >= 3:
        first0 = _norm(data_rows[0][0]) if data_rows[0] else ""
        first1 = _norm(data_rows[1][0]) if data_rows[1] else ""
        if first1 in {"designazione", "designation"} and first0 not in {"name", "designation", "designazione"}:
            for row in data_rows[2:]:
                if name_col_index >= len(row):
                    continue
                name = str(row[name_col_index]).strip()
                if not name or name in {"-", "Designazione"}:
                    continue
                catalog[name] = row
            return catalog

    # Header-based modern CSV
    if _header_looks_like_header(data_rows[0]):
        headers_raw = data_rows[0]
        canonical_headers = []
        seen = set()
        for idx, h in enumerate(headers_raw):
            ck = _canonical_key(h) or _norm(h) or f"col_{idx}"
            # prevent duplicate keys
            base = ck
            j = 2
            while ck in seen:
                ck = f"{base}__dup{j}"
                j += 1
            seen.add(ck)
            canonical_headers.append(ck)

        start_idx = 1
        if len(data_rows) > 1 and _is_units_row(data_rows[1]):
            start_idx = 2

        for row in data_rows[start_idx:]:
            rec = {canonical_headers[i]: row[i] if i < len(row) else None for i in range(len(canonical_headers))}
            name = str(rec.get("name", "")).strip()
            if not name and name_col_index < len(row):
                name = str(row[name_col_index]).strip()
            if not name or name == "-":
                continue
            catalog[name] = rec
        return catalog

    # Legacy CSV with title + units + fixed columns
    for row in data_rows:
        if name_col_index >= len(row):
            continue
        name = str(row[name_col_index]).strip()
        if not name or name in {"-", "Designazione"}:
            continue
        catalog[name] = row
    return catalog


def _extract_from_dict_row(rec: Dict[str, Any], key: str):
    # direct key
    if key in rec:
        return rec.get(key)
    # alias lookup over original headers
    norm_rec = {_norm(k): v for k, v in rec.items()}
    for alias in FIELD_ALIASES.get(key, []):
        a = _norm(alias)
        if a in norm_rec:
            return norm_rec[a]
    return None


def _extract_one_section(row: Any, sec_name: str, units_in: str, units_out: str, require: bool):
    fL = _factor(units_in, units_out)
    fA = fL**2
    fI = fL**4
    fW = fL**3

    parsed: Dict[str, Any] = {"name": sec_name, "raw": row}

    if isinstance(row, dict):
        for key in FIELD_ALIASES:
            parsed[key] = _extract_from_dict_row(row, key)
    else:
        for idx, key in LEGACY_FIXED_COLUMNS.items():
            parsed[key] = row[idx] if idx < len(row) else None

    parsed["section_family"] = None if parsed.get("section_family") is None else str(parsed["section_family"]).strip()
    parsed["fabrication"] = None if parsed.get("fabrication") is None else str(parsed["fabrication"]).strip()
    parsed["ltb_curve"] = None if parsed.get("ltb_curve") is None else str(parsed["ltb_curve"]).strip().lower()
    parsed["buckling_curve_y"] = None if parsed.get("buckling_curve_y") is None else str(parsed["buckling_curve_y"]).strip().lower()
    parsed["buckling_curve_z"] = None if parsed.get("buckling_curve_z") is None else str(parsed["buckling_curve_z"]).strip().lower()

    parsed["section_class"] = _to_int(parsed.get("section_class"))
    parsed["doubly_symmetric"] = _coerce_bool(parsed.get("doubly_symmetric"))

    for key in list(parsed.keys()):
        if key in NO_SCALE_KEYS or key in {"section_family", "fabrication", "section_class", "doubly_symmetric", "raw"}:
            continue
        parsed[key] = _to_float(parsed.get(key))

    for key, value in list(parsed.items()):
        if value is None or key in NO_SCALE_KEYS or key in {"section_family", "fabrication", "section_class", "doubly_symmetric"}:
            continue
        if key in LENGTH_KEYS:
            parsed[key] = value * fL
        elif key in AREA_KEYS:
            parsed[key] = value * fA
        elif key in INERTIA_KEYS:
            parsed[key] = value * fI
        elif key in MODULUS_KEYS:
            parsed[key] = value * fW

    # legacy radius of gyration column
    if parsed.get("i_legacy") is not None:
        parsed["iy"] = parsed.get("i_legacy")
        parsed["iz"] = parsed.get("i_legacy")

    A = parsed.get("A")
    Iy = parsed.get("Iy")
    Iz = parsed.get("Iz")
    if parsed.get("iy") is None and A not in (None, 0.0) and Iy is not None:
        parsed["iy"] = math.sqrt(Iy / A)
    if parsed.get("iz") is None and A not in (None, 0.0) and Iz is not None:
        parsed["iz"] = math.sqrt(Iz / A)

    # aliases expected by rest of project
    parsed["Av"] = parsed.get("Av") if parsed.get("Av") is not None else parsed.get("Av_z") or parsed.get("Av_y")
    parsed["Wy"] = parsed.get("Wel_y")
    parsed["Wz"] = parsed.get("Wel_z")
    parsed["Wply"] = parsed.get("Wpl_y")
    parsed["Wplz"] = parsed.get("Wpl_z")
    parsed["J"] = parsed.get("It")
    parsed["Dext"] = parsed.get("dext")
    parsed["Dint"] = parsed.get("dint")

    if require and (parsed.get("A") is None or parsed.get("Iy") is None or parsed.get("Iz") is None):
        raise ValueError(f"Sezione '{sec_name}' incompleta (A/Iy/Iz mancanti).")

    return parsed


def _print_props(p: dict):
    def fmt(v):
        return "None" if v is None else f"{v:.4e}" if isinstance(v, (int, float)) else str(v)

    keys = [
        "name", "section_family", "fabrication", "section_class",
        "w_kN_per_m", "A", "Aeff", "Av", "Av_y", "Av_z",
        "Iy", "Iz", "It", "Iw", "Wel_y", "Wel_z", "Wpl_y", "Wpl_z",
        "Weff_y", "Weff_z", "Dext", "Dint", "t", "b", "h", "tf", "tw", "iy", "iz"
    ]
    print("SECTION " + " | ".join(f"{k}={fmt(p.get(k))}" for k in keys))


def get_section_properties(
    catalog: dict,
    name: Optional[str] = None,
    units_in: str = "cm",
    units_out: str = "m",
    require: bool = True,
    print_props: bool = False,
    print_all: bool = False,
):
    def _one(sec_name: str):
        if sec_name not in catalog:
            raise KeyError(f"Sezione '{sec_name}' non trovata nel catalogo.")
        return _extract_one_section(catalog[sec_name], sec_name, units_in, units_out, require)

    if name is not None:
        out = _one(name)
        if print_props:
            _print_props(out)
        return out

    sections = []
    for sec_name in catalog.keys():
        try:
            p = _one(sec_name)
            sections.append(p)
            if print_props and print_all:
                _print_props(p)
        except Exception:
            if require:
                raise
            continue

    if print_props and not print_all:
        print(f"\nLoaded sections: {len(sections)} | units: {units_out}")

    return sections
