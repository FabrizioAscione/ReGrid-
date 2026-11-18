import csv
import math
from pathlib import Path
from typing import Any


STEEL_UNIT_WEIGHT_KN_PER_M3 = 78.5

SUPPORTED_MODES = {"all", "flanges", "web"}

GEOMETRY_COLUMNS = ("h", "b", "tf", "tw", "r", "dext", "t", "dint")
PROPERTY_COLUMNS = (
    "w_kN_per_m",
    "A",
    "Aeff",
    "Av_y",
    "Av_z",
    "Iy",
    "Iz",
    "Iuu",
    "Ivv",
    "Iyz",
    "It",
    "Iw",
    "iy",
    "iz",
    "Wel_y",
    "Wel_z",
    "Wpl_y",
    "Wpl_z",
    "Weff_y",
    "Weff_z",
    "theta_principal_deg",
)
TRACKED_COLUMNS = PROPERTY_COLUMNS + GEOMETRY_COLUMNS
MONOTONIC_PROPERTY_COLUMNS = (
    "w_kN_per_m",
    "A",
    "Aeff",
    "Av_y",
    "Av_z",
    "Iy",
    "Iz",
    "Iuu",
    "Ivv",
    "It",
    "Iw",
    "Wel_y",
    "Wel_z",
    "Wpl_y",
    "Wpl_z",
    "Weff_y",
    "Weff_z",
)


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() in {"ND", "NONE", "N/A", "NA", "-"}:
        return None
    text = text.replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return None


def _fmt(value: float | None) -> str:
    if value is None:
        return "ND"
    if not math.isfinite(value):
        return "ND"
    return f"{value:.10g}"


def _norm(value: Any) -> str:
    return "".join(ch for ch in str(value or "").strip().lower() if ch.isalnum() or ch == "_")


HEADER_ALIASES = {
    "name": {"name", "designation", "designazione", "profilo", "sezione"},
    "L": {"l", "lm", "length", "lengthm", "lunghezza"},
    "h": {"h", "hcm", "hmm", "hm", "altezza", "height", "depth"},
    "b": {"b", "bcm", "bmm", "bm", "larghezza", "width"},
    "tw": {"tw", "twcm", "twmm", "twm", "t1", "spessoreanima", "webthickness"},
    "tf": {"tf", "tfcm", "tfmm", "tfm", "t2", "spessoreala", "flangethickness"},
    "t": {"t", "tcm", "tmm", "tm", "s", "thickness", "spessore"},
    "r": {"r", "rcm", "rmm", "radius", "rootradius"},
    "dext": {"dext", "de", "diameterext", "diametroesterno"},
    "dint": {"dint", "di", "diameterint", "diametrointerno"},
    "section_family": {"section_family", "family", "shape", "tipo", "type", "sectiontype"},
}


def _canonical_header(header: str) -> str:
    key = _norm(header)
    for canonical, aliases in HEADER_ALIASES.items():
        if key in aliases:
            return canonical
    for canonical in PROPERTY_COLUMNS:
        if key == _norm(canonical):
            return canonical
    return key


def _is_units_row(row: list[str]) -> bool:
    if not row:
        return False
    count = 0
    for cell in row:
        text = str(cell).strip().lower()
        if any(token in text for token in ("cm", "mm", "m]", "kn/m", "cm2", "cm3", "cm4", "[cm", "(cm")):
            count += 1
    return count >= max(2, len(row) // 4)


def _column_index(headers: list[str]) -> dict[str, int]:
    index: dict[str, int] = {}
    for idx, name in enumerate(headers):
        index[name] = idx
        canonical = _canonical_header(name)
        index.setdefault(canonical, idx)
    return index


def _get(row: list[str], index: dict[str, int], key: str) -> str | None:
    idx = index.get(key)
    if idx is None or idx >= len(row):
        return None
    return row[idx]


def _get_float(row: list[str], index: dict[str, int], key: str) -> float | None:
    return _to_float(_get(row, index, key))


def _set(row: list[str], index: dict[str, int], key: str, value: Any) -> None:
    idx = index[key]
    while len(row) <= idx:
        row.append("")
    row[idx] = str(value)


def _set_float(row: list[str], index: dict[str, int], key: str, value: float | None) -> None:
    _set(row, index, key, _fmt(value))


def _require_positive(values: dict[str, float | None], keys: tuple[str, ...]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in keys:
        value = values.get(key)
        if value is None or value <= 0:
            raise ValueError(f"Invalid or missing positive value for {key}.")
        out[key] = value
    return out


def _weight_from_area(area_cm2: float) -> float:
    return STEEL_UNIT_WEIGHT_KN_PER_M3 * area_cm2 * 1e-4


def _common_properties(area: float, iy: float, iz: float) -> dict[str, float]:
    return {
        "A": area,
        "Aeff": area,
        "Iy": iy,
        "Iz": iz,
        "iy": math.sqrt(iy / area) if area > 0 and iy > 0 else 0.0,
        "iz": math.sqrt(iz / area) if area > 0 and iz > 0 else 0.0,
        "w_kN_per_m": _weight_from_area(area),
    }


def _corroded_plate_dimensions(
    h: float,
    b: float,
    tf: float,
    tw: float,
    loss_cm: float,
    mode: str,
) -> tuple[float, float, float, float]:
    if mode == "all":
        return h - 2 * loss_cm, b - 2 * loss_cm, tf - 2 * loss_cm, tw - 2 * loss_cm
    if mode == "flanges":
        return h - 2 * loss_cm, b - 2 * loss_cm, tf - 2 * loss_cm, tw
    if mode == "web":
        return h, b, tf, tw - 2 * loss_cm
    raise ValueError(f"Unsupported corrosion application mode: {mode}")


def _calc_chs(values: dict[str, float | None], loss_cm: float) -> tuple[dict[str, float], str]:
    dims = _require_positive(values, ("dext", "t"))
    dext = dims["dext"] - 2 * loss_cm
    t = dims["t"] - 2 * loss_cm
    dint = dext - 2 * t

    if dext <= 0 or t <= 0 or dint < 0 or dint >= dext:
        raise ValueError("Corrosion consumes CHS wall thickness or diameter.")

    area = math.pi / 4 * (dext**2 - dint**2)
    inertia = math.pi / 64 * (dext**4 - dint**4)
    torsion = 2 * inertia
    wel = inertia / (dext / 2)
    wpl = (dext**3 - dint**3) / 6

    props = _common_properties(area, inertia, inertia)
    props.update(
        {
            "dext": dext,
            "t": t,
            "dint": dint,
            "Av_y": area,
            "Av_z": area,
            "It": torsion,
            "Iw": 0.0,
            "Wel_y": wel,
            "Wel_z": wel,
            "Wpl_y": wpl,
            "Wpl_z": wpl,
            "Weff_y": wel,
            "Weff_z": wel,
        }
    )
    return props, "valid"


def _calc_rhs(values: dict[str, float | None], loss_cm: float) -> tuple[dict[str, float], str]:
    dims = _require_positive(values, ("h", "b", "t"))
    h = dims["h"] - 2 * loss_cm
    b = dims["b"] - 2 * loss_cm
    t = dims["t"] - 2 * loss_cm
    hi = h - 2 * t
    bi = b - 2 * t

    if h <= 0 or b <= 0 or t <= 0 or hi < 0 or bi < 0:
        raise ValueError("Corrosion consumes RHS/SHS wall thickness or dimensions.")

    area = b * h - bi * hi
    iy = (b * h**3 - bi * hi**3) / 12
    iz = (h * b**3 - hi * bi**3) / 12
    wel_y = iy / (h / 2)
    wel_z = iz / (b / 2)
    wpl_y = (b * h**2 - bi * hi**2) / 4
    wpl_z = (h * b**2 - hi * bi**2) / 4
    torsion = 2 * t * ((h - t) ** 2) * ((b - t) ** 2) / max(h + b - 2 * t, 1e-12)

    props = _common_properties(area, iy, iz)
    props.update(
        {
            "h": h,
            "b": b,
            "t": t,
            "Av_y": area,
            "Av_z": area,
            "It": torsion,
            "Iw": 0.0,
            "Wel_y": wel_y,
            "Wel_z": wel_z,
            "Wpl_y": wpl_y,
            "Wpl_z": wpl_z,
            "Weff_y": wel_y,
            "Weff_z": wel_z,
        }
    )
    return props, "valid"


def _calc_i_section(values: dict[str, float | None], loss_cm: float, mode: str) -> tuple[dict[str, float], str]:
    dims = _require_positive(values, ("h", "b", "tf", "tw"))
    h, b, tf, tw = _corroded_plate_dimensions(
        dims["h"], dims["b"], dims["tf"], dims["tw"], loss_cm, mode
    )
    web_h = h - 2 * tf

    if h <= 0 or b <= 0 or tf <= 0 or tw <= 0 or web_h <= 0 or tw > b:
        raise ValueError("Corrosion consumes I-section plate thickness or dimensions.")

    area = 2 * b * tf + web_h * tw
    iy = (b * h**3 - (b - tw) * web_h**3) / 12
    iz = (2 * tf * b**3 + web_h * tw**3) / 12
    wel_y = iy / (h / 2)
    wel_z = iz / (b / 2)
    wpl_y = b * tf * (h - tf) + tw * web_h**2 / 4
    wpl_z = tf * b**2 / 2 + web_h * tw**2 / 4
    torsion = (2 * b * tf**3 + web_h * tw**3) / 3
    warping = tf * b**3 * (h - tf) ** 2 / 24

    props = _common_properties(area, iy, iz)
    props.update(
        {
            "h": h,
            "b": b,
            "tf": tf,
            "tw": tw,
            "Av_y": 2 * b * tf,
            "Av_z": web_h * tw,
            "It": torsion,
            "Iw": warping,
            "Wel_y": wel_y,
            "Wel_z": wel_z,
            "Wpl_y": wpl_y,
            "Wpl_z": wpl_z,
            "Weff_y": wel_y,
            "Weff_z": wel_z,
        }
    )
    return props, "valid"


def _rect_component(width: float, height: float, x: float, y: float, sign: float = 1.0) -> dict[str, float]:
    area = sign * width * height
    return {
        "A": area,
        "x": x,
        "y": y,
        "Icx": sign * width * height**3 / 12,
        "Icy": sign * height * width**3 / 12,
    }


def _composite_rect_properties(components: list[dict[str, float]]) -> tuple[float, float, float, float, float, float]:
    area = sum(c["A"] for c in components)
    if area <= 0:
        raise ValueError("Composite section has non-positive area.")
    x_bar = sum(c["A"] * c["x"] for c in components) / area
    y_bar = sum(c["A"] * c["y"] for c in components) / area
    iy = sum(c["Icx"] + c["A"] * (c["y"] - y_bar) ** 2 for c in components)
    iz = sum(c["Icy"] + c["A"] * (c["x"] - x_bar) ** 2 for c in components)
    iyz = sum(c["A"] * (c["x"] - x_bar) * (c["y"] - y_bar) for c in components)
    return area, x_bar, y_bar, iy, iz, iyz


def _principal_inertias(iy: float, iz: float, iyz: float) -> tuple[float, float, float]:
    avg = 0.5 * (iy + iz)
    radius = math.sqrt(max(0.0, (0.5 * (iy - iz)) ** 2 + iyz**2))
    iuu = avg + radius
    ivv = avg - radius
    theta = 0.5 * math.atan2(-2.0 * iyz, iz - iy)
    return iuu, ivv, math.degrees(theta)


def _calc_channel(values: dict[str, float | None], loss_cm: float, mode: str) -> tuple[dict[str, float], str]:
    dims = _require_positive(values, ("h", "b", "tf", "tw"))
    h, b, tf, tw = _corroded_plate_dimensions(
        dims["h"], dims["b"], dims["tf"], dims["tw"], loss_cm, mode
    )
    flange_projection = b - tw
    if h <= 0 or b <= 0 or tf <= 0 or tw <= 0 or flange_projection <= 0 or h <= 2 * tf:
        raise ValueError("Corrosion consumes channel plate thickness or dimensions.")

    components = [
        _rect_component(tw, h, tw / 2, h / 2),
        _rect_component(flange_projection, tf, tw + flange_projection / 2, tf / 2),
        _rect_component(flange_projection, tf, tw + flange_projection / 2, h - tf / 2),
    ]
    area, x_bar, y_bar, iy, iz, iyz = _composite_rect_properties(components)
    iuu, ivv, theta_deg = _principal_inertias(iy, iz, iyz)
    wel_y = min(iy / max(y_bar, 1e-12), iy / max(h - y_bar, 1e-12))
    wel_z = min(iz / max(x_bar, 1e-12), iz / max(b - x_bar, 1e-12))
    torsion = (h * tw**3 + 2 * flange_projection * tf**3) / 3

    props = _common_properties(area, iy, iz)
    props.update(
        {
            "h": h,
            "b": b,
            "tf": tf,
            "tw": tw,
            "Av_y": 2 * flange_projection * tf,
            "Av_z": h * tw,
            "It": torsion,
            "Iuu": iuu,
            "Ivv": ivv,
            "Iyz": iyz,
            "theta_principal_deg": theta_deg,
            "Wel_y": wel_y,
            "Wel_z": wel_z,
            "Wpl_y": wel_y,
            "Wpl_z": wel_z,
            "Weff_y": wel_y,
            "Weff_z": wel_z,
        }
    )
    return props, "valid"


def _calc_angle(values: dict[str, float | None], loss_cm: float, mode: str) -> tuple[dict[str, float], str]:
    h = values.get("h")
    b = values.get("b")
    t = values.get("t") if values.get("t") is not None else values.get("tf") or values.get("tw")
    dims = _require_positive({"h": h, "b": b, "t": t}, ("h", "b", "t"))

    h = dims["h"]
    b = dims["b"]
    t = dims["t"]
    if mode == "all":
        h -= 2 * loss_cm
        b -= 2 * loss_cm
        t -= 2 * loss_cm
    elif mode == "flanges":
        b -= 2 * loss_cm
        t -= 2 * loss_cm
    elif mode == "web":
        h -= 2 * loss_cm
        t -= 2 * loss_cm

    if h <= 0 or b <= 0 or t <= 0 or t > h or t > b:
        raise ValueError("Corrosion consumes angle plate thickness or dimensions.")

    components = [
        _rect_component(t, h, t / 2, h / 2),
        _rect_component(b, t, b / 2, t / 2),
        _rect_component(t, t, t / 2, t / 2, sign=-1.0),
    ]
    area, x_bar, y_bar, iy, iz, iyz = _composite_rect_properties(components)
    iuu, ivv, theta_deg = _principal_inertias(iy, iz, iyz)
    wel_y = min(iy / max(y_bar, 1e-12), iy / max(h - y_bar, 1e-12))
    wel_z = min(iz / max(x_bar, 1e-12), iz / max(b - x_bar, 1e-12))
    torsion = (h * t**3 + b * t**3 - t * t**3) / 3

    props = _common_properties(area, iy, iz)
    props.update(
        {
            "h": h,
            "b": b,
            "t": t,
            "tf": t if values.get("tf") is not None else None,
            "tw": t if values.get("tw") is not None else None,
            "Av_y": area,
            "Av_z": area,
            "It": torsion,
            "Iuu": iuu,
            "Ivv": ivv,
            "Iyz": iyz,
            "theta_principal_deg": theta_deg,
            "Wel_y": wel_y,
            "Wel_z": wel_z,
            "Wpl_y": wel_y,
            "Wpl_z": wel_z,
            "Weff_y": wel_y,
            "Weff_z": wel_z,
        }
    )
    return props, "valid"


def _family_from_row(row: list[str], index: dict[str, int]) -> str:
    family = str(_get(row, index, "section_family") or "").strip().upper().replace(" ", "")
    name = str(_get(row, index, "name") or "").strip().upper().replace(" ", "")
    if family and family != "-":
        return family
    if name.startswith("IPE"):
        return "IPE"
    if name.startswith(("HEA", "HEB", "HEM", "HE ")):
        return "HE"
    if name.startswith(("UPN", "UPE", "C")):
        return "C"
    if name.startswith("L"):
        return "L"
    if name.startswith("Ø") or name.startswith("O "):
        return "CHS"
    return family


def _calculate_section_properties(
    row: list[str],
    index: dict[str, int],
    loss_cm: float,
    mode: str,
) -> tuple[dict[str, float], str]:
    values = {key: _get_float(row, index, key) for key in GEOMETRY_COLUMNS}
    family = _family_from_row(row, index)

    if family in {"CHS", "PIPE", "TUBE_CIRCULAR"}:
        return _calc_chs(values, loss_cm)
    if family in {"RHS", "SHS", "HSS"}:
        return _calc_rhs(values, loss_cm)
    if family.startswith(("IPE", "HE", "IPN", "I")):
        return _calc_i_section(values, loss_cm, mode)
    if family in {"UPN", "UPE", "C", "CHANNEL"}:
        return _calc_channel(values, loss_cm, mode)
    if family in {"L", "ANGLE"}:
        return _calc_angle(values, loss_cm, mode)

    raise ValueError(f"Unsupported section family '{family}'.")


def corrode_section_catalog_csv(
    input_csv_path: str | Path,
    output_csv_path: str | Path,
    corrosion_loss_cm: float,
    application_mode: str = "all",
    delimiter: str = ";",
    encoding: str = "utf-8",
    drop_invalid: bool = True,
) -> dict[str, int | float | str]:
    """
    Apply a uniform corrosion loss to a section catalog and save a modified CSV.

    The input/output catalog remains compatible with import_section.py. Values are
    assumed to be in cm-based catalog units, matching the current sections CSV.
    """
    if corrosion_loss_cm < 0:
        raise ValueError("corrosion_loss_cm must be non-negative.")
    application_mode = application_mode.strip().lower()
    if application_mode not in SUPPORTED_MODES:
        raise ValueError(f"application_mode must be one of: {', '.join(sorted(SUPPORTED_MODES))}.")

    input_csv_path = Path(input_csv_path)
    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    with input_csv_path.open(newline="", encoding=encoding) as csvfile:
        rows = list(csv.reader(csvfile, delimiter=delimiter))

    if not rows:
        raise ValueError("Empty section catalog.")

    headers = list(rows[0])
    units_row = list(rows[1]) if len(rows) > 1 and _is_units_row(rows[1]) else None
    data_start = 2 if units_row is not None else 1

    initial_index = _column_index(headers)
    for col in PROPERTY_COLUMNS:
        if col not in initial_index:
            headers.append(col)
            if units_row is not None:
                units_row.append("")

    extra_columns = [
        "corrosion_loss_cm",
        "corrosion_application_mode",
        "corrosion_status",
    ]
    initial_index = _column_index(headers)
    for col in TRACKED_COLUMNS:
        if col in initial_index:
            extra_columns.append(f"{col}_original")

    for col in extra_columns:
        if col not in headers:
            headers.append(col)
            if units_row is not None:
                units_row.append("")

    index = _column_index(headers)
    output_rows = [headers]
    if units_row is not None:
        output_rows.append(units_row)

    processed = 0
    valid = 0
    invalid = 0
    unsupported = 0

    for source_row in rows[data_start:]:
        if not source_row or not any(str(cell).strip() for cell in source_row):
            continue
        row = list(source_row)
        while len(row) < len(headers):
            row.append("")

        processed += 1
        for col in TRACKED_COLUMNS:
            if col in index and f"{col}_original" in index:
                _set(row, index, f"{col}_original", _get(row, index, col) or "")

        _set_float(row, index, "corrosion_loss_cm", corrosion_loss_cm)
        _set(row, index, "corrosion_application_mode", application_mode)

        try:
            updates, status = _calculate_section_properties(row, index, corrosion_loss_cm, application_mode)
            for key, value in updates.items():
                if key in index:
                    if key in MONOTONIC_PROPERTY_COLUMNS and f"{key}_original" in index:
                        original_value = _get_float(row, index, f"{key}_original")
                        if original_value is not None and value is not None:
                            value = min(value, original_value)
                    _set_float(row, index, key, value)
            _set(row, index, "corrosion_status", status)
            valid += 1
            output_rows.append(row)
        except ValueError as exc:
            message = str(exc)
            if "Unsupported section family" in message:
                unsupported += 1
            else:
                invalid += 1
            if not drop_invalid:
                _set(row, index, "corrosion_status", message)
                output_rows.append(row)

    with output_csv_path.open("w", newline="", encoding=encoding) as csvfile:
        writer = csv.writer(csvfile, delimiter=delimiter)
        writer.writerows(output_rows)

    return {
        "input_csv": str(input_csv_path),
        "output_csv": str(output_csv_path),
        "corrosion_loss_cm": corrosion_loss_cm,
        "application_mode": application_mode,
        "processed": processed,
        "valid": valid,
        "invalid": invalid,
        "unsupported": unsupported,
        "written": len(output_rows) - 1 - (1 if units_row is not None else 0),
    }
