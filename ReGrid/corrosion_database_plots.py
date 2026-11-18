import csv
from collections import defaultdict
from pathlib import Path
from typing import Any


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip().replace(",", ".")
    if not text or text.upper() in {"ND", "NONE", "N/A", "NA", "-"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _pick(row: dict[str, Any], *names: str) -> Any:
    normalized = {_norm(k): v for k, v in row.items()}
    for name in names:
        if name in row:
            return row[name]
        key = _norm(name)
        if key in normalized:
            return normalized[key]
    return None


def _norm(value: Any) -> str:
    return "".join(ch for ch in str(value or "").strip().lower() if ch.isalnum())


def _profile_sort_key(name: str) -> tuple[str, float, str]:
    digits = ""
    for ch in name:
        if ch.isdigit() or ch == ".":
            digits += ch
        elif digits:
            break
    return ("".join(ch for ch in name if not ch.isdigit()), _to_float(digits) or 0.0, name)


def _load_profile_stats(csv_path: str | Path, delimiter: str = ";") -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "count": 0,
            "length_total_m": 0.0,
            "length_values": [],
            "area_loss_values": [],
            "area_loss_pct_values": [],
        }
    )

    with Path(csv_path).open(newline="", encoding="utf-8-sig") as csvfile:
        reader = csv.DictReader(csvfile, delimiter=delimiter)
        for row in reader:
            name = str(_pick(row, "Name", "name") or "").strip()
            if not name or name == "-":
                continue

            length = _to_float(_pick(row, "L[m]", "L", "length"))
            area = _to_float(_pick(row, "A", "area"))
            area_original = _to_float(_pick(row, "A_original", "area_original"))
            if area is None:
                area = _area_from_l_geometry(
                    _to_float(_pick(row, "h [cm]", "h")),
                    _to_float(_pick(row, "b [cm]", "b")),
                    _to_float(_pick(row, "tw [cm]", "tw")),
                    _to_float(_pick(row, "tf [cm]", "tf")),
                )
            if area_original is None:
                area_original = _area_from_l_geometry(
                    _to_float(_pick(row, "h_original")),
                    _to_float(_pick(row, "b_original")),
                    _to_float(_pick(row, "tw_original")),
                    _to_float(_pick(row, "tf_original")),
                )

            group = groups[name]
            group["count"] += 1
            if length is not None:
                group["length_total_m"] += length
                group["length_values"].append(length)
            if area is not None and area_original is not None:
                loss = max(0.0, area_original - area)
                group["area_loss_values"].append(loss)
                if area_original > 0:
                    group["area_loss_pct_values"].append(100.0 * loss / area_original)

    stats = []
    for name, values in groups.items():
        length_values = values["length_values"]
        area_loss_values = values["area_loss_values"]
        area_loss_pct_values = values["area_loss_pct_values"]
        stats.append(
            {
                "name": name,
                "count": values["count"],
                "length_total_m": values["length_total_m"],
                "length_mean_m": (
                    sum(length_values) / len(length_values) if length_values else 0.0
                ),
                "area_loss_mean_cm2": (
                    sum(area_loss_values) / len(area_loss_values)
                    if area_loss_values
                    else 0.0
                ),
                "area_loss_mean_pct": (
                    sum(area_loss_pct_values) / len(area_loss_pct_values)
                    if area_loss_pct_values
                    else 0.0
                ),
            }
        )

    return sorted(stats, key=lambda item: _profile_sort_key(item["name"]))


def _area_from_l_geometry(
    h: float | None,
    b: float | None,
    tw: float | None,
    tf: float | None,
) -> float | None:
    if h is None or b is None or tw is None or tf is None:
        return None
    if h <= 0 or b <= 0 or tw <= 0 or tf <= 0:
        return None
    return h * tw + b * tf - tw * tf


def _bar_svg(
    stats: list[dict[str, Any]],
    output_path: str | Path,
    *,
    title: str,
    left_key: str,
    left_label: str,
    right_key: str | None = None,
    right_label: str | None = None,
    left_color: str = "#3577b8",
    right_color: str = "#d9822b",
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    width = max(980, 70 + 34 * max(1, len(stats)))
    height = 560
    ml, mr, mt, mb = 80, 80, 60, 130
    plot_w = width - ml - mr
    plot_h = height - mt - mb
    n = max(1, len(stats))
    step = plot_w / n
    bar_w = min(24.0, step * (0.32 if right_key else 0.55))

    left_max = max([float(s[left_key]) for s in stats] + [1.0])
    right_max = max([float(s[right_key]) for s in stats] + [1.0]) if right_key else 1.0

    def y_left(value: float) -> float:
        return mt + plot_h * (1.0 - value / left_max)

    def y_right(value: float) -> float:
        return mt + plot_h * (1.0 - value / right_max)

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;font-size:13px;fill:#222}.title{font-size:20px;font-weight:bold}.small{font-size:11px}.legend{font-size:13px}</style>',
        f'<text class="title" x="{width / 2:.1f}" y="32" text-anchor="middle">{title}</text>',
    ]

    for i in range(6):
        frac = i / 5
        y = mt + plot_h * (1 - frac)
        left_value = left_max * frac
        lines.append(f'<line x1="{ml}" y1="{y:.2f}" x2="{width - mr}" y2="{y:.2f}" stroke="#e6e6e6"/>')
        lines.append(f'<text class="small" x="{ml - 8}" y="{y + 4:.2f}" text-anchor="end">{left_value:.1f}</text>')
        if right_key:
            right_value = right_max * frac
            lines.append(f'<text class="small" x="{width - mr + 8}" y="{y + 4:.2f}">{right_value:.1f}</text>')

    for i, stat in enumerate(stats):
        x_center = ml + step * (i + 0.5)
        left_value = float(stat[left_key])
        y = y_left(left_value)
        x = x_center - (bar_w if right_key else bar_w / 2)
        lines.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{mt + plot_h - y:.2f}" fill="{left_color}"/>')
        lines.append(f'<text class="small" x="{x + bar_w / 2:.2f}" y="{y - 4:.2f}" text-anchor="middle">{left_value:.1f}</text>')

        if right_key:
            right_value = float(stat[right_key])
            y2 = y_right(right_value)
            x2 = x_center + 2
            lines.append(f'<rect x="{x2:.2f}" y="{y2:.2f}" width="{bar_w:.2f}" height="{mt + plot_h - y2:.2f}" fill="{right_color}"/>')

        label = stat["name"]
        lines.append(
            f'<text class="small" x="{x_center:.2f}" y="{height - mb + 18}" text-anchor="end" transform="rotate(-55 {x_center:.2f} {height - mb + 18})">{label}</text>'
        )

    lines.extend(
        [
            f'<line x1="{ml}" y1="{mt + plot_h}" x2="{width - mr}" y2="{mt + plot_h}" stroke="#333"/>',
            f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt + plot_h}" stroke="#333"/>',
            f'<line x1="{width - mr}" y1="{mt}" x2="{width - mr}" y2="{mt + plot_h}" stroke="#333"/>',
            f'<text x="{22}" y="{mt + plot_h / 2:.1f}" text-anchor="middle" transform="rotate(-90 22 {mt + plot_h / 2:.1f})">{left_label}</text>',
        ]
    )
    if right_key and right_label:
        lines.append(
            f'<text x="{width - 22}" y="{mt + plot_h / 2:.1f}" text-anchor="middle" transform="rotate(90 {width - 22} {mt + plot_h / 2:.1f})">{right_label}</text>'
        )

    legend_x = ml
    legend_y = 48
    lines.append(f'<rect x="{legend_x}" y="{legend_y}" width="14" height="14" fill="{left_color}"/>')
    lines.append(f'<text class="legend" x="{legend_x + 20}" y="{legend_y + 12}">{left_label}</text>')
    if right_key and right_label:
        lines.append(f'<rect x="{legend_x + 190}" y="{legend_y}" width="14" height="14" fill="{right_color}"/>')
        lines.append(f'<text class="legend" x="{legend_x + 210}" y="{legend_y + 12}">{right_label}</text>')

    lines.append("</svg>")
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def _count_area_loss_combo_svg(
    stats: list[dict[str, Any]],
    output_path: str | Path,
    *,
    title: str = "Recovered database and mean area loss",
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    width = max(1040, 70 + 34 * max(1, len(stats)))
    height = 600
    ml, mr, mt, mb = 80, 90, 65, 145
    plot_w = width - ml - mr
    plot_h = height - mt - mb
    n = max(1, len(stats))
    step = plot_w / n
    bar_w = min(24.0, step * 0.52)

    count_max = max([float(s["count"]) for s in stats] + [1.0])
    loss_max = max([float(s["area_loss_mean_pct"]) for s in stats] + [1.0])
    loss_max *= 1.12

    def x_at(i: int) -> float:
        return ml + step * (i + 0.5)

    def y_count(value: float) -> float:
        return mt + plot_h * (1.0 - value / count_max)

    def y_loss(value: float) -> float:
        return mt + plot_h * (1.0 - value / loss_max)

    line_points = " ".join(
        f"{x_at(i):.2f},{y_loss(float(stat['area_loss_mean_pct'])):.2f}"
        for i, stat in enumerate(stats)
    )

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;font-size:13px;fill:#222}.title{font-size:20px;font-weight:bold}.small{font-size:11px}.legend{font-size:13px}</style>',
        f'<text class="title" x="{width / 2:.1f}" y="34" text-anchor="middle">{title}</text>',
    ]

    for i in range(6):
        frac = i / 5
        y = mt + plot_h * (1 - frac)
        count_value = count_max * frac
        loss_value = loss_max * frac
        lines.append(f'<line x1="{ml}" y1="{y:.2f}" x2="{width - mr}" y2="{y:.2f}" stroke="#e6e6e6"/>')
        lines.append(f'<text class="small" x="{ml - 8}" y="{y + 4:.2f}" text-anchor="end">{count_value:.0f}</text>')
        lines.append(f'<text class="small" x="{width - mr + 8}" y="{y + 4:.2f}">{loss_value:.1f}</text>')

    for i, stat in enumerate(stats):
        x_center = x_at(i)
        count = float(stat["count"])
        y = y_count(count)
        x = x_center - bar_w / 2
        lines.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{mt + plot_h - y:.2f}" fill="#3577b8"/>')
        lines.append(f'<text class="small" x="{x_center:.2f}" y="{y - 4:.2f}" text-anchor="middle">{count:.0f}</text>')

        label_y = height - mb + 20
        lines.append(
            f'<text class="small" x="{x_center:.2f}" y="{label_y}" text-anchor="end" transform="rotate(-55 {x_center:.2f} {label_y})">{stat["name"]}</text>'
        )

    lines.append(
        f'<polyline points="{line_points}" fill="none" stroke="#c9573b" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>'
    )
    for i, stat in enumerate(stats):
        x = x_at(i)
        y = y_loss(float(stat["area_loss_mean_pct"]))
        value = float(stat["area_loss_mean_pct"])
        lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.2" fill="#c9573b" stroke="white" stroke-width="1.2"/>')
        lines.append(f'<text class="small" x="{x:.2f}" y="{y - 8:.2f}" text-anchor="middle">{value:.1f}%</text>')

    lines.extend(
        [
            f'<line x1="{ml}" y1="{mt + plot_h}" x2="{width - mr}" y2="{mt + plot_h}" stroke="#333"/>',
            f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt + plot_h}" stroke="#333"/>',
            f'<line x1="{width - mr}" y1="{mt}" x2="{width - mr}" y2="{mt + plot_h}" stroke="#333"/>',
            f'<text x="22" y="{mt + plot_h / 2:.1f}" text-anchor="middle" transform="rotate(-90 22 {mt + plot_h / 2:.1f})">Number of elements</text>',
            f'<text x="{width - 22}" y="{mt + plot_h / 2:.1f}" text-anchor="middle" transform="rotate(90 {width - 22} {mt + plot_h / 2:.1f})">Mean area loss [%]</text>',
            f'<rect x="{ml}" y="49" width="14" height="14" fill="#3577b8"/>',
            f'<text class="legend" x="{ml + 20}" y="61">Number of elements</text>',
            f'<line x1="{ml + 190}" y1="56" x2="{ml + 230}" y2="56" stroke="#c9573b" stroke-width="3"/>',
            f'<circle cx="{ml + 210}" cy="56" r="4" fill="#c9573b"/>',
            f'<text class="legend" x="{ml + 240}" y="61">Mean area loss [%]</text>',
            "</svg>",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def _normalized_database_svg(
    stats: list[dict[str, Any]],
    output_path: str | Path,
    *,
    title: str = "Normalized recovered database metrics",
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    width = max(1040, 70 + 38 * max(1, len(stats)))
    height = 610
    ml, mr, mt, mb = 78, 36, 68, 150
    plot_w = width - ml - mr
    plot_h = height - mt - mb
    n = max(1, len(stats))
    step = plot_w / n
    bar_w = min(16.0, step * 0.25)

    count_max = max([float(s["count"]) for s in stats] + [1.0])
    length_max = max([float(s["length_total_m"]) for s in stats] + [1.0])
    loss_max = max([float(s["area_loss_mean_pct"]) for s in stats] + [1.0])

    def x_at(i: int) -> float:
        return ml + step * (i + 0.5)

    def y_norm(value: float) -> float:
        return mt + plot_h * (1.0 - value)

    def color_scale(value: float, base: tuple[int, int, int]) -> str:
        value = max(0.0, min(1.0, value))
        # Blend from a pale tint to the base color.
        pale = (232, 238, 244)
        rgb = tuple(round(pale[i] + (base[i] - pale[i]) * value) for i in range(3))
        return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"

    normalized = []
    for stat in stats:
        count_norm = float(stat["count"]) / count_max
        length_norm = float(stat["length_total_m"]) / length_max
        loss_norm = float(stat["area_loss_mean_pct"]) / loss_max
        normalized.append((count_norm, length_norm, loss_norm))

    line_points = " ".join(
        f"{x_at(i):.2f},{y_norm(values[2]):.2f}" for i, values in enumerate(normalized)
    )

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;font-size:13px;fill:#222}.title{font-size:20px;font-weight:bold}.small{font-size:11px}.legend{font-size:13px}.faint{fill:#666}</style>',
        f'<text class="title" x="{width / 2:.1f}" y="34" text-anchor="middle">{title}</text>',
    ]

    for i in range(6):
        value = i / 5
        y = y_norm(value)
        lines.append(f'<line x1="{ml}" y1="{y:.2f}" x2="{width - mr}" y2="{y:.2f}" stroke="#e6e6e6"/>')
        lines.append(f'<text class="small" x="{ml - 8}" y="{y + 4:.2f}" text-anchor="end">{value:.1f}</text>')

    for i, (stat, values) in enumerate(zip(stats, normalized)):
        count_norm, length_norm, loss_norm = values
        x_center = x_at(i)

        count_x = x_center - bar_w - 1.5
        count_y = y_norm(count_norm)
        count_color = color_scale(count_norm, (53, 119, 184))
        lines.append(f'<rect x="{count_x:.2f}" y="{count_y:.2f}" width="{bar_w:.2f}" height="{mt + plot_h - count_y:.2f}" fill="{count_color}"/>')

        length_x = x_center + 1.5
        length_y = y_norm(length_norm)
        length_color = color_scale(length_norm, (65, 150, 105))
        lines.append(f'<rect x="{length_x:.2f}" y="{length_y:.2f}" width="{bar_w:.2f}" height="{mt + plot_h - length_y:.2f}" fill="{length_color}"/>')

        label_y = height - mb + 20
        lines.append(
            f'<text class="small" x="{x_center:.2f}" y="{label_y}" text-anchor="end" transform="rotate(-55 {x_center:.2f} {label_y})">{stat["name"]}</text>'
        )

        # Compact real-value labels, useful when reading the normalized plot alone.
        if step >= 42:
            lines.append(f'<text class="small faint" x="{x_center:.2f}" y="{mt + plot_h + 34}" text-anchor="middle">{float(stat["length_total_m"]):.0f} m</text>')

    lines.append(
        f'<polyline points="{line_points}" fill="none" stroke="#c9573b" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>'
    )
    for i, (stat, values) in enumerate(zip(stats, normalized)):
        x = x_at(i)
        y = y_norm(values[2])
        lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.2" fill="#c9573b" stroke="white" stroke-width="1.2"/>')

    lines.extend(
        [
            f'<line x1="{ml}" y1="{mt + plot_h}" x2="{width - mr}" y2="{mt + plot_h}" stroke="#333"/>',
            f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt + plot_h}" stroke="#333"/>',
            f'<text x="22" y="{mt + plot_h / 2:.1f}" text-anchor="middle" transform="rotate(-90 22 {mt + plot_h / 2:.1f})">Normalized value [-]</text>',
            f'<rect x="{ml}" y="50" width="14" height="14" fill="#3577b8"/>',
            f'<text class="legend" x="{ml + 20}" y="62">n / nmax</text>',
            f'<rect x="{ml + 110}" y="50" width="14" height="14" fill="#419669"/>',
            f'<text class="legend" x="{ml + 130}" y="62">Ltot / Ltot,max</text>',
            f'<line x1="{ml + 270}" y1="57" x2="{ml + 310}" y2="57" stroke="#c9573b" stroke-width="3"/>',
            f'<circle cx="{ml + 290}" cy="57" r="4" fill="#c9573b"/>',
            f'<text class="legend" x="{ml + 320}" y="62">area loss % / max</text>',
            "</svg>",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def save_corroded_database_summary_csv(
    input_csv_path: str | Path,
    output_csv_path: str | Path,
    delimiter: str = ";",
) -> Path:
    stats = _load_profile_stats(input_csv_path, delimiter=delimiter)
    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with output_csv_path.open("w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "name",
            "count",
            "length_total_m",
            "length_mean_m",
            "area_loss_mean_cm2",
            "area_loss_mean_pct",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(stats)
    return output_csv_path


def plot_corroded_database_fig12_style(
    input_csv_path: str | Path,
    output_dir: str | Path,
    prefix: str = "corroded_database",
    delimiter: str = ";",
) -> dict[str, Path]:
    stats = _load_profile_stats(input_csv_path, delimiter=delimiter)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = save_corroded_database_summary_csv(
        input_csv_path,
        output_dir / f"{prefix}_summary.csv",
        delimiter=delimiter,
    )
    combined_svg = _count_area_loss_combo_svg(
        stats,
        output_dir / f"{prefix}_count_area_loss.svg",
    )
    normalized_svg = _normalized_database_svg(
        stats,
        output_dir / f"{prefix}_normalized_metrics.svg",
    )
    inventory_svg = _bar_svg(
        stats,
        output_dir / f"{prefix}_inventory.svg",
        title="Recovered database inventory",
        left_key="count",
        left_label="Number of elements",
        right_key="length_total_m",
        right_label="Total length [m]",
    )
    area_loss_svg = _bar_svg(
        stats,
        output_dir / f"{prefix}_area_loss.svg",
        title="Mean cross-sectional area loss",
        left_key="area_loss_mean_pct",
        left_label="Mean area loss [%]",
        right_key="area_loss_mean_cm2",
        right_label="Mean area loss [cm2]",
        left_color="#7f5aa2",
        right_color="#c9573b",
    )

    return {
        "summary_csv": summary_csv,
        "combined_svg": combined_svg,
        "normalized_svg": normalized_svg,
        "inventory_svg": inventory_svg,
        "area_loss_svg": area_loss_svg,
    }
