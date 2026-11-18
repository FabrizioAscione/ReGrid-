import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from transparent_plots import configure_transparent_matplotlib_saves

configure_transparent_matplotlib_saves()


# Upper bounds for carbon steel first-year corrosion rates from ISO 9223.
# Values are stored in mm/year and converted to cm/year during calculations.
CORROSION_RATES_MM_PER_YEAR = {
    "C1": 0.0013,
    "C2": 0.025,
    "C3": 0.050,
    "C4": 0.080,
    "C5": 0.200,
    "CX": 0.700,
}

DEFAULT_H_CORR = 0.575


@dataclass(frozen=True)
class CorrosionPeriod:
    start_year: float
    end_year: float
    category: str


def corrosion_loss_for_category_cm(
    age_years: float,
    category: str,
    h_corr: float = DEFAULT_H_CORR,
) -> float:
    """Return corrosion loss in cm for one constant corrosivity category."""
    if age_years < 0:
        raise ValueError("age_years must be non-negative.")

    category = category.upper()
    if category not in CORROSION_RATES_MM_PER_YEAR:
        valid = ", ".join(CORROSION_RATES_MM_PER_YEAR)
        raise ValueError(f"Unknown corrosion category '{category}'. Use one of: {valid}.")

    r_corr_cm = CORROSION_RATES_MM_PER_YEAR[category] / 10.0

    if age_years <= 20:
        return r_corr_cm * (age_years**h_corr)

    return r_corr_cm * (
        (20.0**h_corr)
        + h_corr * (20.0 ** (h_corr - 1.0)) * (age_years - 20.0)
    )


def build_three_period_scenario(
    end_years: Sequence[float],
    categories: Sequence[str],
    start_year: float = 0.0,
) -> list[CorrosionPeriod]:
    """Build a three-period corrosion scenario from period end years and classes."""
    if len(end_years) != 3:
        raise ValueError("end_years must contain exactly three values.")
    if len(categories) != 3:
        raise ValueError("categories must contain exactly three values.")

    periods = []
    current_start = float(start_year)
    for end_year, category in zip(end_years, categories):
        end_year = float(end_year)
        if end_year <= current_start:
            raise ValueError("Each period end year must be greater than its start year.")
        periods.append(CorrosionPeriod(current_start, end_year, category.upper()))
        current_start = end_year

    validate_corrosion_periods(periods)
    return periods


def validate_corrosion_periods(periods: Sequence[CorrosionPeriod]) -> None:
    if not periods:
        raise ValueError("At least one corrosion period is required.")

    previous_end = None
    for period in periods:
        if period.end_year <= period.start_year:
            raise ValueError("Each corrosion period must have end_year > start_year.")
        if period.category.upper() not in CORROSION_RATES_MM_PER_YEAR:
            valid = ", ".join(CORROSION_RATES_MM_PER_YEAR)
            raise ValueError(
                f"Unknown corrosion category '{period.category}'. Use one of: {valid}."
            )
        if previous_end is not None and abs(period.start_year - previous_end) > 1e-9:
            raise ValueError("Corrosion periods must be contiguous.")
        previous_end = period.end_year


def corrosion_curve_cm(
    periods: Sequence[CorrosionPeriod],
    step_years: float = 0.25,
    h_corr: float = DEFAULT_H_CORR,
) -> tuple[list[float], list[float], list[str]]:
    """
    Calculate a continuous corrosion curve for a piecewise exposure scenario.

    The category change follows the paper logic: when a new category starts,
    the curve keeps the loss already reached and then adopts the increment
    associated with the new category curve at the same structural age.
    """
    if step_years <= 0:
        raise ValueError("step_years must be positive.")

    periods = list(periods)
    validate_corrosion_periods(periods)

    years: list[float] = []
    losses_cm: list[float] = []
    active_categories: list[str] = []
    accumulated_at_start = 0.0

    for period_index, period in enumerate(periods):
        start = period.start_year
        end = period.end_year
        category = period.category.upper()
        base_at_start = corrosion_loss_for_category_cm(start, category, h_corr)

        t = start
        if period_index > 0:
            t += step_years

        while t <= end + 1e-9:
            base_at_t = corrosion_loss_for_category_cm(t, category, h_corr)
            years.append(round(t, 10))
            losses_cm.append(accumulated_at_start + base_at_t - base_at_start)
            active_categories.append(category)
            t += step_years

        base_at_end = corrosion_loss_for_category_cm(end, category, h_corr)
        accumulated_at_start += base_at_end - base_at_start

    return years, losses_cm, active_categories


def plot_corrosion_curve(
    periods: Sequence[CorrosionPeriod],
    output_path: str | Path = "outputs/corrosion_curve.png",
    step_years: float = 0.25,
    h_corr: float = DEFAULT_H_CORR,
    show: bool = False,
) -> Path:
    """Plot the resulting corrosion curve in cm and save it to output_path."""
    years, losses_cm, active_categories = corrosion_curve_cm(
        periods=periods,
        step_years=step_years,
        h_corr=h_corr,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        if output_path.suffix.lower() != ".svg":
            output_path = output_path.with_suffix(".svg")
        _write_corrosion_curve_svg(
            years=years,
            losses_cm=losses_cm,
            periods=periods,
            output_path=output_path,
        )
        return output_path

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(years, losses_cm, color="#1f77b4", linewidth=2.0)

    for period in periods:
        ax.axvline(period.start_year, color="0.82", linewidth=0.8)
        mid_year = 0.5 * (period.start_year + period.end_year)
        y_top = max(losses_cm) if losses_cm else 0.0
        ax.text(
            mid_year,
            y_top * 0.95 if y_top > 0 else 0.0,
            period.category.upper(),
            ha="center",
            va="top",
            fontsize=10,
        )
    ax.axvline(periods[-1].end_year, color="0.82", linewidth=0.8)

    ax.set_xlabel("Time [years]")
    ax.set_ylabel("Corrosion loss [cm]")
    ax.set_title("Corrosion scenario")
    ax.grid(True, linewidth=0.5, alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return output_path


def save_corrosion_curve_data_csv(
    periods: Sequence[CorrosionPeriod],
    output_path: str | Path = "outputs/corrosion_curve_data.csv",
    step_years: float = 0.25,
    h_corr: float = DEFAULT_H_CORR,
) -> Path:
    """Save the calculated corrosion curve points to a CSV file."""
    years, losses_cm, active_categories = corrosion_curve_cm(
        periods=periods,
        step_years=step_years,
        h_corr=h_corr,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(("year", "corrosion_loss_cm", "corrosion_category"))
        for year, loss_cm, category in zip(years, losses_cm, active_categories):
            writer.writerow((year, loss_cm, category))

    return output_path


def _write_corrosion_curve_svg(
    years: Sequence[float],
    losses_cm: Sequence[float],
    periods: Sequence[CorrosionPeriod],
    output_path: Path,
) -> None:
    width = 900
    height = 520
    margin_left = 80
    margin_right = 30
    margin_top = 45
    margin_bottom = 70
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    x_min = min(years)
    x_max = max(years)
    y_min = 0.0
    y_max = max(losses_cm) if losses_cm else 1.0
    if y_max <= 0:
        y_max = 1.0

    def sx(x: float) -> float:
        return margin_left + (x - x_min) / (x_max - x_min) * plot_width

    def sy(y: float) -> float:
        return margin_top + (1.0 - (y - y_min) / (y_max - y_min)) * plot_height

    points = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in zip(years, losses_cm))

    x_ticks = 5
    y_ticks = 5
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;font-size:14px;fill:#222}.small{font-size:12px}.title{font-size:20px;font-weight:bold}</style>',
        f'<text class="title" x="{width / 2:.1f}" y="28" text-anchor="middle">Corrosion scenario</text>',
    ]

    for i in range(x_ticks + 1):
        value = x_min + (x_max - x_min) * i / x_ticks
        x = sx(value)
        lines.append(f'<line x1="{x:.2f}" y1="{margin_top}" x2="{x:.2f}" y2="{height - margin_bottom}" stroke="#e5e5e5" stroke-width="1"/>')
        lines.append(f'<text class="small" x="{x:.2f}" y="{height - margin_bottom + 24}" text-anchor="middle">{value:.0f}</text>')

    for i in range(y_ticks + 1):
        value = y_min + (y_max - y_min) * i / y_ticks
        y = sy(value)
        lines.append(f'<line x1="{margin_left}" y1="{y:.2f}" x2="{width - margin_right}" y2="{y:.2f}" stroke="#e5e5e5" stroke-width="1"/>')
        lines.append(f'<text class="small" x="{margin_left - 10}" y="{y + 4:.2f}" text-anchor="end">{value:.4f}</text>')

    for period in periods:
        x = sx(period.start_year)
        mid = sx(0.5 * (period.start_year + period.end_year))
        lines.append(f'<line x1="{x:.2f}" y1="{margin_top}" x2="{x:.2f}" y2="{height - margin_bottom}" stroke="#bdbdbd" stroke-width="1"/>')
        lines.append(f'<text class="small" x="{mid:.2f}" y="{margin_top + 18}" text-anchor="middle">{period.category.upper()}</text>')

    x = sx(periods[-1].end_year)
    lines.append(f'<line x1="{x:.2f}" y1="{margin_top}" x2="{x:.2f}" y2="{height - margin_bottom}" stroke="#bdbdbd" stroke-width="1"/>')

    lines.extend(
        [
            f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#333" stroke-width="1.2"/>',
            f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#333" stroke-width="1.2"/>',
            f'<polyline points="{points}" fill="none" stroke="#1f77b4" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>',
            f'<text x="{width / 2:.1f}" y="{height - 20}" text-anchor="middle">Time [years]</text>',
            f'<text x="22" y="{height / 2:.1f}" text-anchor="middle" transform="rotate(-90 22 {height / 2:.1f})">Corrosion loss [cm]</text>',
            "</svg>",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def calculate_and_plot_three_period_corrosion(
    end_years: Sequence[float],
    categories: Sequence[str],
    output_path: str | Path = "outputs/corrosion_curve.png",
    data_output_path: str | Path | None = None,
    start_year: float = 0.0,
    step_years: float = 0.25,
    h_corr: float = DEFAULT_H_CORR,
    show: bool = False,
) -> tuple[float, Path]:
    """Convenience function for the current simplified three-period workflow."""
    periods = build_three_period_scenario(
        end_years=end_years,
        categories=categories,
        start_year=start_year,
    )
    _, losses_cm, _ = corrosion_curve_cm(periods, step_years=step_years, h_corr=h_corr)
    plot_path = plot_corrosion_curve(
        periods=periods,
        output_path=output_path,
        step_years=step_years,
        h_corr=h_corr,
        show=show,
    )
    if data_output_path is not None:
        save_corrosion_curve_data_csv(
            periods=periods,
            output_path=data_output_path,
            step_years=step_years,
            h_corr=h_corr,
        )
    return losses_cm[-1], plot_path


if __name__ == "__main__":
    scenario = build_three_period_scenario(
        end_years=(20, 36, 50),
        categories=("C3", "C1", "C2"),
    )
    final_loss_cm, figure_path = calculate_and_plot_three_period_corrosion(
        end_years=(20, 36, 50),
        categories=("C3", "C1", "C2"),
        data_output_path="outputs/corrosion_curve_data.csv",
    )
    print(f"Final corrosion loss: {final_loss_cm:.6f} cm")
    print(f"Curve saved to: {figure_path}")
