import math
from typing import Iterable, Optional, Sequence, Tuple

from transparent_plots import configure_transparent_matplotlib_saves

configure_transparent_matplotlib_saves()

import matplotlib.pyplot as plt
import numpy as np


def _first_n_lambdas(lam: Iterable[float], n: int = 3) -> Sequence[float]:
    vals = []
    for x in lam:
        try:
            xf = float(x)
        except Exception:
            continue
        if math.isfinite(xf):
            vals.append(xf)
        if len(vals) >= int(n):
            break
    if len(vals) < int(n):
        raise ValueError(f"Need at least {int(n)} valid buckling factors; got {len(vals)}.")
    return vals


def merchant_rankine_factor(dcr_max: float, lambda_i: float) -> float:
    dcr_max = float(dcr_max)
    lambda_i = float(lambda_i)
    if dcr_max <= 0.0:
        raise ValueError("dcr_max must be > 0.")
    if lambda_i <= 0.0:
        raise ValueError("lambda_i must be > 0.")
    return 1.0 / ((1.0 / dcr_max) + (1.0 / lambda_i))


def plot_merchant_rankine_comparison(
    dcr_max: float,
    lam: Iterable[float],
    merchant_rankine_factors: Optional[Sequence[float]] = None,
    lf_nonlinear: Optional[float] = None,
    show: bool = True,
    save_path: Optional[str] = None,
    title: str = "Merchant-Rankine and Buckling Comparison",
    dpi: int = 300,
    annotate: bool = True,
) -> Tuple[plt.Figure, plt.Axes, dict]:
    """
    Bar chart comparing:
      - Merchant-Rankine factors MR1, MR2, MR3, MR4, MR5
      - 1 / DCRmax
      - nonlinear analysis LF
      - first 5 linear buckling factors
    """
    dcr_max = float(dcr_max)
    if dcr_max <= 0.0:
        raise ValueError("dcr_max must be > 0.")

    lam5 = list(_first_n_lambdas(lam, n=5))
    lfp_mr = 1.0 / dcr_max

    if merchant_rankine_factors is None:
        mr5 = [merchant_rankine_factor(dcr_max, x) for x in lam5]
    else:
        mr5 = [float(x) for x in merchant_rankine_factors]
        if len(mr5) < 5:
            raise ValueError("merchant_rankine_factors must contain at least 5 values.")
        mr5 = mr5[:5]

    labels = [
        "MR1",
        "MR2",
        "MR3",
        "MR4",
        "MR5",
        "1 / DCRmax",
        "LF nonlinear",
        "LB mode 1",
        "LB mode 2",
        "LB mode 3",
        "LB mode 4",
        "LB mode 5",
    ]
    nl_lf = float(lf_nonlinear) if lf_nonlinear is not None else float("nan")
    values = [mr5[0], mr5[1], mr5[2], mr5[3], mr5[4], lfp_mr, nl_lf, lam5[0], lam5[1], lam5[2], lam5[3], lam5[4]]

    # smaller and cleaner plot
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    x = np.arange(len(labels), dtype=float)
    width = 0.52

    colors = [
        "#8fbcd4",  # MR1
        "#a7d3bf",  # MR2
        "#c7dfb1",  # MR3
        "#d7e8a4",  # MR4
        "#e7f096",  # MR5
        "#cfcfcf",  # 1/DCR
        "#b7b7b7",  # nonlinear LF
        "#d8d8d8",  # LB1
        "#e0e0e0",  # LB2
        "#e8e8e8",  # LB3
        "#f0f0f0",  # LB4
        "#f8f8f8",  # LB5
    ]

    bars = ax.bar(x, values, width=width, color=colors, edgecolor="#666666", linewidth=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=9)
    ax.set_ylabel("Factor", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", alpha=0.20, linewidth=0.6)
    ax.tick_params(axis="y", labelsize=8)

    finite_vals = [v for v in values if np.isfinite(v)]
    ymax = max(finite_vals) if finite_vals else 1.0
    ax.set_ylim(0.0, ymax * 1.12)

    if annotate:
        for rect, val in zip(bars, values):
            txt = "NA" if not np.isfinite(val) else f"{val:.2f}"
            ytxt = rect.get_height() if np.isfinite(val) else 0.0
            ax.text(
                rect.get_x() + rect.get_width() / 2.0,
                ytxt + 0.015 * ymax,
                txt,
                ha="center",
                va="bottom",
                fontsize=8,
                color="#333333",
            )

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    fig.tight_layout(pad=0.8)

    if save_path:
        fig.savefig(save_path, dpi=int(dpi), bbox_inches="tight")

    if show:
        plt.show()

    data = {
        "merchant_rankine_1": mr5[0],
        "merchant_rankine_2": mr5[1],
        "merchant_rankine_3": mr5[2],
        "merchant_rankine_4": mr5[3],
        "merchant_rankine_5": mr5[4],
        "lfp_merchant_rankine": lfp_mr,
        "lf_nonlinear": nl_lf,
        "lambda_1": lam5[0],
        "lambda_2": lam5[1],
        "lambda_3": lam5[2],
        "lambda_4": lam5[3],
        "lambda_5": lam5[4],
        "labels": labels,
        "values": values,
    }
    return fig, ax, data


__all__ = [
    "merchant_rankine_factor",
    "plot_merchant_rankine_comparison",
]
