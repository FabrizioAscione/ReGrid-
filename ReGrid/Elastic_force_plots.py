from __future__ import annotations

from pathlib import Path
from typing import List, Mapping, Optional, Sequence

from transparent_plots import configure_transparent_matplotlib_saves

configure_transparent_matplotlib_saves()

import matplotlib.pyplot as plt
import numpy as np


_FORCE_KEYS = ("N", "Vy", "Vz", "T", "My", "Mz")


def _normalize_components(components: Optional[Sequence[str]]) -> List[str]:
    if components is None:
        return ["N", "My", "Mz", "Vy", "Vz", "T"]
    out = []
    for c in components:
        cc = str(c).strip()
        if cc not in _FORCE_KEYS:
            raise ValueError(f"Unsupported component '{c}'. Use one of {_FORCE_KEYS}.")
        out.append(cc)
    return out


def _extract_arrays(ele_forces: Mapping[int, Mapping[str, Mapping[str, float]]], component: str):
    ele_tags = sorted(int(e) for e in ele_forces.keys())
    vals_i, vals_j, vals_abs = [], [], []
    for e in ele_tags:
        rec = ele_forces[e]
        vi = float(rec["i"][component])
        vj = float(rec["j"][component])
        vals_i.append(vi)
        vals_j.append(vj)
        vals_abs.append(max(abs(vi), abs(vj)))
    return (
        np.array(ele_tags, dtype=int),
        np.array(vals_i, dtype=float),
        np.array(vals_j, dtype=float),
        np.array(vals_abs, dtype=float),
    )


def _order_arrays(ele_tags, vals_i, vals_j, vals_abs, sort_by: str):
    mode = str(sort_by).strip().lower()
    if mode == "abs_max":
        order = np.argsort(-vals_abs)
    elif mode == "element":
        order = np.argsort(ele_tags)
    else:
        raise ValueError("sort_by must be 'element' or 'abs_max'.")
    return ele_tags[order], vals_i[order], vals_j[order], vals_abs[order]


def _xtick_step(n: int, target_labels: int = 24) -> int:
    return max(1, int(np.ceil(n / max(1, target_labels))))


def plot_elastic_force_component(
    ele_forces: Mapping[int, Mapping[str, Mapping[str, float]]],
    component: str,
    *,
    sort_by: str = "element",
    annotate_top_n: int = 0,
    show: bool = True,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
):
    """Single clean plot with only the maximum absolute value per element."""
    component = str(component).strip()
    ele_tags, vals_i, vals_j, vals_abs = _extract_arrays(ele_forces, component)
    ele_tags, vals_i, vals_j, vals_abs = _order_arrays(ele_tags, vals_i, vals_j, vals_abs, sort_by)

    x = np.arange(len(ele_tags), dtype=float)
    fig, ax = plt.subplots(figsize=(16, 5.8))
    ax.plot(x, vals_abs, linewidth=1.8, marker="o", markersize=3.5)
    ax.fill_between(x, 0.0, vals_abs, alpha=0.12)
    ax.set_title(title or f"Elastic internal force |{component}| max per element")
    ax.set_ylabel(f"|{component}| max")
    ax.set_xlabel("Element tag")
    ax.grid(True, axis="y", alpha=0.25)

    step = _xtick_step(len(ele_tags), target_labels=22)
    ax.set_xticks(x[::step])
    ax.set_xticklabels([str(int(e)) for e in ele_tags[::step]], rotation=60, ha="right")

    if annotate_top_n and annotate_top_n > 0:
        idx = np.argsort(-vals_abs)[: min(int(annotate_top_n), len(ele_tags))]
        for ii in idx:
            ax.annotate(
                f"e{int(ele_tags[ii])}\n{vals_abs[ii]:.3g}",
                (x[ii], vals_abs[ii]),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=8,
            )

    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax



def plot_elastic_force_suite(
    ele_forces: Mapping[int, Mapping[str, Mapping[str, float]]],
    *,
    components: Optional[Sequence[str]] = None,
    sort_by: str = "element",
    annotate_top_n: int = 0,
    show: bool = True,
    out_dir: Optional[str] = None,
    save_prefix: str = "Elastic_force",
):
    components = _normalize_components(components)
    figs = {}
    for comp in components:
        save_path = None
        if out_dir:
            save_path = str(Path(out_dir) / f"{save_prefix}_{comp}.png")
        fig, axes = plot_elastic_force_component(
            ele_forces,
            comp,
            sort_by=sort_by,
            annotate_top_n=annotate_top_n,
            show=show,
            save_path=save_path,
        )
        figs[comp] = (fig, axes)
    return figs



def write_elastic_force_report(
    ele_forces: Mapping[int, Mapping[str, Mapping[str, float]]],
    *,
    filepath: str = "outputs/elastic_force_report.txt",
    sort_by: str = "element",
):
    rows = []
    for e, rec in ele_forces.items():
        e = int(e)
        row = {"ele": e}
        for comp in _FORCE_KEYS:
            vi = float(rec["i"][comp])
            vj = float(rec["j"][comp])
            row[f"{comp}_i"] = vi
            row[f"{comp}_j"] = vj
            row[f"{comp}_abs_max"] = max(abs(vi), abs(vj))
        rows.append(row)

    mode = sort_by.lower()
    if mode == "abs_max_n":
        rows.sort(key=lambda r: -r["N_abs_max"])
    elif mode == "abs_max_my":
        rows.sort(key=lambda r: -r["My_abs_max"])
    elif mode == "abs_max_mz":
        rows.sort(key=lambda r: -r["Mz_abs_max"])
    else:
        rows.sort(key=lambda r: r["ele"])

    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("# ELASTIC INTERNAL FORCES REPORT\n")
        for r in rows:
            f.write(
                f"ele {r['ele']:4d} | "
                f"N(i/j/max)=({r['N_i']:.6g}, {r['N_j']:.6g}, {r['N_abs_max']:.6g}) | "
                f"Vy(i/j/max)=({r['Vy_i']:.6g}, {r['Vy_j']:.6g}, {r['Vy_abs_max']:.6g}) | "
                f"Vz(i/j/max)=({r['Vz_i']:.6g}, {r['Vz_j']:.6g}, {r['Vz_abs_max']:.6g}) | "
                f"T(i/j/max)=({r['T_i']:.6g}, {r['T_j']:.6g}, {r['T_abs_max']:.6g}) | "
                f"My(i/j/max)=({r['My_i']:.6g}, {r['My_j']:.6g}, {r['My_abs_max']:.6g}) | "
                f"Mz(i/j/max)=({r['Mz_i']:.6g}, {r['Mz_j']:.6g}, {r['Mz_abs_max']:.6g})\n"
            )
    return filepath
