from __future__ import annotations

import csv
from dataclasses import dataclass
import gc
import json
import os
import sys
from typing import Callable, Dict, Optional, Sequence

import numpy as np
import tkinter as tk
from tkinter import messagebox, ttk

from pareto_plot_style import save_pareto_figure
from transparent_plots import configure_transparent_matplotlib_saves

configure_transparent_matplotlib_saves()

from matplotlib import colors as mcolors
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from reuse_genetic_evaluator import CandidateEvaluation
from reuse_genetic_opt import objective_display_values, pareto_proximity_scores


@dataclass
class ParetoSelectionResult:
    evaluation: Optional[CandidateEvaluation]
    index: Optional[int]
    run_all_executed: bool = False
    run_message: Optional[str] = None
    run_all_solution_kind: str = "reused"
    continue_requested: bool = False
    continue_generations: Optional[int] = None


def _configure_tk_runtime() -> None:
    _patch_tk_destructors()
    base_prefix = getattr(sys, "base_prefix", "") or ""
    if not base_prefix:
        return
    tcl_root = os.path.join(base_prefix, "tcl")
    tcl_library = os.path.join(tcl_root, "tcl8.6")
    tk_library = os.path.join(tcl_root, "tk8.6")
    if os.path.isdir(tcl_library) and "TCL_LIBRARY" not in os.environ:
        os.environ["TCL_LIBRARY"] = tcl_library
    if os.path.isdir(tk_library) and "TK_LIBRARY" not in os.environ:
        os.environ["TK_LIBRARY"] = tk_library


_TK_DESTRUCTORS_PATCHED = False


def _patch_tk_destructors() -> None:
    global _TK_DESTRUCTORS_PATCHED
    if _TK_DESTRUCTORS_PATCHED:
        return
    _TK_DESTRUCTORS_PATCHED = True

    original_image_del = getattr(tk.Image, "__del__", None)
    original_variable_del = getattr(tk.Variable, "__del__", None)

    def _safe_image_del(self):
        if original_image_del is None:
            return
        try:
            original_image_del(self)
        except (RuntimeError, tk.TclError):
            pass

    def _safe_variable_del(self):
        if original_variable_del is None:
            return
        try:
            original_variable_del(self)
        except (RuntimeError, tk.TclError):
            pass

    tk.Image.__del__ = _safe_image_del
    tk.Variable.__del__ = _safe_variable_del


class ParetoSelectionApp:
    _PARALLEL_FONT_FAMILY = "Times New Roman"
    _PARALLEL_TICK_COUNT = 5
    _PARETO_FONT_FAMILY = "Arial"

    def __init__(
        self,
        pareto_front: Sequence[CandidateEvaluation],
        all_evaluations: Sequence[CandidateEvaluation],
        objective_names: Sequence[str],
        *,
        title: str = "Select Pareto solution for nonlinear analysis",
        on_run_selected: Optional[Callable[[CandidateEvaluation, int], str]] = None,
        on_run_all: Optional[Callable[[Sequence[CandidateEvaluation]], str]] = None,
        on_run_reference_selected: Optional[Callable[[CandidateEvaluation, int], str]] = None,
        on_run_reference_all: Optional[Callable[[Sequence[CandidateEvaluation]], str]] = None,
        on_run_fixed_base: Optional[Callable[[], str]] = None,
        save_pareto_only_path: Optional[str] = None,
        save_parallel_coords_path: Optional[str] = None,
    ):
        if not pareto_front:
            raise ValueError("Pareto front is empty.")
        if len(objective_names) not in {2, 3}:
            raise ValueError("Pareto selection GUI supports only 2D or 3D objective sets.")

        self.pareto_front = list(pareto_front)
        self.all_evaluations = list(all_evaluations) if all_evaluations else list(pareto_front)
        self.objective_names = tuple(str(v) for v in objective_names)
        self.result: Optional[ParetoSelectionResult] = None
        self.on_run_selected = on_run_selected
        self.on_run_all = on_run_all
        self.on_run_reference_selected = on_run_reference_selected
        self.on_run_reference_all = on_run_reference_all
        self.on_run_fixed_base = on_run_fixed_base
        self.save_pareto_only_path = save_pareto_only_path
        self.save_parallel_coords_path = save_parallel_coords_path
        self._selection_markers = []
        self._parallel_selection_line = None
        self._parallel_selection_markers = None
        self._parallel_value_labels = []
        self._selected_index: Optional[int] = None
        self._pareto_artist = None
        self._pareto_artists = []
        self._feasible_artists = []
        self._main_feasible_artist = None
        self._feasible_cmap = mcolors.LinearSegmentedColormap.from_list(
            "pareto_proximity_gui",
            ["#7d92ab", "#bcc7d3", "#ecd08a", "#ee8b63"],
        )
        self._feasible_norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
        self._pareto_coords = np.empty((0, len(self.objective_names)), dtype=float)
        self._display_coords = np.empty((0, len(self.objective_names)), dtype=float)
        self._feasible_coords = np.empty((0, len(self.objective_names)), dtype=float)
        self._feasible_closeness = np.empty((0,), dtype=float)
        self._axis_order_2d = (0, 1)
        self._projection_axis_swap_states = [False, False, False]
        self.ax = None
        self._pareto_projection_axes = {}
        self._projection_axes_list = []
        self._parallel_axes = None
        self._parallel_x = None
        self._parallel_ranges = None
        self._pareto_initial_limits_2d = None
        self._pareto_initial_limits_3d = None
        self._pareto_projection_limits = {}
        self._toolbar = None
        self._zoom_enabled = False
        self._pan_enabled = False
        self._closed = False
        self._comparison_canvas = None
        self._comparison_fig = None
        self._comparison_axes = []
        self.canvas = None
        self.fig = None

        _configure_tk_runtime()
        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry("1520x1040")
        self.root.protocol("WM_DELETE_WINDOW", self._on_cancel)

        self._candidate_var = tk.StringVar(value="No point selected")
        default_status = "Click a red Pareto point, then confirm."
        if (
            self.on_run_selected is None
            and self.on_run_all is None
            and self.on_run_reference_selected is None
            and self.on_run_reference_all is None
            and self.on_run_fixed_base is None
        ):
            default_status = "Saved results loaded. Click a red Pareto point to inspect it."
        self._status_var = tk.StringVar(value=default_status)
        self._show_all_var = tk.BooleanVar(value=True)
        self._continue_generations_var = tk.StringVar(value="10")
        self._detail_text = None

        self._build_ui(title)
        self._save_static_plots()

    def _coords(self, rows: Sequence[CandidateEvaluation]) -> np.ndarray:
        if not rows:
            return np.empty((0, len(self.objective_names)), dtype=float)
        return np.asarray([list(objective_display_values(ev, self.objective_names).values()) for ev in rows], dtype=float)

    def _pareto_proximity(self, rows: Sequence[CandidateEvaluation]) -> Dict[str, np.ndarray]:
        return pareto_proximity_scores(rows, self.pareto_front, self.objective_names)

    def _parallel_metric_defs(self):
        return (
            ("candidate_id", lambda ev: float(ev.meta.get("candidate_id", float("nan")))),
            ("%Riuso", lambda ev: 100.0 * float(ev.objectives.get("reuse_ratio", 0.0))),
            ("ReuseEff", lambda ev: 100.0 * float(ev.objectives.get("reuse_efficiency", ev.meta.get("reuse_efficiency", 0.0)))),
            ("Massa", lambda ev: float(ev.objectives.get("mass", float("nan")))),
            ("Cost", lambda ev: float(ev.objectives.get("cost", ev.meta.get("cost", float("nan"))))),
            ("EC", lambda ev: float(ev.meta.get("ec", float("nan")))),
            ("PI_M", lambda ev: float(ev.objectives.get("pi_mass", ev.meta.get("pi_mass", float("nan"))))),
            ("PI_EC", lambda ev: float(ev.objectives.get("pi_ec", ev.meta.get("pi_ec", float("nan"))))),
            ("PI_MR", lambda ev: float(ev.objectives.get("pi_mr", ev.meta.get("pi_mr", float("nan"))))),
            ("DCRmax", lambda ev: float(ev.meta.get("dcr_max", float("nan")))),
            ("DCRmean", lambda ev: float(ev.meta.get("dcr_mean", float("nan")))),
            ("Lambda", lambda ev: float(ev.lambda_ref if ev.lambda_ref is not None else float("nan"))),
            ("MR", lambda ev: float(ev.meta.get("merchant_rankine", float("nan")))),
            ("Mscrap", lambda ev: float(ev.meta.get("mscrap", float("nan")))),
            ("dmax", lambda ev: float(ev.meta.get("max_abs_uz", float("nan")))),
            ("Lscrap", lambda ev: float(ev.meta.get("lscrap", float("nan")))),
            ("n_cut", lambda ev: float(ev.meta.get("n_cuts", float("nan")))),
        )

    def _parallel_values(self, evaluation: CandidateEvaluation) -> np.ndarray:
        values = []
        for _, getter in self._parallel_metric_defs():
            try:
                values.append(float(getter(evaluation)))
            except Exception:
                values.append(float("nan"))
        return np.asarray(values, dtype=float)

    def _finite_parallel_rows(self):
        rows = []
        raw_values = []
        for ev in self.all_evaluations:
            values = self._parallel_values(ev)
            if np.all(np.isfinite(values)):
                rows.append(ev)
                raw_values.append(values)
        return rows, raw_values

    @staticmethod
    def _format_parallel_value(value: float) -> str:
        abs_value = abs(float(value))
        if abs_value >= 1000.0 or (abs_value > 0.0 and abs_value < 0.1):
            return f"{value:.2e}"
        if abs_value >= 100.0:
            return f"{value:.0f}"
        if abs_value >= 10.0:
            return f"{value:.1f}"
        if abs_value >= 1.0:
            return f"{value:.2f}"
        return f"{value:.3f}"

    @staticmethod
    def _metric_value(value) -> float:
        try:
            return float(value)
        except Exception:
            return float("nan")

    def _render_parallel_coordinates(self, ax, raw_values) -> Optional[tuple]:
        if not raw_values:
            ax.text(
                0.5,
                0.5,
                "No finite candidate metrics available.",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontfamily=self._PARALLEL_FONT_FAMILY,
                fontsize=11,
                color="black",
            )
            ax.set_axis_off()
            return None

        raw = np.asarray(raw_values, dtype=float)
        mins = raw.min(axis=0)
        maxs = raw.max(axis=0)
        spans = np.where((maxs - mins) > 1e-12, (maxs - mins), 1.0)
        normalized = (raw - mins) / spans
        labels = [label for label, _ in self._parallel_metric_defs()]
        x = np.arange(len(labels), dtype=float)
        tick_levels = np.linspace(0.0, 1.0, self._PARALLEL_TICK_COUNT)
        tick_label_offset = 0.08
        tick_mark_halfwidth = 0.028

        for row in normalized:
            ax.plot(x, row, color="#8d8d8d", alpha=0.18, linewidth=0.8, zorder=1)

        for idx, label in enumerate(labels):
            x_pos = x[idx]
            ax.axvline(x_pos, color="black", linewidth=1.2, alpha=1.0, zorder=2)
            axis_tick_values = mins[idx] + tick_levels * (maxs[idx] - mins[idx])
            if abs(maxs[idx] - mins[idx]) <= 1e-12:
                axis_tick_values = np.full_like(tick_levels, mins[idx], dtype=float)
            for level, tick_value in zip(tick_levels, axis_tick_values):
                ax.plot(
                    [x_pos - tick_mark_halfwidth, x_pos],
                    [level, level],
                    color="black",
                    linewidth=1.0,
                    solid_capstyle="butt",
                    zorder=3,
                )
                ax.text(
                    x_pos - tick_label_offset,
                    level,
                    self._format_parallel_value(float(tick_value)),
                    ha="right",
                    va="center",
                    fontsize=8,
                    fontfamily=self._PARALLEL_FONT_FAMILY,
                    color="black",
                    zorder=4,
                )
            ax.text(
                x_pos,
                -0.12,
                label,
                ha="center",
                va="top",
                fontsize=10,
                fontweight="bold",
                fontfamily=self._PARALLEL_FONT_FAMILY,
                color="black",
                zorder=4,
            )

        ax.set_xlim(-0.4, len(labels) - 1 + 0.35)
        ax.set_ylim(-0.18, 1.03)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_facecolor("white")

        return x, mins, spans

    def _draw_parallel_coordinates(self, ax) -> None:
        rows, raw_values = self._finite_parallel_rows()
        rendered = self._render_parallel_coordinates(ax, raw_values)

        self._parallel_axes = ax
        if rendered is None:
            self._parallel_x = None
            self._parallel_ranges = None
            return
        x, mins, spans = rendered
        self._parallel_x = x
        self._parallel_ranges = (mins, spans)

    def _clear_parallel_selection_artists(self) -> None:
        if self._parallel_selection_line is not None:
            try:
                self._parallel_selection_line.remove()
            except Exception:
                pass
            self._parallel_selection_line = None
        if self._parallel_selection_markers is not None:
            try:
                self._parallel_selection_markers.remove()
            except Exception:
                pass
            self._parallel_selection_markers = None
        while self._parallel_value_labels:
            label = self._parallel_value_labels.pop()
            try:
                label.remove()
            except Exception:
                pass

    def _highlight_parallel_selection(self, evaluation: CandidateEvaluation) -> None:
        if self._parallel_axes is None or self._parallel_x is None or self._parallel_ranges is None:
            return
        self._clear_parallel_selection_artists()

        values = self._parallel_values(evaluation)
        if not np.all(np.isfinite(values)):
            self.canvas.draw_idle()
            return

        mins, spans = self._parallel_ranges
        normalized = (values - mins) / spans
        self._parallel_selection_line = self._parallel_axes.plot(
            self._parallel_x,
            normalized,
            color="red",
            linewidth=2.2,
            alpha=0.95,
            zorder=4,
        )[0]
        self._parallel_selection_markers = self._parallel_axes.scatter(
            self._parallel_x,
            normalized,
            s=20,
            c="red",
            zorder=5,
        )
        for idx, value in enumerate(values):
            y = float(normalized[idx])
            y_text = min(1.045, max(0.03, y + 0.03))
            label = self._parallel_axes.text(
                self._parallel_x[idx],
                y_text,
                self._format_parallel_value(float(value)),
                ha="center",
                va="bottom",
                fontsize=7,
                color="red",
                zorder=6,
                fontfamily=self._PARALLEL_FONT_FAMILY,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.6},
            )
            self._parallel_value_labels.append(label)
        self.canvas.draw_idle()

    def _ensure_parent_dir(self, path: Optional[str]) -> None:
        if not path:
            return
        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)

    @staticmethod
    def _path_with_suffix(base_path: str, suffix: str) -> str:
        root, ext = os.path.splitext(base_path)
        return f"{root}{suffix}{ext or '.png'}"

    @staticmethod
    def _label_slug(label: str) -> str:
        safe = "".join(ch if ch.isalnum() else "_" for ch in str(label))
        safe = "_".join(part for part in safe.split("_") if part)
        return safe.lower() or "obj"

    @staticmethod
    def _set_axis_limits_2d(ax, coords: np.ndarray, ix: int, iy: int) -> None:
        finite_rows = np.isfinite(coords[:, ix]) & np.isfinite(coords[:, iy])
        if not np.any(finite_rows):
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            return
        x_vals = coords[finite_rows, ix]
        y_vals = coords[finite_rows, iy]
        x_min = float(np.min(x_vals))
        x_max = float(np.max(x_vals))
        y_min = float(np.min(y_vals))
        y_max = float(np.max(y_vals))
        if abs(x_max - x_min) <= 1e-12:
            pad = max(1.0, abs(x_min) * 0.05)
        else:
            pad = max(abs(x_max - x_min) * 0.06, 1e-9)
        ax.set_xlim(x_min - pad, x_max + pad)
        if abs(y_max - y_min) <= 1e-12:
            pad = max(1.0, abs(y_min) * 0.05)
        else:
            pad = max(abs(y_max - y_min) * 0.06, 1e-9)
        ax.set_ylim(y_min - pad, y_max + pad)

    def _style_pareto_axis_2d(self, ax, coords: np.ndarray, ix: int, iy: int, title: Optional[str] = None) -> None:
        ax.set_xlabel(self.objective_names[ix], fontname=self._PARETO_FONT_FAMILY, labelpad=5)
        ax.set_ylabel(self.objective_names[iy], fontname=self._PARETO_FONT_FAMILY, labelpad=4)
        if title:
            ax.set_title(title, fontname=self._PARETO_FONT_FAMILY)
        ax.margins(x=0.04, y=0.04)
        self._set_axis_limits_2d(ax, coords, ix, iy)
        ax.set_facecolor("#fbfbfd")
        ax.tick_params(axis="x", pad=1)
        ax.tick_params(axis="y", pad=1)
        try:
            ax.xaxis.set_label_coords(0.5, -0.12)
            ax.yaxis.set_label_coords(-0.11, 0.5)
        except Exception:
            pass
        for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
            tick_label.set_fontname(self._PARETO_FONT_FAMILY)
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    def _scatter_pareto_projection(
        self,
        ax,
        axis_coords: np.ndarray,
        feasible_coords: np.ndarray,
        feasible_closeness: np.ndarray,
        pareto_coords: np.ndarray,
        ix: int,
        iy: int,
        *,
        title: str,
        show_legend: bool = False,
    ):
        feasible_artist = None
        if len(feasible_coords) > 0:
            feasible_artist = ax.scatter(
                feasible_coords[:, ix],
                feasible_coords[:, iy],
                c=feasible_closeness,
                cmap=self._feasible_cmap,
                norm=self._feasible_norm,
                s=32,
                alpha=0.88,
                edgecolors="white",
                linewidths=0.25,
                label="Feasible",
            )
        artist = ax.scatter(
            pareto_coords[:, ix],
            pareto_coords[:, iy],
            c="#c62828",
            s=66,
            edgecolors="#7f0000",
            linewidths=0.85,
            picker=10,
            label="Pareto front",
        )
        self._style_pareto_axis_2d(ax, axis_coords, ix, iy, title)
        if show_legend:
            ax.legend(loc="best")
        return artist, feasible_artist

    def _save_pareto_only_plot(self) -> None:
        if not self.save_pareto_only_path:
            return
        self._ensure_parent_dir(self.save_pareto_only_path)
        pareto_coords = self._coords(self.pareto_front)
        if len(self.objective_names) == 3:
            fig = Figure(figsize=(7.2, 5.8), dpi=160)
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(
                pareto_coords[:, 0],
                pareto_coords[:, 1],
                pareto_coords[:, 2],
                c="#c62828",
                s=58,
                edgecolors="#7f0000",
                linewidths=0.7,
                depthshade=True,
            )
            ax.set_title("Pareto 3D", fontname=self._PARETO_FONT_FAMILY)
            ax.set_xlabel(self.objective_names[0], fontname=self._PARETO_FONT_FAMILY)
            ax.set_ylabel(self.objective_names[1], fontname=self._PARETO_FONT_FAMILY)
            ax.set_zlabel(self.objective_names[2], fontname=self._PARETO_FONT_FAMILY)
            for tick_label in ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels():
                tick_label.set_fontname(self._PARETO_FONT_FAMILY)
            ax.grid(False)
            try:
                ax.xaxis.set_pane_color((0.98, 0.985, 0.99, 1.0))
                ax.yaxis.set_pane_color((0.98, 0.985, 0.99, 1.0))
                ax.zaxis.set_pane_color((0.98, 0.985, 0.99, 1.0))
            except Exception:
                pass
            save_pareto_figure(fig, self.save_pareto_only_path, is_3d=len(self.objective_names) == 3)

            for ix, iy in ((0, 1), (0, 2), (1, 2)):
                fig2 = Figure(figsize=(7.0, 5.2), dpi=160)
                ax2 = fig2.add_subplot(111)
                self._scatter_pareto_projection(
                    ax2,
                    pareto_coords,
                    np.empty((0, len(self.objective_names)), dtype=float),
                    np.empty((0,), dtype=float),
                    pareto_coords,
                    ix,
                    iy,
                    title=f"{self.objective_names[ix]} vs {self.objective_names[iy]}",
                )
                projection_path = self._path_with_suffix(
                    self.save_pareto_only_path,
                    f"_{self._label_slug(self.objective_names[ix])}_vs_{self._label_slug(self.objective_names[iy])}",
                )
                save_pareto_figure(fig2, projection_path)
            return
        else:
            fig = Figure(figsize=(7.2, 5.8), dpi=160)
            ax = fig.add_subplot(111)
            ax.scatter(
                pareto_coords[:, 0],
                pareto_coords[:, 1],
                c="#c62828",
                s=62,
                edgecolors="#7f0000",
                linewidths=0.7,
            )
            self._style_pareto_axis_2d(ax, pareto_coords, 0, 1)
        save_pareto_figure(fig, self.save_pareto_only_path, is_3d=len(self.objective_names) == 3)

    def _save_parallel_coords_plot(self) -> None:
        if not self.save_parallel_coords_path:
            return
        self._ensure_parent_dir(self.save_parallel_coords_path)
        fig = Figure(figsize=(10.5, 4.2), dpi=160)
        ax = fig.add_subplot(111)

        _, raw_values = self._finite_parallel_rows()
        self._render_parallel_coordinates(ax, raw_values)
        fig.subplots_adjust(left=0.06, right=0.99, top=0.98, bottom=0.22)
        fig.savefig(self.save_parallel_coords_path, dpi=220, bbox_inches="tight")

    def _save_static_plots(self) -> None:
        self._save_pareto_only_plot()
        self._save_parallel_coords_plot()

    def _comparison_pairs(self, evaluation: CandidateEvaluation):
        reference_label = str(evaluation.meta.get("comparison_reference_label", "new_reference") or "new_reference")
        reference_name = "Reference"
        if "fixed" in reference_label.lower() or "base" in reference_label.lower():
            reference_name = "Fixed base"
        return [
            (
                "Mass",
                self._metric_value(
                    evaluation.meta.get(
                        "comparison_reference_mass",
                        evaluation.meta.get("new_reference_mass", float("nan")),
                    )
                ),
                self._metric_value(evaluation.objectives.get("mass", float("nan"))),
            ),
            (
                "EC",
                self._metric_value(
                    evaluation.meta.get(
                        "comparison_reference_ec",
                        evaluation.meta.get("new_reference_ec", float("nan")),
                    )
                ),
                self._metric_value(evaluation.meta.get("ec", float("nan"))),
            ),
            (
                "MR",
                self._metric_value(
                    evaluation.meta.get(
                        "comparison_reference_merchant_rankine",
                        evaluation.meta.get("new_reference_merchant_rankine", float("nan")),
                    )
                ),
                self._metric_value(evaluation.meta.get("merchant_rankine", float("nan"))),
            ),
        ], reference_name

    def _draw_empty_comparison_chart(self) -> None:
        if not self._comparison_axes:
            return
        for ax in self._comparison_axes:
            ax.clear()
            ax.text(
                0.5,
                0.5,
                "Select a Pareto point\nto compare reused vs reference.",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=9,
                color="black",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        if self._comparison_canvas is not None:
            self._comparison_canvas.draw_idle()

    def _update_comparison_chart(self, evaluation: CandidateEvaluation) -> None:
        if not self._comparison_axes:
            return
        pairs, reference_name = self._comparison_pairs(evaluation)
        for ax, (label, new_value, reused_value) in zip(self._comparison_axes, pairs):
            ax.clear()
            values = [float(new_value), float(reused_value)]
            x = np.arange(2, dtype=float)
            colors = ["#4c78a8", "#e45756"]
            bars = ax.bar(x, values, color=colors, width=0.58)
            ax.set_xticks(x, [reference_name, "Reused"])
            ax.set_title(label, fontsize=10, fontweight="bold")
            finite_values = [v for v in values if np.isfinite(v)]
            if finite_values:
                vmax = max(finite_values)
                if abs(vmax) <= 1.0e-12:
                    ymax = 1.0
                else:
                    ymax = vmax * 1.18
                ax.set_ylim(0.0, ymax)
            for bar, value in zip(bars, values):
                if not np.isfinite(value):
                    continue
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height(),
                    self._format_parallel_value(float(value)),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
            ax.grid(False)
            for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
                tick_label.set_fontname(self._PARETO_FONT_FAMILY)
        if self._comparison_canvas is not None:
            self._comparison_canvas.draw_idle()

    def _build_ui(self, title: str) -> None:
        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill="both", expand=True)

        ttk.Label(
            outer,
            text=title,
            font=("TkDefaultFont", 11, "bold"),
        ).pack(anchor="w")

        ttk.Label(
            outer,
            text=(
                "Pareto front in red. Other feasible points move from cool to warm colors "
                "as they get closer to the Pareto front. Only red points are selectable "
                "for the nonlinear analysis."
            ),
            wraplength=1180,
            justify="left",
        ).pack(anchor="w", pady=(4, 10))

        body = ttk.Frame(outer)
        body.pack(fill="both", expand=True)

        plot_frame = ttk.Frame(body)
        plot_frame.pack(side="left", fill="both", expand=True)

        side_container = ttk.Frame(body)
        side_container.pack(side="right", fill="y")
        side_scrollbar = ttk.Scrollbar(side_container, orient="vertical")
        side_scrollbar.pack(side="right", fill="y")
        side_canvas = tk.Canvas(
            side_container,
            highlightthickness=0,
            borderwidth=0,
            width=360,
        )
        side_canvas.pack(side="left", fill="y", expand=False)
        side_scrollbar.configure(command=side_canvas.yview)
        side_canvas.configure(yscrollcommand=side_scrollbar.set)
        side = ttk.Frame(side_canvas, padding=(12, 0, 0, 0))
        side_window = side_canvas.create_window((0, 0), window=side, anchor="nw")

        def _update_side_scrollregion(_event=None) -> None:
            try:
                side_canvas.configure(scrollregion=side_canvas.bbox("all"))
            except Exception:
                pass

        def _sync_side_width(event) -> None:
            try:
                side_canvas.itemconfigure(side_window, width=event.width)
            except Exception:
                pass
            _update_side_scrollregion()

        def _side_mousewheel(event) -> str:
            try:
                delta = int(event.delta)
            except Exception:
                delta = 0
            if delta != 0:
                side_canvas.yview_scroll(int(-delta / 120), "units")
            return "break"

        side.bind("<Configure>", _update_side_scrollregion)
        side_canvas.bind("<Configure>", _sync_side_width)
        side_canvas.bind("<MouseWheel>", _side_mousewheel)

        fig = Figure(figsize=(13.8, 9.6), dpi=100)
        gs = fig.add_gridspec(2, 1, height_ratios=[3.8, 1.8], hspace=0.28)
        if len(self.objective_names) == 3:
            gs_top = gs[0].subgridspec(3, 2, width_ratios=[1.9, 1.0], height_ratios=[1.0, 1.0, 1.0], wspace=0.22, hspace=0.22)
            ax = fig.add_subplot(gs_top[:, 0], projection="3d")
            ax_xy = fig.add_subplot(gs_top[0, 1])
            ax_xz = fig.add_subplot(gs_top[1, 1])
            ax_yz = fig.add_subplot(gs_top[2, 1])
            self._projection_axes_list = [ax_xy, ax_xz, ax_yz]
            self._pareto_projection_axes = {}
        else:
            ax = fig.add_subplot(gs[0])
            self._pareto_projection_axes = {}
            self._projection_axes_list = []
        ax_parallel = fig.add_subplot(gs[1])
        self.ax = ax
        self.fig = fig
        self._pareto_artists = []
        self._feasible_artists = []
        self._pareto_projection_limits = {}

        feasible = [ev for ev in self.all_evaluations if ev.feasible]
        feasible_proximity = self._pareto_proximity(feasible)
        feasible_coords = feasible_proximity["coords"]
        feasible_closeness = feasible_proximity["closeness"]
        pareto_coords = self._coords(self.pareto_front)
        display_coords = np.vstack([feasible_coords, pareto_coords]) if len(feasible_coords) > 0 else pareto_coords
        self._feasible_coords = feasible_coords
        self._feasible_closeness = feasible_closeness
        self._pareto_coords = pareto_coords
        self._display_coords = display_coords

        if len(self.objective_names) == 2:
            self._rebuild_main_2d_plot()
        else:
            if len(feasible_coords) > 0:
                feasible_artist = ax.scatter(
                    feasible_coords[:, 0],
                    feasible_coords[:, 1],
                    feasible_coords[:, 2],
                    c=feasible_closeness,
                    cmap=self._feasible_cmap,
                    norm=self._feasible_norm,
                    s=32,
                    alpha=0.86,
                    edgecolors="white",
                    linewidths=0.22,
                    depthshade=True,
                    label="Feasible",
                )
                self._feasible_artists.append(feasible_artist)
                self._main_feasible_artist = feasible_artist
            self._pareto_artist = ax.scatter(
                pareto_coords[:, 0],
                pareto_coords[:, 1],
                pareto_coords[:, 2],
                c="#c62828",
                s=70,
                edgecolors="#7f0000",
                linewidths=0.82,
                depthshade=True,
                picker=10,
                label="Pareto front",
            )
            self._pareto_artists = [self._pareto_artist]
            ax.set_xlabel(self.objective_names[0], fontname=self._PARETO_FONT_FAMILY)
            ax.set_ylabel(self.objective_names[1], fontname=self._PARETO_FONT_FAMILY)
            ax.set_zlabel(self.objective_names[2], fontname=self._PARETO_FONT_FAMILY)
            try:
                ax.set_box_aspect((1.15, 1.15, 0.95))
            except Exception:
                pass
            for tick_label in ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels():
                tick_label.set_fontname(self._PARETO_FONT_FAMILY)
            ax.grid(False)
            try:
                ax.xaxis.set_pane_color((0.98, 0.985, 0.99, 1.0))
                ax.yaxis.set_pane_color((0.98, 0.985, 0.99, 1.0))
                ax.zaxis.set_pane_color((0.98, 0.985, 0.99, 1.0))
            except Exception:
                pass
            self._pareto_initial_limits_3d = (ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d())
            ax.legend(loc="best")
            self._rebuild_projection_2d_views()

        self._draw_parallel_coordinates(ax_parallel)
        fig.subplots_adjust(left=0.06, right=0.985, top=0.985, bottom=0.06)

        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        self._toolbar = NavigationToolbar2Tk(canvas, plot_frame, pack_toolbar=False)
        self._toolbar.pack(fill="x")
        canvas.mpl_connect("pick_event", self._on_pick)
        canvas.mpl_connect("button_press_event", self._on_click)
        self.canvas = canvas

        ttk.Label(side, text="Selected solution", font=("TkDefaultFont", 10, "bold")).pack(anchor="w")
        ttk.Label(side, textvariable=self._candidate_var, wraplength=300, justify="left").pack(anchor="w", pady=(4, 8))

        self._detail_text = tk.Text(side, width=40, height=18, wrap="word")
        self._detail_text.pack(fill="both", expand=True)
        self._detail_text.configure(state="disabled")

        ttk.Label(side, text="Reused vs New Reference", font=("TkDefaultFont", 10, "bold")).pack(anchor="w", pady=(10, 4))
        comparison_frame = ttk.Frame(side)
        comparison_frame.pack(fill="x")
        comparison_fig = Figure(figsize=(3.7, 3.8), dpi=100)
        comparison_axes = comparison_fig.subplots(3, 1)
        if not isinstance(comparison_axes, np.ndarray):
            comparison_axes = np.asarray([comparison_axes], dtype=object)
        self._comparison_fig = comparison_fig
        self._comparison_axes = list(comparison_axes)
        comparison_fig.subplots_adjust(left=0.18, right=0.98, top=0.95, bottom=0.08, hspace=0.72)
        comparison_canvas = FigureCanvasTkAgg(comparison_fig, master=comparison_frame)
        comparison_canvas.draw()
        comparison_canvas.get_tk_widget().pack(fill="x", expand=False)
        self._comparison_canvas = comparison_canvas
        self._draw_empty_comparison_chart()

        toggle_row = ttk.Frame(side)
        toggle_row.pack(fill="x", pady=(8, 2))
        ttk.Checkbutton(
            toggle_row,
            text="Show all solutions (color scale)",
            variable=self._show_all_var,
            command=self._toggle_show_all_solutions,
        ).pack(anchor="w")

        ttk.Label(side, textvariable=self._status_var, wraplength=300, justify="left").pack(anchor="w", pady=(10, 8))

        buttons = ttk.Frame(side)
        buttons.pack(fill="x")
        ttk.Button(buttons, text="Use selected and close", command=self._on_confirm).pack(side="left", fill="x", expand=True)
        ttk.Button(buttons, text="Close", command=self._on_cancel).pack(side="left", fill="x", expand=True, padx=(8, 0))

        run_buttons = ttk.Frame(side)
        run_buttons.pack(fill="x", pady=(8, 0))
        ttk.Button(run_buttons, text="Run selected reused", command=self._on_run_selected).pack(side="left", fill="x", expand=True)
        ttk.Button(run_buttons, text="Run all reused", command=self._on_run_all).pack(side="left", fill="x", expand=True, padx=(8, 0))

        if self.on_run_reference_selected is not None or self.on_run_reference_all is not None:
            run_buttons_ref = ttk.Frame(side)
            run_buttons_ref.pack(fill="x", pady=(8, 0))
            ttk.Button(run_buttons_ref, text="Run selected new", command=self._on_run_reference_selected).pack(side="left", fill="x", expand=True)
            ttk.Button(run_buttons_ref, text="Run all new", command=self._on_run_reference_all).pack(side="left", fill="x", expand=True, padx=(8, 0))

        if self.on_run_fixed_base is not None:
            run_buttons_fixed = ttk.Frame(side)
            run_buttons_fixed.pack(fill="x", pady=(8, 0))
            ttk.Button(run_buttons_fixed, text="Run fixed base", command=self._on_run_fixed_base).pack(side="left", fill="x", expand=True)

        continue_frame = ttk.LabelFrame(side, text="Continue optimization")
        continue_frame.pack(fill="x", pady=(10, 0))
        ttk.Label(
            continue_frame,
            text="Extra generations",
        ).pack(anchor="w", padx=8, pady=(8, 2))
        ttk.Entry(
            continue_frame,
            textvariable=self._continue_generations_var,
            width=10,
        ).pack(anchor="w", padx=8)
        ttk.Button(
            continue_frame,
            text="Run extra generations and reopen",
            command=self._on_continue_generations,
        ).pack(fill="x", padx=8, pady=(8, 8))

        nav_buttons = ttk.Frame(side)
        nav_buttons.pack(fill="x", pady=(8, 0))
        ttk.Button(nav_buttons, text="Zoom in", command=lambda: self._zoom_pareto(0.8)).pack(side="left", fill="x", expand=True)
        ttk.Button(nav_buttons, text="Zoom out", command=lambda: self._zoom_pareto(1.25)).pack(side="left", fill="x", expand=True, padx=(8, 0))
        ttk.Button(nav_buttons, text="Reset view", command=self._reset_view).pack(side="left", fill="x", expand=True, padx=(8, 0))

        if len(self.objective_names) == 2:
            axis_buttons = ttk.Frame(side)
            axis_buttons.pack(fill="x", pady=(8, 0))
            ttk.Button(axis_buttons, text="Swap X/Y", command=self._swap_axes_2d).pack(side="left", fill="x", expand=True)
        else:
            turn_row_1 = ttk.Frame(side)
            turn_row_1.pack(fill="x", pady=(8, 0))
            ttk.Button(turn_row_1, text="Turn left", command=lambda: self._rotate_pareto_3d(delta_azim=-8.0)).pack(side="left", fill="x", expand=True)
            ttk.Button(turn_row_1, text="Turn right", command=lambda: self._rotate_pareto_3d(delta_azim=8.0)).pack(side="left", fill="x", expand=True, padx=(8, 0))
            turn_row_2 = ttk.Frame(side)
            turn_row_2.pack(fill="x", pady=(8, 0))
            ttk.Button(turn_row_2, text="Tilt up", command=lambda: self._rotate_pareto_3d(delta_elev=5.0)).pack(side="left", fill="x", expand=True)
            ttk.Button(turn_row_2, text="Tilt down", command=lambda: self._rotate_pareto_3d(delta_elev=-5.0)).pack(side="left", fill="x", expand=True, padx=(8, 0))
            proj_swap_row_1 = ttk.Frame(side)
            proj_swap_row_1.pack(fill="x", pady=(8, 0))
            ttk.Button(proj_swap_row_1, text="Swap XY", command=lambda: self._swap_projection_axes_3d(0)).pack(side="left", fill="x", expand=True)
            ttk.Button(proj_swap_row_1, text="Swap XZ", command=lambda: self._swap_projection_axes_3d(1)).pack(side="left", fill="x", expand=True, padx=(8, 0))
            proj_swap_row_2 = ttk.Frame(side)
            proj_swap_row_2.pack(fill="x", pady=(8, 0))
            ttk.Button(proj_swap_row_2, text="Swap YZ", command=lambda: self._swap_projection_axes_3d(2)).pack(side="left", fill="x", expand=True)

        nav_tools = ttk.Frame(side)
        nav_tools.pack(fill="x", pady=(8, 0))
        ttk.Button(nav_tools, text="Zoom box", command=self._toggle_zoom).pack(side="left", fill="x", expand=True)
        ttk.Button(nav_tools, text="Pan", command=self._toggle_pan).pack(side="left", fill="x", expand=True, padx=(8, 0))

        self._toggle_show_all_solutions(redraw=False)

    def _toggle_show_all_solutions(self, redraw: bool = True) -> None:
        visible = bool(self._show_all_var.get())
        for artist in self._feasible_artists:
            try:
                artist.set_visible(visible)
            except Exception:
                pass
        if redraw and self.canvas is not None:
            self.canvas.draw_idle()

    def _current_axis_order_2d(self) -> tuple[int, int]:
        return tuple(int(v) for v in self._axis_order_2d)

    def _rebuild_main_2d_plot(self) -> None:
        if self.ax is None or len(self.objective_names) != 2:
            return
        ix, iy = self._current_axis_order_2d()
        self.ax.clear()
        self._feasible_artists = []
        self._main_feasible_artist = None
        if len(self._feasible_coords) > 0:
            self._main_feasible_artist = self.ax.scatter(
                self._feasible_coords[:, ix],
                self._feasible_coords[:, iy],
                c=self._feasible_closeness,
                cmap=self._feasible_cmap,
                norm=self._feasible_norm,
                s=34,
                alpha=0.88,
                edgecolors="white",
                linewidths=0.28,
                label="Feasible",
            )
            self._feasible_artists.append(self._main_feasible_artist)
        self._pareto_artist = self.ax.scatter(
            self._pareto_coords[:, ix],
            self._pareto_coords[:, iy],
            c="#c62828",
            s=68,
            edgecolors="#7f0000",
            linewidths=0.9,
            picker=10,
            label="Pareto front",
        )
        self._pareto_artists = [self._pareto_artist]
        self._style_pareto_axis_2d(self.ax, self._display_coords, ix, iy)
        self.ax.legend(loc="best")
        self._pareto_initial_limits_2d = (self.ax.get_xlim(), self.ax.get_ylim())
        self._toggle_show_all_solutions(redraw=False)
        if self._selected_index is not None:
            evaluation = self.pareto_front[int(self._selected_index)]
            coords = np.asarray(list(objective_display_values(evaluation, self.objective_names).values()), dtype=float)
            self._highlight_selection(coords)
            self._highlight_parallel_selection(evaluation)
        elif self.canvas is not None:
            self.canvas.draw_idle()

    @staticmethod
    def _base_projection_pairs_3d():
        return [(0, 1), (0, 2), (1, 2)]

    def _projection_pairs_3d(self):
        pairs = []
        for idx, (ix, iy) in enumerate(self._base_projection_pairs_3d()):
            swapped = bool(self._projection_axis_swap_states[idx]) if idx < len(self._projection_axis_swap_states) else False
            pairs.append((iy, ix) if swapped else (ix, iy))
        return pairs

    def _rebuild_projection_2d_views(self) -> None:
        if len(self.objective_names) != 3 or not self._projection_axes_list:
            return
        self._pareto_projection_axes = {}
        self._pareto_projection_limits = {}
        self._pareto_artists = [self._pareto_artist] if self._pareto_artist is not None else []
        self._feasible_artists = [self._main_feasible_artist] if self._main_feasible_artist is not None else []
        for proj_ax, (ix, iy) in zip(self._projection_axes_list, self._projection_pairs_3d()):
            proj_ax.clear()
            artist, feasible_artist = self._scatter_pareto_projection(
                proj_ax,
                self._display_coords,
                self._feasible_coords,
                self._feasible_closeness,
                self._pareto_coords,
                ix,
                iy,
                title="",
            )
            self._pareto_projection_axes[proj_ax] = (ix, iy)
            self._pareto_projection_limits[proj_ax] = (proj_ax.get_xlim(), proj_ax.get_ylim())
            self._pareto_artists.append(artist)
            if feasible_artist is not None:
                self._feasible_artists.append(feasible_artist)
        self._toggle_show_all_solutions(redraw=False)
        if self._selected_index is not None:
            evaluation = self.pareto_front[int(self._selected_index)]
            coords = np.asarray(list(objective_display_values(evaluation, self.objective_names).values()), dtype=float)
            self._highlight_selection(coords)
            self._highlight_parallel_selection(evaluation)
        elif self.canvas is not None:
            self.canvas.draw_idle()

    def _swap_axes_2d(self) -> None:
        if len(self.objective_names) != 2:
            self._status_var.set("Scambio assi disponibile solo per i grafici Pareto a 2 obiettivi.")
            return
        ix, iy = self._current_axis_order_2d()
        self._axis_order_2d = (iy, ix)
        self._rebuild_main_2d_plot()
        self._status_var.set(
            f"Assi scambiati: X={self.objective_names[self._axis_order_2d[0]]}, "
            f"Y={self.objective_names[self._axis_order_2d[1]]}."
        )

    def _swap_projection_axes_3d(self, projection_index: int) -> None:
        if len(self.objective_names) != 3:
            self._status_var.set("Scambio assi 2D disponibile solo nella vista Pareto 3D.")
            return
        if projection_index < 0 or projection_index >= len(self._projection_axis_swap_states):
            return
        self._projection_axis_swap_states[projection_index] = not bool(self._projection_axis_swap_states[projection_index])
        self._rebuild_projection_2d_views()
        base_ix, base_iy = self._base_projection_pairs_3d()[projection_index]
        current_ix, current_iy = self._projection_pairs_3d()[projection_index]
        self._status_var.set(
            f"Proiezione 2D aggiornata: X={self.objective_names[current_ix]}, "
            f"Y={self.objective_names[current_iy]} "
            f"(base: {self.objective_names[base_ix]} vs {self.objective_names[base_iy]})."
        )

    def _rotate_pareto_3d(self, *, delta_azim: float = 0.0, delta_elev: float = 0.0) -> None:
        if self.ax is None or len(self.objective_names) != 3:
            self._status_var.set("Rotazione disponibile solo per il grafico Pareto 3D.")
            return
        current_azim = float(getattr(self.ax, "azim", -60.0))
        current_elev = float(getattr(self.ax, "elev", 30.0))
        self.ax.view_init(elev=current_elev + float(delta_elev), azim=current_azim + float(delta_azim))
        if self.canvas is not None:
            self.canvas.draw_idle()
        self._status_var.set("Vista 3D aggiornata.")

    def _set_detail_text(self, text: str) -> None:
        self._detail_text.configure(state="normal")
        self._detail_text.delete("1.0", "end")
        self._detail_text.insert("1.0", text)
        self._detail_text.configure(state="disabled")

    def _highlight_selection(self, coords: np.ndarray) -> None:
        while self._selection_markers:
            marker = self._selection_markers.pop()
            try:
                marker.remove()
            except Exception:
                pass

        if len(self.objective_names) == 2:
            ix, iy = self._current_axis_order_2d()
            self._selection_markers.append(self.ax.scatter(
                [coords[ix]],
                [coords[iy]],
                s=190,
                facecolors="none",
                edgecolors="gold",
                linewidths=2.0,
                zorder=6,
            ))
        else:
            self._selection_markers.append(self.ax.scatter(
                [coords[0]],
                [coords[1]],
                [coords[2]],
                s=220,
                facecolors="none",
                edgecolors="gold",
                linewidths=2.0,
                depthshade=False,
            ))
            for proj_ax, (ix, iy) in self._pareto_projection_axes.items():
                self._selection_markers.append(proj_ax.scatter(
                    [coords[ix]],
                    [coords[iy]],
                    s=170,
                    facecolors="none",
                    edgecolors="gold",
                    linewidths=2.0,
                    zorder=6,
                ))
        self.canvas.draw_idle()

    def _select_pareto_index(self, idx: int) -> None:
        if idx < 0 or idx >= len(self.pareto_front):
            return

        self._selected_index = idx
        evaluation = self.pareto_front[idx]
        values = objective_display_values(evaluation, self.objective_names)
        coords = np.asarray(list(values.values()), dtype=float)
        self._highlight_selection(coords)

        candidate_id = evaluation.meta.get("candidate_id", "n/a")
        self._candidate_var.set(f"Candidate ID: {candidate_id} | Pareto rank: {idx + 1}")
        lines = [
            f"feasible = {bool(evaluation.feasible)}",
            f"total_violation = {float(evaluation.total_violation):.6f}",
            "",
            "Objectives:",
        ]
        for name, value in values.items():
            lines.append(f"{name} = {float(value):.6f}")
        lines.extend(["", "Constraints:"])
        for key, value in evaluation.constraints.items():
            lines.append(f"{key} = {float(value):.6f}")
        if evaluation.lambda_ref is not None:
            lines.append("")
            lines.append(f"lambda_ref = {float(evaluation.lambda_ref):.6f}")
        lines.append("")
        lines.append(f"Cost = {self._metric_value(evaluation.objectives.get('cost', evaluation.meta.get('cost', float('nan')))):.6f}")
        lines.append(f"EC = {self._metric_value(evaluation.meta.get('ec', float('nan'))):.6f}")
        lines.append(f"PI_M = {self._metric_value(evaluation.objectives.get('pi_mass', evaluation.meta.get('pi_mass', float('nan')))):.6f}")
        lines.append(f"PI_EC = {self._metric_value(evaluation.objectives.get('pi_ec', evaluation.meta.get('pi_ec', float('nan')))):.6f}")
        lines.append(f"PI_MR = {self._metric_value(evaluation.objectives.get('pi_mr', evaluation.meta.get('pi_mr', float('nan')))):.6f}")
        reference_label = str(evaluation.meta.get("comparison_reference_label", "new_reference") or "new_reference")
        reference_prefix = "Reference"
        if "fixed" in reference_label.lower() or "base" in reference_label.lower():
            reference_prefix = "Fixed base"
        lines.append(
            f"{reference_prefix} Mass = "
            f"{self._metric_value(evaluation.meta.get('comparison_reference_mass', evaluation.meta.get('new_reference_mass', float('nan')))):.6f}"
        )
        lines.append(
            f"{reference_prefix} EC = "
            f"{self._metric_value(evaluation.meta.get('comparison_reference_ec', evaluation.meta.get('new_reference_ec', float('nan')))):.6f}"
        )
        lines.append(
            f"{reference_prefix} MR = "
            f"{self._metric_value(evaluation.meta.get('comparison_reference_merchant_rankine', evaluation.meta.get('new_reference_merchant_rankine', float('nan')))):.6f}"
        )
        lines.append(f"DCR_mean = {self._metric_value(evaluation.meta.get('dcr_mean', float('nan'))):.6f}")
        lines.append(f"Mscrap = {self._metric_value(evaluation.meta.get('mscrap', float('nan'))):.6f}")
        lines.append(f"Lscrap = {self._metric_value(evaluation.meta.get('lscrap', 0.0)):.6f}")
        lines.append(f"n_cut = {int(evaluation.meta.get('n_cuts', 0) or 0)}")
        self._set_detail_text("\n".join(lines))
        self._highlight_parallel_selection(evaluation)
        self._update_comparison_chart(evaluation)
        status_message = "Selection ready. You can inspect, run the reused/new reference, or close with this point."
        if self.on_run_fixed_base is not None:
            status_message = "Selection ready. You can inspect, run reused/new/fixed-base analyses, or close with this point."
        self._status_var.set(status_message)

    def _on_pick(self, event) -> None:
        if event.artist not in self._pareto_artists:
            return
        indices = getattr(event, "ind", None)
        if indices is None or len(indices) == 0:
            return
        idx = int(indices[0])
        self._select_pareto_index(idx)

    def _on_click(self, event) -> None:
        if self._zoom_enabled or self._pan_enabled:
            return
        if event.button != 1 or event.inaxes is None:
            return
        if len(self._pareto_coords) == 0:
            return
        if event.x is None or event.y is None:
            return
        if len(self.objective_names) == 2:
            if event.inaxes is not self.ax:
                return
            ix, iy = self._current_axis_order_2d()
            screen_coords = self.ax.transData.transform(self._pareto_coords[:, [ix, iy]])
        else:
            pair = self._pareto_projection_axes.get(event.inaxes, None)
            if pair is None:
                return
            ix, iy = pair
            screen_coords = event.inaxes.transData.transform(self._pareto_coords[:, [ix, iy]])
        deltas = screen_coords - np.asarray([[float(event.x), float(event.y)]], dtype=float)
        distances = np.sqrt((deltas ** 2).sum(axis=1))
        idx = int(np.argmin(distances))
        if float(distances[idx]) <= 14.0:
            self._select_pareto_index(idx)

    def _toggle_zoom(self) -> None:
        if self._toolbar is None:
            return
        if self._pan_enabled:
            self._toolbar.pan()
            self._pan_enabled = False
        self._toolbar.zoom()
        self._zoom_enabled = not self._zoom_enabled
        self._status_var.set(
            "Zoom box attivo: trascina sul grafico Pareto per ingrandire." if self._zoom_enabled else
            "Zoom box disattivato."
        )

    def _toggle_pan(self) -> None:
        if self._toolbar is None:
            return
        if self._zoom_enabled:
            self._toolbar.zoom()
            self._zoom_enabled = False
        self._toolbar.pan()
        self._pan_enabled = not self._pan_enabled
        self._status_var.set(
            "Pan attivo: trascina per spostarti nel grafico." if self._pan_enabled else
            "Pan disattivato."
        )

    def _reset_view(self) -> None:
        if self._toolbar is None:
            return
        if self._zoom_enabled:
            self._toolbar.zoom()
            self._zoom_enabled = False
        if self._pan_enabled:
            self._toolbar.pan()
            self._pan_enabled = False
        if len(self.objective_names) == 2 and self._pareto_initial_limits_2d is not None:
            xlim, ylim = self._pareto_initial_limits_2d
            self.ax.set_xlim(*xlim)
            self.ax.set_ylim(*ylim)
        elif len(self.objective_names) == 3 and self._pareto_initial_limits_3d is not None:
            xlim, ylim, zlim = self._pareto_initial_limits_3d
            self.ax.set_xlim(*xlim)
            self.ax.set_ylim(*ylim)
            self.ax.set_zlim(*zlim)
            for proj_ax, limits in self._pareto_projection_limits.items():
                proj_ax.set_xlim(*limits[0])
                proj_ax.set_ylim(*limits[1])
        else:
            self._toolbar.home()
        self.canvas.draw_idle()
        self._status_var.set("Vista del grafico ripristinata.")

    def _zoom_pareto(self, factor: float) -> None:
        if self.ax is None:
            return
        if self._zoom_enabled and self._toolbar is not None:
            self._toolbar.zoom()
            self._zoom_enabled = False
        if self._pan_enabled and self._toolbar is not None:
            self._toolbar.pan()
            self._pan_enabled = False
        factor = float(factor)
        if factor <= 0.0:
            return
        if len(self.objective_names) == 2:
            x0, x1 = self.ax.get_xlim()
            y0, y1 = self.ax.get_ylim()
            xc = 0.5 * (x0 + x1)
            yc = 0.5 * (y0 + y1)
            hx = 0.5 * (x1 - x0) * factor
            hy = 0.5 * (y1 - y0) * factor
            if hx <= 0.0 or hy <= 0.0:
                return
            self.ax.set_xlim(xc - hx, xc + hx)
            self.ax.set_ylim(yc - hy, yc + hy)
        else:
            x0, x1 = self.ax.get_xlim3d()
            y0, y1 = self.ax.get_ylim3d()
            z0, z1 = self.ax.get_zlim3d()
            xc = 0.5 * (x0 + x1)
            yc = 0.5 * (y0 + y1)
            zc = 0.5 * (z0 + z1)
            hx = 0.5 * (x1 - x0) * factor
            hy = 0.5 * (y1 - y0) * factor
            hz = 0.5 * (z1 - z0) * factor
            if hx <= 0.0 or hy <= 0.0 or hz <= 0.0:
                return
            self.ax.set_xlim(xc - hx, xc + hx)
            self.ax.set_ylim(yc - hy, yc + hy)
            self.ax.set_zlim(zc - hz, zc + hz)
            for proj_ax in self._pareto_projection_axes:
                px0, px1 = proj_ax.get_xlim()
                py0, py1 = proj_ax.get_ylim()
                pxc = 0.5 * (px0 + px1)
                pyc = 0.5 * (py0 + py1)
                phx = 0.5 * (px1 - px0) * factor
                phy = 0.5 * (py1 - py0) * factor
                if phx <= 0.0 or phy <= 0.0:
                    continue
                proj_ax.set_xlim(pxc - phx, pxc + phx)
                proj_ax.set_ylim(pyc - phy, pyc + phy)
        self.canvas.draw_idle()
        self._status_var.set("Zoom aggiornato.")

    def _run_callback_guard(self, action) -> None:
        self.root.config(cursor="watch")
        self.root.update_idletasks()
        try:
            action()
        finally:
            self.root.config(cursor="")
            self.root.update_idletasks()

    def _on_run_selected(self) -> None:
        if self._selected_index is None:
            messagebox.showwarning("No selection", "Select a Pareto point before launching the nonlinear analysis.")
            return
        if self.on_run_selected is None:
            messagebox.showinfo("Run unavailable", "No nonlinear runner callback is configured.")
            return

        idx = int(self._selected_index)
        evaluation = self.pareto_front[idx]

        def _action() -> None:
            self._status_var.set(f"Running nonlinear analysis for Pareto rank {idx + 1}...")
            self.root.update_idletasks()
            try:
                message = self.on_run_selected(evaluation, idx)
            except Exception as exc:
                self._status_var.set(f"Run failed for Pareto rank {idx + 1}.")
                messagebox.showerror("Nonlinear run failed", str(exc))
                return
            self._status_var.set(message or f"Run completed for Pareto rank {idx + 1}.")

        self._run_callback_guard(_action)

    def _on_run_all(self) -> None:
        if self.on_run_all is None:
            messagebox.showinfo("Run unavailable", "No nonlinear runner callback is configured.")
            return
        if not self.pareto_front:
            messagebox.showwarning("No Pareto points", "There are no Pareto points to run.")
            return
        confirm = messagebox.askyesno(
            "Run all Pareto",
            f"Launch the nonlinear analysis for all {len(self.pareto_front)} Pareto solutions?",
        )
        if not confirm:
            return

        def _action() -> None:
            self._status_var.set(f"Running nonlinear analyses for {len(self.pareto_front)} reused Pareto solutions...")
            self.root.update_idletasks()
            try:
                message = self.on_run_all(self.pareto_front)
            except Exception as exc:
                self._status_var.set("Run all reused failed.")
                messagebox.showerror("Run all reused failed", str(exc))
                return
            self._status_var.set(message or f"Completed nonlinear analyses for {len(self.pareto_front)} reused Pareto solutions.")
            messagebox.showinfo("Run all reused completed", self._status_var.get())

        self._run_callback_guard(_action)

    def _on_run_reference_selected(self) -> None:
        if self._selected_index is None:
            messagebox.showwarning("No selection", "Select a Pareto point before launching the new-reference nonlinear analysis.")
            return
        if self.on_run_reference_selected is None:
            messagebox.showinfo("Run unavailable", "No new-reference nonlinear runner callback is configured.")
            return

        idx = int(self._selected_index)
        evaluation = self.pareto_front[idx]

        def _action() -> None:
            self._status_var.set(f"Running new-reference nonlinear analysis for Pareto rank {idx + 1}...")
            self.root.update_idletasks()
            try:
                message = self.on_run_reference_selected(evaluation, idx)
            except Exception as exc:
                self._status_var.set(f"New-reference run failed for Pareto rank {idx + 1}.")
                messagebox.showerror("New-reference nonlinear run failed", str(exc))
                return
            self._status_var.set(message or f"New-reference run completed for Pareto rank {idx + 1}.")

        self._run_callback_guard(_action)

    def _on_run_reference_all(self) -> None:
        if self.on_run_reference_all is None:
            messagebox.showinfo("Run unavailable", "No new-reference nonlinear runner callback is configured.")
            return
        if not self.pareto_front:
            messagebox.showwarning("No Pareto points", "There are no Pareto points to run.")
            return
        confirm = messagebox.askyesno(
            "Run all new references",
            f"Launch the nonlinear analysis for all {len(self.pareto_front)} new-reference solutions?",
        )
        if not confirm:
            return

        def _action() -> None:
            self._status_var.set(f"Running nonlinear analyses for {len(self.pareto_front)} new-reference Pareto solutions...")
            self.root.update_idletasks()
            try:
                message = self.on_run_reference_all(self.pareto_front)
            except Exception as exc:
                self._status_var.set("Run all new-reference failed.")
                messagebox.showerror("Run all new-reference failed", str(exc))
                return
            self._status_var.set(message or f"Completed nonlinear analyses for {len(self.pareto_front)} new-reference Pareto solutions.")
            messagebox.showinfo("Run all new-reference completed", self._status_var.get())

        self._run_callback_guard(_action)

    def _on_run_fixed_base(self) -> None:
        if self.on_run_fixed_base is None:
            messagebox.showinfo("Run unavailable", "No fixed-base nonlinear runner callback is configured.")
            return

        def _action() -> None:
            self._status_var.set("Running fixed-base nonlinear analysis...")
            self.root.update_idletasks()
            try:
                message = self.on_run_fixed_base()
            except Exception as exc:
                self._status_var.set("Fixed-base run failed.")
                messagebox.showerror("Fixed-base nonlinear run failed", str(exc))
                return
            self._status_var.set(message or "Fixed-base nonlinear run completed.")

        self._run_callback_guard(_action)

    def _on_confirm(self) -> None:
        if self._selected_index is None:
            messagebox.showwarning("No selection", "Select a Pareto point before continuing.")
            return
        self.result = ParetoSelectionResult(
            evaluation=self.pareto_front[self._selected_index],
            index=int(self._selected_index),
        )
        self._close_window()

    def _on_continue_generations(self) -> None:
        raw_value = str(self._continue_generations_var.get() or "").strip()
        try:
            extra_generations = int(raw_value)
        except ValueError:
            messagebox.showwarning("Invalid value", "Enter an integer number of extra generations.")
            return
        if extra_generations <= 0:
            messagebox.showwarning("Invalid value", "The number of extra generations must be greater than zero.")
            return
        self.result = ParetoSelectionResult(
            evaluation=None,
            index=None,
            continue_requested=True,
            continue_generations=extra_generations,
        )
        self._close_window()

    def _on_cancel(self) -> None:
        self.result = None
        self._close_window()

    def _cleanup_tk_resources(self) -> None:
        if self._toolbar is not None:
            try:
                self._toolbar.destroy()
            except Exception:
                pass
            self._toolbar = None

        for attr in ("canvas", "_comparison_canvas"):
            canvas = getattr(self, attr, None)
            if canvas is None:
                continue
            try:
                canvas.get_tk_widget().destroy()
            except Exception:
                pass
            try:
                canvas.figure.clear()
            except Exception:
                pass
            setattr(self, attr, None)

        for attr in ("fig", "_comparison_fig"):
            fig = getattr(self, attr, None)
            if fig is None:
                continue
            try:
                fig.clear()
            except Exception:
                pass
            setattr(self, attr, None)

        self.ax = None
        self._comparison_axes = []
        self._pareto_projection_axes = {}
        self._projection_axes_list = []

        for attr in (
            "_candidate_var",
            "_status_var",
            "_show_all_var",
            "_continue_generations_var",
        ):
            try:
                setattr(self, attr, None)
            except Exception:
                pass

    def _close_window(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._cleanup_tk_resources()
        try:
            if self.root.winfo_exists():
                self.root.update_idletasks()
                self.root.destroy()
        except tk.TclError:
            pass
        gc.collect()

    def run(self) -> Optional[ParetoSelectionResult]:
        try:
            self.root.mainloop()
            return self.result
        finally:
            self._close_window()


def select_pareto_solution_gui(
    pareto_front: Sequence[CandidateEvaluation],
    all_evaluations: Sequence[CandidateEvaluation],
    objective_names: Sequence[str],
    *,
    title: str = "Select Pareto solution for nonlinear analysis",
    on_run_selected: Optional[Callable[[CandidateEvaluation, int], str]] = None,
    on_run_all: Optional[Callable[[Sequence[CandidateEvaluation]], str]] = None,
    on_run_reference_selected: Optional[Callable[[CandidateEvaluation, int], str]] = None,
    on_run_reference_all: Optional[Callable[[Sequence[CandidateEvaluation]], str]] = None,
    on_run_fixed_base: Optional[Callable[[], str]] = None,
    save_pareto_only_path: Optional[str] = None,
    save_parallel_coords_path: Optional[str] = None,
) -> Optional[ParetoSelectionResult]:
    try:
        app = ParetoSelectionApp(
            pareto_front=pareto_front,
            all_evaluations=all_evaluations,
            objective_names=objective_names,
            title=title,
            on_run_selected=on_run_selected,
            on_run_all=on_run_all,
            on_run_reference_selected=on_run_reference_selected,
            on_run_reference_all=on_run_reference_all,
            on_run_fixed_base=on_run_fixed_base,
            save_pareto_only_path=save_pareto_only_path,
            save_parallel_coords_path=save_parallel_coords_path,
        )
        return app.run()
    except tk.TclError as exc:
        print(f"# GUI unavailable | falling back to console selection | reason={exc}")
        return _select_pareto_solution_console(pareto_front, objective_names)


def _select_pareto_solution_console(
    pareto_front: Sequence[CandidateEvaluation],
    objective_names: Sequence[str],
) -> Optional[ParetoSelectionResult]:
    if not pareto_front:
        return None
    print("\nPareto candidates:")
    for idx, evaluation in enumerate(pareto_front, start=1):
        candidate_id = evaluation.meta.get("candidate_id", "n/a")
        metrics = objective_display_values(evaluation, objective_names)
        metrics_text = " | ".join(f"{name}={value:.4g}" for name, value in metrics.items())
        print(f"  {idx}: candidate_id={candidate_id} | {metrics_text}")
    if not sys.stdin or not sys.stdin.isatty():
        print("# Console selection skipped: stdin is not interactive.")
        return None
    while True:
        raw = input("\nSelect Pareto rank to continue (Enter to cancel): ").strip()
        if raw == "":
            return None
        try:
            selected = int(raw)
        except ValueError:
            print("Please enter an integer Pareto rank.")
            continue
        if 1 <= selected <= len(pareto_front):
            return ParetoSelectionResult(
                evaluation=pareto_front[selected - 1],
                index=selected - 1,
            )
        print(f"Please enter a value between 1 and {len(pareto_front)}.")


def _safe_float(value, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


_SAVED_OBJECTIVE_COLUMNS = (
    "mass",
    "cost",
    "reuse_ratio",
    "reuse_efficiency",
    "reuse_ratio_members",
    "dcr_max",
    "ec",
    "lscrap",
    "n_cuts",
    "max_abs_uz",
    "lambda_ref",
    "pi_mass",
    "pi_ec",
    "pi_mr",
)


def _restore_saved_objectives(
    objectives: Dict[str, float],
    meta: Dict[str, object],
    source: Dict[str, object],
) -> None:
    for key in _SAVED_OBJECTIVE_COLUMNS:
        raw_value = source.get(key, None)
        if raw_value in {None, ""}:
            continue
        value = _safe_float(raw_value)
        if not np.isfinite(value):
            continue
        if key == "lambda_ref":
            meta.setdefault(key, value)
        elif key in {"dcr_max", "ec", "reuse_ratio_members", "lscrap", "n_cuts", "max_abs_uz"}:
            if key == "n_cuts" and value == int(value):
                meta.setdefault(key, int(value))
            else:
                meta.setdefault(key, value)
        else:
            objectives.setdefault(key, value)


def _candidate_from_pareto_record(record: Dict[str, object]) -> CandidateEvaluation:
    objectives = {str(k): float(v) for k, v in dict(record.get("objectives", {}) or {}).items()}
    constraints = {str(k): float(v) for k, v in dict(record.get("constraints", {}) or {}).items()}
    meta = dict(record.get("meta", {}) or {})
    _restore_saved_objectives(objectives, meta, meta)
    candidate_id = record.get("candidate_id", meta.get("candidate_id", None))
    if candidate_id is not None:
        try:
            meta.setdefault("candidate_id", int(candidate_id))
        except Exception:
            meta.setdefault("candidate_id", candidate_id)
    decision_vector = tuple(float(v) for v in list(record.get("decision_vector", []) or []))
    lambda_ref = record.get("lambda_ref", None)
    lambda_value = None if lambda_ref is None else float(lambda_ref)
    return CandidateEvaluation(
        x=decision_vector,
        feasible=bool(record.get("feasible", False)),
        objectives=objectives,
        constraints=constraints,
        moved_G=None,
        moved_M=None,
        opt_result=None,
        reuse_result=None,
        final_state=None,
        section_by_member=None,
        lambda_ref=lambda_value,
        meta=meta,
    )


def _candidate_from_metrics_row(row: Dict[str, str], objective_names: Sequence[str]) -> CandidateEvaluation:
    objectives: Dict[str, float] = {}
    meta: Dict[str, object] = {}
    for name in objective_names:
        if name == "lambda_ref":
            continue
        if name in {"dcr_max", "max_abs_uz", "reuse_ratio_members", "ec"}:
            meta[name] = _safe_float(row.get(name, None))
        else:
            objectives[name] = _safe_float(row.get(name, None))
    _restore_saved_objectives(objectives, meta, row)
    for key in (
        "cost",
        "reuse_ratio_members",
        "ec",
        "dcr_max",
        "dcr_mean",
        "max_abs_uz",
        "merchant_rankine",
        "reused_mass",
        "new_mass",
        "new_reference_mass",
        "new_reference_mass_tonnes",
        "mscrap",
        "lscrap",
        "n_cuts",
        "new_reference_ec",
        "new_reference_cost",
        "new_reference_dcr_max",
        "new_reference_dcr_mean",
        "new_reference_max_abs_uz",
        "new_reference_lambda_ref",
        "new_reference_merchant_rankine",
        "new_reference_merchant_rankine_lambda",
        "comparison_reference_mass",
        "comparison_reference_mass_tonnes",
        "comparison_reference_ec",
        "comparison_reference_cost",
        "comparison_reference_dcr_max",
        "comparison_reference_dcr_mean",
        "comparison_reference_max_abs_uz",
        "comparison_reference_lambda_ref",
        "comparison_reference_merchant_rankine",
        "comparison_reference_merchant_rankine_lambda",
        "pi_mass",
        "pi_ec",
        "pi_mr",
    ):
        if key in row and row.get(key, "") != "":
            value = _safe_float(row.get(key))
            meta[key] = int(value) if key == "n_cuts" and value == int(value) else value
    for key in ("comparison_reference_mode", "comparison_reference_label"):
        raw_value = row.get(key, "")
        if raw_value != "":
            meta[key] = raw_value
    candidate_id = row.get("candidate_id", "")
    if candidate_id != "":
        try:
            meta["candidate_id"] = int(float(candidate_id))
        except Exception:
            meta["candidate_id"] = candidate_id
    generation = row.get("generation", "")
    if generation != "":
        try:
            meta["generation"] = int(float(generation))
        except Exception:
            meta["generation"] = generation
    decision_vector_raw = row.get("decision_vector", "")
    try:
        decision_vector = tuple(float(v) for v in json.loads(decision_vector_raw)) if decision_vector_raw else tuple()
    except Exception:
        decision_vector = tuple()
    constraints = {
        "g_dcr": _safe_float(row.get("g_dcr", None)),
        "g_disp": _safe_float(row.get("g_disp", None)),
        "g_buckling": _safe_float(row.get("g_buckling", None)),
        "g_merchant_rankine": _safe_float(row.get("g_merchant_rankine", None)),
        "g_analysis": _safe_float(row.get("g_analysis", None)),
    }
    lambda_ref_raw = row.get("lambda_ref", "")
    lambda_ref = None if lambda_ref_raw == "" else _safe_float(lambda_ref_raw)
    return CandidateEvaluation(
        x=decision_vector,
        feasible=_safe_bool(row.get("feasible", "")),
        objectives=objectives,
        constraints=constraints,
        moved_G=None,
        moved_M=None,
        opt_result=None,
        reuse_result=None,
        final_state=None,
        section_by_member=None,
        lambda_ref=lambda_ref,
        meta=meta,
    )


def _candidate_key(evaluation: CandidateEvaluation):
    candidate_id = dict(evaluation.meta or {}).get("candidate_id", None)
    if candidate_id is not None and str(candidate_id).strip() != "":
        try:
            return ("candidate_id", int(float(candidate_id)))
        except Exception:
            return ("candidate_id", str(candidate_id))
    vector = tuple(round(float(v), 12) for v in (getattr(evaluation, "x", None) or ()))
    if vector:
        return ("decision_vector", vector)
    return ("object", id(evaluation))


def _load_metrics_evaluations(path: Optional[str], objective_names: Sequence[str]) -> list[CandidateEvaluation]:
    if not path or not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8", newline="") as fobj:
        return [_candidate_from_metrics_row(row, objective_names) for row in csv.DictReader(fobj)]


def _merge_evaluation_sources(*sources: Sequence[CandidateEvaluation]) -> list[CandidateEvaluation]:
    merged: list[CandidateEvaluation] = []
    seen = set()
    for source in sources:
        for evaluation in source or []:
            key = _candidate_key(evaluation)
            if key in seen:
                continue
            seen.add(key)
            merged.append(evaluation)
    return merged


def load_saved_pareto_front(
    output_dir: Optional[str] = None,
    *,
    pareto_json_path: Optional[str] = None,
    metrics_csv_path: Optional[str] = None,
) -> tuple[list[CandidateEvaluation], list[CandidateEvaluation], tuple[str, ...]]:
    if pareto_json_path is None:
        if not output_dir:
            raise ValueError("Either output_dir or pareto_json_path must be provided.")
        global_pareto_json_path = os.path.join(output_dir, "reuse_genetic_pareto_front_global.json")
        if os.path.exists(global_pareto_json_path):
            pareto_json_path = global_pareto_json_path
        else:
            pareto_json_path = os.path.join(output_dir, "reuse_genetic_pareto_front.json")
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(pareto_json_path))

    with open(pareto_json_path, "r", encoding="utf-8") as fobj:
        raw_records = json.load(fobj)
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError(f"Saved Pareto file is empty or invalid: {pareto_json_path}")

    first_objectives = dict(raw_records[0].get("objectives", {}) or {})
    objective_names = tuple(str(name) for name in first_objectives.keys())
    if len(objective_names) not in {2, 3}:
        raise ValueError(
            f"Saved Pareto GUI supports 2 or 3 objectives, found {len(objective_names)} in {pareto_json_path}."
        )

    pareto_front = [_candidate_from_pareto_record(dict(record)) for record in raw_records]
    metric_paths = []
    if metrics_csv_path is None:
        metric_paths.extend(
            [
                os.path.join(output_dir, "genetic_search", "genetic_candidate_metrics.csv"),
                os.path.join(output_dir, "pareto_candidate_metrics.csv"),
            ]
        )
    else:
        metric_paths.append(metrics_csv_path)
    metric_evaluations = []
    for path in metric_paths:
        metric_evaluations.extend(_load_metrics_evaluations(path, objective_names))

    all_evaluations = _merge_evaluation_sources(metric_evaluations, pareto_front)
    if not all_evaluations:
        all_evaluations = list(pareto_front)
    by_candidate_id = {}
    for evaluation in all_evaluations:
        candidate_id = evaluation.meta.get("candidate_id", None)
        if candidate_id is not None:
            try:
                by_candidate_id[int(candidate_id)] = evaluation
            except Exception:
                pass
    enriched_pareto = []
    for evaluation in pareto_front:
        candidate_id = evaluation.meta.get("candidate_id", None)
        replacement = None
        if candidate_id is not None:
            try:
                replacement = by_candidate_id.get(int(candidate_id), None)
            except Exception:
                replacement = None
        enriched_pareto.append(replacement if replacement is not None else evaluation)
    pareto_front = enriched_pareto
    return pareto_front, all_evaluations, objective_names


def _candidate_saved_metric(evaluation: CandidateEvaluation, *keys: str) -> float:
    for key in keys:
        raw_value = evaluation.lambda_ref if key == "lambda_ref" else None
        if raw_value in {None, ""}:
            raw_value = evaluation.objectives.get(key, None)
        if raw_value in {None, ""}:
            raw_value = evaluation.meta.get(key, None)
        value = _safe_float(raw_value)
        if np.isfinite(value):
            return float(value)
        if key == "mass":
            reused_mass = _safe_float(evaluation.meta.get("reused_mass", None))
            new_mass = _safe_float(evaluation.meta.get("new_mass", None))
            if np.isfinite(reused_mass) and np.isfinite(new_mass):
                return float(reused_mass + new_mass)
    return float("nan")


def _fixed_summary_metric(fixed_reference: Dict[str, object], *keys: str) -> float:
    summary = fixed_reference.get("summary", {}) if isinstance(fixed_reference, dict) else {}
    summary = dict(summary or {}) if isinstance(summary, dict) else {}
    for key in keys:
        raw_value = fixed_reference.get(key, None) if isinstance(fixed_reference, dict) else None
        if raw_value in {None, ""}:
            raw_value = summary.get(key, None)
        value = _safe_float(raw_value)
        if np.isfinite(value):
            return float(value)
    return float("nan")


def _load_fixed_reference_summary(path: Optional[str]) -> Dict[str, object]:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fobj:
            data = json.load(fobj)
    except Exception:
        return {}
    return dict(data) if isinstance(data, dict) else {}


def _fixed_reference_from_pareto_meta(pareto_front: Sequence[CandidateEvaluation]) -> Dict[str, object]:
    for evaluation in pareto_front:
        meta = dict(evaluation.meta or {})
        fixed_reference = {
            "reference_mode": meta.get("comparison_reference_mode", "fixed_base"),
            "label": meta.get("comparison_reference_label", "fixed_base_new"),
            "mass": meta.get("comparison_reference_mass", meta.get("new_reference_mass", None)),
            "ec": meta.get("comparison_reference_ec", meta.get("new_reference_ec", None)),
            "cost": meta.get("comparison_reference_cost", meta.get("new_reference_cost", None)),
            "summary": {
                "dcr_max": meta.get("comparison_reference_dcr_max", meta.get("new_reference_dcr_max", None)),
                "dcr_min": meta.get("comparison_reference_dcr_min", meta.get("new_reference_dcr_min", None)),
                "dcr_mean": meta.get("comparison_reference_dcr_mean", meta.get("new_reference_dcr_mean", None)),
                "max_abs_uz": meta.get("comparison_reference_max_abs_uz", meta.get("new_reference_max_abs_uz", None)),
                "lambda_ref": meta.get("comparison_reference_lambda_ref", meta.get("new_reference_lambda_ref", None)),
                "merchant_rankine": meta.get(
                    "comparison_reference_merchant_rankine",
                    meta.get("new_reference_merchant_rankine", None),
                ),
            },
        }
        values = [
            _fixed_summary_metric(fixed_reference, "mass"),
            _fixed_summary_metric(fixed_reference, "ec"),
            _fixed_summary_metric(fixed_reference, "cost"),
        ]
        if any(np.isfinite(value) for value in values):
            return fixed_reference
    return {}


def _fixed_parallel_metric_defs(
    pareto_front: Sequence[CandidateEvaluation],
    fixed_reference: Dict[str, object],
) -> tuple:
    dcr_min_values = [
        _candidate_saved_metric(evaluation, "dcr_min", "dcrmin")
        for evaluation in pareto_front
    ]
    dcr_min_values.append(_fixed_summary_metric(fixed_reference, "dcr_min", "dcrmin"))
    if any(np.isfinite(value) for value in dcr_min_values):
        dcr_axis = ("DCRmin", ("dcr_min", "dcrmin"), ("dcr_min", "dcrmin"))
    else:
        dcr_axis = ("DCRmean", ("dcr_mean",), ("dcr_mean",))

    return (
        ("Massa", ("mass",), ("mass",)),
        ("EC", ("ec",), ("ec",)),
        ("Costo", ("cost",), ("cost",)),
        ("DCRmax", ("dcr_max",), ("dcr_max",)),
        dcr_axis,
        ("dzmax", ("delta_z_max", "max_abs_uz"), ("delta_z_max", "max_abs_uz")),
        ("LambdaMin", ("lambda_min", "lambda_ref"), ("lambda_min", "lambda_ref")),
        ("MR", ("merchant_rankine",), ("merchant_rankine",)),
    )


def _fixed_parallel_values_from_candidate(
    evaluation: CandidateEvaluation,
    metric_defs: Sequence[tuple],
) -> np.ndarray:
    return np.asarray(
        [_candidate_saved_metric(evaluation, *candidate_keys) for _, candidate_keys, _ in metric_defs],
        dtype=float,
    )


def _fixed_parallel_values_from_reference(
    fixed_reference: Dict[str, object],
    metric_defs: Sequence[tuple],
) -> np.ndarray:
    return np.asarray(
        [_fixed_summary_metric(fixed_reference, *fixed_keys) for _, _, fixed_keys in metric_defs],
        dtype=float,
    )


def _render_fixed_base_parallel_coordinates(
    ax,
    metric_defs: Sequence[tuple],
    pareto_values: Sequence[np.ndarray],
    fixed_values: Optional[np.ndarray],
    *,
    fixed_label: str = "Fixed base",
) -> None:
    labels = [label for label, _, _ in metric_defs]
    n_metrics = len(labels)
    pareto_raw = (
        np.asarray(pareto_values, dtype=float)
        if pareto_values
        else np.empty((0, n_metrics), dtype=float)
    )
    fixed_raw = (
        np.asarray(fixed_values, dtype=float).reshape(1, n_metrics)
        if fixed_values is not None and np.all(np.isfinite(fixed_values))
        else np.empty((0, n_metrics), dtype=float)
    )
    range_parts = []
    if pareto_raw.size:
        range_parts.append(pareto_raw)
    if fixed_raw.size:
        range_parts.append(fixed_raw)
    if not range_parts:
        ax.text(
            0.5,
            0.5,
            "No finite Pareto/fixed-base metrics available.",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontfamily=ParetoSelectionApp._PARALLEL_FONT_FAMILY,
            fontsize=11,
            color="black",
        )
        ax.set_axis_off()
        return

    raw_for_range = np.vstack(range_parts)
    mins = raw_for_range.min(axis=0)
    maxs = raw_for_range.max(axis=0)
    spans = np.where((maxs - mins) > 1e-12, (maxs - mins), 1.0)
    x = np.arange(n_metrics, dtype=float)
    tick_levels = np.linspace(0.0, 1.0, ParetoSelectionApp._PARALLEL_TICK_COUNT)
    tick_label_offset = 0.08
    tick_mark_halfwidth = 0.028

    if pareto_raw.size:
        pareto_normalized = (pareto_raw - mins) / spans
        for row in pareto_normalized:
            ax.plot(x, row, color="#4f5d6a", alpha=0.30, linewidth=1.0, zorder=1)

    for idx, label in enumerate(labels):
        x_pos = x[idx]
        ax.axvline(x_pos, color="black", linewidth=1.2, alpha=1.0, zorder=2)
        axis_tick_values = mins[idx] + tick_levels * (maxs[idx] - mins[idx])
        if abs(maxs[idx] - mins[idx]) <= 1e-12:
            axis_tick_values = np.full_like(tick_levels, mins[idx], dtype=float)
        for level, tick_value in zip(tick_levels, axis_tick_values):
            ax.plot(
                [x_pos - tick_mark_halfwidth, x_pos],
                [level, level],
                color="black",
                linewidth=1.0,
                solid_capstyle="butt",
                zorder=3,
            )
            ax.text(
                x_pos - tick_label_offset,
                level,
                ParetoSelectionApp._format_parallel_value(float(tick_value)),
                ha="right",
                va="center",
                fontsize=9,
                fontfamily=ParetoSelectionApp._PARALLEL_FONT_FAMILY,
                color="black",
                zorder=4,
            )
        ax.text(
            x_pos,
            -0.12,
            label,
            ha="center",
            va="top",
            fontsize=11,
            fontweight="bold",
            fontfamily=ParetoSelectionApp._PARALLEL_FONT_FAMILY,
            color="black",
            zorder=4,
        )

    if fixed_raw.size:
        fixed_normalized = ((fixed_raw[0] - mins) / spans).astype(float)
        ax.plot(
            x,
            fixed_normalized,
            color="#d32f2f",
            linewidth=2.4,
            alpha=0.98,
            marker="o",
            markersize=4.2,
            zorder=5,
        )
        for idx, value in enumerate(fixed_raw[0]):
            y = float(fixed_normalized[idx])
            y_text = min(1.055, max(0.03, y + 0.035))
            ax.text(
                x[idx],
                y_text,
                ParetoSelectionApp._format_parallel_value(float(value)),
                ha="center",
                va="bottom",
                fontsize=8,
                color="#d32f2f",
                zorder=6,
                fontfamily=ParetoSelectionApp._PARALLEL_FONT_FAMILY,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.76, "pad": 0.6},
            )

    ax.set_xlim(-0.4, n_metrics - 1 + 0.35)
    ax.set_ylim(-0.18, 1.08)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor("white")


def _pareto_only_parallel_metric_defs() -> tuple:
    return (
        ("Massa", ("mass",)),
        ("EC", ("ec",)),
        ("PI_Mass", ("pi_mass",)),
        ("PI_Cost", ("pi_cost",)),
        ("PI_EC", ("pi_ec",)),
        ("%Riuso", ("reuse_ratio", "reuse_ratio_length", "reuse_ratio_members")),
        ("Lscrap", ("lscrap",)),
        ("ncut", ("n_cuts", "ncut", "n_cut")),
    )


def _candidate_parallel_metric_defs() -> tuple:
    return (
        ("candidate_id", ("candidate_id",)),
        ("%Riuso", ("reuse_ratio", "reuse_ratio_length", "reuse_ratio_members")),
        ("ReuseEff", ("reuse_efficiency",)),
        ("Massa", ("mass",)),
        ("Cost", ("cost",)),
        ("EC", ("ec",)),
        ("PI_M", ("pi_mass",)),
        ("PI_EC", ("pi_ec",)),
        ("PI_MR", ("pi_mr",)),
        ("DCRmax", ("dcr_max",)),
        ("DCRmean", ("dcr_mean",)),
        ("Lambda", ("lambda_ref",)),
        ("MR", ("merchant_rankine",)),
        ("Mscrap", ("mscrap",)),
        ("dmax", ("max_abs_uz",)),
        ("Lscrap", ("lscrap",)),
        ("n_cut", ("n_cuts", "ncut", "n_cut")),
    )


def _candidate_parallel_values(evaluation: CandidateEvaluation, metric_defs: Sequence[tuple]) -> np.ndarray:
    values = []
    for label, keys in metric_defs:
        value = _candidate_saved_metric(evaluation, *keys)
        if label in {"%Riuso", "ReuseEff"} and np.isfinite(value) and abs(value) <= 1.0:
            value = 100.0 * float(value)
        values.append(value)
    return np.asarray(values, dtype=float)


def _pareto_only_parallel_values(evaluation: CandidateEvaluation, metric_defs: Sequence[tuple]) -> np.ndarray:
    values = []
    for label, keys in metric_defs:
        value = _candidate_saved_metric(evaluation, *keys)
        if label == "%Riuso" and np.isfinite(value) and abs(value) <= 1.0:
            value = 100.0 * float(value)
        values.append(value)
    return np.asarray(values, dtype=float)


def _render_pareto_only_parallel_coordinates(
    ax,
    metric_defs: Sequence[tuple],
    pareto_values: Sequence[np.ndarray],
) -> None:
    labels = [label for label, _ in metric_defs]
    n_metrics = len(labels)
    raw = (
        np.asarray(pareto_values, dtype=float)
        if pareto_values
        else np.empty((0, n_metrics), dtype=float)
    )
    if not raw.size:
        ax.text(
            0.5,
            0.5,
            "No finite Pareto metrics available.",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontfamily=ParetoSelectionApp._PARALLEL_FONT_FAMILY,
            fontsize=11,
            color="black",
        )
        ax.set_axis_off()
        return

    mins = raw.min(axis=0)
    maxs = raw.max(axis=0)
    spans = np.where((maxs - mins) > 1e-12, (maxs - mins), 1.0)
    normalized = (raw - mins) / spans
    x = np.arange(n_metrics, dtype=float)
    tick_levels = np.linspace(0.0, 1.0, ParetoSelectionApp._PARALLEL_TICK_COUNT)
    tick_label_offset = 0.08
    tick_mark_halfwidth = 0.028

    for row in normalized:
        ax.plot(x, row, color="#4f5d6a", alpha=0.36, linewidth=1.05, zorder=1)

    for idx, label in enumerate(labels):
        x_pos = x[idx]
        ax.axvline(x_pos, color="black", linewidth=1.2, alpha=1.0, zorder=2)
        axis_tick_values = mins[idx] + tick_levels * (maxs[idx] - mins[idx])
        if abs(maxs[idx] - mins[idx]) <= 1e-12:
            axis_tick_values = np.full_like(tick_levels, mins[idx], dtype=float)
        for level, tick_value in zip(tick_levels, axis_tick_values):
            ax.plot(
                [x_pos - tick_mark_halfwidth, x_pos],
                [level, level],
                color="black",
                linewidth=1.0,
                solid_capstyle="butt",
                zorder=3,
            )
            ax.text(
                x_pos - tick_label_offset,
                level,
                ParetoSelectionApp._format_parallel_value(float(tick_value)),
                ha="right",
                va="center",
                fontsize=9,
                fontfamily=ParetoSelectionApp._PARALLEL_FONT_FAMILY,
                color="black",
                zorder=4,
            )
        ax.text(
            x_pos,
            -0.12,
            label,
            ha="center",
            va="top",
            fontsize=11,
            fontweight="bold",
            fontfamily=ParetoSelectionApp._PARALLEL_FONT_FAMILY,
            color="black",
            zorder=4,
        )

    ax.set_xlim(-0.4, n_metrics - 1 + 0.35)
    ax.set_ylim(-0.18, 1.08)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor("white")


def save_pareto_fixed_base_parallel_coords_plot(
    output_dir: Optional[str] = None,
    *,
    pareto_front: Optional[Sequence[CandidateEvaluation]] = None,
    fixed_reference: Optional[Dict[str, object]] = None,
    save_path: Optional[str] = None,
    pareto_json_path: Optional[str] = None,
    metrics_csv_path: Optional[str] = None,
    fixed_reference_json_path: Optional[str] = None,
) -> Optional[str]:
    if pareto_front is None:
        pareto_front, _, _ = load_saved_pareto_front(
            output_dir,
            pareto_json_path=pareto_json_path,
            metrics_csv_path=metrics_csv_path,
        )
    pareto_front = list(pareto_front or [])
    if not pareto_front:
        return None

    if output_dir is None:
        if pareto_json_path:
            output_dir = os.path.dirname(os.path.abspath(pareto_json_path))
        elif save_path:
            output_dir = os.path.dirname(os.path.abspath(save_path))
        else:
            output_dir = "outputs"
    if save_path is None:
        save_path = os.path.join(output_dir, "reuse_genetic_parallel_coords_pareto_fixed_base.png")

    if fixed_reference is None:
        fixed_reference = _load_fixed_reference_summary(
            fixed_reference_json_path or os.path.join(output_dir, "genetic_fixed_reference_summary.json")
        )
    if not isinstance(fixed_reference, dict) or not fixed_reference:
        fixed_reference = _fixed_reference_from_pareto_meta(pareto_front)

    metric_defs = _fixed_parallel_metric_defs(pareto_front, fixed_reference)
    pareto_values = []
    for evaluation in pareto_front:
        values = _fixed_parallel_values_from_candidate(evaluation, metric_defs)
        if np.all(np.isfinite(values)):
            pareto_values.append(values)

    fixed_values = None
    if isinstance(fixed_reference, dict) and fixed_reference:
        values = _fixed_parallel_values_from_reference(fixed_reference, metric_defs)
        if np.all(np.isfinite(values)):
            fixed_values = values

    folder = os.path.dirname(save_path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    fig = Figure(figsize=(10.8, 3.55), dpi=160)
    ax = fig.add_subplot(111)
    fixed_label = str((fixed_reference or {}).get("label", "Fixed base") or "Fixed base")
    _render_fixed_base_parallel_coordinates(
        ax,
        metric_defs,
        pareto_values,
        fixed_values,
        fixed_label=fixed_label,
    )
    fig.subplots_adjust(left=0.06, right=0.99, top=0.86, bottom=0.26)
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    return save_path


def save_pareto_metrics_parallel_coords_plot(
    output_dir: Optional[str] = None,
    *,
    pareto_front: Optional[Sequence[CandidateEvaluation]] = None,
    save_path: Optional[str] = None,
    pareto_json_path: Optional[str] = None,
    metrics_csv_path: Optional[str] = None,
) -> Optional[str]:
    if pareto_front is None:
        pareto_front, _, _ = load_saved_pareto_front(
            output_dir,
            pareto_json_path=pareto_json_path,
            metrics_csv_path=metrics_csv_path,
        )
    pareto_front = list(pareto_front or [])
    if not pareto_front:
        return None

    if output_dir is None:
        if pareto_json_path:
            output_dir = os.path.dirname(os.path.abspath(pareto_json_path))
        elif save_path:
            output_dir = os.path.dirname(os.path.abspath(save_path))
        else:
            output_dir = "outputs"
    if save_path is None:
        save_path = os.path.join(output_dir, "reuse_genetic_parallel_coords_pareto_metrics.png")

    metric_defs = _pareto_only_parallel_metric_defs()
    pareto_values = []
    for evaluation in pareto_front:
        values = _pareto_only_parallel_values(evaluation, metric_defs)
        if np.all(np.isfinite(values)):
            pareto_values.append(values)

    folder = os.path.dirname(save_path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    fig = Figure(figsize=(10.8, 3.55), dpi=160)
    ax = fig.add_subplot(111)
    _render_pareto_only_parallel_coordinates(ax, metric_defs, pareto_values)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.86, bottom=0.26)
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    return save_path


def save_candidate_metrics_parallel_coords_plot(
    output_dir: Optional[str] = None,
    *,
    all_evaluations: Optional[Sequence[CandidateEvaluation]] = None,
    save_path: Optional[str] = None,
    pareto_json_path: Optional[str] = None,
    metrics_csv_path: Optional[str] = None,
) -> Optional[str]:
    if all_evaluations is None:
        _, all_evaluations, _ = load_saved_pareto_front(
            output_dir,
            pareto_json_path=pareto_json_path,
            metrics_csv_path=metrics_csv_path,
        )
    all_evaluations = list(all_evaluations or [])
    if not all_evaluations:
        return None

    if output_dir is None:
        if metrics_csv_path:
            output_dir = os.path.dirname(os.path.abspath(metrics_csv_path))
            if os.path.basename(output_dir).lower() == "genetic_search":
                output_dir = os.path.dirname(output_dir)
        elif save_path:
            output_dir = os.path.dirname(os.path.abspath(save_path))
        else:
            output_dir = "outputs"
    if save_path is None:
        save_path = os.path.join(output_dir, "reuse_genetic_parallel_coords.png")

    metric_defs = _candidate_parallel_metric_defs()
    values_rows = []
    for evaluation in all_evaluations:
        values = _candidate_parallel_values(evaluation, metric_defs)
        if np.all(np.isfinite(values)):
            values_rows.append(values)

    folder = os.path.dirname(save_path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    fig = Figure(figsize=(10.8, 3.55), dpi=160)
    ax = fig.add_subplot(111)
    _render_pareto_only_parallel_coordinates(ax, metric_defs, values_rows)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.86, bottom=0.26)
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    return save_path


def reopen_saved_pareto_gui(
    output_dir: Optional[str] = None,
    *,
    pareto_json_path: Optional[str] = None,
    metrics_csv_path: Optional[str] = None,
    title: str = "Saved Pareto results",
) -> Optional[ParetoSelectionResult]:
    pareto_front, all_evaluations, objective_names = load_saved_pareto_front(
        output_dir,
        pareto_json_path=pareto_json_path,
        metrics_csv_path=metrics_csv_path,
    )
    return select_pareto_solution_gui(
        pareto_front,
        all_evaluations,
        objective_names,
        title=title,
    )
