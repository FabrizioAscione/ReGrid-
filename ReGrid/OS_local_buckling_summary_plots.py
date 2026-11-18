import os
from typing import Dict, List, Tuple, Optional

import numpy as np
from transparent_plots import configure_transparent_matplotlib_saves

configure_transparent_matplotlib_saves()

import matplotlib.pyplot as plt


def _ensure_dir(path: str):
    if not path:
        return
    os.makedirs(path, exist_ok=True)


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def _converged(hist, delta_key: str):
    return [
        h for h in hist
        if h.get("ok", 0) == 0 and h.get("LF") is not None and h.get(delta_key) is not None
    ]


def _norm_max_abs(arr) -> Tuple[np.ndarray, float]:
    arr = np.asarray(arr, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.full_like(arr, np.nan, dtype=float), np.nan
    vmax = float(np.nanmax(np.abs(finite)))
    if not np.isfinite(vmax) or vmax <= 0.0:
        return np.full_like(arr, np.nan, dtype=float), vmax
    return arr / vmax, vmax


def _norm_max_pos(arr) -> Tuple[np.ndarray, float]:
    arr = np.asarray(arr, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.full_like(arr, np.nan, dtype=float), np.nan
    vmax = float(np.nanmax(finite))
    if not np.isfinite(vmax) or vmax <= 0.0:
        return np.full_like(arr, np.nan, dtype=float), vmax
    return arr / vmax, vmax


def _get_local_curves(hist) -> Dict:
    if not hist or not isinstance(hist[0], dict):
        return {}
    curves = hist[0].get("local_post_nd", None)
    return curves if isinstance(curves, dict) else {}


def _prepare_member_curve(curve: Dict) -> Dict:
    c = dict(curve)
    N = np.asarray(c.get("N_norm", c.get("N_member", [])), dtype=float)
    if "N_norm" not in c:
        N, Nmax = _norm_max_pos(np.asarray(c.get("N_member", []), dtype=float))
        c["N_norm"] = N
        c["N_max"] = Nmax

    d_end = np.asarray(c.get("d_end_norm", c.get("d_end", [])), dtype=float)
    if "d_end_norm" not in c:
        d_end, d_end_max = _norm_max_abs(np.asarray(c.get("d_end", []), dtype=float))
        c["d_end_norm"] = d_end
        c["d_end_max"] = d_end_max

    d_mid = np.asarray(c.get("d_mid_norm", c.get("d_mid", [])), dtype=float)
    if "d_mid_norm" not in c:
        d_mid, d_mid_max = _norm_max_abs(np.asarray(c.get("d_mid", []), dtype=float))
        c["d_mid_norm"] = d_mid
        c["d_mid_max"] = d_mid_max

    c["event_idx"] = int(c.get("event_idx", 0) or 0)
    c["event_alpha"] = _safe_float(c.get("event_alpha", 1.0))
    if not np.isfinite(c["event_alpha"]):
        c["event_alpha"] = 1.0
    c["step"] = np.asarray(c.get("step", []), dtype=float)
    return c


def _interp_curve_value(arr, idx: int, alpha: float) -> float:
    values = np.asarray(arr, dtype=float)
    if values.size == 0:
        return np.nan
    idx = int(idx)
    alpha = float(max(0.0, min(1.0, alpha)))
    if idx <= 0 or idx >= values.size:
        return float(values[min(max(idx, 0), values.size - 1)])
    v0 = values[idx - 1]
    v1 = values[idx]
    if not (np.isfinite(v0) and np.isfinite(v1)):
        return float(v1 if np.isfinite(v1) else v0)
    return float(v0 + alpha * (v1 - v0))


def _sorted_member_ids(curves: Dict, n_members: Optional[int], sort_by: str = "step") -> List:
    def key_fun(mid):
        c = curves[mid]
        mode = str(sort_by).lower()
        if mode == "n_peak":
            return (-_safe_float(c.get("N_max", np.nan)), str(mid))
        if mode == "member":
            return (str(mid),)
        idx = int(c.get("event_idx", 0) or 0)
        step = np.asarray(c.get("step", []), dtype=float)
        step_val = float(step[idx]) if step.size and idx < step.size else np.inf
        return (step_val, str(mid))

    mids = sorted(curves.keys(), key=key_fun)
    if n_members is None:
        return mids
    return mids[: max(0, int(n_members))]


def plot_gmna_vs_local_buckling_normalized(
    hist,
    n_members: int = 10,
    delta_key: str = "Uz",
    delta_mode: str = "abs",
    sort_by: str = "step",
    show: bool = True,
    out_dir: Optional[str] = None,
    save_prefix: str = "GMNA_local_summary",
    member_labels: bool = True,
):
    """
    Crea due figure riassuntive:
      1) LF/LFmax vs delta/delta_max + prime n curve membro (N/Nmax vs d_mid/d_midmax)
      2) LF/LFmax vs delta/delta_max + prime n curve membro (N/Nmax vs d_end/d_endmax)

    In entrambe, quando un membro si instabilizza, viene marcato il corrispondente
    punto evento sulla curva globale LF-delta con un pallino dello stesso colore.
    """
    data = _converged(hist, delta_key=delta_key)
    if len(data) < 2:
        raise ValueError("Not enough converged steps in hist.")

    steps_global = np.array([int(h.get("step", -1)) for h in data], dtype=int)
    delta = np.array([_safe_float(h.get(delta_key)) for h in data], dtype=float)
    if str(delta_mode).lower() == "abs":
        delta = np.abs(delta)
    delta_norm, delta_max = _norm_max_abs(delta)

    LF = np.array([_safe_float(h.get("LF")) for h in data], dtype=float)
    LF_norm, LF_max = _norm_max_pos(LF)

    step_to_global_idx = {int(st): i for i, st in enumerate(steps_global)}

    curves_raw = _get_local_curves(hist)
    if not curves_raw:
        raise ValueError("No local post-process curves found in hist[0]['local_post_nd'].")

    curves = {mid: _prepare_member_curve(c) for mid, c in curves_raw.items()}
    selected = _sorted_member_ids(curves, n_members=n_members, sort_by=sort_by)
    if not selected:
        raise ValueError("No member curves available after selection.")

    fig_mid, ax_mid = plt.subplots(figsize=(8, 6))
    fig_end, ax_end = plt.subplots(figsize=(8, 6))

    for ax in (ax_mid, ax_end):
        ax.plot(delta_norm, LF_norm, color="black", linewidth=2.8, label="structure LF/LFmax", zorder=1)
        ax.set_xlim(left=0.0)
        ax.set_ylim(bottom=0.0)
        ax.margins(x=0.0, y=0.0)
        ax.grid(True)
        ax.set_ylabel("LF / LFmax")

    ax_mid.set_xlabel("dmid / dmid,max   or   delta / delta,max")
    ax_end.set_xlabel("dend / dend,max   or   delta / delta,max")
    ax_mid.set_title("Normalized GMNA + local curves (P-dmid)")
    ax_end.set_title("Normalized GMNA + local curves (P-dend)")

    for mid in selected:
        c = curves[mid]
        idx = int(c.get("event_idx", 0))
        event_alpha = float(c.get("event_alpha", 1.0))

        x_mid = np.asarray(c.get("d_mid_norm", []), dtype=float)
        x_end = np.asarray(c.get("d_end_norm", []), dtype=float)
        y = np.asarray(c.get("N_norm", []), dtype=float)

        lbl = f"member {mid}" if member_labels else None
        line_mid, = ax_mid.plot(x_mid, y, linewidth=1.25, label=lbl, zorder=2)
        color = line_mid.get_color()
        ax_end.plot(x_end, y, linewidth=1.25, label=lbl, color=color, zorder=2)

        x_mid_event = _safe_float(c.get("event_d_mid_norm", np.nan))
        if not np.isfinite(x_mid_event):
            x_mid_event = _interp_curve_value(x_mid, idx, event_alpha)
        x_end_event = _safe_float(c.get("event_d_end_norm", np.nan))
        if not np.isfinite(x_end_event):
            x_end_event = _interp_curve_value(x_end, idx, event_alpha)
        y_event = _safe_float(c.get("event_N_norm", np.nan))
        if not np.isfinite(y_event):
            y_event = _interp_curve_value(y, idx, event_alpha)

        if np.isfinite(x_mid_event) and np.isfinite(y_event):
            ax_mid.plot(x_mid_event, y_event, linestyle="None", marker="o", markersize=4, color=color, zorder=3)
        if np.isfinite(x_end_event) and np.isfinite(y_event):
            ax_end.plot(x_end_event, y_event, linestyle="None", marker="o", markersize=4, color=color, zorder=3)

        # matching event marker on the global LF-delta curve
        event_delta_raw = _safe_float(c.get(f"event_{delta_key}", np.nan))
        if np.isfinite(event_delta_raw):
            if str(delta_mode).lower() == "abs":
                event_delta_raw = abs(event_delta_raw)
            global_x_event = float(event_delta_raw / delta_max) if np.isfinite(delta_max) and abs(delta_max) > 0.0 else np.nan
        else:
            global_x_event = np.nan
        if not np.isfinite(global_x_event):
            step_arr = np.asarray(c.get("step", []), dtype=float)
            if idx < len(step_arr):
                ev_step = int(step_arr[idx])
                gidx = step_to_global_idx.get(ev_step)
                if gidx is not None and np.isfinite(delta_norm[gidx]):
                    global_x_event = float(delta_norm[gidx])
        global_y_event = _safe_float(c.get("event_LF_norm", np.nan))
        if not np.isfinite(global_y_event):
            step_arr = np.asarray(c.get("step", []), dtype=float)
            if idx < len(step_arr):
                ev_step = int(step_arr[idx])
                gidx = step_to_global_idx.get(ev_step)
                if gidx is not None and np.isfinite(LF_norm[gidx]):
                    global_y_event = float(LF_norm[gidx])
        if np.isfinite(global_x_event) and np.isfinite(global_y_event):
            ax_mid.plot(global_x_event, global_y_event, linestyle="None", marker="o",
                        markersize=5, color=color, zorder=4)
            ax_end.plot(global_x_event, global_y_event, linestyle="None", marker="o",
                        markersize=5, color=color, zorder=4)

    if member_labels and len(selected) <= 15:
        ax_mid.legend(loc="best", fontsize=8)
        ax_end.legend(loc="best", fontsize=8)

    fig_mid.tight_layout()
    fig_end.tight_layout()

    if out_dir:
        _ensure_dir(out_dir)
        fig_mid.savefig(os.path.join(out_dir, f"{save_prefix}_P_dmid.png"), dpi=300, bbox_inches="tight")
        fig_end.savefig(os.path.join(out_dir, f"{save_prefix}_P_dend.png"), dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig_mid)
        plt.close(fig_end)

    meta = {
        "n_members_plotted": len(selected),
        "member_ids": selected,
        "delta_max": None if not np.isfinite(delta_max) else float(delta_max),
        "LF_max": None if not np.isfinite(LF_max) else float(LF_max),
    }
    return fig_mid, fig_end, meta
