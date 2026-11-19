# file: OS_pushdown_plot.py
# file: OS_pushdown_plot.py
# file: OS_pushdown_plot.py
import numpy as np
from transparent_plots import configure_transparent_matplotlib_saves

configure_transparent_matplotlib_saves()

import matplotlib.pyplot as plt


def _converged(hist, delta_key):
    data = [
        h for h in hist
        if h.get("ok", 0) == 0 and h.get("LF") is not None and h.get(delta_key) is not None
    ]
    if len(data) < 2:
        raise ValueError("Not enough converged steps in hist.")
    return data


def _series(data, key):
    return np.array([np.nan if h.get(key) is None else float(h[key]) for h in data], dtype=float)


def _local_summary(h):
    """Return (any_local:bool, reasons:str, modes:str, flags:dict)."""
    events = h.get("local_instab_events", None) or []
    if not isinstance(events, (list, tuple)):
        events = []
    reasons = []
    modes = []
    for ev in events:
        if isinstance(ev, dict):
            r = ev.get("reason", None)
            m = ev.get("buckling_mode", None)
        else:
            r = None; m = None
        if r:
            reasons.append(str(r))
        if m:
            modes.append(str(m))
    reason_set = sorted(set(reasons))
    mode_set = sorted(set(modes))
    any_local = bool(reason_set or mode_set or bool(h.get("local_instab_any", False)))

    has_energy = any("energy" in r for r in reason_set)
    has_pdrop = any("pdrop" in r or r == "pdrop" for r in reason_set)
    has_mode1 = any(m == "mode1" for m in mode_set)
    has_mode2 = any(m == "mode2" for m in mode_set)
    has_mode3 = any(m == "mode3" for m in mode_set)

    flags = dict(
        energy=has_energy,
        pdrop=has_pdrop,
        mode1=has_mode1,
        mode2=has_mode2,
        mode3=has_mode3,
        combined=(has_energy and has_pdrop),
        pdrop_unknown=(has_pdrop and not (has_mode1 or has_mode2 or has_mode3)),
        energy_only=(has_energy and not has_pdrop),
    )

    return any_local, (",".join(reason_set) if reason_set else "-"), (",".join(mode_set) if mode_set else "-"), flags


def _focus_on_load_factor(fig, ax, x, LF, r_onset, r_full):
    """Only crop the extra view; retain true strain data in the line objects."""
    finite_lf = LF[np.isfinite(LF)]
    peak = max(0.0, float(np.max(finite_lf))) if finite_lf.size else 0.0
    upper = 1.12 * peak if peak > 0.0 else 1.0
    fig.set_size_inches(10.0, 6.8)
    ax.set_ylim(0.0, upper)
    ax.set_ylabel(r"LF / strain ratios [-]")
    ax.grid(True, alpha=0.25)

    # Dense event markers should not obscure the load curve in the detail.
    for line in ax.lines:
        if line.get_marker() not in (None, "None", "", " "):
            line.set_markersize(min(line.get_markersize(), 4.0))
            line.set_markeredgewidth(min(line.get_markeredgewidth(), 1.0))
    ax.lines[0].set_zorder(4)
    for line in ax.lines[3:]:
        line.set_zorder(5)

    clipped = []
    for name, values in (("eps_max/eps_lim", r_onset), ("eps_minmax/eps_lim", r_full)):
        finite = values[np.isfinite(values)]
        if finite.size and np.max(finite) > upper:
            clipped.append(f"{name}: max {np.max(finite):.3g}")
            # Mark where the real curve first leaves the displayed window.
            outside = np.flatnonzero(np.isfinite(values) & (values > upper))
            idx = int(outside[0])
            crossing_x = float(x[idx])
            if idx and np.isfinite(values[idx - 1]) and values[idx - 1] <= upper:
                fraction = (upper - values[idx - 1]) / (values[idx] - values[idx - 1])
                crossing_x = float(x[idx - 1] + fraction * (x[idx] - x[idx - 1]))
            ax.plot(crossing_x, upper, marker="^", markersize=5,
                    color="0.45", clip_on=False, zorder=6)
    note = "Fuori scala: " + "; ".join(clipped) if clipped else ""
    ax.set_title("Pushdown - dettaglio LF" + ("\n" + note if note else ""), fontsize=11)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15),
              ncol=3, fontsize=8, frameon=False)


def plot_LF_delta_with_eps_ratios_abs(
    hist,
    eps_lim,
    delta_key="u",
    delta_mode="abs",
    show=True,
    filename=None,
    eps_abs_max_key="eps_abs_max",
    eps_abs_minmax_key="eps_abs_minmax",
    focus_lf=False,
):
    data = _converged(hist, delta_key)

    delta = np.array([float(h[delta_key]) for h in data], dtype=float)
    x = np.abs(delta) if str(delta_mode).lower() == "abs" else delta.copy()

    LF = _series(data, "LF")
    eps_abs_max = _series(data, eps_abs_max_key)
    eps_abs_minmax = _series(data, eps_abs_minmax_key)

    r_onset = eps_abs_max / float(eps_lim)
    r_full  = eps_abs_minmax / float(eps_lim)

    fig, ax = plt.subplots()

    ax.plot(x, LF, color="black", linewidth=3.0, label="LF")
    ax.plot(x, r_onset, color="0.45", linewidth=1.8, label="eps_abs_max/eps_lim")
    ax.plot(x, r_full, color="0.45", linestyle="--", linewidth=1.8, label="eps_abs_minmax/eps_lim")
    # --- markers: local instability (distinguish energy / mode1/2/3 / combined) ---
    flags_list = [ _local_summary(h)[3] for h in data ]
    mask_energy = np.array([f.get("energy_only", False) for f in flags_list], dtype=bool)
    mask_combined = np.array([f.get("combined", False) for f in flags_list], dtype=bool)
    mask_m1 = np.array([f.get("mode1", False) for f in flags_list], dtype=bool)
    mask_m2 = np.array([f.get("mode2", False) for f in flags_list], dtype=bool)
    mask_m3 = np.array([f.get("mode3", False) for f in flags_list], dtype=bool)
    mask_punk = np.array([f.get("pdrop_unknown", False) for f in flags_list], dtype=bool)

    if np.any(mask_energy):
        ax.plot(x[mask_energy], LF[mask_energy], linestyle="None", marker="s",
                color="black", markersize=6.5, markeredgewidth=1.6, label="local instab: energy")
    if np.any(mask_m1):
        ax.plot(x[mask_m1], LF[mask_m1], linestyle="None", marker="^",
                color="orange", markersize=7.5, markeredgewidth=1.8, label="local instab: mode 1")
    if np.any(mask_m2):
        ax.plot(x[mask_m2], LF[mask_m2], linestyle="None", marker="v",
                color="purple", markersize=7.5, markeredgewidth=1.8, label="local instab: mode 2")
    if np.any(mask_m3):
        ax.plot(x[mask_m3], LF[mask_m3], linestyle="None", marker="D",
                color="red", markersize=7.5, markeredgewidth=1.8, label="local instab: mode 3")
    if np.any(mask_combined):
        ax.plot(x[mask_combined], LF[mask_combined], linestyle="None", marker="*",
                color="green", markersize=10.0, markeredgewidth=1.4, label="local instab: energy+pdrop")
    if np.any(mask_punk):
        ax.plot(x[mask_punk], LF[mask_punk], linestyle="None", marker="x",
                color="gray", markersize=7.0, markeredgewidth=2.0, label="local instab: pdrop (unk)")

    # --- markers: plastic onset = blue open circles ---
    mask_onset = np.array([bool(h.get("plastic_onset_any", False)) for h in data], dtype=bool)
    if np.any(mask_onset):
        ax.plot(
            x[mask_onset], LF[mask_onset],
            linestyle="None", marker="o",
            markerfacecolor="none", markeredgecolor="blue",
            color="blue", markersize=7, markeredgewidth=1.8,
            label="plastic onset"
        )

    # --- markers: full plasticization = red open circles ---
    mask_full = np.array([bool(h.get("plastic_full_any", False)) for h in data], dtype=bool)
    if np.any(mask_full):
        ax.plot(
            x[mask_full], LF[mask_full],
            linestyle="None", marker="o",
            markerfacecolor="none", markeredgecolor="red",
            color="red", markersize=8, markeredgewidth=2.0,
            label="full plasticization"
        )

    ax.set_xlabel(f"|{delta_key}| [m]" if str(delta_mode).lower() == "abs" else f"{delta_key} [m]")
    ax.set_ylabel("LF – strain ratios [-]")
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
    ax.margins(x=0.0, y=0.0)
    ax.grid(True)
    ax.legend(loc="best")

    if focus_lf:
        _focus_on_load_factor(fig, ax, x, LF, r_onset, r_full)

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_LF_delta_ratio_with_eps_ratios_abs(
    hist,
    dlim,
    eps_lim,
    delta_key="u",
    show=True,
    filename=None,
    eps_abs_max_key="eps_abs_max",
    eps_abs_minmax_key="eps_abs_minmax",
    focus_lf=False,
):
    data = _converged(hist, delta_key)

    delta = np.abs(np.array([float(h[delta_key]) for h in data], dtype=float))
    x = delta / float(dlim)

    LF = _series(data, "LF")
    eps_abs_max = _series(data, eps_abs_max_key)
    eps_abs_minmax = _series(data, eps_abs_minmax_key)

    r_onset = eps_abs_max / float(eps_lim)
    r_full  = eps_abs_minmax / float(eps_lim)

    fig, ax = plt.subplots()

    ax.plot(x, LF, color="black", linewidth=3.0, label="LF")
    ax.plot(x, r_onset, color="0.45", linewidth=1.8, label="eps_abs_max/eps_lim")
    ax.plot(x, r_full, color="0.45", linestyle="--", linewidth=1.8, label="eps_abs_minmax/eps_lim")

    mask_local = np.array([bool(h.get("local_instab_any", False)) for h in data], dtype=bool)
    if np.any(mask_local):
        ax.plot(
            x[mask_local], LF[mask_local],
            linestyle="None", marker="x",
            color="orange", markersize=7, markeredgewidth=2.0,
            label="local instability"
        )

    mask_onset = np.array([bool(h.get("plastic_onset_any", False)) for h in data], dtype=bool)
    if np.any(mask_onset):
        ax.plot(
            x[mask_onset], LF[mask_onset],
            linestyle="None", marker="o",
            markerfacecolor="none", markeredgecolor="blue",
            color="blue", markersize=7, markeredgewidth=1.8,
            label="plastic onset"
        )

    mask_full = np.array([bool(h.get("plastic_full_any", False)) for h in data], dtype=bool)
    if np.any(mask_full):
        ax.plot(
            x[mask_full], LF[mask_full],
            linestyle="None", marker="o",
            markerfacecolor="none", markeredgecolor="red",
            color="red", markersize=8, markeredgewidth=2.0,
            label="full plasticization"
        )

    ax.set_xlabel("|delta|/dlim [-]")
    ax.set_ylabel("LF – strain ratios [-]")
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
    ax.margins(x=0.0, y=0.0)
    ax.grid(True)
    ax.legend(loc="best")

    if focus_lf:
        _focus_on_load_factor(fig, ax, x, LF, r_onset, r_full)

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_gmna_results(
    hist,
    u_lim,
    fy,
    E,
    delta_key="Uz",
    delta_mode="abs",
    eps_lim=None,
    show=True,
    save_prefix=None,
    out_dir="outputs",
    show_local_nd_curves=False,
    save_lf_zoom=True,
):
    fy = float(fy)
    E = float(E)
    eps_y = (fy / E) if E != 0.0 else 0.0
    eps_lim = float(eps_y) if eps_lim is None else float(eps_lim)
    if eps_lim == 0.0:
        raise ValueError("eps_lim is zero. Check fy/E or provide eps_lim.")

    figA = plot_LF_delta_with_eps_ratios_abs(
        hist,
        eps_lim=eps_lim,
        delta_key=delta_key,
        delta_mode=delta_mode,
        show=show,
        filename=(None if save_prefix is None else f"{out_dir}/{save_prefix}_LF_eps_vs_delta.png"),
    )

    figB = plot_LF_delta_ratio_with_eps_ratios_abs(
        hist,
        dlim=float(u_lim),
        eps_lim=eps_lim,
        delta_key=delta_key,
        show=show,
        filename=(None if save_prefix is None else f"{out_dir}/{save_prefix}_LF_eps_vs_deltaRatio.png"),
    )

    if save_lf_zoom:
        plot_LF_delta_with_eps_ratios_abs(
            hist, eps_lim=eps_lim, delta_key=delta_key, delta_mode=delta_mode,
            show=show, focus_lf=True,
            filename=(None if save_prefix is None else
                      f"{out_dir}/{save_prefix}_LF_eps_vs_delta_zoom.png"),
        )
        plot_LF_delta_ratio_with_eps_ratios_abs(
            hist, dlim=float(u_lim), eps_lim=eps_lim, delta_key=delta_key,
            show=show, focus_lf=True,
            filename=(None if save_prefix is None else
                      f"{out_dir}/{save_prefix}_LF_eps_vs_deltaRatio_zoom.png"),
        )

    return figA, figB


def save_gmna_history_txt(
    hist,
    u_lim,
    filename="GMNA_history.txt",
    delta_mode="raw",
):
    if u_lim is None:
        raise ValueError("u_lim is required")

    data = [h for h in hist if h.get("LF") is not None and h.get("u") is not None]
    if not data:
        raise ValueError("No valid records in hist to export.")

    with open(filename, "w", encoding="utf-8") as f:
        f.write("# GMNA history export (ABS strain metrics)\n")
        f.write(f"# u_lim = {u_lim}\n")
        f.write("# columns:\n")
        f.write("# step ok LF Ftot u u_abs u_over_ulim eps_abs_max eps_abs_minmax "
                "pl_onset_any pl_full_any n_pl_onset_new n_pl_full_new local_any\n")

        for h in data:
            step = int(h.get("step", -1))
            ok = int(h.get("ok", 0))
            LF = float(h["LF"])
            Ftot = float(h.get("Ftot", float("nan")))

            u = float(h["u"])
            u_abs = abs(u)
            u_plot = u_abs if str(delta_mode).lower() == "abs" else u
            u_over = (u_plot / float(u_lim)) if float(u_lim) != 0.0 else float("nan")

            eps_abs_max = h.get("eps_abs_max", None)
            eps_abs_minmax = h.get("eps_abs_minmax", None)
            eps_abs_max = float("nan") if eps_abs_max is None else float(eps_abs_max)
            eps_abs_minmax = float("nan") if eps_abs_minmax is None else float(eps_abs_minmax)

            pl_onset_any = int(bool(h.get("plastic_onset_any", False)))
            pl_full_any = int(bool(h.get("plastic_full_any", False)))
            n_onset = len(h.get("plastic_onset_new", []) or [])
            n_full = len(h.get("plastic_full_new", []) or [])
            any_local, reason_str, mode_str, _ = _local_summary(h)
            local_any = int(bool(any_local))

            f.write(
                f"{step:6d} {ok:2d} "
                f"{LF: .6e} {Ftot: .6e} "
                f"{u: .6e} {u_abs: .6e} {u_over: .6e} "
                f"{eps_abs_max: .6e} {eps_abs_minmax: .6e} "
                f"{pl_onset_any:d} {pl_full_any:d} {n_onset:d} {n_full:d} {local_any:d} {reason_str} {mode_str}\n"
            )

    return filename


import numpy as np
import matplotlib.pyplot as plt
import openseespy.opensees as ops

def plot_plan_with_element_ids(
    G=None,
    ele_tags=None,
    ax=None,
    show=True,
    filename=None,
    title="Plan view (XY) with element IDs",
    label_every=1,                 # 1 = tutti, 2 = uno sì uno no, ecc.
    fontsize=7,
    linewidth=0.8,
    alpha=0.9,
    use_ops_coords=True,           # True: coord da ops.nodeCoord; False: da G["nodes"]
    skip_zero_length=True,
):
    """
    Plot in pianta (XY) con ID elemento al centro di ogni asta.
    Usa connettività da OpenSees: ops.eleNodes(eleTag).
    """

    if ele_tags is None:
        ele_tags = ops.getEleTags()
    ele_tags = list(ele_tags)
    if not ele_tags:
        raise ValueError("No elements found (ops.getEleTags() empty).")

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    def node_xy(nid):
        nid = int(nid)
        if use_ops_coords:
            x, y, _ = ops.nodeCoord(nid)
            return float(x), float(y)
        if G is None:
            raise ValueError("G is required if use_ops_coords=False")
        x, y, *_ = G["nodes"][nid]
        return float(x), float(y)

    # disegna elementi
    for i, et in enumerate(ele_tags):
        do_label = ((i % int(label_every)) == 0)

        try:
            nn = ops.eleNodes(int(et))
        except Exception:
            continue

        # alcuni elementi possono avere più nodi; usiamo i primi due
        if nn is None or len(nn) < 2:
            continue
        n1, n2 = nn[0], nn[1]

        x1, y1 = node_xy(n1)
        x2, y2 = node_xy(n2)

        dx = x2 - x1
        dy = y2 - y1
        L2 = dx * dx + dy * dy
        if skip_zero_length and L2 < 1e-18:
            continue

        ax.plot([x1, x2], [y1, y2], linewidth=linewidth, alpha=alpha)

        if do_label:
            xm = 0.5 * (x1 + x2)
            ym = 0.5 * (y1 + y2)
            ax.text(xm, ym, str(et), fontsize=fontsize, ha="center", va="center")

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True)
    ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=600, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax

