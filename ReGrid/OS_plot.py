# file: OS_plot.py
from transparent_plots import configure_transparent_matplotlib_saves

configure_transparent_matplotlib_saves()

import matplotlib.pyplot as plt

def _import_opsvis():
    try:
        import opsvis as opsv
        return opsv
    except ImportError as e:
        raise ImportError(
            "opsvis non è installato. Installa con: pip install opsvis"
        ) from e


def plot_model_3d(**plot_kwargs):
    """
    Plot del modello OpenSees corrente (undeformed).

    Tutti i parametri vengono passati direttamente a opsvis.plot_model().
    """
    opsv = _import_opsvis()
    
    opsv.plot_model(**plot_kwargs)
    
    plt.title("OpenSees model")
    plt.tight_layout()
    plt.show()


def plot_deformed_shape_3d(sf=1.0):
    """
    Plot della deformata del modello OpenSees corrente.

    Parameters
    ----------
    sf : float
        Fattore di scala della deformata (argomento POSIZIONALE).
    """
    
    opsv = _import_opsvis()
    opsv.plot_defo(sf)
    #plt.title(f"Deformed shape (sf = {sf})")
    plt.tight_layout()
    plt.show()
import numpy as np
import matplotlib.pyplot as plt
import openseespy.opensees as ops


# file: OS_plot.py
import numpy as np
import matplotlib.pyplot as plt
import openseespy.opensees as ops


import numpy as np
import matplotlib.pyplot as plt
import openseespy.opensees as ops



# file: OS_plot.py  (solo la funzione interessata, completa)
import numpy as np
import matplotlib.pyplot as plt
import openseespy.opensees as ops


def plot_deformed_mesh_mpl(
    G, M,
    sf=1.0,
    ax=None,
    mode="subelements",  # "subelements" oppure "members"
    ele_data=None,

    unstable_ele_tags=None,
    plastic_onset_ele_tags=None,   # NEW
    plastic_full_ele_tags=None,    # NEW

    color_stable="black",
    color_unstable="orange",
    color_plastic_onset="blue",    # NEW
    color_plastic_full="red",      # NEW

    lw_stable=1.2,
    lw_unstable=2.6,
    lw_plastic=2.2,

    show_undeformed=False,
):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    unstable_set = set(int(e) for e in (unstable_ele_tags or []))
    onset_set = set(int(e) for e in (plastic_onset_ele_tags or []))
    full_set = set(int(e) for e in (plastic_full_ele_tags or []))

    def _coord(n):
        c = ops.nodeCoord(int(n))
        return float(c[0]), float(c[1]), float(c[2])

    def _disp(n):
        try:
            ux = float(ops.nodeDisp(int(n), 1))
            uy = float(ops.nodeDisp(int(n), 2))
            uz = float(ops.nodeDisp(int(n), 3))
            return ux, uy, uz
        except Exception:
            return 0.0, 0.0, 0.0

    def _style_for_element(eid, fallback_color=color_stable, fallback_lw=lw_stable):
        e = int(eid)
        # priority: full > unstable > onset > stable
        if e in full_set:
            return color_plastic_full, lw_plastic
        if e in unstable_set:
            return color_unstable, lw_unstable
        if e in onset_set:
            return color_plastic_onset, lw_plastic
        return fallback_color, fallback_lw

    # -------------------- PLOT --------------------
    if mode.lower() == "subelements":
        if ele_data is None or not ele_data.get("ele_tags", None):
            raise ValueError("mode='subelements' richiede ele_data con 'ele_tags'.")

        for e in ele_data["ele_tags"]:
            e = int(e)
            try:
                en = ops.eleNodes(e)
                if en is None or len(en) < 2:
                    continue
                na, nb = int(en[0]), int(en[1])
            except Exception:
                continue

            xa, ya, za = _coord(na)
            xb, yb, zb = _coord(nb)

            if show_undeformed:
                ax.plot([xa, xb], [ya, yb], [za, zb], color=color_stable, linestyle="--", linewidth=0.7)

            uxa, uya, uza = _disp(na)
            uxb, uyb, uzb = _disp(nb)

            cc, lw = _style_for_element(e)

            ax.plot(
                [xa + sf*uxa, xb + sf*uxb],
                [ya + sf*uya, yb + sf*uyb],
                [za + sf*uza, zb + sf*uzb],
                color=cc, linewidth=lw
            )

    elif mode.lower() == "members":
        members = list(M.get("members", []))
        if not members:
            raise ValueError("M['members'] is empty.")

        # se ho ele_data posso mappare member->subEles
        member2sub = {}
        if ele_data is not None and "local_instability" in ele_data:
            for key, info in ele_data["local_instability"].items():
                sub = [int(x) for x in info.get("subEles", [])]
                if sub:
                    member2sub[tuple(key)] = sub

        for (na, nb) in members:
            na, nb = int(na), int(nb)

            xa, ya, za = _coord(na)
            xb, yb, zb = _coord(nb)

            if show_undeformed:
                ax.plot([xa, xb], [ya, yb], [za, zb], color=color_stable, linestyle="--", linewidth=0.7)

            uxa, uya, uza = _disp(na)
            uxb, uyb, uzb = _disp(nb)

            key = (na, nb) if na < nb else (nb, na)

            # stato del member = OR sui sub-elementi
            is_full = False
            is_unstable = False
            is_onset = False

            if key in member2sub:
                subs = member2sub[key]
                is_full = any((int(s) in full_set) for s in subs)
                is_unstable = any((int(s) in unstable_set) for s in subs)
                is_onset = any((int(s) in onset_set) for s in subs)

            if is_full:
                cc, lw = color_plastic_full, lw_plastic
            elif is_unstable:
                cc, lw = color_unstable, lw_unstable
            elif is_onset:
                cc, lw = color_plastic_onset, lw_plastic
            else:
                cc, lw = color_stable, lw_stable

            ax.plot(
                [xa + sf*uxa, xb + sf*uxb],
                [ya + sf*uya, yb + sf*uyb],
                [za + sf*uza, zb + sf*uzb],
                color=cc, linewidth=lw
            )

    else:
        raise ValueError("mode must be 'subelements' or 'members'.")

    # -------------------- AXIS LIMITS (da OpenSees) --------------------
    try:
        tags = ops.getNodeTags()
        xyz = np.array([ops.nodeCoord(int(n)) for n in tags], dtype=float)
        x_min, x_max = xyz[:, 0].min(), xyz[:, 0].max()
        y_min, y_max = xyz[:, 1].min(), xyz[:, 1].max()
        z_min, z_max = xyz[:, 2].min(), xyz[:, 2].max()
        max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
        x_mid = 0.5*(x_max + x_min)
        y_mid = 0.5*(y_max + y_min)
        z_mid = 0.5*(z_max + z_min)
        ax.set_xlim(x_mid - max_range/2, x_mid + max_range/2)
        ax.set_ylim(y_mid - max_range/2, y_mid + max_range/2)
        ax.set_zlim(z_mid - max_range/2, z_mid + max_range/2)
    except Exception:
        pass

    ax.set_axis_off()
    return ax



# file: OS_plot.py
import numpy as np
import matplotlib.pyplot as plt
import openseespy.opensees as ops

def _axial_N_basic(eleTag):
    """Ritorna N = basicForce[0] (OpenSees: trazione +, compressione -)."""
    try:
        bf = ops.eleResponse(int(eleTag), "basicForce")
        if bf is not None and len(bf) > 0:
            return float(bf[0])
    except Exception:
        pass
    return None


def plot_deformed_mesh_tension_compression_mpl(
    G, M,
    sf=1.0,
    ax=None,
    mode="subelements",           # "subelements" oppure "members"
    ele_data=None,                # atteso: {"ele_tags":[...]} e/o {"local_instability":{member_key:{"subEles":[...]}}}
    N0=1e-3,                      # soglia per "quasi zero"
    color_tension="red",
    color_compression="blue",
    color_zero="0.7",             # grigio
    lw=1.6,
    show_undeformed=False,
):
    """
    Plot deformata 3D colorando per segno di N assiale:
      - N > +N0  => trazione (rosso)
      - N < -N0  => compressione (blu)
      - |N|<=N0  => quasi zero (grigio)

    mode:
      - "subelements": disegna tutti gli elementi in ele_data["ele_tags"]
      - "members": disegna solo i members in M["members"] e (se possibile) stima N medio dai subEles
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    def _coord(n):
        c = ops.nodeCoord(int(n))
        return float(c[0]), float(c[1]), float(c[2])

    def _disp(n):
        try:
            ux = float(ops.nodeDisp(int(n), 1))
            uy = float(ops.nodeDisp(int(n), 2))
            uz = float(ops.nodeDisp(int(n), 3))
            return ux, uy, uz
        except Exception:
            return 0.0, 0.0, 0.0

    def _color_from_N(N):
        if N is None or (not np.isfinite(N)):
            return color_zero
        if N > +float(N0):
            return color_tension
        if N < -float(N0):
            return color_compression
        return color_zero

    # -------------------- PLOT --------------------
    if mode.lower() == "subelements":
        if ele_data is None or not ele_data.get("ele_tags", None):
            raise ValueError("mode='subelements' richiede ele_data con 'ele_tags'.")

        for e in ele_data["ele_tags"]:
            e = int(e)
            try:
                en = ops.eleNodes(e)
                if en is None or len(en) < 2:
                    continue
                na, nb = int(en[0]), int(en[1])
            except Exception:
                continue

            xa, ya, za = _coord(na)
            xb, yb, zb = _coord(nb)

            if show_undeformed:
                ax.plot([xa, xb], [ya, yb], [za, zb], color="0.85", linestyle="--", linewidth=0.7)

            uxa, uya, uza = _disp(na)
            uxb, uyb, uzb = _disp(nb)

            N = _axial_N_basic(e)
            cc = _color_from_N(N)

            ax.plot(
                [xa + sf*uxa, xb + sf*uxb],
                [ya + sf*uya, yb + sf*uyb],
                [za + sf*uza, zb + sf*uzb],
                color=cc, linewidth=lw
            )

    elif mode.lower() == "members":
        members = list(M.get("members", []))
        if not members:
            raise ValueError("M['members'] is empty.")

        # mappa member_key -> subEles (se disponibile)
        member2sub = {}
        if ele_data is not None and "local_instability" in ele_data:
            for key, info in ele_data["local_instability"].items():
                sub = [int(x) for x in info.get("subEles", [])]
                if sub:
                    member2sub[tuple(key)] = sub

        for (na, nb) in members:
            na, nb = int(na), int(nb)
            xa, ya, za = _coord(na)
            xb, yb, zb = _coord(nb)

            if show_undeformed:
                ax.plot([xa, xb], [ya, yb], [za, zb], color="0.85", linestyle="--", linewidth=0.7)

            uxa, uya, uza = _disp(na)
            uxb, uyb, uzb = _disp(nb)

            key = (na, nb) if na < nb else (nb, na)

            # stima N del member = media N dei sub-elementi (se ho la mappa)
            Nmem = None
            if key in member2sub:
                Ns = []
                for e in member2sub[key]:
                    N = _axial_N_basic(e)
                    if N is not None and np.isfinite(N):
                        Ns.append(float(N))
                if Ns:
                    Nmem = float(np.mean(Ns))

            cc = _color_from_N(Nmem)

            ax.plot(
                [xa + sf*uxa, xb + sf*uxb],
                [ya + sf*uya, yb + sf*uyb],
                [za + sf*uza, zb + sf*uzb],
                color=cc, linewidth=lw
            )

    else:
        raise ValueError("mode must be 'subelements' or 'members'.")

    # -------------------- AXIS LIMITS --------------------
    try:
        tags = ops.getNodeTags()
        xyz = np.array([ops.nodeCoord(int(n)) for n in tags], dtype=float)
        x_min, x_max = xyz[:, 0].min(), xyz[:, 0].max()
        y_min, y_max = xyz[:, 1].min(), xyz[:, 1].max()
        z_min, z_max = xyz[:, 2].min(), xyz[:, 2].max()
        max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
        x_mid = 0.5*(x_max + x_min)
        y_mid = 0.5*(y_max + y_min)
        z_mid = 0.5*(z_max + z_min)
        ax.set_xlim(x_mid - max_range/2, x_mid + max_range/2)
        ax.set_ylim(y_mid - max_range/2, y_mid + max_range/2)
        ax.set_zlim(z_mid - max_range/2, z_mid + max_range/2)
    except Exception:
        pass

    ax.set_axis_off()
    return ax



# =========================================================================================
# ENERGY MAP 3D (DISCRETE): black (low) -> blue -> ... -> red (high)
# =========================================================================================
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import openseespy.opensees as ops


def _setup_3d_view(
    ax,
    elev=25,
    azim=-60,
    show_axes=False,
    show_grid=False,
    background="none",
):
    """Apply a clean 3D view similar to the pushdown plots."""
    try:
        ax.view_init(elev=float(elev), azim=float(azim))
    except Exception:
        pass

    # equal-ish axes
    try:
        tags = ops.getNodeTags()
        xyz = np.array([ops.nodeCoord(int(n)) for n in tags], dtype=float)
        x_min, x_max = float(xyz[:, 0].min()), float(xyz[:, 0].max())
        y_min, y_max = float(xyz[:, 1].min()), float(xyz[:, 1].max())
        z_min, z_max = float(xyz[:, 2].min()), float(xyz[:, 2].max())
        max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
        if max_range <= 0:
            max_range = 1.0
        x_mid = 0.5 * (x_max + x_min)
        y_mid = 0.5 * (y_max + y_min)
        z_mid = 0.5 * (z_max + z_min)
        ax.set_xlim(x_mid - max_range / 2.0, x_mid + max_range / 2.0)
        ax.set_ylim(y_mid - max_range / 2.0, y_mid + max_range / 2.0)
        ax.set_zlim(z_mid - max_range / 2.0, z_mid + max_range / 2.0)
    except Exception:
        pass

    if not show_axes:
        ax.set_axis_off()
    if not show_grid:
        try:
            ax.grid(False)
        except Exception:
            pass

    # transparent panes (nicer)
    try:
        ax.xaxis.pane.set_alpha(0.0)
        ax.yaxis.pane.set_alpha(0.0)
        ax.zaxis.pane.set_alpha(0.0)
    except Exception:
        pass

    # figure background
    try:
        if ax.figure is not None:
            ax.figure.patch.set_alpha(0.0 if str(background).lower() in ("none", "transparent") else 1.0)
    except Exception:
        pass
def energy_to_color_discrete_10(dU, max_dU_pos, min_dU_neg):
    """
    10 colori totali:
    - 4 negativi (grigi -> nero)
    - 1 zero (blu)
    - 5 positivi (blu -> rosso)
    """

    # ZERO esatto
    if dU == 0.0:
        return (0.0, 0.0, 1.0)  # blu

    # NEGATIVI
    if dU < 0.0:
        if min_dU_neg >= 0.0:
            return (0.75, 0.75, 0.75)

        v = dU / abs(min_dU_neg)  # ∈ [-1, 0)

        if v < -0.75:
            return (0.0, 0.0, 0.0)        # nero
        elif v < -0.50:
            return (0.35, 0.35, 0.35)     # grigio scuro
        elif v < -0.25:
            return (0.55, 0.55, 0.55)     # grigio
        else:
            return (0.75, 0.75, 0.75)     # grigio chiaro

    # POSITIVI
    else:
        if max_dU_pos <= 0.0:
            return (0.0, 0.0, 1.0)

        v = dU / max_dU_pos  # ∈ (0,1]

        if v <= 0.20:
            return (0.0, 0.0, 1.0)        # blu
        elif v <= 0.40:
            return (0.0, 0.8, 0.0)        # verde
        elif v <= 0.60:
            return (1.0, 1.0, 0.0)        # giallo
        elif v <= 0.80:
            return (1.0, 0.6, 0.0)        # arancione
        else:
            return (1.0, 0.0, 0.0)        # rosso


def plot_energy_increment_map_3d_discrete(
    ele_tags,
    DU_ele_map,
    outpath,
    step=None,
    load_factor=None,
    show_deformed=True,
    scale_factor=1.0,
    # --- normalization ---
    normalize=True,  # kept for compatibility (normalization happens inside energy_to_color_discrete_10 via max/min)
    # --- discrete bins ---
    n_bins=10,       # kept for compatibility (fixed to 10 in the requested scheme)
    # --- view/style ---
    linewidth=1.8,
    elev=25,
    azim=-60,
    figsize=(9, 7),
    show_axes=False,
    show_grid=False,
    show_title=False,
    show_colorbar=True,  # NOTE: handled as discrete legend (not matplotlib continuous colorbar)
    **kwargs,
):
    """
    3D DISCRETE color map for per-element incremental strain energy ΔU_e.

    Requested style (10 colors TOTAL):
      - 4 negatives: light gray -> gray -> dark gray -> black
      - 1 zero: BLUE
      - 5 positives: BLUE -> GREEN -> YELLOW -> ORANGE -> RED

    Implementation:
      - Uses energy_to_color_discrete_10(dU, max_dU_pos, min_dU_neg) to assign an RGB color per element.
      - No smooth colormap / no BoundaryNorm.
      - Optional discrete legend (recommended). A continuous matplotlib colorbar is not meaningful here.

    kwargs:
      - zero_snap: float (default 0.0)
          If >0, values with |dU| <= zero_snap are forced to 0.0 (BLUE),
          useful to make "vicino allo zero" clearly blue in raw units.
          (This acts BEFORE normalization inside energy_to_color_discrete_10.)
    """
    if ele_tags is None or DU_ele_map is None:
        return

    # optional tuning in raw units (before normalization)
    zero_snap = float(kwargs.pop("zero_snap", 0.0))
    if zero_snap < 0.0:
        zero_snap = 0.0

    # --- collect values once ---
    vals = []
    for e in ele_tags:
        v = DU_ele_map.get(int(e), None)
        if v is not None and np.isfinite(v):
            vv = float(v)
            if zero_snap > 0.0 and abs(vv) <= zero_snap:
                vv = 0.0
            vals.append(vv)

    if not vals:
        return

    # --- signed extrema for normalization inside the color function ---
    pos = [v for v in vals if v > 0.0]
    neg = [v for v in vals if v < 0.0]
    max_dU_pos = max(pos) if pos else 0.0
    min_dU_neg = min(neg) if neg else 0.0  # negative number or 0

    # ---- build segments + per-segment colors ----
    segs = []
    colors = []

    for e in ele_tags:
        e = int(e)
        v = DU_ele_map.get(e, None)
        if v is None or (not np.isfinite(v)):
            continue
        v = float(v)

        if zero_snap > 0.0 and abs(v) <= zero_snap:
            v = 0.0

        # element nodes
        try:
            en = ops.eleNodes(e)
            if en is None or len(en) < 2:
                continue
            na, nb = int(en[0]), int(en[1])
        except Exception:
            continue

        # node coords
        try:
            xa, ya, za = ops.nodeCoord(na)
            xb, yb, zb = ops.nodeCoord(nb)
            xa, ya, za = float(xa), float(ya), float(za)
            xb, yb, zb = float(xb), float(yb), float(zb)
        except Exception:
            continue

        # deformed shape
        if show_deformed:
            try:
                uxa = float(ops.nodeDisp(na, 1)); uya = float(ops.nodeDisp(na, 2)); uza = float(ops.nodeDisp(na, 3))
            except Exception:
                uxa = uya = uza = 0.0
            try:
                uxb = float(ops.nodeDisp(nb, 1)); uyb = float(ops.nodeDisp(nb, 2)); uzb = float(ops.nodeDisp(nb, 3))
            except Exception:
                uxb = uyb = uzb = 0.0

            xa += float(scale_factor) * uxa
            ya += float(scale_factor) * uya
            za += float(scale_factor) * uza
            xb += float(scale_factor) * uxb
            yb += float(scale_factor) * uyb
            zb += float(scale_factor) * uzb

        segs.append([(xa, ya, za), (xb, yb, zb)])

        # --- IMPORTANT: color assignment is discrete and centered on 0 ---
        # (normalization is internal: v/max_dU_pos or v/|min_dU_neg|)
        col = energy_to_color_discrete_10(dU=v, max_dU_pos=max_dU_pos, min_dU_neg=min_dU_neg)
        colors.append(col)

    if not segs:
        return

    # ---- plot ----
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    lc = Line3DCollection(segs, colors=colors, linewidths=float(linewidth))
    ax.add_collection3d(lc)

    _setup_3d_view(ax, elev=elev, azim=azim, show_axes=show_axes, show_grid=show_grid, background="none")

    if show_title:
        t = "Energy increment map (discrete, signed)"
        if step is not None:
            t += f" | step={int(step)}"
        if load_factor is not None:
            t += f" | LF={float(load_factor):.4g}"
        ax.set_title(t)

    # ---- discrete legend (10 colors total) ----
    # A continuous colorbar is not meaningful because we do not use a continuous cmap/norm.
    if show_colorbar:
        from matplotlib.patches import Patch

        # Labels reflect the internal thresholds of energy_to_color_discrete_10
        legend_handles = [
            Patch(facecolor=(0.0, 0.0, 0.0), label="ΔU < -0.75·|ΔU⁻max|"),
            Patch(facecolor=(0.35, 0.35, 0.35), label="-0.75..-0.50·|ΔU⁻max|"),
            Patch(facecolor=(0.55, 0.55, 0.55), label="-0.50..-0.25·|ΔU⁻max|"),
            Patch(facecolor=(0.75, 0.75, 0.75), label="-0.25..0"),
            Patch(facecolor=(0.0, 0.0, 1.0), label="≈ 0 (BLUE pivot)"),
            Patch(facecolor=(0.0, 0.8, 0.0), label="0..0.40·ΔU⁺max (GREEN bin starts at 0.20)"),
            Patch(facecolor=(1.0, 1.0, 0.0), label="0.40..0.60·ΔU⁺max"),
            Patch(facecolor=(1.0, 0.6, 0.0), label="0.60..0.80·ΔU⁺max"),
            Patch(facecolor=(1.0, 0.0, 0.0), label="> 0.80·ΔU⁺max"),
        ]

        # Make the legend compact and readable
        ax.legend(
            handles=legend_handles,
            loc="upper left",
            bbox_to_anchor=(0.02, 0.98),
            frameon=True,
            framealpha=0.85,
            fontsize=8,
            title="ΔU bins (discrete)",
            title_fontsize=9,
        )

    plt.tight_layout()
    try:
        plt.savefig(str(outpath), dpi=300, bbox_inches="tight")
    except Exception:
        pass
    plt.close(fig)


