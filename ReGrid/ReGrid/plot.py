import numpy as np
from transparent_plots import configure_transparent_matplotlib_saves

configure_transparent_matplotlib_saves()

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib import colors as mcolors

#Plot punti generati della geometria
def plot_geometry(G, show_ids=False, elev=25, azim=-60):
    nodes = G["nodes"]
    node_sets = G.get("node_sets", {})
    boundary = set(node_sets.get("boundary", []))
    corners = set(node_sets.get("corners", []))
    # Converti in array
    ids = np.array(sorted(nodes.keys()), dtype=int)
    xyz = np.array([nodes[i] for i in ids], dtype=float)
    # Maschere
    is_boundary = np.array([i in boundary for i in ids])
    is_corner = np.array([i in corners for i in ids])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    # Punti interni
    ax.scatter(xyz[~is_boundary, 0], xyz[~is_boundary, 1], xyz[~is_boundary, 2], s=10)
    # Boundary
    if np.any(is_boundary):
        ax.scatter(xyz[is_boundary, 0], xyz[is_boundary, 1], xyz[is_boundary, 2], s=25)
    # Corners
    if np.any(is_corner):
        ax.scatter(xyz[is_corner, 0], xyz[is_corner, 1], xyz[is_corner, 2], s=60)
    # Opzionale: ID nodi
    if show_ids:
        for nid, (x, y, z) in nodes.items():
            ax.text(x, y, z, str(nid), fontsize=8)
    # Labels e titolo
    meta = G.get("meta", {})
    geom = meta.get("geom", "?")
    grid = meta.get("grid", "?")
    ax.set_title(f"Geometry geom={geom} | grid={grid} | N={len(nodes)}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    # Vista e aspetto
    ax.view_init(elev=elev, azim=azim)
    # Migliora scala: axis equal (approssimato)
    x_min, x_max = xyz[:, 0].min(), xyz[:, 0].max()
    y_min, y_max = xyz[:, 1].min(), xyz[:, 1].max()
    z_min, z_max = xyz[:, 2].min(), xyz[:, 2].max()
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
    x_mid = 0.5 * (x_max + x_min)
    y_mid = 0.5 * (y_max + y_min)
    z_mid = 0.5 * (z_max + z_min)
    ax.set_xlim(x_mid - max_range/2, x_mid + max_range/2)
    ax.set_ylim(y_mid - max_range/2, y_mid + max_range/2)
    ax.set_zlim(z_mid - max_range/2, z_mid + max_range/2)
    plt.tight_layout()
    plt.show()

#Plot di controllo
def plot_boundary_sets(G, elev=25, azim=-60):
    nodes = G["nodes"]
    meta = G.get("meta", {})
    geom = int(meta.get("geom", 0))

    internal_ids = G.get("node_sets", {}).get("internal", [])
    boundary_ids = G.get("node_sets", {}).get("boundary", [])
    corners_ids  = G.get("node_sets", {}).get("corners", [])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # ---- plot internal nodes ----
    if internal_ids:
        xyz_int = np.array([nodes[nid] for nid in internal_ids], dtype=float)
        ax.scatter(xyz_int[:, 0], xyz_int[:, 1], xyz_int[:, 2],
                   s=8, color="black", alpha=0.5, zorder=2, label="internal")

    # ---- plot boundary nodes (all) ----
    if boundary_ids:
        xyz_bnd = np.array([nodes[nid] for nid in boundary_ids], dtype=float)
        ax.scatter(xyz_bnd[:, 0], xyz_bnd[:, 1], xyz_bnd[:, 2],
                   s=18, color="gray", alpha=0.6, zorder=3, label="boundary (all)")

    # ---- plot corners ----
    if corners_ids:
        xyz_cor = np.array([nodes[nid] for nid in corners_ids], dtype=float)
        ax.scatter(xyz_cor[:, 0], xyz_cor[:, 1], xyz_cor[:, 2],
                   s=60, color="limegreen", edgecolors="black", zorder=5, label="corners")

    # ---- highlight groups ----
    colors = {
        "A": "tab:blue",
        "B": "tab:orange",
        "C": "tab:green",
        "D": "tab:red",
    }

    if geom == 1:
        # Barrel: 4 sides
        sides = G.get("node_sets", {}).get("boundary_sides", {})
        # mapping name -> (label, color-key)
        mapping = [
            ("x_min", "x_min", colors["A"]),
            ("x_max", "x_max", colors["B"]),
            ("y_min", "y_min", colors["C"]),
            ("y_max", "y_max", colors["D"]),
        ]

        for key, label, col in mapping:
            ids = sides.get(key, [])
            if not ids:
                continue
            xyz = np.array([nodes[nid] for nid in ids], dtype=float)
            ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                       s=28, color=col, alpha=0.95, zorder=4, label=f"side: {label}")

    elif geom == 2:
        # Dome: 4 boundary spokes
        spokes = G.get("node_sets", {}).get("boundary_spokes", {})
        mapping = [
            ("Q1", "Q1 (0-90°)", colors["A"]),
            ("Q2", "Q2 (90-180°)", colors["B"]),
            ("Q3", "Q3 (180-270°)", colors["C"]),
            ("Q4", "Q4 (270-360°)", colors["D"]),
        ]

        for key, label, col in mapping:
            ids = spokes.get(key, [])
            if not ids:
                continue
            xyz = np.array([nodes[nid] for nid in ids], dtype=float)
            ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                       s=28, color=col, alpha=0.95, zorder=4, label=f"spoke: {label}")

    ax.set_title("Boundary sets (4 sides / 4 spokes)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=elev, azim=azim)

    # axis equal (approx)
    xyz_all = np.array(list(nodes.values()), dtype=float)
    x_min, x_max = xyz_all[:, 0].min(), xyz_all[:, 0].max()
    y_min, y_max = xyz_all[:, 1].min(), xyz_all[:, 1].max()
    z_min, z_max = xyz_all[:, 2].min(), xyz_all[:, 2].max()
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
    x_mid = 0.5 * (x_max + x_min)
    y_mid = 0.5 * (y_max + y_min)
    z_mid = 0.5 * (z_max + z_min)
    ax.set_xlim(x_mid - max_range/2, x_mid + max_range/2)
    ax.set_ylim(y_mid - max_range/2, y_mid + max_range/2)
    ax.set_zlim(z_mid - max_range/2, z_mid + max_range/2)

    # legenda fuori (se hai molti punti è più leggibile)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    plt.tight_layout()
    plt.show()

#Plot della mesh egenrata come linee
def plot_mesh(G, M, show_nodes=True, show_boundary=True, elev=25, azim=-60):
    nodes = G["nodes"]
    members = M.get("members", [])
    faces = M.get("faces", [])
    boundary_edges = M.get("boundary_edges", [])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # ---- draw members (edges) ----
    for (na, nb) in members:
        xa, ya, za = nodes[na]
        xb, yb, zb = nodes[nb]
        ax.plot([xa, xb], [ya, yb], [za, zb], color = "steelblue", linewidth=2.0, alpha = 0.7)

    # ---- boundary highlight ----
    if show_boundary and boundary_edges:
        for (na, nb) in boundary_edges:
            xa, ya, za = nodes[na]
            xb, yb, zb = nodes[nb]
            ax.plot([xa, xb], [ya, yb], [za, zb], color = "steelblue", linewidth=3.0, alpha = 0.95)

    # ---- nodes ----
    if show_nodes:
        xyz = np.array(list(nodes.values()), dtype=float)
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=20, color = "steelblue", alpha = 0.8, zorder = 3)

    meta = M.get("meta", {})
    ax.set_title(
        f"Mesh | members={meta.get('n_members','?')} | faces={meta.get('n_faces','?')}"
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=elev, azim=azim)

    # axis equal (approx)
    xyz = np.array(list(nodes.values()), dtype=float)
    x_min, x_max = xyz[:, 0].min(), xyz[:, 0].max()
    y_min, y_max = xyz[:, 1].min(), xyz[:, 1].max()
    z_min, z_max = xyz[:, 2].min(), xyz[:, 2].max()
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
    x_mid = 0.5 * (x_max + x_min)
    y_mid = 0.5 * (y_max + y_min)
    z_mid = 0.5 * (z_max + z_min)
    ax.set_xlim(x_mid - max_range/2, x_mid + max_range/2)
    ax.set_ylim(y_mid - max_range/2, y_mid + max_range/2)
    ax.set_zlim(z_mid - max_range/2, z_mid + max_range/2)

    plt.tight_layout()
    plt.show()


def plot_section_assignment_plan(
    G,
    M,
    section_by_member,
    *,
    show_nodes=False,
    show_labels=False,
    figsize=(9, 8),
    linewidth=2.6,
    title="Assigned Sections - Plan View",
    save_path=None,
    show=False,
):
    nodes = G["nodes"]
    members = M.get("members", [])
    if not members:
        raise ValueError("M['members'] is empty.")
    if not section_by_member:
        raise ValueError("section_by_member is empty.")

    def _member_key(a, b):
        a = int(a)
        b = int(b)
        return (a, b) if a < b else (b, a)

    section_names = []
    for mk, sec in section_by_member.items():
        try:
            sec_name = str(sec.get("name", "")).strip()
        except Exception:
            sec_name = ""
        if sec_name:
            section_names.append(sec_name)
    unique_names = sorted(set(section_names))
    if not unique_names:
        unique_names = ["(undefined)"]

    color_by_name = {}
    n_colors = max(1, len(unique_names))
    for i, name in enumerate(unique_names):
        hue = (float(i) / float(n_colors)) % 1.0
        sat = 0.72 if (i % 2 == 0) else 0.58
        val = 0.88 if (i % 3 != 0) else 0.75
        color_by_name[name] = mcolors.to_rgba(mcolors.hsv_to_rgb((hue, sat, val)))

    fig, ax = plt.subplots(figsize=figsize)
    legend_handles = []

    for sec_name in unique_names:
        segs = []
        for na, nb in members:
            mk = _member_key(na, nb)
            sec = section_by_member.get(mk, None)
            this_name = str(sec.get("name", "")).strip() if isinstance(sec, dict) else ""
            if not this_name:
                this_name = "(undefined)"
            if this_name != sec_name:
                continue
            xa, ya, _ = nodes[int(na)]
            xb, yb, _ = nodes[int(nb)]
            segs.append([(float(xa), float(ya)), (float(xb), float(yb))])

        if not segs:
            continue

        lc = LineCollection(
            segs,
            colors=[color_by_name[sec_name]],
            linewidths=float(linewidth),
            alpha=0.95,
            capstyle="round",
        )
        ax.add_collection(lc)
        legend_handles.append(
            Line2D([0], [0], color=color_by_name[sec_name], lw=3.0, label=sec_name)
        )

    if show_nodes or show_labels:
        ids = sorted(nodes.keys())
        xy = np.array([[nodes[nid][0], nodes[nid][1]] for nid in ids], dtype=float)
        if show_nodes:
            ax.scatter(xy[:, 0], xy[:, 1], s=10, color="black", alpha=0.55, zorder=3)
        if show_labels:
            for nid in ids:
                x, y, _ = nodes[nid]
                ax.text(float(x), float(y), str(nid), fontsize=7, color="black")

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal", adjustable="box")
    ax.autoscale()
    ax.grid(True, alpha=0.20)

    if legend_handles:
        ax.legend(
            handles=legend_handles,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            frameon=True,
            fontsize=8,
            title="Sections",
            title_fontsize=9,
        )

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=250, bbox_inches="tight")
        print(f"# File scritto: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_ec3_dcr_plan(
    G,
    M,
    ec3_results,
    *,
    save_path=None,
    show=False,
    title=None,
    figsize=(9, 8),
    linewidth=3.0,
    cmap="turbo",
    vmax=None,
):
    nodes = G["nodes"]
    members = M.get("members", [])
    if not members:
        raise ValueError("M['members'] is empty.")

    def _member_key(a, b):
        a = int(a)
        b = int(b)
        return (a, b) if a < b else (b, a)

    dcr_by_member = {}
    for raw_key, rec in (ec3_results or {}).items():
        if not isinstance(raw_key, tuple) or len(raw_key) != 2:
            continue
        try:
            dcr = float(rec.get("dcr", float("nan")))
        except Exception:
            dcr = float("nan")
        if not np.isfinite(dcr):
            continue
        mk = _member_key(*raw_key)
        dcr_by_member[mk] = max(float(dcr), float(dcr_by_member.get(mk, 0.0)))

    segs = []
    vals = []
    for na, nb in members:
        mk = _member_key(na, nb)
        xa, ya, _ = nodes[int(na)]
        xb, yb, _ = nodes[int(nb)]
        segs.append([(float(xa), float(ya)), (float(xb), float(yb))])
        vals.append(float(dcr_by_member.get(mk, 0.0)))

    if not segs:
        raise ValueError("No member segments available for DCR plan plot.")

    vals = np.asarray(vals, dtype=float)
    max_val = float(np.nanmax(vals)) if vals.size else 0.0
    vmax = float(vmax) if vmax is not None else max(1.0, max_val)
    norm = mcolors.Normalize(vmin=0.0, vmax=max(vmax, 1e-12))

    fig, ax = plt.subplots(figsize=figsize)
    lc = LineCollection(
        segs,
        cmap=cmap,
        norm=norm,
        linewidths=float(linewidth),
        alpha=0.95,
        capstyle="round",
    )
    lc.set_array(vals)
    ax.add_collection(lc)

    xy = np.array([[float(x), float(y)] for x, y, _ in nodes.values()], dtype=float)
    ax.set_xlim(float(np.min(xy[:, 0])) - 0.05 * np.ptp(xy[:, 0]), float(np.max(xy[:, 0])) + 0.05 * np.ptp(xy[:, 0]))
    ax.set_ylim(float(np.min(xy[:, 1])) - 0.05 * np.ptp(xy[:, 1]), float(np.max(xy[:, 1])) + 0.05 * np.ptp(xy[:, 1]))
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    ax.set_title(title or f"EC3 DCR - Plan View | max={max_val:.3f}")

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("DCR")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax


def plot_nodal_displacement_plan(
    G,
    disp,
    *,
    M=None,
    save_path=None,
    show=False,
    title=None,
    figsize=(8, 7),
    component="mag",
    node_size=120,
    show_member_lines=True,
    dlim=None,
    vmax=1.0,
):
    nodes = G["nodes"]
    ids = sorted(int(nid) for nid in nodes.keys())
    xy = np.array([[float(nodes[nid][0]), float(nodes[nid][1])] for nid in ids], dtype=float)

    comp = str(component).strip().lower()
    vals = []
    for nid in ids:
        u = disp.get(int(nid), (0.0, 0.0, 0.0)) if disp is not None else (0.0, 0.0, 0.0)
        ux = float(u[0]) if len(u) > 0 else 0.0
        uy = float(u[1]) if len(u) > 1 else 0.0
        uz = float(u[2]) if len(u) > 2 else 0.0
        if comp in {"ux", "x"}:
            vals.append(abs(ux))
            label = "|Ux| [m]"
        elif comp in {"uy", "y"}:
            vals.append(abs(uy))
            label = "|Uy| [m]"
        elif comp in {"uz", "z"}:
            vals.append(abs(uz))
            label = "|Uz| [m]"
        else:
            vals.append((ux * ux + uy * uy + uz * uz) ** 0.5)
            label = "|U| [m]"

    vals = np.asarray(vals, dtype=float)
    raw_max_val = float(np.nanmax(vals)) if vals.size else 0.0
    dlim_val = None
    if dlim is not None:
        try:
            dlim_val = abs(float(dlim))
        except Exception:
            dlim_val = None
    if dlim_val is not None and dlim_val > 1e-12:
        vals = vals / dlim_val
        max_val = float(np.nanmax(vals)) if vals.size else 0.0
        label = label.split("[", 1)[0].strip() + " / dlim [-]"
        title_default = f"Nodal displacement / dlim - Plan View | max={max_val:.3f}"
        norm_vmax = max(float(vmax), 1e-12) if vmax is not None else max(max_val, 1e-12)
    else:
        max_val = raw_max_val
        title_default = f"Nodal displacement - Plan View | max={max_val:.4g} m"
        norm_vmax = max(max_val, 1e-12)
    norm = mcolors.Normalize(vmin=0.0, vmax=norm_vmax, clip=True)

    fig, ax = plt.subplots(figsize=figsize)
    if show_member_lines and M is not None:
        segs = []
        for na, nb in M.get("members", []):
            xa, ya, _ = nodes[int(na)]
            xb, yb, _ = nodes[int(nb)]
            segs.append([(float(xa), float(ya)), (float(xb), float(yb))])
        if segs:
            ax.add_collection(LineCollection(segs, colors="0.86", linewidths=0.8, zorder=1))

    sc = ax.scatter(
        xy[:, 0],
        xy[:, 1],
        c=vals,
        s=float(node_size),
        cmap="Greys",
        norm=norm,
        edgecolors="0.25",
        linewidths=0.35,
        zorder=2,
    )

    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    ax.set_title(title or title_default)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(label)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax
    return fig, ax, color_by_name
