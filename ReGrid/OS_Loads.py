import numpy as np
from transparent_plots import configure_transparent_matplotlib_saves

configure_transparent_matplotlib_saves()

import matplotlib.pyplot as plt
import openseespy.opensees as ops


# ------------------------------------------------------------
# Helpers for load application regions
# ------------------------------------------------------------
def _quadrants_from_xy(x, y, eps=1e-12):
    """
    Quadrants in the plan view:
      1: x>=0, y>=0
      2: x<0,  y>=0
      3: x<0,  y<0
      4: x>=0, y<0

    If the point is on an axis, it belongs to the adjacent quadrants too.
    """
    on_x = abs(x) <= eps
    on_y = abs(y) <= eps

    if on_x and on_y:
        return [1, 2, 3, 4]
    if on_x:
        return [1, 2] if y > eps else [4, 3]
    if on_y:
        return [1, 4] if x > eps else [2, 3]
    if x > 0 and y > 0:
        return [1]
    if x < 0 and y > 0:
        return [2]
    if x < 0 and y < 0:
        return [3]
    return [4]


def _nodes_in_parts_1to4(G, parts, eps=1e-12):
    parts = set(int(p) for p in parts)
    out = []
    for nid, (x, y, z) in G["nodes"].items():
        if any(q in parts for q in _quadrants_from_xy(x, y, eps=eps)):
            out.append(int(nid))
    return out


def _nodes_in_freegrid_lc2_region(G, eps=1e-12):
    """
    FreeGrid LC2 region:
      - barrel vault (geom=1): x >= 0
      - parabolic dome (geom=2): x >= 0
      - hyperbolic paraboloid (geom=3): y >= 0
      - kiewitt-8 dome (geom=4): x >= 0  (same half-domain convention used for domes)

    Nodes on the dividing line are INCLUDED.
    """
    geom = int(G.get("meta", {}).get("geom", 0))
    nodes = G["nodes"]
    out = []

    if geom in (1, 2, 4):
        for nid, (x, y, z) in nodes.items():
            if x >= -eps:
                out.append(int(nid))
    elif geom == 3:
        for nid, (x, y, z) in nodes.items():
            if y >= -eps:
                out.append(int(nid))
    else:
        raise NotImplementedError(f"geom={geom} not supported for FreeGrid LC2 region.")
    return out


def freegrid_parts_for_geom(geom):
    """
    Returns the quadrant union equivalent to the FreeGrid LC2 half-domain.

      geom=1 (barrel) -> (1,4)  <=> x >= 0
      geom=2 (dome)   -> (1,4)  <=> x >= 0
      geom=3 (hypar)  -> (1,2)  <=> y >= 0
      geom=4 (Kiewitt)-> (1,4)  <=> x >= 0
    """
    geom = int(geom)
    if geom in (1, 2, 4):
        return (1, 4)
    if geom == 3:
        return (1, 2)
    raise NotImplementedError(f"geom={geom} not supported.")


# ------------------------------------------------------------
# q specification parsing
# ------------------------------------------------------------
def _parse_qvecspec(q):
    """
    Accepts:
      - scalar number -> interpreted as (0,0,qz,"sym")
      - tuple/list blocks of:
          (qx, qy, qz, "symmetrical")
          (qx, qy, qz, "asymmetrical", 1,2,3,4)
          (qx, qy, qz, "freegrid_lc2")

    Returns a list of dicts:
      {"q": (qx,qy,qz), "mode": "sym"|"asym"|"freegrid_lc2", "parts": None|[...]}
    """
    if isinstance(q, (int, float, np.floating)):
        return [{"q": (0.0, 0.0, float(q)), "mode": "sym", "parts": None}]

    if not isinstance(q, (list, tuple)):
        raise ValueError("q must be a number or a tuple/list spec.")

    items = list(q)
    out = []
    i = 0

    def _is_num(x):
        return isinstance(x, (int, float, np.floating))

    while i < len(items):
        if i + 3 >= len(items):
            raise ValueError("Each load block must be (qx,qy,qz, mode, ...).")

        if not (_is_num(items[i]) and _is_num(items[i + 1]) and _is_num(items[i + 2])):
            raise ValueError(f"Expected (qx,qy,qz) numeric at position {i}.")

        qx = float(items[i])
        qy = float(items[i + 1])
        qz = float(items[i + 2])
        i += 3

        if i >= len(items) or not isinstance(items[i], str):
            raise ValueError(
                "After (qx,qy,qz) you must provide 'symmetrical', 'asymmetrical' or 'freegrid_lc2'."
            )
        mode_raw = items[i].strip().lower()
        i += 1

        if mode_raw.startswith("sym"):
            out.append({"q": (qx, qy, qz), "mode": "sym", "parts": None})
            continue

        if mode_raw in ("freegrid_lc2", "lc2", "freegrid"):
            out.append({"q": (qx, qy, qz), "mode": "freegrid_lc2", "parts": None})
            continue

        if mode_raw.startswith("asym"):
            parts = []
            while i < len(items) and isinstance(items[i], (int, np.integer)):
                parts.append(int(items[i]))
                i += 1
            if not parts and i < len(items) and isinstance(items[i], (list, tuple)):
                parts = [int(p) for p in items[i]]
                i += 1
            if not parts:
                raise ValueError("For 'asymmetrical' specify parts among 1,2,3,4.")
            bad = [p for p in parts if p not in (1, 2, 3, 4)]
            if bad:
                raise ValueError(f"Invalid parts {bad}. Allowed: 1,2,3,4.")
            out.append({"q": (qx, qy, qz), "mode": "asym", "parts": parts})
            continue

        raise ValueError("Mode must be 'symmetrical', 'asymmetrical' or 'freegrid_lc2'.")

    return out


# ------------------------------------------------------------
# Plot helper
# ------------------------------------------------------------
def plot_nodal_load_map(
    G,
    loads,
    filename=None,
    title=None,
    show=True,
    cmap="viridis",
    s=30,
    show_labels=False,
    label_fmt="{:.2f}",
    label_offset=(0.01, 0.01),
    label_fontsize=8,
    label_nodes=None,
    component="Fz",
):
    nodes = G["nodes"]
    ids = sorted(loads.keys())
    xy = np.array([(nodes[nid][0], nodes[nid][1]) for nid in ids], dtype=float)

    comp = str(component).strip().lower()
    vals = []
    for nid in ids:
        v = loads[nid]
        if isinstance(v, (tuple, list, np.ndarray)) and len(v) >= 3:
            Fx, Fy, Fz = float(v[0]), float(v[1]), float(v[2])
            if comp == "fx":
                vals.append(Fx)
            elif comp == "fy":
                vals.append(Fy)
            elif comp == "fz":
                vals.append(Fz)
            elif comp in ("mag", "norm", "module"):
                vals.append((Fx**2 + Fy**2 + Fz**2) ** 0.5)
            else:
                raise ValueError("component must be 'Fx','Fy','Fz' or 'mag'")
        else:
            vals.append(float(v))

    vals = np.array(vals, dtype=float)
    if title is None:
        title = f"Nodal loads {component} [kN]" if comp != "mag" else "Nodal load magnitude [kN]"

    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(
        xy[:, 0],
        xy[:, 1],
        c=vals,
        s=s,
        cmap=cmap,
        edgecolors="k",
        linewidths=0.3,
    )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("kN")

    if show_labels:
        if label_nodes is None:
            label_nodes = ids
        label_nodes = set(int(n) for n in label_nodes)
        dx, dy = label_offset
        for i, nid in enumerate(ids):
            if nid not in label_nodes:
                continue
            ax.text(
                xy[i, 0] + dx,
                xy[i, 1] + dy,
                label_fmt.format(float(vals[i])),
                fontsize=label_fontsize,
                ha="left",
                va="bottom",
            )

    ax.grid(True, linestyle=":", linewidth=0.5)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


# ------------------------------------------------------------
# Tributary areas on the XY projection
# ------------------------------------------------------------
def tributary_areas_xy_from_faces(G, M):
    """
    Compute nodal tributary areas on the HORIZONTAL projection.
    Each face contributes its projected polygon area / n_vertices to its nodes.

    Supports both:
      - M['faces']    : quad or tri faces, optionally with face id in first position
      - M['faces_tri']: triangular faces with (fid, n1, n2, n3)

    This is the most coherent rule with the FreeGrid definition:
      s_j = horizontal projection of the tributary area of joint j.
    """
    nodes = G["nodes"]
    faces = []
    if M is not None:
        if len(M.get("faces", [])) > 0:
            faces = M.get("faces", [])
        elif len(M.get("faces_tri", [])) > 0:
            faces = M.get("faces_tri", [])
    if not faces:
        raise ValueError("M['faces'] and M['faces_tri'] are both empty or missing.")

    areas = {int(nid): 0.0 for nid in nodes.keys()}

    def _poly_area_xy(coords):
        x = np.array([c[0] for c in coords], dtype=float)
        y = np.array([c[1] for c in coords], dtype=float)
        return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    for face in faces:
        if len(face) < 3:
            continue
        # If first entry is an integer id and tuple is longer than 3, skip it.
        if isinstance(face[0], (int, np.integer)) and len(face) >= 4:
            node_ids = [int(n) for n in face[1:]]
        else:
            node_ids = [int(n) for n in face]

        coords = [nodes[nid] for nid in node_ids]
        Af = float(_poly_area_xy(coords))
        if Af <= 0.0:
            continue
        share = Af / float(len(node_ids))
        for nid in node_ids:
            areas[nid] += share

    return areas


def tributary_areas_xy_orthogonal(G, dx=None, dy=None, round_decimals=8, max_gap_factor=1.75):
    """
    Fallback rule based on the orthogonal 4-neighbour stencil in the XY projection.
    Kept for backward compatibility and for cases where faces are not available.
    """
    nodes = G["nodes"]
    ids = np.array(sorted(int(k) for k in nodes.keys()), dtype=int)
    xy = np.array([(nodes[i][0], nodes[i][1]) for i in ids], dtype=float)
    x = xy[:, 0]
    y = xy[:, 1]

    if dx is None:
        ux = np.unique(np.round(x, round_decimals))
        dd = np.diff(np.sort(ux))
        dx = float(np.median(dd)) if len(dd) else 1.0
    if dy is None:
        uy = np.unique(np.round(y, round_decimals))
        dd = np.diff(np.sort(uy))
        dy = float(np.median(dd)) if len(dd) else 1.0

    qx = np.round(x / dx).astype(int)
    qy = np.round(y / dy).astype(int)

    rows, cols = {}, {}
    for idx, nid in enumerate(ids):
        rows.setdefault(qy[idx], []).append(idx)
        cols.setdefault(qx[idx], []).append(idx)

    for ky, lst in rows.items():
        lst.sort(key=lambda i: x[i])
    for kx, lst in cols.items():
        lst.sort(key=lambda i: y[i])

    def _typical_step_from_lists(lists, coord):
        gaps = []
        for lst in lists.values():
            if len(lst) < 2:
                continue
            c = coord[lst]
            dg = np.diff(c)
            dg = dg[dg > 1e-12]
            gaps.extend(dg.tolist())
        if not gaps:
            return 1.0
        gaps = np.array(gaps, float)
        med = np.median(gaps)
        core = gaps[gaps <= 1.5 * med]
        return float(np.median(core)) if core.size else float(med)

    dx0 = _typical_step_from_lists(rows, x)
    dy0 = _typical_step_from_lists(cols, y)
    cap_dx = max_gap_factor * dx0
    cap_dy = max_gap_factor * dy0

    row_pos, col_pos = {}, {}
    for ky, lst in rows.items():
        for p, idx in enumerate(lst):
            row_pos[idx] = (ky, p)
    for kx, lst in cols.items():
        for p, idx in enumerate(lst):
            col_pos[idx] = (kx, p)

    def _d_left_right_capped(lst, coord, p, cap):
        dL = dR = None
        if p > 0:
            dL = min(float(coord[lst[p]] - coord[lst[p - 1]]), cap)
        if p < len(lst) - 1:
            dR = min(float(coord[lst[p + 1]] - coord[lst[p]]), cap)
        return dL, dR

    areas = {}
    for idx, nid in enumerate(ids):
        ky, pr = row_pos[idx]
        kx, pc = col_pos[idx]
        rlist = rows[ky]
        clist = cols[kx]

        dL, dR = _d_left_right_capped(rlist, x, pr, cap_dx)
        if (dL is not None) and (dR is not None):
            Ax = 0.5 * (dL + dR)
        elif dL is not None:
            Ax = 0.5 * dL
        elif dR is not None:
            Ax = 0.5 * dR
        else:
            Ax = 0.0

        dD, dU = _d_left_right_capped(clist, y, pc, cap_dy)
        if (dD is not None) and (dU is not None):
            Ay = 0.5 * (dD + dU)
        elif dD is not None:
            Ay = 0.5 * dD
        elif dU is not None:
            Ay = 0.5 * dU
        else:
            Ay = 0.0

        areas[int(nid)] = float(Ax * Ay)

    return areas


def get_tributary_areas_xy(G, M=None, area_mode="auto", dx=None, dy=None, round_decimals=8):
    """
    area_mode:
      - "faces": use M['faces'] or M['faces_tri'] projected on XY
      - "orthogonal": use 4-neighbour fallback
      - "auto": use faces/faces_tri if available, else orthogonal
    """
    mode = str(area_mode).strip().lower()
    if mode == "auto":
        has_faces = False
        if M is not None:
            has_faces = len(M.get("faces", [])) > 0 or len(M.get("faces_tri", [])) > 0
        mode = "faces" if has_faces else "orthogonal"

    if mode == "faces":
        return tributary_areas_xy_from_faces(G, M)
    if mode in ("orthogonal", "grid"):
        return tributary_areas_xy_orthogonal(G, dx=dx, dy=dy, round_decimals=round_decimals)
    raise ValueError("area_mode must be 'auto', 'faces' or 'orthogonal'.")


# ------------------------------------------------------------
# Main application routine
# ------------------------------------------------------------
def sf_load_as_nodal_xy_projection(
    G,
    q,
    M=None,
    direction="-Z",
    pattern_tag=1,
    ts_tag=1,
    ndf=6,
    area_mode="auto",
    dx=None,
    dy=None,
    round_decimals=8,
    print_ops=True,
    print_mode="all",
    n_preview=10,
    print_summary=True,
    print_components=True,
    plot_loads=False,
    plot_filename=None,
    plot_show=True,
    plot_title=None,
    plot_s=30,
    plot_show_labels=False,
    plot_label_fmt="{:.2f}",
    plot_label_offset=(0.01, 0.01),
    plot_label_fontsize=8,
    plot_label_nodes=None,
    plot_component="Fz",
    diag_check_uniformity=False,
    diag_tol_cv=0.02,
):
    """
    Nodal loads from uniform load(s) q [kN/m^2] referred to the XY-projected tributary area.

    Supported q formats:
      - scalar -> (0,0,qz) symmetrical
      - tuple/list blocks ->
          (qx,qy,qz,"symmetrical", ...)
          (qx,qy,qz,"asymmetrical",1,4, ...)
          (qx,qy,qz,"freegrid_lc2", ...)

    Notes:
      - The load magnitude is always pressure x projected tributary area.
      - The selected region can be fully custom (quadrants) or FreeGrid-compatible.
      - Self-weight g is NOT included here and must be handled separately.
    """
    nodes = G["nodes"]
    ndf = int(ndf)
    if ndf < 3:
        raise ValueError("Need ndf>=3 to apply loads in 3D.")

    dirZ = str(direction).strip().upper()
    signZ = -1.0 if dirZ == "-Z" else +1.0
    q_components = _parse_qvecspec(q)
    ids = np.array(sorted(int(k) for k in nodes.keys()), dtype=int)

    areas = get_tributary_areas_xy(
        G,
        M=M,
        area_mode=area_mode,
        dx=dx,
        dy=dy,
        round_decimals=round_decimals,
    )

    q_eff = {int(nid): np.array([0.0, 0.0, 0.0], dtype=float) for nid in ids}
    all_nodes_set = set(int(n) for n in ids)

    for comp in q_components:
        qvec = np.array(comp["q"], dtype=float)
        if comp["mode"] == "sym":
            target = all_nodes_set
        elif comp["mode"] == "asym":
            target = set(_nodes_in_parts_1to4(G, comp["parts"]))
        elif comp["mode"] == "freegrid_lc2":
            target = set(_nodes_in_freegrid_lc2_region(G))
        else:
            raise RuntimeError(f"Unsupported load mode: {comp['mode']}")

        for nid in target:
            q_eff[nid] += qvec

    try:
        ops.remove("loadPattern", int(pattern_tag))
    except Exception:
        pass
    try:
        ops.remove("timeSeries", int(ts_tag))
    except Exception:
        pass

    ops.timeSeries("Linear", int(ts_tag))
    ops.pattern("Plain", int(pattern_tag), int(ts_tag))

    py_lines = [
        f"ops.timeSeries('Linear', {int(ts_tag)})",
        f"ops.pattern('Plain', {int(pattern_tag)}, {int(ts_tag)})",
    ]

    loads = {}
    for nid in ids:
        Atrib = float(areas.get(int(nid), 0.0))
        qx, qy, qz = q_eff[int(nid)]
        Fx = float(qx * Atrib)
        Fy = float(qy * Atrib)
        Fz = float(signZ * qz * Atrib)

        loads[int(nid)] = (Fx, Fy, Fz)

        vec = [0.0] * ndf
        vec[0] = Fx
        vec[1] = Fy
        vec[2] = Fz
        ops.load(int(nid), *vec)
        py_lines.append(f"ops.load({int(nid)}, {', '.join(str(v) for v in vec)})")

    py_text = "\n".join(py_lines)

    if print_ops:
        mode = str(print_mode).lower()
        if mode == "all":
            print(py_text)
        else:
            total = len(py_lines)
            print(f"# --- OpenSees commands generated: {total} lines ---")
            head = py_lines[:n_preview]
            tail = py_lines[-n_preview:] if total > n_preview else []
            for ln in head:
                print(ln)
            if total > 2 * n_preview:
                print("...")
            for ln in tail:
                print(ln)

    if print_components:
        for k, comp in enumerate(q_components, 1):
            if comp["mode"] == "sym":
                print(f"# q[{k}] = {comp['q']} kN/m^2 | symmetrical (all nodes)")
            elif comp["mode"] == "asym":
                print(f"# q[{k}] = {comp['q']} kN/m^2 | asymmetrical parts={comp['parts']}")
            elif comp["mode"] == "freegrid_lc2":
                print(f"# q[{k}] = {comp['q']} kN/m^2 | FreeGrid LC2 half-domain")

    if print_summary:
        sumA = float(sum(areas.values()))
        Fx_tot = float(sum(F[0] for F in loads.values()))
        Fy_tot = float(sum(F[1] for F in loads.values()))
        Fz_tot = float(sum(F[2] for F in loads.values()))
        print(
            f"# SUMMARY | SumAreaXY={sumA:.6f} m^2 | "
            f"Fx_tot={Fx_tot:.6f} kN | "
            f"Fy_tot={Fy_tot:.6f} kN | "
            f"Fz_tot={Fz_tot:.6f} kN"
        )

    if diag_check_uniformity:
        Avals = np.array(list(areas.values()), dtype=float)
        if Avals.size:
            cv = float(np.std(Avals) / (np.mean(Avals) + 1e-18))
            msg = f"# DIAG | Areas CV = {cv:.4f}"
            if cv > float(diag_tol_cv):
                msg += "  (NOT uniform -> plan grid irregular or faces/rows grouping not matching)"
            print(msg)

    if plot_loads:
        if plot_title is None:
            plot_title = f"Nodal loads from XY projection ({plot_component}) [kN]"
        plot_nodal_load_map(
            G,
            loads,
            filename=plot_filename,
            title=plot_title,
            show=plot_show,
            s=plot_s,
            show_labels=plot_show_labels,
            label_fmt=plot_label_fmt,
            label_offset=plot_label_offset,
            label_fontsize=plot_label_fontsize,
            label_nodes=plot_label_nodes,
            component=plot_component,
        )

    return loads, areas, py_text


# ------------------------------------------------------------
# FreeGrid wrappers (optional convenience)
# ------------------------------------------------------------
def apply_freegrid_lc1(
    G,
    M,
    q1,
    q2,
    pattern_tag=1,
    ts_tag=1,
    ndf=6,
    direction="-Z",
    area_mode="faces",
    **kwargs,
):
    """
    FreeGrid LC1:
      Q_j = (q1 + q2) * s_j   for all j
    Self-weight g is NOT included here and must be handled separately.
    """
    return sf_load_as_nodal_xy_projection(
        G=G,
        M=M,
        q=(0.0, 0.0, float(q1 + q2), "symmetrical"),
        direction=direction,
        pattern_tag=pattern_tag,
        ts_tag=ts_tag,
        ndf=ndf,
        area_mode=area_mode,
        **kwargs,
    )


def apply_freegrid_lc2(
    G,
    M,
    q1,
    q2,
    pattern_tag=2,
    ts_tag=2,
    ndf=6,
    direction="-Z",
    area_mode="faces",
    **kwargs,
):
    """
    FreeGrid LC2:
      - q1 on all nodes
      - q2 only on:
          x >= 0 for barrel vault, parabolic dome and Kiewitt-8 dome
          y >= 0 for hyperbolic paraboloid
    Self-weight g is NOT included here and must be handled separately.
    """
    return sf_load_as_nodal_xy_projection(
        G=G,
        M=M,
        q=(0.0, 0.0, float(q1), "symmetrical", 0.0, 0.0, float(q2), "freegrid_lc2"),
        direction=direction,
        pattern_tag=pattern_tag,
        ts_tag=ts_tag,
        ndf=ndf,
        area_mode=area_mode,
        **kwargs,
    )
