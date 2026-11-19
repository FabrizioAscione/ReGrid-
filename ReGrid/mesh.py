import math
from collections import defaultdict

ROUND_DECIMALS = 6
TOL = 1e-9


def _round6(value):
    return round(float(value), ROUND_DECIMALS)


def _edge(a, b):
    a = int(a)
    b = int(b)
    if a == b:
        raise ValueError("Degenerate edge with coincident node ids.")
    return (a, b) if a < b else (b, a)


def _dist3(nodes, a, b):
    xa, ya, za = nodes[a]
    xb, yb, zb = nodes[b]
    return math.sqrt((xa - xb) ** 2 + (ya - yb) ** 2 + (za - zb) ** 2)




def _x_vals_uniform_arc_barrel(B, n, x_min=None, x_max=None, n_dense=20001):
    if x_min is None:
        x_min = -B / 2.0
    if x_max is None:
        x_max = +B / 2.0
    n = int(n)
    xd = [x_min + (x_max - x_min) * k / (int(n_dense) - 1) for k in range(int(n_dense))]
    s = [0.5 * (x * math.sqrt(1.0 + (x / B) ** 2) + B * math.asinh(x / B)) for x in xd]
    s0 = s[0]
    s = [v - s0 for v in s]
    s_targets = [s[-1] * k / (n - 1) for k in range(n)] if n > 1 else [0.0]
    out = []
    j = 0
    for st in s_targets:
        while j < len(s) - 2 and s[j + 1] < st:
            j += 1
        s1, s2 = s[j], s[j + 1]
        x1, x2 = xd[j], xd[j + 1]
        if abs(s2 - s1) < 1e-15:
            out.append(x1)
        else:
            t = (st - s1) / (s2 - s1)
            out.append(x1 + t * (x2 - x1))
    return [_round6(v) for v in out]


def _expected_parametric_levels(metaG):
    geom = int(metaG.get("geom", 0))
    B = float(metaG.get("B", 0.0))
    L = float(metaG.get("L", 0.0))
    f = float(metaG.get("f", 0.0))
    nx = int(metaG.get("nx", 0))
    ny = int(metaG.get("ny", 0))

    if geom == 1:
        x_levels = _x_vals_uniform_arc_barrel(B, nx, -B / 2.0, B / 2.0)
        y_levels = [_round6(-L / 2.0 + (L * j) / (ny - 1)) for j in range(ny)]
    elif geom == 2:
        R = math.sqrt(2.0 * B * f)
        x_levels = _x_vals_uniform_arc_barrel(B, nx, -R, R)
        y_levels = _x_vals_uniform_arc_barrel(B, ny, -R, R)
    elif geom == 3:
        x_levels = _x_vals_uniform_arc_barrel(B, nx, -B / 2.0, B / 2.0)
        y_levels = _x_vals_uniform_arc_barrel(B, ny, -B / 2.0, B / 2.0)
    else:
        raise NotImplementedError(f"geom={geom} not supported.")
    return x_levels, y_levels


def _ordered_parametric_lines(nodes, x_levels, y_levels, tol=1e-6):
    x_lines = {lvl: [] for lvl in x_levels}
    y_lines = {lvl: [] for lvl in y_levels}

    for nid, (x, y, _) in nodes.items():
        xr = _round6(x)
        yr = _round6(y)

        for lvl in x_levels:
            if abs(xr - lvl) <= tol:
                x_lines[lvl].append((yr, int(nid)))
                break

        for lvl in y_levels:
            if abs(yr - lvl) <= tol:
                y_lines[lvl].append((xr, int(nid)))
                break

    x_lines = {lvl: [nid for _, nid in sorted(vals, key=lambda t: t[0])] for lvl, vals in x_lines.items() if vals}
    y_lines = {lvl: [nid for _, nid in sorted(vals, key=lambda t: t[0])] for lvl, vals in y_lines.items() if vals}
    return x_lines, y_lines

def _ordered_group_lines(nodes, group_by="x"):
    """
    Group nodes by constant x or y and sort each family along the orthogonal coordinate.
    Returns an ordered dict-like mapping: level -> [nid1, nid2, ...]
    """
    groups = defaultdict(list)
    if group_by == "x":
        for nid, (x, y, _) in nodes.items():
            groups[_round6(x)].append((_round6(y), int(nid)))
        return {lvl: [nid for _, nid in sorted(vals, key=lambda t: t[0])] for lvl, vals in sorted(groups.items(), key=lambda t: t[0])}
    elif group_by == "y":
        for nid, (x, y, _) in nodes.items():
            groups[_round6(y)].append((_round6(x), int(nid)))
        return {lvl: [nid for _, nid in sorted(vals, key=lambda t: t[0])] for lvl, vals in sorted(groups.items(), key=lambda t: t[0])}
    else:
        raise ValueError("group_by must be 'x' or 'y'")


def _members_from_ordered_lines(lines):
    members = set()
    for _, ids in lines.items():
        for a, b in zip(ids[:-1], ids[1:]):
            members.add(_edge(a, b))
    return members


def _ordered_boundary_cycle(G, nodes):
    """
    Build an ordered free-edge cycle from the generic boundary node set.
    For dome and hypar this is most robustly obtained by sorting the boundary
    nodes by polar angle around the plan centroid; this avoids the spurious
    long chords produced by naive concatenation of quadrant/side subsets.
    Returns directed boundary edges [(n1,n2), ...].
    """
    node_sets = G.get("node_sets", {})
    boundary = [int(n) for n in node_sets.get("boundary", [])]
    if len(boundary) < 3:
        return []
    cx = sum(nodes[n][0] for n in boundary) / len(boundary)
    cy = sum(nodes[n][1] for n in boundary) / len(boundary)
    order = sorted(boundary, key=lambda n: math.atan2(nodes[n][1] - cy, nodes[n][0] - cx))
    return [(order[i], order[(i + 1) % len(order)]) for i in range(len(order))]


# ---------- face helpers for structured barrel ----------
def _barrel_faces(metaG):
    nx = int(metaG["nx"])
    ny = int(metaG["ny"])

    def nid(i, j):
        return 1 + i * ny + j

    faces = []
    fid = 1
    for i in range(nx - 1):
        for j in range(ny - 1):
            faces.append((fid, nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)))
            fid += 1
    return faces


def _barrel_boundary_edges(metaG):
    nx = int(metaG["nx"])
    ny = int(metaG["ny"])

    def nid(i, j):
        return 1 + i * ny + j

    out = []
    # loop around the perimeter, ordered counter-clockwise in plan
    for i in range(nx - 1):
        out.append((nid(i, 0), nid(i + 1, 0)))
    for j in range(ny - 1):
        out.append((nid(nx - 1, j), nid(nx - 1, j + 1)))
    for i in range(nx - 1, 0, -1):
        out.append((nid(i, ny - 1), nid(i - 1, ny - 1)))
    for j in range(ny - 1, 0, -1):
        out.append((nid(0, j), nid(0, j - 1)))
    return out


# ---------- optional face extraction for cropped dome/hypar ----------
def _faces_from_xy_cells(nodes, x_lines, y_lines):
    """
    Build only true quad cells whose 4 corner nodes exist.
    This is conservative and deterministic. Cropped boundary cells that are not full quads
    are intentionally skipped.
    """
    coord2id = {(_round6(x), _round6(y)): int(nid) for nid, (x, y, _) in nodes.items()}
    x_levels = list(x_lines.keys())
    y_levels = list(y_lines.keys())

    faces = []
    fid = 1
    for i in range(len(x_levels) - 1):
        x0 = x_levels[i]
        x1 = x_levels[i + 1]
        for j in range(len(y_levels) - 1):
            y0 = y_levels[j]
            y1 = y_levels[j + 1]
            n1 = coord2id.get((_round6(x0), _round6(y0)))
            n2 = coord2id.get((_round6(x1), _round6(y0)))
            n3 = coord2id.get((_round6(x1), _round6(y1)))
            n4 = coord2id.get((_round6(x0), _round6(y1)))
            if None not in (n1, n2, n3, n4):
                faces.append((fid, n1, n2, n3, n4))
                fid += 1
    return faces




def _kiewitt8_ring_boundary_node(ring_nodes, ring_index, sector_index):
    ring = list(ring_nodes[int(ring_index)])
    idx = (int(sector_index) * int(ring_index)) % len(ring)
    return int(ring[idx])


def _kiewitt8_faces_and_members(G):
    """Build triangular topology for a Kiewitt-8 dome."""
    node_sets = G.get('node_sets', {})
    ring_nodes = node_sets.get('rings', {})
    apex_list = node_sets.get('apex', [])
    if not ring_nodes or not apex_list:
        raise ValueError("Kiewitt-8 geometry requires node_sets['rings'] and node_sets['apex'].")

    apex = int(apex_list[0])
    n_rings = max(int(k) for k in ring_nodes.keys())

    lat_set = set()
    rad_set = set()
    diag_set = set()
    faces_tri = []
    fid = 1

    # Latitudinal rings
    for r in range(1, n_rings + 1):
        ring = list(ring_nodes[r])
        n = len(ring)
        for j in range(n):
            a = int(ring[j])
            b = int(ring[(j + 1) % n])
            lat_set.add(_edge(a, b))

    # Apex fan + first radial ribs
    ring1 = list(ring_nodes[1])
    for s in range(8):
        a = apex
        b = int(ring1[s])
        rad_set.add(_edge(a, b))
        c = int(ring1[(s + 1) % 8])
        faces_tri.append((fid, a, b, c))
        fid += 1

    # Radial ribs between successive rings
    for r in range(1, n_rings):
        for s in range(8):
            a = _kiewitt8_ring_boundary_node(ring_nodes, r, s)
            b = _kiewitt8_ring_boundary_node(ring_nodes, r + 1, s)
            rad_set.add(_edge(a, b))

    # Diagonals and triangular faces by sector
    for r in range(1, n_rings):
        inner = list(ring_nodes[r])
        outer = list(ring_nodes[r + 1])
        ni = len(inner)
        no = len(outer)
        for s in range(8):
            inner_seq = [inner[(s * r + t) % ni] for t in range(r + 1)]
            outer_seq = [outer[(s * (r + 1) + t) % no] for t in range(r + 2)]

            # fan from each outer interior node to two adjacent inner nodes
            for m in range(1, r + 1):
                a = int(inner_seq[m - 1])
                b = int(inner_seq[m])
                c = int(outer_seq[m])
                diag_set.add(_edge(a, c))
                diag_set.add(_edge(b, c))
                faces_tri.append((fid, a, b, c))
                fid += 1

            # outer strip triangles between consecutive outer nodes and inner nodes
            for m in range(r + 1):
                a = int(inner_seq[m])
                b = int(outer_seq[m])
                c = int(outer_seq[m + 1])
                faces_tri.append((fid, a, b, c))
                fid += 1

    boundary_ring = list(ring_nodes[n_rings])
    boundary_edges = [(int(boundary_ring[j]), int(boundary_ring[(j + 1) % len(boundary_ring)])) for j in range(len(boundary_ring))]

    rib_lines = node_sets.get('rib_lines', {})
    ring_lines = node_sets.get('ring_lines', {})

    members_set = set()
    members_set.update(lat_set)
    members_set.update(rad_set)
    members_set.update(diag_set)
    members_set.update(_edge(a, b) for a, b in boundary_edges)

    return {
        'faces': [],
        'faces_tri': faces_tri,
        'members': sorted(members_set),
        'boundary_edges': boundary_edges,
        'lines': {
            'x': {},
            'y': {},
            'rings': ring_lines,
            'ribs': rib_lines,
        },
        'member_groups': {
            'latitudinal': sorted(lat_set),
            'radial': sorted(rad_set),
            'diagonal': sorted(diag_set),
        },
        'meta': {
            'geom': 4,
            'grid': 2,
            'face_type': 'tri3',
            'member_type': 'kiewitt8',
            'n_faces_tri': len(faces_tri),
            'n_members': len(members_set),
            'n_boundary_edges': len(boundary_edges),
        },
    }


def define_mesh(G, grid):
    """
    FreeGrid-oriented mesh generator.

    Supported geometries:
      geom=1 -> barrel vault
      geom=2 -> parabolic dome
      geom=3 -> hyperbolic paraboloid
      geom=4 -> Kiewitt-8 dome

    For all geometries, the structural members are the two orthogonal families of
    grid lines (x=const and y=const), connected between consecutive nodes.
    This matches the benchmark logic and avoids the old heuristic distance filter.

    Output M:
      {
        "faces": [(fid,n1,n2,n3,n4), ...],
        "members": [(na,nb), ...],
        "boundary_edges": [(na,nb), ...],
        "lines": {"x": {...}, "y": {...}},
        "meta": {...}
      }
    """
    if grid != 1:
        raise NotImplementedError("Only grid=1 (quadrangular) is implemented for now.")

    metaG = G.get("meta", {})
    geom = int(metaG.get("geom", 0))
    nodes = G["nodes"]

    # ------------------------------------------------------------------
    # GEOM = 1  (BARREL VAULT)
    # ------------------------------------------------------------------
    if geom == 1:
        faces = _barrel_faces(metaG)
        members_set = set()
        for _, n1, n2, n3, n4 in faces:
            members_set.add(_edge(n1, n2))
            members_set.add(_edge(n2, n3))
            members_set.add(_edge(n3, n4))
            members_set.add(_edge(n4, n1))

        boundary_edges = _barrel_boundary_edges(metaG)
        members_set.update(_edge(a, b) for a, b in boundary_edges)
        members = sorted(members_set)

        x_levels, y_levels = _expected_parametric_levels(metaG)
        x_lines, y_lines = _ordered_parametric_lines(nodes, x_levels, y_levels)

        return {
            "faces": faces,
            "members": members,
            "boundary_edges": boundary_edges,
            "lines": {"x": x_lines, "y": y_lines},
            "meta": {
                "geom": geom,
                "grid": grid,
                "face_type": "quad4",
                "member_type": "orthogonal-grid-edges",
                "n_faces": len(faces),
                "n_members": len(members),
                "n_boundary_edges": len(boundary_edges),
            },
        }

    # ------------------------------------------------------------------
    # GEOM = 4  (KIEWITT-8 DOME)
    # ------------------------------------------------------------------
    elif geom == 4:
        return _kiewitt8_faces_and_members(G)

    # ------------------------------------------------------------------
    # GEOM = 2/3  (DOME / HYPAR)
    # ------------------------------------------------------------------
    elif geom in (2, 3):
        x_levels, y_levels = _expected_parametric_levels(metaG)
        x_lines, y_lines = _ordered_parametric_lines(nodes, x_levels, y_levels)

        members_set = set()
        members_set.update(_members_from_ordered_lines(x_lines))
        members_set.update(_members_from_ordered_lines(y_lines))

        boundary_edges = _ordered_boundary_cycle(G, nodes)
        members_set.update(_edge(a, b) for a, b in boundary_edges)

        faces = _faces_from_xy_cells(nodes, x_lines, y_lines)
        members = sorted(members_set)

        # Useful diagnostics for QC
        lengths = [_dist3(nodes, a, b) for a, b in members]
        Lmin = min(lengths) if lengths else 0.0
        Lmax = max(lengths) if lengths else 0.0

        return {
            "faces": faces,
            "members": members,
            "boundary_edges": boundary_edges,
            "lines": {"x": x_lines, "y": y_lines},
            "meta": {
                "geom": geom,
                "grid": grid,
                "face_type": "quad4 (full internal cells only)",
                "member_type": "orthogonal-grid-arches",
                "n_faces": len(faces),
                "n_members": len(members),
                "n_boundary_edges": len(boundary_edges),
                "member_length_min": Lmin,
                "member_length_max": Lmax,
            },
        }

    else:
        raise NotImplementedError(f"geom={geom} not supported.")


# ----------------------------------------------------------------------
# Reports / exports
# ----------------------------------------------------------------------
def write_members_to_txt(G, M, filename="mesh_members.txt"):
    """
    Write members to txt:
    member_id, n1, n2, x1 y1 z1, x2 y2 z2, length
    """
    nodes = G["nodes"]
    members = M.get("members", [])

    with open(filename, "w") as f:
        f.write("# mid   n1    n2          x1          y1          z1"
                "          x2          y2          z2        L\n")
        for mid, (n1, n2) in enumerate(sorted(members), start=1):
            x1, y1, z1 = nodes[n1]
            x2, y2, z2 = nodes[n2]
            L = _dist3(nodes, n1, n2)
            f.write(f"{mid:6d} {n1:6d} {n2:6d} "
                    f"{x1:11.6f} {y1:11.6f} {z1:11.6f} "
                    f"{x2:11.6f} {y2:11.6f} {z2:11.6f} "
                    f"{L:10.6f}\n")

    print(f"# File scritto: {filename}  (n_members={len(members)})")


def write_barrel_rows_report(G, filename="barrel_rows_report.txt"):
    """
    Only for geom=1 (barrel vault) with row-major numbering:
      nid(i,j)=1 + i*ny + j

    Writes:
      - rows at x=const (fixed i) -> nodes ordered by y (increasing j)
      - columns at y=const (fixed j) -> nodes ordered by x (increasing i)

    For each consecutive segment, writes 3D length.
    """
    meta = G.get("meta", {})
    if int(meta.get("geom", 0)) != 1:
        raise ValueError("write_barrel_rows_report: works only for geom=1 (barrel vault).")

    nodes = G["nodes"]
    nx = int(meta["nx"])
    ny = int(meta["ny"])

    def nid(i, j):
        return 1 + i * ny + j

    with open(filename, "w") as f:
        f.write("# Barrel vault mesh report\n")
        f.write(f"# nx={nx}, ny={ny}\n\n")

        f.write("=== ROWS (x approximately constant): for each i list nodes along y ===\n\n")
        for i in range(nx):
            n_first = nid(i, 0)
            x_const = nodes[n_first][0]
            f.write(f"--- Row i={i:3d}  x={x_const: .6f} ---\n")

            for j in range(ny):
                n = nid(i, j)
                x, y, z = nodes[n]
                f.write(f"  node {n:6d}  (x={x: .6f}, y={y: .6f}, z={z: .6f})\n")

            f.write("  segments (along +y):\n")
            seg_lengths = []
            for j in range(ny - 1):
                a = nid(i, j)
                b = nid(i, j + 1)
                L = _dist3(nodes, a, b)
                seg_lengths.append(L)
                f.write(f"    {a:6d}-{b:6d}  L={L: .6f}\n")

            if seg_lengths:
                f.write(f"  stats: min={min(seg_lengths):.6f}  max={max(seg_lengths):.6f}  mean={sum(seg_lengths)/len(seg_lengths):.6f}\n")
            f.write("\n")

        f.write("\n=== COLUMNS (y approximately constant): for each j list nodes along x ===\n\n")
        for j in range(ny):
            n_first = nid(0, j)
            y_const = nodes[n_first][1]
            f.write(f"--- Col j={j:3d}  y={y_const: .6f} ---\n")

            for i in range(nx):
                n = nid(i, j)
                x, y, z = nodes[n]
                f.write(f"  node {n:6d}  (x={x: .6f}, y={y: .6f}, z={z: .6f})\n")

            f.write("  segments (along +x / generatrix direction):\n")
            seg_lengths = []
            for i in range(nx - 1):
                a = nid(i, j)
                b = nid(i + 1, j)
                L = _dist3(nodes, a, b)
                seg_lengths.append(L)
                f.write(f"    {a:6d}-{b:6d}  L={L: .6f}\n")

            if seg_lengths:
                f.write(f"  stats: min={min(seg_lengths):.6f}  max={max(seg_lengths):.6f}  mean={sum(seg_lengths)/len(seg_lengths):.6f}\n")
            f.write("\n")

    print(f"# File scritto: {filename}")

