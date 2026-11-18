import math
import numpy as np

ROUND_DECIMALS = 6
TOL = 1e-9


class GeometryZFunction:
    def __init__(self, geom, B, L, f, *, R_sphere=None, z_center_sphere=None):
        self.geom = int(geom)
        self.B = float(B)
        self.L = float(L)
        self.f = float(f)
        self.R_sphere = None if R_sphere is None else float(R_sphere)
        self.z_center_sphere = None if z_center_sphere is None else float(z_center_sphere)

    def __call__(self, x, y):
        x = float(x)
        y = float(y)
        if self.geom == 1:
            return self.f - (x ** 2) / (2.0 * self.B)
        if self.geom == 2:
            return self.f - (x ** 2) / (2.0 * self.B) - (y ** 2) / (2.0 * self.B)
        if self.geom == 3:
            return self.f + (y ** 2 - x ** 2) / (2.0 * self.B)
        if self.geom == 4:
            if self.R_sphere is None or self.z_center_sphere is None:
                raise ValueError("Kiewitt-8 z function requires sphere radius and center.")
            rr = x * x + y * y
            a = self.B / 2.0
            if rr > a * a + 1e-12:
                return float("nan")
            return _round6(self.z_center_sphere + math.sqrt(max(0.0, self.R_sphere * self.R_sphere - rr)))
        raise ValueError(f"Unsupported geometry type for z function: {self.geom}")


def _round6(value):
    return round(float(value), ROUND_DECIMALS)


def _coord_key(x, y, z):
    return (_round6(x), _round6(y), _round6(z))


def _x_vals_uniform_arc_barrel(B, n, x_min=None, x_max=None, n_dense=20001):
    """
    Coordinates equally spaced in arc-length along the parabolic generatrix
        z = -(x^2)/(2B) + f
    The arc-length does not depend on f.
    """
    if x_min is None:
        x_min = -B / 2.0
    if x_max is None:
        x_max = +B / 2.0

    n = int(n)
    xd = np.linspace(x_min, x_max, int(n_dense))

    # Primitive of sqrt(1 + (dz/dx)^2), with dz/dx = -x/B
    s = 0.5 * (xd * np.sqrt(1.0 + (xd / B) ** 2) + B * np.arcsinh(xd / B))
    s = s - s[0]
    s_targets = np.linspace(0.0, float(s[-1]), n)
    return np.interp(s_targets, s, xd)


def _freegrid_constants(B):
    """Main FreeGrid baseline geometric constants."""
    A = (B / 4.0) * math.sqrt(5.0) + B * math.log((1.0 + math.sqrt(5.0)) / 2.0)
    f_ref = B / 8.0
    R_dome = B / 2.0
    return {"A": A, "f_ref": f_ref, "R_dome": R_dome}


def _add_node(nodes, coord2id, x, y, z):
    key = _coord_key(x, y, z)
    if key in coord2id:
        return coord2id[key]
    nid = len(nodes) + 1
    coord2id[key] = nid
    nodes[nid] = key
    return nid


def _sort_by_angle(ids, nodes):
    tmp = []
    for nid in ids:
        x, y, _ = nodes[nid]
        ang = math.atan2(y, x)
        if ang < 0.0:
            ang += 2.0 * math.pi
        tmp.append((ang, nid))
    return [nid for ang, nid in sorted(tmp, key=lambda t: t[0])]




def _spherical_cap_constants(span, rise):
    """Return sphere radius, z-center and maximum polar angle for a spherical cap."""
    a = float(span) / 2.0
    rise = float(rise)
    R = (a * a + rise * rise) / (2.0 * rise)
    zc = rise - R
    phi_max = math.asin(min(1.0, a / R))
    return R, zc, phi_max


def _kiewitt8_ring_boundary_node(ring_nodes, ring_index, sector_index):
    """Node id on the main rib (sector boundary) for a given ring."""
    ring = ring_nodes[int(ring_index)]
    idx = (int(sector_index) * int(ring_index)) % len(ring)
    return ring[idx]


def _build_kiewitt8_geometry(B, f, n_rings):
    """
    Build a standard Kiewitt-8 single-layer dome on a spherical cap.

    Topology:
      - 1 apex node
      - ring r has 8*r nodes
      - 8 main radial ribs
      - concentric latitudinal rings
      - sector-based node indexing suitable for Kiewitt triangulation
    """
    n_rings = int(n_rings)
    if n_rings < 1:
        raise ValueError('For geom=4 (Kiewitt-8), nx must be >= 1 and is used as number of rings.')

    R, zc, phi_max = _spherical_cap_constants(B, f)

    nodes = {}
    ring_nodes = {}

    # apex
    apex = 1
    nodes[apex] = _coord_key(0.0, 0.0, f)
    nid = 2

    for r in range(1, n_rings + 1):
        phi = phi_max * float(r) / float(n_rings)
        rho = R * math.sin(phi)
        z = zc + R * math.cos(phi)
        n_on_ring = 8 * r
        ring_nodes[r] = []
        for j in range(n_on_ring):
            theta = 2.0 * math.pi * float(j) / float(n_on_ring)
            x = rho * math.cos(theta)
            y = rho * math.sin(theta)
            nodes[nid] = _coord_key(x, y, z)
            ring_nodes[r].append(nid)
            nid += 1

    boundary = list(ring_nodes[n_rings])
    internal = [apex] + [nid for r in range(1, n_rings) for nid in ring_nodes[r]]

    rib_lines = {}
    for s in range(8):
        rib_lines[f'rib_{s+1}'] = [apex] + [
            _kiewitt8_ring_boundary_node(ring_nodes, r, s) for r in range(1, n_rings + 1)
        ]

    ring_lines = {f'ring_{r}': list(ring_nodes[r]) for r in range(1, n_rings + 1)}

    boundary_sectors = {}
    for s in range(8):
        start = s * n_rings
        ids = []
        for t in range(n_rings):
            ids.append(boundary[(start + t) % len(boundary)])
        boundary_sectors[f'sector_{s+1}'] = ids

    corners = [_kiewitt8_ring_boundary_node(ring_nodes, n_rings, s) for s in range(8)]

    def z_fun(x, y):
        rr = x * x + y * y
        a = B / 2.0
        if rr > a * a + 1e-12:
            return float('nan')
        return _round6(zc + math.sqrt(max(0.0, R * R - rr)))

    nodes_internal = {nid_: nodes[nid_] for nid_ in internal}
    nodes_boundary = {nid_: nodes[nid_] for nid_ in boundary}

    node_sets = {
        'apex': [apex],
        'internal': internal,
        'boundary': boundary,
        'corners': corners,
        'rings': ring_nodes,
        'rib_lines': rib_lines,
        'ring_lines': ring_lines,
        'boundary_sectors': boundary_sectors,
    }

    meta = {
        'geom': 4,
        'grid': 2,
        'B': B,
        'L': B,
        'f': f,
        'nx': n_rings,
        'ny': 8,
        'n_rings': n_rings,
        'n_sectors': 8,
        'R_base': B / 2.0,
        'R_sphere': R,
        'z_center_sphere': zc,
        'dome_type': 'Kiewitt-8',
    }

    return {
        'nodes': nodes,
        'nodes_internal': nodes_internal,
        'nodes_boundary': nodes_boundary,
        'node_sets': node_sets,
        'domain': {'shape': 'circle', 'R': B / 2.0},
        'meta': meta,
        'z_fun': GeometryZFunction(4, B, B, f, R_sphere=R, z_center_sphere=zc),
    }


def define_geometry(geom, B, L, f, nx, ny, grid):
    """
    FreeGrid baseline geometries:
      geom = 1 -> barrel vault
      geom = 2 -> parabolic dome
      geom = 3 -> hyperbolic paraboloid (hypar)

    Notes:
    - Coordinates are rounded to 6 decimals, in agreement with FreeGrid.
    - For the benchmark baseline, the recommended values are:
          f = B/8
          nx = ny = 21
      and, for the barrel, L = A (generatrix arc-length).
    - The parameter `grid` is kept for interface compatibility.
    """
    if geom not in (1, 2, 3, 4):
        raise ValueError("geom must be 1 (barrel), 2 (dome), 3 (hypar) or 4 (Kiewitt-8 dome)")
    if grid not in (1, 2):
        raise ValueError("grid must be 1 (quad) or 2 (tri)")
    if B <= 0 or L <= 0 or f <= 0:
        raise ValueError("B, L, f must be > 0")
    if nx < 2 or ny < 2:
        raise ValueError("nx and ny must be >= 2")

    nx, ny = int(nx), int(ny)
    const = _freegrid_constants(B)
    A = const["A"]

    # ------------------------------------------------------------------
    # 1) BARREL VAULT
    # ------------------------------------------------------------------
    if geom == 1:
        domain = {"x": (-B / 2.0, B / 2.0), "y": (-L / 2.0, L / 2.0)}

        # FreeGrid: subdivisions along the parabolic arch are uniform in arc-length
        x_vals = _x_vals_uniform_arc_barrel(B, nx, x_min=domain["x"][0], x_max=domain["x"][1])
        y_vals = np.linspace(domain["y"][0], domain["y"][1], ny)

        dx = (domain["x"][1] - domain["x"][0]) / (nx - 1)
        dy = (domain["y"][1] - domain["y"][0]) / (ny - 1)

        def z_fun(x, y):
            return f - (x ** 2) / (2.0 * B)

        def nid(i, j):
            return 1 + i * ny + j

        nodes = {}
        for i, x in enumerate(x_vals):
            for j, y in enumerate(y_vals):
                nodes[nid(i, j)] = _coord_key(x, y, z_fun(x, y))

        corners = [nid(0, 0), nid(0, ny - 1), nid(nx - 1, 0), nid(nx - 1, ny - 1)]
        side_xmin = [nid(0, j) for j in range(ny)]
        side_xmax = [nid(nx - 1, j) for j in range(ny)]
        side_ymin = [nid(i, 0) for i in range(nx)]
        side_ymax = [nid(i, ny - 1) for i in range(nx)]
        boundary = list(dict.fromkeys(side_xmin + side_ymax + side_xmax[::-1] + side_ymin[::-1]))

        internal = [nid(i, j) for i in range(1, nx - 1) for j in range(1, ny - 1)]
        nodes_internal = {nid_: nodes[nid_] for nid_ in internal}
        nodes_boundary = {nid_: nodes[nid_] for nid_ in boundary}

        node_sets = {
            "internal": internal,
            "boundary": boundary,
            "corners": corners,
            "boundary_sides": {
                "x_min": side_xmin,
                "x_max": side_xmax,
                "y_min": side_ymin,
                "y_max": side_ymax,
            },
        }

        meta = {
            "geom": geom,
            "grid": grid,
            "B": B,
            "L": L,
            "f": f,
            "nx": nx,
            "ny": ny,
            "dx": dx,
            "dy": dy,
            "A_arc_generator": A,
            "b_arc_mean": A / (nx - 1),
            "S_plan": B * L,
            "freegrid_reference": {
                "L_barrel_baseline": A,
                "f_baseline": B / 8.0,
            },
        }

        return {
            "nodes": nodes,
            "nodes_internal": nodes_internal,
            "nodes_boundary": nodes_boundary,
            "node_sets": node_sets,
            "domain": domain,
            "meta": meta,
            "z_fun": GeometryZFunction(geom, B, L, f),
        }

    # ------------------------------------------------------------------
    # 2) PARABOLIC DOME
    # ------------------------------------------------------------------
    if geom == 2:
        # FreeGrid surface
        def z_fun(x, y):
            return f - (x ** 2) / (2.0 * B) - (y ** 2) / (2.0 * B)

        R = math.sqrt(2.0 * B * f)
        domain = {"x": (-R, R), "y": (-R, R)}

        # IMPORTANT: FreeGrid OBJ is not built on a uniform Cartesian spacing.
        # The x/y levels inherit the same arc-uniform subdivision used for the barrel.
        x_vals = _x_vals_uniform_arc_barrel(B, nx, x_min=-R, x_max=R)
        y_vals = _x_vals_uniform_arc_barrel(B, ny, x_min=-R, x_max=R)

        dx = (2.0 * R) / (nx - 1)
        dy = (2.0 * R) / (ny - 1)

        nodes = {}
        coord2id = {}

        internal = []
        for x in x_vals:
            for y in y_vals:
                if x * x + y * y < R * R - TOL:
                    internal.append(_add_node(nodes, coord2id, x, y, z_fun(x, y)))

        boundary = []
        for x in x_vals:
            yy = math.sqrt(max(0.0, R * R - x * x))
            boundary.append(_add_node(nodes, coord2id, x, +yy, 0.0))
            boundary.append(_add_node(nodes, coord2id, x, -yy, 0.0))

        for y in y_vals:
            xx = math.sqrt(max(0.0, R * R - y * y))
            boundary.append(_add_node(nodes, coord2id, +xx, y, 0.0))
            boundary.append(_add_node(nodes, coord2id, -xx, y, 0.0))

        boundary = list(dict.fromkeys(boundary))

        corners = [
            _add_node(nodes, coord2id, +R, 0.0, 0.0),
            _add_node(nodes, coord2id, 0.0, +R, 0.0),
            _add_node(nodes, coord2id, -R, 0.0, 0.0),
            _add_node(nodes, coord2id, 0.0, -R, 0.0),
        ]

        b_spokes = {"Q1": [], "Q2": [], "Q3": [], "Q4": []}
        for nid in boundary:
            x, y, _ = nodes[nid]
            ang = math.atan2(y, x)
            if ang < 0.0:
                ang += 2.0 * math.pi
            if 0.0 <= ang < 0.5 * math.pi:
                b_spokes["Q1"].append(nid)
            elif 0.5 * math.pi <= ang < math.pi:
                b_spokes["Q2"].append(nid)
            elif math.pi <= ang < 1.5 * math.pi:
                b_spokes["Q3"].append(nid)
            else:
                b_spokes["Q4"].append(nid)

        for key in b_spokes:
            b_spokes[key] = _sort_by_angle(b_spokes[key], nodes)

        nodes_internal = {nid_: nodes[nid_] for nid_ in internal}
        nodes_boundary = {nid_: nodes[nid_] for nid_ in boundary}

        node_sets = {
            "internal": internal,
            "boundary": boundary,
            "corners": corners,
            "boundary_spokes": b_spokes,
        }

        meta = {
            "geom": geom,
            "grid": grid,
            "B": B,
            "L": L,
            "f": f,
            "nx": nx,
            "ny": ny,
            "dx": dx,
            "dy": dy,
            "A_arc_generator": A,
            "b_arc_mean": A / (nx - 1),
            "S_plan": math.pi * R * R,
            "R_base": R,
            "freegrid_reference": {
                "R_baseline": B / 2.0,
                "f_baseline": B / 8.0,
            },
        }

        return {
            "nodes": nodes,
            "nodes_internal": nodes_internal,
            "nodes_boundary": nodes_boundary,
            "node_sets": node_sets,
            "domain": domain,
            "meta": meta,
            "z_fun": GeometryZFunction(geom, B, L, f),
        }

    # ------------------------------------------------------------------
    # 4) KIEWITT-8 DOME
    # ------------------------------------------------------------------
    if geom == 4:
        # For the Kiewitt-8 test dome, nx is used as number of rings.
        # L and ny are kept only for interface compatibility.
        return _build_kiewitt8_geometry(B=B, f=f, n_rings=nx)

    # ------------------------------------------------------------------
    # 3) HYPAR
    # ------------------------------------------------------------------
    def z_fun(x, y):
        # FreeGrid hypar: saddle obtained from opposite parabolic curvatures
        return f + (y ** 2 - x ** 2) / (2.0 * B)

    # Plan domain: diamond with vertices (±B/2, 0), (0, ±B/2)
    domain = {
        "x": (-B / 2.0, B / 2.0),
        "y": (-B / 2.0, B / 2.0),
        "plan_shape": "diamond_abs(x)+abs(y)<=B/2",
    }

    x_vals = _x_vals_uniform_arc_barrel(B, nx, x_min=-B / 2.0, x_max=B / 2.0)
    y_vals = _x_vals_uniform_arc_barrel(B, ny, x_min=-B / 2.0, x_max=B / 2.0)

    dx = B / (nx - 1)
    dy = B / (ny - 1)

    nodes = {}
    coord2id = {}

    internal = []
    for x in x_vals:
        for y in y_vals:
            if abs(x) + abs(y) < B / 2.0 - TOL:
                internal.append(_add_node(nodes, coord2id, x, y, z_fun(x, y)))

    boundary = []

    # Intersections with the diamond edges, taken on both x- and y-levels,
    # exactly as in the FreeGrid OBJ discretization.
    for x in x_vals:
        yb = B / 2.0 - abs(x)
        boundary.append(_add_node(nodes, coord2id, x, +yb, z_fun(x, +yb)))
        boundary.append(_add_node(nodes, coord2id, x, -yb, z_fun(x, -yb)))

    for y in y_vals:
        xb = B / 2.0 - abs(y)
        boundary.append(_add_node(nodes, coord2id, +xb, y, z_fun(+xb, y)))
        boundary.append(_add_node(nodes, coord2id, -xb, y, z_fun(-xb, y)))

    boundary = list(dict.fromkeys(boundary))

    corners = [
        _add_node(nodes, coord2id, +B / 2.0, 0.0, z_fun(+B / 2.0, 0.0)),
        _add_node(nodes, coord2id, 0.0, +B / 2.0, z_fun(0.0, +B / 2.0)),
        _add_node(nodes, coord2id, -B / 2.0, 0.0, z_fun(-B / 2.0, 0.0)),
        _add_node(nodes, coord2id, 0.0, -B / 2.0, z_fun(0.0, -B / 2.0)),
    ]

    edge_1 = []  # x>=0, y>=0, x+y=B/2
    edge_2 = []  # x<=0, y>=0, -x+y=B/2
    edge_3 = []  # x<=0, y<=0, -x-y=B/2
    edge_4 = []  # x>=0, y<=0, x-y=B/2
    y_nonnegative = []
    y_negative = []

    for nid in boundary:
        x, y, _ = nodes[nid]
        if y >= -1e-6:
            y_nonnegative.append(nid)
        if y <= 1e-6:
            y_negative.append(nid)

        if x >= -1e-6 and y >= -1e-6 and abs(x + y - B / 2.0) <= 2e-6:
            edge_1.append(nid)
        elif x <= 1e-6 and y >= -1e-6 and abs(-x + y - B / 2.0) <= 2e-6:
            edge_2.append(nid)
        elif x <= 1e-6 and y <= 1e-6 and abs(-x - y - B / 2.0) <= 2e-6:
            edge_3.append(nid)
        elif x >= -1e-6 and y <= 1e-6 and abs(x - y - B / 2.0) <= 2e-6:
            edge_4.append(nid)

    # Ordered along the perimeter, counter-clockwise
    def sort_edge(ids):
        return sorted(ids, key=lambda nid: (math.atan2(nodes[nid][1], nodes[nid][0]) + 2.0 * math.pi) % (2.0 * math.pi))

    edge_1 = sort_edge(edge_1)
    edge_2 = sort_edge(edge_2)
    edge_3 = sort_edge(edge_3)
    edge_4 = sort_edge(edge_4)

    nodes_internal = {nid_: nodes[nid_] for nid_ in internal}
    nodes_boundary = {nid_: nodes[nid_] for nid_ in boundary}

    node_sets = {
        "internal": internal,
        "boundary": boundary,
        "corners": corners,
        "boundary_sides": {
            "edge_1": edge_1,
            "edge_2": edge_2,
            "edge_3": edge_3,
            "edge_4": edge_4,
        },
        "boundary_half": {
            "y_ge_0": list(dict.fromkeys(y_nonnegative)),
            "y_le_0": list(dict.fromkeys(y_negative)),
        },
    }

    meta = {
        "geom": geom,
        "grid": grid,
        "B": B,
        "L": L,
        "f": f,
        "nx": nx,
        "ny": ny,
        "dx": dx,
        "dy": dy,
        "A_arc_generator": A,
        "b_arc_mean": A / (nx - 1),
        "S_plan": B * B / 2.0,
        "h_max": f + B / 8.0,
        "freegrid_reference": {
            "f_baseline": B / 8.0,
            "h_baseline": B / 4.0,
            "plan_area_baseline": B * B / 2.0,
        },
    }

    return {
        "nodes": nodes,
        "nodes_internal": nodes_internal,
        "nodes_boundary": nodes_boundary,
        "node_sets": node_sets,
        "domain": domain,
        "meta": meta,
        "z_fun": GeometryZFunction(geom, B, L, f),
    }



def write_nodes_to_txt(geom_data, filename="nodes.txt"):
    """Write nodes in the format: node_id x y z"""
    with open(filename, "w") as fobj:
        fobj.write("# node_id    x           y           z\n")
        for nid in sorted(geom_data["nodes"].keys()):
            x, y, z = geom_data["nodes"][nid]
            fobj.write(f"{nid:6d}  {x:12.6f}  {y:12.6f}  {z:12.6f}\n")

    print(f"# File scritto: {filename}")
