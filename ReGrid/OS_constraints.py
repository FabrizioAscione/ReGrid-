import math
import openseespy.opensees as ops


def apply_costr(G, ndf, c1, c2, c3, c4, c5=None):
    """
    Apply boundary constraints based on geometry-specific node sets.

    Mapping used:

    BARREL (geom=1)
      c1 -> boundary_sides['x_max']
      c2 -> boundary_sides['x_min']
      c3 -> boundary_sides['y_max']
      c4 -> boundary_sides['y_min']
      c5 -> corners only

    DOME (geom=2)
      c1 -> boundary_spokes['Q1']
      c2 -> boundary_spokes['Q2']
      c3 -> boundary_spokes['Q3']
      c4 -> boundary_spokes['Q4']
      c5 ignored

    HYPAR (geom=3)
      c1 -> boundary_sides['edge_1']
      c2 -> boundary_sides['edge_2']
      c3 -> boundary_sides['edge_3']
      c4 -> boundary_sides['edge_4']
      c5 -> corners only

    KIEWITT-8 DOME (geom=4)
      c1 -> boundary nodes in plan quadrant Q1 (x>=0, y>=0)
      c2 -> boundary nodes in plan quadrant Q2 (x<0,  y>=0)
      c3 -> boundary nodes in plan quadrant Q3 (x<0,  y<0)
      c4 -> boundary nodes in plan quadrant Q4 (x>=0, y<0)
      c5 -> main rib end nodes on the boundary (8 nodes) if present in node_sets['corners']

    Returns
    -------
    py_text : str
        Text of executed ops.fix(...) commands.
    fixed_groups : dict
        Node tags fixed in each boundary group.
    """

    def _check_c(c, name):
        if c is None:
            return None
        if len(c) != ndf:
            raise ValueError(f"{name} must have length ndf={ndf}. Got {len(c)}.")
        return tuple(int(v) for v in c)

    def _unique_keep_order(seq):
        seen = set()
        out = []
        for item in seq:
            item = int(item)
            if item not in seen:
                seen.add(item)
                out.append(item)
        return out

    def _fix_many(node_list, c, label):
        node_list = _unique_keep_order(node_list)
        if not node_list or c is None:
            fixed_groups[label] = []
            return
        fixed = []
        for nid in node_list:
            if nid in globally_fixed:
                continue
            ops.fix(int(nid), *c)
            py_lines.append(f"ops.fix({int(nid)}, {', '.join(str(v) for v in c)})")
            fixed.append(int(nid))
            globally_fixed.add(int(nid))
        fixed_groups[label] = fixed

    def _quadrants_from_xy(x, y, eps=1e-12):
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

    def _nodes_in_boundary_quadrants(nodes, boundary_ids, quadrant):
        out = []
        for nid in boundary_ids:
            x, y, _ = nodes[int(nid)]
            if quadrant in _quadrants_from_xy(x, y):
                out.append(int(nid))
        return out

    ndf = int(ndf)
    if ndf <= 0:
        raise ValueError("ndf must be a positive integer.")

    c1 = _check_c(c1, "c1")
    c2 = _check_c(c2, "c2")
    c3 = _check_c(c3, "c3")
    c4 = _check_c(c4, "c4")
    c5 = _check_c(c5, "c5") if c5 is not None else None

    meta = G.get("meta", {})
    geom = int(meta.get("geom", 0))
    node_sets = G.get("node_sets", {})
    nodes = G.get("nodes", {})

    py_lines = []
    fixed_groups = {}
    globally_fixed = set()

    # ---------------- BARREL ----------------
    if geom == 1:
        sides = node_sets.get("boundary_sides", {})
        if not sides:
            raise ValueError("For geom=1, expected G['node_sets']['boundary_sides'].")

        corners = set(int(n) for n in node_sets.get("corners", []))

        x_max = [n for n in sides.get("x_max", []) if int(n) not in corners]
        x_min = [n for n in sides.get("x_min", []) if int(n) not in corners]
        y_max = [n for n in sides.get("y_max", []) if int(n) not in corners]
        y_min = [n for n in sides.get("y_min", []) if int(n) not in corners]

        _fix_many(x_max, c1, "c1_x_max")
        _fix_many(x_min, c2, "c2_x_min")
        _fix_many(y_max, c3, "c3_y_max")
        _fix_many(y_min, c4, "c4_y_min")
        _fix_many(sorted(corners), c5, "c5_corners")

    # ---------------- DOME ----------------
    elif geom == 2:
        spokes = node_sets.get("boundary_spokes", {})
        if not spokes:
            raise ValueError("For geom=2, expected G['node_sets']['boundary_spokes'] with Q1..Q4.")

        _fix_many(spokes.get("Q1", []), c1, "c1_Q1")
        _fix_many(spokes.get("Q2", []), c2, "c2_Q2")
        _fix_many(spokes.get("Q3", []), c3, "c3_Q3")
        _fix_many(spokes.get("Q4", []), c4, "c4_Q4")
        fixed_groups["c5_corners"] = []

    # ---------------- HYPAR ----------------
    elif geom == 3:
        sides = node_sets.get("boundary_sides", {})
        if not sides:
            raise ValueError("For geom=3, expected G['node_sets']['boundary_sides'] with edge_1..edge_4.")

        corners = set(int(n) for n in node_sets.get("corners", []))

        edge_1 = [n for n in sides.get("edge_1", []) if int(n) not in corners]
        edge_2 = [n for n in sides.get("edge_2", []) if int(n) not in corners]
        edge_3 = [n for n in sides.get("edge_3", []) if int(n) not in corners]
        edge_4 = [n for n in sides.get("edge_4", []) if int(n) not in corners]

        _fix_many(edge_1, c1, "c1_edge_1")
        _fix_many(edge_2, c2, "c2_edge_2")
        _fix_many(edge_3, c3, "c3_edge_3")
        _fix_many(edge_4, c4, "c4_edge_4")
        _fix_many(sorted(corners), c5, "c5_corners")

    # ---------------- KIEWITT-8 DOME ----------------
    elif geom == 4:
        boundary = [int(n) for n in node_sets.get("boundary", [])]
        if not boundary:
            raise ValueError("For geom=4, expected G['node_sets']['boundary'].")

        corners = set(int(n) for n in node_sets.get("corners", []))

        q1 = [n for n in _nodes_in_boundary_quadrants(nodes, boundary, 1) if n not in corners]
        q2 = [n for n in _nodes_in_boundary_quadrants(nodes, boundary, 2) if n not in corners]
        q3 = [n for n in _nodes_in_boundary_quadrants(nodes, boundary, 3) if n not in corners]
        q4 = [n for n in _nodes_in_boundary_quadrants(nodes, boundary, 4) if n not in corners]

        _fix_many(q1, c1, "c1_Q1")
        _fix_many(q2, c2, "c2_Q2")
        _fix_many(q3, c3, "c3_Q3")
        _fix_many(q4, c4, "c4_Q4")
        _fix_many(sorted(corners), c5, "c5_corners")

    else:
        raise NotImplementedError(f"geom={geom} not supported in apply_costr().")

    return "\n".join(py_lines), fixed_groups
