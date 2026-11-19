import copy
import math
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Tuple, Union


NodeId = int
MoveRange = Tuple[float, float]
MemberKey = Tuple[int, int]

_TOL = 1e-9


def parse_move_range(spec: Any, *, name: str) -> MoveRange:
    if isinstance(spec, str):
        parts = [p.strip() for p in spec.split(",") if p.strip()]
        if len(parts) != 2:
            raise ValueError(f"{name} must be like '+0.50,-0.50' or '-0.50,+0.50'.")
        values = [float(parts[0]), float(parts[1])]
    elif isinstance(spec, (list, tuple)) and len(spec) == 2:
        values = [float(spec[0]), float(spec[1])]
    else:
        raise ValueError(f"{name} must be a string or a 2-item list/tuple.")

    return (min(values), max(values))


def _round6(value: float) -> float:
    return round(float(value), 6)


def _group_key(x: float, y: float) -> Tuple[float, float]:
    return (_round6(abs(x)), _round6(abs(y)))


def _sign(value: float) -> int:
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0


def _member_key(a: int, b: int) -> MemberKey:
    a = int(a)
    b = int(b)
    return (a, b) if a < b else (b, a)


def _member_group_key(
    G: Mapping[str, Any],
    member: MemberKey,
    *,
    tol_digits: int = 6,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], float]:
    nodes = G.get("nodes", {})
    (na, nb) = member
    xa, ya, za = nodes[int(na)]
    xb, yb, zb = nodes[int(nb)]
    end_a = (round(abs(float(xa)), tol_digits), round(abs(float(ya)), tol_digits), round(float(za), tol_digits))
    end_b = (round(abs(float(xb)), tol_digits), round(abs(float(yb)), tol_digits), round(float(zb), tol_digits))
    ordered = tuple(sorted((end_a, end_b)))
    length = round(
        math.sqrt((float(xb) - float(xa)) ** 2 + (float(yb) - float(ya)) ** 2 + (float(zb) - float(za)) ** 2),
        tol_digits,
    )
    return ordered[0], ordered[1], length


def build_quarter_mirror_move_spec(
    G: Mapping[str, Any],
    *,
    x_move_range: Any,
    y_move_range: Any,
    tol: float = _TOL,
) -> Dict[str, Any]:
    nodes = G.get("nodes", {})
    if not nodes:
        raise ValueError("Geometry G has no nodes.")

    x_range = parse_move_range(x_move_range, name="x_move_range")
    y_range = parse_move_range(y_move_range, name="y_move_range")
    boundary_ids = set(int(v) for v in G.get("node_sets", {}).get("boundary", []))

    grouped: Dict[Tuple[float, float], List[Dict[str, Any]]] = {}
    for nid, (x, y, z) in nodes.items():
        grouped.setdefault(_group_key(x, y), []).append(
            {
                "nid": int(nid),
                "x": float(x),
                "y": float(y),
                "z": float(z),
                "sx": _sign(float(x)),
                "sy": _sign(float(y)),
                "is_boundary": int(nid) in boundary_ids,
            }
        )

    movable_groups: List[Dict[str, Any]] = []
    fixed_groups: List[Dict[str, Any]] = []

    for key, members in sorted(grouped.items(), key=lambda item: item[0]):
        members_sorted = sorted(members, key=lambda item: (item["sx"], item["sy"], item["nid"]))
        leader = next(
            (item for item in members_sorted if item["sx"] > 0 and item["sy"] > 0),
            None,
        )
        has_axis_node = any(abs(item["x"]) <= tol or abs(item["y"]) <= tol for item in members_sorted)
        has_boundary_node = any(item["is_boundary"] for item in members_sorted)

        group_info = {
            "key": key,
            "leader_nid": None if leader is None else int(leader["nid"]),
            "leader_xy": None if leader is None else (float(leader["x"]), float(leader["y"])),
            "members": [
                {
                    "nid": int(item["nid"]),
                    "sx": int(item["sx"]),
                    "sy": int(item["sy"]),
                    "x0": float(item["x"]),
                    "y0": float(item["y"]),
                    "z0": float(item["z"]),
                    "is_boundary": bool(item["is_boundary"]),
                }
                for item in members_sorted
            ],
        }

        if leader is None or len(members_sorted) != 4 or has_axis_node or has_boundary_node:
            fixed_groups.append(group_info)
            continue

        movable_groups.append(group_info)

    return {
        "symmetry_mode": "quarter_mirror_xy",
        "x_move_range": x_range,
        "y_move_range": y_range,
        "movable_groups": movable_groups,
        "fixed_groups": fixed_groups,
        "leader_node_ids": [group["leader_nid"] for group in movable_groups],
        "n_var": 2 * len(movable_groups),
    }


def build_reuse_member_mask_spec(
    G: Mapping[str, Any],
    M: Mapping[str, Any],
    *,
    mode: str = "member",
) -> Dict[str, Any]:
    members = [_member_key(a, b) for (a, b) in M.get("members", [])]
    normalized_mode = str(mode or "member").strip().lower()
    if normalized_mode not in {"member", "symmetric", "global"}:
        raise ValueError("reuse member mask mode must be 'member', 'symmetric', or 'global'.")

    groups: List[Dict[str, Any]] = []
    if normalized_mode == "global":
        groups.append(
            {
                "label": "reuse_all_members",
                "members": tuple(sorted(members)),
            }
        )
    elif normalized_mode == "member":
        for mk in sorted(members):
            groups.append(
                {
                    "label": f"reuse_flag_{int(mk[0])}_{int(mk[1])}",
                    "members": (mk,),
                }
            )
    else:
        grouped: Dict[Tuple[Tuple[float, float, float], Tuple[float, float, float], float], List[MemberKey]] = {}
        for mk in members:
            grouped.setdefault(_member_group_key(G, mk), []).append(mk)
        for idx, key in enumerate(sorted(grouped.keys()), start=1):
            group_members = tuple(sorted(grouped[key]))
            if len(group_members) == 1:
                mk = group_members[0]
                label = f"reuse_sym_1_{int(mk[0])}_{int(mk[1])}"
            else:
                label = f"reuse_sym_group_{idx}"
            groups.append(
                {
                    "label": label,
                    "members": group_members,
                }
            )

    return {
        "mode": normalized_mode,
        "groups": groups,
        "n_var": len(groups),
    }



def build_quarter_mirror_diagonal_spec(
    G: Mapping[str, Any],
    M: Mapping[str, Any],
    *,
    pattern: str = "alternating",
    tol: float = _TOL,
) -> Dict[str, Any]:
    """Build symmetry-linked binary variables for optional grid diagonals.

    One decision variable controls one diagonal in the +x/+y quarter together
    with all of its reflections about the global x/y symmetry axes.  The
    quadrilateral faces remain unchanged; only the structural ``members`` list
    is modified when a mask is applied.

    Parameters
    ----------
    pattern:
        ``"alternating"`` (default), ``"slash"`` or ``"backslash"``.
        The pattern only selects which of the two possible diagonals of each
        leader quadrilateral is eligible.  The GA variable then decides whether
        that diagonal group is present (1) or absent (0).
    """
    nodes = G.get("nodes", {})
    faces = list(M.get("faces", []) or [])
    if not nodes:
        raise ValueError("Geometry G has no nodes.")

    normalized_pattern = str(pattern or "alternating").strip().lower()
    aliases = {
        "a": "slash",
        "n1_n3": "slash",
        "slash": "slash",
        "/": "slash",
        "b": "backslash",
        "n2_n4": "backslash",
        "backslash": "backslash",
        "\\": "backslash",
        "checkerboard": "alternating",
        "alternating": "alternating",
    }
    if normalized_pattern not in aliases:
        raise ValueError("diagonal pattern must be 'alternating', 'slash', or 'backslash'.")
    normalized_pattern = aliases[normalized_pattern]

    # Coordinate lookup in plan.  z is deliberately not used because mirrored
    # geometries such as hypars can have different z values at mirrored x/y.
    coord_to_nid: Dict[Tuple[float, float], int] = {}
    for nid, xyz in nodes.items():
        coord_to_nid[(_round6(float(xyz[0])), _round6(float(xyz[1])))] = int(nid)

    quad_faces: List[Tuple[int, Tuple[int, int, int, int], float, float]] = []
    possible_diagonals = set()
    for idx, face in enumerate(faces, start=1):
        vals = list(face)
        if len(vals) >= 5:
            fid = int(vals[0])
            n1, n2, n3, n4 = map(int, vals[-4:])
        elif len(vals) == 4:
            fid = idx
            n1, n2, n3, n4 = map(int, vals)
        else:
            continue
        if any(nid not in nodes for nid in (n1, n2, n3, n4)):
            continue
        possible_diagonals.add(_member_key(n1, n3))
        possible_diagonals.add(_member_key(n2, n4))
        cx = sum(float(nodes[nid][0]) for nid in (n1, n2, n3, n4)) / 4.0
        cy = sum(float(nodes[nid][1]) for nid in (n1, n2, n3, n4)) / 4.0
        quad_faces.append((fid, (n1, n2, n3, n4), cx, cy))

    # The +x/+y faces are the independent quarter.  Faces centred exactly on a
    # symmetry axis are intentionally left fixed; this avoids assigning the
    # same physical diagonal to two independent variables.
    leaders = [item for item in quad_faces if item[2] > tol and item[3] > tol]
    if not leaders:
        return {
            "symmetry_mode": "quarter_mirror_xy",
            "pattern": normalized_pattern,
            "groups": [],
            "candidate_members": (),
            "n_var": 0,
        }

    x_levels = sorted({_round6(item[2]) for item in leaders})
    y_levels = sorted({_round6(item[3]) for item in leaders})
    x_rank = {value: idx for idx, value in enumerate(x_levels)}
    y_rank = {value: idx for idx, value in enumerate(y_levels)}

    existing_members = {_member_key(a, b) for a, b in M.get("members", [])}
    groups: List[Dict[str, Any]] = []
    seen_groups = set()

    for fid, (n1, n2, n3, n4), cx, cy in sorted(leaders, key=lambda item: (item[2], item[3], item[0])):
        use_slash = normalized_pattern == "slash"
        if normalized_pattern == "backslash":
            use_slash = False
        elif normalized_pattern == "alternating":
            parity = (x_rank[_round6(cx)] + y_rank[_round6(cy)]) % 2
            use_slash = parity == 0

        leader_edge = _member_key(n1, n3) if use_slash else _member_key(n2, n4)
        a0, b0 = leader_edge
        xa, ya, _ = nodes[a0]
        xb, yb, _ = nodes[b0]

        mirrored_edges = set()
        for sx in (-1, 1):
            for sy in (-1, 1):
                na = coord_to_nid.get((_round6(sx * float(xa)), _round6(sy * float(ya))))
                nb = coord_to_nid.get((_round6(sx * float(xb)), _round6(sy * float(yb))))
                if na is None or nb is None or int(na) == int(nb):
                    continue
                edge = _member_key(int(na), int(nb))
                if edge in possible_diagonals:
                    mirrored_edges.add(edge)

        if not mirrored_edges:
            continue
        group_members = tuple(sorted(mirrored_edges))
        if group_members in seen_groups:
            continue
        seen_groups.add(group_members)
        group_idx = len(groups) + 1
        groups.append(
            {
                "label": f"diagonal_group_{group_idx}",
                "leader_face_id": int(fid),
                "leader_member": leader_edge,
                "members": group_members,
                "initial_flags": tuple(1 if mk in existing_members else 0 for mk in group_members),
            }
        )

    candidate_members = tuple(sorted({mk for group in groups for mk in group["members"]}))
    return {
        "symmetry_mode": "quarter_mirror_xy",
        "pattern": normalized_pattern,
        "groups": groups,
        "candidate_members": candidate_members,
        "n_var": len(groups),
    }


def apply_diagonal_topology_mask(
    M: Mapping[str, Any],
    diagonal_spec: Mapping[str, Any],
    diagonal_flags: Mapping[MemberKey, int],
) -> Dict[str, Any]:
    """Return a mesh with candidate diagonal members switched on/off.

    Existing non-candidate members are preserved. Candidate diagonals are first
    removed from the base topology and then re-added when their binary flag is 1.
    This makes the operation valid both for an initially quadrangular mesh and
    for a mesh that already contains some of the candidate diagonals.
    """
    new_M = copy.deepcopy(dict(M))
    members = {_member_key(a, b) for a, b in new_M.get("members", [])}
    candidate_members = {
        _member_key(int(mk[0]), int(mk[1]))
        for mk in diagonal_spec.get("candidate_members", ())
    }
    members.difference_update(candidate_members)

    active_diagonals = set()
    for mk in candidate_members:
        if int(diagonal_flags.get(mk, 0)) == 1:
            members.add(mk)
            active_diagonals.add(mk)

    new_M["members"] = sorted(members)
    meta = dict(new_M.get("meta", {}))
    meta.update(
        {
            "diagonal_topology_optimization_active": True,
            "diagonal_symmetry_mode": str(diagonal_spec.get("symmetry_mode", "quarter_mirror_xy")),
            "diagonal_pattern": str(diagonal_spec.get("pattern", "alternating")),
            "n_candidate_diagonals": int(len(candidate_members)),
            "n_active_diagonals": int(len(active_diagonals)),
            "n_members": int(len(new_M["members"])),
            "active_diagonals": [list(mk) for mk in sorted(active_diagonals)],
        }
    )
    new_M["meta"] = meta
    return new_M


def describe_diagonal_spec(diagonal_spec: Mapping[str, Any]) -> str:
    groups = list(diagonal_spec.get("groups", []) or [])
    members = list(diagonal_spec.get("candidate_members", []) or [])
    return (
        f"symmetry={diagonal_spec.get('symmetry_mode', 'quarter_mirror_xy')} | "
        f"pattern={diagonal_spec.get('pattern', 'alternating')} | "
        f"groups={len(groups)} | candidate_diagonals={len(members)} | "
        f"n_var={int(diagonal_spec.get('n_var', len(groups)))}"
    )

def move_map_from_vector(move_spec: Mapping[str, Any], vector: Sequence[float]) -> Dict[int, Tuple[float, float]]:
    groups = list(move_spec.get("movable_groups", []))
    values = [float(v) for v in vector]
    expected = 2 * len(groups)
    if len(values) != expected:
        raise ValueError(f"Expected {expected} decision variables, got {len(values)}.")

    out: Dict[int, Tuple[float, float]] = {}
    for idx, group in enumerate(groups):
        leader_nid = int(group["leader_nid"])
        out[leader_nid] = (values[2 * idx], values[2 * idx + 1])
    return out


def apply_quarter_mirror_moves(
    G: Mapping[str, Any],
    move_spec: Mapping[str, Any],
    moves: Union[Mapping[NodeId, Sequence[float]], Sequence[float]],
    *,
    strict: bool = True,
    tol: float = _TOL,
) -> Dict[str, Any]:
    z_fun = G.get("z_fun", None)
    if z_fun is None:
        raise ValueError("Geometry G has no z_fun, cannot rebuild shape-consistent z coordinates.")

    move_map = moves if isinstance(moves, Mapping) else move_map_from_vector(move_spec, moves)
    x_min, x_max = move_spec["x_move_range"]
    y_min, y_max = move_spec["y_move_range"]

    new_G = copy.deepcopy(dict(G))
    new_nodes: MutableMapping[int, Tuple[float, float, float]] = dict(new_G["nodes"])

    for group in move_spec.get("movable_groups", []):
        leader_nid = int(group["leader_nid"])
        dx, dy = move_map.get(leader_nid, (0.0, 0.0))
        dx = float(dx)
        dy = float(dy)

        if dx < x_min - tol or dx > x_max + tol:
            if strict:
                raise ValueError(f"Move dx={dx} for node {leader_nid} is outside bounds [{x_min}, {x_max}].")
            dx = min(max(dx, x_min), x_max)
        if dy < y_min - tol or dy > y_max + tol:
            if strict:
                raise ValueError(f"Move dy={dy} for node {leader_nid} is outside bounds [{y_min}, {y_max}].")
            dy = min(max(dy, y_min), y_max)

        x0, y0 = group["leader_xy"]
        x_abs = float(x0) + dx
        y_abs = float(y0) + dy

        if x_abs <= tol or y_abs <= tol:
            raise ValueError(
                f"Move on node {leader_nid} crosses a symmetry axis. "
                f"Got x={x_abs}, y={y_abs} after applying dx/dy."
            )

        for member in group["members"]:
            nid = int(member["nid"])
            sx = int(member["sx"])
            sy = int(member["sy"])
            x = float(sx) * x_abs
            y = float(sy) * y_abs
            z = float(z_fun(x, y))
            if not math.isfinite(z):
                raise ValueError(
                    f"Move on node group {leader_nid} generated an invalid point "
                    f"({x}, {y}, {z})."
                )
            new_nodes[nid] = (_round6(x), _round6(y), _round6(z))

    new_G["nodes"] = new_nodes

    if "nodes_internal" in new_G:
        new_G["nodes_internal"] = {
            int(nid): new_nodes[int(nid)]
            for nid in G.get("nodes_internal", {}).keys()
        }
    if "nodes_boundary" in new_G:
        new_G["nodes_boundary"] = {
            int(nid): new_nodes[int(nid)]
            for nid in G.get("nodes_boundary", {}).keys()
        }
    if "meta" in new_G:
        new_G["meta"] = dict(new_G["meta"])
        new_G["meta"]["quarter_mirror_moves_active"] = True

    return new_G


def describe_move_spec(move_spec: Mapping[str, Any]) -> str:
    x_min, x_max = move_spec["x_move_range"]
    y_min, y_max = move_spec["y_move_range"]
    return (
        f"symmetry={move_spec['symmetry_mode']} | "
        f"movable_groups={len(move_spec.get('movable_groups', []))} | "
        f"n_var={int(move_spec.get('n_var', 0))} | "
        f"x_range=[{x_min}, {x_max}] | "
        f"y_range=[{y_min}, {y_max}]"
    )
