# =========================
# file: ELASTICO_ELEMENTO.py
# =========================
import math
import openseespy.opensees as ops

from fiber_section_shapes import get_section_rotation_deg


def _norm(v):
    x, y, z = v
    n = math.sqrt(x * x + y * y + z * z)
    if n < 1e-14:
        return None
    return (x / n, y / n, z / n)


def _cross(a, b):
    ax, ay, az = a
    bx, by, bz = b
    return (ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx)


def _rotate_section_vecxz(p1, p2, sec, rotation_opts=None):
    base_vecxz = _pick_vecxz_for_element(p1, p2)
    angle_deg = float(get_section_rotation_deg(sec, rotation_opts))
    if abs(angle_deg) <= 1e-12:
        return base_vecxz

    x_axis = _norm((p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]))
    if x_axis is None:
        return base_vecxz

    y_axis = _norm(_cross(base_vecxz, x_axis))
    if y_axis is None:
        return base_vecxz

    z_axis = _norm(_cross(x_axis, y_axis))
    if z_axis is None:
        return base_vecxz

    ang = math.radians(angle_deg)
    c = math.cos(ang)
    s = math.sin(ang)
    z_rot = (
        (c * z_axis[0]) - (s * y_axis[0]),
        (c * z_axis[1]) - (s * y_axis[1]),
        (c * z_axis[2]) - (s * y_axis[2]),
    )
    z_rot = _norm(z_rot)
    return z_rot if z_rot is not None else base_vecxz


def _pick_vecxz_for_element(p1, p2):
    """
    Sceglie un "vecxz" robusto per geomTransf:
    prova Z globale, poi Y, poi X; prende il primo non collineare all'asse locale.
    """
    x_axis = _norm((p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]))
    if x_axis is None:
        return (0.0, 0.0, 1.0)

    candidates = [(0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0)]
    for c in candidates:
        cr = _cross(x_axis, c)
        if _norm(cr) is not None:
            return c
    return (0.0, 0.0, 1.0)


def _member_key(a, b):
    a = int(a); b = int(b)
    return (a, b) if a < b else (b, a)


def _get_subdivisions(subdivide, na, nb):
    if subdivide is None:
        return 1
    if isinstance(subdivide, int):
        return max(1, int(subdivide))
    if isinstance(subdivide, dict):
        return max(1, int(subdivide.get(_member_key(na, nb), 1)))
    raise TypeError("subdivide must be None, int, or dict with keys (minNode,maxNode)->n_sub")


def _parse_material(material):
    """
    el_ele lavora con dict: {"E":..., "G":...} (oltre ad altre chiavi).
    Se arriva tuple/list (utente) la converte.
    """
    if isinstance(material, dict):
        return material

    if isinstance(material, (tuple, list)):
        if len(material) < 4:
            raise ValueError("material tuple must be ('elastic', fyk, E, G) at least.")
        mtype = str(material[0]).strip()
        fyk = float(material[1])
        E = float(material[2])
        G = float(material[3])
        out = {"type": mtype, "fyk": fyk, "E": E, "G": G}
        if mtype.lower() == "steel01":
            if len(material) < 5:
                raise ValueError("steel01 requires ('steel01', fyk, E, G, b)")
            out["b"] = float(material[4])
        if mtype.lower() in ("ppmat", "elasticpp"):
            if len(material) < 5:
                raise ValueError("PPmat requires ('PPmat', fyk, E, G, epsyP)")
            out["epsyP"] = float(material[4])
            out["epsyN"] = float(material[4])
            out["eps0"] = 0.0
        return out

    raise TypeError("material must be dict or tuple/list.")


def _make_section_by_member(M, section):
    members = M.get("members", [])
    if not members:
        raise ValueError("M['members'] is empty.")
    sec_map = {}
    for (na, nb) in members:
        sec_map[_member_key(na, nb)] = section
    return sec_map




def _sec_get(sec, *names, default=None):
    for name in names:
        if name in sec and sec[name] is not None:
            return sec[name]
    return default
def el_ele(
    G,
    M,
    material,                 # dict o tuple -> viene normalizzato
    section=None,             # dict globale (fallback)
    section_by_member=None,   # dict {(na,nb): sec_dict} (opzionale)
    section_rotation_opts=None,
    transf_type="Linear",
    first_ele_tag=1,
    first_sec_tag=1,
    first_tr_tag=1,
    subdivide=1,
    first_new_node_tag=None,
    update_G=False,
    print_ops=True,
    print_mode="headtail",
    n_preview=10,
    print_summary=True,
):
    """
    Crea:
      - (opzionale) nuovi nodi interni se subdivide>1
      - 1 geomTransf per OGNI sub-elemento
      - 1 section('Elastic') per OGNI sub-elemento
      - 1 elasticBeamColumn per OGNI sub-elemento

    Sezioni:
      - puoi passare section_by_member (multi-sezione)
      - oppure section globale (1-sezione) e la funzione crea internamente la map
    """

    nodes_in = G.get("nodes", {})
    members = M.get("members", [])
    if not nodes_in:
        raise ValueError("G['nodes'] is empty.")
    if not members:
        raise ValueError("M['members'] is empty.")

    # normalizzo materiale (FIX: accetta tuple)
    material = _parse_material(material)

    # preparo sezione per-member
    if section_by_member is None:
        if section is None:
            raise ValueError("Provide either section (global) or section_by_member.")
        section_by_member = _make_section_by_member(M, section)

    nodes = dict(nodes_in)

    # Tag iniziali
    eid = int(first_ele_tag)
    sid = int(first_sec_tag)
    tid = int(first_tr_tag)

    # Tag nuovi nodi
    max_existing = max(int(k) for k in nodes.keys())
    next_new_node = int(first_new_node_tag) if first_new_node_tag is not None else (max_existing + 1)

    py_lines = []
    ele_tags, sec_tags, tr_tags = [], [], []
    new_node_tags = []
    member_nodechains = {}

    # ----------------------------
    # 1) subdivision: nuovi nodi interni
    # ----------------------------
    for (na0, nb0) in members:
        na0 = int(na0); nb0 = int(nb0)
        if na0 not in nodes or nb0 not in nodes:
            raise KeyError(f"Member ({na0},{nb0}) refers to missing node(s).")

        n_sub = _get_subdivisions(subdivide, na0, nb0)
        if n_sub <= 1:
            member_nodechains[_member_key(na0, nb0)] = [na0, nb0]
            continue

        x1, y1, z1 = nodes[na0]
        x2, y2, z2 = nodes[nb0]

        chain = [na0]
        for k in range(1, n_sub):
            tpar = k / float(n_sub)
            xn = (1 - tpar) * x1 + tpar * x2
            yn = (1 - tpar) * y1 + tpar * y2
            zn = (1 - tpar) * z1 + tpar * z2

            tag = next_new_node
            next_new_node += 1

            nodes[tag] = (float(xn), float(yn), float(zn))
            new_node_tags.append(tag)

            ops.node(int(tag), float(xn), float(yn), float(zn))
            py_lines.append(f"ops.node({int(tag)}, {float(xn)}, {float(yn)}, {float(zn)})")

            chain.append(tag)

        chain.append(nb0)
        member_nodechains[_member_key(na0, nb0)] = chain

    if update_G and new_node_tags:
        G["nodes"] = nodes

    # ----------------------------
    # 2) sezioni+transf+elementi per ogni sub-elemento
    # ----------------------------
    for (na0, nb0) in members:
        na0 = int(na0); nb0 = int(nb0)
        key = _member_key(na0, nb0)
        if key not in section_by_member:
            raise KeyError(f"Missing section for member {key} in section_by_member.")
        sec = section_by_member[key]

        # proprietà per quel member
        E  = float(material["E"])
        Gs = float(material["G"])
        A  = float(_sec_get(sec, "A"))
        Iy = float(_sec_get(sec, "Iy"))
        Iz = float(_sec_get(sec, "Iz"))
        Jv = _sec_get(sec, "J", "It")
        J  = float(Jv) if Jv is not None else 0.0

        chain = member_nodechains[_member_key(na0, nb0)]
        for i in range(len(chain) - 1):
            na = int(chain[i])
            nb = int(chain[i + 1])

            p1 = nodes[na]
            p2 = nodes[nb]

            # geomTransf per sub-elemento
            vx, vy, vz = _rotate_section_vecxz(p1, p2, sec, section_rotation_opts)
            ops.geomTransf(transf_type, tid, vx, vy, vz)
            py_lines.append(f"ops.geomTransf('{transf_type}', {tid}, {vx}, {vy}, {vz})")
            tr_tags.append(tid)

            # sezione elastica per sub-elemento
            ops.section("Elastic", sid, E, A, Iz, Iy, Gs, J)
            py_lines.append(f"ops.section('Elastic', {sid}, {E}, {A}, {Iz}, {Iy}, {Gs}, {J})")
            sec_tags.append(sid)

            # elemento elasticBeamColumn
            ops.element("elasticBeamColumn", eid, na, nb, sid, tid)
            py_lines.append(f"ops.element('elasticBeamColumn', {eid}, {na}, {nb}, {sid}, {tid})")
            ele_tags.append(eid)

            eid += 1
            sid += 1
            tid += 1

    py_text = "\n".join(py_lines)

    # ---- stampa righe OpenSees ----
    if print_ops:
        if str(print_mode).lower() == "all":
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

    # ---- summary opzionale ----
    if print_summary:
        print(
            f"# SUMMARY | "
            f"new nodes generated: {len(new_node_tags)} | "
            f"sections generated: {len(sec_tags)} | "
            f"elements generated: {len(ele_tags)} | "
            f"transfTags generated: {len(tr_tags)}"
        )

    out = {
        "ele_tags": ele_tags,
        "sec_tags": sec_tags,
        "tr_tags": tr_tags,
        "new_node_tags": new_node_tags,
        "member_nodechains": member_nodechains,
        "meta": {
            "elem_type": "elasticBeamColumn",
            "sec_type": "elastic",
            "n_ele": len(ele_tags),
            "n_new_nodes": len(new_node_tags),
            "first_ele_tag": int(first_ele_tag),
            "first_sec_tag": int(first_sec_tag),
            "first_tr_tag": int(first_tr_tag),
            "first_new_node_tag": int(first_new_node_tag) if first_new_node_tag is not None else (max_existing + 1),
            "transf_type": transf_type,
            "section_rotation_opts": section_rotation_opts,
            "subdivide": subdivide,
            "update_G": bool(update_G),
        },
    }

    return py_text, out
