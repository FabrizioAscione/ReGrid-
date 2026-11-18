# =========================
# file: FIBER_ELEMENTO.py
# =========================
import math
import openseespy.opensees as ops
from fiber_section_shapes import build_fiber_patch_specs, plot_unique_fiber_sections, require_supported_fiber_family


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


def _pick_vecxz_for_element(p1, p2):
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
    fb_ele lavora con dict: {"type","E","G","fyk",...}
    Se arriva tuple/list utente la converte.
    """
    if isinstance(material, dict):
        out = dict(material)
        if "type" in out:
            out["type"] = str(out["type"]).strip()
        return out

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
            
        if mtype.lower() == "steel02":
            if len(material) < 5:
                raise ValueError("steel02 requires ('steel02', fyk, E, G, b)")
            out["b"] = float(material[4])

        if mtype.lower() in ("ppmat", "elasticpp"):
            if len(material) < 5:
                raise ValueError("PPmat requires ('PPmat', fyk, E, G, epsyP)")
            out["epsyP"] = float(material[4])
            out["epsyN"] = float(material[4])
            out["eps0"] = 0.0

        if mtype.lower() == "hardening":
            if len(material) < 5:
                raise ValueError("Hardening requires ('Hardening', E, fyk, H_iso, H_kin)")
            # For backward compatibility with the rest of the codebase we remap to the dict keys used elsewhere.
            out = {"type": mtype, "E": float(material[1]), "fyk": float(material[2]), "H_iso": float(material[3]), "H_kin": float(material[4]), "G": 0.0}

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
    """Return first non-None section property among aliases."""
    for name in names:
        if name in sec and sec[name] is not None:
            return sec[name]
    return default


def _section_family(sec):
    fam = _sec_get(sec, "section_family", "family", "type", default=None)
    if fam is None:
        return None
    return str(fam).strip().upper()

def _append_patch_commands(py_lines, matTag, patch_specs):
    for patch in patch_specs:
        if patch["kind"] == "circ":
            ops.patch(
                "circ",
                matTag,
                int(patch["n_circ"]),
                int(patch["n_rad"]),
                float(patch["yC"]),
                float(patch["zC"]),
                float(patch["rI"]),
                float(patch["rO"]),
                float(patch["a0"]),
                float(patch["a1"]),
            )
            py_lines.append(
                f"ops.patch('circ', {matTag}, {int(patch['n_circ'])}, {int(patch['n_rad'])}, "
                f"{float(patch['yC'])}, {float(patch['zC'])}, {float(patch['rI'])}, {float(patch['rO'])}, "
                f"{float(patch['a0'])}, {float(patch['a1'])})"
            )
        elif patch["kind"] == "rect":
            ops.patch(
                "rect",
                matTag,
                int(patch["nfy"]),
                int(patch["nfz"]),
                float(patch["y1"]),
                float(patch["z1"]),
                float(patch["y2"]),
                float(patch["z2"]),
            )
            py_lines.append(
                f"ops.patch('rect', {matTag}, {int(patch['nfy'])}, {int(patch['nfz'])}, "
                f"{float(patch['y1'])}, {float(patch['z1'])}, {float(patch['y2'])}, {float(patch['z2'])})"
            )
        elif patch["kind"] == "quad":
            p1, p2, p3, p4 = patch["points"]
            ops.patch(
                "quad",
                matTag,
                int(patch["nfy"]),
                int(patch["nfz"]),
                float(p1[0]), float(p1[1]),
                float(p2[0]), float(p2[1]),
                float(p3[0]), float(p3[1]),
                float(p4[0]), float(p4[1]),
            )
            py_lines.append(
                f"ops.patch('quad', {matTag}, {int(patch['nfy'])}, {int(patch['nfz'])}, "
                f"{float(p1[0])}, {float(p1[1])}, {float(p2[0])}, {float(p2[1])}, "
                f"{float(p3[0])}, {float(p3[1])}, {float(p4[0])}, {float(p4[1])})"
            )
        else:
            raise NotImplementedError(f"Unsupported patch kind '{patch['kind']}'.")
def fb_L_ele(
    G,
    M,
    material,
    section=None,
    section_by_member=None,
    transf_type="PDelta",
    subdivide=1,
    first_new_node_tag=None,
    update_G=False,

    n_integration_pts=5,
    fiber_opts=None,
    first_mat_tag=1,
    first_integ_tag=1,

    first_ele_tag=1,
    first_sec_tag=1,
    first_tr_tag=1,

    plot_fiber_sections=False,
    plot_fiber_sections_dir=None,
    plot_fiber_sections_show=False,

    print_ops=True,
    print_mode="headtail",
    n_preview=10,
    print_summary=True,
):
    nodes_in = G.get("nodes", {})
    members = M.get("members", [])
    if not nodes_in:
        raise ValueError("G['nodes'] is empty.")
    if not members:
        raise ValueError("M['members'] is empty.")

    material = _parse_material(material)
    n_integration_pts = int(n_integration_pts)
    if n_integration_pts < 2:
        raise ValueError("Lobatto integration requires at least 2 integration points.")

    if section_by_member is None:
        if section is None:
            raise ValueError("Provide either section (global) or section_by_member.")
        section_by_member = _make_section_by_member(M, section)

    nodes = dict(nodes_in)

    if fiber_opts is None:
        fiber_opts = {}
    n_circ = int(fiber_opts.get("n_circ", 32))
    n_rad  = int(fiber_opts.get("n_rad", 4))
    yC     = float(fiber_opts.get("yC", 0.0))
    zC     = float(fiber_opts.get("zC", 0.0))
    a0     = float(fiber_opts.get("a0_deg", 0.0))
    a1     = float(fiber_opts.get("a1_deg", 360.0))
    use_gj = bool(fiber_opts.get("use_GJ", True))

    mtype = str(material.get("type", "")).strip().lower()
    matTag = int(first_mat_tag)

    py_lines = []

    def _mk_uniaxial_material():
        if mtype == "elastic":
            E = float(material["E"])
            ops.uniaxialMaterial("Elastic", matTag, E)
            return [f"ops.uniaxialMaterial('Elastic', {matTag}, {E})"]

        if mtype == "steel01":
            Fy = float(material["fyk"])
            E0 = float(material["E"])
            b  = float(material.get("b", 0.0))
            ops.uniaxialMaterial("Steel01", matTag, Fy, E0, b)
            return [f"ops.uniaxialMaterial('Steel01', {matTag}, {Fy}, {E0}, {b})"]

        if mtype == "steel02":
            Fy = float(material["fyk"])
            E0 = float(material["E"])
            b  = float(material.get("b", 0.0))
            R0 = float(material.get("R0", 20.0))
            cR1 = float(material.get("cR1", 0.925))
            cR2 = float(material.get("cR2", 0.15))
            ops.uniaxialMaterial("Steel02", matTag, Fy, E0, b, R0, cR1, cR2)
            return [f"ops.uniaxialMaterial('Steel02', {matTag}, {Fy}, {E0}, {b}, {R0}, {cR1}, {cR2})"]

        if mtype in ("ppmat", "elasticpp"):
            E = float(material["E"])
            epsyP = float(material["epsyP"])
            epsyN = float(material.get("epsyN", epsyP))
            eps0  = float(material.get("eps0", 0.0))
            ops.uniaxialMaterial("ElasticPP", matTag, E, epsyP, epsyN, eps0)
            return [f"ops.uniaxialMaterial('ElasticPP', {matTag}, {E}, {epsyP}, {epsyN}, {eps0})"]

        if mtype == "hardening":
            Fy = float(material["fyk"])
            E0 = float(material["E"])
            H_iso = float(material.get("H_iso", 0.0))
            H_kin = float(material.get("H_kin", 0.0))
            ops.uniaxialMaterial("Hardening", matTag, E0, Fy, H_iso, H_kin)
            return [f"ops.uniaxialMaterial('Hardening', {matTag}, {E0}, {Fy}, {H_iso}, {H_kin})"]

        raise ValueError(f"Unsupported fiber material type='{mtype}'.")

    py_lines.extend(_mk_uniaxial_material())

    eid = int(first_ele_tag)
    sid = int(first_sec_tag)
    tid = int(first_tr_tag)
    itg = int(first_integ_tag)

    max_existing = max(int(k) for k in nodes.keys())
    next_new_node = int(first_new_node_tag) if first_new_node_tag is not None else (max_existing + 1)

    ele_tags, sec_tags, tr_tags, integ_tags = [], [], [], []
    new_node_tags = []
    member_nodechains = {}

    # NEW: per-member list of sub-elements
    member_subele_tags = {}

    # 1) subdivision nodes
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

    if bool(plot_fiber_sections):
        try:
            sec_map_for_plot = section_by_member if section_by_member is not None else _make_section_by_member(M, section)
            saved_plots = plot_unique_fiber_sections(
                sec_map_for_plot,
                fiber_opts=fiber_opts,
                out_dir=plot_fiber_sections_dir,
                show=plot_fiber_sections_show,
            )
            if print_summary:
                print(f"# SUMMARY | fiber section plots saved: {len(saved_plots)}")
        except Exception as exc:
            print(f"# WARNING | could not generate fiber section plots: {exc}")

    # 2) build sections + transf + integration + elements per sub-element
    for (na0, nb0) in members:
        na0 = int(na0); nb0 = int(nb0)
        key = _member_key(na0, nb0)
        if key not in section_by_member:
            raise KeyError(f"Missing section for member {key} in section_by_member.")
        sec = section_by_member[key]
        require_supported_fiber_family(sec, key, context="fb_L_ele")
        patch_specs = build_fiber_patch_specs(sec, fiber_opts)

        Gs = float(material.get("G", 0.0))
        J  = _sec_get(sec, "J", "It")
        GJ = None
        if use_gj and J is not None:
            GJ = Gs * float(J)

        chain = member_nodechains[key]
        member_subele_tags[key] = []

        for i in range(len(chain) - 1):
            na = int(chain[i])
            nb = int(chain[i + 1])
            p1 = nodes[na]
            p2 = nodes[nb]

            vx, vy, vz = _pick_vecxz_for_element(p1, p2)
            ops.geomTransf(transf_type, tid, vx, vy, vz)
            py_lines.append(f"ops.geomTransf('{transf_type}', {tid}, {vx}, {vy}, {vz})")
            tr_tags.append(tid)

            if GJ is not None:
                ops.section("Fiber", sid, "-GJ", float(GJ))
                py_lines.append(f"ops.section('Fiber', {sid}, '-GJ', {float(GJ)})")
            else:
                ops.section("Fiber", sid)
                py_lines.append(f"ops.section('Fiber', {sid})")

            _append_patch_commands(py_lines, matTag, patch_specs)
            sec_tags.append(sid)

            ops.beamIntegration("Lobatto", itg, sid, n_integration_pts)
            py_lines.append(f"ops.beamIntegration('Lobatto', {itg}, {sid}, {n_integration_pts})")
            integ_tags.append(itg)

            ops.element("forceBeamColumn", eid, na, nb, tid, itg)
            py_lines.append(f"ops.element('forceBeamColumn', {eid}, {na}, {nb}, {tid}, {itg})")
            ele_tags.append(eid)
            member_subele_tags[key].append(eid)

            eid += 1
            sid += 1
            tid += 1
            itg += 1

    py_text = "\n".join(py_lines)

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

    if print_summary:
        print(
            f"# SUMMARY | new nodes: {len(new_node_tags)} | mats: 1 | "
            f"sections: {len(sec_tags)} | beamInt: {len(integ_tags)} | "
            f"elements: {len(ele_tags)} | transf: {len(tr_tags)}"
        )

    # NEW: local instability mapping ready-to-use
    local_instability = {}
    for (na0, nb0) in members:
        key = _member_key(na0, nb0)
        chain = member_nodechains[key]
        mid_node = chain[len(chain)//2]
        local_instability[key] = {
            "i": int(chain[0]),
            "j": int(chain[-1]),
            "m": int(mid_node),
            "subEles": [int(e) for e in member_subele_tags.get(key, [])],
        }

    out = {
        "ele_tags": ele_tags,
        "sec_tags": sec_tags,
        "tr_tags": tr_tags,
        "integ_tags": integ_tags,
        "new_node_tags": new_node_tags,
        "member_nodechains": member_nodechains,
        "member_subele_tags": member_subele_tags,   # NEW
        "local_instability": local_instability,     # NEW
        "meta": {
            "elem_type": "forceBeamColumn",
            "sec_type": "fiber",
            "integration": "Lobatto",
            "material_type": mtype,
            "n_ele": len(ele_tags),
            "n_new_nodes": len(new_node_tags),
            "n_ip": int(n_integration_pts),
            "fiber": {"n_circ": n_circ, "n_rad": n_rad, "use_GJ": bool(use_gj)},
            "first_ele_tag": int(first_ele_tag),
            "first_sec_tag": int(first_sec_tag),
            "first_tr_tag": int(first_tr_tag),
            "first_integ_tag": int(first_integ_tag),
            "first_new_node_tag": int(first_new_node_tag) if first_new_node_tag is not None else (max_existing + 1),
            "transf_type": transf_type,
            "subdivide": subdivide,
            "update_G": bool(update_G),
        },
    }
    return py_text, out
