
# file: OS_results_updated_member_env.py
import numpy as np
import openseespy.opensees as ops


# -----------------------------------------------------------------------------
# low-level helpers
# -----------------------------------------------------------------------------

def _member_key(a, b):
    a = int(a)
    b = int(b)
    return (a, b) if a < b else (b, a)


def _safe_ele_force(ele_tag):
    """
    Ritorna il vettore delle forze resistenti dell'elemento.

    Per elasticBeamColumn la query documentata e' 'force'; se non disponibile
    proviamo localForce e infine eleForce().
    """
    try:
        v = ops.eleResponse(int(ele_tag), "force")
        if v is None or len(v) == 0:
            raise RuntimeError("empty force")
        return np.array(v, dtype=float)
    except Exception:
        try:
            v = ops.eleResponse(int(ele_tag), "localForce")
            if v is None or len(v) == 0:
                raise RuntimeError("empty localForce")
            return np.array(v, dtype=float)
        except Exception:
            v = ops.eleForce(int(ele_tag))
            if v is None or len(v) == 0:
                raise RuntimeError(f"Cannot get forces for element {ele_tag}")
            return np.array(v, dtype=float)


def _safe_ele_nodes(ele_tag):
    try:
        nds = ops.eleNodes(int(ele_tag))
        if nds is None or len(nds) < 2:
            return None, None
        return int(nds[0]), int(nds[1])
    except Exception:
        return None, None


def _parse_beam3d_end_forces_local(f):
    """
    Parser beam 3D:
    [N1, Vy1, Vz1, T1, My1, Mz1,  N2, Vy2, Vz2, T2, My2, Mz2]
    """
    f = np.asarray(f, dtype=float).flatten()
    if f.size != 12:
        raise ValueError(f"Expected 12 local end forces for 3D beam, got {f.size}")

    endI = {"N": f[0], "Vy": f[1], "Vz": f[2], "T": f[3], "My": f[4], "Mz": f[5]}
    endJ = {"N": f[6], "Vy": f[7], "Vz": f[8], "T": f[9], "My": f[10], "Mz": f[11]}
    return endI, endJ


def _safe_max_abs(vals):
    vals = [abs(float(v)) for v in vals if v is not None]
    return max(vals) if vals else 0.0


def _signed_value_with_max_abs(vals):
    vals = [float(v) for v in vals if v is not None]
    if not vals:
        return 0.0
    return max(vals, key=lambda x: abs(x))


def _compression_and_tension_from_physical_N(N_phys_vals, compression_sign="negative"):
    vals = [float(v) for v in N_phys_vals if v is not None]
    if not vals:
        return 0.0, 0.0, 0.0, 0.0
    sign = str(compression_sign).strip().lower()
    if sign == "negative":
        N_comp = max([0.0] + [-v for v in vals])
        N_tens = max([0.0] + vals)
    elif sign == "positive":
        N_comp = max([0.0] + vals)
        N_tens = max([0.0] + [-v for v in vals])
    else:
        raise ValueError("compression_sign must be 'negative' or 'positive'.")
    return min(vals), max(vals), max(abs(v) for v in vals), N_comp, N_tens


def _build_seg_to_member(ele_data):
    chains = ele_data.get("member_nodechains", {}) or {}
    out = {}
    for mkey, chain in chains.items():
        mk = _member_key(*mkey)
        ch = [int(n) for n in chain]
        for i in range(len(ch) - 1):
            out[_member_key(ch[i], ch[i + 1])] = mk
    return out


def _build_member_lengths(G, ele_data):
    chains = ele_data.get("member_nodechains", {}) or {}
    out = {}
    for mkey, chain in chains.items():
        mk = _member_key(*mkey)
        ch = [int(n) for n in chain]
        L = 0.0
        for i in range(len(ch) - 1):
            p1 = np.array(G["nodes"][ch[i]], dtype=float)
            p2 = np.array(G["nodes"][ch[i + 1]], dtype=float)
            L += float(np.linalg.norm(p2 - p1))
        out[mk] = L
    return out


def _find_oriented_segment_record(member_chain, ele_forces):
    """
    Trova i record del primo e ultimo subelemento lungo la chain del member.
    """
    if len(member_chain) < 2:
        return None, None

    first_pair = (int(member_chain[0]), int(member_chain[1]))
    last_pair = (int(member_chain[-2]), int(member_chain[-1]))

    first_rec = None
    last_rec = None

    for eTag, data in ele_forces.items():
        ni, nj = data.get("nodes", (None, None))
        if (ni, nj) == first_pair:
            first_rec = ("i", eTag, data)
        elif (nj, ni) == first_pair:
            first_rec = ("j", eTag, data)

        if (ni, nj) == last_pair:
            last_rec = ("j", eTag, data)
        elif (nj, ni) == last_pair:
            last_rec = ("i", eTag, data)

    return first_rec, last_rec


# -----------------------------------------------------------------------------
# nodal and element results
# -----------------------------------------------------------------------------

def extract_nodal_results(node_tags=None, ndf=6):
    if node_tags is None:
        node_tags = ops.getNodeTags()
    node_tags = list(map(int, node_tags))

    disp = {}
    for nd in node_tags:
        u = []
        for dof in range(1, int(ndf) + 1):
            u.append(float(ops.nodeDisp(nd, dof)))
        disp[nd] = tuple(u)
    return disp


def max_vertical_displacement(disp_dict, uz_dof=3, uz_ref=None):
    idx = int(uz_dof) - 1

    min_node = None
    max_node = None
    min_uz = +1e20
    max_uz = -1e20
    max_abs_nd_node = None
    max_abs_nd_val = -1.0

    for nd, u in disp_dict.items():
        uz = float(u[idx])
        if uz < min_uz:
            min_uz = uz
            min_node = nd
        if uz > max_uz:
            max_uz = uz
            max_node = nd

        if uz_ref is not None:
            ref = float(uz_ref)
            if ref == 0.0:
                raise ValueError("uz_ref must be non-zero for normalization.")
            uz_nd_abs = abs(uz) / ref
            if uz_nd_abs > max_abs_nd_val:
                max_abs_nd_val = uz_nd_abs
                max_abs_nd_node = nd

    ux_min, uy_min, uz_min = disp_dict[min_node][0:3]
    ux_max, uy_max, uz_max = disp_dict[max_node][0:3]

    min_res = {"node": min_node, "Ux": ux_min, "Uy": uy_min, "Uz": uz_min}
    max_res = {"node": max_node, "Ux": ux_max, "Uy": uy_max, "Uz": uz_max}

    max_abs_nd_res = None
    if uz_ref is not None and max_abs_nd_node is not None:
        ux, uy, uz = disp_dict[max_abs_nd_node][0:3]
        ref = float(uz_ref)
        max_abs_nd_res = {
            "node": max_abs_nd_node,
            "Ux": float(ux),
            "Uy": float(uy),
            "Uz": float(uz),
            "Uz_abs": float(abs(uz)),
            "Uz_nd": float(uz / ref),
            "Uz_nd_abs": float(abs(uz) / ref),
            "ref": ref,
        }

    return min_res, max_res, max_abs_nd_res


def extract_element_forces(ele_data=None, prefer_ele_data=True):
    """
    Estrae le forze di estremita' per ogni subelemento.
    """
    if prefer_ele_data and ele_data is not None and "ele_tags" in ele_data:
        ele_tags = list(map(int, ele_data["ele_tags"]))
    else:
        ele_tags = list(map(int, ops.getEleTags()))

    ele_forces = {}

    for e in ele_tags:
        f_loc = _safe_ele_force(e)
        endI, endJ = _parse_beam3d_end_forces_local(f_loc)
        ni, nj = _safe_ele_nodes(e)

        maxvals = {
            "N_max_abs": max(abs(endI["N"]), abs(endJ["N"])),
            "Vy_max_abs": max(abs(endI["Vy"]), abs(endJ["Vy"])),
            "Vz_max_abs": max(abs(endI["Vz"]), abs(endJ["Vz"])),
            "T_max_abs": max(abs(endI["T"]), abs(endJ["T"])),
            "My_max_abs": max(abs(endI["My"]), abs(endJ["My"])),
            "Mz_max_abs": max(abs(endI["Mz"]), abs(endJ["Mz"])),
            "N_min": min(endI["N"], endJ["N"]),
            "N_max": max(endI["N"], endJ["N"]),
        }

        ele_forces[int(e)] = {
            "i": endI,
            "j": endJ,
            "max": maxvals,
            "nodes": (ni, nj),
        }

    return ele_forces


# -----------------------------------------------------------------------------
# subelement report (as before)
# -----------------------------------------------------------------------------

def build_axial_force_rows(ele_forces, compression_sign="negative"):
    comp_negative = str(compression_sign).strip().lower() != "positive"
    rows = []

    for eTag, data in sorted(ele_forces.items()):
        Ni = float(data["i"]["N"])
        Nj = float(data["j"]["N"])
        ni, nj = data.get("nodes", (None, None))

        if comp_negative:
            Ncomp_i = max(0.0, -Ni)
            Ncomp_j = max(0.0, -Nj)
        else:
            Ncomp_i = max(0.0, Ni)
            Ncomp_j = max(0.0, Nj)

        Myi = float(data["i"]["My"])
        Myj = float(data["j"]["My"])
        Mzi = float(data["i"]["Mz"])
        Mzj = float(data["j"]["Mz"])

        rows.append({
            "element": int(eTag),
            "node_i": ni,
            "node_j": nj,
            "N_i": Ni,
            "N_j": Nj,
            "N_min": min(Ni, Nj),
            "N_max": max(Ni, Nj),
            "N_abs_max": max(abs(Ni), abs(Nj)),
            "N_comp_i": Ncomp_i,
            "N_comp_j": Ncomp_j,
            "N_comp_max": max(Ncomp_i, Ncomp_j),
            "My_i": Myi,
            "My_j": Myj,
            "My_abs_max": max(abs(Myi), abs(Myj)),
            "Mz_i": Mzi,
            "Mz_j": Mzj,
            "Mz_abs_max": max(abs(Mzi), abs(Mzj)),
        })

    return rows


def _sort_rows(rows, sort_by, descending=True):
    return sorted(
        rows,
        key=lambda r: float(r[sort_by]) if r.get(sort_by) is not None else -np.inf,
        reverse=bool(descending),
    )


def print_axial_force_report(
    ele_forces,
    compression_sign="negative",
    sort_by="N_comp_max",
    descending=True,
    top_n=None,
    print_header=True,
):
    rows = build_axial_force_rows(ele_forces, compression_sign=compression_sign)
    rows = _sort_rows(rows, sort_by=sort_by, descending=descending)
    if top_n is not None:
        rows = rows[:int(top_n)]

    if print_header:
        print(
            "# SUBELEMENT FORCE REPORT | "
            f"compression_sign={compression_sign} | sort_by={sort_by} | descending={descending}"
        )
        print(
            f"{'eTag':>6s} {'nodeI':>6s} {'nodeJ':>6s} "
            f"{'N_i':>14s} {'N_j':>14s} {'N_min':>14s} {'N_max':>14s} "
            f"{'|N|_max':>14s} {'N_comp_max':>14s} "
            f"{'My_i':>14s} {'My_j':>14s} {'|My|_max':>14s} "
            f"{'Mz_i':>14s} {'Mz_j':>14s} {'|Mz|_max':>14s}"
        )
        print("-" * 202)

    for r in rows:
        ni = "-" if r["node_i"] is None else str(r["node_i"])
        nj = "-" if r["node_j"] is None else str(r["node_j"])
        print(
            f"{r['element']:6d} {ni:>6s} {nj:>6s} "
            f"{r['N_i']:14.6f} {r['N_j']:14.6f} {r['N_min']:14.6f} {r['N_max']:14.6f} "
            f"{r['N_abs_max']:14.6f} {r['N_comp_max']:14.6f} "
            f"{r['My_i']:14.6f} {r['My_j']:14.6f} {r['My_abs_max']:14.6f} "
            f"{r['Mz_i']:14.6f} {r['Mz_j']:14.6f} {r['Mz_abs_max']:14.6f}"
        )
    return rows


def write_force_report(
    ele_forces,
    filepath="outputs/axial_force_report.txt",
    compression_sign="negative",
    sort_by="N_comp_max",
    descending=True,
    top_n=None,
):
    rows = build_axial_force_rows(ele_forces, compression_sign=compression_sign)
    rows = _sort_rows(rows, sort_by=sort_by, descending=descending)
    if top_n is not None:
        rows = rows[:int(top_n)]

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(
            "# SUBELEMENT FORCE REPORT | "
            f"compression_sign={compression_sign} | sort_by={sort_by} | descending={descending}\n"
        )
        f.write(
            f"{'eTag':>6s} {'nodeI':>6s} {'nodeJ':>6s} "
            f"{'N_i':>14s} {'N_j':>14s} {'N_min':>14s} {'N_max':>14s} "
            f"{'|N|_max':>14s} {'N_comp_max':>14s} "
            f"{'My_i':>14s} {'My_j':>14s} {'|My|_max':>14s} "
            f"{'Mz_i':>14s} {'Mz_j':>14s} {'|Mz|_max':>14s}\n"
        )
        f.write("-" * 202 + "\n")
        for r in rows:
            ni = "-" if r["node_i"] is None else str(r["node_i"])
            nj = "-" if r["node_j"] is None else str(r["node_j"])
            f.write(
                f"{r['element']:6d} {ni:>6s} {nj:>6s} "
                f"{r['N_i']:14.6f} {r['N_j']:14.6f} {r['N_min']:14.6f} {r['N_max']:14.6f} "
                f"{r['N_abs_max']:14.6f} {r['N_comp_max']:14.6f} "
                f"{r['My_i']:14.6f} {r['My_j']:14.6f} {r['My_abs_max']:14.6f} "
                f"{r['Mz_i']:14.6f} {r['Mz_j']:14.6f} {r['Mz_abs_max']:14.6f}\n"
            )
    return rows


# -----------------------------------------------------------------------------
# member envelopes from subelements
# -----------------------------------------------------------------------------

def build_member_force_envelopes(ele_forces, ele_data, compression_sign="negative"):
    """
    Costruisce l'inviluppo delle sollecitazioni sul member originale
    a partire dai subelementi elastici.

    Convenzione per l'assiale:
      N_phys(I) = -N_i
      N_phys(J) =  N_j

    Questo consente di mantenere un segno fisico coerente lungo il member,
    invece di leggere semplicemente N_i/N_j come azioni nodali di estremita'.
    """
    seg_to_member = _build_seg_to_member(ele_data)
    chains = ele_data.get("member_nodechains", {}) or {}
    member_lengths = _build_member_lengths({"nodes": {}}, ele_data) if False else None

    envelopes = {}

    # init with known members
    for mkey, chain in chains.items():
        mk = _member_key(*mkey)
        envelopes[mk] = {
            "member_key": mk,
            "nodes": (int(chain[0]), int(chain[-1])),
            "source_elements": [],
            "_N_phys": [],
            "_My": [],
            "_Mz": [],
            "_Vy": [],
            "_Vz": [],
            "_T": [],
        }

    for eTag, data in sorted(ele_forces.items()):
        ni, nj = data.get("nodes", (None, None))
        if ni is None or nj is None:
            ni, nj = _safe_ele_nodes(eTag)
        if ni is None or nj is None:
            continue

        mk = seg_to_member.get(_member_key(ni, nj))
        if mk is None:
            continue

        env = envelopes.setdefault(
            mk,
            {
                "member_key": mk,
                "nodes": (ni, nj),
                "source_elements": [],
                "_N_phys": [],
                "_My": [],
                "_Mz": [],
                "_Vy": [],
                "_Vz": [],
                "_T": [],
            },
        )
        env["source_elements"].append(int(eTag))

        Ni = float(data["i"]["N"])
        Nj = float(data["j"]["N"])
        env["_N_phys"].extend([-Ni, Nj])

        env["_My"].extend([float(data["i"]["My"]), float(data["j"]["My"])])
        env["_Mz"].extend([float(data["i"]["Mz"]), float(data["j"]["Mz"])])
        env["_Vy"].extend([float(data["i"]["Vy"]), float(data["j"]["Vy"])])
        env["_Vz"].extend([float(data["i"]["Vz"]), float(data["j"]["Vz"])])
        env["_T"].extend([float(data["i"]["T"]), float(data["j"]["T"])])

    # add outer-end values from first and last subelements, when identifiable
    for mkey, chain in chains.items():
        mk = _member_key(*mkey)
        env = envelopes.get(mk, None)
        if env is None:
            continue

        first_rec, last_rec = _find_oriented_segment_record([int(n) for n in chain], ele_forces)
        My_start = My_end = 0.0
        Mz_start = Mz_end = 0.0

        if first_rec is not None:
            which, eTag, data = first_rec
            My_start = float(data[which]["My"])
            Mz_start = float(data[which]["Mz"])
        if last_rec is not None:
            which, eTag, data = last_rec
            My_end = float(data[which]["My"])
            Mz_end = float(data[which]["Mz"])

        N_phys_min, N_phys_max, N_abs_max, N_comp_max, N_tens_max = _compression_and_tension_from_physical_N(
            env["_N_phys"], compression_sign=compression_sign
        )

        env["envelope"] = {
            "N_phys_min": N_phys_min,
            "N_phys_max": N_phys_max,
            "N_abs_max": N_abs_max,
            "N_comp_max": N_comp_max,
            "N_tension_max": N_tens_max,
            "My_abs_max": _safe_max_abs(env["_My"]),
            "Mz_abs_max": _safe_max_abs(env["_Mz"]),
            "Vy_abs_max": _safe_max_abs(env["_Vy"]),
            "Vz_abs_max": _safe_max_abs(env["_Vz"]),
            "T_abs_max": _safe_max_abs(env["_T"]),
            "My_crit_value": _signed_value_with_max_abs(env["_My"]),
            "Mz_crit_value": _signed_value_with_max_abs(env["_Mz"]),
            "Vy_crit_value": _signed_value_with_max_abs(env["_Vy"]),
            "Vz_crit_value": _signed_value_with_max_abs(env["_Vz"]),
            "T_crit_value": _signed_value_with_max_abs(env["_T"]),
            "My_member_end_i": My_start,
            "My_member_end_j": My_end,
            "Mz_member_end_i": Mz_start,
            "Mz_member_end_j": Mz_end,
            "n_subelements": len(env["source_elements"]),
        }

        # cleanup private lists in final structure
        del env["_N_phys"]
        del env["_My"]
        del env["_Mz"]
        del env["_Vy"]
        del env["_Vz"]
        del env["_T"]

    return envelopes


def build_member_force_rows(ele_forces, ele_data, compression_sign="negative"):
    envelopes = build_member_force_envelopes(
        ele_forces=ele_forces,
        ele_data=ele_data,
        compression_sign=compression_sign,
    )
    rows = []

    for mk, env in sorted(envelopes.items()):
        e = env["envelope"]
        ni, nj = env.get("nodes", (None, None))
        rows.append({
            "member": mk,
            "node_i": ni,
            "node_j": nj,
            "n_sub": int(e.get("n_subelements", len(env.get("source_elements", [])))),
            "N_phys_min": float(e["N_phys_min"]),
            "N_phys_max": float(e["N_phys_max"]),
            "N_abs_max": float(e["N_abs_max"]),
            "N_comp_max": float(e["N_comp_max"]),
            "N_tension_max": float(e["N_tension_max"]),
            "My_abs_max": float(e["My_abs_max"]),
            "Mz_abs_max": float(e["Mz_abs_max"]),
            "Vy_abs_max": float(e["Vy_abs_max"]),
            "Vz_abs_max": float(e["Vz_abs_max"]),
            "T_abs_max": float(e["T_abs_max"]),
            "My_end_i": float(e["My_member_end_i"]),
            "My_end_j": float(e["My_member_end_j"]),
            "Mz_end_i": float(e["Mz_member_end_i"]),
            "Mz_end_j": float(e["Mz_member_end_j"]),
            "source_elements": tuple(env.get("source_elements", [])),
        })
    return rows


def print_member_force_report(
    ele_forces,
    ele_data,
    compression_sign="negative",
    sort_by="N_comp_max",
    descending=True,
    top_n=None,
    print_header=True,
):
    rows = build_member_force_rows(
        ele_forces=ele_forces,
        ele_data=ele_data,
        compression_sign=compression_sign,
    )
    rows = _sort_rows(rows, sort_by=sort_by, descending=descending)
    if top_n is not None:
        rows = rows[:int(top_n)]

    if print_header:
        print(
            "# MEMBER ENVELOPE REPORT | "
            f"compression_sign={compression_sign} | sort_by={sort_by} | descending={descending}"
        )
        print(
            f"{'member':>13s} {'nodeI':>6s} {'nodeJ':>6s} {'nSub':>5s} "
            f"{'N_min_phys':>14s} {'N_max_phys':>14s} {'|N|_max':>14s} "
            f"{'N_comp_max':>14s} {'N_tens_max':>14s} "
            f"{'|My|_max':>14s} {'|Mz|_max':>14s} "
            f"{'|Vy|_max':>14s} {'|Vz|_max':>14s} {'|T|_max':>14s}"
        )
        print("-" * 183)

    for r in rows:
        mk = f"({r['member'][0]},{r['member'][1]})"
        ni = "-" if r["node_i"] is None else str(r["node_i"])
        nj = "-" if r["node_j"] is None else str(r["node_j"])
        print(
            f"{mk:>13s} {ni:>6s} {nj:>6s} {r['n_sub']:5d} "
            f"{r['N_phys_min']:14.6f} {r['N_phys_max']:14.6f} {r['N_abs_max']:14.6f} "
            f"{r['N_comp_max']:14.6f} {r['N_tension_max']:14.6f} "
            f"{r['My_abs_max']:14.6f} {r['Mz_abs_max']:14.6f} "
            f"{r['Vy_abs_max']:14.6f} {r['Vz_abs_max']:14.6f} {r['T_abs_max']:14.6f}"
        )
    return rows


def write_member_force_report(
    ele_forces,
    ele_data,
    filepath="outputs/member_force_report.txt",
    compression_sign="negative",
    sort_by="N_comp_max",
    descending=True,
    top_n=None,
):
    rows = build_member_force_rows(
        ele_forces=ele_forces,
        ele_data=ele_data,
        compression_sign=compression_sign,
    )
    rows = _sort_rows(rows, sort_by=sort_by, descending=descending)
    if top_n is not None:
        rows = rows[:int(top_n)]

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(
            "# MEMBER ENVELOPE REPORT | "
            f"compression_sign={compression_sign} | sort_by={sort_by} | descending={descending}\n"
        )
        f.write(
            f"{'member':>13s} {'nodeI':>6s} {'nodeJ':>6s} {'nSub':>5s} "
            f"{'N_min_phys':>14s} {'N_max_phys':>14s} {'|N|_max':>14s} "
            f"{'N_comp_max':>14s} {'N_tens_max':>14s} "
            f"{'|My|_max':>14s} {'|Mz|_max':>14s} "
            f"{'|Vy|_max':>14s} {'|Vz|_max':>14s} {'|T|_max':>14s}\n"
        )
        f.write("-" * 183 + "\n")
        for r in rows:
            mk = f"({r['member'][0]},{r['member'][1]})"
            ni = "-" if r["node_i"] is None else str(r["node_i"])
            nj = "-" if r["node_j"] is None else str(r["node_j"])
            f.write(
                f"{mk:>13s} {ni:>6s} {nj:>6s} {r['n_sub']:5d} "
                f"{r['N_phys_min']:14.6f} {r['N_phys_max']:14.6f} {r['N_abs_max']:14.6f} "
                f"{r['N_comp_max']:14.6f} {r['N_tension_max']:14.6f} "
                f"{r['My_abs_max']:14.6f} {r['Mz_abs_max']:14.6f} "
                f"{r['Vy_abs_max']:14.6f} {r['Vz_abs_max']:14.6f} {r['T_abs_max']:14.6f}\n"
            )
    return rows


# -----------------------------------------------------------------------------
# reactions and high-level result bundle
# -----------------------------------------------------------------------------

def extract_reactions(node_tags=None, ndf=6):
    ops.reactions()

    if node_tags is None:
        node_tags = ops.getNodeTags()
    node_tags = list(map(int, node_tags))

    reac = {}
    for nd in node_tags:
        r = []
        for dof in range(1, int(ndf) + 1):
            try:
                r.append(float(ops.nodeReaction(nd, dof)))
            except Exception:
                r.append(0.0)

        if any(abs(x) > 0.0 for x in r):
            reac[nd] = tuple(r)

    return reac


def sum_reactions(reactions_dict, dof=3):
    idx = int(dof) - 1
    return sum(float(r[idx]) for r in reactions_dict.values())


def extract_results(
    G,
    M,
    ele_data=None,
    ndf=6,
    uz_dof=3,
    include_reactions=True,
    prefer_ele_data=True,
    uz_ref=None,
    build_member_envelopes=True,
    compression_sign="negative",
):
    node_tags = list(map(int, G["nodes"].keys()))

    disp = extract_nodal_results(node_tags=node_tags, ndf=ndf)
    min_res, max_res, max_abs_nd = max_vertical_displacement(
        disp,
        uz_dof=uz_dof,
        uz_ref=uz_ref,
    )

    ele_forces = extract_element_forces(
        ele_data=ele_data,
        prefer_ele_data=prefer_ele_data,
    )

    out = {
        "disp": disp,
        "max_vertical": {
            "min": min_res,
            "max": max_res,
            "max_abs_nd": max_abs_nd,
        },
        "ele_forces": ele_forces,
    }

    if build_member_envelopes and ele_data is not None and "member_nodechains" in ele_data:
        out["member_forces"] = build_member_force_envelopes(
            ele_forces=ele_forces,
            ele_data=ele_data,
            compression_sign=compression_sign,
        )

    if include_reactions:
        reac = extract_reactions(node_tags=node_tags, ndf=ndf)
        Rz_tot = sum_reactions(reac, dof=uz_dof)
        out["reactions"] = reac
        out["reactions_sum"] = {"Rz": Rz_tot}

    return out


def print_results_summary(res, uz_dof=3, print_summary=True):
    if not print_summary:
        return
    mn = res["max_vertical"]["min"]
    mx = res["max_vertical"]["max"]
    mabs = res["max_vertical"].get("max_abs_nd", None)

    print(f"# SUMMARY | MAX DOWNWARD Uz(dof={uz_dof}): node={mn['node']} Uz={mn['Uz']}")
    print(f"# SUMMARY | MAX UPWARD  Uz(dof={uz_dof}): node={mx['node']} Uz={mx['Uz']}")

    if mabs is not None:
        print(
            f"# SUMMARY | ABS  Uz(dof={uz_dof}): node={mabs['node']} "
            f"Uz={mabs['Uz']} | Uz_nd_abs={mabs['Uz_nd_abs']}"
        )
    else:
        print(f"# SUMMARY | ABS  Uz(dof={uz_dof}): (not computed) pass uz_ref=... to extract_results()")

    if "reactions_sum" in res:
        print(f"# SUMMARY | Sum reactions Rz = {res['reactions_sum']['Rz']}")
        if "reactions" in res:
            print(f"# SUMMARY | n_reaction_nodes = {len(res['reactions'])}")

    if "member_forces" in res:
        print(f"# SUMMARY | member force envelopes available = {len(res['member_forces'])}")
