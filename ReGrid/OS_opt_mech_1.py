# =========================
# file: opt_mech_updated.py
# =========================
import numpy as np
import openseespy.opensees as ops

from OS_results import extract_results
from OS_LB_fast import get_tangent_stiffness, linear_buckling_sparse
from Structural_check import compute_DCR_elements
from OS_self_weigth import apply_selfweight


def _member_key(a, b):
    a = int(a)
    b = int(b)
    return (a, b) if a < b else (b, a)


def make_section_by_member(M, sec):
    """
    Crea mapping automatico {(na,nb)->sec_dict} per ogni member in M['members'].
    """
    members = M.get("members", [])
    if not members:
        raise ValueError("M['members'] is empty.")
    return {_member_key(na, nb): sec for (na, nb) in members}


def _parse_material_for_elastic(material):
    """
    Accetta:
      - dict da define_material
      - tuple/list utente:
          ('elastic', fyk, E, G)
          ('steel01', fyk, E, G, b)
          ('PPmat',  fyk, E, G, epsyP)
    Ritorna dict coerente per el_ele.
    """
    if isinstance(material, dict):
        out = dict(material)
        if "type" in out:
            out["type"] = str(out["type"]).strip()
        return out

    if isinstance(material, (tuple, list)):
        if len(material) < 4:
            raise ValueError("material tuple must be like ('elastic', fyk, E, G) at least.")

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


def _run_static_ok(run_static):
    """
    Normalizza il risultato del wrapper run_static.

    Accetta:
      - bool
      - int (OpenSees: 0 = ok)
      - tuple/list, es. (py_text, ok)
    """
    out = run_static()

    if isinstance(out, bool):
        return out

    if isinstance(out, (int, np.integer)):
        return int(out) == 0

    if isinstance(out, (tuple, list)):
        if len(out) == 0:
            return False
        last = out[-1]
        if isinstance(last, bool):
            return last
        if isinstance(last, (int, np.integer)):
            return int(last) == 0
        return bool(last)

    return bool(out)


def build_and_analyze_global_section(
    G, M,
    ndm, ndf,
    c1, c2, c3, c4, c5,
    q1,
    sec,
    mat,
    transf_type="PDelta",
    subdivide=1,
    first_new_node_tag=2000,
    update_G=True,
    init_model=None,
    define_nodes=None,
    apply_costr=None,
    el_ele=None,
    apply_loads=None,
    run_static=None,
    uz_limit=0.10,
    gamma_M=1.0,
    require_dcr=True,
    print_iter_summary=False,
    print_model_ops=False,
    print_mode="headtail",
    n_preview=5,
    print_load_ops=False,
    print_load_mode="headtail",
    print_buckling_summary=False,
    debug_kgeo=False,
):
    """
    Ricostruisce modello OpenSees, crea elementi, calcola K0,
    applica carichi, esegue statica elastica e buckling lineare.
    """
    if init_model is None or define_nodes is None or apply_costr is None:
        raise ValueError("Missing init_model/define_nodes/apply_costr")
    if el_ele is None or apply_loads is None or run_static is None:
        raise ValueError("Missing el_ele/apply_loads/run_static")

    matd = _parse_material_for_elastic(mat)
    fyk = float(matd.get("fyk", 0.0))

    init_model(ndm, ndf)
    define_nodes(G, ndf, set_mass=False)
    apply_costr(G, ndf, c1, c2, c3, c4, c5)

    _, ele_data = el_ele(
        G, M,
        material=matd,
        section=sec,
        transf_type=transf_type,
        subdivide=int(subdivide),
        first_new_node_tag=int(first_new_node_tag),
        update_G=bool(update_G),
        print_ops=bool(print_model_ops),
        print_mode=str(print_mode),
        n_preview=int(n_preview),
        print_summary=False,
    )

    loads_sw = apply_selfweight(
        G,
        ele_data,
        mode="eleload",
        section=sec,
        pattern_tag=1,
        ts_tag=1,
    )

    tangent_kwargs = dict(system="FullGeneral", numberer="RCM", constraints="Plain", algorithm="Newton")
    K0, _ = get_tangent_stiffness(**tangent_kwargs, print_ops=False)

    loads, areas, _ = apply_loads(
        G,
        M=M,
        q=q1,
        ndf=ndf,
        direction="-Z",
        ts_tag=2,
        pattern_tag=2,
        print_ops=bool(print_load_ops),
        print_mode=str(print_load_mode),
        n_preview=int(n_preview),
        print_summary=False,
        plot_loads=False,
    )

    ok_static = _run_static_ok(run_static)
    if not ok_static:
        return {
            "ok": False,
            "reason": "static_not_converged",
            "section": sec.get("name", "?"),
            "sec": sec,
            "sec_assignment": {"mode": "global", "sec": sec},
            "sec_by_member": make_section_by_member(M, sec),
            "Uz_min": None,
            "DCR_max": None,
            "LFB": None,
        }

    res = extract_results(G, M, ele_data=ele_data, ndf=ndf, uz_dof=3)
    uz_min = float(res["max_vertical"]["min"]["Uz"])
    uz_ok = abs(uz_min) <= float(uz_limit)

    Wy = sec.get("Wy", None)
    Wz = sec.get("Wz", None) or Wy

    _, dcr_max = compute_DCR_elements(
        res["ele_forces"],
        fy=fyk,
        A=float(sec["A"]),
        Iy=float(sec["Iy"]),
        Iz=float(sec["Iz"]),
        Wy=Wy,
        Wz=Wz,
        gamma_M=float(gamma_M),
    )
    dcr_ok = float(dcr_max) <= 1.0

    if debug_kgeo:
        Ktot, _ = get_tangent_stiffness(**tangent_kwargs, print_ops=False)
        Kgeo = np.array(K0, dtype=float) - np.array(Ktot, dtype=float)

        print("# DEBUG norms:")
        print("||K0||   =", np.linalg.norm(np.array(K0, dtype=float)))
        print("||Ktot|| =", np.linalg.norm(np.array(Ktot, dtype=float)))
        print("||Kgeo|| =", np.linalg.norm(np.array(Kgeo, dtype=float)))
        print("ratio ||Kgeo||/||K0|| =", np.linalg.norm(Kgeo) / max(np.linalg.norm(K0), 1e-16))

        Fxq = sum(F[0] for F in loads.values())
        Fyq = sum(F[1] for F in loads.values())
        Fzq = sum(F[2] for F in loads.values())
        print(f"Q resultant = Fx={Fxq:.6f} | Fy={Fyq:.6f} | Fz={Fzq:.6f}")

        try:
            if isinstance(loads_sw, dict):
                Fxg = sum(F[0] for F in loads_sw.values())
                Fyg = sum(F[1] for F in loads_sw.values())
                Fzg = sum(F[2] for F in loads_sw.values())
                print(f"G resultant = Fx={Fxg:.6f} | Fy={Fyg:.6f} | Fz={Fzg:.6f}")
        except Exception:
            pass

        try:
            ops.reactions()
            rz = 0.0
            for nid in ops.getNodeTags():
                try:
                    r = ops.nodeReaction(nid)
                    if len(r) >= 3:
                        rz += float(r[2])
                except Exception:
                    pass
            print(f"Sum reactions Rz = {rz:.6f}")
        except Exception:
            pass

    lam, vecs = linear_buckling_sparse(
        K0,
        n_modes=6,
        drop_tol=1e-12,
        eig_tol=1e-12,
        tangent_kwargs=tangent_kwargs,
        print_ops=False,
        print_summary=bool(print_buckling_summary),
    )

    if lam is None or len(lam) == 0:
        return {
            "ok": False,
            "reason": "buckling_failed",
            "section": sec.get("name", "?"),
            "sec": sec,
            "sec_assignment": {"mode": "global", "sec": sec},
            "sec_by_member": make_section_by_member(M, sec),
            "Uz_min": uz_min,
            "Uz_ok": uz_ok,
            "DCR_max": float(dcr_max),
            "DCR_ok": dcr_ok,
            "LFB": None,
        }

    lfb = float(lam[0])
    lfb_ok = lfb >= 1.0

    ok_all = bool(uz_ok and lfb_ok and (dcr_ok if require_dcr else True))

    out = {
        "ok": ok_all,
        "section": sec.get("name", "?"),
        "sec": sec,
        "sec_assignment": {"mode": "global", "sec": sec},
        "sec_by_member": make_section_by_member(M, sec),
        "Uz_min": uz_min,
        "Uz_ok": uz_ok,
        "DCR_max": float(dcr_max),
        "DCR_ok": dcr_ok,
        "LFB": lfb,
        "LFB_ok": lfb_ok,
        "max_vertical": res["max_vertical"],
    }

    if print_iter_summary:
        def _fmt(x):
            return "-" if x is None else f"{float(x):.4g}"

        print(
            f"[mecc_opt=1] {out['section']} | ok={out['ok']} | "
            f"LFB={_fmt(out.get('LFB'))} | DCR={_fmt(out.get('DCR_max'))} | Uzmin={_fmt(out.get('Uz_min'))}"
        )

    return out


def optimize_mecc_opt_1(
    G, M,
    ndm, ndf,
    c1, c2, c3, c4, c5,
    q1,
    sections_sorted_light_to_heavy,
    mat,
    uz_limit,
    transf_type,
    init_model,
    define_nodes,
    apply_costr,
    el_ele,
    apply_loads,
    run_static,
    subdivide=1,
    first_new_node_tag=2000,
    update_G=True,
    gamma_M=1.0,
    require_dcr=True,
    print_iter_summary=True,
    debug_kgeo=False,
):
    """
    Lista sezioni già ordinata: piccolo -> grande.
    Si ferma alla prima sezione che verifica i criteri.
    Se nessuna verifica, ritorna il miglior fallback.
    """

    def score(out):
        big = 1e9
        if not out:
            return big
        if out.get("reason") == "static_not_converged":
            return big - 2
        if out.get("reason") == "buckling_failed":
            return big - 1

        dcr = out.get("DCR_max", None)
        lfb = out.get("LFB", None)
        uz = out.get("Uz_min", None)

        dcr = float(dcr) if dcr is not None else 1e6
        lfb = float(lfb) if lfb is not None else 0.0
        uz = float(uz) if uz is not None else 1e6

        dcr_violation = max(0.0, dcr - 1.0) if require_dcr else 0.0
        lfb_violation = max(0.0, 1.0 - lfb)
        uz_violation = max(0.0, abs(uz) - float(uz_limit))
        return 100.0 * dcr_violation + 50.0 * lfb_violation + 10.0 * uz_violation

    best_any = None
    best_any_score = float("inf")

    for sec in sections_sorted_light_to_heavy:
        out = build_and_analyze_global_section(
            G, M,
            ndm, ndf,
            c1, c2, c3, c4, c5,
            q1,
            sec=sec,
            mat=mat,
            transf_type=transf_type,
            subdivide=subdivide,
            first_new_node_tag=first_new_node_tag,
            update_G=update_G,
            init_model=init_model,
            define_nodes=define_nodes,
            apply_costr=apply_costr,
            el_ele=el_ele,
            apply_loads=apply_loads,
            run_static=run_static,
            uz_limit=uz_limit,
            gamma_M=gamma_M,
            require_dcr=require_dcr,
            print_iter_summary=print_iter_summary,
            debug_kgeo=debug_kgeo,
        )

        s = score(out)
        if s < best_any_score:
            best_any = out
            best_any_score = s

        if out.get("ok", False):
            return out

    return best_any

