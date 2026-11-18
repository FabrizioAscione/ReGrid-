# file: buckling_fast.py
import numpy as np
import openseespy.opensees as ops
import math
import contextlib
import os
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union
from transparent_plots import configure_transparent_matplotlib_saves

configure_transparent_matplotlib_saves()

import matplotlib.pyplot as plt


def _norm_vec(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-14:
        return None
    return v / n


@contextlib.contextmanager
def _suppress_native_output(enabled: bool):
    if not enabled:
        yield
        return
    devnull_fd = None
    saved_stdout_fd = None
    saved_stderr_fd = None
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    try:
        try:
            ops.logFile("NUL", "-noEcho")
        except Exception:
            pass
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        saved_stdout_fd = os.dup(1)
        saved_stderr_fd = os.dup(2)
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        yield
    finally:
        try:
            if saved_stdout_fd is not None:
                os.dup2(saved_stdout_fd, 1)
            if saved_stderr_fd is not None:
                os.dup2(saved_stderr_fd, 2)
        finally:
            try:
                ops.logFile("stdout")
            except Exception:
                pass
            if saved_stdout_fd is not None:
                os.close(saved_stdout_fd)
            if saved_stderr_fd is not None:
                os.close(saved_stderr_fd)
            if devnull_fd is not None:
                os.close(devnull_fd)


def _beam_local_axes(p1, p2):
    x_axis = _norm_vec(np.asarray(p2, dtype=float) - np.asarray(p1, dtype=float))
    if x_axis is None:
        return None

    candidates = (
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
    )
    ref = None
    for cand in candidates:
        if np.linalg.norm(np.cross(x_axis, cand)) > 1e-10:
            ref = cand
            break
    if ref is None:
        return None

    y_axis = _norm_vec(np.cross(ref, x_axis))
    if y_axis is None:
        return None
    z_axis = _norm_vec(np.cross(x_axis, y_axis))
    if z_axis is None:
        return None
    return x_axis, y_axis, z_axis


def _hermite_shape_functions(t, L):
    h1 = 1.0 - 3.0 * t**2 + 2.0 * t**3
    h2 = L * (t - 2.0 * t**2 + t**3)
    h3 = 3.0 * t**2 - 2.0 * t**3
    h4 = L * (-t**2 + t**3)
    return h1, h2, h3, h4


def _interpolate_member_curve(nodes, na, nb, da, db, sf, n_curve_points):
    p1 = np.asarray(nodes[na], dtype=float)
    p2 = np.asarray(nodes[nb], dtype=float)
    axes = _beam_local_axes(p1, p2)
    if axes is None:
        return np.array([p1, p2], dtype=float)

    x_axis, y_axis, z_axis = axes
    T = np.column_stack((x_axis, y_axis, z_axis))
    L = np.linalg.norm(p2 - p1)
    if L <= 1e-14:
        return np.array([p1, p2], dtype=float)

    ua_g = np.array(da[:3], dtype=float)
    ub_g = np.array(db[:3], dtype=float)
    ra_g = np.array(da[3:6], dtype=float) if len(da) >= 6 else np.zeros(3, dtype=float)
    rb_g = np.array(db[3:6], dtype=float) if len(db) >= 6 else np.zeros(3, dtype=float)

    ua = T.T @ ua_g
    ub = T.T @ ub_g
    ra = T.T @ ra_g
    rb = T.T @ rb_g

    pts = []
    for t in np.linspace(0.0, 1.0, max(2, int(n_curve_points))):
        x0 = p1 + t * (p2 - p1)

        u_local = (1.0 - t) * ua[0] + t * ub[0]

        h1, h2, h3, h4 = _hermite_shape_functions(t, L)
        v_local = h1 * ua[1] + h2 * ra[2] + h3 * ub[1] + h4 * rb[2]
        w_local = h1 * ua[2] - h2 * ra[1] + h3 * ub[2] - h4 * rb[1]

        disp_global = T @ np.array([u_local, v_local, w_local], dtype=float)
        pts.append(x0 + sf * disp_global)

    return np.asarray(pts, dtype=float)

def get_tangent_stiffness(
    system="FullGeneral",
    numberer="RCM",
    constraints="Plain",
    algorithm="Newton",
    print_ops=False,
    print_mode="all",
    n_preview=10,
):
    quiet_native = not bool(print_ops)
    with _suppress_native_output(quiet_native):
        ops.wipeAnalysis()
        ops.system(system)
        ops.numberer(numberer)
        ops.constraints(constraints)
        ops.integrator("LoadControl", 0.0)
        ops.algorithm(algorithm)
        ops.analysis("Static")
        ops.analyze(1)

    py_lines = [
        "ops.wipeAnalysis()",
        f"ops.system('{system}')",
        f"ops.numberer('{numberer}')",
        f"ops.constraints('{constraints}')",
        "ops.integrator('LoadControl', 0.0)",
        f"ops.algorithm('{algorithm}')",
        "ops.analysis('Static')",
        "ops.analyze(1)",
    ]
    py_text = "\n".join(py_lines)

    if print_ops:
        if str(print_mode).lower() == "all":
            print(py_text)
        else:
            total = len(py_lines)
            print(f"# --- OpenSees commands generated: {total} lines ---")
            for ln in py_lines[:n_preview]:
                print(ln)

    n = ops.systemSize()
    K_flat = ops.printA("-ret")
    K = np.array(K_flat, dtype=float).reshape((n, n))
    return K, py_text


import numpy as np

def linear_buckling_first_mode_sparse(
    K0,
    n_search_modes=3,
    eig_tol=1e-12,
    imag_tol=1e-7,
    drop_tol=1e-12,
    tangent_kwargs=None,
    print_summary=True
):
    if tangent_kwargs is None:
        tangent_kwargs = {}

    from scipy.sparse import csc_matrix
    from scipy.sparse.linalg import eigs, splu, LinearOperator

    Ktot, _ = get_tangent_stiffness(**tangent_kwargs)
    K0 = np.array(K0, dtype=float)
    Ktot = np.array(Ktot, dtype=float)

    if K0.shape != Ktot.shape:
        raise ValueError(f"K0 shape {K0.shape} != Ktot shape {Ktot.shape}")

    Kgeo = K0 - Ktot

    K0 = 0.5 * (K0 + K0.T)
    Kgeo = 0.5 * (Kgeo + Kgeo.T)

    K0[np.abs(K0) < drop_tol] = 0.0
    Kgeo[np.abs(Kgeo) < drop_tol] = 0.0

    K0_sp = csc_matrix(K0)
    Kgeo_sp = csc_matrix(Kgeo)
    lu = splu(K0_sp)

    n = K0.shape[0]
    if n <= 2:
        if print_summary:
            print("# SUMMARY | buckling first mode FAILED | system too small.")
        return np.array([]), None

    def matvec(x):
        return lu.solve(Kgeo_sp @ x)

    A = LinearOperator((n, n), matvec=matvec, dtype=float)
    k = min(max(1, int(n_search_modes)), n - 2)

    try:
        vals_mu, vecs = eigs(
            A,
            k=k,
            which="LR",
            tol=1e-9,
            maxiter=5000,
        )
    except Exception as e:
        if print_summary:
            print(f"# SUMMARY | buckling first mode FAILED | last_err={e}")
        return np.array([]), None

    mask = np.isfinite(vals_mu) & (np.abs(np.imag(vals_mu)) < imag_tol)
    vals_mu = np.real(vals_mu[mask])
    vecs = np.real(vecs[:, mask])

    pos_mask = vals_mu > eig_tol
    vals_mu = vals_mu[pos_mask]
    vecs = vecs[:, pos_mask]

    if vals_mu.size == 0:
        if print_summary:
            print("# SUMMARY | buckling first mode FAILED | no positive reciprocal eigenvalues found.")
        return np.array([]), None

    lam = 1.0 / vals_mu
    idx = np.argsort(lam)
    lam = lam[idx]
    vecs = vecs[:, idx]

    if print_summary:
        print(f"# SUMMARY | buckling first mode lambda_cr={lam[0]:.6g}")

    return lam[:1], vecs[:, 0]

import numpy as np

import numpy as np

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import eigs

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import eigs, splu, LinearOperator

def linear_buckling_sparse(
    K0,
    n_modes=8,
    eig_tol=1e-10,
    imag_tol=1e-7,
    drop_tol=1e-12,
    tangent_kwargs=None,
    print_ops=False,
    print_mode="headtail",
    n_preview=10,
    print_summary=True,
    max_print_modes=10,
):
    if tangent_kwargs is None:
        tangent_kwargs = {}

    Ktot, _ = get_tangent_stiffness(
        print_ops=print_ops,
        print_mode=print_mode,
        n_preview=n_preview,
        **tangent_kwargs
    )

    K0 = np.array(K0, dtype=float)
    Ktot = np.array(Ktot, dtype=float)

    if K0.shape != Ktot.shape:
        raise ValueError(f"K0 shape {K0.shape} != Ktot shape {Ktot.shape}")

    Kgeo = K0 - Ktot

    # pulizia numerica
    K0 = 0.5 * (K0 + K0.T)
    Kgeo = 0.5 * (Kgeo + Kgeo.T)

    K0[np.abs(K0) < drop_tol] = 0.0
    Kgeo[np.abs(Kgeo) < drop_tol] = 0.0

    K0_sp = csc_matrix(K0)
    Kgeo_sp = csc_matrix(Kgeo)

    # fattorizza K0 una sola volta
    lu = splu(K0_sp)

    n = K0.shape[0]

    def matvec(x):
        return lu.solve(Kgeo_sp @ x)

    A = LinearOperator((n, n), matvec=matvec, dtype=float)

    # cerco i Î¼ con parte reale maggiore
    vals_mu, vecs = eigs(
        A,
        k=min(int(n_modes), n - 2),
        which="LR",
        tol=1e-9,
        maxiter=5000
    )

    # tengo solo autovalori quasi reali
    mask = np.isfinite(vals_mu) & (np.abs(np.imag(vals_mu)) < imag_tol)
    vals_mu = np.real(vals_mu[mask])
    vecs = np.real(vecs[:, mask])

    # tengo solo Î¼ positivi
    pos_mask = vals_mu > eig_tol
    vals_mu = vals_mu[pos_mask]
    vecs = vecs[:, pos_mask]

    if vals_mu.size == 0:
        if print_summary:
            print("# SUMMARY | buckling: no positive reciprocal eigenvalues found.")
        return np.array([]), np.array([[]])

    # Î» = 1/Î¼
    lam = 1.0 / vals_mu

    # ordino per Î» crescente
    idx = np.argsort(lam)
    lam = lam[idx]
    vecs = vecs[:, idx]

    if print_summary:
        nshow = min(int(max_print_modes), lam.size)
        txt = ", ".join(f"{lam[i]:.6g}" for i in range(nshow))
        print(f"# SUMMARY | buckling eigenvalues lambda (first {nshow}/{lam.size}): {txt}")

    return lam, vecs

from scipy.linalg import eig
import numpy as np

def linear_buckling_dense_debug_current_state(
    K0,
    tangent_kwargs=None,
    eig_tol=1e-10,
    imag_tol=1e-7,
    symmetrize=True,
    print_summary=True,
    n_show=20,
):
    if tangent_kwargs is None:
        tangent_kwargs = {}

    Ktot, _ = get_tangent_stiffness(**tangent_kwargs)

    K0 = np.array(K0, dtype=float)
    Ktot = np.array(Ktot, dtype=float)

    if K0.shape != Ktot.shape:
        raise ValueError(f"K0 shape {K0.shape} != Ktot shape {Ktot.shape}")

    Kgeo = K0 - Ktot

    if symmetrize:
        K0 = 0.5 * (K0 + K0.T)
        Kgeo = 0.5 * (Kgeo + Kgeo.T)

    vals, vecs = eig(K0, Kgeo)

    mask = np.isfinite(vals) & (np.abs(np.imag(vals)) < imag_tol)
    vals = np.real(vals[mask])
    vecs = np.real(vecs[:, mask])

    idx = np.argsort(vals)
    vals = vals[idx]
    vecs = vecs[:, idx]

    if print_summary:
        print("All real eigenvalues (sorted):")
        print(vals[:n_show])

        pos = vals[vals > eig_tol]
        neg = vals[vals < -eig_tol]

        if len(pos):
            print("First positive eigenvalues:", pos[:min(10, len(pos))])
        else:
            print("No positive real eigenvalues found.")

        if len(neg):
            print("First negative eigenvalues:", neg[:min(10, len(neg))])
        else:
            print("No negative real eigenvalues found.")

        print("||K0||   =", np.linalg.norm(K0))
        print("||Ktot|| =", np.linalg.norm(Ktot))
        print("||Kgeo|| =", np.linalg.norm(Kgeo))
        print("sym err K0   =", np.linalg.norm(K0 - K0.T))
        print("sym err Kgeo =", np.linalg.norm(Kgeo - Kgeo.T))

    return vals, vecs, Ktot, Kgeo

    
def build_dof_map(node_tags, ndf):
    """
    Mappa (nodeTag, dof_idx 0-based) -> eq number (>=0) oppure -1 se vincolato.
    """
    dof_map = {}
    for nd in node_tags:
        eq_nums = ops.nodeDOFs(int(nd))
        if len(eq_nums) != int(ndf):
            raise ValueError(f"nodeDOFs({nd}) returned {len(eq_nums)} dofs, expected ndf={ndf}.")
        for dof_idx, eq in enumerate(eq_nums):
            dof_map[(int(nd), int(dof_idx))] = int(eq)
    return dof_map


def eigenvector_to_nodal_mode(phi, node_tags, dof_map, ndf):
    """
    Converte un autovettore globale phi (per eqn) in dizionario nodale.
    """
    nodal = {}
    ndf = int(ndf)

    for nd in node_tags:
        vals = []
        for dof_idx in range(ndf):
            eq = dof_map[(int(nd), dof_idx)]
            vals.append(float(phi[eq]) if eq >= 0 else 0.0)
        nodal[int(nd)] = tuple(vals)

    return nodal


def vectors_to_modes(vecs, ndf, node_tags=None):
    """
    Converte vecs (matrice autovettori) in lista di modes nodali.
    vecs: shape (neq, n_modes) -> ogni colonna Ã¨ un modo

    Returns: modes = [ {nodeTag:(...)} , ... ]
    """
    if node_tags is None:
        node_tags = ops.getNodeTags()
    node_tags = list(map(int, node_tags))

    dof_map = build_dof_map(node_tags, ndf)

    modes = []
    if vecs is None or vecs.size == 0:
        return modes

    n_modes = vecs.shape[1]
    for i in range(n_modes):
        phi = vecs[:, i]
        modes.append(eigenvector_to_nodal_mode(phi, node_tags, dof_map, ndf))

    return modes


def plot_buckling_modes(
    G,
    M,
    modes,
    lam=None,
    mode_index=1,
    scale=0.20,
    include_boundary_ring=True,
    elev=25,
    azim=-60,
    lw_undef=0.8,
    lw_def=2.0,
    clean=True,
    title=True,
    show_undeformed=False,
    subplot=False,
    ncols=2,
    figsize_per_plot=(6, 5),
    show=True,
    save=False,
    out_dir="outputs",
    save_prefix="buckling_mode",
    dpi=300,
    bbox_inches="tight",
    n_curve_points=2,
):
    """
    Plot one or more buckling modes.

    New options:
    - show: if False, figures are not displayed.
    - save: if True, figures are saved to out_dir.
    - save_prefix: prefix used for output filenames.

    Returns
    -------
    If subplot=False or a single mode is requested:
        list[matplotlib.figure.Figure]
    If subplot=True with multiple modes:
        matplotlib.figure.Figure
    """
    if isinstance(mode_index, (int, np.integer)):
        mode_ids = [int(mode_index)]
    else:
        mode_ids = [int(i) for i in mode_index]

    n_modes_avail = len(modes)
    valid_ids: List[int] = []
    for mid in mode_ids:
        if mid < 1 or mid > n_modes_avail:
            print(f"# WARNING | mode {mid} not available (1..{n_modes_avail}), skipped.")
            continue
        valid_ids.append(mid)

    if not valid_ids:
        return []

    out_path = Path(out_dir)
    if save:
        out_path.mkdir(parents=True, exist_ok=True)

    # old behavior: one figure per mode
    if not subplot or len(valid_ids) == 1:
        figs = []
        for mid in valid_ids:
            mode_dict = modes[mid - 1]
            lam_i = lam[mid - 1] if lam is not None and mid - 1 < len(lam) else None

            fig, _ = _plot_single_mode(
                G,
                M,
                mode_dict,
                mode_index=mid,
                lam=lam_i,
                scale=scale,
                include_boundary_ring=include_boundary_ring,
                elev=elev,
                azim=azim,
                lw_undef=lw_undef,
                lw_def=lw_def,
                clean=clean,
                title=title,
                show_undeformed=show_undeformed,
                n_curve_points=n_curve_points,
                ax=None,
                show=show,
            )
            figs.append(fig)

            if save:
                lam_suffix = ""
                if lam_i is not None:
                    lam_suffix = f"_lam_{float(lam_i):.3f}".replace(".", "p")
                filename = out_path / f"{save_prefix}_{mid:02d}{lam_suffix}.png"
                fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches)
                print(f"# SAVED | {filename}")

        return figs

    # subplot mode
    n = len(valid_ids)
    ncols = max(1, int(ncols))
    nrows = math.ceil(n / ncols)

    fig = plt.figure(figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows))

    for i, mid in enumerate(valid_ids, start=1):
        ax = fig.add_subplot(nrows, ncols, i, projection="3d")
        mode_dict = modes[mid - 1]
        lam_i = lam[mid - 1] if lam is not None and mid - 1 < len(lam) else None

        _plot_single_mode(
            G,
            M,
            mode_dict,
            mode_index=mid,
            lam=lam_i,
            scale=scale,
            include_boundary_ring=include_boundary_ring,
            elev=elev,
            azim=azim,
            lw_undef=lw_undef,
            lw_def=lw_def,
            clean=clean,
            title=title,
            show_undeformed=show_undeformed,
            n_curve_points=n_curve_points,
            ax=ax,
            show=False,
        )

    plt.tight_layout()
    plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99, wspace=0.0, hspace=0.0)

    if save:
        first_mid = valid_ids[0]
        last_mid = valid_ids[-1]
        filename = out_path / f"{save_prefix}_{first_mid:02d}_to_{last_mid:02d}_subplot.png"
        fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches)
        print(f"# SAVED | {filename}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def _plot_single_mode(
    G,
    M,
    mode_dict,
    mode_index=1,
    lam=None,
    scale=0.20,
    include_boundary_ring=True,
    elev=25,
    azim=-60,
    lw_undef=0.8,
    lw_def=2.0,
    clean=True,
    title=True,
    show_undeformed=False,
    n_curve_points=2,
    ax=None,
    show=True,
):
    nodes = G["nodes"]

    edges = list(M.get("members", []))
    if include_boundary_ring:
        edges += list(M.get("boundary_edges", []))

    disp = []
    for nd, d in mode_dict.items():
        ux = d[0] if len(d) > 0 else 0.0
        uy = d[1] if len(d) > 1 else 0.0
        uz = d[2] if len(d) > 2 else 0.0
        disp.append((ux**2 + uy**2 + uz**2) ** 0.5)
    max_disp = max(disp) if disp and max(disp) > 0 else 1.0

    xyz = np.array(list(nodes.values()), float)
    x_min, x_max = xyz[:, 0].min(), xyz[:, 0].max()
    y_min, y_max = xyz[:, 1].min(), xyz[:, 1].max()
    z_min, z_max = xyz[:, 2].min(), xyz[:, 2].max()

    Lchar = max(x_max - x_min, y_max - y_min, z_max - z_min)
    if Lchar <= 0:
        Lchar = 1.0

    sf = scale * Lchar / max_disp

    created_fig = False
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        created_fig = True
    else:
        fig = ax.figure

    if show_undeformed:
        for (na, nb) in edges:
            xa, ya, za = nodes[na]
            xb, yb, zb = nodes[nb]
            ax.plot([xa, xb], [ya, yb], [za, zb], color="k", linestyle="--", linewidth=lw_undef)

    for (na, nb) in edges:
        xa, ya, za = nodes[na]
        xb, yb, zb = nodes[nb]
        da = mode_dict.get(na, (0, 0, 0, 0, 0, 0))
        db = mode_dict.get(nb, (0, 0, 0, 0, 0, 0))
        if int(n_curve_points) <= 2:
            ax.plot(
                [xa + sf * da[0], xb + sf * db[0]],
                [ya + sf * da[1], yb + sf * db[1]],
                [za + sf * da[2], zb + sf * db[2]],
                color="k",
                linestyle="-",
                linewidth=lw_def,
            )
        else:
            pts = _interpolate_member_curve(nodes, na, nb, da, db, sf, n_curve_points)
            ax.plot(
                pts[:, 0],
                pts[:, 1],
                pts[:, 2],
                color="k",
                linestyle="-",
                linewidth=lw_def,
            )

    ax.view_init(elev=elev, azim=azim)

    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
    x_mid = 0.5 * (x_max + x_min)
    y_mid = 0.5 * (y_max + y_min)
    z_mid = 0.5 * (z_max + z_min)
    ax.set_xlim(x_mid - max_range / 2, x_mid + max_range / 2)
    ax.set_ylim(y_mid - max_range / 2, y_mid + max_range / 2)
    ax.set_zlim(z_mid - max_range / 2, z_mid + max_range / 2)

    if title:
        t = f"Mode {mode_index}"
        if lam is not None:
            t += f" | lambda = {lam:.3f}"
        ax.set_title(t)

    if clean:
        ax.set_axis_off()
        ax.grid(False)
        try:
            ax.xaxis.pane.set_alpha(0.0)
            ax.yaxis.pane.set_alpha(0.0)
            ax.zaxis.pane.set_alpha(0.0)
        except Exception:
            pass

    if created_fig:
        plt.tight_layout()
        if show:
            plt.show()
        else:
            plt.close(fig)

    return fig, ax

