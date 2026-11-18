import math
import os
import re
from typing import Any, Dict, List, Mapping, Optional

from transparent_plots import configure_transparent_matplotlib_saves

configure_transparent_matplotlib_saves()


PatchSpec = Dict[str, Any]


def _sec_get(sec: Mapping[str, Any], *names: str, default=None):
    for name in names:
        if name in sec and sec[name] is not None:
            return sec[name]
    return default


def section_family(sec: Mapping[str, Any]) -> Optional[str]:
    fam = _sec_get(sec, "section_family", "family", "type", default=None)
    if fam is None:
        return None
    return str(fam).strip().upper()


def _as_float(value: Any, *, label: str) -> float:
    if value is None:
        raise ValueError(f"Missing section property '{label}'.")
    try:
        return float(value)
    except Exception as exc:
        raise ValueError(f"Invalid section property '{label}': {value!r}") from exc


def _rect_patch(y1: float, z1: float, y2: float, z2: float, *, nfy: int, nfz: int) -> PatchSpec:
    return {
        "kind": "rect",
        "nfy": max(1, int(nfy)),
        "nfz": max(1, int(nfz)),
        "y1": float(y1),
        "z1": float(z1),
        "y2": float(y2),
        "z2": float(z2),
    }


def _quad_patch(points, *, nfy: int, nfz: int) -> PatchSpec:
    return {
        "kind": "quad",
        "nfy": max(1, int(nfy)),
        "nfz": max(1, int(nfz)),
        "points": [(float(y), float(z)) for y, z in points],
    }


def _family_fiber_cfg(sec: Mapping[str, Any], fiber_opts: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    fiber_opts = fiber_opts or {}
    fam = section_family(sec)
    if fam in {"L", "2L", "4L"}:
        cfg = fiber_opts.get(fam, None)
        if isinstance(cfg, Mapping):
            return cfg
    return {}


def _fiber_mesh_opts(sec: Mapping[str, Any], fiber_opts: Optional[Mapping[str, Any]]) -> Dict[str, int]:
    fiber_opts = fiber_opts or {}
    fam_cfg = _family_fiber_cfg(sec, fiber_opts)
    return {
        "n_web_y": max(1, int(fam_cfg.get("n_web_y", fiber_opts.get("n_web_y", 2)))),
        "n_web_z": max(1, int(fam_cfg.get("n_web_z", fiber_opts.get("n_web_z", 8)))),
        "n_flange_y": max(1, int(fam_cfg.get("n_flange_y", fiber_opts.get("n_flange_y", 8)))),
        "n_flange_z": max(1, int(fam_cfg.get("n_flange_z", fiber_opts.get("n_flange_z", 2)))),
        "n_circ": max(1, int(fiber_opts.get("n_circ", 32))),
        "n_rad": max(1, int(fiber_opts.get("n_rad", 4))),
    }


def _rotation_deg(sec: Mapping[str, Any], fiber_opts: Optional[Mapping[str, Any]]) -> float:
    # Section rotation is globally disabled to keep elastic and nonlinear
    # section orientation consistent with the unrotated local axes.
    return 0.0


def get_section_rotation_deg(sec: Mapping[str, Any], fiber_opts: Optional[Mapping[str, Any]] = None) -> float:
    return _rotation_deg(sec, fiber_opts)


def _rotate_point(y: float, z: float, angle_deg: float) -> tuple[float, float]:
    ang = math.radians(float(angle_deg))
    c = math.cos(ang)
    s = math.sin(ang)
    return (c * y) - (s * z), (s * y) + (c * z)


def _apply_rotation_to_patch_specs(patch_specs: List[PatchSpec], angle_deg: float) -> List[PatchSpec]:
    if abs(float(angle_deg)) <= 1e-12:
        return patch_specs

    rotated: List[PatchSpec] = []
    for patch in patch_specs:
        if patch["kind"] == "rect":
            y1 = float(patch["y1"])
            z1 = float(patch["z1"])
            y2 = float(patch["y2"])
            z2 = float(patch["z2"])
            corners = [
                (y1, z1),
                (y2, z1),
                (y2, z2),
                (y1, z2),
            ]
            points = [_rotate_point(y, z, angle_deg) for y, z in corners]
            rotated.append(_quad_patch(points, nfy=int(patch["nfy"]), nfz=int(patch["nfz"])))
        elif patch["kind"] == "quad":
            points = [_rotate_point(y, z, angle_deg) for y, z in patch["points"]]
            rotated.append(_quad_patch(points, nfy=int(patch["nfy"]), nfz=int(patch["nfz"])))
        elif patch["kind"] == "circ":
            pc = dict(patch)
            yC, zC = _rotate_point(float(patch["yC"]), float(patch["zC"]), angle_deg)
            pc["yC"] = yC
            pc["zC"] = zC
            rotated.append(pc)
        else:
            raise NotImplementedError(f"Unsupported patch kind '{patch['kind']}'.")
    return rotated


def _build_chs_patches(sec: Mapping[str, Any], fiber_opts: Optional[Mapping[str, Any]]) -> List[PatchSpec]:
    mesh = _fiber_mesh_opts(sec, fiber_opts)
    yC = float((fiber_opts or {}).get("yC", 0.0))
    zC = float((fiber_opts or {}).get("zC", 0.0))
    a0 = float((fiber_opts or {}).get("a0_deg", 0.0))
    a1 = float((fiber_opts or {}).get("a1_deg", 360.0))

    dext = _sec_get(sec, "Dext", "dext", default=None)
    dint = _sec_get(sec, "Dint", "dint", default=None)
    t = _sec_get(sec, "t", "thickness", default=None)
    if dext is None:
        raise ValueError("CHS fiber section needs sec['Dext'] or sec['dext'].")
    rO = float(dext) / 2.0
    if dint is not None:
        rI = float(dint) / 2.0
    elif t is not None:
        rI = rO - float(t)
    else:
        rI = 0.0
    if rI < 0.0 or rI > rO:
        raise ValueError(f"Invalid CHS geometry: rI={rI}, rO={rO}")

    return [{
        "kind": "circ",
        "n_circ": mesh["n_circ"],
        "n_rad": mesh["n_rad"],
        "yC": yC,
        "zC": zC,
        "rI": float(rI),
        "rO": float(rO),
        "a0": a0,
        "a1": a1,
    }]


def _build_single_L_patches(sec: Mapping[str, Any], fiber_opts: Optional[Mapping[str, Any]]) -> List[PatchSpec]:
    mesh = _fiber_mesh_opts(sec, fiber_opts)
    h = _as_float(_sec_get(sec, "h"), label="h")
    b = _as_float(_sec_get(sec, "b"), label="b")
    tw = _as_float(_sec_get(sec, "tw"), label="tw")
    tf = _as_float(_sec_get(sec, "tf"), label="tf")

    A1 = h * tw
    A2 = max(0.0, b - tw) * tf
    A = A1 + A2
    if A <= 0.0:
        raise ValueError("Invalid L section geometry: area <= 0.")

    yG1 = tw / 2.0
    zG1 = h / 2.0
    yG2 = tw + max(0.0, b - tw) / 2.0
    zG2 = tf / 2.0
    yG = (A1 * yG1 + A2 * yG2) / A
    zG = (A1 * zG1 + A2 * zG2) / A

    patches = [
        _rect_patch(-yG, -zG, tw - yG, h - zG, nfy=mesh["n_web_y"], nfz=mesh["n_web_z"]),
    ]
    if b > tw and tf > 0.0:
        patches.append(
            _rect_patch(tw - yG, -zG, b - yG, tf - zG, nfy=mesh["n_flange_y"], nfz=mesh["n_flange_z"])
        )
    return patches


def _build_double_L_patches(sec: Mapping[str, Any], fiber_opts: Optional[Mapping[str, Any]]) -> List[PatchSpec]:
    mesh = _fiber_mesh_opts(sec, fiber_opts)
    h_total = _as_float(_sec_get(sec, "h"), label="h")
    b_total = _as_float(_sec_get(sec, "b"), label="b")
    web_width = _as_float(_sec_get(sec, "tf"), label="tf")
    flange_height = _as_float(_sec_get(sec, "tw"), label="tw")

    if h_total <= 0.0 or b_total <= 0.0 or web_width <= 0.0 or flange_height <= 0.0:
        raise ValueError("Invalid 2L section geometry.")

    A1 = h_total * web_width
    A2 = max(0.0, b_total - web_width) * flange_height
    A = A1 + A2
    if A <= 0.0:
        raise ValueError("Invalid 2L section geometry: area <= 0.")

    zG1 = h_total / 2.0
    yG1 = web_width / 2.0
    zG2 = h_total / 2.0
    yG2 = web_width + max(0.0, b_total - web_width) / 2.0
    zG = (A1 * zG1 + A2 * zG2) / A
    yG = (A1 * yG1 + A2 * yG2) / A

    z_flange_1 = (h_total / 2.0) - (flange_height / 2.0)
    z_flange_2 = (h_total / 2.0) + (flange_height / 2.0)

    patches = [
        _rect_patch(-yG, -zG, web_width - yG, h_total - zG, nfy=mesh["n_web_y"], nfz=mesh["n_web_z"]),
    ]
    if b_total > web_width and flange_height > 0.0:
        patches.append(
            _rect_patch(
                web_width - yG,
                z_flange_1 - zG,
                b_total - yG,
                z_flange_2 - zG,
                nfy=mesh["n_flange_y"],
                nfz=mesh["n_flange_z"],
            )
        )
    return patches


def _build_four_L_patches(sec: Mapping[str, Any], fiber_opts: Optional[Mapping[str, Any]]) -> List[PatchSpec]:
    mesh = _fiber_mesh_opts(sec, fiber_opts)
    H = _as_float(_sec_get(sec, "h"), label="h")
    B = _as_float(_sec_get(sec, "b"), label="b")
    tf = _as_float(_sec_get(sec, "tf"), label="tf")
    tw = _as_float(_sec_get(sec, "tw"), label="tw")

    if H <= 0.0 or B <= 0.0 or tf <= 0.0 or tw <= 0.0 or 2.0 * tf >= H or tw >= B:
        raise ValueError("Invalid 4L section geometry.")

    y1 = -B / 2.0
    y2 = B / 2.0
    z1 = -H / 2.0
    z2 = H / 2.0

    patches = [
        _rect_patch(y1, z1, y2, z1 + tf, nfy=mesh["n_flange_y"], nfz=mesh["n_flange_z"]),
        _rect_patch(y1, z2 - tf, y2, z2, nfy=mesh["n_flange_y"], nfz=mesh["n_flange_z"]),
        _rect_patch(-tw / 2.0, z1 + tf, tw / 2.0, z2 - tf, nfy=mesh["n_web_y"], nfz=mesh["n_web_z"]),
    ]
    return patches


def supported_fiber_family(sec: Mapping[str, Any]) -> bool:
    fam = section_family(sec)
    return fam in {"CHS", "O", "PIPE", "TUBE", "ROUND", "ROUND_HOLLOW", "CIRCULAR_HOLLOW", "L", "2L", "4L"}


def require_supported_fiber_family(sec: Mapping[str, Any], key: Any, *, context: str) -> str:
    fam = section_family(sec)
    if fam is None:
        raise ValueError(f"{context} needs section_family for member {key}.")
    if not supported_fiber_family(sec):
        raise NotImplementedError(
            f"{context} currently supports CHS, L, 2L, and 4L; member {key} has section_family='{fam}'."
        )
    return fam


def build_fiber_patch_specs(sec: Mapping[str, Any], fiber_opts: Optional[Mapping[str, Any]] = None) -> List[PatchSpec]:
    fam = section_family(sec)
    if fam in {"CHS", "O", "PIPE", "TUBE", "ROUND", "ROUND_HOLLOW", "CIRCULAR_HOLLOW"}:
        patch_specs = _build_chs_patches(sec, fiber_opts)
    elif fam == "L":
        patch_specs = _build_single_L_patches(sec, fiber_opts)
    elif fam == "2L":
        patch_specs = _build_double_L_patches(sec, fiber_opts)
    elif fam == "4L":
        patch_specs = _build_four_L_patches(sec, fiber_opts)
    else:
        raise NotImplementedError(f"Unsupported fiber section family '{fam}'.")

    return _apply_rotation_to_patch_specs(patch_specs, _rotation_deg(sec, fiber_opts))


def _sanitize_filename(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"[^\w\-\.]+", "_", text)
    return text[:180] if len(text) > 180 else text


def _update_bounds(bounds: List[float], y_values: List[float], z_values: List[float]) -> None:
    bounds[0] = min(bounds[0], *y_values)
    bounds[1] = max(bounds[1], *y_values)
    bounds[2] = min(bounds[2], *z_values)
    bounds[3] = max(bounds[3], *z_values)


def plot_fiber_section(
    sec: Mapping[str, Any],
    fiber_opts: Optional[Mapping[str, Any]] = None,
    *,
    save_path: Optional[str] = None,
    show: bool = False,
    title: Optional[str] = None,
) -> Optional[str]:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Polygon, Rectangle, Wedge

    patch_specs = build_fiber_patch_specs(sec, fiber_opts)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    cmap = plt.get_cmap("tab20")
    bounds = [float("inf"), float("-inf"), float("inf"), float("-inf")]

    def _lerp(p0, p1, t):
        return ((1.0 - t) * p0[0] + t * p1[0], (1.0 - t) * p0[1] + t * p1[1])

    for idx, patch in enumerate(patch_specs):
        color = cmap(idx % 20)
        if patch["kind"] == "rect":
            y1 = float(patch["y1"])
            y2 = float(patch["y2"])
            z1 = float(patch["z1"])
            z2 = float(patch["z2"])
            x0 = min(y1, y2)
            y0 = min(z1, z2)
            width = abs(y2 - y1)
            height = abs(z2 - z1)
            ax.add_patch(Rectangle((x0, y0), width, height, facecolor=color, edgecolor="black", alpha=0.45, linewidth=1.2))

            nfy = max(1, int(patch["nfy"]))
            nfz = max(1, int(patch["nfz"]))
            for k in range(1, nfy):
                yy = x0 + width * k / nfy
                ax.plot([yy, yy], [y0, y0 + height], color="black", linewidth=0.5, alpha=0.6)
            for k in range(1, nfz):
                zz = y0 + height * k / nfz
                ax.plot([x0, x0 + width], [zz, zz], color="black", linewidth=0.5, alpha=0.6)

            ax.text(x0 + width / 2.0, y0 + height / 2.0, f"P{idx+1}\n{nfy}x{nfz}", ha="center", va="center", fontsize=7)
            _update_bounds(bounds, [x0, x0 + width], [y0, y0 + height])

        elif patch["kind"] == "quad":
            pts = [(float(y), float(z)) for y, z in patch["points"]]
            poly = Polygon(pts, closed=True, facecolor=color, edgecolor="black", alpha=0.45, linewidth=1.2)
            ax.add_patch(poly)

            nfy = max(1, int(patch["nfy"]))
            nfz = max(1, int(patch["nfz"]))
            p1, p2, p3, p4 = pts
            for k in range(1, nfy):
                t = k / nfy
                a = _lerp(p1, p4, t)
                b = _lerp(p2, p3, t)
                ax.plot([a[0], b[0]], [a[1], b[1]], color="black", linewidth=0.5, alpha=0.6)
            for k in range(1, nfz):
                t = k / nfz
                a = _lerp(p1, p2, t)
                b = _lerp(p4, p3, t)
                ax.plot([a[0], b[0]], [a[1], b[1]], color="black", linewidth=0.5, alpha=0.6)

            yc = sum(p[0] for p in pts) / 4.0
            zc = sum(p[1] for p in pts) / 4.0
            ax.text(yc, zc, f"P{idx+1}\n{nfy}x{nfz}", ha="center", va="center", fontsize=7)
            _update_bounds(bounds, [p[0] for p in pts], [p[1] for p in pts])

        elif patch["kind"] == "circ":
            yC = float(patch["yC"])
            zC = float(patch["zC"])
            rI = float(patch["rI"])
            rO = float(patch["rO"])
            a0 = float(patch["a0"])
            a1 = float(patch["a1"])
            n_circ = max(1, int(patch["n_circ"]))
            n_rad = max(1, int(patch["n_rad"]))

            ax.add_patch(Wedge((yC, zC), rO, a0, a1, width=max(0.0, rO - rI), facecolor=color, edgecolor="black", alpha=0.45, linewidth=1.2))
            if rI > 0.0:
                ax.add_patch(Circle((yC, zC), rI, fill=False, edgecolor="black", linewidth=0.8, alpha=0.8))
            ax.add_patch(Circle((yC, zC), rO, fill=False, edgecolor="black", linewidth=0.8, alpha=0.8))

            for k in range(1, n_rad):
                rr = rI + (rO - rI) * k / n_rad
                ax.add_patch(Circle((yC, zC), rr, fill=False, edgecolor="black", linewidth=0.4, alpha=0.5))
            for k in range(n_circ):
                ang = math.radians(a0 + (a1 - a0) * k / n_circ)
                ax.plot([yC + rI * math.cos(ang), yC + rO * math.cos(ang)],
                        [zC + rI * math.sin(ang), zC + rO * math.sin(ang)],
                        color="black", linewidth=0.4, alpha=0.5)

            ax.text(yC, zC, f"P{idx+1}\n{n_circ}x{n_rad}", ha="center", va="center", fontsize=7)
            _update_bounds(bounds, [yC - rO, yC + rO], [zC - rO, zC + rO])
        else:
            raise NotImplementedError(f"Unsupported patch kind '{patch['kind']}'.")

    ax.axhline(0.0, color="0.5", linewidth=0.8, linestyle="--")
    ax.axvline(0.0, color="0.5", linewidth=0.8, linestyle="--")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.2)
    ax.set_xlabel("local y [m]")
    ax.set_ylabel("local z [m]")
    sec_name = str(sec.get("name", sec.get("section_family", "section")))
    ax.set_title(title or f"Fiber Section: {sec_name}")

    if math.isfinite(bounds[0]) and math.isfinite(bounds[1]) and math.isfinite(bounds[2]) and math.isfinite(bounds[3]):
        dy = bounds[1] - bounds[0]
        dz = bounds[3] - bounds[2]
        pad = 0.12 * max(dy, dz, 1e-3)
        ax.set_xlim(bounds[0] - pad, bounds[1] + pad)
        ax.set_ylim(bounds[2] - pad, bounds[3] + pad)

    final_path = None
    if save_path:
        folder = os.path.dirname(save_path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        final_path = save_path
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return final_path


def plot_unique_fiber_sections(
    section_by_member: Mapping[Any, Mapping[str, Any]],
    fiber_opts: Optional[Mapping[str, Any]] = None,
    *,
    out_dir: Optional[str] = None,
    show: bool = False,
) -> List[str]:
    saved: List[str] = []
    seen = set()
    for sec in section_by_member.values():
        sec_name = str(sec.get("name", "")).strip()
        if not sec_name or sec_name in seen:
            continue
        seen.add(sec_name)
        save_path = None
        if out_dir:
            save_path = os.path.join(out_dir, f"{_sanitize_filename(sec_name)}.png")
        out = plot_fiber_section(sec, fiber_opts, save_path=save_path, show=show)
        if out:
            saved.append(out)
    return saved
