import os
from collections import defaultdict

import numpy as np
from transparent_plots import configure_transparent_matplotlib_saves

configure_transparent_matplotlib_saves()

import matplotlib.pyplot as plt


def _ensure_dir(path: str):
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass


def _converged_local_hist(hist):
    return [h for h in hist if h.get("ok", 0) == 0 and isinstance(h.get("local_response", None), dict)]


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def _select_N_value(mrec, N_use="abs"):
    mode = str(N_use).lower()
    if mode == "raw":
        return _safe_float(mrec.get("N_member_raw", np.nan))
    if mode == "comp_pos":
        return _safe_float(mrec.get("N_member_comp", np.nan))
    return _safe_float(mrec.get("N_member_abs", np.nan))


def _normalize_by_absmax(arr):
    arr = np.asarray(arr, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.full_like(arr, np.nan, dtype=float), np.nan
    amax = float(np.nanmax(np.abs(finite)))
    if not np.isfinite(amax) or amax <= 0.0:
        return np.full_like(arr, np.nan, dtype=float), amax
    return arr / amax, amax


def _clamp01(value):
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 1.0


def _interp_segment_value(arr, idx, alpha):
    values = np.asarray(arr, dtype=float)
    if values.size == 0:
        return np.nan
    idx = int(idx)
    alpha = _clamp01(alpha)
    if idx <= 0 or idx >= values.size or not np.isfinite(alpha):
        return float(values[min(max(idx, 0), values.size - 1)])
    v0 = values[idx - 1]
    v1 = values[idx]
    if not (np.isfinite(v0) and np.isfinite(v1)):
        return float(v1 if np.isfinite(v1) else v0)
    return float(v0 + alpha * (v1 - v0))


def _segment_crossing_alpha(v0, v1, threshold, prefer=1.0):
    if not (np.isfinite(v0) and np.isfinite(v1) and np.isfinite(threshold)):
        return _clamp01(prefer)
    dv = float(v1 - v0)
    if abs(dv) <= 1e-14:
        return _clamp01(prefer)
    alpha = (float(threshold) - float(v0)) / dv
    return _clamp01(alpha)


def _build_member_series(hist, N_use="abs"):
    data = _converged_local_hist(hist)
    by_member = defaultdict(lambda: defaultdict(list))
    for h in data:
        step = int(h.get("step", -1))
        lf = _safe_float(h.get("LF", np.nan))
        u = _safe_float(h.get("u", np.nan))
        uz = _safe_float(h.get("Uz", np.nan))
        for member_id, mrec in h.get("local_response", {}).items():
            md = by_member[member_id]
            md["step"].append(step)
            md["LF"].append(lf)
            md["u"].append(u)
            md["Uz"].append(uz)
            md["d_end"].append(_safe_float(mrec.get("d_end", np.nan)))
            md["d_mid"].append(_safe_float(mrec.get("d_mid", np.nan)))
            md["N_member"].append(_select_N_value(mrec, N_use=N_use))
            md["N_member_raw"].append(_safe_float(mrec.get("N_member_raw", np.nan)))
            md["N_i"].append(_safe_float(mrec.get("N_i_raw", np.nan)))
            md["N_j"].append(_safe_float(mrec.get("N_j_raw", np.nan)))
            md["N_mid"].append(_safe_float(mrec.get("N_mid_raw", np.nan)))
            md["dU_local"].append(_safe_float(mrec.get("dU_local", np.nan)))
            md["U_local"].append(_safe_float(mrec.get("U_local", np.nan)))
            md["ele_tags"] = list(mrec.get("ele_tags", []))
            md["end_nodes"] = tuple(mrec.get("end_nodes", (None, None)))
            md["mid_node"] = mrec.get("mid_node", None)

    for member_id, md in by_member.items():
        for k, v in list(md.items()):
            if isinstance(v, list):
                md[k] = np.array(v, dtype=float)

        N = md.get("N_member", np.array([], dtype=float))
        if len(N) and np.isfinite(N).any():
            Nmax = float(np.nanmax(N))
        else:
            Nmax = np.nan
        md["N_max"] = Nmax
        if np.isfinite(Nmax) and Nmax > 0.0:
            md["N_norm"] = N / Nmax
        else:
            md["N_norm"] = np.full_like(N, np.nan, dtype=float)

        md["d_end_norm"], md["d_end_max"] = _normalize_by_absmax(md.get("d_end", np.array([], dtype=float)))
        md["d_mid_norm"], md["d_mid_max"] = _normalize_by_absmax(md.get("d_mid", np.array([], dtype=float)))
        md["U_norm"], md["U_max"] = _normalize_by_absmax(md.get("U_local", np.array([], dtype=float)))
        md["dU_norm"], md["dU_abs_max"] = _normalize_by_absmax(md.get("dU_local", np.array([], dtype=float)))
    return by_member


def _detect_pdrop(N, d, drop_ratio=0.05, drop_consec=2, d_tol=1e-6, d_event_min=0.0):
    if N is None or d is None:
        return None
    N = np.asarray(N, dtype=float)
    d = np.asarray(d, dtype=float)
    valid = np.isfinite(N) & np.isfinite(d)
    if np.count_nonzero(valid) < max(5, int(drop_consec) + 2):
        return None

    peak_idx = int(np.nanargmax(N)) if np.isfinite(N).any() else None
    if peak_idx is None:
        return None
    peakN = float(N[peak_idx])
    if not np.isfinite(peakN) or peakN <= 0.0:
        return None

    thr = (1.0 - float(drop_ratio)) * peakN
    d_peak = float(d[peak_idx]) if np.isfinite(d[peak_idx]) else np.nan

    consec = 0
    start_idx = None
    for k in range(peak_idx + 1, len(N)):
        if not valid[k]:
            consec = 0
            start_idx = None
            continue
        dd = d[k] - d[k - 1] if (k - 1) >= 0 and np.isfinite(d[k - 1]) else np.nan
        drop_ok = bool(np.isfinite(N[k]) and (N[k] <= thr))
        incr_ok = bool(np.isfinite(dd) and (dd > float(d_tol)))
        dmag_ok = bool(np.isfinite(d[k]) and (d[k] >= max(float(d_event_min), d_peak + float(d_tol))))
        if drop_ok and incr_ok and dmag_ok:
            consec += 1
            if start_idx is None:
                start_idx = k
            if consec >= int(drop_consec):
                return {
                    "event_idx": int(start_idx),
                    "event_alpha": 1.0,
                    "peak_idx": int(peak_idx),
                    "N_peak": peakN,
                    "N_event": float(N[start_idx]),
                }
        else:
            consec = 0
            start_idx = None
    return None


def _detect_energy(U, dU, dU_eps=0.0, energy_consec=3):
    U = np.asarray(U, dtype=float)
    dU = np.asarray(dU, dtype=float)
    if len(U) < max(5, int(energy_consec) + 1):
        return None

    consec = 0
    start_idx = None
    for k in range(len(dU)):
        if np.isfinite(dU[k]) and (dU[k] < -abs(float(dU_eps))):
            consec += 1
            if start_idx is None:
                start_idx = k
            if consec >= int(energy_consec):
                peak_idx = int(np.nanargmax(U)) if np.isfinite(U).any() else int(start_idx)
                return {
                    "event_idx": int(start_idx),
                    "event_alpha": 1.0,
                    "peak_idx": int(peak_idx),
                    "U_peak": float(U[peak_idx]) if np.isfinite(U[peak_idx]) else np.nan,
                    "U_event": float(U[start_idx]) if np.isfinite(U[start_idx]) else np.nan,
                    "criterion": "dU_neg",
                }
        else:
            consec = 0
            start_idx = None
    return None


def _classify_buckling_mode(d_end, d_mid, event_idx, d_tol=1e-6):
    if event_idx is None or event_idx <= 0:
        return "unknown"
    ddend = np.nan
    ddmid = np.nan
    try:
        ddend = float(d_end[event_idx] - d_end[event_idx - 1])
    except Exception:
        pass
    try:
        ddmid = float(d_mid[event_idx] - d_mid[event_idx - 1])
    except Exception:
        pass
    end_inc = np.isfinite(ddend) and (ddend > float(d_tol))
    mid_inc = np.isfinite(ddmid) and (ddmid > float(d_tol))
    if end_inc and not mid_inc:
        return "mode1"
    if mid_inc and not end_inc:
        return "mode2"
    if end_inc and mid_inc:
        return "mode3"
    return "unknown"


def _required_count(decision_rule, n_active):
    rule = str(decision_rule).lower()
    n_active = int(max(1, n_active))
    if rule in ("all", "3of3"):
        return n_active
    if rule in ("2of3", "twoofthree"):
        return min(2, n_active)
    return 1


def _combine_method_events(method_event_map, decision_rule="either", active_methods=("pdrop_dend", "pdrop_dmid", "energy")):
    active = [m for m in active_methods if m in method_event_map]
    candidates = sorted(set(int(method_event_map[m]["event_idx"]) for m in active if method_event_map.get(m) is not None))
    if not candidates:
        return None, []
    need = _required_count(decision_rule, len(active))
    for idx in candidates:
        methods = [m for m in active if method_event_map.get(m) is not None and int(method_event_map[m]["event_idx"]) <= idx]
        if len(methods) >= need:
            return idx, methods
    return None, []


def _event_alpha_for_methods(method_map, methods, event_idx):
    alphas = []
    for method in methods:
        payload = method_map.get(method)
        if not payload:
            continue
        if int(payload.get("event_idx", -1)) != int(event_idx):
            continue
        alpha = payload.get("event_alpha", None)
        if alpha is not None and np.isfinite(alpha):
            alphas.append(float(alpha))
    if alphas:
        return _clamp01(min(alphas))
    return 1.0


def _write_events_txt(path, events):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# member local buckling events\n")
        f.write("# member step LF u Uz methods buckling_mode N_peak N_event d_end d_mid\n")
        for ev in events:
            methods = ",".join(ev.get("methods", []))
            f.write(
                f"{ev['member']} {ev['step']} {ev['LF']:.6e} {ev['u']:.6e} {ev['Uz']:.6e} "
                f"{methods} {ev['buckling_mode']} {ev['N_peak']:.6e} {ev['N_event']:.6e} "
                f"{ev['d_end']:.6e} {ev['d_mid']:.6e}\n"
            )


def _write_curves_txt(path, curves):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# local member curves\n")
        for member_id, c in curves.items():
            f.write(f"\n# member {member_id}\n")
            f.write("# step LF u Uz N Nnorm d_end d_mid dU_local U_local\n")
            n = len(c["step"])
            for i in range(n):
                f.write(
                    f"{int(c['step'][i])} {c['LF'][i]:.6e} {c['u'][i]:.6e} {c['Uz'][i]:.6e} "
                    f"{c['N_member'][i]:.6e} {c['N_norm'][i]:.6e} {c['d_end'][i]:.6e} {c['d_mid'][i]:.6e} "
                    f"{c['dU_local'][i]:.6e} {c['U_local'][i]:.6e}\n"
                )


def _sort_member_ids(curves, events_by_member, sort_by="step"):
    def key(member_id):
        ev = events_by_member[member_id]
        c = curves[member_id]
        mode = str(sort_by).lower()
        if mode == "n_peak":
            return (-float(ev.get("N_peak", np.nan)), str(member_id))
        if mode == "u_max":
            return (-float(c.get("U_max", np.nan)), str(member_id))
        if mode == "member":
            return (str(member_id),)
        return (int(ev.get("step", 10**12)), str(member_id))

    return sorted(events_by_member.keys(), key=key)


def _plot_member_curves(
    curves,
    events_by_member,
    out_dir,
    max_members=None,
    sort_by="step",
    normalized=True,
    add_raw_pd_plots=True,
    show=False,
):
    _ensure_dir(out_dir)
    member_ids = _sort_member_ids(curves, events_by_member, sort_by=sort_by)
    if max_members is not None:
        member_ids = member_ids[: int(max_members)]

    for member_id in member_ids:
        c = curves[member_id]
        ev = events_by_member[member_id]
        idx = int(ev["event_idx"])
        event_alpha = float(ev.get("event_alpha", 1.0))

        x_end = c["d_end_norm"] if normalized else c["d_end"]
        x_mid = c["d_mid_norm"] if normalized else c["d_mid"]
        yN = c["N_norm"] if normalized else c["N_member"]
        yU = c["U_norm"] if normalized else c["U_local"]
        ydU = c["dU_norm"] if normalized else c["dU_local"]

        x_end_event = _interp_segment_value(x_end, idx, event_alpha)
        x_mid_event = _interp_segment_value(x_mid, idx, event_alpha)
        yN_event = _interp_segment_value(yN, idx, event_alpha)
        step_event = _interp_segment_value(c["step"], idx, event_alpha)

        xlabel_end = "d_end/d_end,max" if normalized else "d_end"
        xlabel_mid = "d_mid/d_mid,max" if normalized else "d_mid"
        ylabel_N = "N/Nmax" if normalized else "N"
        title_suffix = "normalized" if normalized else "raw"

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        axes[0].plot(x_end, yN, linewidth=1.8)
        if np.isfinite(x_end_event) and np.isfinite(yN_event):
            axes[0].plot(x_end_event, yN_event, marker="o")
        axes[0].set_xlabel(xlabel_end)
        axes[0].set_ylabel(ylabel_N)
        axes[0].set_title(f"Member {member_id} | P-d_end ({title_suffix})")
        axes[0].grid(True)

        axes[1].plot(x_mid, yN, linewidth=1.8)
        if np.isfinite(x_mid_event) and np.isfinite(yN_event):
            axes[1].plot(x_mid_event, yN_event, marker="o")
        axes[1].set_xlabel(xlabel_mid)
        axes[1].set_ylabel(ylabel_N)
        axes[1].set_title(f"Member {member_id} | P-d_mid ({title_suffix})")
        axes[1].grid(True)

        axes[2].plot(c["step"], yU, label=("U/Umax" if normalized else "U_local"), linewidth=1.8)
        axes[2].plot(c["step"], ydU, label=("dU/max|dU|" if normalized else "dU_local"), linewidth=1.2)
        if np.isfinite(step_event):
            axes[2].axvline(step_event, linestyle="--", linewidth=1.2)
        axes[2].set_xlabel("step")
        axes[2].set_title(f"Member {member_id} | energy ({title_suffix})")
        axes[2].grid(True)
        axes[2].legend(loc="best")

        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"local_member_{member_id}_{title_suffix}.png"), dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)

        if add_raw_pd_plots and normalized:
            fig_raw, axes_raw = plt.subplots(1, 2, figsize=(10, 4))
            axes_raw[0].plot(c["d_end"], c["N_member"], linewidth=1.8)
            raw_x_end_event = _interp_segment_value(c["d_end"], idx, event_alpha)
            raw_x_mid_event = _interp_segment_value(c["d_mid"], idx, event_alpha)
            raw_y_event = _interp_segment_value(c["N_member"], idx, event_alpha)
            if np.isfinite(raw_x_end_event) and np.isfinite(raw_y_event):
                axes_raw[0].plot(raw_x_end_event, raw_y_event, marker="o")
            axes_raw[0].set_xlabel("d_end")
            axes_raw[0].set_ylabel("N")
            axes_raw[0].set_title(f"Member {member_id} | P-d_end (raw)")
            axes_raw[0].grid(True)

            axes_raw[1].plot(c["d_mid"], c["N_member"], linewidth=1.8)
            if np.isfinite(raw_x_mid_event) and np.isfinite(raw_y_event):
                axes_raw[1].plot(raw_x_mid_event, raw_y_event, marker="o")
            axes_raw[1].set_xlabel("d_mid")
            axes_raw[1].set_ylabel("N")
            axes_raw[1].set_title(f"Member {member_id} | P-d_mid (raw)")
            axes_raw[1].grid(True)

            fig_raw.tight_layout()
            fig_raw.savefig(os.path.join(out_dir, f"local_member_{member_id}_raw_pd.png"), dpi=300, bbox_inches="tight")
            if show:
                plt.show()
            else:
                plt.close(fig_raw)


def _plot_energy_maps_from_hist(hist, ele_data, out_dir, every=10, show_deformed=False, scale_factor=1.0):
    if ele_data is None:
        return
    try:
        ele_tags_for_map = ele_data.get("ele_tags", None)
    except Exception:
        ele_tags_for_map = None
    if ele_tags_for_map is None:
        return
    try:
        from OS_plot import plot_energy_increment_map_3d_discrete
    except Exception:
        return
    _ensure_dir(out_dir)
    converged = [h for h in hist if h.get("ok", 0) == 0 and isinstance(h.get("DU_ele_map", None), dict)]
    if not converged:
        return
    for h in converged:
        step = int(h.get("step", -1))
        if step > 0 and (step % int(every)) != 0 and step != int(converged[-1].get("step", -1)):
            continue
        try:
            plot_energy_increment_map_3d_discrete(
                ele_tags=ele_tags_for_map,
                DU_ele_map=h["DU_ele_map"],
                outpath=os.path.join(out_dir, f"energy_step_{step:05d}.png"),
                step=step,
                load_factor=float(h.get("LF", np.nan)),
                show_deformed=bool(show_deformed),
                scale_factor=float(scale_factor),
                elev=35,
                azim=-55,
                linewidth=1.7,
            )
        except Exception:
            continue


def postprocess_local_instability(
    hist,
    summary=None,
    N_use="abs",
    drop_ratio=0.05,
    drop_consec=2,
    d_tol=1e-6,
    dU_eps=None,
    U_eps=None,
    energy_consec=3,
    decision_rule="either",
    active_methods=("pdrop_dend", "pdrop_dmid", "energy"),
    N_peak_min_abs=0.0,
    N_peak_min_ratio_global=0.0,
    d_event_min=0.0,
    rb_tol=0.02,
    out_dir="outputs",
    events_txt="local_instability_events_post.txt",
    curves_txt="local_instability_curves_post.txt",
    plot_member_curves=True,
    plot_member_curves_max=None,
    plot_member_curves_sort_by="step",
    plot_member_curves_normalized=True,
    plot_member_curves_add_raw=True,
    plot_energy_maps=False,
    energy_map_every=10,
    energy_map_show_deformed=False,
    energy_map_scale_factor=1.0,
    ele_data=None,
    show_plots=False,
):
    if dU_eps is None:
        dU_eps = 0.0 if U_eps is None else U_eps

    active_methods = tuple(str(x) for x in active_methods) if active_methods is not None else ("pdrop_dend", "pdrop_dmid", "energy")
    by_member = _build_member_series(hist, N_use=N_use)
    events = []
    events_by_member = {}
    unstable_ele_tags = set()

    global_member_peaks = [float(c.get("N_max", np.nan)) for c in by_member.values()]
    global_member_peaks = [v for v in global_member_peaks if np.isfinite(v)]
    N_peak_global = max(global_member_peaks) if global_member_peaks else np.nan

    for member_id, c in by_member.items():
        N_peak = float(c.get("N_max", np.nan))
        if not np.isfinite(N_peak) or N_peak <= 0.0:
            continue
        peak_threshold = float(N_peak_min_abs)
        if np.isfinite(N_peak_global) and float(N_peak_min_ratio_global) > 0.0:
            peak_threshold = max(peak_threshold, float(N_peak_min_ratio_global) * float(N_peak_global))
        if N_peak < peak_threshold:
            continue

        method_map = {
            "pdrop_dend": _detect_pdrop(c["N_member"], c["d_end"], drop_ratio=drop_ratio, drop_consec=drop_consec, d_tol=d_tol, d_event_min=d_event_min),
            "pdrop_dmid": _detect_pdrop(c["N_member"], c["d_mid"], drop_ratio=drop_ratio, drop_consec=drop_consec, d_tol=d_tol, d_event_min=d_event_min),
            "energy": _detect_energy(c["U_local"], c["dU_local"], dU_eps=dU_eps, energy_consec=energy_consec),
        }

        event_idx, methods = _combine_method_events(method_map, decision_rule=decision_rule, active_methods=active_methods)
        if event_idx is None:
            continue

        d_end_ev = float(c["d_end"][event_idx]) if np.isfinite(c["d_end"][event_idx]) else np.nan
        d_mid_ev = float(c["d_mid"][event_idx]) if np.isfinite(c["d_mid"][event_idx]) else np.nan
        if max(d_end_ev if np.isfinite(d_end_ev) else -np.inf, d_mid_ev if np.isfinite(d_mid_ev) else -np.inf) < float(d_event_min):
            continue

        mode = _classify_buckling_mode(c["d_end"], c["d_mid"], event_idx, d_tol=d_tol)
        event_alpha = _event_alpha_for_methods(method_map, methods, event_idx)
        ev = {
            "member": member_id,
            "event_idx": int(event_idx),
            "event_alpha": float(event_alpha),
            "step": int(c["step"][event_idx]),
            "step_exact": _interp_segment_value(c["step"], event_idx, event_alpha),
            "LF": _interp_segment_value(c["LF"], event_idx, event_alpha),
            "u": _interp_segment_value(c["u"], event_idx, event_alpha),
            "Uz": _interp_segment_value(c["Uz"], event_idx, event_alpha),
            "methods": list(methods),
            "buckling_mode": mode,
            "N_peak": float(N_peak),
            "N_event": _interp_segment_value(c["N_member"], event_idx, event_alpha),
            "d_end": _interp_segment_value(c["d_end"], event_idx, event_alpha),
            "d_mid": _interp_segment_value(c["d_mid"], event_idx, event_alpha),
            "ele_tags": list(c.get("ele_tags", [])),
        }
        events.append(ev)
        events_by_member[member_id] = ev
        unstable_ele_tags.update(int(x) for x in c.get("ele_tags", []))

    events.sort(key=lambda x: (x["step"], str(x["member"])))

    step_to_hist = {int(h.get("step", -1)): h for h in hist}
    for h in hist:
        h["local_instab_new"] = []
        h["local_instab_any"] = False
        h["local_instab_events"] = []
    for ev in events:
        h = step_to_hist.get(int(ev["step"]))
        if h is None:
            continue
        if ev["member"] not in h["local_instab_new"]:
            h["local_instab_new"].append(ev["member"])
        h["local_instab_any"] = True
        for reason in ev["methods"]:
            h["local_instab_events"].append({
                "member": ev["member"],
                "reason": str(reason),
                "buckling_mode": ev["buckling_mode"],
                "Nmax": ev["N_peak"],
                "N_at": ev["N_event"],
            })

    curves_payload = {}
    for member_id, ev in events_by_member.items():
        curves_payload[member_id] = by_member[member_id]
        curves_payload[member_id]["event_idx"] = int(ev["event_idx"])
        curves_payload[member_id]["event_alpha"] = float(ev["event_alpha"])
        curves_payload[member_id]["event_step_exact"] = float(ev["step_exact"])
        curves_payload[member_id]["event_LF"] = float(ev["LF"])
        curves_payload[member_id]["event_u"] = float(ev["u"])
        curves_payload[member_id]["event_Uz"] = float(ev["Uz"])
        curves_payload[member_id]["event_d_end"] = float(ev["d_end"])
        curves_payload[member_id]["event_d_mid"] = float(ev["d_mid"])
        curves_payload[member_id]["event_N"] = float(ev["N_event"])
        d_end_max = _safe_float(by_member[member_id].get("d_end_max", np.nan))
        d_mid_max = _safe_float(by_member[member_id].get("d_mid_max", np.nan))
        n_max = _safe_float(by_member[member_id].get("N_max", np.nan))
        curves_payload[member_id]["event_d_end_norm"] = (
            float(ev["d_end"]) / d_end_max if np.isfinite(d_end_max) and abs(d_end_max) > 0.0 else np.nan
        )
        curves_payload[member_id]["event_d_mid_norm"] = (
            float(ev["d_mid"]) / d_mid_max if np.isfinite(d_mid_max) and abs(d_mid_max) > 0.0 else np.nan
        )
        curves_payload[member_id]["event_N_norm"] = (
            float(ev["N_event"]) / n_max if np.isfinite(n_max) and abs(n_max) > 0.0 else np.nan
        )
        curves_payload[member_id]["buckling_mode"] = ev["buckling_mode"]
        curves_payload[member_id]["methods"] = list(ev["methods"])

    if hist:
        hist[0]["local_post_nd"] = curves_payload

    conv_hist = _converged_local_hist(hist)
    lf_conv = [float(h.get("LF", np.nan)) for h in conv_hist if np.isfinite(_safe_float(h.get("LF", np.nan)))]
    uz_conv = [abs(float(h.get("Uz", np.nan))) for h in conv_hist if np.isfinite(_safe_float(h.get("Uz", np.nan)))]
    lf_max = float(np.nanmax(lf_conv)) if lf_conv else np.nan
    uz_max = float(np.nanmax(uz_conv)) if uz_conv else np.nan
    for member_id in curves_payload.keys():
        curves_payload[member_id]["event_LF_norm"] = (
            float(curves_payload[member_id]["event_LF"]) / lf_max if np.isfinite(lf_max) and lf_max > 0.0 else np.nan
        )
        curves_payload[member_id]["event_Uz_norm"] = (
            abs(float(curves_payload[member_id]["event_Uz"])) / uz_max if np.isfinite(uz_max) and uz_max > 0.0 else np.nan
        )

    Ps = float(np.nanmax(lf_conv)) if lf_conv else np.nan
    if events:
        first_event = min(events, key=lambda x: (float(x.get("step_exact", x["step"])), str(x["member"])))
        Pm = float(first_event["LF"])
        Rb = (Pm / Ps) if (np.isfinite(Pm) and np.isfinite(Ps) and abs(Ps) > 0.0) else np.nan
    else:
        first_event = None
        Pm = np.nan
        Rb = np.nan

    instability_pattern = None
    if np.isfinite(Rb):
        instability_pattern = "synchronous" if abs(Rb - 1.0) <= float(rb_tol) else ("progressive" if Rb < (1.0 - float(rb_tol)) else "synchronous")

    _ensure_dir(out_dir)
    if events_txt:
        _write_events_txt(os.path.join(out_dir, events_txt), events)
    if curves_txt:
        _write_curves_txt(os.path.join(out_dir, curves_txt), curves_payload)
    if plot_member_curves and curves_payload:
        _plot_member_curves(
            curves_payload,
            events_by_member,
            os.path.join(out_dir, "local_member_curves"),
            max_members=plot_member_curves_max,
            sort_by=plot_member_curves_sort_by,
            normalized=plot_member_curves_normalized,
            add_raw_pd_plots=plot_member_curves_add_raw,
            show=show_plots,
        )
    if plot_energy_maps:
        _plot_energy_maps_from_hist(
            hist,
            ele_data=ele_data,
            out_dir=os.path.join(out_dir, "energy_maps_post"),
            every=energy_map_every,
            show_deformed=energy_map_show_deformed,
            scale_factor=energy_map_scale_factor,
        )

    result = {
        "events": events,
        "curves": curves_payload,
        "unstable_ele_tags": sorted(unstable_ele_tags),
        "first_event_step": None if first_event is None else int(first_event["step"]),
        "first_event_lf": None if first_event is None else float(first_event["LF"]),
        "first_event_u": None if first_event is None else float(first_event["u"]),
        "Pm": None if not np.isfinite(Pm) else float(Pm),
        "Ps": None if not np.isfinite(Ps) else float(Ps),
        "Rb": None if not np.isfinite(Rb) else float(Rb),
        "instability_pattern": instability_pattern,
        "decision_rule": str(decision_rule),
        "active_methods": list(active_methods),
        "N_use": str(N_use),
        "drop_ratio": float(drop_ratio),
        "drop_consec": int(drop_consec),
        "energy_consec": int(energy_consec),
        "dU_eps": float(dU_eps),
        "N_peak_min_abs": float(N_peak_min_abs),
        "N_peak_min_ratio_global": float(N_peak_min_ratio_global),
        "d_event_min": float(d_event_min),
    }

    if isinstance(summary, dict):
        summary["unstable_ele_tags"] = sorted(unstable_ele_tags)
        summary["local_post"] = result
        summary["first_local_buckling_step"] = result["first_event_step"]
        summary["first_local_buckling_lf"] = result["first_event_lf"]
        summary["first_local_buckling_u"] = result["first_event_u"]
        summary["Rb"] = result["Rb"]
        summary["instability_pattern"] = result["instability_pattern"]

    return result
