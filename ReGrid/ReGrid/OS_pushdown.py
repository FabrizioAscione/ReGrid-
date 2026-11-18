# file: OS_pushdown_clean.py
# Cleaned version: local-instability detection removed from online solvers.
import time
import numpy as np
import openseespy.opensees as ops
import os, sys
from contextlib import contextmanager
import signal
import csv
import json


try:
    import msvcrt
except Exception:
    msvcrt = None

from OS_fiber_monitor import (
    fiber_dump_step_txt,
    fiber_abs_minmax_metrics,
    fiber_abs_metrics_per_element,   # NEW
)

@contextmanager
def silence_native_output(enabled=True):
    if not enabled:
        yield
        return
    old_stdout_fd = os.dup(1)
    old_stderr_fd = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        yield
    finally:
        os.dup2(old_stdout_fd, 1)
        os.dup2(old_stderr_fd, 2)
        os.close(old_stdout_fd)
        os.close(old_stderr_fd)
        os.close(devnull_fd)


def _headtail(lines, n=10):
    n = int(n)
    if n <= 0:
        return []
    if len(lines) <= 2 * n:
        return lines
    return lines[:n] + ["..."] + lines[-n:]


def _print_ops_block(py_lines, print_ops=True, print_mode="headtail", n_preview=10):
    if not print_ops:
        return
    mode = str(print_mode).lower()
    if mode == "all":
        print("\n".join(py_lines))
    else:
        for ln in _headtail(py_lines, n_preview):
            print(ln)


def _as_writer_add(writer, py_text_or_lines):
    if writer is None or py_text_or_lines is None:
        return
    if isinstance(py_text_or_lines, str):
        writer.add(py_text_or_lines)
    else:
        writer.add("\n".join(py_text_or_lines))


class _ForceStepRecovery:
    """Retry from committed states, with optional iteration-based step adaptation.

    Adaptive mode tries alternate algorithms before halving the increment, down
    to a bounded minimum. Slow successful steps also reduce the next increment;
    only consecutive easy steps allow growth, retaining the working algorithm.
    The default preserves the legacy explicit-fallback policy for other callers.
    """

    def __init__(self, *, constraints, numberer, system, test, algorithm, analysis,
                 dLambda, fallbacks, growth_steps=5, growth_factor=2.0, verbose=True,
                 adaptive=False, min_increment=1e-6, cutback_factor=0.5,
                 easy_iterations=8, slow_iterations=16, max_attempts=60,
                 log_prefix=None, should_abort=None):
        if not np.isfinite(dLambda) or float(dLambda) == 0.0:
            raise ValueError("dLambda must be finite and nonzero.")
        if int(growth_steps) < 1 or not np.isfinite(growth_factor) or float(growth_factor) <= 1.0:
            raise ValueError("Fallback growth requires positive growth_steps and growth_factor > 1.")
        self.constraints, self.numberer, self.analysis = constraints, numberer, analysis
        self.base = dict(system=system, test=tuple(test), algorithm=tuple(algorithm), dLambda=float(dLambda))
        self.active = dict(self.base)
        self.configured = None
        self.fallbacks = list(fallbacks or [])
        self.growth_steps, self.growth_factor = int(growth_steps), float(growth_factor)
        self.verbose = bool(verbose)
        self.success_streak = 0
        self.attempts, self.last_attempts = [], []
        self.last_increment = float(dLambda)
        self.recovered_steps = 0
        self.adaptive = bool(adaptive)
        self.min_increment = float(min_increment)
        self.cutback_factor = float(cutback_factor)
        self.easy_iterations, self.slow_iterations = int(easy_iterations), int(slow_iterations)
        self.max_attempts = int(max_attempts)
        if self.adaptive and (
            not np.isfinite(self.min_increment) or not 0 < self.min_increment <= abs(dLambda)
            or not np.isfinite(self.cutback_factor) or not 0 < self.cutback_factor < 1
            or not 0 < self.easy_iterations < self.slow_iterations or self.max_attempts < 1
        ):
            raise ValueError("Invalid adaptive force-step limits.")
        self.stop_reason = None
        self.should_abort = should_abort
        self.attempt_context = {}
        self.decisions = []
        self.log_prefix = os.fspath(log_prefix) if log_prefix is not None else None
        self.log_fields = ["attempt", "step", "phase", "cutback_level", "fallback", "ok",
                           "time_before", "time_after", "system", "algorithm", "test", "dLambda",
                           "iterations", "final_norm", "residual_norms", "elapsed_s", "error"]
        if self.log_prefix:
            os.makedirs(os.path.dirname(os.path.abspath(self.log_prefix)), exist_ok=True)
            with open(self.log_prefix + "_attempts.jsonl", "w", encoding="utf-8"):
                pass
            with open(self.log_prefix + "_attempts.csv", "w", newline="", encoding="utf-8") as stream:
                csv.DictWriter(stream, fieldnames=self.log_fields).writeheader()
            self._save_status("running")

    def _save_status(self, status, **extra):
        if not self.log_prefix:
            return
        data = dict(status=status, adaptive=self.adaptive, base=self.base, fallbacks=self.fallbacks,
                    min_increment=self.min_increment, cutback_factor=self.cutback_factor,
                    easy_iterations=self.easy_iterations, slow_iterations=self.slow_iterations,
                    growth_steps=self.growth_steps, growth_factor=self.growth_factor,
                    max_attempts_per_step=self.max_attempts, attempts=len(self.attempts),
                    recovered_steps=self.recovered_steps, next_configuration=self.active,
                    stop_reason=self.stop_reason, step_decisions=self.decisions)
        data.update(extra)
        with open(self.log_prefix + "_summary.json", "w", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2)

    def _save_attempt(self, record):
        if not self.log_prefix:
            return
        # Close each write: failed attempts remain available even if the run is interrupted.
        with open(self.log_prefix + "_attempts.jsonl", "a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        row = dict(record)
        for key in ("algorithm", "test", "residual_norms"):
            row[key] = json.dumps(row[key])
        with open(self.log_prefix + "_attempts.csv", "a", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=self.log_fields).writerow(row)

    def _configure(self, cfg):
        self.configured = None
        ops.wipeAnalysis()
        ops.constraints(self.constraints)
        ops.numberer(self.numberer)
        ops.system(cfg["system"])
        ops.test(cfg["test"][0], *cfg["test"][1:])
        ops.algorithm(cfg["algorithm"][0], *cfg["algorithm"][1:])
        ops.integrator("LoadControl", cfg["dLambda"])
        ops.analysis(self.analysis)
        self.configured = dict(cfg)

    def _attempt(self, cfg, step, fallback):
        started = time.perf_counter()
        time_before = float(ops.getTime())
        error = None
        try:
            if cfg != self.configured:
                self._configure(cfg)
            with silence_native_output(True):
                ok = int(ops.analyze(1))
        except Exception as exc:
            ok, error = -1, str(exc)
        time_after = float(ops.getTime())
        # StaticAnalysis restores the last committed domain after a failure.
        # Never retry against a trial load state that was not rolled back.
        bad_rollback = ok != 0 and not np.isclose(time_after, time_before, rtol=0.0, atol=1e-12)
        if bad_rollback:
            error = f"Failed force step did not restore committed time: {time_before} -> {time_after}."
        try:
            iterations = int(ops.testIter())
            norms = [float(x) for x in ops.testNorms()][:iterations]
        except Exception:
            iterations, norms = None, []
        record = dict(attempt=len(self.attempts) + 1, step=int(step),
                      phase=self.attempt_context.get("phase", "fallback" if fallback else "active"),
                      cutback_level=self.attempt_context.get("cutback_level", 0),
                      fallback=bool(fallback), ok=ok,
                      time_before=time_before, time_after=time_after,
                      system=cfg["system"], algorithm=list(cfg["algorithm"]), test=list(cfg["test"]),
                      dLambda=cfg["dLambda"], iterations=iterations, residual_norms=norms,
                      final_norm=norms[-1] if norms else None,
                      elapsed_s=time.perf_counter() - started, error=error)
        self.attempts.append(record)
        self.last_attempts.append(record)
        self._save_attempt(record)
        if bad_rollback:
            self.stop_reason = "rollback_error"
            self._save_status("error")
            raise RuntimeError(error)
        if self.verbose and (fallback or ok != 0):
            final_norm = norms[-1] if norms else None
            print(f"# GMNA RECOVERY | step={step} | algorithm={cfg['algorithm']} "
                  f"| dLambda={cfg['dLambda']:.6g} | ok={ok} | iterations={iterations} "
                  f"| final_norm={final_norm}", flush=True)
        return ok

    def take_step(self, step):
        if self.adaptive:
            return self._take_adaptive_step(step)
        self.last_attempts = []
        current = dict(self.active)
        ok = self._attempt(current, step, False)
        recovered = False
        if ok != 0:
            self.success_streak = 0
            seen = {(current["system"], current["test"], current["algorithm"], current["dLambda"])}
            for fb in self.fallbacks:
                cfg = dict(current)
                for name in ("system", "test", "algorithm"):
                    if name in fb:
                        cfg[name] = tuple(fb[name]) if name != "system" else fb[name]
                cfg["dLambda"] = float(fb.get("dLambda", current["dLambda"]))
                if not np.isfinite(cfg["dLambda"]) or cfg["dLambda"] * self.base["dLambda"] <= 0:
                    raise ValueError("Fallback dLambda must be finite and have the same sign as the base step.")
                if abs(cfg["dLambda"]) > abs(current["dLambda"]):
                    continue
                if (cfg["system"] == current["system"] and cfg["algorithm"] == current["algorithm"]
                        and cfg["test"][:2] == current["test"][:2]
                        and cfg["test"][2] <= current["test"][2]
                        and cfg["dLambda"] == current["dLambda"]):
                    continue
                key = (cfg["system"], cfg["test"], cfg["algorithm"], cfg["dLambda"])
                if key in seen:
                    continue
                seen.add(key)
                ok = self._attempt(cfg, step, True)
                if ok == 0:
                    current, recovered = cfg, True
                    self.recovered_steps += 1
                    break
        self.last_increment = self.last_attempts[-1]["dLambda"]
        if ok == 0:
            self.active = dict(current)
            self.success_streak += 1
            if self.fallbacks and not recovered and self.success_streak >= self.growth_steps and self.active != self.base:
                cfg = dict(self.base)
                magnitude = min(abs(self.base["dLambda"]), abs(current["dLambda"]) * self.growth_factor)
                cfg["dLambda"] = np.copysign(magnitude, self.base["dLambda"])
                self.active, self.success_streak = cfg, 0
        return ok

    def _take_adaptive_step(self, step):
        self.last_attempts = []
        self.stop_reason = None
        current = dict(self.active)
        magnitude = abs(current["dLambda"])
        seen = set()
        level = 0
        recovered = False
        while True:
            signed_step = float(np.copysign(magnitude, self.base["dLambda"]))
            preferred = dict(current, dLambda=signed_step)
            # Also retry plain Newton at smaller increments, even if a different
            # algorithm was retained from the preceding committed step.
            candidates = [preferred, dict(self.base, dLambda=signed_step)]
            for fb in self.fallbacks:
                cfg = dict(preferred)
                for name in ("system", "test", "algorithm"):
                    if name in fb:
                        cfg[name] = tuple(fb[name]) if name != "system" else fb[name]
                if "dLambda" in fb:
                    value = float(fb["dLambda"])
                    if not np.isfinite(value) or value * self.base["dLambda"] <= 0:
                        raise ValueError("Fallback dLambda must be finite and have the same sign as the base step.")
                    if not self.min_increment <= abs(value) <= magnitude:
                        continue
                    cfg["dLambda"] = value
                candidates.append(cfg)
            for cfg in candidates:
                if self.should_abort is not None and self.should_abort():
                    self.stop_reason = "user_abort"
                    ok = -1
                    break
                key = (cfg["system"], cfg["test"], cfg["algorithm"], cfg["dLambda"])
                if key in seen:
                    continue
                if len(self.last_attempts) >= self.max_attempts:
                    self.stop_reason = "attempt_limit"
                    break
                seen.add(key)
                self.attempt_context = dict(phase="active" if not self.last_attempts else
                                           ("cutback" if level else "algorithm_retry"), cutback_level=level)
                ok = self._attempt(cfg, step, bool(self.last_attempts))
                if ok == 0:
                    current = cfg
                    recovered = len(self.last_attempts) > 1
                    break
            if ok == 0 or self.stop_reason:
                break
            if magnitude <= self.min_increment:
                self.stop_reason = "minimum_increment_exhausted"
                break
            magnitude = max(self.min_increment, magnitude * self.cutback_factor)
            level += 1
        if self.last_attempts:
            self.last_increment = self.last_attempts[-1]["dLambda"]
        decision = "failed"
        if ok == 0:
            self.recovered_steps += int(recovered)
            self.active = dict(current)
            iterations = self.last_attempts[-1]["iterations"]
            if iterations is not None and iterations >= self.slow_iterations:
                self.success_streak = 0
                self.active["dLambda"] = float(np.copysign(
                    max(self.min_increment, abs(current["dLambda"]) * self.cutback_factor), self.base["dLambda"]))
                decision = "reduce_after_slow_convergence"
            elif not recovered and iterations is not None and iterations <= self.easy_iterations:
                self.success_streak += 1
                decision = "hold"
                if self.success_streak >= self.growth_steps:
                    self.active["dLambda"] = float(np.copysign(
                        min(abs(self.base["dLambda"]), abs(current["dLambda"]) * self.growth_factor), self.base["dLambda"]))
                    self.success_streak = 0
                    decision = "grow_after_easy_convergence"
            else:
                self.success_streak = 0
                decision = "hold_after_recovery" if recovered else "hold"
        else:
            self.success_streak = 0
        self.decisions.append(dict(step=int(step), decision=decision, ok=ok,
                                   dLambda_used=self.last_increment,
                                   dLambda_next=self.active["dLambda"],
                                   algorithm_next=list(self.active["algorithm"]),
                                   attempts=len(self.last_attempts), stop_reason=self.stop_reason))
        self._save_status("running" if ok == 0 else "nonconvergence")
        return ok


def _get_total_vertical_load(loads_dict, z_index=2):
    if not loads_dict:
        return 0.0
    tot = 0.0
    for v in loads_dict.values():
        if isinstance(v, (tuple, list, np.ndarray)):
            if len(v) <= z_index:
                raise ValueError(f"Load tuple has no z component at index {z_index}: {v}")
            tot += float(v[z_index])
        else:
            tot += float(v)
    return float(tot)


def _infer_control_node_from_elastic_res(elastic_res, prefer="minUz"):
    if elastic_res is None:
        return None
    try:
        if prefer == "maxUz":
            return int(elastic_res["max_vertical"]["max"]["node"])
        return int(elastic_res["max_vertical"]["min"]["node"])
    except Exception:
        return None


def _node_disp_safe(nid: int):
    try:
        ux = float(ops.nodeDisp(int(nid), 1))
    except Exception:
        ux = 0.0
    try:
        uy = float(ops.nodeDisp(int(nid), 2))
    except Exception:
        uy = 0.0
    try:
        uz = float(ops.nodeDisp(int(nid), 3))
    except Exception:
        uz = 0.0
    return ux, uy, uz


def _make_keyboard_abort_controller(keyboard_abort=True, ctrlc_abort=True, stop_keys=(b"\x1b", b"q", b"Q")):
    abort = {"flag": False}
    old_sigint = None

    if bool(ctrlc_abort):
        def _sigint_handler(sig, frame):
            abort["flag"] = True
            print("\n# Ctrl+C received -> stop requested (will stop after current step).")
        try:
            old_sigint = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, _sigint_handler)
        except Exception:
            old_sigint = None

    def poll():
        if not keyboard_abort:
            return
        if msvcrt is None:
            return
        try:
            if msvcrt.kbhit():
                ch = msvcrt.getch()
                if ch in stop_keys:
                    abort["flag"] = True
                    print("\n# Key stop requested (ESC/Q) -> stopping after current step.")
        except Exception:
            pass

    def restore():
        if old_sigint is not None:
            try:
                signal.signal(signal.SIGINT, old_sigint)
            except Exception:
                pass

    return abort, poll, restore


def _infer_eps_lim(fy=None, E=None, eps_lim=None):
    if eps_lim is not None:
        return float(eps_lim)
    if fy is None or E is None:
        return None
    fy = float(fy); E = float(E)
    if E == 0.0:
        return None
    return float(fy / E)


def _ensure_dir(path: str):
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass


# =========================================================================================
# LOCAL RESPONSE RECORDER (for post-process member buckling checks)
# =========================================================================================
def _as_int_list(values):
    out = []
    if values is None:
        return out
    if isinstance(values, (int, np.integer, str)):
        values = [values]
    for v in values:
        try:
            out.append(int(v))
        except Exception:
            continue
    return out


def _safe_ele_nodes(ele_tag):
    try:
        return [int(x) for x in ops.eleNodes(int(ele_tag))]
    except Exception:
        return []


def _extract_member_ele_tags(member_info):
    if isinstance(member_info, (list, tuple)):
        return _as_int_list(member_info)
    if not isinstance(member_info, dict):
        return []
    # Accept both the new recorder keys and the legacy keys returned by hinge_radau/fb_H_ele.
    for key in (
        "ele_tags", "sub_ele_tags", "subelement_tags", "subelements",
        "elements", "eles", "elems",
        "subEles",  # legacy local-instability map
    ):
        if key in member_info:
            tags = _as_int_list(member_info.get(key))
            if tags:
                return tags
    return []


def _extract_member_nodes_explicit(member_info):
    if not isinstance(member_info, dict):
        return None, None, None
    end_nodes = member_info.get("end_nodes", None)
    if end_nodes is None:
        end_nodes = member_info.get("member_end_nodes", None)

    ni = nj = nk = None
    if isinstance(end_nodes, (list, tuple)) and len(end_nodes) >= 2:
        try:
            ni = int(end_nodes[0]); nj = int(end_nodes[1])
        except Exception:
            ni = nj = None

    # Accept legacy keys from hinge_radau/fb_H_ele local_instability map.
    if ni is None and "i" in member_info:
        try:
            ni = int(member_info.get("i"))
        except Exception:
            ni = None
    if nj is None and "j" in member_info:
        try:
            nj = int(member_info.get("j"))
        except Exception:
            nj = None

    for key in ("mid_node", "mid_node_tag", "member_mid_node", "center_node", "centre_node", "m"):
        if key in member_info:
            try:
                nk = int(member_info.get(key))
            except Exception:
                nk = None
            break
    return ni, nj, nk


def _chain_nodes_from_ele_tags(ele_tags):
    ele_tags = _as_int_list(ele_tags)
    if not ele_tags:
        return []
    node_pairs = []
    for et in ele_tags:
        nn = _safe_ele_nodes(et)
        if len(nn) >= 2:
            node_pairs.append((int(nn[0]), int(nn[1])))
    if not node_pairs:
        return []
    chain = [node_pairs[0][0], node_pairs[0][1]]
    for a, b in node_pairs[1:]:
        if chain[-1] == a:
            chain.append(b)
        elif chain[-1] == b:
            chain.append(a)
        elif chain[0] == b:
            chain.insert(0, a)
        elif chain[0] == a:
            chain.insert(0, b)
        else:
            if a not in chain:
                chain.append(a)
            if b not in chain:
                chain.append(b)
    uniq = []
    for n in chain:
        if not uniq or uniq[-1] != n:
            uniq.append(n)
    return uniq


def _normalize_local_response_map(ele_data=None, local_response_map=None):
    src_map = local_response_map
    if src_map is None and isinstance(ele_data, dict):
        for key in ("local_instability", "local_response_map", "member_map", "members"):
            src_map = ele_data.get(key, None)
            if src_map is not None:
                break
    norm = {}

    if isinstance(src_map, dict) and src_map:
        iterable = src_map.items()
    else:
        iterable = []
        if isinstance(ele_data, dict):
            ele_tags = _as_int_list(ele_data.get("ele_tags", None))
            iterable = [(int(et), {"ele_tags": [int(et)]}) for et in ele_tags]

    for member_id, member_info in iterable:
        try:
            mid = int(member_id)
        except Exception:
            mid = str(member_id)

        ele_tags = _extract_member_ele_tags(member_info)
        if not ele_tags:
            continue

        ni, nj, nk = _extract_member_nodes_explicit(member_info)
        chain_nodes = _chain_nodes_from_ele_tags(ele_tags)
        if ni is None and chain_nodes:
            ni = int(chain_nodes[0])
        if nj is None and len(chain_nodes) >= 2:
            nj = int(chain_nodes[-1])
        if nk is None and chain_nodes:
            nk = int(chain_nodes[len(chain_nodes) // 2])
        if ni is None or nj is None:
            continue

        norm[mid] = {
            "member_id": mid,
            "ele_tags": ele_tags,
            "end_nodes": (int(ni), int(nj)),
            "mid_node": (None if nk is None else int(nk)),
        }
    return norm


def _node_xyz_disp(nid):
    nid = int(nid)
    try:
        x0, y0, z0 = ops.nodeCoord(nid)
        x0 = float(x0); y0 = float(y0); z0 = float(z0)
    except Exception:
        x0 = y0 = z0 = 0.0
    try:
        ux = float(ops.nodeDisp(nid, 1))
    except Exception:
        ux = 0.0
    try:
        uy = float(ops.nodeDisp(nid, 2))
    except Exception:
        uy = 0.0
    try:
        uz = float(ops.nodeDisp(nid, 3))
    except Exception:
        uz = 0.0
    xyz0 = np.array([x0, y0, z0], dtype=float)
    uvec = np.array([ux, uy, uz], dtype=float)
    xyz = xyz0 + uvec
    return xyz0, uvec, xyz


def _safe_ele_force(ele_tag):
    try:
        vals = ops.eleForce(int(ele_tag))
        return np.array([float(v) for v in vals], dtype=float)
    except Exception:
        return None


def _safe_ele_response(ele_tag, response_type):
    try:
        vals = ops.eleResponse(int(ele_tag), response_type)
        if vals is None:
            return None
        return np.array([float(v) for v in vals], dtype=float)
    except Exception:
        return None


def _axial_metrics_from_force_vector(f):
    if f is None or len(f) < 1:
        return dict(N_i_raw=np.nan, N_j_raw=np.nan, N_rep_raw=np.nan,
                    N_i_abs=np.nan, N_j_abs=np.nan, N_rep_abs=np.nan,
                    N_i_comp=np.nan, N_j_comp=np.nan, N_rep_comp=np.nan)

    fi = float(f[0])
    fj = float(f[6]) if len(f) >= 7 else (-fi)

    # Signed representative axial force.
    # For many OpenSees beam-column responses, a pure axial state gives end actions of opposite sign.
    # Using 0.5*(fj-fi) keeps a consistent signed scalar; abs/comp_pos filters are handled in post-process.
    N_i_raw = -fi
    N_j_raw = fj
    N_rep_raw = 0.5 * (fj - fi)

    return dict(
        N_i_raw=float(N_i_raw),
        N_j_raw=float(N_j_raw),
        N_rep_raw=float(N_rep_raw),
        N_i_abs=float(abs(N_i_raw)),
        N_j_abs=float(abs(N_j_raw)),
        N_rep_abs=float(abs(N_rep_raw)),
        N_i_comp=float(max(0.0, -N_i_raw)),
        N_j_comp=float(max(0.0, -N_j_raw)),
        N_rep_comp=float(max(0.0, -N_rep_raw)),
    )


def _record_local_response_step(norm_local_map, prev_local_state):
    local_response = {}
    DU_ele_map = {}
    next_prev = {}

    for member_id, minfo in norm_local_map.items():
        ele_tags = list(minfo["ele_tags"])
        ni, nj = minfo["end_nodes"]
        nk = minfo.get("mid_node", None)

        _, ui, xi = _node_xyz_disp(ni)
        _, uj, xj = _node_xyz_disp(nj)
        d_end = float(np.linalg.norm(uj - ui))

        d_mid = None
        if nk is not None:
            _, uk, xk = _node_xyz_disp(nk)
            xline_mid = 0.5 * (xi + xj)
            d_mid = float(np.linalg.norm(xk - xline_mid))

        # axial snapshots: first, middle, last subelement
        force_first = _safe_ele_force(ele_tags[0]) if ele_tags else None
        force_last = _safe_ele_force(ele_tags[-1]) if ele_tags else None
        force_mid = _safe_ele_force(ele_tags[len(ele_tags) // 2]) if ele_tags else None

        ax_first = _axial_metrics_from_force_vector(force_first)
        ax_last = _axial_metrics_from_force_vector(force_last)
        ax_mid = _axial_metrics_from_force_vector(force_mid)

        # member-average axial from all available subelements
        rep_raw_vals = []
        rep_abs_vals = []
        rep_comp_vals = []
        member_dU = 0.0
        member_U_prev = 0.0
        prev_member = prev_local_state.get(member_id, {}) if isinstance(prev_local_state, dict) else {}
        if isinstance(prev_member, dict):
            member_U_prev = float(prev_member.get("U_local", 0.0) or 0.0)

        per_ele_prev = prev_member.get("per_ele", {}) if isinstance(prev_member, dict) else {}
        per_ele_next = {}

        for et in ele_tags:
            f = _safe_ele_force(et)
            ax = _axial_metrics_from_force_vector(f)
            if np.isfinite(ax["N_rep_raw"]):
                rep_raw_vals.append(ax["N_rep_raw"])
                rep_abs_vals.append(ax["N_rep_abs"])
                rep_comp_vals.append(ax["N_rep_comp"])

            q = _safe_ele_response(et, "basicForce")
            v = _safe_ele_response(et, "basicDeformation")
            du = 0.0
            prev_ele = per_ele_prev.get(int(et), {}) if isinstance(per_ele_prev, dict) else {}
            q_prev = prev_ele.get("q", None)
            v_prev = prev_ele.get("v", None)
            try:
                if q is not None and v is not None and q_prev is not None and v_prev is not None:
                    if len(q) == len(q_prev) and len(v) == len(v_prev):
                        du = float(0.5 * np.dot((q_prev + q), (v - v_prev)))
            except Exception:
                du = 0.0

            member_dU += float(du)
            DU_ele_map[int(et)] = float(du)
            per_ele_next[int(et)] = {"q": q, "v": v}

        N_member_raw = float(np.nanmean(rep_raw_vals)) if rep_raw_vals else np.nan
        N_member_abs = float(np.nanmean(rep_abs_vals)) if rep_abs_vals else np.nan
        N_member_comp = float(np.nanmean(rep_comp_vals)) if rep_comp_vals else np.nan
        U_local = float(member_U_prev + member_dU)

        local_response[member_id] = {
            "member_id": member_id,
            "ele_tags": ele_tags,
            "end_nodes": (int(ni), int(nj)),
            "mid_node": (None if nk is None else int(nk)),
            "d_end": float(d_end),
            "d_mid": (None if d_mid is None else float(d_mid)),
            "N_i_raw": ax_first["N_i_raw"],
            "N_j_raw": ax_last["N_j_raw"],
            "N_mid_raw": ax_mid["N_rep_raw"],
            "N_i_abs": ax_first["N_i_abs"],
            "N_j_abs": ax_last["N_j_abs"],
            "N_mid_abs": ax_mid["N_rep_abs"],
            "N_i_comp": ax_first["N_i_comp"],
            "N_j_comp": ax_last["N_j_comp"],
            "N_mid_comp": ax_mid["N_rep_comp"],
            "N_member_raw": N_member_raw,
            "N_member_abs": N_member_abs,
            "N_member_comp": N_member_comp,
            "dU_local": float(member_dU),
            "U_local": float(U_local),
        }

        next_prev[member_id] = {
            "per_ele": per_ele_next,
            "U_local": float(U_local),
        }

    return local_response, DU_ele_map, next_prev



# =========================================================================================
# FORCE CONTROL
# =========================================================================================
def run_gmna_force_control(
    G, M,
    ele_data,
    loads_dict,
    ndf=6,
    pattern_tag=1,

    control_node=None,
    control_dof=3,
    elastic_res=None,
    control_node_prefer="minUz",

    dLambda=0.02,
    n_steps=200,
    LF_max=None,
    stop_if_unstable=True,

    constraints="Transformation",
    numberer="RCM",
    system="ProfileSPD",
    test=("NormUnbalance", 1e-10, 2),
    algorithm=("Newton",),
    analysis="Static",

    fallback_strategies=None,

    monitor_fibers=True,
    fiber_ele_tags=None,
    fiber_nIP=None,
    fiber_sec_nums=None,
    T_fiber_ele="Lobatto",
    fiber_strain_col=None,
    fiber_chunk_size=None,

    fiber_area_min=0.0,
    fiber_area_col=2,
    exclude_neutral_axis=True,
    neutral_axis_tol=1e-12,

    dump_fibers=False,
    dump_fibers_txt="fiber_dump.txt",
    dump_every=1,
    dump_include_full_row=True,

    fy=None,
    E=None,
    eps_lim=None,

    print_ops=True,
    print_mode="headtail",
    n_preview=10,
    print_step_info=True,
    print_every=10,
    writer=None,

    live_plot=False,
    live_plot_every=5,
    live_monitor=None,
    keyboard_abort=True,
    ctrlc_abort=True,
    stop_keys=(b"\x1b", b"q", b"Q"),

    record_local_response=False,
    local_response_map=None,
    local_response_verbose=False,
    fallback_growth_steps=5,
    fallback_growth_factor=2.0,
    fallback_verbose=True,
    adaptive_force_step=False,
    force_min_increment=1e-6,
    force_cutback_factor=0.5,
    force_easy_iterations=8,
    force_slow_iterations=16,
    force_max_attempts=60,
    force_log_prefix=None,
):


    if control_node is None:
        control_node = _infer_control_node_from_elastic_res(elastic_res, prefer=control_node_prefer)
    if control_node is None:
        control_node = int(sorted(G["nodes"].keys())[0])

    control_node = int(control_node)
    control_dof = int(control_dof)

    Fref = _get_total_vertical_load(loads_dict)
    eps_lim_val = _infer_eps_lim(fy=fy, E=E, eps_lim=eps_lim)
    # Local-instability arguments are kept only for backward compatibility.

    # ---------------- fiber tags / nIP ----------------
    if monitor_fibers or dump_fibers:
        if fiber_ele_tags is None:
            fiber_ele_tags = ele_data.get("ele_tags", None) if ele_data is not None else None
        if fiber_ele_tags is None:
            fiber_ele_tags = ops.getEleTags()

        if fiber_nIP is None:
            try:
                fiber_nIP = int(ele_data.get("meta", {}).get("n_ip", None))
            except Exception:
                fiber_nIP = None

    norm_local_response_map = None
    prev_local_state = {}
    if bool(record_local_response):
        norm_local_response_map = _normalize_local_response_map(ele_data=ele_data, local_response_map=local_response_map)
        if bool(local_response_verbose):
            print(f"[LOCAL-REC] members tracked = {len(norm_local_response_map)}")

    # ---------------- analysis commands ----------------
    py_lines = []
    py_lines.append("ops.wipeAnalysis()")
    ops.wipeAnalysis()

    py_lines.append(f"ops.constraints('{constraints}')"); ops.constraints(constraints)
    py_lines.append(f"ops.numberer('{numberer}')");       ops.numberer(numberer)
    py_lines.append(f"ops.system('{system}')");           ops.system(system)

    tname = test[0]
    py_lines.append(f"ops.test('{tname}', {', '.join(str(x) for x in test[1:])})")
    ops.test(tname, *test[1:])

    aname = algorithm[0]
    py_lines.append(
        f"ops.algorithm('{aname}'{'' if len(algorithm)==1 else ', ' + ', '.join(str(x) for x in algorithm[1:])})"
    )
    ops.algorithm(aname, *algorithm[1:])

    py_lines.append(f"ops.integrator('LoadControl', {float(dLambda)})")
    ops.integrator("LoadControl", float(dLambda))

    py_lines.append(f"ops.analysis('{analysis}')")
    ops.analysis(analysis)

    _print_ops_block(py_lines, print_ops=print_ops, print_mode=print_mode, n_preview=n_preview)
    _as_writer_add(writer, py_lines)

    # ---------------- run loop ----------------
    hist = []
    t0 = time.time()
    abort_ctl, poll_key, restore_sig = _make_keyboard_abort_controller(
        keyboard_abort=keyboard_abort,
        ctrlc_abort=ctrlc_abort,
        stop_keys=stop_keys
    )

    def _get_LF():
        return float(ops.getLoadFactor(int(pattern_tag)))

    def _get_u_control():
        return float(ops.nodeDisp(control_node, control_dof))

    def _recovery_should_abort():
        poll_key()
        return bool(abort_ctl["flag"])

    recovery = _ForceStepRecovery(
        constraints=constraints, numberer=numberer, system=system, test=test,
        algorithm=algorithm, analysis=analysis, dLambda=dLambda, fallbacks=fallback_strategies,
        growth_steps=fallback_growth_steps, growth_factor=fallback_growth_factor,
        verbose=fallback_verbose,
        adaptive=adaptive_force_step, min_increment=force_min_increment,
        cutback_factor=force_cutback_factor, easy_iterations=force_easy_iterations,
        slow_iterations=force_slow_iterations, max_attempts=force_max_attempts,
        log_prefix=force_log_prefix,
        should_abort=_recovery_should_abort,
    )
    plastic_onset_set = set()
    plastic_full_set = set()

    termination_reason = "step_limit"
    for step in range(int(n_steps) + 1):
        ok = 0

        poll_key()
        if abort_ctl["flag"]:
            termination_reason = "user_abort"
            break

        if step > 0:
            ok = recovery.take_step(step)

        poll_key()
        if abort_ctl["flag"]:
            termination_reason = "user_abort"
            break

        try:
            lf = _get_LF()
        except Exception:
            lf = 0.0

        u = _get_u_control()
        ux, uy, uz = _node_disp_safe(control_node)
        Ftot = lf * Fref

        # --- dump fibres ---
        if dump_fibers and (step > 0) and (fiber_nIP is not None):
            if (int(step) % int(dump_every)) == 0:
                fiber_dump_step_txt(
                    txt_path=str(dump_fibers_txt),
                    step=int(step),
                    lf=float(lf),
                    ele_tags=fiber_ele_tags,
                    nIP=int(fiber_nIP),
                    T_fiber_ele=str(T_fiber_ele),
                    sec_num_list=fiber_sec_nums,
                    fiber_chunk_size=fiber_chunk_size,
                    fiber_strain_col=fiber_strain_col,
                    include_full_row=bool(dump_include_full_row),
                    write_header_if_new=True,
                )

        # --- ABS metrics GLOBAL ---
        eps_abs_max = eps_abs_minmax = None
        info_abs_max = info_abs_minmax = None
        if monitor_fibers and (step > 0) and (fiber_nIP is not None):
            eps_abs_max, eps_abs_minmax, info_abs_max, info_abs_minmax = fiber_abs_minmax_metrics(
                ele_tags=fiber_ele_tags,
                nIP=int(fiber_nIP),
                T_fiber_ele=str(T_fiber_ele),
                sec_num_list=fiber_sec_nums,
                fiber_strain_col=fiber_strain_col,
                fiber_chunk_size=fiber_chunk_size,
                area_col=int(fiber_area_col) if fiber_area_col is not None else None,
                area_min=float(fiber_area_min),
                exclude_neutral_axis=bool(exclude_neutral_axis),
                neutral_axis_tol=float(neutral_axis_tol),
            )

        # ================= PER-ELEMENT plasticity flags =================
        new_onset = []
        new_full = []
        plastic_onset_any = False
        plastic_full_any = False

        if monitor_fibers and (step > 0) and (fiber_nIP is not None) and (eps_lim_val is not None):
            ele_abs_max_map, ele_min_abs_map, _, _ = fiber_abs_metrics_per_element(
                ele_tags=fiber_ele_tags,
                nIP=int(fiber_nIP),
                T_fiber_ele=str(T_fiber_ele),
                sec_num_list=fiber_sec_nums,
                fiber_strain_col=fiber_strain_col,
                fiber_chunk_size=fiber_chunk_size,
                area_col=int(fiber_area_col) if fiber_area_col is not None else None,
                area_min=float(fiber_area_min),
                exclude_neutral_axis=bool(exclude_neutral_axis),
                neutral_axis_tol=float(neutral_axis_tol),
            )

            for e, v in ele_abs_max_map.items():
                if float(v) >= float(eps_lim_val) and (int(e) not in plastic_onset_set) and (int(e) not in plastic_full_set):
                    new_onset.append(int(e))

            for e, v in ele_min_abs_map.items():
                if float(v) >= float(eps_lim_val) and (int(e) not in plastic_full_set):
                    new_full.append(int(e))

            if new_onset:
                plastic_onset_any = True
                for e in new_onset:
                    plastic_onset_set.add(int(e))

            if new_full:
                plastic_full_any = True
                for e in new_full:
                    plastic_full_set.add(int(e))
                    if int(e) in plastic_onset_set:
                        plastic_onset_set.discard(int(e))
        newly_local = []
        local_response = None
        DU_ele_map = None
        if bool(record_local_response) and (norm_local_response_map is not None) and (ok == 0):
            local_response, DU_ele_map, prev_local_state = _record_local_response_step(
                norm_local_response_map, prev_local_state
            )

        rec = {
            "step": int(step),
            "ok": int(ok),
            "dLambda_used": float(recovery.last_increment) if step > 0 else 0.0,
            "step_attempts": list(recovery.last_attempts) if step > 0 else [],
            "LF": float(lf),
            "Ftot": float(Ftot),
            "control_node": int(control_node),
            "control_dof": int(control_dof),
            "u": float(u),
            "Ux": float(ux),
            "Uy": float(uy),
            "Uz": float(uz),

            "eps_abs_max": None if eps_abs_max is None else float(eps_abs_max),
            "eps_abs_minmax": None if eps_abs_minmax is None else float(eps_abs_minmax),
            "info_eps_abs_max": info_abs_max,
            "info_eps_abs_minmax": info_abs_minmax,

            "local_instab_new": newly_local,
            "local_instab_any": bool(newly_local),
            "local_instab_events": [],

            "DU_ele_map": DU_ele_map,
            "local_response": local_response,

            "plastic_onset_new": new_onset,
            "plastic_full_new": new_full,
            "plastic_onset_any": bool(new_onset),
            "plastic_full_any": bool(new_full),
        }
        hist.append(rec)
        # ---------------- prints ----------------
        if print_step_info and (
            (step % int(print_every)) == 0 or ok != 0 or step == 0
            or rec["local_instab_any"] or rec["plastic_onset_any"] or rec["plastic_full_any"]
        ):
            msg = f"# SUMMARY | GMNA [Force Controlled] step={step} | ok={ok} | LF={lf:.6g} | u={u:.6g} | Ftot={Ftot:.6g} kN"
            if eps_abs_max is not None:
                msg += f" | eps_abs_max={eps_abs_max:.3e}"
            if eps_abs_minmax is not None:
                msg += f" | eps_abs_minmax={eps_abs_minmax:.3e}"
            if rec["plastic_onset_any"]:
                msg += f" | PL_ONSET(n={len(new_onset)})"
            if rec["plastic_full_any"]:
                msg += f" | PL_FULL(n={len(new_full)})"
            if rec["local_instab_any"]:
                msg += " | LOCAL_INSTAB=YES"
            print(msg)

        if live_plot and (live_monitor is not None) and ((step % int(live_plot_every)) == 0):
            live_monitor.update(hist)

        if ok != 0 and stop_if_unstable:
            termination_reason = "nonconvergence"
            break
        if LF_max is not None and float(lf) >= float(LF_max):
            termination_reason = "load_limit"
            break
    dt = time.time() - t0

    summary = {
        "n_steps_done": int(len(hist) - 1),
        "ok_last": int(hist[-1]["ok"]) if hist else None,
        "LF_last": float(hist[-1]["LF"]) if hist else None,
        "u_last": float(hist[-1]["u"]) if hist else None,
        "Ftot_last": float(hist[-1]["Ftot"]) if hist else None,
        "runtime_s": float(dt),
        "fallback_recovered_steps": int(recovery.recovered_steps),
        "n_converged_steps": sum(1 for rec in hist if rec["step"] > 0 and rec["ok"] == 0),
        "termination_reason": termination_reason,
        "recovery_stop_reason": recovery.stop_reason,
        "adaptive_step_decisions": list(recovery.decisions),
        "force_log_prefix": recovery.log_prefix,
        "analysis_attempts": list(recovery.attempts),
        "dLambda_next": float(recovery.active["dLambda"]),
        "control_node": int(control_node),
        "control_dof": int(control_dof),
        "Fref": float(Fref),
        "pattern_tag": int(pattern_tag),
        "T_fiber_ele": str(T_fiber_ele),
        "fiber_nIP": None if fiber_nIP is None else int(fiber_nIP),
        "dump_fibers": bool(dump_fibers),
        "dump_fibers_txt": str(dump_fibers_txt) if dump_fibers_txt is not None else None,
        "exclude_neutral_axis": bool(exclude_neutral_axis),
        "neutral_axis_tol": float(neutral_axis_tol),
        "fiber_area_min": float(fiber_area_min),

        "local_response_recorded": bool(record_local_response),
        "local_response_n_members": 0 if norm_local_response_map is None else int(len(norm_local_response_map)),
        "unstable_ele_tags": [],

        "eps_lim": None if eps_lim_val is None else float(eps_lim_val),
        "plastic_onset_ele_tags": sorted([int(x) for x in plastic_onset_set]),
        "plastic_full_ele_tags": sorted([int(x) for x in plastic_full_set]),

    }

    print(
        f"# SUMMARY | GMNA done | steps={summary['n_steps_done']} | ok_last={summary['ok_last']} "
        f"| LF_last={summary['LF_last']:.6g} | runtime={summary['runtime_s']:.2f}s "
        f"| termination={termination_reason} | recovery_stop={recovery.stop_reason}"
    )
    recovery._save_status("finished", termination_reason=termination_reason,
                          n_steps_done=summary["n_steps_done"], LF_last=summary["LF_last"],
                          u_last=summary["u_last"], ok_last=summary["ok_last"], runtime_s=dt)
    restore_sig()
    return hist, summary



# =========================================================================================
# DISPLACEMENT CONTROL
# =========================================================================================
def run_gmna_disp_control(
    G, M,
    ele_data,
    loads_dict,
    ndf=6,
    pattern_tag=1,

    control_node=None,
    control_dof=3,
    elastic_res=None,
    control_node_prefer="minUz",

    dU=-1e-4,
    n_steps=200,
    U_max=None,
    stop_if_unstable=True,

    constraints="Transformation",
    numberer="RCM",
    system="ProfileSPD",
    test=("NormUnbalance", 1e-10, 50),
    algorithm=("Newton",),
    analysis="Static",

    fallback_strategies=None,

    monitor_fibers=True,
    fiber_ele_tags=None,
    fiber_nIP=None,
    fiber_sec_nums=None,
    T_fiber_ele="Lobatto",
    fiber_strain_col=None,
    fiber_chunk_size=None,

    fiber_area_min=0.0,
    fiber_area_col=2,
    exclude_neutral_axis=True,
    neutral_axis_tol=1e-12,

    dump_fibers=False,
    dump_fibers_txt="fiber_dump.txt",
    dump_every=1,
    dump_include_full_row=True,

    fy=None,
    E=None,
    eps_lim=None,

    print_ops=True,
    print_mode="headtail",
    n_preview=10,
    print_step_info=True,
    print_every=10,
    writer=None,

    live_plot=False,
    live_plot_every=5,
    live_monitor=None,
    keyboard_abort=True,
    ctrlc_abort=True,
    stop_keys=(b"\x1b", b"q", b"Q"),

    record_local_response=False,
    local_response_map=None,
    local_response_verbose=False,
):

    if control_node is None:
        control_node = _infer_control_node_from_elastic_res(elastic_res, prefer=control_node_prefer)
    if control_node is None:
        control_node = int(sorted(G["nodes"].keys())[0])

    control_node = int(control_node)
    control_dof = int(control_dof)

    Fref = _get_total_vertical_load(loads_dict)
    eps_lim_val = _infer_eps_lim(fy=fy, E=E, eps_lim=eps_lim)
    # Local-instability arguments are kept only for backward compatibility.

    # ---------------- fibers ----------------
    if monitor_fibers or dump_fibers:
        if fiber_ele_tags is None:
            fiber_ele_tags = ele_data.get("ele_tags", None) if ele_data is not None else None
        if fiber_ele_tags is None:
            fiber_ele_tags = ops.getEleTags()

        if fiber_nIP is None:
            try:
                fiber_nIP = int(ele_data.get("meta", {}).get("n_ip", None))
            except Exception:
                fiber_nIP = None

    norm_local_response_map = None
    prev_local_state = {}
    if bool(record_local_response):
        norm_local_response_map = _normalize_local_response_map(ele_data=ele_data, local_response_map=local_response_map)
        if bool(local_response_verbose):
            print(f"[LOCAL-REC] members tracked = {len(norm_local_response_map)}")

    # ---------------- analysis commands ----------------
    py_lines = []
    py_lines.append("ops.wipeAnalysis()"); ops.wipeAnalysis()
    py_lines.append(f"ops.constraints('{constraints}')"); ops.constraints(constraints)
    py_lines.append(f"ops.numberer('{numberer}')");       ops.numberer(numberer)
    py_lines.append(f"ops.system('{system}')");           ops.system(system)

    tname = test[0]
    py_lines.append(f"ops.test('{tname}', {', '.join(str(x) for x in test[1:])})")
    ops.test(tname, *test[1:])

    aname = algorithm[0]
    py_lines.append(
        f"ops.algorithm('{aname}'{'' if len(algorithm)==1 else ', ' + ', '.join(str(x) for x in algorithm[1:])})"
    )
    ops.algorithm(aname, *algorithm[1:])

    py_lines.append(f"ops.integrator('DisplacementControl', {int(control_node)}, {int(control_dof)}, {float(dU)})")
    ops.integrator("DisplacementControl", int(control_node), int(control_dof), float(dU))

    py_lines.append(f"ops.analysis('{analysis}')")
    ops.analysis(analysis)

    _print_ops_block(py_lines, print_ops=print_ops, print_mode=print_mode, n_preview=n_preview)
    _as_writer_add(writer, py_lines)

    # ---------------- run loop ----------------
    hist = []
    t0 = time.time()
    abort_ctl, poll_key, restore_sig = _make_keyboard_abort_controller(
        keyboard_abort=keyboard_abort,
        ctrlc_abort=ctrlc_abort,
        stop_keys=stop_keys
    )

    def _get_LF():
        return float(ops.getLoadFactor(int(pattern_tag)))

    def _get_u_control():
        return float(ops.nodeDisp(control_node, control_dof))

    def _try_step_with_fallbacks():
        with silence_native_output(True):
            ok = ops.analyze(1)
        if ok == 0:
            return 0
        if not fallback_strategies:
            return ok

        main_cfg = dict(system=system, test=test, algorithm=algorithm)

        for fb in fallback_strategies:
            try:
                ops.wipeAnalysis()
                ops.constraints(constraints)
                ops.numberer(numberer)

                sys_fb = fb.get("system", main_cfg["system"])
                ops.system(sys_fb)

                test_fb = fb.get("test", main_cfg["test"])
                ops.test(test_fb[0], *test_fb[1:])

                alg_fb = fb.get("algorithm", main_cfg["algorithm"])
                ops.algorithm(alg_fb[0], *alg_fb[1:])

                ops.integrator("DisplacementControl", int(control_node), int(control_dof), float(dU))
                ops.analysis(analysis)

                with silence_native_output(True):
                    ok2 = ops.analyze(1)
                if ok2 == 0:
                    return 0
            except Exception:
                continue

        try:
            ops.wipeAnalysis()
            ops.constraints(constraints)
            ops.numberer(numberer)
            ops.system(system)
            ops.test(test[0], *test[1:])
            ops.algorithm(algorithm[0], *algorithm[1:])
            ops.integrator("DisplacementControl", int(control_node), int(control_dof), float(dU))
            ops.analysis(analysis)
        except Exception:
            pass

        return ok
    plastic_onset_set = set()
    plastic_full_set = set()

    for step in range(int(n_steps) + 1):
        ok = 0

        poll_key()
        if abort_ctl["flag"]:
            break

        if step > 0:
            ok = _try_step_with_fallbacks()

        poll_key()
        if abort_ctl["flag"]:
            break

        try:
            lf = _get_LF()
        except Exception:
            lf = 0.0

        u = _get_u_control()
        ux, uy, uz = _node_disp_safe(control_node)
        Ftot = lf * Fref

        if dump_fibers and (step > 0) and (fiber_nIP is not None):
            if (int(step) % int(dump_every)) == 0:
                fiber_dump_step_txt(
                    txt_path=str(dump_fibers_txt),
                    step=int(step),
                    lf=float(lf),
                    ele_tags=fiber_ele_tags,
                    nIP=int(fiber_nIP),
                    T_fiber_ele=str(T_fiber_ele),
                    sec_num_list=fiber_sec_nums,
                    fiber_chunk_size=fiber_chunk_size,
                    fiber_strain_col=fiber_strain_col,
                    include_full_row=bool(dump_include_full_row),
                    write_header_if_new=True,
                )

        eps_abs_max = eps_abs_minmax = None
        info_abs_max = info_abs_minmax = None

        if monitor_fibers and (step > 0) and (fiber_nIP is not None):
            eps_abs_max, eps_abs_minmax, info_abs_max, info_abs_minmax = fiber_abs_minmax_metrics(
                ele_tags=fiber_ele_tags,
                nIP=int(fiber_nIP),
                T_fiber_ele=str(T_fiber_ele),
                sec_num_list=fiber_sec_nums,
                fiber_strain_col=fiber_strain_col,
                fiber_chunk_size=fiber_chunk_size,
                area_col=int(fiber_area_col) if fiber_area_col is not None else None,
                area_min=float(fiber_area_min),
                exclude_neutral_axis=bool(exclude_neutral_axis),
                neutral_axis_tol=float(neutral_axis_tol),
            )

        new_onset = []
        new_full = []

        if monitor_fibers and (step > 0) and (fiber_nIP is not None) and (eps_lim_val is not None):
            ele_abs_max_map, ele_min_abs_map, _, _ = fiber_abs_metrics_per_element(
                ele_tags=fiber_ele_tags,
                nIP=int(fiber_nIP),
                T_fiber_ele=str(T_fiber_ele),
                sec_num_list=fiber_sec_nums,
                fiber_strain_col=fiber_strain_col,
                fiber_chunk_size=fiber_chunk_size,
                area_col=int(fiber_area_col) if fiber_area_col is not None else None,
                area_min=float(fiber_area_min),
                exclude_neutral_axis=bool(exclude_neutral_axis),
                neutral_axis_tol=float(neutral_axis_tol),
            )

            for e, v in ele_abs_max_map.items():
                if float(v) >= float(eps_lim_val) and (int(e) not in plastic_onset_set) and (int(e) not in plastic_full_set):
                    new_onset.append(int(e))

            for e, v in ele_min_abs_map.items():
                if float(v) >= float(eps_lim_val) and (int(e) not in plastic_full_set):
                    new_full.append(int(e))

            for e in new_onset:
                plastic_onset_set.add(int(e))
            for e in new_full:
                plastic_full_set.add(int(e))
                plastic_onset_set.discard(int(e))
        newly_local = []
        local_response = None
        DU_ele_map = None
        if bool(record_local_response) and (norm_local_response_map is not None) and (ok == 0):
            local_response, DU_ele_map, prev_local_state = _record_local_response_step(
                norm_local_response_map, prev_local_state
            )

        rec = {
            "step": int(step),
            "ok": int(ok),
            "LF": float(lf),
            "Ftot": float(Ftot),
            "control_node": int(control_node),
            "control_dof": int(control_dof),
            "u": float(u),
            "Ux": float(ux),
            "Uy": float(uy),
            "Uz": float(uz),

            "eps_abs_max": None if eps_abs_max is None else float(eps_abs_max),
            "eps_abs_minmax": None if eps_abs_minmax is None else float(eps_abs_minmax),
            "info_eps_abs_max": info_abs_max,
            "info_eps_abs_minmax": info_abs_minmax,

            "local_instab_new": newly_local,
            "local_instab_any": bool(newly_local),
            "local_instab_events": [],

            "DU_ele_map": DU_ele_map,
            "local_response": local_response,

            "plastic_onset_new": new_onset,
            "plastic_full_new": new_full,
            "plastic_onset_any": bool(new_onset),
            "plastic_full_any": bool(new_full),
        }
        hist.append(rec)
        if print_step_info and (
            (step % int(print_every)) == 0 or ok != 0 or step == 0
            or rec["local_instab_any"] or rec["plastic_onset_any"] or rec["plastic_full_any"]
        ):
            msg = f"# SUMMARY | GMNA [Displacements] step={step} | ok={ok} | u={u:.6g} | LF={lf:.6g} | Ftot={Ftot:.6g} kN"
            if eps_abs_max is not None:
                msg += f" | eps_abs_max={eps_abs_max:.3e}"
            if eps_abs_minmax is not None:
                msg += f" | eps_abs_minmax={eps_abs_minmax:.3e}"
            if rec["plastic_onset_any"]:
                msg += f" | PL_ONSET(n={len(new_onset)})"
            if rec["plastic_full_any"]:
                msg += f" | PL_FULL(n={len(new_full)})"
            if rec["local_instab_any"]:
                msg += " | LOCAL_INSTAB=YES"
            print(msg)

        if live_plot and (live_monitor is not None) and ((step % int(live_plot_every)) == 0):
            live_monitor.update(hist)

        if ok != 0 and stop_if_unstable:
            break
        if U_max is not None and abs(float(u)) >= float(abs(U_max)):
            break
    dt = time.time() - t0
    summary = {
        "n_steps_done": int(len(hist) - 1),
        "ok_last": int(hist[-1]["ok"]) if hist else None,
        "LF_last": float(hist[-1]["LF"]) if hist else None,
        "u_last": float(hist[-1]["u"]) if hist else None,
        "Ftot_last": float(hist[-1]["Ftot"]) if hist else None,
        "runtime_s": float(dt),
        "control_node": int(control_node),
        "control_dof": int(control_dof),
        "Fref": float(Fref),
        "pattern_tag": int(pattern_tag),
        "dU": float(dU),
        "T_fiber_ele": str(T_fiber_ele),
        "fiber_nIP": None if fiber_nIP is None else int(fiber_nIP),
        "dump_fibers": bool(dump_fibers),
        "dump_fibers_txt": str(dump_fibers_txt) if dump_fibers_txt is not None else None,
        "exclude_neutral_axis": bool(exclude_neutral_axis),
        "neutral_axis_tol": float(neutral_axis_tol),
        "fiber_area_min": float(fiber_area_min),

        "local_response_recorded": bool(record_local_response),
        "local_response_n_members": 0 if norm_local_response_map is None else int(len(norm_local_response_map)),
        "unstable_ele_tags": [],

        "eps_lim": None if eps_lim_val is None else float(eps_lim_val),
        "plastic_onset_ele_tags": sorted([int(x) for x in plastic_onset_set]),
        "plastic_full_ele_tags": sorted([int(x) for x in plastic_full_set]),

    }

    print(f"# SUMMARY | GMNA-DC done | steps={summary['n_steps_done']} | ok_last={summary['ok_last']} "
          f"| u_last={summary['u_last']:.6g} | LF_last={summary['LF_last']:.6g} | runtime={summary['runtime_s']:.2f}s")
    restore_sig()
    return hist, summary


# =========================================================================================
# ARC-LENGTH CONTROL
# =========================================================================================
def run_gmna_arclength_control(
    G, M,
    ele_data,
    loads_dict,
    ndf=6,
    pattern_tag=1,

    control_node=None,
    control_dof=3,
    elastic_res=None,
    control_node_prefer="minUz",

    s=0.05,
    alpha=1.0,
    n_steps=200,
    LF_max=None,
    stop_if_unstable=True,

    constraints="Transformation",
    numberer="RCM",
    system="ProfileSPD",
    test=("NormUnbalance", 1e-10, 50),
    algorithm=("Newton",),
    analysis="Static",

    fallback_strategies=None,

    monitor_fibers=True,
    fiber_ele_tags=None,
    fiber_nIP=None,
    fiber_sec_nums=None,
    T_fiber_ele="Lobatto",
    fiber_strain_col=None,
    fiber_chunk_size=None,

    fiber_area_min=0.0,
    fiber_area_col=2,
    exclude_neutral_axis=True,
    neutral_axis_tol=1e-12,

    dump_fibers=False,
    dump_fibers_txt="fiber_dump.txt",
    dump_every=1,
    dump_include_full_row=True,

    fy=None,
    E=None,
    eps_lim=None,

    print_ops=True,
    print_mode="headtail",
    n_preview=10,
    print_step_info=True,
    print_every=10,
    writer=None,

    live_plot=False,
    live_plot_every=5,
    live_monitor=None,

    integrator_name="ArcLength",
    keyboard_abort=True,
    ctrlc_abort=True,
    stop_keys=(b"\x1b", b"q", b"Q"),
    record_local_response=False,
    local_response_map=None,
    local_response_verbose=False,
):

    if control_node is None:
        control_node = _infer_control_node_from_elastic_res(elastic_res, prefer=control_node_prefer)
    if control_node is None:
        control_node = int(sorted(G["nodes"].keys())[0])

    control_node = int(control_node)
    control_dof = int(control_dof)

    Fref = _get_total_vertical_load(loads_dict)
    eps_lim_val = _infer_eps_lim(fy=fy, E=E, eps_lim=eps_lim)
    # Local-instability arguments are kept only for backward compatibility.

    if monitor_fibers or dump_fibers:
        if fiber_ele_tags is None:
            fiber_ele_tags = ele_data.get("ele_tags", None) if ele_data is not None else None
        if fiber_ele_tags is None:
            fiber_ele_tags = ops.getEleTags()

        if fiber_nIP is None:
            try:
                fiber_nIP = int(ele_data.get("meta", {}).get("n_ip", None))
            except Exception:
                fiber_nIP = None

    norm_local_response_map = None
    prev_local_state = {}
    if bool(record_local_response):
        norm_local_response_map = _normalize_local_response_map(ele_data=ele_data, local_response_map=local_response_map)
        if bool(local_response_verbose):
            print(f"[LOCAL-REC] members tracked = {len(norm_local_response_map)}")

    py_lines = []
    py_lines.append("ops.wipeAnalysis()"); ops.wipeAnalysis()
    py_lines.append(f"ops.constraints('{constraints}')"); ops.constraints(constraints)
    py_lines.append(f"ops.numberer('{numberer}')");       ops.numberer(numberer)
    py_lines.append(f"ops.system('{system}')");           ops.system(system)

    tname = test[0]
    py_lines.append(f"ops.test('{tname}', {', '.join(str(x) for x in test[1:])})")
    ops.test(tname, *test[1:])

    aname = algorithm[0]
    py_lines.append(
        f"ops.algorithm('{aname}'{'' if len(algorithm)==1 else ', ' + ', '.join(str(x) for x in algorithm[1:])})"
    )
    ops.algorithm(aname, *algorithm[1:])

    py_lines.append(f"ops.integrator('{str(integrator_name)}', {float(s)}, {float(alpha)})")
    ops.integrator(str(integrator_name), float(s), float(alpha))

    py_lines.append(f"ops.analysis('{analysis}')"); ops.analysis(analysis)

    _print_ops_block(py_lines, print_ops=print_ops, print_mode=print_mode, n_preview=n_preview)
    _as_writer_add(writer, py_lines)

    hist = []
    t0 = time.time()
    abort_ctl, poll_key, restore_sig = _make_keyboard_abort_controller(
        keyboard_abort=keyboard_abort,
        ctrlc_abort=ctrlc_abort,
        stop_keys=stop_keys
    )

    def _get_LF():
        return float(ops.getLoadFactor(int(pattern_tag)))

    def _get_u_control():
        return float(ops.nodeDisp(control_node, control_dof))

    def _try_step_with_fallbacks():
        with silence_native_output(True):
            ok = ops.analyze(1)
        if ok == 0:
            return 0
        if not fallback_strategies:
            return ok

        main_cfg = dict(system=system, test=test, algorithm=algorithm, integrator_name=str(integrator_name))

        for fb in fallback_strategies:
            try:
                ops.wipeAnalysis()
                ops.constraints(constraints)
                ops.numberer(numberer)

                sys_fb = fb.get("system", main_cfg["system"])
                ops.system(sys_fb)

                test_fb = fb.get("test", main_cfg["test"])
                ops.test(test_fb[0], *test_fb[1:])

                alg_fb = fb.get("algorithm", main_cfg["algorithm"])
                ops.algorithm(alg_fb[0], *alg_fb[1:])

                s_fb = float(fb.get("s", s))
                a_fb = float(fb.get("alpha", alpha))
                integrator_fb = fb.get("integrator_name", main_cfg["integrator_name"])

                ops.integrator(str(integrator_fb), s_fb, a_fb)
                ops.analysis(analysis)

                with silence_native_output(True):
                    ok2 = ops.analyze(1)
                if ok2 == 0:
                    return 0
            except Exception:
                continue

        try:
            ops.wipeAnalysis()
            ops.constraints(constraints)
            ops.numberer(numberer)
            ops.system(system)
            ops.test(test[0], *test[1:])
            ops.algorithm(algorithm[0], *algorithm[1:])
            ops.integrator(str(integrator_name), float(s), float(alpha))
            ops.analysis(analysis)
        except Exception:
            pass

        return ok
    plastic_onset_set = set()
    plastic_full_set = set()

    for step in range(int(n_steps) + 1):
        ok = 0

        poll_key()
        if abort_ctl["flag"]:
            break

        if step > 0:
            ok = _try_step_with_fallbacks()

        poll_key()
        if abort_ctl["flag"]:
            break

        try:
            lf = _get_LF()
        except Exception:
            lf = 0.0

        u = _get_u_control()
        ux, uy, uz = _node_disp_safe(control_node)
        Ftot = lf * Fref

        if dump_fibers and (step > 0) and (fiber_nIP is not None):
            if (int(step) % int(dump_every)) == 0:
                fiber_dump_step_txt(
                    txt_path=str(dump_fibers_txt),
                    step=int(step),
                    lf=float(lf),
                    ele_tags=fiber_ele_tags,
                    nIP=int(fiber_nIP),
                    T_fiber_ele=str(T_fiber_ele),
                    sec_num_list=fiber_sec_nums,
                    fiber_chunk_size=fiber_chunk_size,
                    fiber_strain_col=fiber_strain_col,
                    include_full_row=bool(dump_include_full_row),
                    write_header_if_new=True,
                )

        eps_abs_max = eps_abs_minmax = None
        info_abs_max = info_abs_minmax = None
        if monitor_fibers and (step > 0) and (fiber_nIP is not None):
            eps_abs_max, eps_abs_minmax, info_abs_max, info_abs_minmax = fiber_abs_minmax_metrics(
                ele_tags=fiber_ele_tags,
                nIP=int(fiber_nIP),
                T_fiber_ele=str(T_fiber_ele),
                sec_num_list=fiber_sec_nums,
                fiber_strain_col=fiber_strain_col,
                fiber_chunk_size=fiber_chunk_size,
                area_col=int(fiber_area_col) if fiber_area_col is not None else None,
                area_min=float(fiber_area_min),
                exclude_neutral_axis=bool(exclude_neutral_axis),
                neutral_axis_tol=float(neutral_axis_tol),
            )

        new_onset = []
        new_full = []

        if monitor_fibers and (step > 0) and (fiber_nIP is not None) and (eps_lim_val is not None):
            ele_abs_max_map, ele_min_abs_map, _, _ = fiber_abs_metrics_per_element(
                ele_tags=fiber_ele_tags,
                nIP=int(fiber_nIP),
                T_fiber_ele=str(T_fiber_ele),
                sec_num_list=fiber_sec_nums,
                fiber_strain_col=fiber_strain_col,
                fiber_chunk_size=fiber_chunk_size,
                area_col=int(fiber_area_col) if fiber_area_col is not None else None,
                area_min=float(fiber_area_min),
                exclude_neutral_axis=bool(exclude_neutral_axis),
                neutral_axis_tol=float(neutral_axis_tol),
            )

            for e, v in ele_abs_max_map.items():
                if float(v) >= float(eps_lim_val) and (int(e) not in plastic_onset_set) and (int(e) not in plastic_full_set):
                    new_onset.append(int(e))

            for e, v in ele_min_abs_map.items():
                if float(v) >= float(eps_lim_val) and (int(e) not in plastic_full_set):
                    new_full.append(int(e))

            for e in new_onset:
                plastic_onset_set.add(int(e))
            for e in new_full:
                plastic_full_set.add(int(e))
                plastic_onset_set.discard(int(e))
        newly_local = []
        local_response = None
        DU_ele_map = None
        if bool(record_local_response) and (norm_local_response_map is not None) and (ok == 0):
            local_response, DU_ele_map, prev_local_state = _record_local_response_step(
                norm_local_response_map, prev_local_state
            )

        rec = {
            "step": int(step),
            "ok": int(ok),
            "LF": float(lf),
            "Ftot": float(Ftot),
            "control_node": int(control_node),
            "control_dof": int(control_dof),
            "u": float(u),
            "Ux": float(ux),
            "Uy": float(uy),
            "Uz": float(uz),

            "eps_abs_max": None if eps_abs_max is None else float(eps_abs_max),
            "eps_abs_minmax": None if eps_abs_minmax is None else float(eps_abs_minmax),
            "info_eps_abs_max": info_abs_max,
            "info_eps_abs_minmax": info_abs_minmax,

            "local_instab_new": newly_local,
            "local_instab_any": bool(newly_local),
            "local_instab_events": [],

            "DU_ele_map": DU_ele_map,
            "local_response": local_response,

            "plastic_onset_new": new_onset,
            "plastic_full_new": new_full,
            "plastic_onset_any": bool(new_onset),
            "plastic_full_any": bool(new_full),
        }
        hist.append(rec)
        if print_step_info and (
            (step % int(print_every)) == 0 or ok != 0 or step == 0
            or rec["local_instab_any"] or rec["plastic_onset_any"] or rec["plastic_full_any"]
        ):
            msg = f"# SUMMARY | GMNA [ArchLength] step={step} | ok={ok} | LF={lf:.6g} | u={u:.6g} | Ftot={Ftot:.6g} kN"
            if eps_abs_max is not None:
                msg += f" | eps_abs_max={eps_abs_max:.3e}"
            if eps_abs_minmax is not None:
                msg += f" | eps_abs_minmax={eps_abs_minmax:.3e}"
            if rec["plastic_onset_any"]:
                msg += f" | PL_ONSET(n={len(new_onset)})"
            if rec["plastic_full_any"]:
                msg += f" | PL_FULL(n={len(new_full)})"
            if rec["local_instab_any"]:
                msg += " | LOCAL_INSTAB=YES"
            print(msg)

        if live_plot and (live_monitor is not None) and ((step % int(live_plot_every)) == 0):
            live_monitor.update(hist)

        if ok != 0 and stop_if_unstable:
            break
        if LF_max is not None and float(lf) >= float(LF_max):
            break
    dt = time.time() - t0
    summary = {
        "n_steps_done": int(len(hist) - 1),
        "ok_last": int(hist[-1]["ok"]) if hist else None,
        "LF_last": float(hist[-1]["LF"]) if hist else None,
        "u_last": float(hist[-1]["u"]) if hist else None,
        "Ftot_last": float(hist[-1]["Ftot"]) if hist else None,
        "runtime_s": float(dt),
        "control_node": int(control_node),
        "control_dof": int(control_dof),
        "Fref": float(Fref),
        "pattern_tag": int(pattern_tag),

        "integrator_name": str(integrator_name),
        "s": float(s),
        "alpha": float(alpha),

        "T_fiber_ele": str(T_fiber_ele),
        "fiber_nIP": None if fiber_nIP is None else int(fiber_nIP),
        "dump_fibers": bool(dump_fibers),
        "dump_fibers_txt": str(dump_fibers_txt) if dump_fibers_txt is not None else None,

        "exclude_neutral_axis": bool(exclude_neutral_axis),
        "neutral_axis_tol": float(neutral_axis_tol),
        "fiber_area_min": float(fiber_area_min),

        "local_response_recorded": bool(record_local_response),
        "local_response_n_members": 0 if norm_local_response_map is None else int(len(norm_local_response_map)),
        "unstable_ele_tags": [],

        "eps_lim": None if eps_lim_val is None else float(eps_lim_val),
        "plastic_onset_ele_tags": sorted([int(x) for x in plastic_onset_set]),
        "plastic_full_ele_tags": sorted([int(x) for x in plastic_full_set]),

    }

    print(f"# SUMMARY | GMNA-AL done | steps={summary['n_steps_done']} | ok_last={summary['ok_last']} "
          f"| LF_last={summary['LF_last']:.6g} | runtime={summary['runtime_s']:.2f}s")
    restore_sig()
    return hist, summary



