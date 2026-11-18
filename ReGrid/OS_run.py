import openseespy.opensees as ops
import contextlib
import os
import sys


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

def run_elastic_static_analysis(
    load_control=1.0,
    tol=1e-12,
    max_iter=100,
    analyze_steps=1,
    set_load_const=True,
    constraints="Transformation",
    numberer="AMD",
    system="ProfileSPD",
    algorithm="Linear",
    test_type="NormUnbalance",
    print_ops=True,
    print_mode="all",
    n_preview=10,
    print_summary=False,
):
    quiet_native = (not bool(print_ops)) and (not bool(print_summary))
    with _suppress_native_output(quiet_native):
        ops.wipeAnalysis()
        ops.constraints(constraints)
        ops.numberer(numberer)
        ops.system(system)
        ops.test(test_type, float(tol), int(max_iter))
        ops.algorithm(algorithm)
        ops.integrator("LoadControl", float(load_control))
        ops.analysis("Static")

        ok = ops.analyze(int(analyze_steps))

    if set_load_const:
        ops.loadConst("-time", 0.0)

    py_lines = [
        "ops.wipeAnalysis()",
        f"ops.constraints('{constraints}')",
        f"ops.numberer('{numberer}')",
        f"ops.system('{system}')",
        f"ops.test('{test_type}', {float(tol)}, {int(max_iter)})",
        f"ops.algorithm('{algorithm}')",
        f"ops.integrator('LoadControl', {float(load_control)})",
        "ops.analysis('Static')",
        f"ops.analyze({int(analyze_steps)})  # -> {ok}",
    ]
    if set_load_const:
        py_lines.append("ops.loadConst('-time', 0.0)")

    py_text = "\n".join(py_lines)

    if print_ops:
        if str(print_mode).lower() == "all":
            print(py_text)
        else:
            total = len(py_lines)
            print(f"# --- OpenSees commands generated: {total} lines ---")
            for ln in py_lines[:n_preview]:
                print(ln)

    if print_summary:
        print(f"# SUMMARY | elastic static analyze ok={ok} | steps={int(analyze_steps)} | dLambda={float(load_control)}")

    return py_text, ok
