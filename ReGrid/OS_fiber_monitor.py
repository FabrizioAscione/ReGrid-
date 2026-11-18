# file: OS_fiber_monitor.py
import numpy as np
import openseespy.opensees as ops


def _infer_chunk_size(arr_len: int, candidates=(5, 6, 7, 8, 9, 10, 12, 14, 16)):
    for c in candidates:
        if arr_len % c == 0:
            return int(c)
    return None


def _default_strain_col_from_chunk(chunk: int) -> int:
    return 4 if chunk >= 5 else -1


def _sec_list_from_integration(T_fiber_ele: str, nIP: int):
    if nIP is None or int(nIP) < 1:
        return []
    t = (T_fiber_ele or "").strip().lower()
    if t == "hingeradautwo":
        return sorted({ip for ip in (1, 2, int(nIP)-1, int(nIP)) if 1 <= ip <= int(nIP)})
    if "hingeradau" in t or ("hinge" in t and "radau" in t):
        if int(nIP) == 1:
            return [1]
        return [1, int(nIP)]
    return list(range(1, int(nIP) + 1))


def _ensure_txt_header(path: str, strain_col, include_full_row: bool):
    try:
        with open(path, "r", encoding="utf-8") as f:
            has = (f.read(1) != "")
    except Exception:
        has = False
    if has:
        return

    with open(path, "a", encoding="utf-8") as f:
        f.write("# Fiber dump\n")
        f.write("# columns: step, LF, ele, sec, iFib, y, z")
        if strain_col is not None:
            f.write(f", eps(col={int(strain_col)})")
        if include_full_row:
            f.write(", row[0..chunk-1]")
        f.write("\n")


def fiber_dump_step_txt(
    txt_path: str,
    step: int,
    lf: float,
    ele_tags,
    nIP: int,
    T_fiber_ele: str = "Lobatto",
    sec_num_list=None,
    fiber_chunk_size: int = None,
    fiber_strain_col: int = None,
    include_full_row: bool = True,
    write_header_if_new: bool = True,
):
    if txt_path is None:
        return

    if sec_num_list is None:
        sec_num_list = _sec_list_from_integration(T_fiber_ele, int(nIP))

    if write_header_if_new:
        _ensure_txt_header(str(txt_path), fiber_strain_col, include_full_row)

    lines = []

    for e in ele_tags:
        e = int(e)
        for secNum in sec_num_list:
            secNum = int(secNum)

            try:
                fd = ops.eleResponse(e, "section", secNum, "fiberData")
            except Exception:
                continue

            if fd is None or len(fd) == 0:
                continue

            fd = np.asarray(fd, dtype=float).flatten()

            chunk = int(fiber_chunk_size) if fiber_chunk_size is not None else _infer_chunk_size(fd.size)
            if chunk is None:
                lines.append(
                    f"{int(step)}\t{float(lf):.10e}\t{e}\t{secNum}\t-1\tNaN\tNaN\t# UNK_CHUNK len={fd.size}\n"
                )
                continue

            n_fib = fd.size // chunk
            mat = fd.reshape((n_fib, chunk))

            for i in range(n_fib):
                row = mat[i, :]
                y = float(row[0]) if chunk >= 1 else float("nan")
                z = float(row[1]) if chunk >= 2 else float("nan")

                parts = [
                    str(int(step)),
                    f"{float(lf):.10e}",
                    str(e),
                    str(secNum),
                    str(int(i)),
                    f"{y:.10e}",
                    f"{z:.10e}",
                ]

                if fiber_strain_col is not None and 0 <= int(fiber_strain_col) < chunk:
                    eps = float(row[int(fiber_strain_col)])
                    parts.append(f"{eps:.10e}")

                if include_full_row:
                    parts.extend([f"{float(v):.10e}" for v in row])

                lines.append("\t".join(parts) + "\n")

    with open(str(txt_path), "a", encoding="utf-8") as f:
        f.writelines(lines)


# =========================================================================================
# NEW: per-element ABS metrics (quello che ti serve per blu/rosso GLOBALI)
# =========================================================================================
def fiber_abs_metrics_per_element(
    ele_tags,
    nIP: int,
    T_fiber_ele: str = "Lobatto",
    sec_num_list=None,
    fiber_strain_col: int = None,
    fiber_chunk_size: int = None,

    # filtri robusti
    area_col: int = 2,
    area_min: float = 0.0,

    # esclusione asse neutro per "full plast"
    exclude_neutral_axis: bool = True,
    neutral_axis_tol: float = 1e-12,
):
    """
    Calcola, PER OGNI elemento e:

      ele_abs_max[e] = max(|eps|) su tutte le fibre (tutte le sezioni selezionate)
      ele_min_abs[e] = min(|eps|) su tutte le fibre (tutte le sezioni), con filtri:
                       - area_min
                       - e (opzionale) esclusione NA: |eps| > neutral_axis_tol

    Nota concettuale:
      - onset plast:   ele_abs_max[e] >= eps_lim
      - full plast:    ele_min_abs[e] >= eps_lim   (min(|eps|) esclusa NA)

    Returns:
      ele_abs_max (dict int->float),
      ele_min_abs (dict int->float),
      info_abs_max (dict int->info),
      info_min_abs (dict int->info)

    info format:
      {"ele","sec","y","z","eps","abs_eps"}
    """
    if sec_num_list is None:
        sec_num_list = _sec_list_from_integration(T_fiber_ele, int(nIP))

    ele_abs_max = {}
    ele_min_abs = {}
    info_abs_max = {}
    info_min_abs = {}

    for e in ele_tags:
        e = int(e)

        # init
        max_abs = -1e30
        min_abs = +1e30
        max_info = None
        min_info = None

        for secNum in sec_num_list:
            secNum = int(secNum)

            try:
                fd = ops.eleResponse(e, "section", secNum, "fiberData")
            except Exception:
                continue

            if fd is None or len(fd) == 0:
                continue

            fd = np.asarray(fd, dtype=float).flatten()

            chunk = int(fiber_chunk_size) if fiber_chunk_size is not None else _infer_chunk_size(fd.size)
            if chunk is None:
                continue

            n_fib = fd.size // chunk
            mat = fd.reshape((n_fib, chunk))

            ys = mat[:, 0]
            zs = mat[:, 1]

            sc = int(fiber_strain_col) if fiber_strain_col is not None else _default_strain_col_from_chunk(chunk)
            if sc < 0 or sc >= chunk:
                continue

            eps = mat[:, sc]
            abs_eps = np.abs(eps)

            # ---- FILTRI ----
            mask = np.isfinite(abs_eps)

            if area_min is not None and float(area_min) > 0.0 and area_col is not None and 0 <= int(area_col) < chunk:
                A = mat[:, int(area_col)]
                mask &= np.isfinite(A) & (A > float(area_min))

            # per max_abs NON serve escludere NA; per min_abs sì (se richiesto)
            mask_for_max = mask.copy()
            mask_for_min = mask.copy()
            if exclude_neutral_axis:
                mask_for_min &= (abs_eps > float(neutral_axis_tol))

            # ---- max(|eps|) ----
            if np.any(mask_for_max):
                abs_f = abs_eps[mask_for_max]
                eps_f = eps[mask_for_max]
                ys_f = ys[mask_for_max]
                zs_f = zs[mask_for_max]

                iM = int(np.argmax(abs_f))
                sec_max = float(abs_f[iM])
                if sec_max > max_abs:
                    max_abs = sec_max
                    max_info = {
                        "ele": e, "sec": secNum,
                        "y": float(ys_f[iM]), "z": float(zs_f[iM]),
                        "eps": float(eps_f[iM]), "abs_eps": float(sec_max)
                    }

            # ---- min(|eps|) (con NA exclusion) ----
            if np.any(mask_for_min):
                abs_f = abs_eps[mask_for_min]
                eps_f = eps[mask_for_min]
                ys_f = ys[mask_for_min]
                zs_f = zs[mask_for_min]

                im = int(np.argmin(abs_f))
                sec_min = float(abs_f[im])
                if sec_min < min_abs:
                    min_abs = sec_min
                    min_info = {
                        "ele": e, "sec": secNum,
                        "y": float(ys_f[im]), "z": float(zs_f[im]),
                        "eps": float(eps_f[im]), "abs_eps": float(sec_min)
                    }

        # salva se valido
        if max_info is not None and max_abs > -1e20:
            ele_abs_max[e] = float(max_abs)
            info_abs_max[e] = max_info

        if min_info is not None and min_abs < +1e20:
            ele_min_abs[e] = float(min_abs)
            info_min_abs[e] = min_info

    return ele_abs_max, ele_min_abs, info_abs_max, info_min_abs


# =========================================================================================
# (resta anche la tua funzione globale ABS se vuoi, ma non è più necessaria per blu/rosso)
# =========================================================================================
def fiber_abs_minmax_metrics(
    ele_tags,
    nIP: int,
    T_fiber_ele: str = "Lobatto",
    sec_num_list=None,
    fiber_strain_col: int = None,
    fiber_chunk_size: int = None,
    area_col: int = 2,
    area_min: float = 0.0,
    exclude_neutral_axis: bool = True,
    neutral_axis_tol: float = 1e-12,
):
    if sec_num_list is None:
        sec_num_list = _sec_list_from_integration(T_fiber_ele, int(nIP))

    eps_abs_max_global = -1e30
    info_abs_max = None

    best_ele_min_abs = -1e30
    info_abs_minmax = None

    for e in ele_tags:
        e = int(e)

        ele_min_abs = +1e30
        ele_min_info = None

        for secNum in sec_num_list:
            secNum = int(secNum)

            try:
                fd = ops.eleResponse(e, "section", secNum, "fiberData")
            except Exception:
                continue

            if fd is None or len(fd) == 0:
                continue

            fd = np.asarray(fd, dtype=float).flatten()

            chunk = int(fiber_chunk_size) if fiber_chunk_size is not None else _infer_chunk_size(fd.size)
            if chunk is None:
                continue

            n_fib = fd.size // chunk
            mat = fd.reshape((n_fib, chunk))

            ys = mat[:, 0]
            zs = mat[:, 1]

            sc = int(fiber_strain_col) if fiber_strain_col is not None else _default_strain_col_from_chunk(chunk)
            if sc < 0 or sc >= chunk:
                continue

            eps = mat[:, sc]
            abs_eps = np.abs(eps)

            mask = np.isfinite(abs_eps)

            if area_min is not None and float(area_min) > 0.0 and area_col is not None and 0 <= int(area_col) < chunk:
                A = mat[:, int(area_col)]
                mask &= np.isfinite(A) & (A > float(area_min))

            if exclude_neutral_axis:
                mask &= (abs_eps > float(neutral_axis_tol))

            if not np.any(mask):
                continue

            eps_f = eps[mask]
            abs_f = abs_eps[mask]
            ys_f = ys[mask]
            zs_f = zs[mask]

            i_abs_max = int(np.argmax(abs_f))
            sec_abs_max = float(abs_f[i_abs_max])
            if sec_abs_max > eps_abs_max_global:
                eps_abs_max_global = sec_abs_max
                info_abs_max = {
                    "ele": e, "sec": secNum,
                    "y": float(ys_f[i_abs_max]), "z": float(zs_f[i_abs_max]),
                    "eps": float(eps_f[i_abs_max]),
                    "abs_eps": sec_abs_max
                }

            i_abs_min = int(np.argmin(abs_f))
            sec_abs_min = float(abs_f[i_abs_min])
            if sec_abs_min < ele_min_abs:
                ele_min_abs = sec_abs_min
                ele_min_info = {
                    "ele": e, "sec": secNum,
                    "y": float(ys_f[i_abs_min]), "z": float(zs_f[i_abs_min]),
                    "eps": float(eps_f[i_abs_min]),
                    "abs_eps": sec_abs_min
                }

        if ele_min_info is not None and ele_min_abs < +1e29:
            if ele_min_abs > best_ele_min_abs:
                best_ele_min_abs = ele_min_abs
                info_abs_minmax = ele_min_info

    if eps_abs_max_global == -1e30:
        return None, None, None, None

    eps_abs_minmax_global = None if best_ele_min_abs == -1e30 else float(best_ele_min_abs)
    return float(eps_abs_max_global), eps_abs_minmax_global, info_abs_max, info_abs_minmax
