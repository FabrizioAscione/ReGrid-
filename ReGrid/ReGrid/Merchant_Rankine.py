from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple, Union

Number = Union[int, float]


def merchant_rankine_factor(dcr_max: Number, linear_buckling_factor: Number) -> float:
    """
    Compute the Merchant-Rankine factor as requested by the user:

        MR = 1 / ( (1 / DCRmax) + (1 / lambda_cr) )

    Parameters
    ----------
    dcr_max : float
        Maximum DCR from the EC3 checks. Must be positive.
    linear_buckling_factor : float
        Linear buckling factor. Must be positive.
    """
    dcr = float(dcr_max)
    lbf = float(linear_buckling_factor)

    if dcr <= 0.0:
        raise ValueError(f"dcr_max must be > 0. Got {dcr!r}.")
    if lbf <= 0.0:
        raise ValueError(f"linear_buckling_factor must be > 0. Got {lbf!r}.")

    return 1.0 / ((1.0 / (1/dcr)) + (1.0 / lbf))


def first_n_linear_buckling_factors(
    lam: Union[Sequence[Number], Iterable[Number], Number],
    n: int = 3,
) -> List[float]:
    """
    Extract the first n linear buckling factors from a scalar or iterable.
    """
    n = int(n)
    if n < 1:
        raise ValueError("n must be >= 1.")

    if isinstance(lam, (int, float)):
        vals = [float(lam)]
    else:
        vals = [float(x) for x in lam]

    if len(vals) < n:
        raise ValueError(f"Need at least {n} buckling factors; got {len(vals)}.")

    out = vals[:n]
    for i, v in enumerate(out, start=1):
        if v <= 0.0:
            raise ValueError(f"Buckling factor #{i} must be > 0. Got {v!r}.")
    return out


def merchant_rankine_factors_from_results(
    dcr_max: Number,
    lam: Union[Sequence[Number], Iterable[Number], Number],
    n_modes: int = 3,
) -> Tuple[List[float], List[float]]:
    """
    Compute Merchant-Rankine factors using the first n buckling factors.

    Returns
    -------
    mr_factors : list[float]
        Merchant-Rankine factors [MR1, MR2, ...].
    lambdas_used : list[float]
        Linear buckling factors actually used.
    """
    lambdas_used = first_n_linear_buckling_factors(lam, n=n_modes)
    mr_factors = [merchant_rankine_factor(dcr_max=dcr_max, linear_buckling_factor=lbf) for lbf in lambdas_used]
    return mr_factors, lambdas_used


def merchant_rankine_from_results(dcr_max: Number, lam: Union[Sequence[Number], Iterable[Number], Number]) -> Tuple[float, float]:
    """
    Backward-compatible wrapper returning only MR1 and lambda_1.
    """
    mr_factors, lambdas_used = merchant_rankine_factors_from_results(dcr_max=dcr_max, lam=lam, n_modes=1)
    return mr_factors[0], lambdas_used[0]


if __name__ == "__main__":
    dcr_max_example = 0.72
    lambdas = [1.64, 1.80, 2.10]
    mr_list, lam_list = merchant_rankine_factors_from_results(dcr_max_example, lambdas, n_modes=3)
    for i, (mr, lam_i) in enumerate(zip(mr_list, lam_list), start=1):
        print(f"lambda_{i} = {lam_i:.6f} | MR{i} = {mr:.6f}")
