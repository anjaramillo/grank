from typing import Any, Optional

import numpy as np
import numpy.typing as npt


def grank(
    a: npt.ArrayLike,
    g: Optional[npt.ArrayLike] = None,
    method: str = "average",
    axis: Optional[int] = None,
) -> npt.NDArray[Any]:
    """Assign ranks independently within groups.

    When `g` is not None, the rankings are assigned independently within groups. When
    `g` is None, the behaviour is the same as scipy.stats.rankdata.

    """
    a = np.asarray(a)
    if axis is not None:
        if a.size == 0:
            dt = np.float64 if method == "average" else np.int_
            return np.empty(a.shape, dtype=dt)
        return np.apply_along_axis(grank, axis, a, g, method)

    if g is None:
        g = np.zeros_like(a)
    g = np.asarray(g)
    assert a.shape == g.shape, "`a` and `g` must be the same shape."

    try:
        nan_indexes = np.isnan(a)
    except Exception:
        nan_indexes = None

    # These helper indexes are inspired by numpy-indexed.
    sorter = np.lexsort((a, g))
    sorted_a = a[sorter]
    sorted_g = g[sorter]
    flag = sorted_g[:-1] != sorted_g[1:]
    slices = np.r_[0, np.flatnonzero(flag) + 1, len(a)]
    start = slices[:-1]

    # Convert sorted group id into the original rank
    g2inv = np.empty(sorter.size, int)
    g2inv[sorter] = np.cumsum(np.r_[False, flag])

    offset = start

    if method == "ordinal":
        rank = np.empty(len(a), int)
        rank[sorter] = np.arange(len(a))
        rank -= offset[g2inv]
        result = rank + 1
    else:
        # Convert sorted data into the original rank
        x2inv = np.empty(sorter.size, int)
        x2inv[sorter] = np.arange(sorter.size, dtype=np.intp)

        obs = np.r_[True, sorted_a[1:] != sorted_a[:-1]]
        obs[start] = True
        dense = obs.cumsum()
        if method == "dense":
            # the offset needs to count g unique values
            dense_offset = dense[start]
            result = dense[x2inv] - dense_offset[g2inv] + 1
        else:
            count = np.r_[np.flatnonzero(obs), len(obs)]
            if method == "max" or method == "average":
                max_rank = count[dense][x2inv] - offset[g2inv]

            if method == "min" or method == "average":
                min_rank = count[dense - 1][x2inv] - offset[g2inv] + 1

            if method == "average":
                avg_rank = 0.5 * (max_rank + min_rank)

            if method == "max":
                result = max_rank
            elif method == "min":
                result = min_rank
            elif method == "average":
                result = avg_rank

    if nan_indexes is not None:
        result = result.astype("float64")
        result[nan_indexes] = np.nan
    return result


def npi_rank_n(
    a: npt.ArrayLike,
    g: Optional[npt.ArrayLike] = None,
    method: str = "average",
    axis: Optional[int] = None,
    nan_policy: str = "propagate",
    n_jobs: Optional[int] = None,
) -> npt.NDArray[np.float64]:
    """Assign ranks independently within groups. Non-vectorized implementation with
    numpy-indexed

    """
    import numpy_indexed as npi
    from joblib import Parallel, delayed
    from scipy.stats import rankdata

    if g is None:
        g = np.zeros_like(a)
    a, g = np.asarray(a), np.asarray(g)

    if a.size == 0:
        dt = np.float64 if method == "average" else np.int_
        return np.empty(a.shape, dtype=dt)

    gb = npi.group_by(g)
    with Parallel(n_jobs=n_jobs) as parallel:
        ranks = parallel(
            delayed(rankdata)(a=x, method=method, axis=axis, nan_policy=nan_policy)
            for x in gb.split(a)
        )
    rank = np.empty_like(a, dtype=float)
    rank[gb.index.sorter] = np.concatenate(ranks)
    return rank


def npi_rank_v(
    a: npt.ArrayLike,
    g: Optional[npt.ArrayLike] = None,
    method: str = "average",
    axis: Optional[int] = None,
) -> npt.NDArray[Any]:
    """Assign ranks independently within groups. Vectorized implementation with
    numpy-indexed

    """
    if method not in ("average", "min", "max", "dense", "ordinal"):
        raise ValueError(f"Unknown method `{method}`")

    import numpy_indexed as npi

    a = np.asarray(a)
    if axis is not None:
        if a.size == 0:
            dt = np.float64 if method == "average" else np.int_
            return np.empty(a.shape, dtype=dt)
        return np.apply_along_axis(npi_rank_v, axis, a, g, method)

    assert a.ndim == 1, "Only 1-dimensional arrays are supported"
    if g is None:
        g = np.zeros_like(a)

    lexidx = npi.as_index((a, g))

    if method == "dense":
        # Get the unique count of each g. lexidx.unique[1] is the sorted `g`.
        unique_idx = npi.as_index(lexidx.unique[1])
        # Get the offset based on the previous g size
        offset = np.concatenate([[0], np.cumsum(unique_idx.count[:-1])])
        # Inflate the offset to the same shape as the groups
        offset = offset[unique_idx.inverse]
        dense_ranks = unique_idx.rank - offset + 1
        ranks: npt.NDArray[np.float64] = dense_ranks[lexidx.inverse]
        return ranks

    gidx = npi.as_index(g)
    offset = np.concatenate([[0], np.cumsum(gidx.count[:-1])])
    offset = offset[gidx.inverse]
    ranks = lexidx.rank - offset + 1

    if method == "ordinal":
        return ranks

    if method == "average":
        k, v = npi.group_by(lexidx).mean(ranks)
    elif method == "max":
        k, v = npi.group_by(lexidx).max(ranks)
    elif method == "min":
        k, v = npi.group_by(lexidx).min(ranks)
    ranks = v[lexidx.inverse]
    return ranks
