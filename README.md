# Numpy in-group ranking

This package provides numpy-based vectorized in-group
[ranking](https://en.wikipedia.org/wiki/Ranking) as an alternative to
[pandas.GroupBy.rank](https://pandas.pydata.org/docs/reference/api/pandas.core.groupby.DataFrameGroupBy.rank.html). It's
mainly for fun and is not optimized, but I do use it to avoid importing pandas
when I only need pandas for ranking.

The package is greatly inspired by
[numpy-indexed](https://github.com/EelcoHoogendoorn/Numpy_arraysetops_EP) and
[scipy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rankdata.html).

The test cases are mostly adopted from
[scipy](https://github.com/scipy/scipy/blob/main/scipy/stats/tests/test_rank.py).

## Usage

```
from grank import grank
gid = [0, 1, 0, 0, 1, 0, 1, 2, 2, 1]
val = [1, 6, 6, 4, 6, 4, 7, 7, 7, 6]
ranks = grank(val, gid, method="average")
assert_array_equal(ranks, [1, 2, 4, 2.5, 2, 2.5, 4, 1.5, 1.5, 2])
```
