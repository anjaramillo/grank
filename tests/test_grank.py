import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_equal

from grank import grank


class TestRankData:
    def test_empty(self):
        """stats.grank([]) should return an empty array."""
        a = np.array([], dtype=int)
        r = grank(a)
        assert_array_equal(r, np.array([], dtype=np.float64))
        r = grank([])
        assert_array_equal(r, np.array([], dtype=np.float64))

    def test_one(self):
        """Check stats.grank with an array of length 1."""
        data = [100]
        a = np.array(data, dtype=int)
        r = grank(a)
        assert_array_equal(r, np.array([1.0], dtype=np.float64))
        r = grank(data)
        assert_array_equal(r, np.array([1.0], dtype=np.float64))

    def test_basic(self):
        """Basic tests of stats.grank."""
        data = [100, 10, 50]
        expected = np.array([3.0, 1.0, 2.0], dtype=np.float64)
        a = np.array(data, dtype=int)
        r = grank(a)
        assert_array_equal(r, expected)
        r = grank(data)
        assert_array_equal(r, expected)

        data = [40, 10, 30, 10, 50]
        expected = np.array([4.0, 1.5, 3.0, 1.5, 5.0], dtype=np.float64)
        a = np.array(data, dtype=int)
        r = grank(a)
        assert_array_equal(r, expected)
        r = grank(data)
        assert_array_equal(r, expected)

        data = [20, 20, 20, 10, 10, 10]
        expected = np.array([5.0, 5.0, 5.0, 2.0, 2.0, 2.0], dtype=np.float64)
        a = np.array(data, dtype=int)
        r = grank(a)
        assert_array_equal(r, expected)
        r = grank(data)
        assert_array_equal(r, expected)
        # The docstring states explicitly that the argument is flattened.
        # a2d = a.reshape(2, 3)
        # r = grank(a2d)
        # assert_array_equal(r, expected)

    def test_rankdata_object_string(self):
        min_rank = lambda a: [1 + sum(i < j for i in a) for j in a]
        max_rank = lambda a: [sum(i <= j for i in a) for j in a]
        ordinal_rank = lambda a: min_rank([(x, i) for i, x in enumerate(a)])

        def average_rank(a):
            return [(i + j) / 2.0 for i, j in zip(min_rank(a), max_rank(a))]

        def dense_rank(a):
            b = np.unique(a)
            return [1 + sum(i < j for i in b) for j in a]

        rankf = dict(
            min=min_rank,
            max=max_rank,
            ordinal=ordinal_rank,
            average=average_rank,
            dense=dense_rank,
        )

        def check_ranks(a):
            for method in "min", "max", "dense", "ordinal", "average":
                out = grank(a, method=method)
                assert_array_equal(out, rankf[method](a))

        val = ["foo", "bar", "qux", "xyz", "abc", "efg", "ace", "qwe", "qaz"]
        check_ranks(np.random.choice(val, 200))
        check_ranks(np.random.choice(val, 200).astype("object"))

        val = np.array([0, 1, 2, 2.718, 3, 3.141], dtype="object")
        check_ranks(np.random.choice(val, 200).astype("object"))

    def test_large_int(self):
        data = np.array([2**60, 2**60 + 1], dtype=np.uint64)
        r = grank(data)
        assert_array_equal(r, [1.0, 2.0])

        data = np.array([2**60, 2**60 + 1], dtype=np.int64)
        r = grank(data)
        assert_array_equal(r, [1.0, 2.0])

        data = np.array([2**60, -(2**60) + 1], dtype=np.int64)
        r = grank(data)
        assert_array_equal(r, [2.0, 1.0])

    def test_big_tie(self):
        for n in [10000, 100000, 1000000]:
            data = np.ones(n, dtype=int)
            r = grank(data)
            expected_rank = 0.5 * (n + 1)
            assert_array_equal(r, expected_rank * data, "test failed with n=%d" % n)

    def test_axis(self):
        data = [[0, 2, 1], [4, 2, 2]]
        expected0 = [[1.0, 1.5, 1.0], [2.0, 1.5, 2.0]]
        r0 = grank(data, axis=0)
        assert_array_equal(r0, expected0)
        expected1 = [[1.0, 3.0, 2.0], [3.0, 1.5, 1.5]]
        r1 = grank(data, axis=1)
        assert_array_equal(r1, expected1)

    methods = ["average", "min", "max", "dense", "ordinal"]
    dtypes = [np.float64] + [np.int_] * 4

    @pytest.mark.parametrize("axis", [0, 1])
    @pytest.mark.parametrize("method, dtype", zip(methods, dtypes))
    def test_size_0_axis(self, axis, method, dtype):
        shape = (3, 0)
        data = np.zeros(shape)
        r = grank(data, method=method, axis=axis)
        assert_equal(r.shape, shape)
        assert_equal(r.dtype, dtype)


_cases = (
    # values, method, expected
    ([], "average", []),
    ([], "min", []),
    ([], "max", []),
    ([], "dense", []),
    ([], "ordinal", []),
    #
    ([100], "average", [1.0]),
    ([100], "min", [1.0]),
    ([100], "max", [1.0]),
    ([100], "dense", [1.0]),
    ([100], "ordinal", [1.0]),
    #
    ([100, 100, 100], "average", [2.0, 2.0, 2.0]),
    ([100, 100, 100], "min", [1.0, 1.0, 1.0]),
    ([100, 100, 100], "max", [3.0, 3.0, 3.0]),
    ([100, 100, 100], "dense", [1.0, 1.0, 1.0]),
    ([100, 100, 100], "ordinal", [1.0, 2.0, 3.0]),
    #
    ([100, 300, 200], "average", [1.0, 3.0, 2.0]),
    ([100, 300, 200], "min", [1.0, 3.0, 2.0]),
    ([100, 300, 200], "max", [1.0, 3.0, 2.0]),
    ([100, 300, 200], "dense", [1.0, 3.0, 2.0]),
    ([100, 300, 200], "ordinal", [1.0, 3.0, 2.0]),
    #
    ([100, 200, 300, 200], "average", [1.0, 2.5, 4.0, 2.5]),
    ([100, 200, 300, 200], "min", [1.0, 2.0, 4.0, 2.0]),
    ([100, 200, 300, 200], "max", [1.0, 3.0, 4.0, 3.0]),
    ([100, 200, 300, 200], "dense", [1.0, 2.0, 3.0, 2.0]),
    ([100, 200, 300, 200], "ordinal", [1.0, 2.0, 4.0, 3.0]),
    #
    ([100, 200, 300, 200, 100], "average", [1.5, 3.5, 5.0, 3.5, 1.5]),
    ([100, 200, 300, 200, 100], "min", [1.0, 3.0, 5.0, 3.0, 1.0]),
    ([100, 200, 300, 200, 100], "max", [2.0, 4.0, 5.0, 4.0, 2.0]),
    ([100, 200, 300, 200, 100], "dense", [1.0, 2.0, 3.0, 2.0, 1.0]),
    ([100, 200, 300, 200, 100], "ordinal", [1.0, 3.0, 5.0, 4.0, 2.0]),
    #
    ([10] * 30, "ordinal", np.arange(1.0, 31.0)),
)


def test_cases():
    for values, method, expected in _cases:
        group = np.tile(np.arange(3), len(values))
        values = np.repeat(values, 3)
        expected = np.repeat(expected, 3)
        r = grank(values, group, method=method)
        assert_array_equal(r, expected)


def test_manual():
    gid = [0, 1, 0, 1, 0, 1]
    val = [1, 3, 6, 8, 4, 3]
    ranks = grank(val, gid, method="average")
    assert_array_equal(ranks, [1, 1.5, 3, 3, 2, 1.5])

    ranks = grank(val, gid, method="min")
    assert_array_equal(ranks, [1, 1, 3, 3, 2, 1])

    ranks = grank(val, gid, method="max")
    assert_array_equal(ranks, [1, 2, 3, 3, 2, 2])

    ranks = grank(val, gid, method="ordinal")
    assert_array_equal(ranks, [1, 1, 3, 3, 2, 2])

    ranks = grank(val, gid, method="dense")
    assert_array_equal(ranks, [1, 1, 3, 2, 2, 1])


def test_manual2():
    gid = [0, 1, 0, 0, 1, 0, 1, 2, 2, 1]
    val = [1, 6, 6, 4, 6, 4, 7, 7, 7, 6]
    ranks = grank(val, gid, method="average")
    assert_array_equal(ranks, [1, 2, 4, 2.5, 2, 2.5, 4, 1.5, 1.5, 2])

    ranks = grank(val, gid, method="min")
    assert_array_equal(ranks, [1, 1, 4, 2, 1, 2, 4, 1, 1, 1])

    ranks = grank(val, gid, method="max")
    assert_array_equal(ranks, [1, 3, 4, 3, 3, 3, 4, 2, 2, 3])

    ranks = grank(val, gid, method="ordinal")
    assert_array_equal(ranks, [1, 1, 4, 2, 2, 3, 4, 1, 2, 3])

    ranks = grank(val, gid, method="dense")
    assert_array_equal(ranks, [1, 1, 3, 2, 1, 2, 2, 1, 1, 1])
