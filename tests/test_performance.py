import time
from typing import Any, Dict

import numpy as np
import pandas as pd
import seaborn as sns
from numpy.testing import assert_array_equal

from grank import grank, npi_rank_n, npi_rank_v


def run_once(groups: int, group_size: int, method: str) -> pd.DataFrame:
    a = np.random.randint(0, group_size, group_size * groups)
    g = np.random.randint(0, groups, group_size * groups)

    t1 = time.time()
    result_a = grank(a, g, method)
    t2 = time.time()
    time_grank = t2 - t1
    print(f"grank: {t2-t1}")

    t1 = time.time()
    result_b = npi_rank_n(a, g, method)
    t2 = time.time()
    time_nprank = t2 - t1
    print(f"npi_rank_n: {t2-t1}")

    t1 = time.time()
    result_c = npi_rank_v(a, g, method)
    t2 = time.time()
    time_npirank = t2 - t1
    print(f"npi_rank_v: {t2-t1}")

    df = pd.DataFrame({"a": a, "g": g})
    method = method if method != "ordinal" else "first"
    t1 = time.time()
    result_d = df.groupby("g")["a"].rank(method)
    t2 = time.time()
    time_pandas = t2 - t1
    print(f"pandas: {t2-t1}")

    assert_array_equal(result_a, result_b)
    assert_array_equal(result_a, result_c)
    assert_array_equal(result_a, result_d)

    return {
        "grank": time_grank,
        "npi_rank_n": time_nprank,
        "npi_rank_v": time_npirank,
        "pandas.rank": time_pandas,
    }


def main() -> None:
    np.random.seed(0)
    group_size = 100
    for method in ["average", "min", "max", "dense", "ordinal"]:
        data = []
        for groups in [10, 100, 1000, 10000, 100000, 1000000]:
            result: Dict[str, Any] = {}
            result["groups"] = groups
            print(f"Groups: {groups}")
            result.update(run_once(groups, group_size, method))
            data.append(result)
        df = pd.DataFrame(data)
        print(
            df.to_markdown(
                f"assets/{method}.md", index=False, floatfmt=[".0f"] * 1 + [".3f"] * 4
            )
        )
        df = df.melt(
            id_vars=["groups"],
            value_vars=["grank", "npi_rank_n", "npi_rank_v", "pandas.rank"],
            var_name="function",
            value_name="seconds",
        )
        sns.set_theme("notebook", "darkgrid", font="Linux Biolinum O")
        g = sns.relplot(
            kind="line",
            data=df,
            x="groups",
            y="seconds",
            hue="function",
            aspect=2,
        )
        g.set(xscale="log", yscale="log")
        g.tight_layout()
        g.savefig(f"assets/{method}.png", dpi=600)


if __name__ == "__main__":
    main()
