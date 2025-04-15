"""
We piggy-back on top of subset2evaluate for now. This might change later.
"""

from typing import Callable, List, Optional
import subset2evaluate.evaluate
import collections
import numpy as np


# compute cluster count and rank correlation
eval_clu_cor = subset2evaluate.evaluate.eval_clucor

MainResult = collections.namedtuple(
    "MainResult", ["avg_score", "avg_diff", "diff_corr", "avg_perfect"]
)


def main_eval(
    data: List,
    method_name: str,
    B: int = 100,
):
    """
    TODO description
    """
    import scipy.stats

    data_y = [np.average([x["scores"][sys][method_name] for sys in x["scores"].keys()]) for x in data]
    # take top-B from data based on data_y
    data_new = [x[0] for x in sorted(list(zip(data, data_y)), key=lambda x: x[1])[:B]]

    result_avg_score = np.average([
        line["scores"][sys]["human"]
        for line in data_new
        for sys in line["scores"].keys()
    ])

    result_avg_diff = np.average([
        line["scores"][sys]["human_z"]
        for line in data_new
        for sys in line["scores"].keys()
    ])

    # result_clusters = subset2evaluate.evaluate.eval_subset_clusters(data_new, "human")
    result_avg_perfect = np.average([
        1 if line["scores"][sys]["human"] == 100 else 0
        for line in data_new
        for sys in line["scores"].keys()
    ])

    result_diff_corr = []
    for sys in data[0]["scores"].keys():
        data_y_sys = [x["scores"][sys]["human"] for x in data]
        data_diff_sys = [x["scores"][sys][method_name] for x in data]
        result_diff_corr.append(scipy.stats.kendalltau(data_y_sys, data_diff_sys, variant="b").correlation)
    result_diff_corr = np.average(result_diff_corr)

    return MainResult(
        avg_score=result_avg_score,
        avg_diff=result_avg_diff,
        diff_corr=result_diff_corr,
        avg_perfect=result_avg_perfect,
    )


def main_eval_avg(
    method_fn: Callable,
    data: List,
    B: Optional[int] = None,
    p: Optional[float] = None,
):
    """
    TODO description
    """
    assert B is not None or p is not None, "Either B or p must be specified"

    results = []

    for data in data.lp2src_data_list.values():
        if B is not None:
            results.append(main_eval(data, method_fn, B))
        elif p is not None:
            results.append(main_eval(data, method_fn, int(len(data)*p)))

    # average results
    return MainResult(
        avg_score=np.average([x.avg_score for x in results]),
        avg_diff=np.average([x.avg_diff for x in results]),
        diff_corr=np.average([x.diff_corr for x in results]),
        avg_perfect=np.average([x.avg_perfect for x in results]),
    )
