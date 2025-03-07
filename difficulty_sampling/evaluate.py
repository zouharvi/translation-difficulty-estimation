"""
We piggy-back on top of subset2evaluate for now. This might change later.
"""

from typing import Callable, List
import subset2evaluate.evaluate
import collections
import numpy as np

# compute cluster count and rank correlation
eval_clu_cor = subset2evaluate.evaluate.eval_clucor

MainResult = collections.namedtuple(
    "MainResult", ["avg_score", "diff_corr", "clusters"]
)


def main_eval(
    data: List,
    method_name: str,
    B: int = 100,
):
    """
    TODO description
    """

    data_y = [np.average([x["scores"][sys][method_name] for sys in x["scores"].keys()]) for x in data]
    # take top-B from data based on data_y
    data_new = [x[0] for x in sorted(list(zip(data, data_y)), key=lambda x: x[1])[:B]]

    result_avg_score = np.average([
        line["scores"][sys]["human"]
        for line in data_new
        for sys in line["scores"].keys()
    ])

    result_clusters = subset2evaluate.evaluate.eval_subset_clusters(data_new, "human")

    # TODO:
    result_diff_corr = 0

    return MainResult(
        avg_score=result_avg_score,
        diff_corr=result_diff_corr,
        clusters=result_clusters,
    )


def main_eval_avg(
    method_fn: Callable,
    data: List,
    B: int = 100,
):
    """
    TODO description
    """

    results = []

    for data in data.lp2src_data_list.values():
        results.append(main_eval(data, method_fn, B))

    # average results
    return MainResult(
        avg_score=np.average([x.avg_score for x in results]),
        diff_corr=np.average([x.diff_corr for x in results]),
        clusters=np.average([x.clusters for x in results]),
    )
