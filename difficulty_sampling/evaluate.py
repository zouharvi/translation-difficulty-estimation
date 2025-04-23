"""
We piggy-back on top of subset2evaluate for now. This might change later.
"""

from typing import List, Optional, Dict

import scipy
import subset2evaluate.evaluate
import collections
import numpy as np

from difficulty_sampling.data import Data, SrcData, get_src_score

# compute cluster count and rank correlation
eval_clu_cor = subset2evaluate.evaluate.eval_clucor

MainResult = collections.namedtuple(
    "MainResult", ["avg_score", "avg_score_z", "diff_corr", "avg_perfect"]
)


def main_eval(
    src_data_list: List[SrcData],
    method_name: str,
    budget: int = 100,
    sorted_src_data_ids: Optional[List[int]] = None,
):
    """
    Return the subset difficulty evaluation metrics on the input source text data using the input method.

    Args:
        src_data_list: List of source texts data.
        method_name: Name of the method whose scores will be used for subsampling.
        budget: Number of source texts to maintain in the sampled subset.
        sorted_src_data_ids: List of source text IDs for subsampling. If None, subsampling will be run. Default: None.

    Returns:
        MainResult: Evaluation results with `AvgScore`, `AvgScore-z`, `DiffCorr`, and `% Perfect`.
    """
    if sorted_src_data_ids is None:
        avg_method_scores = [
            np.average(
                [
                    method_name2score[method_name]
                    for method_name2score in src_data["scores"].values()
                ]
            )
            for src_data in src_data_list
        ]
        # Subsample according to average scores extracted from the input method.
        filtered_src_data_list: List[SrcData] = [
            src_data
            for (src_data, avg_score) in sorted(
                list(zip(src_data_list, avg_method_scores)), key=lambda pair: pair[1]
            )[:budget]
        ]
    else:
        filtered_src_data_list: List[SrcData] = [
            src_data_list[src_idx] for src_idx in sorted_src_data_ids[:budget]
        ]

    result_avg_score = np.average(
        [
            method_name2score["human"]
            for src_data in filtered_src_data_list
            for method_name2score in src_data["scores"].values()
        ]
    )

    result_avg_score_z = np.average(
        [
            method_name2score["human_z"]
            for src_data in filtered_src_data_list
            for method_name2score in src_data["scores"].values()
        ]
    )

    # result_clusters = subset2evaluate.evaluate.eval_subset_clusters(filtered_src_data_list, "human")
    result_avg_perfect = np.average(
        [
            int(method_name2score["human"] == 100)
            for src_data in filtered_src_data_list
            for method_name2score in src_data["scores"].values()
        ]
    )

    result_diff_corr = []
    for sys in src_data_list[0]["scores"]:
        result_diff_corr.append(
            scipy.stats.kendalltau(
                [src_data["scores"][sys][method_name] for src_data in src_data_list],
                [src_data["scores"][sys]["human"] for src_data in src_data_list],
                variant="b",
            ).statistic
        )
    result_diff_corr = np.average(result_diff_corr)

    return MainResult(
        avg_score=result_avg_score,
        avg_score_z=result_avg_score_z,
        diff_corr=result_diff_corr,
        avg_perfect=result_avg_perfect,
    )


def get_round_robin_sorted_src_data_ids(
    lp2src_data_list: Dict[str, List[SrcData]], method_name: str
) -> List[int]:
    """
    Get the sorted source text IDs for round-robin subsampling across several language pairs.

    Args:
        lp2src_data_list: Dictionary mapping language pairs to their source text data lists.
        method_name: Name of the method whose scores will be used for subsampling.

    Returns:
        sorted_src_data_ids: A list of sorted source text IDs for round-robin subsampling.
    """
    # Sort each language pair's src_data_list by the specified method's score
    lp2sorted_src_data_ids = {
        lp: [
            idx
            for idx, src_data in sorted(
                enumerate(src_data_list),
                key=lambda pair: get_src_score(pair[1], method_name),
            )
        ]
        for lp, src_data_list in lp2src_data_list.items()
    }

    # We'll preserve insertion order by collecting new IDs in a list.
    # We also keep a set in parallel for O(1) membership checks.
    sorted_src_data_ids: List[int] = []
    seen_src_ids = set()

    while len(sorted_src_data_ids) < len(next(iter(lp2src_data_list.values()))):
        for sorted_lp_src_data_ids in lp2sorted_src_data_ids.values():
            candidate_src_id = sorted_lp_src_data_ids.pop(0)
            if candidate_src_id not in seen_src_ids:
                seen_src_ids.add(candidate_src_id)
                sorted_src_data_ids.append(candidate_src_id)

    # Double-check that all language-pair lists are now the same size
    # (i.e., each has popped an equal number of items).
    lengths = {len(v) for v in lp2sorted_src_data_ids.values()}
    assert (
        len(lengths) == 1
    ), "All source data lists should be the same length after round-robin."

    return sorted_src_data_ids


def main_eval_avg(
    method_name: str,
    data: Data,
    budget: int = 100,
    single_src_subset: bool = False,
    is_method_tgt_lang_dep: bool = False,
) -> MainResult:
    """
    Run the subset difficulty evaluation on the input data using the input method for subsampling.

    Args:
        method_name: Name of the method whose scores will be used for subsampling.
        data: Data to use for evaluation.
        budget: Number of source texts to maintain in the sampled subset.
        single_src_subset: Whether to use a shared source text subset across language pairs. Default: False.
        is_method_tgt_lang_dep: Method is tgt lang dependent. Used only if `single_src_subset` is True. Default: False.

    Returns:
        MainResult: Evaluation results with macro `AvgScore`, `AvgScore-z`, `DiffCorr`, and `% Perfect`.
    """
    results = []

    if single_src_subset and is_method_tgt_lang_dep:
        sorted_src_data_ids = get_round_robin_sorted_src_data_ids(
            data.lp2src_data_list, method_name
        )
        for src_data_list in data.lp2src_data_list.values():
            results.append(
                main_eval(src_data_list, method_name, budget, sorted_src_data_ids)
            )
    else:
        for src_data_list in data.lp2src_data_list.values():
            results.append(main_eval(src_data_list, method_name, budget))

    # average results
    return MainResult(
        avg_score=np.average([r.avg_score for r in results]),
        avg_score_z=np.average([r.avg_score_z for r in results]),
        diff_corr=np.average([r.diff_corr for r in results]),
        avg_perfect=np.average([r.avg_perfect for r in results]),
    )
