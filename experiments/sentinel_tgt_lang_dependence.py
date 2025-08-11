from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Literal, List, Tuple

import numpy as np

from difficulty_estimation.data import Data, get_src_score
import analysis.translation_difficulty_across_tgt_langs
import subsampling.sentinel


def read_arguments() -> ArgumentParser:
    parser = ArgumentParser(
        description="Command to evaluate the sensitivity of Sentinel-MQM-tgt to target language variation using "
        "wmt24.esa en-x language pairs."
    )

    parser.add_argument(
        "--correlation",
        type=str,
        choices=["kendall", "spearman", "pearson"],
        default="kendall",
        help="Correlation function to use in the analysis. Allowed values: 'kendall', 'spearman', 'spearman'. "
        "Default: 'kendall'.",
    )

    parser.add_argument(
        "--out-plot-path",
        type=Path,
        help="Path to the local file where the correlation matrix plot from the analysis will be saved.",
    )

    return parser


def compute_correlation_matrix_from_sentinel_scored_data(
    scored_data: Data,
    correlation_method: Literal["kendall", "spearman", "pearson"] = "kendall",
) -> Tuple[np.ndarray, List[str]]:
    """
    Computes the correlation matrix for source text Sentinel-MQM-tgt scores across multiple language pairs.

    Args:
        scored_data: Data object containing the loaded WMT data scored with Sentinel-MQM-tgt. .
        correlation_method: Correlation method ("kendall", "spearman", "pearson"). Default: "kendall".

    Returns:
        (correlation_matrix, lp): Symmetric correlation matrix and language pairs list.
    """
    correlation_func = analysis.translation_difficulty_across_tgt_langs.get_correlation_function(correlation_method)

    lps = sorted(scored_data.lp2src_data_list)
    num_lps = len(lps)

    lp2scores = {lp: [] for lp in lps}
    for lp in lps:
        lp2scores[lp] = [
            get_src_score(src_data, "sentinel-src-mqm-tgt-lang")
            for src_data in scored_data.lp2src_data_list[lp]
        ]

    # Compute correlation matrix using numpy
    correlation_matrix = np.zeros((num_lps, num_lps))
    for i, lp1 in enumerate(lps):
        for j, lp2 in enumerate(lps):
            if i < j:
                corr_value, _ = correlation_func(lp2scores[lp1], lp2scores[lp2])
                correlation_matrix[i, j] = corr_value
                correlation_matrix[j, i] = corr_value
    np.fill_diagonal(correlation_matrix, 1.0)

    return correlation_matrix, lps


def sentinel_tgt_lang_dependence_command() -> None:
    """
    Command to evaluate the sensitivity of Sentinel-MQM-tgt to target language variation using wmt24.esa en-x lps.
    """
    args: Namespace = read_arguments().parse_args()

    data = Data.load(dataset_name="wmt24", lps=["en-x"], domains="all", protocol="esa")

    # Score all data with Sentinel-MQM-tgt.
    subsampling.sentinel.sentinel_src_metric_model_score(
        subsampling.sentinel.get_sentinel_src_metric_model("Prosho/sentinel-src-mqm-tgt-lang"),
        scorer_name="sentinel-src-mqm-tgt-lang",
        data=data,
        use_tgt_lang_token=True,
    )

    # Compute symmetric correlation matrix.
    correlation_matrix, lps = compute_correlation_matrix_from_sentinel_scored_data(
        data, args.correlation
    )

    # Save the plot.
    analysis.translation_difficulty_across_tgt_langs.save_correlation_plot(
        correlation_matrix,
        lps,
        "Language Pair",
        args.out_plot_path,
        args.correlation,
        "Sentinel-MQM-tgt",
    )


if __name__ == "__main__":
    sentinel_tgt_lang_dependence_command()
