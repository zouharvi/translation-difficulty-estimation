from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Literal, List, Tuple, Dict

import numpy as np
from matplotlib import pyplot as plt

from difficulty_estimation.data import Data, SrcData
import analysis.translation_difficulty_across_tgt_langs


def read_arguments() -> ArgumentParser:
    parser = ArgumentParser(
        description="Command to analyze how translation human scores vary across distinct MT systems."
    )

    parser.add_argument(
        "--dataset-name",
        type=str,
        default="wmt24",
        help="WMT test set to analyze. Default: 'wmt24'.",
    )

    parser.add_argument(
        "--lp",
        type=str,
        default="en-de",
        help="Language pair to consider in the input WMT test set. Default: 'en-de'.",
    )

    parser.add_argument(
        "--protocol",
        type=str,
        default="mqm",
        help="Annotation protocol to use when loading human scores for the WMT test set. Default: 'mqm'.",
    )

    parser.add_argument(
        "--domains",
        type=str,
        nargs="+",
        default="all",
        help="Domains to be analyzed. If not specified, all domains are considered ('all'). Default: 'all'.",
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
        "--compute-diff-corr",
        action="store_true",
        help="Compute a single DiffCorr score per system—its average Kendall τ_b against all MT systems—instead of the "
        "full correlation matrix. Note: when this flag is used, --correlation is ignored and Kendall τ_b is always "
        "applied.",
    )

    parser.add_argument(
        "--out-plot-path",
        type=Path,
        help="Path to the local file where the correlation matrix plot from the analysis will be saved.",
    )

    return parser


def compute_diff_corr(
    src_data_list: List[SrcData],
) -> Tuple[Dict[str, float], List[str]]:
    """
    For each system, compute its average Kendall τ_b against all MT systems.
    Returns a dict system -> DiffCorr and the list of systems (sorted).

    Args
        src_data_list: List of source data objects containing the loaded WMT data with human scores.

    Returns:
        diff_corr: Dictionary with DiffCorr scores for each system.
        mt_systems: List of systems.
    """
    corr_func = (
        analysis.translation_difficulty_across_tgt_langs.get_correlation_function(
            "kendall"
        )
    )

    # 1) Gather all systems and separately the MT subset
    all_systems = sorted({sys for sample in src_data_list for sys in sample["scores"]})
    mt_systems = [s for s in all_systems if not s.lower().startswith("ref")]

    diff_corr: Dict[str, float] = dict()
    for sys1 in all_systems:
        taus: List[float] = []

        # correlate sys1 _only_ against each MT system
        for sys2 in mt_systems:
            if sys1 == sys2:
                continue

            h1, h2 = [], []
            for sample in src_data_list:
                sc = sample["scores"]
                if sys1 in sc and sys2 in sc:
                    v1 = sc[sys1].get("human")
                    v2 = sc[sys2].get("human")
                    if v1 is not None and v2 is not None:
                        h1.append(v1)
                        h2.append(v2)

            if len(h1) >= 2:
                tau, _ = corr_func(h1, h2)
                taus.append(tau)

        # arithmetic mean of the pairwise taus
        diff_corr[sys1] = np.nanmean(taus) if taus else np.nan

    return diff_corr, all_systems


def compute_corr_matrix_across_sys(
    src_data_list: List[SrcData],
    correlation_method: Literal["kendall", "spearman", "pearson"] = "kendall",
) -> Tuple[np.ndarray, List[str]]:
    """
    Computes the correlation matrix across distinct MT systems for source text human scores.

    Args:
        src_data_list: List of source data objects containing the loaded WMT data with human scores.
        correlation_method: Correlation method ("kendall", "spearman", "pearson"). Default: "kendall".

    Returns:
        (correlation_matrix, lp): Symmetric correlation matrix and MT systems list.
    """
    corr_func = (
        analysis.translation_difficulty_across_tgt_langs.get_correlation_function(
            correlation_method
        )
    )

    mt_systems = sorted({sys for sample in src_data_list for sys in sample["scores"]})

    n = len(mt_systems)
    corr_matrix = np.eye(n)  # diagonal already 1

    for i, sys1 in enumerate(mt_systems):
        for j in range(i + 1, n):
            sys2 = mt_systems[j]

            human1, human2 = [], []
            for sample in src_data_list:
                if sys1 in sample["scores"] and sys2 in sample["scores"]:
                    h1, h2 = sample["scores"][sys1].get("human"), sample["scores"][
                        sys2
                    ].get("human")
                    if h1 is not None and h2 is not None:
                        human1.append(h1)
                        human2.append(h2)

            if len(human1) >= 2:
                tau, _ = corr_func(human1, human2)
            else:
                tau = np.nan

            corr_matrix[i, j] = tau
            corr_matrix[j, i] = tau

    return corr_matrix, mt_systems


def translation_scores_across_sys_command() -> None:
    """
    Command to analyze how translation human scores vary across distinct MT systems.
    """
    args: Namespace = read_arguments().parse_args()

    src_data_list = Data.load(
        dataset_name=args.dataset_name,
        lps=[args.lp],
        protocol=args.protocol,
        domains=args.domains,
        include_ref=True,
        include_human=True,
    ).lp2src_data_list[args.lp]

    if args.compute_diff_corr:
        diff_corr, all_systems = compute_diff_corr(src_data_list)

        # sort by descending DiffCorr, then plot a horizontal bar chart
        sorted_sys = sorted(all_systems, key=lambda s: diff_corr[s], reverse=True)
        scores = [diff_corr[s] for s in sorted_sys]

        fig, ax = plt.subplots(figsize=(8, len(sorted_sys) * 0.4))
        y = np.arange(len(sorted_sys))
        ax.barh(y, scores, align="center")
        ax.set_yticks(y)
        ax.set_yticklabels(sorted_sys)
        ax.invert_yaxis()  # highest score on top
        ax.set_xlabel("DiffCorr (avg Kendall τ₍b₎ vs. MT systems)")
        ax.set_title(f"DiffCorr per System ({args.lp}, {args.protocol})")
        fig.tight_layout()
        fig.savefig(args.out_plot_path)
    else:
        # Compute symmetric correlation matrix.
        corr_matrix, mt_systems = compute_corr_matrix_across_sys(
            src_data_list, args.correlation
        )

        # Save the plot.
        analysis.translation_difficulty_across_tgt_langs.save_correlation_plot(
            corr_matrix,
            mt_systems,
            "MT System",
            args.out_plot_path,
            args.correlation,
            f"Human ({args.protocol})",
        )


if __name__ == "__main__":
    translation_scores_across_sys_command()
