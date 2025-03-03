import logging
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from typing import List


logger = logging.getLogger(__name__)


def plot_human_scores_hist(
    sorted_human_scores: List[List[float]],
    scorer_name: str,
    bins: np.ndarray,
    out_plot_path: Path,
) -> None:
    """
    Plot the hist of the human scores contained in the scored input data.

    Args:
        sorted_human_scores (List[List[float]]): Human scores (for several language pairs) sorted wrt the used scorer.
        scorer_name (str): Which name to use to identify the scorer used for subsampling.
        bins (np.ndarray): Bins to use for the hist plot.
        out_plot_path (Path): Path where to save the output hist plot.
    """
    all_human_scores = np.array(
        [
            human_score
            for system_human_scores in sorted_human_scores
            for human_score in system_human_scores
        ]
    )
    logger.info(f"Total number of human scores: {len(all_human_scores)}.")
    most_diff_half_human_scores, most_diff_quarter_human_scores = np.array(
        [
            human_score
            for system_human_scores in sorted_human_scores[
                : len(sorted_human_scores) // 2
            ]
            for human_score in system_human_scores
        ]
    ), np.array(
        [
            human_score
            for system_human_scores in sorted_human_scores[
                : len(sorted_human_scores) // 4
            ]
            for human_score in system_human_scores
        ]
    )

    plt.hist(
        [all_human_scores, most_diff_half_human_scores, most_diff_quarter_human_scores],
        density=True,
        bins=bins,
        label=["All", "Selected 50%", "Selected 25%"],
    )

    plt.legend()
    plt.title(
        f"Selection with {scorer_name}. Number of src: {len(sorted_human_scores)}."
    )

    # Save the figure
    out_plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_plot_path, dpi=300, bbox_inches="tight")
    plt.close()  # Close the figure to free memory and avoid displaying
