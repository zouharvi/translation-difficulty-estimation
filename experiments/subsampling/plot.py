from pathlib import Path
from typing import Dict, Union, List

import numpy as np
from matplotlib import pyplot as plt

from difficulty_sampling.data import Data


def plot_human_scores_hist(
    scored_data: Data, scorer_name: str, bins: np.ndarray, out_plot_path: Path
) -> None:
    """
    Plot the hist of the human scores contained in the scored input data.

    Args:
        scored_data (Data): Data containing scores to use for subsampling.
        scorer_name (str): Which name to use to identify the scorer used for subsampling.
        bins (np.ndarray): Bins to use for the hist plot.
        out_plot_path (Path): Path where to save the output hist plot.
    """
    data_y1 = np.array(
        [
            np.average([v["human"] for v in l["scores"].values()])
            for l in scored_data.src_data_list
        ]
    )
    len_flat = len(data_y1)
    data_y2 = np.array(
        [
            np.average([v["human"] for v in l["scores"].values()])
            for l in scored_data.src_data_list
        ][: int(len_flat * 0.50)]
    )
    data_y3 = np.array(
        [
            np.average([v["human"] for v in l["scores"].values()])
            for l in scored_data.src_data_list
        ][: int(len_flat * 0.25)]
    )

    plt.hist(
        [data_y1, data_y2, data_y3],
        density=True,
        bins=bins,
        label=["All", "Selected 50%", "Selected 25%"],
    )

    plt.legend()
    plt.title(f"Selection with {scorer_name}. Total number of src: {len_flat}.")

    # Save the figure
    out_plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_plot_path, dpi=300, bbox_inches="tight")
    plt.close()  # Close the figure to free memory and avoid displaying
