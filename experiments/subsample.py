from typing import Dict
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from difficulty_sampling.data import SrcData
from subsampling.sentinel import subsample_with_sentinel_src
from subsampling.word_frequency import subsample_with_word_frequency
from subsampling.plot import plot_human_scores_hist


def read_arguments() -> ArgumentParser:
    parser = ArgumentParser(
        description="Command to subsample WMT data using the outputs returned by a given scoring method."
    )

    parser.add_argument(
        "--dataset-name",
        type=str,
        default="wmt24",
        help="Name of the WMT dataset to subsample. Default: 'wmt24'.",
    )

    parser.add_argument(
        "--lp",
        type=str,
        default="en-es",
        help="Language pair to consider in the WMT dataset passed in input. If 'all_en' is passed, all the wmt24 "
        "language pairs with English on the source side will be used, and the '--dataset-name' argument will be "
        "ignored. Default: 'en-es'.",
    )

    parser.add_argument(
        "--protocol",
        type=str,
        choices=["esa", "mqm"],
        default="esa",
        help="Which annotation protocol to consider when loading human scores. Allowed values: 'esa', 'mqm'. "
        "Default: 'esa'.",
    )

    parser.add_argument(
        "--domains",
        type=str,
        nargs="+",
        default="all",
        help="Domains to be analyzed. If not specified, all domains are considered ('all'). Default: 'all'.",
    )

    parser.add_argument(
        "--sentinel-src-metric-model",
        type=str,
        default="sapienzanlp/sentinel-src-mqm",
        help="String that identifies a local file system path to a sentinel-src metric model checkpoint, or a string "
        "that identifies it on the Hugging Face Hub.",
    )

    parser.add_argument(
        "--batch-size",
        default=32,
        type=int,
        help="Batch size to use when running inference with the input sentinel-src metric model. Default: 32.",
    )

    parser.add_argument(
        "--scorer-name",
        type=str,
        default="sentinel-src-mqm",
        help="Which name to use to identify the sentinel-src metric model used for the subsampling (it will be used in "
        "the output plot and for the output path where to save it). Default: 'sentinel-src-mqm'.",
    )

    parser.add_argument(
        "--out-plot-path",
        type=Path,
        required=True,
        help="Local file system path where to save the output hist.",
    )

    return parser


def get_src_score(src_data: SrcData, scorer_name: str) -> float:
    """
    Return the score assigned by the input scorer to the src data.

    Args:
        src_data (SrcData): SrcData Dictionary containing all the info for a given src segment.
        scorer_name (str): Name of the scorer to use to extract the score from the data.

    Returns:
        score (float): Score assigned by the input scorer to the src data.
    """
    scores: Dict[str, Dict[str, float]] = src_data["scores"]  # More explicit typing
    return scores[next(iter(scores))][scorer_name]


def subsample_command() -> None:
    args = read_arguments().parse_args()

    if args.scorer_name in {"sentinel-src-mqm", "sentinel-src-da"}:
        command = subsample_with_sentinel_src
    elif args.scorer_name in {"word_frequency", "word_zipf_frequency"}:
        command = subsample_with_word_frequency
    else:
        raise ValueError(f"Scorer name '{args.scorer_name}' not recognized.")

    data = command(args)

    # Sort the src data in ascending order in terms of the scorer output
    data.src_data_list.sort(
        key=lambda src_data: get_src_score(src_data, args.scorer_name)
    )

    plot_human_scores_hist(
        data,
        args.scorer_name,
        (
            np.arange(0, 100 + 10, 10)
            if args.protocol == "esa"
            else np.arange(-25, 0 + 1, 1)
        ),
        args.out_plot_path,
    )


if __name__ == "__main__":
    subsample_command()
