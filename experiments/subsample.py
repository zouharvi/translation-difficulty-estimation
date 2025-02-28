import logging
from typing import Dict
from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np

from difficulty_sampling.data import SrcData
from subsampling.sentinel import subsample_with_sentinel_src
from subsampling.word_frequency import subsample_with_word_frequency
from subsampling.syntactic_complexity import subsample_with_syntactic_complexity
from subsampling.plot import plot_human_scores_hist


logging.basicConfig(level=logging.INFO)


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
        help="Language pair to consider in the WMT dataset passed in input. If 'en-x' is passed, all the wmt24 "
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
        "--use-tgt-lang",
        action="store_true",
        help="Whether to use the target language token in the input data for the sentinel-src metric. Default: False.",
    )

    parser.add_argument(
        "--batch-size",
        default=32,
        type=int,
        help="Batch size to use when running inference with the input sentinel-src metric. Default: 32.",
    )

    parser.add_argument(
        "--syntactic-model-name",
        type=str,
        default="en_core_web_sm",
        help="Which spaCy Dependency Parsing model to use for Syntactic Structure Complexity subsampling. "
        "Default: 'en_core_web_sm'.",
    )

    parser.add_argument(
        "--scorer-name",
        type=str,
        choices=[
            "sentinel-src-mqm",
            "sentinel-src-da",
            "word_frequency",
            "word_zipf_frequency",
            "syntactic_complexity",
        ],
        default="sentinel-src-mqm",
        help="Which name to use to identify the scorer used. Allowed values: 'sentinel-src-mqm', 'sentinel-src-da', "
        "'word_frequency', 'word_zipf_frequency', 'syntactic_complexity'. Default: 'sentinel-src-mqm'.",
    )

    parser.add_argument(
        "--systems-to-filter",
        type=str,
        nargs="+",
        help="Systems to exclude from the analysis.",
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
    """
    Command to subsample WMT data using the outputs returned by a given scoring method.
    """
    args: Namespace = read_arguments().parse_args()

    if args.scorer_name in {"sentinel-src-mqm", "sentinel-src-da"}:
        command = subsample_with_sentinel_src
    elif args.scorer_name in {"word_frequency", "word_zipf_frequency"}:
        command = subsample_with_word_frequency
    elif args.scorer_name == "syntactic_complexity":
        command = subsample_with_syntactic_complexity
    else:
        raise ValueError(
            f"Scorer name '{args.scorer_name}' not recognized! Allowed values: 'sentinel-src-mqm', 'sentinel-src-da', "
            f"'word_frequency', 'word_zipf_frequency', 'syntactic_complexity'."
        )

    data = command(args)

    # Sort the human scores in ascending order in terms of the scorer output
    sorted_human_scores = []
    if command == subsample_with_sentinel_src and args.use_tgt_lang:
        lp2sorted_src_data_list = {
            lp: sorted(
                enumerate(src_data_list),
                key=lambda pair: get_src_score(pair[1], args.scorer_name),
            )
            for lp, src_data_list in data.lp2src_data_list.items()
        }

        added_src_ids = set()
        while len(sorted_human_scores) < len(
            next(iter(data.lp2src_data_list.values()))
        ):
            for sorted_src_data_list in lp2sorted_src_data_list.values():
                if len(sorted_src_data_list) == 0:
                    break

                src_idx, src_data = sorted_src_data_list.pop(0)
                if src_idx in added_src_ids:
                    continue

                sorted_human_scores.append(
                    [
                        src_data_list[src_idx]["scores"][sys]["human"]
                        for src_data_list in data.lp2src_data_list.values()
                        for sys in src_data_list[src_idx]["scores"]
                        if sys not in args.systems_to_filter
                    ]
                )
                added_src_ids.add(src_idx)
        assert (
            len(
                set(
                    len(sorted_src_data_list)
                    for sorted_src_data_list in lp2sorted_src_data_list.values()
                )
            )
            == 1
        )
    else:
        for src_idx in [
            src_idx
            for src_idx, src_data in sorted(
                enumerate(next(iter(data.lp2src_data_list.values()))),
                key=lambda pair: get_src_score(pair[1], args.scorer_name),
            )
        ]:
            sorted_human_scores.append(
                [
                    src_data_list[src_idx]["scores"][sys]["human"]
                    for src_data_list in data.lp2src_data_list.values()
                    for sys in src_data_list[src_idx]["scores"]
                    if sys not in args.systems_to_filter
                ]
            )

    plot_human_scores_hist(
        sorted_human_scores,
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
