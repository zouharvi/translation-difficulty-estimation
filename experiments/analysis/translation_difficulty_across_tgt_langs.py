import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
import random
from typing import Tuple, List, Optional, Callable, Literal, Dict

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from mt_metrics_eval.data import EvalSet
from scipy.stats import kendalltau, spearmanr, pearsonr

from difficulty_sampling import ROOT
from difficulty_sampling.data import Data

logging.basicConfig(level=logging.INFO)


dataset2annotations_to_use = {
    "wmt20": {
        "en-de": {
            "scores": {
                "mqm-col1",
                "mqm-col2",
                "mqm-col3",
                "psqm-rater1",
                "psqm-rater2",
                "psqm-rater3",
                "psqm-rater4",
                "psqm-rater5",
                "psqm-rater6",
            }
        }
    },
    "wmt22": {
        "en-de": {"ratings": {"mqm.merged", "round2.mqm.merged", "round3.mqm.merged"}},
        "en-zh": {
            "ratings": {
                "mqm.rater1",
                "mqm.rater2",
                "mqm.rater3",
                "mqm.rater4",
                "mqm.rater5",
                "mqm.rater6",
                "mqm.rater7",
                "mqm.rater8",
                "mqm.rater9",
            }
        },
        "en-ru": {"scores": {"mqm"}},
    },
    "wmt23": {
        "en-de": {"ratings": {"mqm.merged", "round2.mqm.merged", "round3.mqm.merged"}}
    },
    "wmt24": {"en-es": {"scores": {"mqm", "esa"}}, "en-de": {"scores": {"mqm"}}},
}
annotation_name_mapping = {
    "mqm.merged": "mqm-col1",
    "round2.mqm.merged": "mqm-col2",
    "round3.mqm.merged": "mqm-col3",
}


def read_arguments() -> ArgumentParser:
    parser = ArgumentParser(
        description="Command to compute and save on a file the translation difficulty correlation matrix across en-x "
        "target languages."
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
        help="Domains to take into account. If not specified, all domains are considered ('all'). Default: 'all'.",
    )

    parser.add_argument(
        "--systems-to-filter",
        type=str,
        nargs="+",
        help="Systems to exclude from the analysis.",
    )

    parser.add_argument(
        "--correlation",
        type=str,
        choices=["kendall", "spearman", "pearson"],
        default="kendall",
        help="Correlation function to use. Allowed values: 'kendall', 'spearman', 'spearman'. Default: 'kendall'.",
    )

    parser.add_argument(
        "--out-plot-path",
        type=Path,
        help="Local file system path where to save the output plot with the resulting correlation matrix.",
    )

    parser.add_argument(
        "--compute-upper-bounds",
        action="store_true",
        help="Whether to compute several correlations upper bounds starting from MT Eval datasets with multiple "
        "parallel annotations. If passed, all the above arguments will be ignored.",
    )

    return parser


def get_correlation_function(
    method: Literal["kendall", "spearman", "pearson"] = "kendall"
) -> Callable:
    """
    Returns the appropriate correlation function based on the specified method.

    Args:
        method (Literal["kendall", "spearman", "pearson"]): Correlation method name. Default: "kendall".

    Returns:
        method (Callable): Selected correlation callable function from `scipy.stats`.
    """
    if method == "kendall":
        return kendalltau
    elif method == "spearman":
        return spearmanr
    elif method == "pearson":
        return pearsonr
    else:
        raise ValueError(
            f"Unsupported correlation method: {method}. Allowed values: 'kendall', 'spearman', 'pearson'."
        )


def compute_correlation_matrix(
    annotation_id2src_data_list: Dict[str, List[Dict]],
    correlation_method: Literal["kendall", "spearman", "pearson"],
    systems_to_filter: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Computes the correlation matrix for source text avg human scores across multiple annotations.

    Args:
        annotation_id2src_data_list (Dict[str, List[Dict]]): The several annotations to consider for the corr matrix.
        correlation_method (Literal["kendall", "spearman", "pearson"]): The correlation method to use.
        systems_to_filter (Optional[List[str]]): Systems to exclude from the analysis. Default: None.

    Returns:
        np.ndarray: The computed correlation matrix.
        List[str]: The list of annotation string names.
    """
    annotations = sorted(
        annotation_id2src_data_list.keys()
    )  # Get all annotation string names
    num_annotations = len(annotations)

    correlation_func = get_correlation_function(correlation_method)
    annotation2avg_human_scores = {annotation_id: [] for annotation_id in annotations}

    n_sources = 0
    if systems_to_filter is None:
        systems_to_filter = []
    for annotation in annotations:
        src_data_list = annotation_id2src_data_list[annotation]
        if n_sources == 0:
            n_sources = len(src_data_list)
            logging.info(f"Number of source texts: {n_sources}.")
        elif n_sources != len(src_data_list):
            raise ValueError(
                f"Not all input annotations contain the same number of source texts! Found {n_sources} and "
                f"{len(src_data_list)}."
            )

        # Extract average human scores for each source text.
        annotation2avg_human_scores[annotation] = [
            np.mean(
                [
                    sys_scores["human"]
                    for sys, sys_scores in sample["scores"].items()
                    if sys not in systems_to_filter
                ]
            )
            for sample in src_data_list
        ]

    # Compute the correlation matrix
    correlation_matrix = np.zeros((num_annotations, num_annotations))
    for i, annotation1 in enumerate(annotations):
        for j, annotation2 in enumerate(annotations):
            if i < j:
                corr_value, _ = correlation_func(
                    annotation2avg_human_scores[annotation1],
                    annotation2avg_human_scores[annotation2],
                )
                correlation_matrix[i, j] = corr_value
                correlation_matrix[j, i] = corr_value  # Symmetric matrix

    # Fill diagonal with 1s (self-correlation)
    np.fill_diagonal(correlation_matrix, 1.0)

    return correlation_matrix, annotations


def get_mt_eval_annotation_sys2seg_scores(
    eval_set: EvalSet,
    annotation_type: Literal["scores", "ratings"],
    annotation_name: str,
    sorted_seg_indexes: Optional[List[int]] = None,
) -> Dict[str, List[Optional[float]]]:
    """
    Returns the human scores for each system for each source text in the given MT Eval dataset.

    Args:
        eval_set (EvalSet): The MT Eval dataset object.
        annotation_type (Literal["scores", "ratings"]): Annotation type to load. Allowed values: 'scores', 'ratings'.
        annotation_name (str): The name of the annotation to use.
        sorted_seg_indexes (Optional[List[int]]): Sorted indexes to ensure the same order across all lps. Default: None.

    Returns:
        Dict[str, List[Optional[float]]]: A dictionary mapping system names to lists of human scores.
    """
    if annotation_type != "scores" and annotation_type != "ratings":
        raise ValueError(
            f"Invalid `annotation_type` ({annotation_type})! Allowed values: 'scores', 'ratings'."
        )

    sys2seg_scores = dict()

    if annotation_type == "scores":
        for sys, seg_scores in eval_set.Scores("seg", annotation_name).items():
            if sys in eval_set.human_sys_names:
                continue

            sys2seg_scores[sys] = []

            for score in seg_scores:
                if score is not None and (
                    ("mqm" in annotation_name and score > 0)
                    or ("psqm" in annotation_name and score < 0)
                ):
                    sys2seg_scores[sys].append(-score)
                else:
                    sys2seg_scores[sys].append(score)

    else:
        for sys, error_spans in eval_set.Ratings(annotation_name).items():
            if sys in eval_set.human_sys_names:
                continue

            sys2seg_scores[sys] = []

            for rating in error_spans:
                sys2seg_scores[sys].append(
                    sum(-error.score for error in rating.errors)
                    if rating is not None
                    else None
                )

    if sorted_seg_indexes is not None:
        for sys, seg_scores in sys2seg_scores.items():
            assert len(seg_scores) == len(sorted_seg_indexes)
            # Sort float_list according to sorted_indices
            sys2seg_scores[sys] = [
                seg_scores[seg_idx] for seg_idx in sorted_seg_indexes
            ]

    return sys2seg_scores


def merge_mt_eval_annotations(
    annotations_to_merge: List[Dict[str, List[Optional[float]]]],
    n_annotations_per_seg: int = 3,
) -> List[Dict[str, List[float]]]:
    """
    Merges multiple MT Eval annotations into `n_annotations_per_seg` dictionaries.

    Args:
        annotations_to_merge (List[Dict[str, List[Optional[float]]]]): List of annotations to merge.
        n_annotations_per_seg (int): Number of annotations to merge per segment. Default: 3.

    Returns:
        List[Dict[str, List[float]]]: List of three merged MT Eval annotations.
    """
    # Extract MT system names (same across all annotations to merge).
    mt_systems = annotations_to_merge[0].keys()
    num_segs = len(
        next(iter(annotations_to_merge[0].values()))
    )  # All annotations to merge contain the same number of segments.

    # Initialize three output dictionaries (merged annotations).
    merged_annotations = [
        {sys: [None] * num_segs for sys in mt_systems}
        for _ in range(n_annotations_per_seg)
    ]

    for sys in mt_systems:
        for seg_idx in range(num_segs):
            # Collect all non-None scores at the `seg_idx` segment index for the `sys` MT system.
            available_scores = [
                sys2seg_scores[sys][seg_idx]
                for sys2seg_scores in annotations_to_merge
                if sys2seg_scores[sys][seg_idx] is not None
            ]

            if len(available_scores) != n_annotations_per_seg:
                raise ValueError(
                    f"Expected exactly {n_annotations_per_seg} non-None scores, but found {len(available_scores)} at "
                    f"segment index {seg_idx} for MT system {sys}!"
                )

            # Shuffle the scores to ensure randomness before assigning.
            random.shuffle(available_scores)

            # Assign the scores to the three merged annotations.
            for merged_annotation_idx in range(n_annotations_per_seg):
                merged_annotations[merged_annotation_idx][sys][
                    seg_idx
                ] = available_scores[merged_annotation_idx]

    return merged_annotations


def get_upper_bounds() -> (
    Tuple[List[np.ndarray], List[List[str]], List[Path], List[str]]
):
    """
    Get correlation matrices data associated with the upper bounds computation.

    Returns:
        Tuple[List[np.ndarray], List[List[str]], List[Path], List[str]]: Corr matrices, axes list, output paths, corrs.
    """
    # Set random seed for reproducibility across a single `get_upper_bounds` call.
    random.seed(42)

    upper_bound_plots_dir_path = (
        ROOT / "data" / "plots" / "translation_difficulty_corr_matrix" / "upper_bounds"
    )

    correlation_matrices, axes_list, output_paths, correlation_methods = [], [], [], []
    for dataset_name, lp2annotations in dataset2annotations_to_use.items():
        lp2annotation_scores = dict()

        for lp, annotation_type2annotation_names in lp2annotations.items():
            assert len(annotation_type2annotation_names) == 1

            lp2annotation_scores[lp] = dict()
            annotation_type, annotation_names = next(
                iter(annotation_type2annotation_names.items())
            )
            eval_set = EvalSet(
                dataset_name, lp, read_stored_ratings=annotation_type == "ratings"
            )
            sorted_seg_indexes = None
            if len(lp2annotations) > 1:
                # Get the sorted src indexes to ensure the same order across all language pair annotations.
                sorted_seg_indexes = sorted(
                    range(len(eval_set.src)), key=lambda i: eval_set.src[i]
                )

            annotations_to_merge, merged_annotation_name = [], None
            for annotation_name in annotation_names:
                sys2seg_scores = get_mt_eval_annotation_sys2seg_scores(
                    eval_set, annotation_type, annotation_name, sorted_seg_indexes
                )
                if "rater" in annotation_name:
                    annotations_to_merge.append(sys2seg_scores)
                    merged_annotation_name = (
                        "psqm" if annotation_name.startswith("psqm") else "mqm"
                    )
                else:
                    lp2annotation_scores[lp][
                        annotation_name_mapping.get(annotation_name, annotation_name)
                    ] = sys2seg_scores

            if len(annotations_to_merge) > 0:
                merged_annotations = merge_mt_eval_annotations(annotations_to_merge)
                for merged_annotation_idx, merged_annotation in enumerate(
                    merged_annotations, start=1
                ):
                    lp2annotation_scores[lp][
                        f"{merged_annotation_name}-col{merged_annotation_idx}"
                    ] = merged_annotation

        for lp, annotation_name2scores in lp2annotation_scores.items():
            for annotation_name, sys2seg_scores in annotation_name2scores.items():
                annotation_name2scores[annotation_name] = {
                    sys: seg_scores
                    for sys, seg_scores in sys2seg_scores.items()
                    if len(seg_scores) > 0
                    and any(score is not None for score in seg_scores)
                }

        segs_intersection = None
        for lp, annotation_name2scores in lp2annotation_scores.items():
            for sys2seg_scores in annotation_name2scores.values():
                valid_seg_indexes = []
                for seg_idx, seg_score in enumerate(
                    next(iter(sys2seg_scores.values()))
                ):
                    if seg_score is not None:
                        valid_seg_indexes.append(seg_idx)
                valid_seg_indexes = set(valid_seg_indexes)
                if segs_intersection is None:
                    segs_intersection = valid_seg_indexes
                else:
                    segs_intersection &= valid_seg_indexes

        segs_intersection = sorted(segs_intersection)
        logging.info("\n")
        logging.info(
            f"Number of segments for {dataset_name} dataset: {len(segs_intersection)}."
        )
        for lp, annotation_name2scores in lp2annotation_scores.items():
            mt_systems_intersection = None
            for annotation_name, sys2seg_scores in annotation_name2scores.items():
                for sys, seg_scores in sys2seg_scores.items():
                    sys2seg_scores[sys] = [
                        seg_scores[seg_idx] for seg_idx in segs_intersection
                    ]

                if mt_systems_intersection is None:
                    mt_systems_intersection = set(sys2seg_scores.keys())
                else:
                    mt_systems_intersection &= set(sys2seg_scores.keys())

            logging.info(
                f"Number of MT systems for {dataset_name} dataset and {lp} language pair: "
                f"{len(mt_systems_intersection)}."
            )
            for annotation_name, sys2seg_scores in annotation_name2scores.items():
                annotation_name2scores[annotation_name] = {
                    sys: sys2seg_scores[sys] for sys in mt_systems_intersection
                }
        logging.info("\n")

        annotation_id2src_data_list = dict()
        for lp, annotation_name2scores in lp2annotation_scores.items():
            for annotation_name, sys2seg_scores in annotation_name2scores.items():
                src_data_list = []
                for sys, seg_scores in sys2seg_scores.items():
                    for seg_idx, seg_score in enumerate(seg_scores):
                        if len(src_data_list) <= seg_idx:
                            src_data_list.append({"scores": dict()})
                        assert sys not in src_data_list[seg_idx]["scores"]
                        assert seg_score is not None
                        src_data_list[seg_idx]["scores"][sys] = {"human": seg_score}
                annotation_id2src_data_list[f"{lp}.{annotation_name}"] = src_data_list

        for correlation_method in ["kendall", "spearman", "pearson"]:
            correlation_matrix, annotations = compute_correlation_matrix(
                annotation_id2src_data_list, correlation_method
            )

            correlation_matrices.append(correlation_matrix)

            axes_list.append(annotations)

            output_path = (
                upper_bound_plots_dir_path
                / f"{dataset_name}"
                / f"{correlation_method}.png"
            )
            # Create the directories before the file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_paths.append(output_path)

            correlation_methods.append(correlation_method)

    return correlation_matrices, axes_list, output_paths, correlation_methods


def save_correlation_plot(
    correlation_matrix: np.ndarray,
    axes: List[str],
    axes_name: str,
    output_path: Path,
    correlation_method: str,
) -> None:
    """
    Saves the correlation matrix as a heatmap plot.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        xticklabels=axes,
        yticklabels=axes,
        vmin=-1,
        vmax=1,
    )
    plt.title(
        f"{correlation_method.capitalize()} Correlation Matrix for Source Text Difficulty Scores"
    )
    plt.xlabel(axes_name)
    plt.ylabel(axes_name)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    # Save plot
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def translation_difficulty_across_tgt_langs_command() -> None:
    args: Namespace = read_arguments().parse_args()

    if args.compute_upper_bounds:
        # Compute correlation matrices
        (
            correlation_matrices,
            axes_list,
            output_paths,
            correlation_methods,
        ) = get_upper_bounds()
        assert (
            len(correlation_matrices)
            == len(axes_list)
            == len(output_paths)
            == len(correlation_methods)
        )
        for correlation_matrix, axes, output_path, correlation_method in zip(
            correlation_matrices, axes_list, output_paths, correlation_methods
        ):
            # Save the plot
            save_correlation_plot(
                correlation_matrix,
                axes,
                "MT Eval Annotation",
                output_path,
                correlation_method,
            )

    else:
        # Compute correlation matrix
        correlation_matrix, annotations = compute_correlation_matrix(
            Data.load(
                dataset_name="wmt24",
                lps=["en-x"],
                protocol=args.protocol,
                domains=args.domains,
            ).lp2src_data_list,
            args.correlation,
            args.systems_to_filter,
        )

        # Save the plot
        save_correlation_plot(
            correlation_matrix,
            annotations,
            "Language Pair",
            args.out_plot_path,
            args.correlation,
        )


if __name__ == "__main__":
    translation_difficulty_across_tgt_langs_command()
