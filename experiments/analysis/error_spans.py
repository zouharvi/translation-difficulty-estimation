import json
import logging
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from collections import Counter, defaultdict

from typing import Dict, List, Union, TypedDict, Optional

from difficulty_sampling import wmt24_from_en_lps_mqm

from mt_metrics_eval import data


logging.basicConfig(level=logging.INFO)


class EsaSpan(TypedDict, total=False):
    start_i: int
    end_i: int
    severity: str
    error_type: Optional[str]


class EsaAnnotation(TypedDict, total=False):
    langs: int
    line_id: int
    src: str
    tgt: str
    doc_id: str
    domain: str
    esa_spans: List[EsaSpan]
    esa_score: int
    system: str
    annotator: str
    speech_info: Optional[str]


def read_arguments() -> ArgumentParser:
    parser = ArgumentParser(
        description="Command to perform the error spans analysis on WMT24 submissions for the news domain."
    )

    parser.add_argument(
        "--wmt24-esa-jsonl-path",
        type=Path,
        help="Path to the jsonl file containing the WMT24 ESA annotations. If not passed, the analysis will be "
        "conducted on the WMT24 MQM error spans.",
    )

    parser.add_argument(
        "--easy-systems-threshold",
        type=float,
        default=0.8,
        help="Proportion of systems that need to classify a document as easy for it to be considered 'EASY'. "
        "Default: 0.8",
    )

    parser.add_argument(
        "--hard-systems-threshold",
        type=float,
        default=0.8,
        help="Proportion of systems that need to classify a document as hard for it to be considered 'HARD'. "
        "Default: 0.8",
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
        help="Local file system path where to save the output plot.",
    )

    parser.add_argument(
        "--granularity",
        choices=["segment", "document"],
        default="segment",
        help="What to consider as a single 'item' in the analysis. If MQM analysis is conducted, this argument will be "
        "ignored, and such analysis will be with respect to segments. Allowed values: 'segment', 'document'. "
        "Default: 'segment'.",
    )

    return parser


def segment_error_category(esa_spans: List[EsaSpan]) -> str:
    """
    Determine the overall category of a single segment's error spans.
    Returns 'no_error', 'minor_error', or 'major_error'.

    Args:
        esa_spans (List[EsaSpan]): ESA error spans annotations for a single segment.

    Returns:
        str: The overall category of the segment's error spans.
    """
    if len(esa_spans) == 0:  # no error spans => no_error
        return "no_error"

    # If any span is major => major_error
    for span in esa_spans:
        assert (
            span["severity"] == "minor"
            or span["severity"] == "major"
            or span["severity"] == "undecided"
        )

        if span["severity"] == "major":
            return "major_error"

    return "minor_error"


def analyze_docs_condition(
    doc_id2sys_translations: Dict[str, Dict[str, Dict[str, List[EsaAnnotation]]]],
    easy_systems_threshold: float = 0.8,
    hard_systems_threshold: float = 0.8,
) -> Dict[
    str, Dict[str, Dict[str, Union[Dict[str, Dict[str, int]], Dict[str, str], str]]]
]:
    """
    Classifies each document under strict rules:
      - 'easy' if:
         major_error_count == 0
         AND minor_error_count <= total_segments
      - 'hard' if:
         major_error_count >= total_segments // 2
      - otherwise 'mixed' at system-level.

    Then aggregates system-level classification to doc-level:
      - 'EASY' if >= 80% systems see doc as easy
      - 'HARD' if >= 80% systems see doc as hard
      - else 'MIXED'

    Args:
        doc_id2sys_translations: Dictionary containing wmt24 human annotations.
        easy_systems_threshold: Proportion of systems that need to classify a document as easy for 'EASY' doc class.
        hard_systems_threshold: Proportion of systems that need to classify a document as hard for 'HARD' doc class.

    Returns:
        Dict: Dictionary containing document-level classification results.
    """
    # Return structure
    analysis_res = dict()

    for doc_id, lp_dict in doc_id2sys_translations.items():
        analysis_res[doc_id] = dict()

        for lp, sys_dict in lp_dict.items():
            doc_stats_per_system = dict()

            # 1) Collect stats for each system
            for system_name, seg_list in sys_dict.items():
                total_segments = len(seg_list)
                cat_counts = Counter()

                # Tally up errors for each segment
                for seg_info in seg_list:
                    esa_spans = seg_info["esa_spans"]
                    cat = segment_error_category(esa_spans)
                    cat_counts[cat] += 1

                doc_stats_per_system[system_name] = {
                    "no_error_count": cat_counts["no_error"],
                    "minor_error_count": cat_counts["minor_error"],
                    "major_error_count": cat_counts["major_error"],
                    "total_segments": total_segments,
                }

            # 2) System-level classification (easy/hard/mixed)
            system_classification = dict()
            for system_name, stats in doc_stats_per_system.items():
                total_segs = stats["total_segments"]
                minor_count = stats["minor_error_count"]
                major_count = stats["major_error_count"]

                # easy condition:
                #  no major errors
                #  at most 1 minor error per segment => minor_count <= total_segs
                is_easy = (major_count == 0) and (minor_count <= total_segs)

                # hard condition:
                #  major_count >= total_segments // 2 => at least as many major errors as half the number of segments
                is_hard = major_count >= total_segs // 2

                if is_easy:
                    system_classification[system_name] = "easy"
                elif is_hard:
                    system_classification[system_name] = "hard"
                else:
                    system_classification[system_name] = "mixed"

            # 3) Aggregate to doc-level
            num_systems = len(system_classification)
            assert num_systems > 0
            easy_count = sum(val == "easy" for val in system_classification.values())
            hard_count = sum(val == "hard" for val in system_classification.values())

            if easy_count >= easy_systems_threshold * num_systems:
                doc_classification = "EASY"
            elif hard_count >= hard_systems_threshold * num_systems:
                doc_classification = "HARD"
            else:
                doc_classification = "MIXED"

            analysis_res[doc_id][lp] = {
                "doc_stats_per_system": doc_stats_per_system,
                "system_classification": system_classification,
                "classification": doc_classification,
            }

    return analysis_res


def analyze_segments_condition(
    doc_id2sys_translations: Dict[str, Dict[str, Dict[str, List[EsaAnnotation]]]],
    easy_systems_threshold: float = 0.8,
    hard_systems_threshold: float = 0.8,
) -> Dict[str, Dict[str, Dict[int, Dict[str, Union[Dict[str, str], str]]]]]:
    """
    Groups annotations by document, language pair, and line_id.
    For each segment (identified by line_id), each system's annotation is classified as follows:
      - 'hard' if the annotation contains any major error or 5 or more minor errors.
      - 'easy' if it contains no major errors and at most one minor error.
      - 'mixed' if it contains no major errors and the number of minor errors is greater than 1 and less than 5.
    Then, for each segment, the overall classification is:
      - 'EASY' if >= easy_systems_threshold classify it as easy,
      - 'HARD' if >= hard_systems_threshold classify it as hard,
      - otherwise 'MIXED'.

    Args:
        doc_id2sys_translations: Dictionary containing wmt24 human annotations.
        easy_systems_threshold: Proportion of systems that need to classify a segment as easy for 'EASY' seg class.
        hard_systems_threshold: Proportion of systems that need to classify a segment as hard for 'HARD' seg class.

    Returns:
        Dict: Dictionary containing segment-level classification results.
    """
    analysis_seg_res = dict()
    for doc_id, lp_dict in doc_id2sys_translations.items():
        analysis_seg_res[doc_id] = dict()
        for lp, sys_dict in lp_dict.items():
            # Group annotations by line_id across systems
            line_id_groups = defaultdict(
                list
            )  # line_id -> list of tuples (system, annotation)
            for system_name, seg_list in sys_dict.items():
                for seg in seg_list:
                    line_id = seg["line_id"]
                    line_id_groups[line_id].append((system_name, seg))
            seg_results = dict()
            for line_id, annotations in line_id_groups.items():
                system_classification = dict()
                for system, annotation in annotations:
                    esa_spans = annotation["esa_spans"]
                    major_errors = sum(
                        1 for span in esa_spans if span["severity"] == "major"
                    )
                    minor_errors = sum(
                        1 for span in esa_spans if span["severity"] == "minor"
                    )

                    if major_errors > 0:
                        system_classification[system] = "hard"
                    else:
                        # No major errors; now check number of minor errors.
                        if minor_errors <= 1:
                            system_classification[system] = "easy"
                        elif 1 < minor_errors < 5:
                            system_classification[system] = "mixed"
                        else:  # minor_errors >= 5
                            system_classification[system] = "hard"
                num_systems = len(system_classification)
                assert num_systems > 0
                easy_count = sum(
                    1 for v in system_classification.values() if v == "easy"
                )
                hard_count = sum(
                    1 for v in system_classification.values() if v == "hard"
                )
                if easy_count >= easy_systems_threshold * num_systems:
                    seg_classification = "EASY"
                elif hard_count >= hard_systems_threshold * num_systems:
                    seg_classification = "HARD"
                else:
                    seg_classification = "MIXED"
                seg_results[line_id] = {
                    "system_classification": system_classification,
                    "classification": seg_classification,
                }
            analysis_seg_res[doc_id][lp] = seg_results
    return analysis_seg_res


def analyze_mqm_segments_condition(
    easy_systems_threshold: float = 0.8, hard_systems_threshold: float = 0.8
) -> Dict:
    """
    For each segment, each system's annotation is classified as follows:
      - 'hard' if the annotation contains any major error or 5 or more minor errors.
      - 'easy' if it contains no major errors and at most one minor error.
      - 'mixed' if it contains no major errors and the number of minor errors is greater than 1 and less than 5.
    Then, for each segment, the overall classification is:
      - 'EASY' if >= easy_systems_threshold classify it as easy,
      - 'HARD' if >= hard_systems_threshold classify it as hard,
      - otherwise 'MIXED'.

    Args:
        easy_systems_threshold: Proportion of systems that need to classify a segment as easy for 'EASY' seg class.
        hard_systems_threshold: Proportion of systems that need to classify a segment as hard for 'HARD' seg class.

    Returns:
        Dict: Dictionary containing segment-level classification results.
    """
    analysis_seg_res = dict()

    # Iterate over each language pair
    for lp in wmt24_from_en_lps_mqm:
        analysis_seg_res[lp] = dict()

        eval_set = data.EvalSet("wmt24", lp, read_stored_ratings=True)
        domains_per_seg = eval_set.DomainsPerSeg()
        # Obtain merged MQM ratings per system for this language pair
        # This returns a dict: system -> list of ratings (one per segment)
        system_ratings = eval_set.Ratings("mqm.merged")

        # Group annotations by segment (using line_id)
        seg_groups = defaultdict(list)  # line_id -> list of tuples (system, rating)
        for sys, ratings in system_ratings.items():
            assert len(ratings) == len(eval_set.src) == len(domains_per_seg)
            for line_id, (rating, domain) in enumerate(zip(ratings, domains_per_seg)):
                if rating is None or domain != "news":
                    continue
                seg_groups[line_id].append((sys, rating))

        # For each segment, aggregate system classifications
        seg_results = dict()
        for line_id, sys_ratings in seg_groups.items():
            system_classification = dict()
            for sys, rating in sys_ratings:
                # Count errors by severity in this segment
                minor_count = sum(
                    1 for error in rating.errors if error.severity == "minor"
                )
                major_count = sum(
                    1 for error in rating.errors if error.severity == "major"
                )

                # Classify per system:
                # - "hard" if any major error or >= 5 minor errors
                # - "easy" if no major errors and at most 1 minor error
                # - "mixed" if no major errors and  minor errors > 1 and < 5
                if major_count > 0 or minor_count >= 5:
                    system_classification[sys] = "hard"
                elif minor_count <= 1:
                    system_classification[sys] = "easy"
                else:
                    system_classification[sys] = "mixed"

            num_systems = len(system_classification)
            assert num_systems > 0
            easy_count = sum(1 for v in system_classification.values() if v == "easy")
            hard_count = sum(1 for v in system_classification.values() if v == "hard")
            if easy_count >= easy_systems_threshold * num_systems:
                overall_class = "EASY"
            elif hard_count >= hard_systems_threshold * num_systems:
                overall_class = "HARD"
            else:
                overall_class = "MIXED"

            seg_results[line_id] = {
                "system_classification": system_classification,
                "classification": overall_class,
            }
        analysis_seg_res[lp] = seg_results

    return analysis_seg_res


def plot_item_classes(analysis_res: Dict, plot_path: Path) -> None:
    """
    Aggregates item (document or segment) classification counts per language pair and saves a stacked bar chart.

    Args:
        analysis_res (Dict): Classification results.
        plot_path (Path): File path to save the plot.
    """
    counts_per_lp = defaultdict(lambda: defaultdict(int))

    granularity = None
    # We iterate over analysis_res with two possibilities:
    # 1) Top-level keys are language pairs (e.g., MQM segment-level)
    # 2) Top-level keys are document IDs (which may contain either a direct "classification"
    #    or a dict keyed by segment IDs).
    for top_key, lp_dict in analysis_res.items():
        # Determine if top_key is a language pair (str) or a document ID.
        # If top_key is in a predefined set of language pairs (wmt24_from_en_lps_mqm), then it's MQM segment-level.
        if top_key in wmt24_from_en_lps_mqm:
            # MQM segment-level analysis: top-level keys are language pairs.
            lp = top_key
            for seg_id, seg_info in lp_dict.items():
                cls_val = seg_info["classification"]
                counts_per_lp[lp][cls_val] += 1

            granularity = "Segment"
        else:
            # Top-level key is not a language pair, assume it's a document ID.
            # For each language pair in the document...
            for lp, info in lp_dict.items():
                # Check the type of keys in 'info'.
                # If all keys are integers, we assume segment-level analysis.
                if all(isinstance(k, int) for k in info.keys()):
                    for seg_id, seg_info in info.items():
                        cls_val = seg_info["classification"]
                        counts_per_lp[lp][cls_val] += 1

                    granularity = "Segment"
                else:
                    # Otherwise, assume document-level analysis.
                    cls_val = info["classification"]
                    counts_per_lp[lp][cls_val] += 1

                    granularity = "Document"

    # Define the classification classes to display.
    classes = ["EASY", "MIXED", "HARD"]
    lp_list = sorted(counts_per_lp.keys())
    aggregated_data = []
    x_labels = []
    for lp in lp_list:
        row = [counts_per_lp[lp].get(cls, 0) for cls in classes]
        aggregated_data.append(row)
        total = sum(row)
        # Include total count in tick label (important for MQM, where counts may vary).
        x_labels.append(f"{lp} (N={total})")

    data_array = np.array(aggregated_data)
    x = np.arange(len(lp_list))
    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = np.zeros(len(lp_list))
    for i, cls in enumerate(classes):
        ax.bar(x, data_array[:, i], label=cls, bottom=bottom, width=0.6)
        bottom += data_array[:, i]

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_xlabel("Language Pair")
    ylabel = f"Number of {granularity}s"
    ax.set_ylabel(ylabel)
    ax.set_title(f"{granularity} Classification Counts per Language Pair")
    ax.legend()
    plt.tight_layout()
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)


def error_spans_analysis_command() -> None:
    """
    Command to perform the error spans analysis on WMT24 submissions for the news domain.
    """
    args = read_arguments().parse_args()

    if args.wmt24_esa_jsonl_path is not None:
        data = []
        with open(args.wmt24_esa_jsonl_path, encoding="utf-8") as f:
            for line in f:
                # Strip whitespace to avoid blank lines
                line = line.strip()
                if not line:
                    continue  # skip empty lines if any
                # Parse the line as JSON
                obj = json.loads(line)
                data.append(obj)

        systems_to_filter = (
            set(args.systems_to_filter) if args.systems_to_filter else set()
        )
        doc_id2sys_translations, news_lps = dict(), set()
        for segment_annotation in data:
            if (
                segment_annotation["domain"] != "news"
                or segment_annotation["langs"][:2] != "en"
                or segment_annotation["system"] in systems_to_filter
            ):
                continue

            if segment_annotation["doc_id"] not in doc_id2sys_translations:
                doc_id2sys_translations[segment_annotation["doc_id"]] = dict()

            if (
                segment_annotation["langs"]
                not in doc_id2sys_translations[segment_annotation["doc_id"]]
            ):
                doc_id2sys_translations[segment_annotation["doc_id"]][
                    segment_annotation["langs"]
                ] = dict()
            news_lps.add(segment_annotation["langs"])

            if (
                segment_annotation["system"]
                not in doc_id2sys_translations[segment_annotation["doc_id"]][
                    segment_annotation["langs"]
                ]
            ):
                doc_id2sys_translations[segment_annotation["doc_id"]][
                    segment_annotation["langs"]
                ][segment_annotation["system"]] = []

            doc_id2sys_translations[segment_annotation["doc_id"]][
                segment_annotation["langs"]
            ][segment_annotation["system"]].append(segment_annotation)

        logging.info(
            f"Total number of docs in the news domain for wmt24 with ESA annotations: {len(doc_id2sys_translations)}\t"
            f"Total number of language pairs in the news domain for wmt24 with ESA annotations: {len(news_lps)}."
        )

        # Choose analysis based on granularity argument
        if args.granularity == "document":
            analysis_res = analyze_docs_condition(
                doc_id2sys_translations,
                args.easy_systems_threshold,
                args.hard_systems_threshold,
            )
        else:
            analysis_res = analyze_segments_condition(
                doc_id2sys_translations,
                args.easy_systems_threshold,
                args.hard_systems_threshold,
            )

    else:
        analysis_res = analyze_mqm_segments_condition(
            args.easy_systems_threshold, args.hard_systems_threshold
        )

    plot_item_classes(analysis_res, args.out_plot_path)


if __name__ == "__main__":
    error_spans_analysis_command()
