import logging
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import subset2evaluate

from difficulty_sampling.data import Data


logger = logging.getLogger(__name__)


tgt_lang2lp = {
    "chinese": "en-zh",
    "czech": "en-cs",
    "hindi": "en-hi",
    "icelandic": "en-is",
    "japanese": "en-ja",
    "russian": "en-ru",
    "spanish": "en-es",
    "ukrainian": "en-uk",
}


def apply_src_len(data):
    for data in data.lp2src_data_list.values():
        for line in data:
            for sys in line["scores"].keys():
                line["scores"][sys]["src_len"] = -len(line["src"])


def apply_subset2evaluate_cache(data, method):
    load_model = None
    for data in data.lp2src_data_list.values():
        scores, load_model = subset2evaluate.methods.METHODS[method](
            data, return_model=True, load_model=load_model
        )
        for line in data:
            score = scores.pop(0)
            for sys in line["scores"].keys():
                line["scores"][sys][method] = -score


def apply_subset2evaluate(data, method):
    for data in data.lp2src_data_list.values():
        scores = subset2evaluate.methods.METHODS[method](data)
        for line in data:
            score = scores.pop(0)
            for sys in line["scores"].keys():
                line["scores"][sys][method] = -score


def apply_artificial_crowd_metrics(data, model, metric):
    for data in data.lp2src_data_list.values():
        for line in data:
            score = line["scores"][model][metric]
            for sys in line["scores"].keys():
                line["scores"][sys]["artcrowd|" + model + "|" + metric] = score


def apply_external_artificial_crowd_metrics(
    data: Data,
    sys2translations_path: Path,
    metric: Literal["MetricX-24-Hybrid-QE-XXL", "XCOMET-QE-XXL", "CometKiwi-XXL"],
) -> None:
    """
    Add External Artificial Crowd scores to the input data.

    Args:
        data: Data to add the External Artificial Crowd scores to.
        sys2translations_path: Path to the pickle file containing the scored system translations.
        metric: QE metric to use for the External Artificial Crowd scores.
    """
    if (
        metric != "MetricX-24-Hybrid-QE-XXL"
        and metric != "XCOMET-QE-XXL"
        and metric != "CometKiwi-XXL"
    ):
        raise ValueError(
            f"Invalid metric '{metric}'! The metric must be one of 'MetricX-24-Hybrid-QE-XXL', 'XCOMET-QE-XXL', or "
            "'CometKiwi-XXL'."
        )

    with sys2translations_path.open("rb") as handle:
        sys2translations = pickle.load(handle)

    lp2artificial_crowd_metric_scores = dict()
    for tgt_lang2scored_translations in sys2translations.values():
        for tgt_lang, scores in tgt_lang2scored_translations.items():
            lp = tgt_lang2lp[tgt_lang]
            if lp not in lp2artificial_crowd_metric_scores:
                lp2artificial_crowd_metric_scores[lp] = []
            for seg_idx, score_dict in enumerate(scores):
                if len(lp2artificial_crowd_metric_scores[lp]) <= seg_idx:
                    lp2artificial_crowd_metric_scores[lp].append(
                        {
                            "MetricX-24-Hybrid-QE-XXL": [
                                score_dict["metricx24-hybrid-qe_score"]
                            ],
                            "XCOMET-QE-XXL": [score_dict["xcomet-qe_score"]],
                            "CometKiwi-XXL": [score_dict["cometkiwi-xxl_score"]],
                        }
                    )
                else:
                    lp2artificial_crowd_metric_scores[lp][seg_idx][
                        "MetricX-24-Hybrid-QE-XXL"
                    ].append(score_dict["metricx24-hybrid-qe_score"])
                    lp2artificial_crowd_metric_scores[lp][seg_idx][
                        "XCOMET-QE-XXL"
                    ].append(score_dict["xcomet-qe_score"])
                    lp2artificial_crowd_metric_scores[lp][seg_idx][
                        "CometKiwi-XXL"
                    ].append(score_dict["cometkiwi-xxl_score"])

    for lp, src_data_list in data.lp2src_data_list.items():
        assert len(src_data_list) == len(lp2artificial_crowd_metric_scores[lp])
        for line, artificial_crowd_metric_scores in zip(
            src_data_list, lp2artificial_crowd_metric_scores[lp]
        ):
            score = sum(artificial_crowd_metric_scores[metric]) / len(
                artificial_crowd_metric_scores[metric]
            )
            for sys in line["scores"].keys():
                line["scores"][sys]["ext_artcrowd|" + metric] = score


def apply_llm_as_a_judge(
    data: Data, scored_source_texts_df_path: Path, llm_name: str
) -> None:
    """
    Add LLM-as-a-judge scores to the input data.

    Args:
        data: Data to add the LLM-as-a-judge scores to.
        scored_source_texts_df_path: Path to the CSV file containing the scored source texts pandas dataframe.
        llm_name: Name of the LLM used for scoring.
    """
    scored_source_texts = pd.read_csv(scored_source_texts_df_path, na_filter=False)
    scored_source_texts["numeric_score"] = pd.to_numeric(
        scored_source_texts["numeric_score"], errors="coerce"
    )

    source_text_scores, lp2source_text_scores = None, None
    n_nan_scores = 0
    if "src_lang" in scored_source_texts.columns:
        source_text_scores = scored_source_texts[
            scored_source_texts["src_lang"] == "en"
        ]["numeric_score"].tolist()
        n_nan_scores = sum(np.isnan(score) for score in source_text_scores)
    else:
        lp2source_text_scores = defaultdict(list)
        for _, row in scored_source_texts.iterrows():
            lp = row["lp"]
            if lp.startswith("en-"):
                lp2source_text_scores[lp].append(row["numeric_score"])
                if np.isnan(row["numeric_score"]):
                    n_nan_scores += 1
    scores_type = "src" if source_text_scores is not None else "tgt"
    logger.info(
        f"Number of NaN scores in LLM-as-a-Judge predictions with {llm_name} LLM ({scores_type}-based): {n_nan_scores}."
    )

    for lp, src_data_list in data.lp2src_data_list.items():
        if lp2source_text_scores is not None:
            source_text_scores = lp2source_text_scores[lp]
        assert len(src_data_list) == len(source_text_scores)
        for line, score in zip(src_data_list, source_text_scores):
            for sys in line["scores"].keys():
                line["scores"][sys][
                    f"LLM-as-a-Judge ({llm_name}, {scores_type}-based)"
                ] = -(
                    score if not np.isnan(score) else 60
                )  # LLM-as-a-Judge returns difficulty scores.
