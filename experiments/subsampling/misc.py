import logging
import pickle
from collections import defaultdict
from pathlib import Path
import random
from typing import Literal

import numpy as np
import pandas as pd
import subset2evaluate

from difficulty_estimation.data import Data


logger = logging.getLogger(__name__)


wmt24_encs_mt_sys_ranking = [
    "Unbabel-Tower70B",
    "Claude-3.5",
    "ONLINE-W",
    "CUNI-MH",
    "Gemini-1.5-Pro",
    "GPT-4",
    "CommandR-plus",
    "IOL_Research",
    "SCIR-MT",
    "CUNI-DocTransformer",
    "Aya23",
    "CUNI-GA",
    "IKUN",
    "Llama3-70B",
    "IKUN-C",
]


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


def apply_subset2evaluate(data, method, **kwargs):
    for data in data.lp2src_data_list.values():
        scores = subset2evaluate.methods.METHODS[method](data, **kwargs)
        for line in data:
            score = scores.pop(0)
            for sys in line["scores"].keys():
                line["scores"][sys][method] = -score


def apply_random(data: Data, scorer_name: str = "random", seed: int = 42) -> None:
    """
    Assign a random score to each MT output.

    Args:
        data: Data to add the random scores to.
        scorer_name: Name of the random score to be added. Default: "random".
        seed: Random seed. Default: 42.
    """
    random.seed(seed)

    for src_data_list in data.lp2src_data_list.values():
        for sample in src_data_list:
            for metric2score in sample["scores"].values():
                metric2score[scorer_name] = random.random()


def apply_oracle_with_fixed_scores(
    data: Data,
    scorer_name: str = "oracle_with_fixed_scores",
    use_tgt_lang: bool = False,
) -> None:
    """
    Assign the average human score of each source text to all MT system translations.

    Args:
        data: Data to add the oracle scores to.
        scorer_name: Name of the oracle score to be added. Default: "oracle_with_fixed_scores".
        use_tgt_lang: If True, there will be a single oracle score computation per target language. Default: False.
    """
    src2scores = defaultdict(lambda: defaultdict(float))
    for lp, src_data_list in data.lp2src_data_list.items():
        tgt_lang = lp.split("-")[1]
        for sample in src_data_list:
            src2scores[sample["src"]][tgt_lang] = sum(
                metric2score["human"] for metric2score in sample["scores"].values()
            ) / len(sample["scores"])

    for lp, src_data_list in data.lp2src_data_list.items():
        tgt_lang = lp.split("-")[1]
        for sample in src_data_list:
            oracle_score = (
                src2scores[sample["src"]][tgt_lang]
                if use_tgt_lang
                else sum(src2scores[sample["src"]].values())
                / len(src2scores[sample["src"]])
            )
            for metric2score in sample["scores"].values():
                metric2score[scorer_name] = oracle_score


def apply_internal_artificial_crowd_metrics(
    data: Data,
    model: str,
    metric: str,
) -> None:
    """
    Add Internal Artificial Crowd scores to the input data.

    Args:
        data: Data to add the Internal Artificial Crowd scores to.
        model: Name of the MT model used for Internal Artificial Crowd scores. If "all", all MT models will be used.
        metric: Name of the metric used for Internal Artificial Crowd scores.
    """
    for lp, src_data_list in data.lp2src_data_list.items():
        for sample in src_data_list:
            if model == "all":
                for metric2score in sample["scores"].values():
                    metric2score["artcrowd|" + model + "|" + metric] = metric2score[
                        metric
                    ]

            else:
                curr_model = model
                metric2score_for_model, back_off_model_idx = (
                    sample["scores"].get(curr_model),
                    0,
                )

                while metric2score_for_model is None:
                    assert lp == "en-cs"
                    curr_model = wmt24_encs_mt_sys_ranking[back_off_model_idx]
                    metric2score_for_model = sample["scores"].get(curr_model)
                    back_off_model_idx += 1

                score = metric2score_for_model[metric]

                for metric2score in sample["scores"].values():
                    metric2score["artcrowd|" + model + "|" + metric] = score


def apply_external_artificial_crowd_metrics(
    data: Data,
    sys2translations_path: Path,
    metric: Literal["MetricX-24-Hybrid-QE-XXL", "XCOMET-QE-XXL", "CometKiwi-XXL"],
    protocol: Literal["esa", "mqm"] = "esa",
) -> None:
    """
    Add External Artificial Crowd scores to the input data.

    Args:
        data: Data to add the External Artificial Crowd scores to.
        sys2translations_path: Path to the pickle file containing the scored system translations.
        metric: QE metric to use for the External Artificial Crowd scores.
        protocol: Protocol to use for loading the External Artificial Crowd scores. Default: "esa".
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
    if protocol != "esa" and protocol != "mqm":
        raise ValueError(
            f"Invalid protocol '{protocol}'! The protocol must be either 'esa' or 'mqm'."
        )

    with sys2translations_path.open("rb") as handle:
        sys2translations = pickle.load(handle)

    lp2artificial_crowd_metric_scores = defaultdict(list)
    for protocol2lp_scored_translations in sys2translations.values():
        for lp, scores in protocol2lp_scored_translations[protocol].items():
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
        for sample, artificial_crowd_metric_scores in zip(
            src_data_list, lp2artificial_crowd_metric_scores[lp]
        ):
            score = sum(artificial_crowd_metric_scores[metric]) / len(
                artificial_crowd_metric_scores[metric]
            )
            for metric2score in sample["scores"].values():
                metric2score["ext_artcrowd|" + metric] = score


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
    scored_src_texts = pd.read_csv(scored_source_texts_df_path, na_filter=False)
    scored_src_texts["numeric_score"] = pd.to_numeric(
        scored_src_texts["numeric_score"], errors="coerce"
    )

    src_lang2scores, lp2scores, n_nan_scores = None, None, 0
    if "src_lang" in scored_src_texts.columns:
        needed_src_langs = {lp.split("-")[0] for lp in data.lps}

        src_lang2scores = defaultdict(list)

        for _, row in scored_src_texts.iterrows():
            if row["src_lang"] in needed_src_langs:
                src_lang2scores[row["src_lang"]].append(row["numeric_score"])
                n_nan_scores += np.isnan(row["numeric_score"])

    else:
        needed_lps = set(data.lps)

        lp2scores = defaultdict(list)

        for _, row in scored_src_texts.iterrows():
            if row["lp"] in needed_lps:
                lp2scores[row["lp"]].append(row["numeric_score"])
                n_nan_scores += np.isnan(row["numeric_score"])

    logger.info(
        f"Number of NaN scores in LLM-as-a-Judge predictions with {llm_name} LLM: {n_nan_scores}."
    )

    for lp, src_data_list in data.lp2src_data_list.items():
        src_text_scores = (
            src_lang2scores[lp.split("-")[0]] if src_lang2scores else lp2scores[lp]
        )

        median_score = np.nanmedian(src_text_scores)
        if np.isnan(median_score):
            median_score = 60

        assert len(src_data_list) == len(src_text_scores)

        for sample, score in zip(src_data_list, src_text_scores):
            for metric2score in sample["scores"].values():
                metric2score[f"LLM-as-a-Judge ({llm_name})"] = -(
                    score if not np.isnan(score) else median_score
                )  # LLM-as-a-Judge returns difficulty scores.
