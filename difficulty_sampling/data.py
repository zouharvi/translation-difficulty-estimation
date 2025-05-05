import collections
import json
import logging

from mt_metrics_eval.data import EvalSet
import numpy as np
from typing import Dict, List, Union, TypedDict, Optional

from difficulty_sampling import WMT24_GENMT_DATA
from difficulty_sampling.utils import load_data_wmt


logger = logging.getLogger(__name__)


class SrcData(TypedDict, total=False):
    i: int
    src: str
    ref: str
    tgt: Dict[str, str]  # System translations.
    cost: float
    domain: str
    doc: str
    scores: Dict[
        str, Dict[str, float]
    ]  # Segment-level scores given by metrics to systems.


class Data:
    def __init__(
        self,
        lp2src_data_list: Dict[str, List[SrcData]],
        lps: List[str],
        dataset_name: str,
        protocol: str,
        domains: str,
    ):
        self.lp2src_data_list = lp2src_data_list

        self.lps = lps
        self.dataset_name = dataset_name
        self.protocol = protocol
        self.domains = domains

    @classmethod
    def load(
        cls,
        dataset_name: str,
        lps: List[str],
        protocol: str,
        domains: Union[str, List[str]] = "all",
        include_ref: bool = False,
    ):
        """
        Load the data for the given dataset, language pair, protocol and domains

        Args:
            dataset_name (str): Name of the dataset (e.g. wmt24, wmt23, ...)
            lps (List[str]): Language pairs (e.g., en-es, en-de, ...). ['en-x'] -> all EN-X wmt24 data will be used.
            protocol (str): Protocol used for evaluation (e.g., esa, mqm, ...)
            domains (Union[str, List[str]], optional): List of domains to analyze (e.g., ['news']). Defaults to "all".
        """
        from difficulty_sampling import wmt24_from_en_lps_esa, wmt24_from_en_lps_mqm

        assert (
            protocol is not None
        ), "You need to specify the protocol, such as 'esa', 'da' or 'mqm'."

        if lps == ["en-x"]:
            dataset_name, lps = (
                "wmt24",
                wmt24_from_en_lps_esa if protocol == "esa" else wmt24_from_en_lps_mqm,
            )

        logger.info(
            f"Loading dataset: {dataset_name}\tLanguage pairs: {' '.join(lps)}\tProtocol: {protocol}."
        )

        lp2src_data_list = {
            lp: load_data_wmt(
                year=dataset_name,
                langs=lp,
                normalize=False,
                file_protocol=protocol,
                include_ref=include_ref,
            )
            for lp in lps
            if lp != "en-cs"
        }

        if "en-cs" in lps:
            line_id2annotated_translations = collections.defaultdict(list)
            with WMT24_GENMT_DATA.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    annotated_translation = json.loads(line)
                    if annotated_translation["langs"] == "en-cs":
                        line_id2annotated_translations[
                            annotated_translation["line_id"]
                        ].append(annotated_translation)

            wmt_metrics_eval_set = EvalSet("wmt24", "en-cs", True)
            metric_name2seg_scores = dict()
            for metric_name in wmt_metrics_eval_set.metric_names:
                metric_name2seg_scores[
                    wmt_metrics_eval_set.BaseMetric(metric_name)
                ] = wmt_metrics_eval_set.Scores("seg", metric_name)

            encs_data = []
            for line_id_true, (line_id, annotated_translations) in enumerate(
                sorted(line_id2annotated_translations.items())
            ):
                assert (
                    len(set(a["src"] for a in annotated_translations))
                    == len(set(a["domain"] for a in annotated_translations))
                    == len(set(a["doc_id"] for a in annotated_translations))
                    == 1
                )

                word_count = len(annotated_translations[0]["src"].split())

                new_sample = {
                    "i": line_id_true,
                    "src": annotated_translations[0]["src"],
                    "ref": wmt_metrics_eval_set.all_refs[wmt_metrics_eval_set.std_ref][
                        line_id
                    ],
                    "cost": 0.15 * word_count + 33.7,
                    "domain": annotated_translations[0]["domain"],
                    "doc": annotated_translations[0]["doc_id"],
                }

                sys2annotations = dict()
                for annotation_dict in annotated_translations:
                    if annotation_dict["system"] == "refA":
                        continue

                    sys = (
                        "IOL_Research"
                        if annotation_dict["system"] == "IOL-Research"
                        else annotation_dict["system"]
                    )

                    if sys not in sys2annotations:
                        assert annotation_dict["tgt"] is not None
                        sys2annotations[sys] = (
                            annotation_dict["tgt"],
                            [float(annotation_dict["esa_score"])],
                        )
                    else:
                        assert sys2annotations[sys][0] == annotation_dict["tgt"]
                        sys2annotations[sys][1].append(
                            float(annotation_dict["esa_score"])
                        )

                new_sample["tgt"], new_sample["scores"] = dict(), dict()
                for sys, (tgt, scores) in sys2annotations.items():
                    new_sample["tgt"][sys] = tgt

                    new_sample["scores"][sys] = {"human": sum(scores) / len(scores)}

                    for (
                        metric_name,
                        sys2metric_scores,
                    ) in metric_name2seg_scores.items():
                        assert sys2metric_scores[sys][line_id] is not None
                        new_sample["scores"][sys][metric_name] = sys2metric_scores[sys][
                            line_id
                        ]

                assert len(new_sample["tgt"]) == len(new_sample["scores"]) > 0

                encs_data.append(new_sample)

            encs_data_flat = [line["cost"] for line in encs_data]
            cost_avg = np.average(encs_data_flat)
            cost_std = np.std(encs_data_flat)
            for line in encs_data:
                # z-normalize and make mean 1
                line["cost"] = (line["cost"] - cost_avg) / cost_std + 1

            encs_data_flat = [line["cost"] for line in encs_data]
            cost_min = np.min(encs_data_flat)
            for line in encs_data:
                # make sure it's positive
                line["cost"] = (line["cost"] - cost_min) / (1 - cost_min)

            lp2src_data_list["en-cs"] = encs_data

        logger.info(
            "Num segments before domain filtering: {}".format(
                len(next(iter(lp2src_data_list.values())))
            )
        )

        if domains != "all":
            logger.info(f"Filtering data to the domains: {domains}.")

            lp2src_data_list = {
                lp: [sample for sample in src_data_list if sample["domain"] in domains]
                for lp, src_data_list in lp2src_data_list.items()
            }

            domains = "_".join(sorted(domains))

            logger.info(
                "Num segments after domain filtering: {}.".format(
                    len(next(iter(lp2src_data_list.values())))
                )
            )

        # add z-normalized human scores
        for data in lp2src_data_list.values():
            score_system = collections.defaultdict(list)
            for line in data:
                for sys in line["scores"].keys():
                    score_system[sys].append(line["scores"][sys]["human"])
            # compute mean and variance
            score_system = {
                sys: (np.average(sys_l), np.std(sys_l))
                for sys, sys_l in score_system.items()
            }
            for line in data:
                for sys in line["scores"].keys():
                    line["scores"][sys]["human_z"] = (
                        line["scores"][sys]["human"] - score_system[sys][0]
                    ) / score_system[sys][1]

        return cls(lp2src_data_list, lps, dataset_name, protocol, domains)


def get_src_score(
    src_data: SrcData, scorer_name: str, systems_to_filter: Optional[List[str]] = None
) -> float:
    """
    Return the score assigned by the input scorer to the src data.

    Args:
        src_data (SrcData): SrcData Dictionary containing all the info for a given src segment.
        scorer_name (str): Name of the scorer to use to extract the score from the data.
        systems_to_filter (Optional[List[str]]): Sys to exclude from the analysis (used iff `scorer_name` is `'human'`).

    Returns:
        score (float): Score assigned by the input scorer to the src data.
    """
    scores: Dict[str, Dict[str, float]] = src_data["scores"]  # More explicit typing

    if scorer_name == "human":
        human_scores_sum, n_sys = 0, 0
        for sys in scores:
            if systems_to_filter is None or sys not in systems_to_filter:
                human_scores_sum += scores[sys]["human"]
                n_sys += 1
        return human_scores_sum / n_sys

    return scores[next(iter(scores))][scorer_name]
