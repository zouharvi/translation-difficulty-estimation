import collections
import logging
import numpy as np
from typing import Dict, List, Union, TypedDict

from difficulty_sampling import wmt24_from_en_lps_esa, wmt24_from_en_lps_mqm
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
        assert protocol is not None, "You need to specify the protocol, such as 'esa', 'da' or 'mqm'."

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
        }

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
                    line["scores"][sys]["human_z"] = (line["scores"][sys]["human"] - score_system[sys][0]) / score_system[sys][1]

        return cls(lp2src_data_list, lps, dataset_name, protocol, domains)
