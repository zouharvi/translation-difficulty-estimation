from typing import Dict, List, Union, TypedDict, Optional
import logging

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
    ):
        """
        Load the data for the given dataset, language pair, protocol and domains

        Args:
            dataset_name (str): Name of the dataset (e.g. wmt24, wmt23, ...)
            lp (str): Language pair (e.g., en-es, en-de, ...). 'en-x' -> all EN-X wmt24 data will be used.
            protocol (str): Protocol used for evaluation (e.g., esa, mqm, ...)
            domains (Union[str, List[str]], optional): List of domains to analyze (e.g., ['news']). Defaults to "all".
        """

        if len(lps) == 1 and lps[0] == "en-x":
            lps = wmt24_from_en_lps_esa if protocol == "esa" else wmt24_from_en_lps_mqm

        dataset_name = "wmt24" if dataset_name == "en-x" else dataset_name

        logger.info(
            f"Loading dataset: {dataset_name}\tLanguage pairs: {' '.join(lps)}\tProtocol: {protocol}."
        )

        lp2src_data_list = {
            lp: load_data_wmt(
                year="wmt24" if lp == "en-x" else dataset_name,
                langs=lp,
                normalize=False,
                file_protocol=protocol,
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

        return cls(lp2src_data_list, lps, dataset_name, protocol, domains)
