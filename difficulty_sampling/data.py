from typing import Dict, List, Union
import logging

from difficulty_sampling import wmt24_from_en_lps, wmt24_from_en_lps_mqm
from difficulty_sampling.utils import load_data_wmt


class Data:
    def __init__(
        self,
        data: List[Dict],
        dataset_name: str,
        lp: str,
        protocol: str,
        domains: str,
    ):
        self.data = data

        self.dataset_name = dataset_name
        self.lp = lp
        self.protocol = protocol
        self.domains = domains

    @classmethod
    def load(
        cls,
        dataset_name: str,
        lp: str,
        protocol: str,
        domains: Union[str, List[str]] = "all",
    ):
        """
        Load the data for the given dataset, language pair, protocol and domains

        Args:
            dataset_name (str): Name of the dataset (e.g. wmt24, wmt23, ...)
            lp (str): Language pair (e.g., en-es, en-de, ...). 'all_en' -> all EN-X wmt24 data will be used.
            protocol (str): Protocol used for evaluation (e.g., esa, mqm, ...)
            domains (Union[str, List[str]], optional): List of domains to analyze (e.g., ['news']). Defaults to "all".
        """
        logging.info(
            f"Loading dataset: {dataset_name}\tLanguage pair: {lp}\tProtocol: {protocol}."
        )
        if lp == "all_en":
            data = []
            lps = wmt24_from_en_lps if protocol == "esa" else wmt24_from_en_lps_mqm
            for lp in lps:
                data += load_data_wmt(
                    year="wmt24", langs=lp, normalize=False, file_protocol=protocol
                )
        else:
            data = load_data_wmt(
                year=dataset_name, langs=lp, normalize=False, file_protocol=protocol
            )

        logging.info("Num segments before domain filtering: {}".format(len(data)))

        if domains != "all":
            logging.info(f"Filtering data to the domains: {domains}.")

            data = [sample for sample in data if sample["domain"] in domains]
            domains = "_".join(sorted(domains))

            logging.info("Num segments after domain filtering: {}.".format(len(data)))

        return cls(data, dataset_name, lp, protocol, domains)
