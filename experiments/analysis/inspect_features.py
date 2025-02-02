from pathlib import Path
from typing import List, Dict, Union
import logging
import argparse
import pickle
import scipy.stats

import matplotlib.pyplot as plt
from tqdm import tqdm

import numpy as np
from scipy.stats import linregress

import difficulty_sampling

logging.basicConfig(level=logging.INFO)

plt.rcParams["text.usetex"] = True


def src_NER(
    data: List[str],
    filepath: Path,
    batch_size=128,
):
    """
    Counts the number of named entities in the source of each sample

    The named entities are extracted using the GLiNER model, for which you need to run `pip install gliner`
    """

    if not filepath.exists():
        from gliner import GLiNER

        model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1").to("cuda")

        counts = []
        for idx in tqdm(range(0, len(data), batch_size), total=len(data) // batch_size):
            srcs = [sample["src"] for sample in data[idx : idx + batch_size]]

            labels = ["person", "award", "date", "competitions", "teams"]
            batch_result = model.batch_predict_entities(srcs, labels)

            counts.extend(
                [
                    {"segment": data[idx + batch_idx]["i"], "count": len(ner_sample)}
                    for batch_idx, ner_sample in enumerate(batch_result)
                ]
            )

        pickle.dump(counts, open(filepath, "wb"))
    else:
        counts = pickle.load(open(filepath, "rb"))

    return counts


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
            lp (str): Language pair (e.g., en-es, en-de, ...)
            protocol (str): Protocol used for evaluation (e.g., esa, mqm, ...)
            domains (Union[str, List[str]], optional): List of domains to analyze (e.g., ['news']). Defaults to "all".
        """

        logging.info(
            f"Loading dataset: {dataset_name}\tLanguage pair: {lp}\tProtocol: {protocol}"
        )
        data = difficulty_sampling.utils.load_data_wmt(
            year=dataset_name, langs=lp, normalize=False, file_protocol=protocol
        )

        logging.info("Num segments: {}".format(len(data)))

        if domains != "all":
            logging.info(f"Filtering data to the domains: {domains}")

            data = [sample for sample in data if sample["domain"] in domains]
            domains = "_".join(sorted(domains))

        return cls(data, dataset_name, lp, protocol, domains)


class Features:
    """
    Class to measure and report the relationship between features of the data and human scores

    Features are listed in Features.available_features.
    To add a new feature, add it to the list and implement how it gets added to the data in the measure_feature method

    Usage:
        1) Load the data using the Data class
        2) Create an instance of the Features class, passing the data as argument
        3) Measure the feature using the measure_feature method
        4) Plot the feature using the plot_feature method
    """

    available_features = ["src_length", "src_NE_count"]

    def __init__(self, data: Data):
        self.data = data

        self.savedir = Path(
            f"logs/analysis/{data.dataset_name}/{data.lp}/{data.domains}"
        )
        if not self.savedir.exists():
            self.savedir.mkdir(parents=True)

    def measure_feature(self, feature: str):
        if feature not in self.available_features:
            raise (ValueError(f"Feature {feature} not available"))

        if feature == "src_length":
            for sample in self.data.data:
                for system in sample["scores"]:
                    sample["scores"][system]["src_length"] = len(sample["src"])

        elif feature == "src_NE_count":

            ner_savedir = Path("dumps/ner_counts")
            if not ner_savedir.exists():
                ner_savedir.mkdir(parents=True)

            filepath = (
                ner_savedir
                / f"ner_counts.{self.data.dataset_name}.{self.data.lp}.{self.data.domains}.pkl"
            )

            counts = src_NER(self.data.data, filepath=filepath)
            for idx, sample in enumerate(self.data.data):
                assert sample["i"] == counts[idx]["segment"]
                for system in sample["scores"]:
                    sample["scores"][system][feature] = counts[idx]["count"]

    def plot_feature(self, feature: str):
        """
        Plot a feature against the human scores

        Args:
            feature (str): Feature to be plotted (e.g., src_length, src_NE_count),
                which should have been previously computed and added to the score of each system, for each translation
        """

        data = self.data.data
        protocol = self.data.protocol
        datadir = self.savedir / feature

        if not datadir.exists():
            datadir.mkdir(parents=True)

        x = [
            sample["scores"][system][feature]
            for sample in data
            for system in sample["scores"]
        ]
        y = [
            sample["scores"][system]["human"]
            for sample in data
            for system in sample["scores"]
        ]

        fig, ax = plt.subplots(figsize=(6, 4), dpi=72)

        scatter = ax.scatter(
            x, y, alpha=0.6,
            edgecolors="w",
            s=10,
            linewidth=0,
            rasterized=True,
        )

        ax.grid(True, linestyle="--", alpha=0.6)

        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        line = slope * np.array(x) + intercept

        ax.plot(x, line, color="r")

        corr_pearson = scipy.stats.pearsonr(x, y)[0]
        corr_spearman = scipy.stats.spearmanr(x, y)[0]
        plt.text(
            0.95,
            0.05,
            f"Pearson: {corr_pearson:.1%}\nSpearman: {corr_spearman:.1%}".replace("%", r"\%"),
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=dict(facecolor="white", alpha=0.8),
        )

        ax.set_xlabel(feature, fontsize=16)
        ax.set_ylabel(f"{protocol} score", fontsize=16)

        fig.tight_layout(pad=0)

        plt_name = f"{protocol}.png"
        plt.savefig(datadir / plt_name, format="png", dpi=300)
        plt.close()


if __name__ == "__main__":

    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        "--dataset-name",
        type=str,
        default="wmt24",
    )

    argparser.add_argument(
        "--lp",
        type=str,
        default="en-es",
    )

    argparser.add_argument(
        "--protocol",
        type=str,
        default="esa",
    )

    argparser.add_argument(
        "--domains",
        type=str,
        nargs="+",
        default="all",
        help="Domains to be analyzed. If not specified, all domains are considered",
    )

    argparser.add_argument(
        "--feature",
        type=str,
        default="src_length",
        help="Feature to be computed",
    )

    args = argparser.parse_args()

    data = Data.load(
        dataset_name=args.dataset_name,
        lp=args.lp,
        protocol=args.protocol,
        domains=args.domains,
    )

    features = Features(data)
    features.measure_feature(args.feature)
    features.plot_feature(args.feature)
