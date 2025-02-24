from pathlib import Path
from typing import List
import logging
import argparse
import pickle
import scipy.stats
import subset2evaluate.select_subset

import matplotlib.pyplot as plt
from tqdm import tqdm

import numpy as np
from scipy.stats import linregress

from difficulty_sampling.data import Data


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

    available_features = [
        "src_length",
        "src_NE_count",
        "cometsrc_avg",
        "cometsrc_diversity",
        "sentinel_src",
    ]

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
            for sample in self.data.src_data_list:
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

            counts = src_NER(self.data.src_data_list, filepath=filepath)
            for idx, sample in enumerate(self.data.src_data_list):
                assert sample["i"] == counts[idx]["segment"]
                for system in sample["scores"]:
                    sample["scores"][system][feature] = counts[idx]["count"]

        elif feature == "cometsrc_avg":
            data_new = subset2evaluate.select_subset.basic(
                method="cometsrc_avg", data=self.data.src_data_list
            )
            data_new = {
                sample["i"]: sample["subset2evaluate_utility"] for sample in data_new
            }
            for sample in self.data.src_data_list:
                for system in sample["scores"]:
                    sample["scores"][system]["cometsrc_avg"] = data_new[sample["i"]]

        elif feature == "cometsrc_diversity":
            data_new = subset2evaluate.select_subset.basic(
                method="cometsrc_diversity", data=self.data.src_data_list
            )
            data_new = {
                sample["i"]: sample["subset2evaluate_utility"] for sample in data_new
            }
            for sample in self.data.src_data_list:
                for system in sample["scores"]:
                    sample["scores"][system]["cometsrc_diversity"] = data_new[
                        sample["i"]
                    ]

        elif feature == "sentinel_src":
            # to use, you need to install the sentinel_metric package following the readme of
            # https://github.com/SapienzaNLP/guardians-mt-eval
            from sentinel_metric import download_model, load_from_checkpoint

            model_path = download_model("sapienzanlp/sentinel-src-mqm")
            model = load_from_checkpoint(model_path)

            sources = [{"src": sample["src"]} for sample in self.data.src_data_list]
            scores = model.predict(sources, batch_size=32, gpus=1).scores

            assert len(scores) == len(self.data.src_data_list)

            for idx, sample in enumerate(self.data.src_data_list):
                for system in sample["scores"]:
                    sample["scores"][system]["sentinel_src"] = scores[idx]

    def filter_outliers(self, x, y):
        """
        Filter outliers (in terms of the feature x) from the data

        Args:
            x (List): Feature values
            y (List): Human scores

        Returns:
            List: Filtered feature values
            List: Filtered human scores
        """

        x = np.array(x)
        y = np.array(y)

        # Filter outliers
        z = np.abs(scipy.stats.zscore(x))
        x = x[z < 3]
        y = y[z < 3]
        sys_names = [sys_names[idx] for idx in range(len(z)) if z[idx] < 3]

        assert len(x) == len(y) == len(sys_names)

        return x, y, sys_names

    def plot_feature(self, feature: str, filter_feature_outliers=False):
        """
        Plot a feature against the human scores

        Args:
            feature (str): Feature to be plotted (e.g., src_length, src_NE_count),
                which should have been previously computed and added to the score of each system, for each translation
        """

        data = self.data.src_data_list
        protocol = self.data.protocol
        datadir = self.savedir / feature

        if not datadir.exists():
            datadir.mkdir(parents=True)

        # Collect x, y, and the system labels in parallel
        x = []
        y = []
        sys_names = []
        for sample in data:
            for system in sample["scores"]:
                x.append(sample["scores"][system][feature])
                y.append(sample["scores"][system]["human"])
                sys_names.append(system)

        if filter_feature_outliers:
            x, y, sys_names = self.filter_outliers(x, y, sys_names)

        fig, ax = plt.subplots(figsize=(6, 4), dpi=72)

        # Color each system with a different color
        unique_sys_names = sorted(set(sys_names))
        system_to_id = {sys: idx for idx, sys in enumerate(unique_sys_names)}
        cmap = plt.get_cmap("tab20")
        point_colors = [cmap(system_to_id[sys_name]) for sys_name in sys_names]

        scatter = ax.scatter(
            x,
            y,
            c=point_colors,
            alpha=0.9,
            edgecolors="w",
            s=10,
            linewidth=0,
            rasterized=True,
        )

        # scatter an empty set of points to automatically create handles for the legend
        for sys_name, color_id in system_to_id.items():
            ax.scatter([], [], color=point_colors[color_id], label=sys_name)

        ax.legend(
            loc="best",
            fontsize="x-small",
            markerscale=0.5,
            labelspacing=0.2,
            borderpad=0.3,
        )

        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        line = slope * np.array(x) + intercept

        ax.plot(x, line, color="r")

        corr_pearson = scipy.stats.pearsonr(x, y)[0]
        corr_spearman = scipy.stats.spearmanr(x, y)[0]

        plt.text(
            0.95,
            0.05,
            f"Pearson: {round(corr_pearson,2):.2f}\nSpearman: {round(corr_spearman, 2):.2f}",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=dict(facecolor="white", alpha=0.8),
        )

        ax.set_xlabel(feature, fontsize=16)
        ax.set_ylabel(f"{protocol} score", fontsize=16)
        ax.grid(True, linestyle="--", alpha=0.6)
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

    argparser.add_argument(
        "--filter-feature-outliers",
        action="store_true",
        help="Filter outliers in the feature values",
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
    features.plot_feature(args.feature, args.filter_feature_outliers)
