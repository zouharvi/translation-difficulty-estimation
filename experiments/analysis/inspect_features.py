from typing import List, Union
import logging
import argparse
import pickle
import scipy.stats

import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from scipy.stats import linregress
import precomet
import sentinel_metric

from difficulty_estimation.data import Data


logging.basicConfig(level=logging.INFO)


try:
    plt.rcParams["text.usetex"] = True
except Exception:
    logging.warning("LaTeX unavailable, using standard text rendering")


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
        "human",
        "src_length",
        "src_NE_count",
        "PreCOMET-avg",
        "PreCOMET-diversity",
        "PreCOMET-var",
        "PreCOMET-cons",
        "sentinel-src-mqm",
        "sentinel-src-mqm-tgt-lang",
    ]

    def __init__(self, data: Data):
        self.data = data

        self.savedir = Path(
            f"logs/analysis/{data.dataset_name}/{'.'.join(data.lps)}/{data.domains}"
        )
        if not self.savedir.exists():
            self.savedir.mkdir(parents=True)

    def measure_feature(self, feature: str):
        if feature not in self.available_features:
            raise (ValueError(f"Feature {feature} not available"))

        if feature == "human":
            return

        if feature == "src_length":
            for lp, src_data_list in self.data.lp2src_data_list.items():
                for sample in src_data_list:
                    for system in sample["scores"]:
                        sample["scores"][system]["src_length"] = len(sample["src"])

        elif feature == "src_NE_count":
            ner_savedir = Path("dumps/ner_counts")
            if not ner_savedir.exists():
                ner_savedir.mkdir(parents=True)

            filepath = (
                ner_savedir
                / f"ner_counts.{self.data.dataset_name}.{'.'.join(self.data.lps)}.{self.data.domains}.pkl"
            )

            for lp, src_data_list in self.data.lp2src_data_list.items():
                counts = src_NER(src_data_list, filepath=filepath)
                for idx, sample in enumerate(src_data_list):
                    assert sample["i"] == counts[idx]["segment"]
                    for system in sample["scores"]:
                        sample["scores"][system][feature] = counts[idx]["count"]

        elif feature in {
            "PreCOMET-avg",
            "PreCOMET-diversity",
            "PreCOMET-var",
            "PreCOMET-cons",
        }:

            model = precomet.load_from_checkpoint(
                precomet.download_model(f"zouharvi/{feature}")
            )

            for lp, src_data_list in self.data.lp2src_data_list.items():
                sources = [{"src": sample["src"]} for sample in src_data_list]

                scores = model.predict(sources)["scores"]
                assert len(scores) == len(src_data_list)
                for idx, sample in enumerate(src_data_list):
                    assert sample["i"] == idx
                    for system in sample["scores"]:
                        sample["scores"][system][feature] = scores[idx]

        elif feature == "sentinel-src-mqm":

            model_path = sentinel_metric.download_model("sapienzanlp/sentinel-src-mqm")
            model = sentinel_metric.load_from_checkpoint(model_path)

            for lp, src_data_list in self.data.lp2src_data_list.items():
                sources = [{"src": sample["src"]} for sample in src_data_list]
                scores = model.predict(sources, batch_size=32, gpus=1).scores

                assert len(scores) == len(src_data_list)

                for idx, sample in enumerate(src_data_list):
                    assert sample["i"] == idx
                    for system in sample["scores"]:
                        sample["scores"][system][feature] = scores[idx]

        elif feature == "sentinel-src-mqm-tgt-lang":

            model_path = Path("models/sentinel-src-mqm-tgt-lang.ckpt")
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")

            model = sentinel_metric.load_from_checkpoint(model_path)

            for lp, src_data_list in self.data.lp2src_data_list.items():
                sources = [{"src": sample["src"]} for sample in src_data_list]
                scores = model.predict(
                    [{"src": sample["src"], "lp": lp} for sample in src_data_list],
                    batch_size=32,
                    gpus=1,
                ).scores

                assert len(scores) == len(src_data_list)

                for idx, sample in enumerate(src_data_list):
                    assert sample["i"] == idx
                    for system in sample["scores"]:
                        sample["scores"][system][feature] = scores[idx]

    def filter_outliers(self, x, y, sys_names_or_tgt_langs):
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
        sys_names_or_tgt_langs = [
            sys_names_or_tgt_langs[idx] for idx in range(len(z)) if z[idx] < 3
        ]

        assert len(x) == len(y) == len(sys_names_or_tgt_langs)

        return x, y, sys_names_or_tgt_langs

    def plot_features_with_colored_systems(
        self,
        feature: str,
        filter_feature_outliers=False,
        feature2: str = "human",
        aggregate_feature1: str = None,
        aggregate_feature2: str = None,
    ):
        """
        Plot a feature against the human scores or another feature. The color of each scatter element depends
        on the MT system that produced the corresponding translation.

        Args:
            feature (str): Feature to be plotted on the x-axis (e.g., src_length, src_NE_count),
                which should have been previously computed and a dded to the score of each system, for each translation
            filter_feature_outliers (bool, optional): Whether to filter outliers in the feature values. Defaults to False.
            feature2 (str, optional): Second feature, to be plotted on the y-axis. Defaults to "human".
            aggregate_feature1 (Union[str, bool], optional): Whether to aggregate the feature values for each translation.
                - 'mean': Compute the mean of the feature values for each translation
                - 'std' : Compute the standard deviation of the feature values for each translation
                - None : Do not aggregate the feature values
            aggregate_feature2 (Union[str, bool], optional): Whether to aggregate the feature2 values for each translation.
                - 'mean': Compute the mean of the feature2 values for each translation
                - 'std' : Compute the standard deviation of the feature2 values for each translation
                - None : Do not aggregate the feature2 values
        """

        protocol = self.data.protocol
        if feature == "human":
            datadir = self.savedir / protocol
        else:
            datadir = self.savedir / feature
        if not datadir.exists():
            datadir.mkdir(parents=True)

        if aggregate_feature1 not in {None, "mean", "std"}:
            raise ValueError(
                f"Invalid value for aggregate_feature1: {aggregate_feature1}"
            )

        if aggregate_feature2 not in {None, "mean", "std"}:
            raise ValueError(
                f"Invalid value for aggregate_feature2: {aggregate_feature2}"
            )

        # Collect x, y, and the system labels in parallel
        x = []
        y = []
        sys_names = []
        for lp, src_data_list in self.data.lp2src_data_list.items():
            data = src_data_list

            lp_x, lp_y, lp_sys_names = [], [], []

            for sample in data:
                lp_sample_x, lp_sample_y = [], []

                for system in sample["scores"]:
                    lp_sample_x.append(sample["scores"][system][feature])
                    lp_sample_y.append(sample["scores"][system][feature2])
                    lp_sys_names.append(system)

                if aggregate_feature1 is not None:
                    if aggregate_feature1 == "mean":
                        agg_x = np.mean(lp_sample_x)
                    elif aggregate_feature1 == "std":
                        agg_x = np.std(lp_sample_x)
                    lp_x.extend([agg_x] * len(lp_sample_x))
                else:
                    lp_x.extend(lp_sample_x)

                if aggregate_feature2 is not None:
                    if aggregate_feature2 == "mean":
                        agg_y = np.mean(lp_sample_y)
                    elif aggregate_feature2 == "std":
                        agg_y = np.std(lp_sample_y)
                    lp_y.extend([agg_y] * len(lp_sample_y))
                else:
                    lp_y.extend(lp_sample_y)

            if filter_feature_outliers:
                lp_x, lp_y, lp_sys_names = self.filter_outliers(
                    lp_x, lp_y, lp_sys_names
                )

            x.extend(lp_x)
            y.extend(lp_y)
            sys_names.extend(lp_sys_names)

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
            ax.scatter([], [], color=cmap(color_id), label=sys_name)

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

        plt_name, x_label, y_label = "", "", ""

        if aggregate_feature1 is not None:
            plt_name += f"{aggregate_feature1}."

        if feature == "human":
            x_label += f"{protocol}"
        else:
            x_label += f"{feature}"

        if feature2 == "human":
            y_label += f"{protocol}"
            plt_name += f"{protocol}"
        else:
            y_label += f"{feature2}"
            plt_name += f"{feature2}"

        if aggregate_feature1 is not None:
            x_label += f" ({aggregate_feature1})"

        if aggregate_feature2 is not None:
            plt_name += f".{aggregate_feature2}"
            y_label += f" ({aggregate_feature2})"
        plt_name += ".png"

        ax.set_xlabel(f"{x_label}", fontsize=16)
        ax.set_ylabel(f"{y_label}", fontsize=16)

        ax.grid(True, linestyle="--", alpha=0.6)
        fig.tight_layout(pad=0)

        plt.savefig(datadir / plt_name, format="png", dpi=300)
        plt.close()

    def plot_features_with_colored_langs(
        self,
        feature: str,
        filter_feature_outliers=False,
        feature2: str = "human",
    ):
        """
        Plot a feature against the human scores or another feature. The color of each scatter element depends
        on the target language of the corresponding translation.

        Args:
            feature (str): Feature to be plotted on the x-axis (e.g., src_length, src_NE_count),
                which should have been previously computed and a dded to the score of each system, for each translation
            filter_feature_outliers (bool, optional): Whether to filter outliers in the feature values. Defaults to False.
            feature2 (str, optional): Second feature, to be plotted on the y-axis. Defaults to "human".
        """

        datadir = self.savedir / feature
        if not datadir.exists():
            datadir.mkdir(parents=True)

        # Collect x, y, and the system labels in parallel
        x = []
        y = []
        tgt_langs = []
        for lp, src_data_list in self.data.lp2src_data_list.items():
            data = src_data_list
            protocol = self.data.protocol

            lp_x, lp_y, lp_tgt_langs = [], [], []

            for sample in data:
                for system in sample["scores"]:
                    lp_x.append(sample["scores"][system][feature])
                    lp_y.append(sample["scores"][system][feature2])
                    lp_tgt_langs.append(lp.split("-")[1])

            if filter_feature_outliers:
                lp_x, lp_y, lp_tgt_langs = self.filter_outliers(
                    lp_x, lp_y, lp_tgt_langs
                )

            x.extend(lp_x)
            y.extend(lp_y)
            tgt_langs.extend(lp_tgt_langs)

        fig, ax = plt.subplots(figsize=(6, 4), dpi=72)

        # Color each system with a different color
        unique_tgt_langs = sorted(set(tgt_langs))
        lang2id = {lang: idx for idx, lang in enumerate(unique_tgt_langs)}
        cmap = plt.get_cmap("tab20")
        point_colors = [cmap(lang2id[lang]) for lang in tgt_langs]

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
        for lang, color_id in lang2id.items():
            ax.scatter([], [], color=cmap(color_id), label=lang)

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
        if feature2 == "human":
            ax.set_ylabel(f"{protocol} score", fontsize=16)
            plt_name = f"colored.lang.{protocol}.png"
        else:
            ax.set_ylabel(feature2, fontsize=16)
            plt_name = f"colored.lang.{feature2}.png"

        ax.grid(True, linestyle="--", alpha=0.6)
        fig.tight_layout(pad=0)

        plt.savefig(datadir / plt_name, format="png", dpi=300)
        plt.close()

    def plot_features_distribution_per_lang(
        self, feature: str, filter_feature_outliers=False
    ):
        """
        Plot the probability density function of a feature's values for each target language.

        Args:
            feature (str): Feature to be plotted (e.g., src_length, sentinel-src-mqm)
            filter_feature_outliers (bool): Whether to filter outliers in feature values. Defaults to False.
        """

        datadir = self.savedir / feature
        if not datadir.exists():
            datadir.mkdir(parents=True)

        lang_to_values = {}

        for lp, src_data_list in self.data.lp2src_data_list.items():

            if lp not in lang_to_values:
                lang_to_values[lp] = []

            values = []
            for sample in src_data_list:
                for system in sample["scores"]:
                    values.append(sample["scores"][system][feature])

            if filter_feature_outliers and len(values) > 0:
                z = np.abs(scipy.stats.zscore(np.array(values)))
                values = [values[i] for i in range(len(values)) if z[i] < 3]

            lang_to_values[lp].extend(values)

        fig, ax = plt.subplots(figsize=(6, 4), dpi=72)

        cmap = plt.get_cmap("tab20")

        # Plot density for each language
        for i, (lang, values) in enumerate(sorted(lang_to_values.items())):
            # Use kernel density estimation to get smooth PDF
            if len(values) > 5:  # Need enough data points for density estimation
                density = scipy.stats.gaussian_kde(values)
                x = np.linspace(min(values), max(values), 1000)
                ax.plot(x, density(x), color=cmap(i), label=lang, rasterized=True)

        if feature == "human":
            ax.set_ylabel(f"{self.data.protocol}", fontsize=16)
            plt_name = f"{self.data.protocol}.distribution_per_lang.png"
        else:
            ax.set_xlabel(feature, fontsize=16)
            plt_name = f"distribution_per_lang.png"

        ax.set_ylabel("Density", fontsize=16)
        ax.legend(
            loc="best",
            fontsize="x-small",
            markerscale=0.5,
            labelspacing=0.2,
            borderpad=0.3,
        )
        ax.grid(True, linestyle="--", alpha=0.6)

        fig.tight_layout(pad=0)
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
        "--lps",
        type=str,
        nargs="+",
        default="en-x",
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
        "--feature2",
        type=str,
        default="human",
        help="Second feature to be computed",
    )

    argparser.add_argument(
        "--filter-feature-outliers",
        action="store_true",
        help="Filter outliers in the feature values",
    )

    argparser.add_argument(
        "--aggregate-feature1",
        type=str,
        default=None,
        help="Aggregate the feature values for each translation",
    )

    argparser.add_argument(
        "--aggregate-feature2",
        type=str,
        default=None,
        help="Aggregate the feature2 values for each translation",
    )

    args = argparser.parse_args()

    data = Data.load(
        dataset_name=args.dataset_name,
        lps=args.lps,
        protocol=args.protocol,
        domains=args.domains,
    )

    features = Features(data)
    features.measure_feature(args.feature)
    features.measure_feature(args.feature2)

    features.plot_features_with_colored_systems(
        args.feature,
        args.filter_feature_outliers,
        args.feature2,
        args.aggregate_feature1,
        args.aggregate_feature2,
    )
    features.plot_features_with_colored_langs(
        args.feature, args.filter_feature_outliers, args.feature2
    )
    features.plot_features_distribution_per_lang(
        args.feature, args.filter_feature_outliers
    )
    features.plot_features_distribution_per_lang(
        args.feature2, args.filter_feature_outliers
    )
