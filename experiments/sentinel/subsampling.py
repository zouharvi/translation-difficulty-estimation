import logging
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import sentinel_metric
from typing import Dict, Union, List

from matplotlib import pyplot as plt

import difficulty_sampling

logging.basicConfig(level=logging.INFO)

wmt24_from_en_lps, wmt24_from_en_lps_mqm = [
    "en-cs",
    "en-de",
    "en-es",
    "en-hi",
    "en-is",
    "en-ja",
    "en-ru",
    "en-uk",
    "en-zh",
], ["en-de", "en-es"]


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
                data += difficulty_sampling.utils.load_data_wmt(
                    year="wmt24", langs=lp, normalize=False, file_protocol=protocol
                )
        else:
            data = difficulty_sampling.utils.load_data_wmt(
                year=dataset_name, langs=lp, normalize=False, file_protocol=protocol
            )

        logging.info("Num segments before domain filtering: {}".format(len(data)))

        if domains != "all":
            logging.info(f"Filtering data to the domains: {domains}.")

            data = [sample for sample in data if sample["domain"] in domains]
            domains = "_".join(sorted(domains))

            logging.info("Num segments after domain filtering: {}.".format(len(data)))

        return cls(data, dataset_name, lp, protocol, domains)


def read_arguments() -> ArgumentParser:
    parser = ArgumentParser(
        description="Command to subsample WMT data using the scores returned by a sentinel-src metric model."
    )

    parser.add_argument(
        "--dataset-name",
        type=str,
        default="wmt24",
        help="Name of the WMT dataset to subsample. Default: 'wmt24'.",
    )

    parser.add_argument(
        "--lp",
        type=str,
        default="en-es",
        help="Language pair to consider in the WMT dataset passed in input. If 'all_en' is passed, all the wmt24 "
        "language pairs with English on the source side will be used, and the '--dataset-name' argument will be "
        "ignored. Default: 'en-es'.",
    )

    parser.add_argument(
        "--protocol",
        type=str,
        choices=["esa", "mqm"],
        default="esa",
        help="Which annotation protocol to consider when loading human scores. Allowed values: 'esa', 'mqm'. "
        "Default: 'esa'.",
    )

    parser.add_argument(
        "--domains",
        type=str,
        nargs="+",
        default="all",
        help="Domains to be analyzed. If not specified, all domains are considered ('all'). Default: 'all'.",
    )

    parser.add_argument(
        "--sentinel-src-metric-model",
        type=str,
        default="sapienzanlp/sentinel-src-mqm",
        help="String that identifies a local file system path to a sentinel-src metric model checkpoint, or a string "
        "that identifies it on the Hugging Face Hub.",
    )

    parser.add_argument(
        "--batch-size",
        default=32,
        type=int,
        help="Batch size to use when running inference with the input sentinel-src metric model. Default: 32.",
    )

    parser.add_argument(
        "--scorer-name",
        type=str,
        default="sentinel-src-mqm",
        help="Which name to use to identify the sentinel-src metric model used for the subsampling (it will be used in "
        "the output plot and for the output path where to save it). Default: 'sentinel-src-mqm'.",
    )

    parser.add_argument(
        "--out-plot-path",
        type=Path,
        required=True,
        help="Local file system path where to save the output hist.",
    )

    return parser


def get_sentinel_src_metric_model(
    sentinel_src_metric_model: str = "sapienzanlp/sentinel-src-mqm",
) -> sentinel_metric.models.SentinelRegressionMetric:
    """
    Return a sentinel-src metric model identified by the input parameter.

    Args:
        sentinel_src_metric_model (str): String that identifies a path or a sentinel-src model on the Hugging Face Hub.

    Returns:
        sentinel_src_metric_model (sentinel_metric.models.SentinelRegressionMetric): A sentinel-src metric model.
    """
    sentinel_src_metric_model_checkpoint_path = Path(sentinel_src_metric_model)
    if sentinel_src_metric_model_checkpoint_path.exists():
        return sentinel_metric.load_from_checkpoint(
            sentinel_src_metric_model_checkpoint_path,
            strict=True,
            class_identifier="sentinel_regression_metric",
        )
    else:
        sentinel_src_metric_model_checkpoint_path = sentinel_metric.download_model(
            sentinel_src_metric_model
        )
        return sentinel_metric.load_from_checkpoint(
            sentinel_src_metric_model_checkpoint_path,
            strict=True,
            class_identifier="sentinel_regression_metric",
        )


def sentinel_src_metric_model_score(
    sentinel_src_metric_model: sentinel_metric.models.SentinelRegressionMetric,
    data: Data,
    batch_size: int = 32,
    scorer_name: str = "sentinel-src-mqm",
) -> Data:
    """
    Score the input data with the input sentinel-src metric model, adding "sentinel_src" to their available scores.

    Args:
        sentinel_src_metric_model (sentinel_metric.models.SentinelRegressionMetric): Sentinel-src metric model to use.
        data (Data): Data to score.
        batch_size (int): Batch size to use for the inference with the input sentinel-src metric model. Default: 32.
        scorer_name (str): Name to use to identify the sentinel-src metric model used. Default: 'sentinel-src-mqm'.

    Returns:
        scored_data (Data): Input data with "sentinel_src" as additional available score for each MT system.
    """
    sources = [{"src": sample["src"]} for sample in data.data]
    scores = sentinel_src_metric_model.predict(
        sources, batch_size=batch_size, gpus=1
    ).scores

    assert len(scores) == len(data.data)

    for idx, sample in enumerate(data.data):
        for system in sample["scores"]:
            sample["scores"][system][scorer_name] = scores[idx]

    return data


def get_src_score(
    src_data: Dict[
        str, Union[int, str, Dict[str, str], float, Dict[str, Dict[str, float]]]
    ],
    scorer_name: str = "sentinel-src-mqm",
) -> float:
    """
    Return the score assigned by the input scorer to the src data.

    Args:
        src_data (Dict): Dictionary containing all the data for a given src segment.
        scorer_name (str): Name of the scorer to use to extract the score from the data. Default: 'sentinel-src-mqm'.

    Returns:
        score (float): Score assigned by the input scorer to the src data.
    """
    return src_data["scores"][next(iter(src_data["scores"]))][scorer_name]


def plot_human_scores_hist(
    scored_data: Data, scorer_name: str, bins: np.ndarray, out_plot_path: Path
) -> None:
    """
    Plot the hist of the human scores contained in the scored input data.

    Args:
        scored_data (Data): Data containing scores to use for subsampling.
        scorer_name (str): Which name to use to identify the scorer used for subsampling.
        bins (np.ndarray): Bins to use for the hist plot.
        out_plot_path (Path): Path where to save the output hist plot.
    """
    data_y1 = np.array(
        [
            np.average([v["human"] for v in l["scores"].values()])
            for l in scored_data.data
        ]
    )
    len_flat = len(data_y1)
    data_y2 = np.array(
        [
            np.average([v["human"] for v in l["scores"].values()])
            for l in scored_data.data
        ][: int(len_flat * 0.50)]
    )
    data_y3 = np.array(
        [
            np.average([v["human"] for v in l["scores"].values()])
            for l in scored_data.data
        ][: int(len_flat * 0.25)]
    )

    plt.hist(
        [data_y1, data_y2, data_y3],
        density=True,
        bins=bins,
        label=["All", "Selected 50%", "Selected 25%"],
    )

    plt.legend()
    plt.title(f"Selection with {scorer_name}. Total number of src: {len_flat}.")

    # Save the figure
    plt.savefig(out_plot_path, dpi=300, bbox_inches="tight")
    plt.close()  # Close the figure to free memory and avoid displaying


def subsample_with_sentinel_src_command() -> None:
    """
    Command to subsample WMT data using the scores returned by a sentinel-src metric.
    """
    args = read_arguments().parse_args()

    scored_data = sentinel_src_metric_model_score(
        get_sentinel_src_metric_model(args.sentinel_src_metric_model),
        Data.load(
            dataset_name=args.dataset_name,
            lp=args.lp,
            protocol=args.protocol,
            domains=args.domains,
        ),
        args.batch_size,
        args.scorer_name,
    )
    # Sort the src data in ascending order using the input sentinel-src metric model scores.
    scored_data.data.sort(
        key=lambda src_data: get_src_score(src_data, args.scorer_name)
    )

    plot_human_scores_hist(
        scored_data,
        args.scorer_name,
        np.arange(0, 100 + 10, 10)
        if args.protocol == "esa"
        else np.arange(-25, 0 + 1, 1),
        args.out_plot_path,
    )


if __name__ == "__main__":
    subsample_with_sentinel_src_command()
