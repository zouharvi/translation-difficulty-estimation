import logging
from argparse import Namespace
from pathlib import Path

import sentinel_metric

from difficulty_sampling.data import Data

logging.basicConfig(level=logging.INFO)


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
    sources = [
        {"src": sample["src"], "lp": sample["lp"]} for sample in data.src_data_list
    ]
    scores = sentinel_src_metric_model.predict(
        sources, batch_size=batch_size, gpus=1
    ).scores

    assert len(scores) == len(data.src_data_list)

    for idx, sample in enumerate(data.src_data_list):
        for system in sample["scores"]:
            sample["scores"][system][scorer_name] = scores[idx]

    return data


def subsample_with_sentinel_src(args: Namespace) -> Data:
    """
    Command to subsample WMT data using the scores returned by a sentinel-src metric.

    Args:
        args (Namespace): Arguments parsed from the command line.

    Returns:
        scored_data (Data): Data with sentinel-src scores added.
    """
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

    return scored_data
