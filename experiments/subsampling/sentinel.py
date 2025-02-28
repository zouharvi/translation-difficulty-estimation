import logging
from argparse import Namespace
from pathlib import Path

import sentinel_metric

from difficulty_sampling.data import Data


logger = logging.getLogger(__name__)


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
    use_tgt_lang_token: bool = False,
) -> Data:
    """
    Score the input data with the input sentinel-src metric model, adding "sentinel_src" to their available scores.

    Args:
        sentinel_src_metric_model (sentinel_metric.models.SentinelRegressionMetric): Sentinel-src metric model to use.
        data (Data): Data to score.
        batch_size (int): Batch size to use for the inference with the input sentinel-src metric model. Default: 32.
        scorer_name (str): Name to use to identify the sentinel-src metric model used. Default: 'sentinel-src-mqm'.
        use_tgt_lang_token (bool): Whether to use the target language token in the input data. Default: False.

    Returns:
        scored_data (Data): Input data with "sentinel_src" as additional available score for each MT system.
    """
    if sentinel_src_metric_model.hparams.target_languages and not use_tgt_lang_token:
        raise ValueError(
            "The employed sentinel-src metric model was trained with target language token, but the '--use-tgt-lang' "
            "flag has not been set!"
        )
    if use_tgt_lang_token:
        scores = dict()
        for lp, src_data_list in data.lp2src_data_list.items():
            scores[lp] = sentinel_src_metric_model.predict(
                [{"src": sample["src"], "lp": lp} for sample in src_data_list],
                batch_size=batch_size,
                gpus=1,
            ).scores
        assert set(scores) == set(data.lp2src_data_list)
    else:
        scores = sentinel_src_metric_model.predict(
            [
                {"src": sample["src"]}
                for sample in next(iter(data.lp2src_data_list.values()))
            ],
            batch_size=batch_size,
            gpus=1,
        ).scores

    for lp, src_data_list in data.lp2src_data_list.items():
        assert len(src_data_list) == (
            len(scores) if isinstance(scores, list) else len(scores[lp])
        )
        for idx, sample in enumerate(src_data_list):
            for system in sample["scores"]:
                sample["scores"][system][scorer_name] = (
                    scores[idx] if isinstance(scores, list) else scores[lp][idx]
                )

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
        args.use_tgt_lang,
    )

    return scored_data
