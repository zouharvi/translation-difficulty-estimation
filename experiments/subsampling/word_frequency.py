import logging
from argparse import Namespace
from typing import Literal

from difficulty_sampling.data import Data


logger = logging.getLogger(__name__)


def word_frequency_score(
    data: Data,
    scorer_name: Literal["word_frequency", "word_zipf_frequency"],
) -> Data:
    """
    Assign to each source a score that depends on the word frequency of its words

    Args:
        data (Data): Data to score.
        scorer_name (str): Name to use to identify the scorer used in {'word_frequency', 'word_zipf_frequency'}.

    Returns:
        scored_data (Data): Input data with "scorer_name" as additional available score for each MT system.
        Higher scores indicate higher frequency, therefore lower rarity.
    """
    from wordfreq import word_frequency, zipf_frequency, tokenize

    if scorer_name != "word_frequency" and scorer_name != "word_zipf_frequency":
        raise ValueError(
            f"Scorer name '{scorer_name}' not recognized! Allowed values: 'word_frequency', 'word_zipf_frequency'."
        )

    src_lang = data.lps[0].split("-")[0]
    logger.info(f"Counting frequencies for source language: {src_lang}.")

    scoring_funct, scores = (
        word_frequency if scorer_name == "word_frequency" else zipf_frequency,
        [],
    )
    for sample in next(iter(data.lp2src_data_list.values())):
        tokens = tokenize(sample["src"], src_lang)
        scores.append(
            sum(scoring_funct(token, src_lang) for token in tokens) / len(tokens)
        )

    for lp, src_data_list in data.lp2src_data_list.items():
        assert len(src_data_list) == len(scores)
        for idx, sample in enumerate(src_data_list):
            for system in sample["scores"]:
                sample["scores"][system][scorer_name] = scores[idx]

    return data


def subsample_with_word_frequency(args: Namespace) -> Data:
    """
    Command to subsample WMT data using the frequency of words in the source sentences.

    Args:
        args (Namespace): Arguments parsed from the command line.

    Returns:
        scored_data (Data): Data with word frequency scores added.
    """
    scored_data = word_frequency_score(
        Data.load(
            dataset_name=args.dataset_name,
            lps=[args.lp],
            protocol=args.protocol,
            domains=args.domains,
        ),
        args.scorer_name,
    )

    return scored_data
