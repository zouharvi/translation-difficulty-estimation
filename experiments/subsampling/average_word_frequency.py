import logging
from argparse import Namespace
from typing import Literal

from difficulty_estimation.data import Data


logger = logging.getLogger(__name__)


def avg_word_freq_score(
    data: Data,
    scorer_name: Literal["avg_word_freq", "avg_word_zipf_freq"],
    score_all_source_texts: bool = False,
) -> Data:
    """
    Assign to each source text a score that depends on the word frequency of its tokens (average).

    Args:
        data: Data to score.
        scorer_name: Name to use for scorer. Allowed values: 'avg_word_freq', 'avg_word_zipf_freq'.
        score_all_source_texts: If True, score all source texts regardless of language pair. Default: False.

    Returns:
        scored_data: Input data with `scorer_name` as an additional available score for each MT system.
    """
    from wordfreq import word_frequency, zipf_frequency, tokenize

    if scorer_name != "avg_word_freq" and scorer_name != "avg_word_zipf_freq":
        raise ValueError(
            f"Scorer name '{scorer_name}' not recognized! Allowed values: 'avg_word_freq', 'avg_word_zipf_freq'."
        )

    src_lang2scores = (
        {lp.split("-")[0]: [] for lp in data.lps}
        if not score_all_source_texts
        else None
    )
    scoring_funct = word_frequency if scorer_name == "avg_word_freq" else zipf_frequency

    for lp, src_data_list in data.lp2src_data_list.items():
        src_lang = lp.split("-")[0]
        for src_idx, sample in enumerate(src_data_list):
            if not score_all_source_texts and len(src_lang2scores[src_lang]) > src_idx:
                for scorer_name2score in sample["scores"].values():
                    scorer_name2score[scorer_name] = src_lang2scores[src_lang][src_idx]
            else:
                tokens = tokenize(sample["src"], src_lang)
                score = (
                    sum(scoring_funct(tok, src_lang) for tok in tokens) / len(tokens)
                    if len(tokens) > 0
                    else 0.0
                )
                for scorer_name2score in sample["scores"].values():
                    scorer_name2score[scorer_name] = score
                if not score_all_source_texts:
                    src_lang2scores[src_lang].append(score)

    return data


def subsample_with_negative_word_frequency(args: Namespace) -> Data:
    """
    Command to subsample WMT data using the frequency of words in the source sentences.

    Args:
        args (Namespace): Arguments parsed from the command line.

    Returns:
        scored_data (Data): Data with word frequency scores added.
    """
    scored_data = avg_word_freq_score(
        Data.load(
            dataset_name=args.dataset_name,
            lps=[args.lp],
            protocol=args.protocol,
            domains=args.domains,
        ),
        args.scorer_name,
    )

    return scored_data
