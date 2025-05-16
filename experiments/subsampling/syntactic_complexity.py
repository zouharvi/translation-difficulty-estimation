import logging
from argparse import Namespace
from typing import List, Dict

import spacy
from spacy.tokens import Token, Doc
from spacy_udpipe import load as udpipe_load

from difficulty_sampling.data import Data


logger = logging.getLogger(__name__)


def compute_token_depth(token: Token) -> int:
    """
    Compute the dependency depth of a token: the number of hops from the token to the root.

    Args:
        token (Token): A spaCy Token.

    Returns:
        int: The dependency depth of the token.
    """
    depth = 0
    while token.head != token:
        depth += 1
        token = token.head
    return depth


def compute_dependency_tree_height(doc: Doc) -> int:
    """
    Compute the dependency tree height for a spaCy Doc.
    The height is defined as the maximum dependency depth among all tokens in the doc.

    Args:
        doc (Doc): A spaCy Doc.

    Returns:
        int: The maximum token depth in the doc.
    """
    if len(doc) == 0:
        return 0
    return max(compute_token_depth(token) for token in doc)


def get_src_lang2nlp(lps: List[str]) -> Dict:
    """
    Get a mapping of source languages to their corresponding NLP models.

    Args:
        lps (List[str]): List of language pairs.

    Returns:
        Dict: Mapping of source languages to their corresponding NLP models.
    """
    src_lang2nlp = dict()
    for lp in lps:
        src_lang = lp.split("-")[0]
        if src_lang != "en" and src_lang != "ja" and src_lang != "cs":
            raise ValueError(
                f"Language '{src_lang}' not supported! Supported languages: 'en', 'ja', 'cs'."
            )
        if src_lang not in src_lang2nlp:
            # udpipe_load gives tokenization + UD parsing for Czech
            src_lang2nlp[src_lang] = (
                spacy.load(f"{src_lang}_core_web_sm")
                if src_lang != "cs"
                else udpipe_load("cs")
            )
    return src_lang2nlp


def syntactic_complexity_score(
    data: Data,
    scorer_name: str = "syntactic_complexity",
    score_all_source_texts: bool = False,
) -> Data:
    """
    Score the input data using a syntactic structure complexity metric based on dependency tree height.

    For each source sentence, the dependency tree is computed with spaCy, and the negative of its tree height
    (maximum dependency depth) is used as the complexity score. Higher scores indicate less complex syntactic
    structure (and therefore, less difficulty).

    The computed score is added to each sample's scores for every MT system.

    Args:
        data: Data to score.
        scorer_name: Name to assign to the syntactic complexity score. Default: "syntactic_complexity".
        score_all_source_texts: If True, score all source texts regardless of language pair. Default: False.

    Returns:
        Data: The input data with an additional score (`scorer_name`) for each MT system.
    """
    src_lang2nlp = get_src_lang2nlp(data.lps)
    src_lang2scores = (
        {lp.split("-")[0]: [] for lp in data.lps}
        if not score_all_source_texts
        else None
    )

    logger.info("Computing Syntactic Complexity scores for each source text...")
    for lp, src_data_list in data.lp2src_data_list.items():
        src_lang = lp.split("-")[0]
        for src_idx, sample in enumerate(src_data_list):
            if not score_all_source_texts and len(src_lang2scores[src_lang]) > src_idx:
                for scorer_name2score in sample["scores"].values():
                    scorer_name2score[scorer_name] = src_lang2scores[src_lang][src_idx]
            else:
                doc = src_lang2nlp[src_lang](sample["src"])
                height = compute_dependency_tree_height(doc)
                score = -height
                for scorer_name2score in sample["scores"].values():
                    scorer_name2score[scorer_name] = score
                if not score_all_source_texts:
                    src_lang2scores[src_lang].append(score)

    logger.info("Syntactic Complexity scoring complete.")
    return data


def src_len_score(
    data: Data,
    scorer_name: str = "src_len",
    score_all_source_texts: bool = False,
) -> Data:
    """
    Score the input data using the source sentence length (number of tokens).

    Args:
        data: Data to score.
        scorer_name: Name to assign to the source length score. Default: "src_len".
        score_all_source_texts: If True, score all source texts regardless of language pair. Default: False.

    Returns:
        Data: The input data with an additional score (`scorer_name`) for each MT system.
    """
    src_lang2nlp = get_src_lang2nlp(data.lps)
    src_lang2scores = (
        {lp.split("-")[0]: [] for lp in data.lps}
        if not score_all_source_texts
        else None
    )

    logger.info("Computing Source Length scores for each source text...")
    for lp, src_data_list in data.lp2src_data_list.items():
        src_lang = lp.split("-")[0]
        for src_idx, sample in enumerate(src_data_list):
            if not score_all_source_texts and len(src_lang2scores[src_lang]) > src_idx:
                for scorer_name2score in sample["scores"].values():
                    scorer_name2score[scorer_name] = src_lang2scores[src_lang][src_idx]
            else:
                doc = src_lang2nlp[src_lang](sample["src"])
                score = -len(doc)
                for scorer_name2score in sample["scores"].values():
                    scorer_name2score[scorer_name] = score
                if not score_all_source_texts:
                    src_lang2scores[src_lang].append(score)

    logger.info("Source Length scoring complete.")
    return data


def subsample_with_syntactic_complexity(args: Namespace) -> Data:
    """
    Command to subsample WMT data using a syntactic structure complexity score.

    The syntactic complexity is computed as the dependency tree height for each source sentence.
    The computed score is added as an additional score for each MT system.

    Args:
        args (Namespace): Arguments parsed from the command line.

    Returns:
        Data: Data with syntactic complexity scores added.
    """
    scored_data = syntactic_complexity_score(
        Data.load(
            dataset_name=args.dataset_name,
            lps=[args.lp],
            protocol=args.protocol,
            domains=args.domains,
        ),
        args.scorer_name,
        args.syntactic_model_name,
    )

    return scored_data
