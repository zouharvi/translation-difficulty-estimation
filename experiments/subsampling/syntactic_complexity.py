import logging
from argparse import Namespace

import spacy
from spacy.tokens import Token, Doc

from difficulty_sampling.data import Data

logging.basicConfig(level=logging.INFO)


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


def syntactic_complexity_score(
    data: Data,
    scorer_name: str = "syntactic_complexity",
    model_name: str = "en_core_web_sm",
) -> Data:
    """
    Score the input data using a syntactic structure complexity metric based on dependency tree height.

    For each source sentence, the dependency tree is computed with spaCy, and its tree height (maximum dependency depth)
    is used as the complexity score. Higher scores indicate more complex syntactic
    structure (and therefore, higher difficulty).

    The computed score is added to each sample's scores for every MT system.

    Args:
        data (Data): Data to score.
        scorer_name (str): Name to assign to the syntactic complexity score. Default: "syntactic_complexity".
        model_name (str): spaCy model to use for dependency parsing. Default: "en_core_web_sm".

    Returns:
        Data: The input data with an additional score (scorer_name) for each MT system.
    """
    logging.info(f"Loading spaCy model: {model_name}.")
    nlp = spacy.load(model_name)

    scores = []
    logging.info("Computing syntactic complexity scores for each source sentence...")
    for sample in data.src_data_list:
        doc = nlp(sample["src"])
        height = compute_dependency_tree_height(doc)
        scores.append(height)

    assert len(scores) == len(data.src_data_list)
    # Since this is a source-based metric, the same score applies for all systems.
    for idx, sample in enumerate(data.src_data_list):
        for system in sample["scores"]:
            sample["scores"][system][scorer_name] = scores[idx]

    logging.info("Syntactic complexity scoring complete.")
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
            lp=args.lp,
            protocol=args.protocol,
            domains=args.domains,
        ),
        scorer_name=getattr(args, "scorer_name", "syntactic_complexity"),
        model_name=getattr(args, "syntactic_model_name", "en_core_web_sm"),
    )

    return scored_data
