import argparse
import os
from typing import List, Dict

import difficulty_estimation
from difficulty_estimation.data import Data

from translation import nllb_models, gemma_models, command_a_models, qwen_models
from translation.models import gemma3, command_a, nllb, qwen
from translation.utils import save_translations


def read_arguments():
    parser = argparse.ArgumentParser(
        description="Translate wmt24 sources from English into predefined target languages"
    )

    parser.add_argument(
        "--system",
        type=str,
        default="google/gemma-3-1b-it",
        help="The model to use for translation",
    )

    parser.add_argument(
        "--target-languages",
        type=str,
        nargs="+",
        default=[
            "Czech",
            "Spanish",
            "Hindi",
            "Icelandic",
            "Japanese",
            "Russian",
            "Ukrainian",
            "Chinese",
        ],
        help="List of target languages to translate into",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of sources to process in one batch",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "translations"
        ),
        help="Directory to save the translations (default: repository root/translations)",
    )

    return parser.parse_args()


def translate(
    sources: List[str],
    system: str,
    target_languages: List[str],
    batch_size: int,
) -> Dict[str, List[str]]:
    """
    Translate the given sources using the given system.

    Args:
        sources (List[str]): List of sources to translate
        system (str): The model to use for translation
        batch_size (int): Number of sources to process in one batch

    Returns:
        Dict[str, List[str]]: Dictionary containing a list of translations for each target language
    """

    if system in nllb_models:
        model = nllb.NLLB(system)
    elif system in gemma_models:
        model = gemma3.Gemma3(system)
    elif system in command_a_models:
        model = command_a.CommandA(system)
    elif system in qwen_models:
        model = qwen.Qwen(system)
    else:
        raise ValueError(f"Unknown system: {system}")

    return model.translate(sources, target_languages, batch_size)


if __name__ == "__main__":

    args = read_arguments()

    # The English sources in WMT24 are the same across target languages (except from Czech, where there are less sources)
    data = Data.load(
        dataset_name="wmt24",
        lps=["en-es"],
        protocol="esa",
        domains="all",
    )
    sources = [elem["src"] for elem in data.lp2src_data_list["en-es"]]

    print(
        f"Translating {len(sources)} English source segments using system {args.system}"
    )
    translations: Dict[str, List[str]] = translate(
        sources, args.system, args.target_languages, args.batch_size
    )

    # Save the translations to the specified output directory
    save_translations(translations, sources, args.output_dir, args.system)

    print(f"Translations saved to {args.output_dir}")
