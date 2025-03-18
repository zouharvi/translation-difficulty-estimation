import argparse
import os

import difficulty_sampling
from difficulty_sampling.data import Data, SrcData
from translating.translation import translate
from translating.utils import save_translations


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


if __name__ == "__main__":

    args = read_arguments()
    print(args)

    # The English sources in WMT24 are the same across target languages
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
    translations = translate(sources, args.system, args.batch_size)

    # Save the translations to the specified output directory
    save_translations(translations, sources, args.output_dir, args.system)

    print(f"Translations saved to {args.output_dir}")
