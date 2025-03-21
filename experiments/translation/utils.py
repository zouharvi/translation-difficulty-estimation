import os
import json
from typing import Dict, List, Tuple
from datetime import datetime
import argparse


def postprocess_translation(translation: str) -> str:
    """
    Clean up LLM-generated translations by removing any content after a newline.

    Args:
        translation (str): The translation text that might contain comments after one or more newlines

    Returns:
        str: The cleaned translation with only the actual translation content
    """
    # Split by double newline and take only the first part
    if "\n" in translation:
        return translation.split("\n")[0]

    return translation.strip()


def save_translations(
    translations: Dict[str, List[str]],
    sources: List[str],
    output_dir: str,
    system: str,
    postprocessed: bool = False,
) -> None:
    """
    Save translations to files in the specified directory.

    Args:
        translations (Dict[str, List[str]]): Dictionary with language keys and lists of translations
        sources (List[str]): Original source sentences in English
        output_dir (str): Directory to save the translations
        system (str): The system used for translation
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create system-specific subdirectory (clean up system name for filesystem)
    system_dir = os.path.join(output_dir, system.replace("/", "_"))
    os.makedirs(system_dir, exist_ok=True)

    # Also save the sources
    with open(os.path.join(system_dir, "sources.jsonl"), "w", encoding="utf-8") as f:
        for src in sources:
            f.write(f"{json.dumps(src, ensure_ascii=False)}\n")

    # For each language, create a jsonl file with one translation per line
    for lang, trans_list in translations.items():
        # Create filename with language
        filename = f"{lang.lower()}.jsonl"
        filepath = os.path.join(system_dir, filename)

        # Write translations, one per line as JSON
        with open(filepath, "w", encoding="utf-8") as f:
            for tgt in trans_list:
                f.write(f"{json.dumps(tgt, ensure_ascii=False)}\n")

        print(f"Saved {len(trans_list)} translations to {filepath}")


def load_translations(
    output_dir: str, system: str, languages: List[str]
) -> Tuple[Dict[str, List[str]], List[str]]:
    """
    Load translations from files in the specified directory.

    Args:
        output_dir (str): Directory where translations are saved
        system (str): The system used for translation
        languages (List[str]): List of language codes to load

    Returns:
        Tuple[Dict[str, List[str]], List[str]]: A tuple containing:
            - Dictionary with language keys and lists of translations
            - Original source sentences in English
    """
    system_dir = os.path.join(output_dir, system.replace("/", "_"))

    # Load sources
    sources = []
    with open(os.path.join(system_dir, "sources.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            sources.append(json.loads(line))

    # Load translations for each language
    translations = {}
    for lang in languages:
        filename = f"{lang.lower()}.jsonl"
        filepath = os.path.join(system_dir, filename)

        translations[lang] = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                translations[lang].append(json.loads(line))

    return translations, sources


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Postprocess and save translations")
    parser.add_argument(
        "--output_dir", required=True, help="Directory where translations are saved"
    )
    parser.add_argument(
        "--system", required=True, help="The system used for translation"
    )
    parser.add_argument(
        "--languages", required=True, nargs="+", help="Languages to process"
    )

    args = parser.parse_args()

    # Load translations
    translations, sources = load_translations(
        args.output_dir, args.system, args.languages
    )

    # Apply postprocessing to each translation
    postprocessed_translations = {}
    for lang, trans_list in translations.items():
        postprocessed_translations[lang] = [
            postprocess_translation(tgt) for tgt in trans_list
        ]

    # Save the postprocessed translations
    save_translations(postprocessed_translations, sources, args.output_dir, args.system)

    print(
        f"Successfully loaded, postprocessed, and saved translations for languages: {', '.join(args.languages)}"
    )
