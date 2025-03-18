import os
from typing import Dict, List
from datetime import datetime


def save_translations(
    translations: Dict[str, List[str]], sources: List[str], output_dir: str, system: str
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
    with open(os.path.join(system_dir, "sources.txt"), "w", encoding="utf-8") as f:
        for src in sources:
            f.write(f"{src}\n")

    # For each language, create a text file with one translation per line
    for lang, trans_list in translations.items():
        # Create filename with language
        filename = f"{lang.lower()}.txt"
        filepath = os.path.join(system_dir, filename)

        # Write translations, one per line
        with open(filepath, "w", encoding="utf-8") as f:
            for tgt in trans_list:
                f.write(f"{tgt}\n")

        print(f"Saved {len(trans_list)} translations to {filepath}")
