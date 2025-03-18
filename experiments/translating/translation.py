from typing import List, Dict
from tqdm import tqdm
from typing import Iterable, Optional, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from translating import (
    mt_systems,
    llms,
    gemma_models,
    gemma_multimodal_models,
    translation_user_prompt,
    translation_system_prompt,
)


def translate(
    sources: List[str],
    system: str,
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
    if system in mt_systems:
        return mt_system_translate(sources, system, batch_size)
    elif system in llms:
        return llm_translate(sources, system, batch_size)
    else:
        raise ValueError(f"Unknown system: {system}")


def mt_system_translate(
    sources: List[str],
    system: str,
    batch_size: int,
) -> Dict[str, List[str]]:
    """
    Translate the given sources using the given machine translation system.

    Args:
        sources (List[str]): List of sources to translate
        system (str): The model to use for translation
        batch_size (int): Number of sources to process in one batch

    Returns:
        Dict[str, List[str]]: Dictionary containing a list of translations for each target language
    """
    pass


def llm_translate(
    sources: List[str],
    system: str,
    batch_size: int,
) -> Dict[str, List[str]]:
    """
    Translate the given sources using the given LLM.

    Args:
        sources (List[str]): List of sources to translate
        system (str): The model to use for translation
        batch_size (int): Number of sources to process in one batch

    Returns:
        Dict[str, List[str]]: Dictionary containing a list of translations for each target language
    """

    if system in gemma_models:
        return translate_with_gemma(sources, system, batch_size)
    elif system in gemma_multimodal_models:
        return translate_with_gemma_multimodal(sources, system, batch_size)
    else:
        raise ValueError(f"Unknown LLM: {system}")


def translate_with_gemma_multimodal(
    sources: List[str],
    system: str,
    batch_size: int,
) -> Dict[str, List[str]]:
    """
    Translate the given sources using one of the new multimodal gemma-3 models.

    Args:
        sources (List[str]): List of sources to translate
        system (str): The model to use for translation
        batch_size (int): Number of sources to process in one batch

    Returns:
        Dict[str, List[str]]: Dictionary containing a list of translations for each target language
    """
    from transformers import AutoProcessor, Gemma3ForConditionalGeneration

    tokenizer = AutoProcessor.from_pretrained(system)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        system, torch_dtype=torch.float16, device_map="auto"
    ).eval()

    # Dictionary to store translations for each target language
    translations = {}

    # List of target languages (can be extended in the future)
    target_languages = ["Spanish"]

    for lang in tqdm(target_languages, desc="Processing languages"):
        translations[lang] = []

        # Process sources in batches
        for i in tqdm(
            range(0, len(sources), batch_size),
            desc=f"Translating to {lang}",
            total=(len(sources) + batch_size - 1) // batch_size,
        ):
            batch_sources = sources[i : i + batch_size]

            messages = [
                [
                    {
                        "role": "system",
                        "content": [
                            {"type": "text", "text": translation_system_prompt},
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": translation_user_prompt.format(
                                    lang=lang, text=text
                                ),
                            },
                        ],
                    },
                ]
                for text in batch_sources
            ]

            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
            ).to(model.device, dtype=torch.float16)

            with torch.inference_mode():
                outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False)

            decoded_output = tokenizer.batch_decode(
                outputs[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )

            translations[lang].extend(decoded_output)

    return translations


def translate_with_gemma(
    sources: List[str],
    system: str,
    batch_size: int,
) -> Dict[str, List[str]]:
    """
    Translate the given sources using one of the new gemma-3 models.

    Args:
        sources (List[str]): List of sources to translate
        system (str): The model to use for translation
        batch_size (int): Number of sources to process in one batch

    Returns:
        Dict[str, List[str]]: Dictionary containing a list of translations for each target language
    """
    from transformers import Gemma3ForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(system)
    model = Gemma3ForCausalLM.from_pretrained(
        system, torch_dtype=torch.float16, device_map="cuda:0"
    ).eval()

    # Dictionary to store translations for each target language
    translations = {}

    # List of target languages (can be extended in the future)
    target_languages = ["Spanish"]

    for lang in tqdm(target_languages, desc="Processing languages"):
        translations[lang] = []

        # Process sources in batches
        for i in tqdm(
            range(0, len(sources), batch_size),
            desc=f"Translating to {lang}",
            total=(len(sources) + batch_size - 1) // batch_size,
        ):
            batch_sources = sources[i : i + batch_size]

            messages = [
                [
                    {
                        "role": "system",
                        "content": [
                            {"type": "text", "text": translation_system_prompt},
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": translation_user_prompt.format(
                                    lang=lang, text=text
                                ),
                            },
                        ],
                    },
                ]
                for text in batch_sources
            ]

            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
            ).to(model.device, dtype=torch.float16)

            with torch.inference_mode():
                outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False)

            decoded_output = tokenizer.batch_decode(
                outputs[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )

            translations[lang].extend(decoded_output)

    return translations
