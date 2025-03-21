from typing import List, Dict
from tqdm import tqdm

import torch
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    Gemma3ForCausalLM,
    Gemma3ForConditionalGeneration,
)

from translation.utils import postprocess_translation


class Gemma3:

    text2text_models = {"google/gemma-3-1b-it"}
    imagetext2text_models = {"google/gemma-3-27b-it"}

    translation_system_prompt = "You are a professional translator. Return only the translated text, with no additional comments, notes, or explanations."
    translation_user_prompt = (
        "Translate the following English text into {lang}.\n\n{text}"
    )

    def __init__(self, model_name: str):

        if model_name in self.text2text_models:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = Gemma3ForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.bfloat16, device_map="auto"
            ).eval()
        elif model_name in self.imagetext2text_models:
            self.tokenizer = AutoProcessor.from_pretrained(model_name)
            self.model = Gemma3ForConditionalGeneration.from_pretrained(
                model_name, torch_dtype=torch.bfloat16, device_map="auto"
            ).eval()
        else:
            raise ValueError(f"Unknown Gemma model: {model_name}")

        self.tokenizer.padding_side = "left"

    def translate(
        self,
        sources: List[str],
        target_languages: List[str],
        batch_size: int,
    ) -> Dict[str, List[str]]:
        """
        Translate the given sources into the target languages.

        Args:
            sources (List[str]): List of sources to translate
            target_languages (List[str]): List of target languages to translate into
            batch_size (int): Number of sources to process in one batch

        Returns:
            Dict[str, List[str]]: Dictionary containing a list of translations for each target language
        """

        # Dictionary to store translations for each target language
        translations = {}

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
                                {
                                    "type": "text",
                                    "text": self.translation_system_prompt,
                                },
                            ],
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": self.translation_user_prompt.format(
                                        lang=lang, text=text
                                    ),
                                },
                            ],
                        },
                    ]
                    for text in batch_sources
                ]

                inputs = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    padding=True,
                ).to(self.model.device)

                with torch.inference_mode():
                    outputs = self.model.generate(
                        **inputs, max_new_tokens=512, do_sample=False
                    )

                decoded_translations = self.tokenizer.batch_decode(
                    outputs[:, inputs["input_ids"].shape[1] :],
                    skip_special_tokens=True,
                )

                postprocessed_translations = [
                    postprocess_translation(t) for t in decoded_translations
                ]

                translations[lang].extend(postprocessed_translations)

        return translations
