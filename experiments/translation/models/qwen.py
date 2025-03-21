from typing import List, Dict
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from translation.utils import postprocess_translation


class Qwen:

    models = {"Qwen/Qwen2.5-72B-Instruct"}

    translation_system_prompt = "You are a professional translator. Return only the translated text, with no additional comments, notes, or explanations."
    translation_user_prompt = (
        "Translate the following English text into {lang}.\n\n{text}"
    )

    def __init__(self, model_name: str):

        if model_name in self.models:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.bfloat16, device_map="auto"
            ).eval()
        else:
            raise ValueError(f"Unknown CommandA model: {model_name}")

        self.tokenizer.padding_side = "left"

    def translate(
        self,
        sources: List[str],
        target_languages: List[str],
        batch_size: int,
    ) -> Dict[str, List[str]]:
        """
        Translate the given sources using a Qwen model.

        Args:
            sources (List[str]): List of sources to translate
            system (str): The model to use for translation
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
                            "content": self.translation_system_prompt,
                        },
                        {
                            "role": "user",
                            "content": self.translation_user_prompt.format(
                                lang=lang, text=text
                            ),
                        },
                    ]
                    for text in batch_sources
                ]

                inputs = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_tensors="pt",
                    padding=True,
                    return_dict=True,
                ).to(self.model.device)

                with torch.inference_mode():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=512,
                        do_sample=False,
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
