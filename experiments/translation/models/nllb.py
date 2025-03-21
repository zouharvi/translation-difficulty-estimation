from typing import List, Dict
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class NLLB:

    models = {"facebook/nllb-moe-54b"}

    lang2lang_code = {
        "English": "eng_Latn",
        "Chinese": "zho_Hans",
        "German": "deu_Latn",
        "Spanish": "spa_Latn",
        "Italian": "ita_Latn",
        "Russian": "rus_Cyrl",
        "Bulgarian": "bul_Cyrl",
        "Dutch": "nld_Latn",
        "Slovene": "slv_Latn",
        "Czech": "ces_Latn",
        "Hindi": "hin_Deva",
        "Icelandic": "isl_Latn",
        "Japanese": "jpn_Japn",
        "Ukrainian": "ukr_Cyrl",
    }

    def __init__(self, model_name: str):
        if model_name in self.models:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name, device_map="auto", torch_dtype=torch.bfloat16
            ).eval()
        else:
            raise ValueError(f"Unknown NLLB model: {model_name}")

    def get_lang_code(self, lang: str):
        return self.lang2lang_code.get(lang, None)

    def translate(
        self, sources: List[str], target_languages: List[str], batch_size: int
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

            src_lang_code = self.get_lang_code("English")
            tgt_lang_code = self.get_lang_code(lang)

            self.tokenizer.src_lang = src_lang_code

            # Process sources in batches
            for i in tqdm(
                range(0, len(sources), batch_size),
                desc=f"Translating to {lang}",
                total=(len(sources) + batch_size - 1) // batch_size,
            ):
                batch_sources = sources[i : i + batch_size]

                inputs = self.tokenizer(
                    batch_sources,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to(self.model.device)

                with torch.inference_mode():
                    output = self.model.generate(
                        **inputs,
                        forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(
                            tgt_lang_code
                        ),
                        return_dict_in_generate=True,
                        max_new_tokens=512,
                        do_sample=False,
                    )

                decoded_translations = [
                    t.strip()
                    for t in self.tokenizer.batch_decode(
                        output.sequences, skip_special_tokens=True
                    )
                ]

                translations[lang].extend(decoded_translations)

        return translations
