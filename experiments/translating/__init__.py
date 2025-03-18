translation_system_prompt = "You are a professional translator."
translation_user_prompt = "Translate the following English text into {lang}. Return only the translated text, with no additional comments.\n\n{text}"

mt_systems = {}
llms = {"google/gemma-3-1b-it", "google/gemma-3-27b-it"}

gemma_models = {"google/gemma-3-1b-it"}
gemma_multimodal_models = {"google/gemma-3-27b-it"}
