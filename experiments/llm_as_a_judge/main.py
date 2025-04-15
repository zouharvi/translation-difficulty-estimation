import pandas as pd
import hashlib
import ipdb
import asyncio
import os
from tqdm import tqdm
# TODO class estimator needs implementation



async def call_model(prompts, model="command-a-03-2025"):
    if "command" in model:
        estimator = CommandEstimator(model=model, temperature=0.0, max_tokens=16000)
    elif "gpt" in model:
        estimator = GPTEstimator(model=model, temperature=0.0, max_tokens=16000)
    await estimator.apply("chat", prompts)
    await estimator.cleanup() # close connections, avoids warnings for many estimators
    return prompts

TEMPLATE_source = "You are given a source text. Your goal is to determine the approximate proficiency level required to translate this text, based on a detailed analysis of its complexity. The final result should be reported as a single numeric score on a scale of 0 to 120, where higher numbers correspond to a higher difficulty (i.e., more advanced language proficiency requirements). You should also relate this numeric score to commonly recognized proficiency levels (e.g., A1, A2, B1, B2, C1, C2). Here is the expected mapping: 0-20 for A1 (Beginner); 21-40 for A2 (Elementary); 41-60 for B1 (Intermediate); 61-80 for B2 (Upper Intermediate); 81-100 for C1 (Advanced); 101-120 for C2 (Mastery).\n\nInstructions: First, examine the text to identify features that affect reading difficulty, including complexity of vocabulary, grammar, semantic density, and any specialized knowledge required. Then, provide a brief explanation of your reasoning for each major factor. Consider whether the text includes domain-specific terminology, cultural references, idiomatic expressions, or advanced grammatical constructions. Finally, assign a numeric score from 0 to 120 and map that score to one of the CEFR levels. Conclude with a final statement that clearly states your numeric score and the corresponding proficiency level surrounded by triple square brackets, for example [[[86, C1 (Advanced)]]].\n\nAnalyze following text:\n{src}"


TEMPLATE_target = "You are given a source text. Your goal is to determine the approximate proficiency level required to translate this text into {target_language}, based on a detailed analysis of its complexity. The final result should be reported as a single numeric score on a scale of 0 to 120, where higher numbers correspond to a higher difficulty (i.e., more advanced language proficiency requirements). You should also relate this numeric score to commonly recognized proficiency levels (e.g., A1, A2, B1, B2, C1, C2). Here is the expected mapping: 0-20 for A1 (Beginner); 21-40 for A2 (Elementary); 41-60 for B1 (Intermediate); 61-80 for B2 (Upper Intermediate); 81-100 for C1 (Advanced); 101-120 for C2 (Mastery).\n\nInstructions: First, examine the text to identify features affecting the translation into {target_language}, which affect reading difficulty, including complexity of vocabulary, grammar, semantic density, and any specialized knowledge required. Then, provide a brief explanation of your reasoning for each major factor. Consider whether the text includes domain-specific terminology, cultural references, idiomatic expressions, or advanced grammatical constructions. Finally, assign a numeric score from 0 to 120 and map that score to one of the CEFR levels. Conclude with a final statement that clearly states your numeric score and the corresponding proficiency level surrounded by triple square brackets, for example [[[86, C1 (Advanced)]]].\n\nAnalyze following text:\n{src}"

def get_answers(df, model="command-a-03-2025", template=TEMPLATE_source):
    prompts = []
    ids = []
    for idx, row in df.iterrows():
        prompt = template.format(**row)
        prompts.append(ChatSample.from_prompt(prompt))
        ids.append(idx)
        
    answers = asyncio.run(call_model(prompts, model))
    checked_answers = {}
    for idx, answer in zip(ids, answers):
        # parse final statement
        score = None
        assessment = None
        if answer.generations is not None:
            assessment = answer.generations[0].strip()
            score = assessment.split("]]]")[0].split("[[[")[-1].strip()

        checked_answers[idx] = {"assessment": assessment, "score": score, "hash_id": idx}
    return checked_answers


language_mapping = {
    "cs": "Czech",
    "de": "German",
    "en": "English",
    "he": "Hebrew",
    "ja": "Japanese",
    "zh": "Chinese",
    "ru": "Russian",
    "sah": "Yakut",
    "kk": "Kazakh",
    "gu": "Gujarati",
    "lt": "Lithuanian",
    "uk": "Ukrainian",
    "es": "Spanish",
    "hi": "Hindi",
    "is": "Icelandic",
    "hr": "Croatian"
}


SOURCE_BASED = True
model = "command-a-03-2025"
# model = "gpt-4o-1120"
SAMPLE_SIZE = 100


data_filename = "../../wmt24_data_src_only.csv" if SOURCE_BASED else "../../wmt24_data.csv"
filename = f"cache/{model}_source_answers.json" if SOURCE_BASED else f"cache/{model}_target_answers.json"
TEMPLATE = TEMPLATE_source if SOURCE_BASED else TEMPLATE_target

data = pd.read_csv(data_filename)
if not SOURCE_BASED:
    data['target_language'] = data['lp'].apply(lambda x: x.split("-")[1])
    data['target_language'] = data['target_language'].map(language_mapping)
    data['hash_id'] = data.apply(lambda x: hashlib.md5((x['src'] + x['lp']).encode()).hexdigest(), axis=1)
else:
    data['hash_id'] = data.apply(lambda x: hashlib.md5(x['src'].encode()).hexdigest(), axis=1)

data.set_index('hash_id', inplace=True)

if os.path.exists(filename):
    # preload the answers
    answers_cache = pd.read_json(filename, orient="records")
    df = data.merge(answers_cache, on="hash_id", how="left")
    df.set_index('hash_id', inplace=True)
else:
    answers_cache = pd.DataFrame()
    df = data.copy()
    df["assessment"] = None
    df["score"] = None

missing_df = df[df["assessment"].isnull()]

for i in tqdm(range(0, len(missing_df), SAMPLE_SIZE), total=len(missing_df)//SAMPLE_SIZE):
    # get i to i+100
    subdf_sample = missing_df.iloc[i:i+SAMPLE_SIZE]
    answers = get_answers(subdf_sample, model, TEMPLATE)
    for idx, row in answers.items():
        hash_id = row["hash_id"]
        assert isinstance(hash_id, str), f"hash_id is not str: {hash_id}"

        df.loc[hash_id, "assessment"] = row["assessment"]
        df.loc[hash_id, "score"] = row["score"]

        answers_cache.loc[hash_id, "assessment"] = row["assessment"]
        answers_cache.loc[hash_id, "score"] = row["score"]
        answers_cache.loc[hash_id, "hash_id"] = row["hash_id"]
    # keep index as a column hash_id
    answers_cache.to_json(filename, force_ascii=False, orient="records")

if SOURCE_BASED:
    data_with_scores = data.merge(df[['src', 'assessment', 'score']], on='src', how='left')
    data_with_scores.to_csv(f"scoring/{model}_source_based_answers.csv", index=False)
else:
    data_with_scores = data.merge(df[['lp', 'src', 'assessment', 'score']], on=['lp', 'src'], how='left')
    data_with_scores.to_csv(f"scoring/{model}_target_based_answers.csv", index=False)




