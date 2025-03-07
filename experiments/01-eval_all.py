"""
TODO: resolve the structure, things are a bit messy with subsampling not being in difficulty_sampling package
"""

# %%

import difficulty_sampling
import difficulty_sampling.evaluate
import difficulty_sampling.utils
import difficulty_sampling.data
import subsampling.sentinel
import subsampling.syntactic_complexity
import subsampling.word_frequency
import subset2evaluate.methods

def apply_src_len(data):
    for data in data.lp2src_data_list.values():
        for line in data:
            for sys in line["scores"].keys():
                line["scores"][sys]["src_len"] = -len(line["src"])


def apply_precomet(data, model="precomet_avg"):
    for data in data.lp2src_data_list.values():
        scores = subset2evaluate.methods.METHODS[model](data)
        for line in data:
            score = scores.pop(0)
            for sys in line["scores"].keys():
                line["scores"][sys][model] = -score

data = difficulty_sampling.data.Data.load(dataset_name="wmt24", lps=["en-x"], domains="all", protocol="esa")

# apply scorers to the whole data
apply_src_len(data)
subsampling.syntactic_complexity.syntactic_complexity_score(data, "syntactic_complexity")
apply_precomet(data, model="precomet_avg")
apply_precomet(data, model="precomet_var")
# subsampling.word_frequency.word_frequency_score(data, "word_frequency")
# subsampling.word_frequency.word_frequency_score(data, "word_zipf_frequency")

# TODO: format line for table

# full dataset to get the cluster count
print(difficulty_sampling.evaluate.main_eval_avg("src_len", data=data, B=None))

print(difficulty_sampling.evaluate.main_eval_avg("src_len", data=data, B=100))
print(difficulty_sampling.evaluate.main_eval_avg("syntactic_complexity", data=data, B=100))