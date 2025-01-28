# %%

import difficulty_sampling
import copy
import numpy as np

# load 572 items
data = difficulty_sampling.utils.load_data_wmt_all()[("wmt24", "en-cs")]
len(data)

# %%
# Example 1: difficulty sampler is based on the length of the source sentence (bad, I know)
def my_difficulty_sampler_1(line):
    return len(line["src"])

data_new = copy.deepcopy(data)
data_new.sort(key=my_difficulty_sampler_1)

clus, cors = difficulty_sampling.evaluate.eval_clu_cor(data_new, data)
# print average clusters and average correlation
print(f"CLU: {np.mean(clus):.2f}, {np.mean(cors):.1%}")

# %%
# Example 2: difficulty sampler is based on average word length in source
def my_difficulty_sampler_2(line):
    word_len = [len(word) for word in line["src"].split()]
    return np.average(word_len)

data_new = copy.deepcopy(data)
data_new.sort(key=my_difficulty_sampler_2)

clus, cors = difficulty_sampling.evaluate.eval_clu_cor(data_new, data)
# print average clusters and average correlation
print(f"CLU: {np.mean(clus):.2f}, {np.mean(cors):.1%}")


# %%
# Example 3: use py-readability-metrics
# For this, you need to install the package first: pip install py-readability-metrics
from readability import Readability

def my_difficulty_sampler_3(line):
    # hack: duplicate to have at least 100 sentences
    return Readability((line["src"]+" apple. ")*100).flesch_kincaid().score

data_new = copy.deepcopy(data)
data_new.sort(key=my_difficulty_sampler_3)

clus, cors = difficulty_sampling.evaluate.eval_clu_cor(data_new, data)
# print average clusters and average correlation
print(f"CLU: {np.mean(clus):.2f}, {np.mean(cors):.1%}")