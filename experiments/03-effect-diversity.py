# %%

import pickle
from typing import List

data_external = pickle.load(open("/home/vilda/Downloads/sys2translations.pickle", "rb"))

# %%

def compute_diversity(data: List):
    for line in data:
        line["diversity"] = {}
        pass
    pass

# %%
print(data.keys())
print(len(data['google_gemma-3-27b-it']["chinese"]))