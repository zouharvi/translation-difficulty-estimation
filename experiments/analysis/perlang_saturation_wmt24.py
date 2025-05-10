# %%

from typing import Dict

import numpy as np
import difficulty_sampling
import difficulty_sampling.evaluate
import difficulty_sampling.utils
import difficulty_sampling.data

data_all = difficulty_sampling.utils.load_data_wmt_all(min_items=100, normalize=False)
data_all = {
    k[1]: v for k, v in data_all.items()
    if (k[0] == "wmt24") and (k[1].split("-")[0] == "en") and (k[1] != "en-de")
}

# %%

def get_easy_hard(data_dict: Dict):
    # compute proportion of easy/hard/mixsed examples
    out = {}
    for lp, data in data_dict.items():
        out[lp] = {
            "easy": np.average([
                np.average([line["scores"][sys]["human"] >= 90 for sys in line["scores"].keys()]) >= 0.5
                for line in data
            ]),
            "hard": np.average([
                np.average([line["scores"][sys]["human"] <= 66 for sys in line["scores"].keys()]) >= 0.5
                for line in data
            ]),
            "mixed": np.average([
                np.average([line["scores"][sys]["human"] <= 66 for sys in line["scores"].keys()]) < 0.5
                and
                np.average([line["scores"][sys]["human"] >= 90 for sys in line["scores"].keys()]) < 0.5
                for line in data
            ])
        }
    return out

data_to_plot = get_easy_hard(data_all)

# %%

import matplotlib.pyplot as plt
from difficulty_sampling.utils import tgt2lp, difficulty2color
lp2tgt = {v: k for k, v in tgt2lp.items()}
plt.rc("font", family="serif")
plt.figure(figsize=(3.5, 1.9))

# sort data_to_plot by easy
data_to_plot = dict(sorted(data_to_plot.items(), key=lambda item: item[1]["easy"]))
labels = list(data_to_plot.keys())

left = np.array([0.0]*len(data_to_plot))
for cls in ["easy", "mixed", "hard"]:
    labels_txt = [lp2tgt[lp].capitalize() for lp in labels]
    plt.barh(
        labels_txt,
        [data_to_plot[lp][cls] for lp in labels],
        left=left,
        label=cls,
        color=difficulty2color[cls],
    )
    for i, k in enumerate(data_to_plot):
        val = data_to_plot[k][cls]*100
        if val < 10:
            continue
        plt.text(
            left[i] + 0.01, i,
            f"{int(val)}%",
            fontsize=8,
            va="center",
        )
    left += np.array([data_to_plot[lp][cls] for lp in labels])

plt.gca().spines[["left", "top", "right", "bottom"]].set_visible(False)
plt.xticks([])

plt.tight_layout(pad=0)
plt.savefig("../../generated/perlang_saturation_wmt24.pdf")
plt.show()