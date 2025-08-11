# %%

import collections
from typing import Dict

import numpy as np
import difficulty_estimation
import difficulty_estimation.evaluate
import difficulty_estimation.utils
import difficulty_estimation.data
from difficulty_estimation.utils import difficulty2color

data_all = difficulty_estimation.utils.load_data_wmt_all(min_items=100, normalize=False)
data_year = collections.defaultdict(dict)

for (year, lp), data in data_all.items():
    if year in {"wmt21.tedtalks", "wmt21.flores", "wmt23.sent"}:
        continue
    year = year.split(".")[0]
    data_year[year][lp] = data

lps_in_all = set(data_year["wmt19"])
for year, lp2data in data_year.items():
    lps_in_all &= set(lp2data)
    print(lps_in_all)

print(lps_in_all)
years = sorted(data_year.keys())

# %%

def get_easy_hard_year(data_dict: Dict):
    # compute proportion of easy examples
    out = {}
    for lp in lps_in_all:
        data = data_dict[lp]
        # line = data[0]
        # print([line["scores"][sys]["human"] for sys in line["scores"].keys()])
        # print(([line["scores"][sys]["human"] >= 90 for sys in line["scores"].keys()]))
        # at least 50% of the systems have a human score >= 90
        # np.average([line["scores"][sys]["human"] for sys in line["scores"].keys()])
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

data_to_plot = collections.defaultdict(dict)
for year in years:
    lp2data = data_year[year]
    data_easy = get_easy_hard_year(lp2data)
    data_to_plot[year] = data_easy    
    print(year, data_easy)

# %%

import matplotlib.pyplot as plt
plt.rc("font", family="serif")
plt.figure(figsize=(3, 1.5))

years_txt = ["20" + year[-2:] for year in years]

# for lp in lps_in_all:
lp = "en-cs"

bottom = np.array([0.0] * len(years))
for cls in ["easy", "mixed", "hard"]:
    y = [data_to_plot[year][lp][cls] for year in years]
    plt.bar(
        years_txt, y,
        label=cls,
        color=difficulty2color[cls],
        bottom=bottom,
        width=0.7,
    )
    print(cls, difficulty2color[cls])
    for year, year_txt in zip(years, years_txt):
        val = int(y[years.index(year)] * 100)
        if val < 10:
            continue
        plt.text(
            year_txt, bottom[years.index(year)],
            f"{val}%",
            fontsize=8,
            ha="center",
            va="bottom",
        )
    bottom += np.array(y)

# plt.title("Difficulty for English$\\rightarrow$Czech across years", fontsize=9)
plt.yticks([])
plt.gca().spines[["left", "top", "right"]].set_visible(False)

plt.tight_layout(pad=0)
plt.savefig("../../generated/diachronic_saturation.pdf")
plt.show()