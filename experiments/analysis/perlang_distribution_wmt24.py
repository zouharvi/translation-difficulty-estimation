# %%

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

def get_all_scores(data_local):
    return [
        line["scores"][sys]["human"]
        for line in data_local
        for sys in line["scores"].keys()
    ]

def get_top_score(data_local):
    return [
        max([
            line["scores"][sys]["human"]
            for sys in line["scores"].keys()
        ])
        for line in data_local
    ]

def get_top_system(data_local):
    # get top-k systems
    sys_best = sorted(
        list(data_local[0]["scores"].keys()),
        key=lambda sys: np.average([line["scores"][sys]["human"] for line in data_local]),
        reverse=True
    )[:1]

    return [
        line["scores"][sys]["human"]
        for line in data_local
        for sys in sys_best
    ]

import matplotlib.pyplot as plt
from difficulty_sampling.utils import tgt2lp
lp2tgt = {v: k for k, v in tgt2lp.items()}
difficulty_sampling.utils.matplotlib_default()

_, axs= plt.subplots(4, 2, figsize=(3.9, 3.6), sharex=True, sharey=True)
axs = list(axs.flatten())
for (lp, data_local), ax in zip(data_all.items(), axs):
    data_flat1 = get_all_scores(data_local)
    data_flat2 = get_top_system(data_local)
    data_flat3 = get_top_score(data_local)

    ax.hist(
        [data_flat1, data_flat2, data_flat3],
        bins=range(0, 100+15, 15),
        density=True,
        label=["All", "Top system", "Top translation"],
    )
    # r"En$\rightarrow$" + 
    ax.text(
        0.05,
        0.75,
        lp2tgt[lp].capitalize(),
        # top left
        transform=ax.transAxes,
        weight="bold",
    )
    if lp in {"en-cs", "en-hi", "en-ja", "en-uk"}:
        ax.set_ylabel("Freq.")
    ax.set_yticks([])
    if lp in {"en-uk", "en-zh"}:
        ax.set_xlabel("ESA Score")
    ax.spines[["top", "right"]].set_visible(False)

handles = axs[0].get_legend_handles_labels()
plt.tight_layout(pad=0.2)
plt.savefig(difficulty_sampling.ROOT / "generated/perlang_wmt24.pdf")
plt.show()

# %%

# only legend

plt.figure(figsize=(3.9, 0.5))
plt.legend(
    *handles,
    loc="center",
    ncol=3,
    frameon=False,
)
# turn off axis
plt.gca().spines[["left", "top", "right", "bottom"]].set_visible(False)
plt.xticks([])
plt.yticks([])
plt.tight_layout(pad=0)
plt.savefig(difficulty_sampling.ROOT / "generated/perlang_wmt24_legend.pdf")
plt.show()