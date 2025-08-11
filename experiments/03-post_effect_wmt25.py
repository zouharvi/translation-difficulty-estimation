# %%

import json
import sentence_transformers
import numpy as np
import tqdm
import difficulty_estimation.data
import subsampling.misc
import subsampling.syntactic_complexity
import subsampling.average_word_frequency
import matplotlib.pyplot as plt
import difficulty_estimation.utils
import scipy.stats
import subsampling.sentinel
from pathlib import Path

with open("../data/wmt25.jsonl", "r") as f:
    data_raw = [
        {
            "src": json.loads(x),
            "scores": {"mock": {}},
        }
        for x in f.readlines()
    ]

data_all = difficulty_estimation.data.Data(
    lp2src_data_list={"en": data_raw},
    lps=["en"],
    dataset_name="wmt25",
    protocol=None,
    domains="news",
)


import language_tool_python

grammarcheck = language_tool_python.LanguageTool("en-US")
src2error = list(
    {
        line["src"]
        for data_name, data in data_all.lp2src_data_list.items()
        for line in data
        # do this only for English
        if data_name.split("-")[0] == "en"
    }
)
src2error = {src: len(grammarcheck.check(src)) / len(src.split()) for src in src2error}
grammarcheck.close()

# compute some protected attributes
for data_name, data in list(data_all.lp2src_data_list.items()):
    for line in tqdm.tqdm(data):
        line["effect"] = {
            "grammaticality": src2error[line["src"]],
            "length": len(line["src"]),
        }

# %%

# apply scorers to the whole data
subsampling.sentinel.sentinel_src_metric_model_score(
    subsampling.sentinel.get_sentinel_src_metric_model(
        "Prosho/sentinel-src-mqm-wmt1723"
    ),
    scorer_name="sentinel-src-mqm-wmt1723",
    data=data_all,
    use_tgt_lang_token=False,
)
subsampling.misc.apply_subset2evaluate(data_all, method="random")
subsampling.syntactic_complexity.syntactic_complexity_score(
    data_all, "syntactic_complexity"
)
subsampling.syntactic_complexity.src_len_score(
    data_all,
    "src_len",
)

# %%


data_size = len(list(data_all.lp2src_data_list.values())[0])


def avg_effect_across_langs(key, en_only=False):
    return [
        data[i]["effect"][key]
        for data_name, data in data_all.lp2src_data_list.items()
        if (not en_only) or (data_name.split("-")[0] == "en")
        for i in range(data_size)
    ]


def avg_difficulty_across_langs(key, en_only=False):
    return [
        np.average(
            [
                np.average(
                    [data[i]["scores"][sys][key] for sys in data[i]["scores"].keys()]
                )
                for data in data_all.lp2src_data_list.values()
                if (not en_only) or (data_name.split("-")[0] == "en")
            ]
        )
        for i in range(data_size)
    ]


import collections

METHOD_CORR = collections.defaultdict(dict)


def plot_problem(ax, data_x, data_y, key_x, key_y):
    # filter out None from data_y, but in parallel
    data_x, data_y = zip(
        *[(x, y) for x, y in zip(data_x, data_y) if x is not None and y is not None]
    )
    ax.scatter(
        data_x,
        data_y,
        s=5,
        color="black",
        alpha=0.7,
        linewidth=0,
    )
    ax.spines[["top", "right"]].set_visible(False)

    corr_pearson = scipy.stats.pearsonr(data_x, data_y)[0]
    corr_spearman = scipy.stats.spearmanr(data_x, data_y)[0]

    if key_x == "random":
        corr_pearson = 0.0
        corr_spearman = 0.0

    METHOD_CORR[key_y][key_x] = {
        "pearson": corr_pearson,
        "spearman": corr_spearman,
    }
    ax.text(
        0.95,
        0.05,
        f"ρ={corr_pearson:.2f} (Pearson)\nρ={corr_spearman:.2f} (Spearman)",
        transform=ax.transAxes,
        fontsize=8,
        ha="right",
        va="bottom",
    )


METHOD_TO_NAME = {
    "random": "Random",
    "syntactic_complexity": "Syntactic Complexity",
    "sentinel-src-mqm-wmt1723": "Sentinel",
    "src_len": "Length",
}

fig, axss = plt.subplots(
    len(METHOD_TO_NAME),
    5,
    figsize=(11, 2.1 * len(METHOD_TO_NAME)),
    width_ratios=[1, 1, 1, 1, 1],
)
for axs_i, (axs, (method, method_name)) in enumerate(zip(axss, METHOD_TO_NAME.items())):
    plot_problem(
        axs[0],
        avg_effect_across_langs("length"),
        avg_difficulty_across_langs(method),
        key_x="length (characters)",
        key_y=method_name,
    )

    plot_problem(
        axs[4],
        avg_effect_across_langs("grammaticality", en_only=True),
        avg_difficulty_across_langs(method),
        key_x="grammar errors per word",
        key_y=method_name,
    )

    axs[0].set_ylabel(method_name)

    if axs_i == len(METHOD_TO_NAME) - 1:
        axs[0].set_xlabel("Length")
        axs[4].set_xlabel("Grammaticality")


plt.tight_layout(pad=0)
plt.savefig("../generated/03-post_effect_wmt25.pdf")
plt.show()


# %%

fout = open("../generated/03-post_effect_corr_wmt25.tex", "w")


print(
    "\\begin{tabular}{l" + "r" * len(METHOD_CORR["Length"]) + "} \n \\toprule",
    file=fout,
)
METHODNAME_TO_SHORT = {
    "Random": "Random",
    "Syntactic Complexity": "Syntax Complexity",
    "Sentinel": "Sentinel",
    "Length": "Length",
}
print(
    r"""
    & \multicolumn{2}{c}{\bf Source} & \multicolumn{2}{c}{\bf Diversity} & \bf Unique \\
    & \bf length & \bf errors & \bf embd & \bf chrF\hspace{0.5mm} & \bf outputs \\
    \midrule
    """,
    file=fout,
)


def format_cell(v, minv=0, maxv=1.1):
    va = abs(v)
    color = int(100 * (va - minv) / (maxv - minv))
    return f"\\cellcolor{{red!{color}}} ${v:.2f}$"


METHOD_CORR["Random"] = {
    k: {"pearson": 0.0, "spearman": 0.0} for k in METHOD_CORR["Random"].keys()
}

KEYX = [
    "length (characters)",
    "grammar errors per word",
]
for method_name, method_v in METHOD_CORR.items():
    print(f"{METHODNAME_TO_SHORT[method_name]} & ", file=fout)
    print(
        *[format_cell(method_v[k]["pearson"]) for k in KEYX],
        sep=" & ",
        end="\\\\\n",
        file=fout,
    )

print("\\bottomrule ", file=fout)
print("\\end{tabular}", file=fout)
fout.close()
