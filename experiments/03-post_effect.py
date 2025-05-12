# %%

import sentence_transformers
import itertools
import numpy as np
import tqdm
import difficulty_sampling.data
import subsampling.misc
import subsampling.syntactic_complexity
import subsampling.negative_word_frequency
import matplotlib.pyplot as plt
import difficulty_sampling.utils
import scipy.stats
import subsampling.sentinel
from pathlib import Path
from fastchrf import pairwise_chrf

data_all = difficulty_sampling.data.Data.load(
    dataset_name="wmt24", lps=["en-x"], domains="all", protocol="esa"
)

model = sentence_transformers.SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# map tgt to embedding
tgt2embd = list({
    tgt
    for data in data_all.lp2src_data_list.values()
    for line in data
    for tgt in line["tgt"].values()
})
tgt2embd = dict(zip(tgt2embd, model.encode(tgt2embd)))

# %%
import language_tool_python
grammarcheck = language_tool_python.LanguageTool('en-US')
src2error = list({
    line["src"]
    for data in data_all.lp2src_data_list.values()
    for line in data
})
src2error = {src: len(grammarcheck.check(src))/len(src.split()) for src in src2error}
grammarcheck.close()

# %%

def symmetric_chrf(tgt1, tgt2):
    out = pairwise_chrf([[tgt1], [tgt2]], [[tgt2], [tgt1]])
    return (out[0][0][0]+out[1][0][0])/2

# compute some protected attributes
for data in tqdm.tqdm(list(data_all.lp2src_data_list.values())):
    for line in data:
        output_unique = len({tgt for tgt in line["tgt"].values()})/len(line["tgt"])
        output_diversity_ip = np.average([
            # cosine similarity
            -sentence_transformers.util.pytorch_cos_sim(
                tgt2embd[tgt1], tgt2embd[tgt2]
            ).item()
            for (sys1, tgt1), (sys2, tgt2) in itertools.product(line["tgt"].items(), line["tgt"].items())
            if sys1 != sys2
        ])
        output_diversity_chrf = np.average([
            100-symmetric_chrf(tgt1, tgt2)
            for (sys1, tgt1), (sys2, tgt2) in itertools.product(line["tgt"].items(), line["tgt"].items())
            if sys1 != sys2
        ])
        line["effect"] = {
            "output_diversity_ip": output_diversity_ip,
            "output_diversity_chrf": output_diversity_chrf,
            "output_unique": output_unique,
            "grammaticality": src2error[line["src"]],
            "length": len(line["src"]),
        }

# %%
# apply scorers to the whole data
subsampling.sentinel.sentinel_src_metric_model_score(
    subsampling.sentinel.get_sentinel_src_metric_model("Prosho/sentinel-src-mqm-wmt1923"),
    scorer_name="sentinel-src-mqm-wmt1923",
    data=data_all,
    use_tgt_lang_token=True,
)
subsampling.misc.apply_subset2evaluate(data_all, method="random")
subsampling.syntactic_complexity.syntactic_complexity_score(
    data_all, "syntactic_complexity"
)
subsampling.misc.apply_external_artificial_crowd_metrics(
    data_all,
    sys2translations_path=Path(
        "../data/artificial_crowd/scored_translations/sys2translations.pickle"
    ),
    metric="MetricX-24-Hybrid-QE-XXL", 
)
subsampling.misc.apply_llm_as_a_judge(
    data_all,
    scored_source_texts_df_path=Path(
        "../data/LLM-as-a-Judge/new/command-a/wmt_data_with_source_based_num_scores.csv"
    ),
    llm_name="Command-A_new",
)
subsampling.misc.apply_llm_as_a_judge(
    data_all,
    scored_source_texts_df_path=Path(
        "../data/LLM-as-a-Judge/new/gpt-4o/gpt-4o-1120_source_based_num_scores.csv"
    ),
    llm_name="GPT-4o",
)


# %%


data_size = len(list(data_all.lp2src_data_list.values())[0])

def avg_effect_across_langs(key):
    return [
        np.average([data[i]["effect"][key] for data in data_all.lp2src_data_list.values()])
        for i in range(data_size)
    ]

def avg_difficulty_across_langs(key):
    return [
        np.average([np.average([data[i]["scores"][sys][key] for sys in data[i]["scores"].keys()]) for data in data_all.lp2src_data_list.values()])
        for i in range(data_size)
    ]

import collections
METHOD_CORR = collections.defaultdict(dict)

def plot_problem(ax, data_x, data_y, key_x, key_y):
    ax.scatter(
        data_x, data_y,
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
        0.95, 0.05,
        f"ρ={corr_pearson:.2f} (Pearson)\nρ={corr_spearman:.2f} (Spearman)",
        transform=ax.transAxes,
        fontsize=8,
        ha="right",
        va="bottom",
    )


METHOD_TO_NAME = {
    "random": "Random",
    "LLM-as-a-Judge (Command-A_new, src-based)": "LLM-as-a-Judge",
    "syntactic_complexity": "Syntactic Complexity",
    "ext_artcrowd|MetricX-24-Hybrid-QE-XXL": "Artificial Crowd",
    "sentinel-src-mqm-wmt1923": "Sentinel",
    "human": "Oracle",
}

fig, axss = plt.subplots(
    len(METHOD_TO_NAME), 5,
    figsize=(11, 2.5*len(METHOD_TO_NAME)),
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
        axs[1],
        avg_effect_across_langs("output_unique"),
        avg_difficulty_across_langs(method),
        key_x="% output unique",
        key_y=method_name,
    )

    plot_problem(
        axs[2],
        avg_effect_across_langs("output_diversity_ip"),
        avg_difficulty_across_langs(method),
        key_x="output diversity (embd)",
        key_y=method_name,
    )

    plot_problem(
        axs[3],
        avg_effect_across_langs("output_diversity_chrf"),
        avg_difficulty_across_langs(method),
        key_x="output diversity (chrF)",
        key_y=method_name,
    )

    plot_problem(
        axs[4],
        avg_effect_across_langs("grammaticality"),
        avg_difficulty_across_langs(method),
        key_x="grammar errors per word",
        key_y=method_name,
    )

    axs[0].set_ylabel(method_name)

    if axs_i == len(METHOD_TO_NAME)-1:
        axs[0].set_xlabel("Length")
        axs[1].set_xlabel("Output Uniqueness")
        axs[2].set_xlabel("Output Diversity (Embedding)")
        axs[3].set_xlabel("Output Diversity (chrF)")
        axs[4].set_xlabel("Grammaticality")


plt.tight_layout(pad=0)
plt.savefig("../generated/03-post_effect.pdf")
plt.show()


# %%

fout = open("../generated/03-post_effect_corr.tex", "w")

# import sys
# fout = sys.stdout

print("\\begin{tabular}{l" + "r" * len(METHOD_CORR["Oracle"]) + "} \n \\toprule", file=fout)
METHODNAME_TO_SHORT = {
    "Random": "Random",
    "LLM-as-a-Judge": "LLM-as-a-Judge",
    "Syntactic Complexity": "Syntax Complexity",
    "Artificial Crowd": "Artificial Crowd",
    "Sentinel": "Sentinel",
    "Oracle": "Oracle",
}
print(r"""
    & \multicolumn{2}{c}{Source} & \multicolumn{2}{c}{Diversity} & Unique \\
    & length & errors & embd & chrF & outputs \\
    \midrule
    """,
    file=fout
)

def format_cell(v, minv=0, maxv=1.1):
    va = abs(v)
    color = int(100  * (va-minv) / (maxv-minv))
    return f"\\cellcolor{{red!{color}}} {v:.2f}"

METHOD_CORR["Random"] = {
    k: {"pearson": 0.0, "spearman": 0.0}
    for k in METHOD_CORR["Random"].keys()
}

KEYX = [
    "length (characters)",
    "grammar errors per word",
    "output diversity (embd)",
    "output diversity (chrF)",
    "% output unique",
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