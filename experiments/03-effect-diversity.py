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

def symmetric_chrf(tgt1, tgt2):
    out = pairwise_chrf([[tgt1], [tgt2]], [[tgt2], [tgt1]])
    return (out[0][0][0]+out[1][0][0])/2

# compute some protected attributes
for data in tqdm.tqdm(list(data_all.lp2src_data_list.values())):
    for line in data:
        output_unique = len({tgt for tgt in line["tgt"].values()})/len(line["tgt"])
        output_diversity_ip = np.average([
            # cosine similarity
            sentence_transformers.util.pytorch_cos_sim(
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
            "length": len(line["src"]),
        }

# %%

subsampling.sentinel.sentinel_src_metric_model_score(
    subsampling.sentinel.get_sentinel_src_metric_model("sapienzanlp/sentinel-src-mqm"),
    scorer_name="sentinel-src-mqm",
    data=data_all,
    use_tgt_lang_token=False,
)

subsampling.sentinel.sentinel_src_metric_model_score(
    subsampling.sentinel.get_sentinel_src_metric_model(
        "Prosho/sentinel-src-mqm-tgt-lang"
    ),
    scorer_name="sentinel-src-mqm-tgt-lang",
    data=data_all,
    use_tgt_lang_token=True,
)
subsampling.misc.apply_subset2evaluate(data_all, method="random")
subsampling.misc.apply_src_len(data_all)
subsampling.misc.apply_src_len(data_all)
subsampling.syntactic_complexity.syntactic_complexity_score(
    data_all, "syntactic_complexity"
)
subsampling.negative_word_frequency.negative_word_frequency_score(
    data_all, "negative_word_frequency"
)
subsampling.negative_word_frequency.negative_word_frequency_score(
    data_all, "negative_word_zipf_frequency"
)
subsampling.misc.apply_subset2evaluate_cache(data_all, method="precomet_avg")
subsampling.misc.apply_subset2evaluate_cache(data_all, method="precomet_diff")
subsampling.misc.apply_subset2evaluate_cache(data_all, method="precomet_var")
subsampling.misc.apply_subset2evaluate_cache(data_all, method="precomet_diversity")

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


def plot_problem(ax, data_x, data_y, key_x, key_y):
    ax.scatter(
        data_x, data_y,
        s=10,
        color="#722",
        alpha=0.5,
        linewidth=0,
    )
    ax.set_xlabel(key_x)
    ax.set_ylabel(key_y)
    ax.spines[["top", "right"]].set_visible(False)

    corr_pearson = scipy.stats.pearsonr(data_x, data_y)[0]
    corr_spearman = scipy.stats.spearmanr(data_x, data_y)[0]
    ax.text(
        0.95, 0.05,
        f"ρ={corr_pearson:.2f} (Pearson)\nρ={corr_spearman:.2f} (Spearman)",
        transform=ax.transAxes,
        fontsize=8,
        ha="right",
        va="bottom",
    )

def plot_method(method_name):
    fig, axs = plt.subplots(1, 4, figsize=(12, 3.5))

    plot_problem(
        axs[0],
        avg_effect_across_langs("length"),
        avg_difficulty_across_langs(method_name),
        key_x="length (characters)",
        key_y=method_name,
    )

    plot_problem(
        axs[1],
        avg_effect_across_langs("output_unique"),
        avg_difficulty_across_langs(method_name),
        key_x="% output unique",
        key_y=method_name,
    )

    plot_problem(
        axs[2],
        avg_effect_across_langs("output_diversity_ip"),
        avg_difficulty_across_langs(method_name),
        key_x="output diversity (embd)",
        key_y=method_name,
    )

    plot_problem(
        axs[3],
        avg_effect_across_langs("output_diversity_chrf"),
        avg_difficulty_across_langs(method_name),
        key_x="output diversity (chrf)",
        key_y=method_name,
    )

    plt.tight_layout()
    plt.show()

plot_method("sentinel-src-mqm")
plot_method("sentinel-src-mqm-tgt-lang")
plot_method("syntactic_complexity")
plot_method("negative_word_frequency")
plot_method("negative_word_zipf_frequency")
plot_method("precomet_avg")
plot_method("precomet_var")
plot_method("precomet_diff")
plot_method("precomet_diversity")
plot_method("src_len")
plot_method("random")