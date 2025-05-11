# %%
import sentence_transformers.util
import difficulty_sampling
import difficulty_sampling.evaluate
import difficulty_sampling.utils
import difficulty_sampling.data
import subsampling.sentinel
import subsampling.syntactic_complexity
import subsampling.negative_word_frequency
import subsampling.misc
import sentence_transformers
import collections
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

data_all = difficulty_sampling.data.Data.load(
    dataset_name="wmt24", lps=["en-x"], domains="all", protocol="esa"
)

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
# embedd all sources
# map tgt to embedding
src2embd = list({
    line["src"]
    for data in data_all.lp2src_data_list.values()
    for line in data
})
src2embd = dict(zip(src2embd, sentence_transformers.SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2").encode(src2embd)))

# %%

def measure_avg_sim(method_name, p: float):
    mean_dists = []
    for data in data_all.lp2src_data_list.values():
        # take top-B from data based on data_y
        B = int(len(data) * p)
        data_y = [np.average([x["scores"][sys][method_name] for sys in x["scores"].keys()]) for x in data]
        data_local = [x[0] for x in sorted(list(zip(data, data_y)), key=lambda x: x[1])[:B]]

        # measure average point distance based on embedding
        data_embd = [
            src2embd[line["src"]]
            for line in data_local
        ]
        data_embd = np.array(data_embd)
        # measure distance
        mean_dist = sentence_transformers.util.cos_sim(data_embd, data_embd).mean()
        mean_dists.append(mean_dist)
    return np.average(mean_dists)



METHOD_TO_NAME = {
    "random": "Random",
    "LLM-as-a-Judge (Command-A_new, src-based)": "LLM-as-a-Judge",
    "syntactic_complexity": "Syntax Complexity",
    "ext_artcrowd|MetricX-24-Hybrid-QE-XXL": "Artificial Crowd",
    "sentinel-src-mqm-wmt1923": "Sentinel",
    "human": "Oracle",
}

def measure_closest_sim(method_name, p: float):
    closest_dists = []
    for data in data_all.lp2src_data_list.values():
        # take top-B from data based on data_y
        B = int(len(data) * p)
        data_y = [np.average([x["scores"][sys][method_name] for sys in x["scores"].keys()]) for x in data]
        data_local = [x[0] for x in sorted(list(zip(data, data_y)), key=lambda x: x[1])[:B]]

        # measure average point distance based on embedding
        data_embd = [
            src2embd[line["src"]]
            for line in data_local
        ]
        data_embd = np.array(data_embd)
        # measure distance
        dists = sentence_transformers.util.cos_sim(data_embd, data_embd)
        # take closest point for each point
        dists = dists - np.eye(dists.shape[0])
        dists = np.max(dists.numpy(), axis=1)
        closest_dists.append(dists.mean())
    return np.average(closest_dists)

results_avg = collections.defaultdict(list)
results_avg_random = collections.defaultdict(list)
results_closest = collections.defaultdict(list)
results_closest_random = collections.defaultdict(list)
for p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    for method in METHOD_TO_NAME.keys():
        results_avg[method].append(measure_avg_sim(method, p))
        results_closest[method].append(measure_closest_sim(method, p))

for i in range(10):
    subsampling.misc.apply_subset2evaluate(data_all, method="random")
    for p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        results_avg_random[i].append(measure_avg_sim("random", p))
        results_closest_random[i].append(measure_closest_sim("random", p))




# %%


METHOD_TO_COLOR = {
    "random": "black",
    "human": difficulty_sampling.utils.COLORS[0],
    "syntactic_complexity": difficulty_sampling.utils.COLORS[3],
    "LLM-as-a-Judge (Command-A_new, src-based)": difficulty_sampling.utils.COLORS[1],
    "ext_artcrowd|MetricX-24-Hybrid-QE-XXL": difficulty_sampling.utils.COLORS[4],
    "sentinel-src-mqm-wmt1923": difficulty_sampling.utils.COLORS[2],
}

import difficulty_sampling.utils

difficulty_sampling.utils.matplotlib_default()

plt.figure(figsize=(3.5, 2))


for i, results_v in results_closest_random.items():
    plt.plot(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        results_v,
        color="black",
        alpha=0.2,
        label="Random" if i == 0 else None,
    )

for method in METHOD_TO_COLOR.keys():
    results_v = results_closest[method]
    if method == "random":
        continue
    plt.plot(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        results_v,
        label=METHOD_TO_NAME[method],
        color=METHOD_TO_COLOR[method],
        linewidth=2,
    )

# turn off spines
plt.gca().spines[["top", "right"]].set_visible(False)

# xticks formatter percentage
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x*100)}%"))

plt.ylabel("Closest Neighbour\nCosine Similarity")
plt.xlabel("Proportion of original data")
plt.ylim(0.434, None)

handles = plt.gca().get_legend_handles_labels()

plt.tight_layout(pad=0)
plt.savefig("../generated/05-post_effect_src_diversity.pdf")
plt.show()

# %%
# plot just the legend

# fix the line for Random
handles[0][0].set_alpha(1)

plt.figure(figsize=(3.5, 0.4))
plt.legend(
    *handles,
    loc="center",
    fontsize=9,
    ncol=3,
    frameon=False,
    handlelength=0.7,
    handletextpad=0.5,
    columnspacing=0.5,
)
plt.axis("off")
plt.tight_layout(pad=0)
plt.savefig("../generated/05-post_effect_src_diversity_legend.pdf")
plt.show()

# %%

for i, results_v in results_avg_random.items():
    plt.plot(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        results_v,
        color="black",
        alpha=0.5,
        label="Random" if i == 0 else None,
    )

for method, results_v in results_avg.items():
    plt.plot([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], results_v, label=METHOD_TO_NAME[method])

# turn off spines
plt.gca().spines[["top", "right"]].set_visible(False)

plt.ylabel("Average Source Cosine Similarity")
plt.xlabel("Proportion of original data")
plt.legend(frameon=False)
plt.show()
