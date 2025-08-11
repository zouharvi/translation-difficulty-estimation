# %%
import json
import sentence_transformers.util
import difficulty_estimation
import difficulty_estimation.evaluate
import difficulty_estimation.utils
import difficulty_estimation.data
import subsampling.sentinel
import subsampling.syntactic_complexity
import subsampling.average_word_frequency
import subsampling.misc
import sentence_transformers
import collections
import matplotlib.pyplot as plt
import numpy as np
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
# embedd all sources
# map tgt to embedding
src2embd = list(
    {line["src"] for data in data_all.lp2src_data_list.values() for line in data}
)
src2embd = dict(
    zip(
        src2embd,
        sentence_transformers.SentenceTransformer(
            "paraphrase-multilingual-MiniLM-L12-v2"
        ).encode(src2embd),
    )
)

# %%

data_x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def measure_avg_sim(method_name, p: float):
    mean_dists = []
    for data in data_all.lp2src_data_list.values():
        # take top-B from data based on data_y
        B = int(len(data) * p)
        data_y = [
            np.average([x["scores"][sys][method_name] for sys in x["scores"].keys()])
            for x in data
        ]
        data_local = [
            x[0] for x in sorted(list(zip(data, data_y)), key=lambda x: x[1])[:B]
        ]

        # measure average point distance based on embedding
        data_embd = [src2embd[line["src"]] for line in data_local]
        data_embd = np.array(data_embd)
        # measure distance
        mean_dist = sentence_transformers.util.cos_sim(data_embd, data_embd).mean()
        mean_dists.append(mean_dist)
    return np.average(mean_dists)


METHOD_TO_NAME = {
    "random": "Random",
    "syntactic_complexity": "Syntax Complexity",
    "sentinel-src-mqm-wmt1723": "Sentinel",
    "src_len": "Length",
}


def measure_closest_sim(method_name, p: float):
    closest_dists = []
    for data in data_all.lp2src_data_list.values():
        # take top-B from data based on data_y
        B = int(len(data) * p)
        data_y = [
            np.average([x["scores"][sys][method_name] for sys in x["scores"].keys()])
            for x in data
        ]
        data_local = [
            x[0] for x in sorted(list(zip(data, data_y)), key=lambda x: x[1])[:B]
        ]

        # measure average point distance based on embedding
        data_embd = [src2embd[line["src"]] for line in data_local]
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
for p in data_x:
    for method in METHOD_TO_NAME.keys():
        results_avg[method].append(measure_avg_sim(method, p))
        results_closest[method].append(measure_closest_sim(method, p))

for i in range(10):
    subsampling.misc.apply_subset2evaluate(data_all, method="random")
    for p in data_x:
        results_avg_random[i].append(measure_avg_sim("random", p))
        results_closest_random[i].append(measure_closest_sim("random", p))


# %%

METHOD_TO_COLOR = {
    "random": "black",
    "src_len": "#a0a010",
    "syntactic_complexity": difficulty_estimation.utils.COLORS[3],
    "sentinel-src-mqm-wmt1723": difficulty_estimation.utils.COLORS[2],
}

difficulty_estimation.utils.matplotlib_default()

fig, axs = plt.subplots(1, 2, figsize=(7.5, 2.5))

# plot closest
data_y_rand_closets_interval = [
    difficulty_estimation.utils.confidence_interval(l, confidence=0.99)
    for l in np.array(list(results_closest_random.values())).T
]
axs[0].fill_between(
    data_x,
    [x[0] for x in data_y_rand_closets_interval],
    [x[1] for x in data_y_rand_closets_interval],
    color=METHOD_TO_COLOR["random"],
    linewidth=0,
    alpha=0.4,
    zorder=-10,
    label="Random",
)
for method in METHOD_TO_COLOR.keys():
    results_v = results_closest[method]
    if method == "random":
        continue
    axs[0].plot(
        data_x,
        results_v,
        label=METHOD_TO_NAME[method],
        color=METHOD_TO_COLOR[method],
        linewidth=2,
    )

# plot avg
data_y_rand_avg_interval = [
    difficulty_estimation.utils.confidence_interval(l, confidence=0.99)
    for l in np.array(list(results_avg_random.values())).T
]
axs[1].fill_between(
    data_x,
    [x[0] for x in data_y_rand_avg_interval],
    [x[1] for x in data_y_rand_avg_interval],
    color=METHOD_TO_COLOR["random"],
    linewidth=0,
    alpha=0.4,
    zorder=-10,
)
for method in METHOD_TO_COLOR.keys():
    results_v = results_avg[method]
    if method == "random":
        continue
    axs[1].plot(
        data_x,
        results_v,
        label=METHOD_TO_NAME[method],
        color=METHOD_TO_COLOR[method],
        linewidth=2,
    )

# configure decorations
axs[0].spines[["top", "right"]].set_visible(False)
axs[1].spines[["top", "right"]].set_visible(False)

axs[0].set_xticks(data_x[::2])
axs[0].set_xticklabels([f"{int(p*100)}%" for p in data_x[::2]])
axs[1].set_xticks(data_x[::2])
axs[1].set_xticklabels([f"{int(p*100)}%" for p in data_x[::2]])

# set formatter to three decimal places
axs[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.3f}"))
axs[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.3f}"))

axs[0].set_ylabel("Closest Neighbour\nCosine Similarity")
axs[1].set_ylabel("Average Pairwise\nCosine Similarity")
axs[0].set_xlabel("Proportion of original data")
axs[1].set_xlabel("Proportion of original data")
axs[0].set_ylim(0.434, None)

handles, handles_txt = plt.gca().get_legend_handles_labels()

plt.tight_layout(pad=0)
# negative spacing between subplots
plt.subplots_adjust(wspace=0.25)
plt.savefig("../generated/05-post_effect_src_diversity_wmt25.pdf")
plt.show()

# %%

# fix the line for Random
import copy

handle_random = copy.deepcopy(handles[0])
handle_random.set_color(METHOD_TO_COLOR["random"])

# plot just the legend
plt.figure(figsize=(7.5, 0.4))
plt.legend(
    [handle_random] + handles,
    ["Random"] + handles_txt,
    loc="center",
    fontsize=9,
    ncol=7,
    frameon=False,
    handlelength=0.7,
    handletextpad=0.5,
    columnspacing=0.9,
)
plt.axis("off")
plt.savefig("../generated/05-post_effect_src_diversity_wmt25_length.pdf")
plt.tight_layout()
