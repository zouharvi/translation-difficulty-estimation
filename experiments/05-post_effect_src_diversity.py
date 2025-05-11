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

data_all = difficulty_sampling.data.Data.load(
    dataset_name="wmt24", lps=["en-x"], domains="all", protocol="esa"
)

# %%

subsampling.misc.apply_subset2evaluate(data_all, method="random")
subsampling.misc.apply_src_len(data_all)
subsampling.negative_word_frequency.negative_word_frequency_score(
    data_all, "negative_word_frequency"
)
subsampling.sentinel.sentinel_src_metric_model_score(
    subsampling.sentinel.get_sentinel_src_metric_model(
        "Prosho/sentinel-src-mqm-tgt-lang"
    ),
    scorer_name="sentinel-src-mqm-tgt-lang",
    data=data_all,
    use_tgt_lang_token=True,
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
    for method in [
        "sentinel-src-mqm-tgt-lang",
        "negative_word_frequency",
        "src_len",
        "human",
    ]:
        results_avg[method].append(measure_avg_sim(method, p))
        results_closest[method].append(measure_closest_sim(method, p))

for i in range(10):
    subsampling.misc.apply_subset2evaluate(data_all, method="random")
    for p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        results_avg_random[i].append(measure_avg_sim("random", p))
        results_closest_random[i].append(measure_closest_sim("random", p))



# %%

METHOD_TO_NAME = {
    "human": "Oracle",
    "src_len": "Source Length",
    "syntactic_complexity": "Syntactic Complexity",
    "negative_word_frequency": "Negative Word Frequency",
    "sentinel-src-mqm-tgt-lang": "Sentinel-MQM-TGT",
}

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
plt.xlabel("Subset size")
plt.legend(frameon=False)
plt.show()


# %%


for i, results_v in results_closest_random.items():
    plt.plot(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        results_v,
        color="black",
        alpha=0.5,
        label="Random" if i == 0 else None,
    )

for method, results_v in results_closest.items():
    plt.plot([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], results_v, label=METHOD_TO_NAME[method])

# turn off spines
plt.gca().spines[["top", "right"]].set_visible(False)

plt.ylabel("Closest Neighbour Cosine Similarity")
plt.xlabel("Subset size")
plt.legend(frameon=False)
plt.show()