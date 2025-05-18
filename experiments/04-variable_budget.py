# %%

from pathlib import Path
import numpy as np
import difficulty_sampling
import difficulty_sampling.evaluate
import difficulty_sampling.utils
import difficulty_sampling.data
import matplotlib.pyplot as plt
import subsampling.sentinel
import subsampling.syntactic_complexity
import subsampling.average_word_frequency
import subsampling.misc

data_all = difficulty_sampling.data.Data.load(
    dataset_name="wmt24", lps=["all"], domains="all", protocol="esa"
)

# apply scorers to the whole data
subsampling.sentinel.sentinel_src_metric_model_score(
    subsampling.sentinel.get_sentinel_src_metric_model(
        "Prosho/sentinel-src-mqm-wmt1723"
    ),
    scorer_name="sentinel-src-mqm-wmt1723",
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
        "../data/external_artificial_crowd/sys2translations.pickle"
    ),
    metric="XCOMET-QE-XXL",
)
subsampling.misc.apply_llm_as_a_judge(
    data_all,
    scored_source_texts_df_path=Path(
        "../data/LLM-as-a-Judge/esa/command-a/command-a-03-2025_target_based_num_scores.csv"
    ),
    llm_name="Command-A",
)
subsampling.misc.apply_oracle_with_fixed_scores(
    data_all, scorer_name="oracle-src", use_tgt_lang=False
)
subsampling.misc.apply_oracle_with_fixed_scores(
    data_all, scorer_name="oracle-tgt", use_tgt_lang=True
)

# %%

data_x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# plot random spread
data_y_rand_all = []
for seed in range(10):
    subsampling.misc.apply_subset2evaluate(data_all, method="random", seed=seed)
    data_y = []
    for p in data_x:
        out = difficulty_sampling.evaluate.main_eval_avg(
            "random",
            data=data_all,
            proportion=p,
        )
        data_y.append((out.avg_score, out.avg_perfect))
    data_y_rand_all.append(data_y)

data_y_rand_all = np.array(data_y_rand_all)

# %%


difficulty_sampling.utils.matplotlib_default()

METHOD_TO_NAME = {
    "random": "Random",
    "LLM-as-a-Judge (Command-A)": "LLM-as-a-Judge",
    "syntactic_complexity": "Syntax Complexity",
    "ext_artcrowd|XCOMET-QE-XXL": "Artificial Crowd",
    "sentinel-src-mqm-wmt1723": "Sentinel",
    "oracle-src": "Oracle-src",
    "oracle-tgt": "Oracle-tgt",
}
METHOD_TO_COLOR = {
    "random": "black",
    "oracle-src": difficulty_sampling.utils.COLORS[0],
    "oracle-tgt": "#600000",
    "syntactic_complexity": difficulty_sampling.utils.COLORS[3],
    "sentinel-src-mqm-wmt1723": difficulty_sampling.utils.COLORS[2],
    "ext_artcrowd|XCOMET-QE-XXL": difficulty_sampling.utils.COLORS[4],
    "LLM-as-a-Judge (Command-A)": difficulty_sampling.utils.COLORS[1],
}

fig, axs = plt.subplots(ncols=2, figsize=(7.5, 2.5), sharex=True)

for method, method_name in METHOD_TO_NAME.items():
    if method == "random":
        continue
    data_y = []
    for p in data_x:
        data_y.append(
            difficulty_sampling.evaluate.main_eval_avg(
                method,
                data=data_all,
                proportion=p,
            )
        )

    axs[0].plot(
        data_x,
        [y.avg_score for y in data_y],
        label=method_name,
        color=METHOD_TO_COLOR[method],
        linewidth=2,
    )
    axs[1].plot(
        data_x,
        [y.avg_perfect for y in data_y],
        label=method_name,
        color=METHOD_TO_COLOR[method],
        linewidth=2,
    )


data_y_rand_score = data_y_rand_all[:, :, 0].T
data_y_rand_perfe = data_y_rand_all[:, :, 1].T
data_y_rand_score_interval = [
    difficulty_sampling.utils.confidence_interval(l, confidence=0.99)
    for l in data_y_rand_score
]
data_y_rand_perfe_interval = [
    difficulty_sampling.utils.confidence_interval(l, confidence=0.99)
    for l in data_y_rand_perfe
]

axs[0].fill_between(
    data_x,
    [i[0] for i in data_y_rand_score_interval],
    [i[1] for i in data_y_rand_score_interval],
    color=METHOD_TO_COLOR["random"],
    linewidth=0,
    alpha=0.4,
    zorder=-10,
)
axs[1].fill_between(
    data_x,
    [i[0] for i in data_y_rand_perfe_interval],
    [i[1] for i in data_y_rand_perfe_interval],
    color=METHOD_TO_COLOR["random"],
    linewidth=0,
    alpha=0.4,
    zorder=-10,
)


axs[0].set_ylabel("Average Score")
axs[1].set_ylabel("%Perfect")

for ax in axs.flatten():
    ax.set_xticks(data_x[::2])
    ax.set_xticklabels([f"{int(p*100)}%" for p in data_x[::2]])

    ax.spines[["top", "right"]].set_visible(False)


axs[0].set_xlabel("Proportion of original data")
axs[1].set_xlabel("Proportion of original data")
axs[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x*100)}%"))

handles, handles_txt = axs[0].get_legend_handles_labels()

for method, coords0, coords1 in [
    ("oracle-src", (0.04, 0.35, 35), (0.15, 0.29, 20)),
    ("oracle-tgt", (0.07, 0.05, 53), (0.15, 0.05, 20)),
    ("sentinel-src-mqm-wmt1723", (0.04, 0.63, 10), (0.04, 0.10, 25)),
    ("ext_artcrowd|XCOMET-QE-XXL", None, None),
    ("syntactic_complexity", None, None),
    ("LLM-as-a-Judge (Command-A)", None, None),
    ("random", None, (0.05, 0.75, 0)),
]:
    if coords0 is not None:
        axs[0].text(
            coords0[0],
            coords0[1],
            METHOD_TO_NAME[method].replace("Artificial", "Art."),
            fontsize=8,
            color=METHOD_TO_COLOR[method],
            transform=axs[0].transAxes,
            rotation=coords0[2],
        )
    if coords1 is not None:
        axs[1].text(
            coords1[0],
            coords1[1],
            METHOD_TO_NAME[method].replace("Artificial", "Art."),
            fontsize=8,
            color=METHOD_TO_COLOR[method],
            transform=axs[1].transAxes,
            rotation=coords1[2],
        )


plt.tight_layout(pad=0)
# negative spacing between subplots
plt.subplots_adjust(wspace=0.25)
plt.savefig("../generated/04-variable_budget.pdf")
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
plt.savefig("../generated/04-variable_budget_legend.pdf")
plt.tight_layout()
