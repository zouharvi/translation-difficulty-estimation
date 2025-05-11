# %%

from pathlib import Path
import difficulty_sampling
import difficulty_sampling.evaluate
import difficulty_sampling.utils
import difficulty_sampling.data
import subsampling.sentinel
import subsampling.syntactic_complexity
import subsampling.negative_word_frequency
import subsampling.misc


data = difficulty_sampling.data.Data.load(
    dataset_name="wmt24", lps=["en-x"], domains="all", protocol="esa"
)

# apply scorers to the whole data
subsampling.sentinel.sentinel_src_metric_model_score(
    subsampling.sentinel.get_sentinel_src_metric_model("Prosho/sentinel-src-mqm-wmt1923"),
    scorer_name="sentinel-src-mqm-wmt1923",
    data=data,
    use_tgt_lang_token=True,
)
subsampling.misc.apply_subset2evaluate(data, method="random")
subsampling.syntactic_complexity.syntactic_complexity_score(
    data, "syntactic_complexity"
)
subsampling.misc.apply_external_artificial_crowd_metrics(
    data,
    sys2translations_path=Path(
        "../data/artificial_crowd/scored_translations/sys2translations.pickle"
    ),
    metric="MetricX-24-Hybrid-QE-XXL", 
)
subsampling.misc.apply_llm_as_a_judge(
    data,
    scored_source_texts_df_path=Path(
        "../data/LLM-as-a-Judge/new/command-a/wmt_data_with_source_based_num_scores.csv"
    ),
    llm_name="Command-A_new",
)
subsampling.misc.apply_llm_as_a_judge(
    data,
    scored_source_texts_df_path=Path(
        "../data/LLM-as-a-Judge/new/gpt-4o/gpt-4o-1120_source_based_num_scores.csv"
    ),
    llm_name="GPT-4o",
)


# %%
import matplotlib.pyplot as plt
import difficulty_sampling.utils
import importlib
importlib.reload(difficulty_sampling.utils)

difficulty_sampling.utils.matplotlib_default()

METHOD_TO_NAME = {
    "random": "Random",
    "LLM-as-a-Judge (Command-A_new, src-based)": "LLM-as-a-Judge",
    "syntactic_complexity": "Syntactic Complexity",
    "ext_artcrowd|MetricX-24-Hybrid-QE-XXL": "Artificial Crowd",
    "sentinel-src-mqm-wmt1923": "Sentinel",
    "human": "Oracle",
}
METHOD_TO_COLOR = {
    "random": "black",
    "human": difficulty_sampling.utils.COLORS[0],
    "syntactic_complexity": difficulty_sampling.utils.COLORS[3],
    "sentinel-src-mqm-wmt1923": difficulty_sampling.utils.COLORS[2],
    "ext_artcrowd|MetricX-24-Hybrid-QE-XXL": difficulty_sampling.utils.COLORS[4],
    "LLM-as-a-Judge (Command-A_new, src-based)": difficulty_sampling.utils.COLORS[1],
}

fig, axs = plt.subplots(ncols=2, figsize=(7.5, 2), sharex=True)

data_x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
for method, method_name in METHOD_TO_NAME.items():
    data_y = []
    for p in data_x:
        data_y.append(
            difficulty_sampling.evaluate.main_eval_avg(
                method,
                data=data,
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
        [y.avg_easy for y in data_y],
        label=method_name,
        color=METHOD_TO_COLOR[method],
        linewidth=2,
    )

axs[0].set_ylabel("Average Score")
axs[1].set_ylabel("% easy")

for ax in axs.flatten():
    ax.set_xticks(data_x[::2])
    ax.set_xticklabels([f"{int(p*100)}%" for p in data_x[::2]])

    ax.spines[["top", "right"]].set_visible(False)


axs[0].set_xlabel("Proportion of original data")
axs[1].set_xlabel("Proportion of original data")
axs[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x*100)}%"))

handles = axs[0].get_legend_handles_labels()

for method, coords0, coords1 in [
    ("human", (0.2, 0.23), (0.2, 0.12)),
    ("sentinel-src-mqm-wmt1923", (0.04, 0.52), (0.04, 0.29)),
    ("ext_artcrowd|MetricX-24-Hybrid-QE-XXL", (0.04, 0.76), (0.04, 0.66)),
]:
    axs[0].text(
        coords0[0], coords0[1],
        METHOD_TO_NAME[method].replace("Artificial", "Art."),
        fontsize=9,
        color=METHOD_TO_COLOR[method],
        transform=axs[0].transAxes,
    )
    axs[1].text(
        coords1[0], coords1[1],
        METHOD_TO_NAME[method].replace("Artificial", "Art."),
        fontsize=9,
        color=METHOD_TO_COLOR[method],
        transform=axs[1].transAxes,
    )


plt.tight_layout(pad=0)
# negative spacing between subplots
plt.subplots_adjust(wspace=0.25)
plt.savefig("../generated/04-variable_budget.pdf")
plt.show()

# %%

# plot just the legend
plt.figure(figsize=(7.5, 0.4))
plt.legend(
    *handles,
    loc="center",
    fontsize=9,
    ncol=6,
    frameon=False,
    handlelength=1,
    handletextpad=0.5,
)
plt.axis("off")
plt.savefig("../generated/04-variable_budget_legend.pdf")
plt.tight_layout()