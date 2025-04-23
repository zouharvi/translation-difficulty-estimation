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
    subsampling.sentinel.get_sentinel_src_metric_model(
        "Prosho/sentinel-src-mqm-tgt-lang"
    ),
    scorer_name="sentinel-src-mqm-tgt-lang",
    data=data,
    use_tgt_lang_token=True,
)
subsampling.misc.apply_subset2evaluate(data, method="random")
subsampling.misc.apply_src_len(data)
subsampling.syntactic_complexity.syntactic_complexity_score(
    data, "syntactic_complexity"
)
subsampling.negative_word_frequency.negative_word_frequency_score(
    data, "negative_word_frequency"
)
subsampling.negative_word_frequency.negative_word_frequency_score(
    data, "negative_word_zipf_frequency"
)
subsampling.misc.apply_external_artificial_crowd_metrics(
    data,
    sys2translations_path=Path(
        "../data/artificial_crowd/scored_translations/sys2translations.pickle"
    ),
    metric="MetricX-24-Hybrid-XXL",
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

METHOD_TO_NAME = {
    "random": "Random",
    "human": "Oracle",
    "src_len": "Source Length",
    "syntactic_complexity": "Syntactic Complexity",
    "sentinel-src-mqm-tgt-lang": "Sentinel-MQM",
    "ext_artcrowd|MetricX-24-Hybrid-XXL": "External Artificial Crowd (MetricX-24-Hybrid-XXL)",
}

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(9, 6), sharex=True)

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

    axs[0, 0].plot(
        data_x,
        [y.avg_score for y in data_y],
        label=method_name,
    )
    axs[0, 1].plot(
        data_x,
        [y.diff_corr for y in data_y],
        label=method_name,
    )
    axs[1, 0].plot(
        data_x,
        [y.avg_diff for y in data_y],
        label=method_name,
    )
    axs[1, 1].plot(
        data_x,
        [y.avg_perfect for y in data_y],
        label=method_name,
    )

axs[0, 0].set_ylabel("Average Score")
axs[0, 1].set_ylabel("Correlation")
axs[1, 0].set_ylabel("Average Difficulty")
axs[1, 1].set_ylabel("Average Perfect Score")

# axs[0,0].legend()
for ax in axs.flatten():
    ax.set_xticks(data_x[::2])
    ax.set_xticklabels([f"{int(p*100)}%" for p in data_x[::2]])

    ax.spines[["top", "right"]].set_visible(False)


axs[1, 0].set_xlabel("Proportion of original data")
axs[1, 1].set_xlabel("Proportion of original data")


handles = axs[0, 0].get_legend_handles_labels()

plt.tight_layout()
plt.show()

# %%

# plot just the legend
plt.figure(figsize=(3, 1))
plt.legend(
    *handles,
    loc="center",
    fontsize=10,
    ncol=2,
    frameon=False,
)
plt.axis("off")
plt.tight_layout()


# with open(difficulty_sampling.ROOT / "generated/01-eval_all.tex", "w") as f:
