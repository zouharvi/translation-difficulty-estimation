"""
TODO: resolve the structure, things are a bit messy with subsampling not being in difficulty_sampling package
"""

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
    subsampling.sentinel.get_sentinel_src_metric_model("sapienzanlp/sentinel-src-mqm"),
    scorer_name="sentinel-src-mqm",
    data=data,
    use_tgt_lang_token=False,
)
subsampling.sentinel.sentinel_src_metric_model_score(
    subsampling.sentinel.get_sentinel_src_metric_model(
        "Prosho/sentinel-src-mqm-tgt-lang"
    ),
    scorer_name="sentinel-src-mqm-tgt-lang",
    data=data,
    use_tgt_lang_token=True,
)
subsampling.sentinel.sentinel_src_metric_model_score(
    subsampling.sentinel.get_sentinel_src_metric_model(
        "Prosho/sentinel-src-mqm-lin-tgt-lang"
    ),
    scorer_name="sentinel-src-mqm-lin-tgt-lang",
    data=data,
    use_tgt_lang_token=True,
)
subsampling.sentinel.sentinel_src_metric_model_score(
    subsampling.sentinel.get_sentinel_src_metric_model(
        "Prosho/sentinel-src-mqm-znorm-per-rater-tgt-lang"
    ),
    scorer_name="sentinel-src-mqm-znorm-per-rater-tgt-lang",
    data=data,
    use_tgt_lang_token=True,
)
subsampling.sentinel.sentinel_src_metric_model_score(
    subsampling.sentinel.get_sentinel_src_metric_model("sapienzanlp/sentinel-src-da"),
    scorer_name="sentinel-src-da",
    data=data,
    use_tgt_lang_token=False,
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
subsampling.misc.apply_subset2evaluate_cache(data, method="precomet_avg")
subsampling.misc.apply_subset2evaluate_cache(data, method="precomet_var")
subsampling.misc.apply_subset2evaluate_cache(data, method="precomet_diff")
subsampling.misc.apply_subset2evaluate_cache(data, method="precomet_diversity")
subsampling.misc.apply_artificial_crowd_metrics(data, model="GPT-4", metric="XCOMET")
subsampling.misc.apply_artificial_crowd_metrics(data, model="GPT-4", metric="human")
subsampling.misc.apply_external_artificial_crowd_metrics(
    data,
    sys2translations_path=Path(
        "../data/artificial_crowd/scored_translations/sys2translations.pickle"
    ),
    metric="MetricX-24-Hybrid-XXL",
)
subsampling.misc.apply_external_artificial_crowd_metrics(
    data,
    sys2translations_path=Path(
        "../data/artificial_crowd/scored_translations/sys2translations.pickle"
    ),
    metric="XCOMET-XXL",
)
subsampling.misc.apply_external_artificial_crowd_metrics(
    data,
    sys2translations_path=Path(
        "../data/artificial_crowd/scored_translations/sys2translations.pickle"
    ),
    metric="CometKiwi-XXL",
)
subsampling.misc.apply_llm_as_a_judge(
    data,
    scored_source_texts_df_path=Path(
        "../data/LLM-as-a-Judge/old/wmt_data_with_source_based_num_scores.csv"
    ),
    llm_name="Command-A_old",
)
subsampling.misc.apply_llm_as_a_judge(
    data,
    scored_source_texts_df_path=Path(
        "../data/LLM-as-a-Judge/old/wmt_data_with_target_based_num_scores.csv"
    ),
    llm_name="Command-A_old",
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
        "../data/LLM-as-a-Judge/new/command-a/wmt_data_with_target_based_num_scores.csv"
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
subsampling.misc.apply_llm_as_a_judge(
    data,
    scored_source_texts_df_path=Path(
        "../data/LLM-as-a-Judge/new/gpt-4o/gpt-4o-1120_target_based_num_scores.csv"
    ),
    llm_name="GPT-4o",
)


# %%
METHOD_TO_NAME = {
    "random": "Random",
    "human": "Oracle",
    "src_len": "Source Length",
    "syntactic_complexity": "Syntactic Complexity",
    "negative_word_frequency": "Negative Word Frequency",
    "negative_word_zipf_frequency": "Negative Word Zipf Frequency",
    "precomet_avg": "PreCOMET Average",
    "precomet_var": "PreCOMET Variance",
    "precomet_diff": "PreCOMET Difficulty",
    "precomet_diversity": "PreCOMET Diversity",
    "sentinel-src-mqm": "Sentinel-MQM",
    "sentinel-src-mqm-tgt-lang": "Sentinel-MQM-tgt",
    "sentinel-src-mqm-lin-tgt-lang": "Sentinel-MQM-lin-tgt",
    "sentinel-src-mqm-znorm-per-rater-tgt-lang": "Sentinel-MQM-znorm-tgt",
    "sentinel-src-da": "Sentinel-DA",
    "artcrowd|GPT-4|human": "Artificial Crowd (Oracle)",
    "artcrowd|GPT-4|XCOMET": "Artificial Crowd (XCOMET)",
    "ext_artcrowd|MetricX-24-Hybrid-XXL": "External Artificial Crowd (MetricX-24-Hybrid-XXL)",
    "ext_artcrowd|XCOMET-XXL": "External Artificial Crowd (XCOMET-XXL)",
    "ext_artcrowd|CometKiwi-XXL": "External Artificial Crowd (CometKiwi-XXL)",
}

with open(difficulty_sampling.ROOT / "generated/01-eval_all.tex", "w") as f:

    def eval_print_table(method_name, B=100):
        results = difficulty_sampling.evaluate.main_eval_avg(
            method_name, data=data, B=B
        )
        method_name = METHOD_TO_NAME.get(
            method_name, method_name.replace("_", " ").title()
        )
        print(
            f"{method_name:>20}",
            f"{results.avg_score:.1f}",
            f"{results.avg_score_z:.2f}",
            f"{results.diff_corr:.3f}",
            f"{results.avg_perfect:.1%}".replace("%", "\\%"),
            sep=" & ",
            end=" \\\\\n",
            file=f,
        )

    for method_name in [
        "random",
        "human",
        "src_len",
        "syntactic_complexity",
        "negative_word_frequency",
        "negative_word_zipf_frequency",
        "precomet_avg",
        "precomet_var",
        "precomet_diff",
        "precomet_diversity",
        "sentinel-src-da",
        "sentinel-src-mqm",
        "sentinel-src-mqm-tgt-lang",
        "sentinel-src-mqm-lin-tgt-lang",
        "sentinel-src-mqm-znorm-per-rater-tgt-lang",
        "artcrowd|GPT-4|human",
        "artcrowd|GPT-4|XCOMET",
        "ext_artcrowd|MetricX-24-Hybrid-XXL",
        "ext_artcrowd|XCOMET-XXL",
        "ext_artcrowd|CometKiwi-XXL",
        "LLM-as-a-Judge (Command-A_old, src-based)",
        "LLM-as-a-Judge (Command-A_old, tgt-based)",
        "LLM-as-a-Judge (Command-A_new, src-based)",
        "LLM-as-a-Judge (Command-A_new, tgt-based)",
        "LLM-as-a-Judge (GPT-4o, src-based)",
        "LLM-as-a-Judge (GPT-4o, tgt-based)",
    ]:
        eval_print_table(method_name)
