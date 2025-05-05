"""
TODO: resolve the structure, things are a bit messy with subsampling not being in difficulty_sampling package
"""
# %%

from pathlib import Path
from typing import Literal

import difficulty_sampling
import difficulty_sampling.evaluate
import difficulty_sampling.utils
import difficulty_sampling.data
import subsampling.sentinel
import subsampling.syntactic_complexity
import subsampling.negative_word_frequency
import subsampling.misc
from difficulty_sampling import DiffCorrTasks

SINGLE_SRC_SUBSET = False

RUN_STAT_SIGN_ON_DIFFCORR, K = True, 0

protocol: Literal["esa", "mqm"] = "esa"
lps = difficulty_sampling.wmt24_lps_esa
data = difficulty_sampling.data.Data.load(
    dataset_name="wmt24", lps=lps, domains="all", protocol=protocol
)
domains = ["news", "social", "literary", "speech"]

score_all_source_texts = True

# apply scorers to the whole data
subsampling.sentinel.sentinel_src_metric_model_score(
    subsampling.sentinel.get_sentinel_src_metric_model("sapienzanlp/sentinel-src-da"),
    scorer_name="sentinel-src-da",
    data=data,
    use_tgt_lang_token=False,
    score_all_source_texts=score_all_source_texts,
)
subsampling.sentinel.sentinel_src_metric_model_score(
    subsampling.sentinel.get_sentinel_src_metric_model("sapienzanlp/sentinel-src-mqm"),
    scorer_name="sentinel-src-mqm",
    data=data,
    use_tgt_lang_token=False,
    score_all_source_texts=score_all_source_texts,
)
subsampling.sentinel.sentinel_src_metric_model_score(
    subsampling.sentinel.get_sentinel_src_metric_model(
        "Prosho/sentinel-src-da-wmt1723"
    ),
    scorer_name="sentinel-src-da-wmt1723",
    data=data,
    use_tgt_lang_token=False,
    score_all_source_texts=score_all_source_texts,
)
subsampling.sentinel.sentinel_src_metric_model_score(
    subsampling.sentinel.get_sentinel_src_metric_model(
        "Prosho/sentinel-src-mqm-wmt1723"
    ),
    scorer_name="sentinel-src-mqm-wmt1723",
    data=data,
    use_tgt_lang_token=False,
    score_all_source_texts=score_all_source_texts,
)
subsampling.sentinel.sentinel_src_metric_model_score(
    subsampling.sentinel.get_sentinel_src_metric_model(
        "Prosho/sentinel-src-da-wmt1923"
    ),
    scorer_name="sentinel-src-da-wmt1923",
    data=data,
    use_tgt_lang_token=False,
    score_all_source_texts=score_all_source_texts,
)
subsampling.sentinel.sentinel_src_metric_model_score(
    subsampling.sentinel.get_sentinel_src_metric_model(
        "Prosho/sentinel-src-mqm-wmt1923"
    ),
    scorer_name="sentinel-src-mqm-wmt1923",
    data=data,
    use_tgt_lang_token=False,
    score_all_source_texts=score_all_source_texts,
)
subsampling.sentinel.sentinel_src_metric_model_score(
    subsampling.sentinel.get_sentinel_src_metric_model(
        "Prosho/sentinel-src-da-z-norm-per-sys"
    ),
    scorer_name="sentinel-src-da-z-norm-per-sys",
    data=data,
    use_tgt_lang_token=False,
    score_all_source_texts=score_all_source_texts,
)
subsampling.sentinel.sentinel_src_metric_model_score(
    subsampling.sentinel.get_sentinel_src_metric_model(
        "Prosho/sentinel-src-mqm-z-norm-per-sys"
    ),
    scorer_name="sentinel-src-mqm-z-norm-per-sys",
    data=data,
    use_tgt_lang_token=False,
    score_all_source_texts=score_all_source_texts,
)
"""
subsampling.misc.apply_subset2evaluate(data, method="random")
subsampling.misc.apply_src_len(data)
subsampling.syntactic_complexity.syntactic_complexity_score(
    data,
    "syntactic_complexity",
    score_all_source_texts=protocol == "mqm",
)
subsampling.negative_word_frequency.negative_word_frequency_score(
    data,
    "negative_word_frequency",
    score_all_source_texts=protocol == "mqm",
)
subsampling.negative_word_frequency.negative_word_frequency_score(
    data,
    "negative_word_zipf_frequency",
    score_all_source_texts=protocol == "mqm",
)
subsampling.misc.apply_subset2evaluate_cache(data, method="precomet_avg")
subsampling.misc.apply_subset2evaluate_cache(data, method="precomet_var")
subsampling.misc.apply_subset2evaluate_cache(data, method="precomet_diff")
subsampling.misc.apply_subset2evaluate_cache(data, method="precomet_diversity")
subsampling.misc.apply_artificial_crowd_metrics(
    data, model="GPT-4", metric="MetricX-24-Hybrid-QE"
)
subsampling.misc.apply_artificial_crowd_metrics(data, model="GPT-4", metric="XCOMET-QE")
subsampling.misc.apply_artificial_crowd_metrics(data, model="GPT-4", metric="human")
subsampling.misc.apply_external_artificial_crowd_metrics(
    data,
    sys2translations_path=Path(
        "../data/external_artificial_crowd/scored_translations/sys2translations.pickle"
    ),
    metric="MetricX-24-Hybrid-QE-XXL",
)
subsampling.misc.apply_external_artificial_crowd_metrics(
    data,
    sys2translations_path=Path(
        "../data/external_artificial_crowd/scored_translations/sys2translations.pickle"
    ),
    metric="XCOMET-QE-XXL",
)
subsampling.misc.apply_external_artificial_crowd_metrics(
    data,
    sys2translations_path=Path(
        "../data/external_artificial_crowd/scored_translations/sys2translations.pickle"
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
"""

# %%
METHOD_TO_NAME = {
    # "random": "Random",
    # "human": "Oracle",
    # "src_len": "Source Length",
    # "syntactic_complexity": "Syntactic Complexity",
    # "negative_word_frequency": "Negative Word Frequency",
    # "negative_word_zipf_frequency": "Negative Word Zipf Frequency",
    # "precomet_avg": "PreCOMET Average",
    # "precomet_var": "PreCOMET Variance",
    # "precomet_diff": "PreCOMET Difficulty",
    # "precomet_diversity": "PreCOMET Diversity",
    "sentinel-src-da": "Sentinel-DA",
    "sentinel-src-da-wmt1723": "Sentinel-DA-WMT1723",
    "sentinel-src-da-wmt1923": "Sentinel-DA-WMT1923",
    "sentinel-src-da-z-norm-per-sys": "Sentinel-DA-znorm-per-sys",
    "sentinel-src-mqm": "Sentinel-MQM",
    "sentinel-src-mqm-wmt1723": "Sentinel-MQM-WMT1723",
    "sentinel-src-mqm-wmt1923": "Sentinel-MQM-WMT1923",
    "sentinel-src-mqm-z-norm-per-sys": "Sentinel-MQM-znorm-per-sys",
    # "artcrowd|GPT-4|MetricX-24-Hybrid-QE": "Artificial Crowd (MetricX-24-Hybrid-QE-XXL)",
    # "artcrowd|GPT-4|XCOMET-QE": "Artificial Crowd (XCOMET-QE-XXL)",
    # "artcrowd|GPT-4|human": "Artificial Crowd (Oracle)",
    # "ext_artcrowd|MetricX-24-Hybrid-QE-XXL": "External Artificial Crowd (MetricX-24-Hybrid-QE-XXL)",
    # "ext_artcrowd|XCOMET-QE-XXL": "External Artificial Crowd (XCOMET-QE-XXL)",
    # "ext_artcrowd|CometKiwi-XXL": "External Artificial Crowd (CometKiwi-XXL)",
}


"""
with open(
    difficulty_sampling.ROOT
    / f"generated/{'01-gen_mt_like_eval' if SINGLE_SRC_SUBSET else '01-eval_all'}_{protocol}.tex",
    "w",
) as f:

    def eval_print_table(method_name, is_method_tgt_lang_dep, budget=100):
        results = difficulty_sampling.evaluate.main_eval_avg(
            method_name,
            data=data,
            single_src_subset=SINGLE_SRC_SUBSET,
            is_method_tgt_lang_dep=is_method_tgt_lang_dep,
            protocol=protocol,
            budget=budget,
        )
        method_name = METHOD_TO_NAME.get(method_name, method_name.replace("_", " "))
        print(
            f"{method_name:>20}{'*' if is_method_tgt_lang_dep else ''}",
            f"{results.avg_score:.1f}",
            f"{results.avg_score_z:.2f}",
            f"{results.diff_corr:.3f}",
            f"{results.avg_perfect:.1%}".replace("%", "\\%"),
            sep=" & ",
            end=" \\\\\n",
            file=f,
        )

    for method_name, is_method_tgt_lang_dep in [
        #("random", False),
        #("human", True),
        #("src_len", False),
        #("syntactic_complexity", False),
        #("negative_word_frequency", False),
        #("negative_word_zipf_frequency", False),
        #("precomet_avg", False),
        #("precomet_var", False),
        #("precomet_diff", False),
        #("precomet_diversity", False),
        ("sentinel-src-da", False),
        ("sentinel-src-da-more-data", False),
        ("sentinel-src-da-z-norm-per-sys", False),
        ("sentinel-src-da-z-norm-per-sys-no-domain", False),
        ("sentinel-src-da-beta", False),
        ("sentinel-src-da-beta-no-domain", False),
        ("sentinel-src-mqm", False),
        ("sentinel-src-mqm-z-norm-more-data", False),
        ("sentinel-src-mqm-from-da-more-data", False),
        ("sentinel-src-mqm-z-norm-per-sys", False),
        ("sentinel-src-mqm-z-norm-per-sys-no-domain", False),
        ("sentinel-src-mqm-beta", False),
        ("sentinel-src-mqm-beta-no-domain", False),
        ("sentinel-src-da-for-genmt25", False),
        ("sentinel-src-mqm-for-genmt25", False),
        # ("artcrowd|GPT-4|MetricX-24-Hybrid-QE", True),
        # ("artcrowd|GPT-4|XCOMET-QE", True),
        # ("artcrowd|GPT-4|human", True),
        # ("ext_artcrowd|MetricX-24-Hybrid-QE-XXL", True),
        # ("ext_artcrowd|XCOMET-QE-XXL", True),
        # ("ext_artcrowd|CometKiwi-XXL", True),
        # ("LLM-as-a-Judge (Command-A_old, src-based)", False),
        # ("LLM-as-a-Judge (Command-A_old, tgt-based)", True),
        # ("LLM-as-a-Judge (Command-A_new, src-based)", False),
        # ("LLM-as-a-Judge (Command-A_new, tgt-based)", True),
        # ("LLM-as-a-Judge (GPT-4o, src-based)", False),
        # ("LLM-as-a-Judge (GPT-4o, tgt-based)", True),
    ]:
        eval_print_table(method_name, is_method_tgt_lang_dep)


with open(
    difficulty_sampling.ROOT
    / f"generated/{'01-gen_mt_like_eval_domains' if SINGLE_SRC_SUBSET else '01-eval_all_domains'}_{protocol}.tex",
    "w",
) as f:

    def eval_print_table(method_name, is_method_tgt_lang_dep, proportion=0.5):
        # 1) collect each domain’s results
        domain_results = []
        for domain in domains:
            domain_results.append(
                difficulty_sampling.evaluate.main_eval_avg(
                    method_name,
                    data=data,
                    single_src_subset=SINGLE_SRC_SUBSET,
                    is_method_tgt_lang_dep=is_method_tgt_lang_dep,
                    protocol=protocol,
                    proportion=proportion,
                    domains={domain},
                )
            )

        # 2) build per‐metric lists
        # 2a) AvgScore for each domain + macro‐avg
        scores = [results.avg_score for results in domain_results]
        macro_score = sum(scores) / len(scores)
        scores.append(macro_score)

        # 2b) DiffCorr for each domain + macro‐avg
        corrs = [results.diff_corr for results in domain_results]
        macro_corr = sum(corrs) / len(corrs)
        corrs.append(macro_corr)

        # 2c) %Perfect for each domain + macro‐avg
        perfs = [100 * results.avg_perfect for results in domain_results]
        macro_perf = sum(perfs) / len(perfs)
        perfs.append(macro_perf)

        # 3) format for LaTeX
        formatted = []
        # first 5 are AvgScore → one decimal
        for val in scores:
            formatted.append(f"{val:.1f}")
        # next 5 are DiffCorr → three decimals
        for val in corrs:
            formatted.append(f"{val:.3f}")
        # last 5 are %Perfect → one decimal + “\%”
        for val in perfs:
            formatted.append(f"{val:.1f}\\%")

        # 4) write the row
        display_name = METHOD_TO_NAME.get(method_name, method_name.replace("_", " "))
        star = "*" if is_method_tgt_lang_dep else ""
        row = " & ".join(formatted)
        f.write(f"{display_name}{star} & {row} \\\\\n")

    for method_name, is_method_tgt_lang_dep in [
        ("random", False),
        ("human", True),
        ("src_len", False),
        ("syntactic_complexity", False),
        ("negative_word_frequency", False),
        ("negative_word_zipf_frequency", False),
        ("precomet_avg", False),
        ("precomet_var", False),
        ("precomet_diff", False),
        ("precomet_diversity", False),
        ("sentinel-src-da", False),
        ("sentinel-src-da-more-data", False),
        ("sentinel-src-da-z-norm-per-sys", False),
        ("sentinel-src-da-z-norm-per-sys-no-domain", False),
        ("sentinel-src-da-beta", False),
        ("sentinel-src-da-beta-no-domain", False),
        ("sentinel-src-mqm", False),
        ("sentinel-src-mqm-z-norm-more-data", False),
        ("sentinel-src-mqm-from-da-more-data", False),
        ("sentinel-src-mqm-z-norm-per-sys", False),
        ("sentinel-src-mqm-z-norm-per-sys-no-domain", False),
        ("sentinel-src-mqm-beta", False),
        ("sentinel-src-mqm-beta-no-domain", False),
        ("sentinel-src-da-for-genmt25", False),
        ("sentinel-src-mqm-for-genmt25", False)
        # ("artcrowd|GPT-4|MetricX-24-Hybrid-QE", True),
        # ("artcrowd|GPT-4|XCOMET-QE", True),
        # ("artcrowd|GPT-4|human", True),
        # ("ext_artcrowd|MetricX-24-Hybrid-QE-XXL", True),
        # ("ext_artcrowd|XCOMET-QE-XXL", True),
        # ("ext_artcrowd|CometKiwi-XXL", True),
    ]:
        eval_print_table(method_name, is_method_tgt_lang_dep)
"""

if RUN_STAT_SIGN_ON_DIFFCORR:
    diff_corr_tasks, wts = DiffCorrTasks.diff_correlations_on_wmt24(
        list(data.lp2src_data_list), k=K
    )

    new_results = diff_corr_tasks.Run(data, list(METHOD_TO_NAME), METHOD_TO_NAME)
    if K > 0:
        avg_corrs, matrix = new_results.AverageCorrMatrix(wts)
    else:
        avg_corrs = new_results.AverageCorrs(wts)

    with open(
        difficulty_sampling.ROOT / f"generated/diff_corr_table_{protocol}.txt", "w"
    ) as f:
        table = new_results.Table(
            metrics=list(avg_corrs),
            initial_column=avg_corrs,
            initial_column_header="avg-corr",
            attr_list=["lang", "corr_fcn"],
            nicknames={"kendall": "DiffCorr"},
            fmt="text",
        )
        f.write(table)

    with open(
        difficulty_sampling.ROOT / f"generated/diff_corr_table_{protocol}.tex", "w"
    ) as f:
        table = new_results.Table(
            metrics=list(avg_corrs),
            initial_column=avg_corrs,
            initial_column_header="avg-corr",
            attr_list=["lang", "corr_fcn"],
            nicknames={"kendall": "DiffCorr"},
            fmt="latex",
        )
        f.write(table)
