"""
TODO: resolve the structure, things are a bit messy with subsampling not being in difficulty_estimation package
"""
# %%

from pathlib import Path
from typing import Literal

import difficulty_estimation
import difficulty_estimation.evaluate
import difficulty_estimation.utils
import difficulty_estimation.data
import subsampling.sentinel
import subsampling.syntactic_complexity
import subsampling.average_word_frequency
import subsampling.misc
from difficulty_estimation import DiffCorrTasks

SINGLE_SRC_SUBSET = False

RUN_STAT_SIGN_ON_DIFFCORR, K = False, 1000
corr_fcn_for_diff: Literal["kendall", "pearson"] = "kendall"

protocol: Literal["esa", "mqm"] = "esa"
data = difficulty_estimation.data.Data.load(
    dataset_name="wmt24",
    lps=["all"],
    domains="all",
    protocol=protocol,
    # include_ref=True,
    # include_human=True,
)
domains = ["news", "social", "literary", "speech"]

score_all_source_texts = True

# apply scorers to the whole data

subsampling.sentinel.sentinel_src_metric_model_score(
    subsampling.sentinel.get_sentinel_src_metric_model("sapienzanlp/sentinel-src-mqm"),
    scorer_name="sentinel-src-mqm",
    data=data,
    score_all_source_texts=score_all_source_texts,
)
subsampling.sentinel.sentinel_src_metric_model_score(
    subsampling.sentinel.get_sentinel_src_metric_model(
        "Prosho/sentinel-src-mqm-wmt1723"
    ),
    scorer_name="sentinel-src-mqm-wmt1723",
    data=data,
    score_all_source_texts=score_all_source_texts,
)

subsampling.misc.apply_oracle_with_fixed_scores(
    data, scorer_name="oracle_with_fixed_scores", use_tgt_lang=False
)
subsampling.misc.apply_oracle_with_fixed_scores(
    data, scorer_name="oracle_with_fixed_scores_tgt", use_tgt_lang=True
)

subsampling.misc.apply_random(data, scorer_name="random", seed=42)
# subsampling.misc.apply_subset2evaluate(data, method="random")

# subsampling.misc.apply_src_len(data)
subsampling.syntactic_complexity.syntactic_complexity_score(
    data,
    "syntactic_complexity",
    score_all_source_texts=score_all_source_texts,
)
subsampling.syntactic_complexity.src_len_score(
    data,
    "src_len",
    score_all_source_texts=score_all_source_texts,
)
subsampling.average_word_frequency.avg_word_freq_score(
    data,
    "avg_word_freq",
    score_all_source_texts=score_all_source_texts,
)

subsampling.misc.apply_subset2evaluate_cache(data, method="precomet_diff")
subsampling.misc.apply_subset2evaluate_cache(data, method="precomet_diversity")

subsampling.misc.apply_internal_artificial_crowd_metrics(
    data, model="all", metric="MetricX-24-Hybrid-QE"  # GPT-4
)
subsampling.misc.apply_internal_artificial_crowd_metrics(
    data, model="all", metric="XCOMET-QE"  # GPT-4
)
subsampling.misc.apply_external_artificial_crowd_metrics(
    data,
    sys2translations_path=Path(
        "../data/external_artificial_crowd/sys2translations.pickle"
    ),
    metric="MetricX-24-Hybrid-QE-XXL",
    protocol=protocol,
)
subsampling.misc.apply_external_artificial_crowd_metrics(
    data,
    sys2translations_path=Path(
        "../data/external_artificial_crowd/sys2translations.pickle"
    ),
    metric="XCOMET-QE-XXL",
    protocol=protocol,
)

subsampling.misc.apply_llm_as_a_judge(
    data,
    scored_source_texts_df_path=Path(
        f"../data/LLM-as-a-Judge/{protocol}/command-a/command-a-03-2025_source_based_num_scores.csv"
    ),
    llm_name="Command A, src-based",
)
subsampling.misc.apply_llm_as_a_judge(
    data,
    scored_source_texts_df_path=Path(
        f"../data/LLM-as-a-Judge/{protocol}/gpt-4o/gpt-4o-1120_source_based_num_scores.csv"
    ),
    llm_name="GPT-4o, src-based",
)
subsampling.misc.apply_llm_as_a_judge(
    data,
    scored_source_texts_df_path=Path(
        f"../data/LLM-as-a-Judge/{protocol}/command-a/command-a-03-2025_target_based_num_scores.csv"
    ),
    llm_name="Command A, tgt-based",
)
subsampling.misc.apply_llm_as_a_judge(
    data,
    scored_source_texts_df_path=Path(
        f"../data/LLM-as-a-Judge/{protocol}/gpt-4o/gpt-4o-1120_target_based_num_scores.csv"
    ),
    llm_name="GPT-4o, tgt-based",
)

METHOD_TO_NAME = {
    "random": "Random",
    "human": "Oracle",
    "oracle_with_fixed_scores": "Oracle with Fixed Scores",
    "oracle_with_fixed_scores_tgt": "Oracle with Fixed Scores (Tgt)",
    "src_len": "Source Length",
    "syntactic_complexity": "Syntactic Complexity",
    "avg_word_freq": "Word Rarity",
    "precomet_diff": "PreCOMET Difficulty",
    "precomet_diversity": "PreCOMET Diversity",
    "sentinel-src-mqm": "Sentinel-MQM",
    "sentinel-src-mqm-wmt1723": "Sentinel-MQM-WMT1723",
    "artcrowd|all|MetricX-24-Hybrid-QE": "Internal Artificial Crowd (MetricX-24-Hybrid-QE-XXL)",  # GPT-4
    "artcrowd|all|XCOMET-QE": "Internal Artificial Crowd (XCOMET-QE-XXL)",  # GPT-4
    "ext_artcrowd|MetricX-24-Hybrid-QE-XXL": "External Artificial Crowd (MetricX-24-Hybrid-QE-XXL)",
    "ext_artcrowd|XCOMET-QE-XXL": "External Artificial Crowd (XCOMET-QE-XXL)",
    "LLM-as-a-Judge (Command A, src-based)": "LLM-as-a-Judge (Command A, src-based)",
    "LLM-as-a-Judge (GPT-4o, src-based)": "LLM-as-a-Judge (GPT-4o, src-based)",
    "LLM-as-a-Judge (Command A, tgt-based)": "LLM-as-a-Judge (Command A, tgt-based)",
    "LLM-as-a-Judge (GPT-4o, tgt-based)": "LLM-as-a-Judge (GPT-4o, tgt-based)",
}

with open(
    difficulty_estimation.ROOT
    / f"generated/{'01-gen_mt_like_eval' if SINGLE_SRC_SUBSET else '01-eval_all'}_{protocol}.tex",
    "w",
) as f:

    def eval_print_table(method_name, is_method_tgt_lang_dep, proportion=0.25):
        results = difficulty_estimation.evaluate.main_eval_avg(
            method_name,
            data=data,
            single_src_subset=SINGLE_SRC_SUBSET,
            is_method_tgt_lang_dep=is_method_tgt_lang_dep,
            protocol=protocol,
            proportion=proportion,
        )
        method_name = METHOD_TO_NAME.get(method_name, method_name.replace("_", " "))
        print(
            f"{method_name:>20}{'*' if is_method_tgt_lang_dep else ''}",
            f"{results.avg_score:.1f}",
            # f"{results.diff_tau:.3f}",
            # f"{results.diff_pearson:.3f}",
            f"{(results.avg_perfect * 100):.1f}%".replace("%", "\\%"),
            sep=" & ",
            end=" \\\\\n",
            file=f,
        )

    for method_name, is_method_tgt_lang_dep in [
        ("random", True),
        ("human", True),
        ("oracle_with_fixed_scores", False),
        ("oracle_with_fixed_scores_tgt", True),
        ("src_len", False),
        ("syntactic_complexity", False),
        ("avg_word_freq", False),
        ("precomet_diff", False),
        ("precomet_diversity", False),
        ("sentinel-src-mqm", False),
        ("sentinel-src-mqm-wmt1723", False),
        ("artcrowd|all|MetricX-24-Hybrid-QE", True),  # GPT-4
        ("artcrowd|all|XCOMET-QE", True),  # GPT-4
        ("ext_artcrowd|MetricX-24-Hybrid-QE-XXL", True),
        ("ext_artcrowd|XCOMET-QE-XXL", True),
        ("LLM-as-a-Judge (Command A, src-based)", False),
        ("LLM-as-a-Judge (GPT-4o, src-based)", False),
        ("LLM-as-a-Judge (Command A, tgt-based)", True),
        ("LLM-as-a-Judge (GPT-4o, tgt-based)", True),
    ]:
        eval_print_table(method_name, is_method_tgt_lang_dep)

with open(
    difficulty_estimation.ROOT
    / f"generated/{'01-gen_mt_like_eval_domains' if SINGLE_SRC_SUBSET else '01-eval_all_domains'}_{protocol}.tex",
    "w",
) as f:

    def eval_print_table(method_name, is_method_tgt_lang_dep, proportion=0.25):
        # 1) collect each domain’s results
        domain_results = []
        for domain in domains:
            domain_results.append(
                difficulty_estimation.evaluate.main_eval_avg(
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
        # corrs = [results.diff_corr for results in domain_results]
        # macro_corr = sum(corrs) / len(corrs)
        # corrs.append(macro_corr)

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
        # for val in corrs:
        # formatted.append(f"{val:.3f}")
        # last 5 are %Perfect → one decimal + “\%”
        for val in perfs:
            formatted.append(f"{val:.1f}\\%")

        # 4) write the row
        display_name = METHOD_TO_NAME.get(method_name, method_name.replace("_", " "))
        star = "*" if is_method_tgt_lang_dep else ""
        row = " & ".join(formatted)
        f.write(f"{display_name}{star} & {row} \\\\\n")

    for method_name, is_method_tgt_lang_dep in [
        ("random", True),
        ("human", True),
        ("oracle_with_fixed_scores", False),
        ("oracle_with_fixed_scores_tgt", True),
        ("src_len", False),
        ("syntactic_complexity", False),
        ("avg_word_freq", False),
        ("precomet_diff", False),
        ("precomet_diversity", False),
        ("sentinel-src-mqm", False),
        ("sentinel-src-mqm-wmt1723", False),
        ("artcrowd|all|MetricX-24-Hybrid-QE", True),  # GPT-4
        ("artcrowd|all|XCOMET-QE", True),  # GPT-4
        ("ext_artcrowd|MetricX-24-Hybrid-QE-XXL", True),
        ("ext_artcrowd|XCOMET-QE-XXL", True),
        ("LLM-as-a-Judge (Command A, src-based)", False),
        ("LLM-as-a-Judge (GPT-4o, src-based)", False),
        ("LLM-as-a-Judge (Command A, tgt-based)", True),
        ("LLM-as-a-Judge (GPT-4o, tgt-based)", True),
    ]:
        eval_print_table(method_name, is_method_tgt_lang_dep)

if RUN_STAT_SIGN_ON_DIFFCORR:
    diff_corr_tasks, wts = DiffCorrTasks.diff_correlations_on_wmt24(
        list(data.lp2src_data_list), k=K, corr_fcn=corr_fcn_for_diff
    )

    new_results = diff_corr_tasks.Run(data, list(METHOD_TO_NAME), METHOD_TO_NAME)
    if K > 0:
        avg_corrs, matrix = new_results.AverageCorrMatrix(wts)
    else:
        avg_corrs = new_results.AverageCorrs(wts)

    with open(
        difficulty_estimation.ROOT
        / f"generated/diff_corr_table_{corr_fcn_for_diff}_{protocol}.txt",
        "w",
    ) as f:
        table = new_results.Table(
            metrics=list(avg_corrs),
            initial_column=avg_corrs,
            initial_column_header="avg-corr",
            attr_list=["lang", "corr_fcn"],
            nicknames={corr_fcn_for_diff: f"DiffCorr ({corr_fcn_for_diff})"},
            fmt="text",
        )
        f.write(table)

    with open(
        difficulty_estimation.ROOT
        / f"generated/diff_corr_table_{corr_fcn_for_diff}_{protocol}.tsv",
        "w",
    ) as f:
        table = new_results.Table(
            metrics=list(avg_corrs),
            initial_column=avg_corrs,
            initial_column_header="avg-corr",
            attr_list=["lang", "corr_fcn"],
            nicknames={corr_fcn_for_diff: f"DiffCorr ({corr_fcn_for_diff})"},
            fmt="tsv",
        )
        f.write(table)

    with open(
        difficulty_estimation.ROOT
        / f"generated/diff_corr_table_{corr_fcn_for_diff}_{protocol}.tex",
        "w",
    ) as f:
        table = new_results.Table(
            metrics=list(avg_corrs),
            initial_column=avg_corrs,
            initial_column_header="avg-corr",
            attr_list=["lang", "corr_fcn"],
            nicknames={corr_fcn_for_diff: f"DiffCorr ({corr_fcn_for_diff})"},
            fmt="latex",
        )
        f.write(table)
