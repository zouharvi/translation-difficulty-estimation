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

data_all = difficulty_sampling.data.Data.load(dataset_name="wmt24", lps=["all"], domains="all", protocol="esa")

# %%
subsampling.misc.apply_subset2evaluate(data_all, method="random")
subsampling.syntactic_complexity.src_len_score(data_all)
subsampling.syntactic_complexity.syntactic_complexity_score(data_all, "syntactic_complexity")
subsampling.negative_word_frequency.negative_word_frequency_score(data_all, "negative_word_frequency")
subsampling.misc.apply_subset2evaluate_cache(data_all, method="precomet_avg")
subsampling.misc.apply_subset2evaluate_cache(data_all, method="precomet_var")
subsampling.misc.apply_subset2evaluate_cache(data_all, method="precomet_diff")
subsampling.misc.apply_subset2evaluate_cache(data_all, method="precomet_diversity")

# apply scorers to the whole data
subsampling.sentinel.sentinel_src_metric_model_score(
    subsampling.sentinel.get_sentinel_src_metric_model("sapienzanlp/sentinel-src-mqm"),
    scorer_name="sentinel-src-mqm",
    data=data_all,
    use_tgt_lang_token=False,
)
subsampling.sentinel.sentinel_src_metric_model_score(
    subsampling.sentinel.get_sentinel_src_metric_model(
        "Prosho/sentinel-src-mqm-wmt1923"
    ),
    scorer_name="sentinel-src-mqm-wmt1923",
    data=data_all,
    use_tgt_lang_token=False,
)

# NOTE: temporary hack for artificial crowd
# GPT-4 does not exist for 7 examples in en-cs
model = "GPT-4"
for metric in ["human", "XCOMET-QE"]:
    for data_name, data_local in data_all.lp2src_data_list.items():
        for line in data_local:
            if model not in line["scores"]:
                print("skipping", data_name)
                score = line["scores"]["Aya23"][metric]
            else:
                score = line["scores"][model][metric]
                
            for sys in line["scores"].keys():
                line["scores"][sys]["artcrowd|" + model + "|" + metric] = score

# subsampling.misc.apply_artificial_crowd_metrics(data_all, model="GPT-4", metric="XCOMET-QE")
# subsampling.misc.apply_artificial_crowd_metrics(data_all, model="GPT-4", metric="human")

subsampling.misc.apply_external_artificial_crowd_metrics(
    data_all,
    sys2translations_path=Path(
        "../data/external_artificial_crowd/scored_translations/sys2translations.pickle"
    ),
    metric="MetricX-24-Hybrid-QE-XXL",
)
subsampling.misc.apply_external_artificial_crowd_metrics(
    data_all,
    sys2translations_path=Path(
        "../data/external_artificial_crowd/scored_translations/sys2translations.pickle"
    ),
    metric="XCOMET-QE-XXL",
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
METHOD_TO_NAME = {
    "random": "Random",
    "human": "Oracle",
    "src_len": "Source Length",
    "syntactic_complexity": "Syntactic Complexity",
    "negative_word_frequency": "Negative Word Frequency",
    "precomet_avg": "PreCOMET Average",
    "precomet_var": "PreCOMET Variance",
    "precomet_diff": "PreCOMET Difficulty",
    "precomet_diversity": "PreCOMET Diversity",
    "sentinel-src-mqm": "Sentinel MQM (original)",
    "sentinel-src-mqm-wmt1923": "Sentinel MQM (new)",
    "artcrowd|GPT-4|XCOMET-QE": "Art. Crowd (XCOMET)",
    "artcrowd|GPT-4|human": "Art. Crowd (Oracle)",
    "ext_artcrowd|MetricX-24-Hybrid-QE-XXL": "Ext. Crowd (MetricX)",
    "ext_artcrowd|XCOMET-QE-XXL": "Ext. Crowd (XCOMET)",
}

import random

def format_cell_tau(v, minv=0, maxv=0.250):
    vcol = min(maxv, abs(v))
    vcol = (vcol-minv) / (maxv-minv) * 80
    return f"\\cellcolor{{SpringGreen3!{int(vcol)}}} {random.randint(1, 10)} \\hspace{{2mm}} {v:.3f}"

def format_cell_pearson(v, minv=0, maxv=0.250):
    vcol = min(maxv, abs(v))
    vcol = (vcol-minv) / (maxv-minv) * 80
    return f"\\cellcolor{{CadetBlue3!{int(vcol)}}} {random.randint(1, 10)} \\hspace{{2mm}} {v:.3f}"

fout = open(difficulty_sampling.ROOT / f"generated/06-eval_some.tex", "w")
print(
r"""
\begin{tabular}{lrr}
\toprule
Method & \hspace{-3mm} Kendall's $\tau_b$ \hspace{-3mm} & Pearson \\
\midrule
""",
    file=fout
)

for method, method_name in METHOD_TO_NAME.items():
    results = difficulty_sampling.evaluate.main_eval_avg(
        method, data=data_all,
        budget=100,
    )
    diff_tau = results.diff_tau
    diff_pearson = results.diff_pearson
    if method == "random":
        diff_tau = 0.0
        diff_pearson = 0.0
    print(
        f"{method_name:>20}",
        format_cell_tau(diff_tau),
        format_cell_pearson(diff_pearson),
        sep=" & ",
        end=" \\\\\n",
        file=fout,
    )

print(r"""
\bottomrule
\end{tabular}
""",
    file=fout
)
fout.close()