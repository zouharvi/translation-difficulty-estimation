"""
TODO: resolve the structure, things are a bit messy with subsampling not being in difficulty_sampling package
"""

# %%

import difficulty_sampling
import difficulty_sampling.evaluate
import difficulty_sampling.utils
import difficulty_sampling.data
import subsampling.sentinel
import subsampling.syntactic_complexity
import subsampling.word_frequency
import subsampling.misc


data = difficulty_sampling.data.Data.load(dataset_name="wmt24", lps=["en-x"], domains="all", protocol="esa")

# apply scorers to the whole data
subsampling.misc.apply_src_len(data)
subsampling.syntactic_complexity.syntactic_complexity_score(data, "syntactic_complexity")
subsampling.word_frequency.word_frequency_score(data, "word_frequency")
subsampling.word_frequency.word_frequency_score(data, "word_zipf_frequency")
subsampling.misc.apply_subset2evaluate(data, method="precomet_avg")
subsampling.misc.apply_subset2evaluate(data, method="precomet_var")
subsampling.misc.apply_subset2evaluate(data, method="precomet_diff")
subsampling.misc.apply_subset2evaluate(data, method="precomet_diversity")
# NOTE: does not work now
# subsampling.sentinel.sentinel_src_metric_model_score(subsampling.sentinel.get_sentinel_src_metric_model("sapienzanlp/sentinel-src-mqm"), data, use_tgt_lang_token=True)

METHOD_TO_NAME = {
    "src_len": "Source Length",
    "syntactic_complexity": "Syntactic Complexity",
    "word_frequency": "Word Frequency",
    "word_zipf_frequency": "Word Zipf Frequency",
    "precomet_avg": "PreCOMET Average",
    "precomet_var": "PreCOMET Variance",
    "precomet_diff": "PreCOMET Difficulty",
    "precomet_diversity": "PreCOMET Diversity",
    "sentinel_src_metric_model": "Sentinel-SRC-MQM",
}

with open(difficulty_sampling.ROOT / "generated/01-eval_all.tex", "w") as f:
    def eval_print_table(method_name, B=100):
        results = difficulty_sampling.evaluate.main_eval_avg(method_name, data=data, B=B)
        method_name = METHOD_TO_NAME.get(method_name, method_name.replace('_', ' ').title())
        print(
            f"{method_name:>20}",
            f"{results.avg_score:.2f}",
            f"{results.diff_corr:.3f}",
            f"{results.clusters:.2f}",
            sep = " & ",
            end=" \\\\\n",
            file=f,
        )


    for method_name in [
        "src_len", "syntactic_complexity", "word_frequency", "word_zipf_frequency",
        "precomet_avg", "precomet_var", "precomet_diff", "precomet_diversity",
    ]:
        eval_print_table(method_name)
