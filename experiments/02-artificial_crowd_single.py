"""
TODO: resolve the structure, things are a bit messy with subsampling not being in difficulty_estimation package
"""

# %%

import numpy as np
import difficulty_estimation
import difficulty_estimation.evaluate
import difficulty_estimation.utils
import difficulty_estimation.data
import subsampling.misc

data = difficulty_estimation.data.Data.load(
    dataset_name="wmt24", lps=["en-x"], domains="all", protocol="esa", include_ref=True
)

# models in all data
data_all = list(data.lp2src_data_list.values())
models = set(data_all[0][0]["scores"].keys())
# filter to models in all data
for lp2src_data in data_all:
    models = models.intersection(set(lp2src_data[0]["scores"].keys()))

# sort models by average human score across all languages
models = list(models)
models.sort(
    key=lambda model: np.average(
        [line["scores"][model]["human"] for data in data_all for line in data]
    ),
    reverse=True,
)

# metrics in all data
metrics = set(data_all[0][0]["scores"][models[0]].keys())
# filter to metrics in all data
for lp2src_data in data_all:
    metrics = metrics.intersection(set(lp2src_data[0]["scores"][models[0]].keys()))

# sort metrics by correlation with human across all languages
metrics = list(metrics)
metrics.sort(
    key=lambda metric: np.corrcoef(
        [
            line["scores"][model][metric]
            for data in data_all
            for line in data
            for model in models
        ],
        [
            line["scores"][model]["human"]
            for data in data_all
            for line in data
            for model in models
        ],
    )[0, 1],
    reverse=True,
)

# print correlations
for metric in metrics:
    # average
    corr_avg = np.corrcoef(
        [line["scores"]["GPT-4"][metric] for data in data_all for line in data],
        [line["scores"]["GPT-4"]["human"] for data in data_all for line in data],
    )[0, 1]
    corr_hum = np.corrcoef(
        [line["scores"]["refA"][metric] for data in data_all for line in data],
        [line["scores"]["refA"]["human"] for data in data_all for line in data],
    )[0, 1]
    print(f"{metric:>50}", f"{corr_hum:.3f}", f"{corr_avg:.3f}")

# %%
# remove some metrics that are near duplicates
# metrics.remove("metametrics_mt_mqm_same_source_targ")
# metrics.remove("metametrics_mt_mqm_hybrid_kendall")
# metrics.remove("MetricX-24-Hybrid")
# metrics.remove("PrismRefSmall")

results = {}
# apply scorers to the whole data
for model in models:
    for metric in metrics:
        subsampling.misc.apply_internal_artificial_crowd_metrics(
            data, model=model, metric=metric
        )
        results[(model, metric)] = difficulty_estimation.evaluate.main_eval_avg(
            f"artcrowd|{model}|{metric}", data=data, budget=100
        )._asdict()


# %%
for metametric in ["avg_score", "diff_corr", "avg_perfect"]:
    formatter = {
        "avg_score": "{:.1f}",
        "diff_corr": "{:.3f}",
        "avg_perfect": "{:.1%}",
    }[metametric]
    with open(
        difficulty_estimation.ROOT
        / f"generated/02-artificial_crowd_single-{metametric}.tex",
        "w",
    ) as f:
        print(r"\begin{tabular}{l" + "c" * len(models) + "}", file=f)
        print(r"\toprule", file=f)
        print(
            "\\bf Metric",
            *[model.replace("_", " ") for model in models],
            sep=" & \\bf ",
            end=" \\\\\n",
            file=f,
        )
        print(r"\midrule", file=f)
        for metric in metrics:
            # TODO: cell color
            print(
                metric.replace("_", " "),
                *[
                    formatter.format(results[(model, metric)][metametric]).replace(
                        "%", "\\%"
                    )
                    for model in models
                ],
                sep=" & ",
                end=" \\\\\n",
                file=f,
            )
        print(r"\bottomrule", file=f)
        print(r"\end{tabular}", file=f)
