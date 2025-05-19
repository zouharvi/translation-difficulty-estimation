# %%

import numpy as np
import difficulty_sampling
import difficulty_sampling.evaluate
import difficulty_sampling.utils
import difficulty_sampling.data
import matplotlib.pyplot as plt
from difficulty_sampling.utils import tgt2lp
lp2tgt = {v: k for k, v in tgt2lp.items()}

LPS = ["en-cs", "en-es", "en-hi", "en-is", "en-ja", "en-ru", "en-uk", "en-zh"]
data_all = difficulty_sampling.data.Data.load(
    dataset_name="wmt24", lps=LPS, domains="all", protocol="esa"
)

# %%

def get_all_scores(data_local):
    return [
        line["scores"][sys]["human"]
        for line in data_local
        for sys in line["scores"].keys()
    ]

def get_top_score(data_local):
    return [
        max([
            line["scores"][sys]["human"]
            for sys in line["scores"].keys()
        ])
        for line in data_local
    ]

def get_top_system(data_local):
    # get top-k systems
    sys_best = sorted(
        list(data_local[0]["scores"].keys()),
        key=lambda sys: np.nanmean([
            line["scores"][sys]["human"] if sys in line["scores"] else np.nan
            for line in data_local
        ]),
        reverse=True
    )[:1]

    return [
        line["scores"][sys]["human"]
        for line in data_local
        for sys in sys_best
        if sys in line["scores"]
    ]

difficulty_sampling.utils.matplotlib_default()

def plot_eightpack(data_all: difficulty_sampling.data.Data, suffix: str = ""):
    _, axs= plt.subplots(4, 2, figsize=(3.9, 3.6), sharex=True, sharey=True)
    axs = list(axs.flatten())
    for lp, ax in zip(LPS, axs):
        data_local = data_all.lp2src_data_list[lp]
        data_flat1 = get_all_scores(data_local)
        data_flat2 = get_top_system(data_local)
        data_flat3 = get_top_score(data_local)

        ax.hist(
            [data_flat1, data_flat2, data_flat3],
            bins=range(0, 100+15, 15),
            density=True,
            label=["All", "Top system", "Top translation"],
        )
        # r"En$\rightarrow$" + 
        ax.text(
            0.05,
            0.75,
            lp2tgt[lp].capitalize(),
            # top left
            transform=ax.transAxes,
            weight="bold",
        )
        ax.set_ylim(0, 0.07)
        if lp in {"en-cs", "en-hi", "en-ja", "en-uk"}:
            ax.set_ylabel("Freq.")
        ax.set_yticks([])
        if lp in {"en-uk", "en-zh"}:
            ax.set_xlabel("ESA Score")
        ax.spines[["top", "right"]].set_visible(False)

    handles = axs[0].get_legend_handles_labels()
    plt.tight_layout(pad=0.2)
    plt.savefig(difficulty_sampling.ROOT / f"generated/07-perlang_wmt24{suffix}.pdf")
    plt.show()

    # only legend
    plt.figure(figsize=(3.9, 0.5))
    plt.legend(
        *handles,
        loc="center",
        ncol=3,
        frameon=False,
    )
    # turn off axis
    plt.gca().spines[["left", "top", "right", "bottom"]].set_visible(False)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout(pad=0)
    plt.savefig(difficulty_sampling.ROOT / "generated/07-perlang_wmt24_legend.pdf")
    plt.show()


plot_eightpack(data_all)

# %%
from pathlib import Path
import subsampling.sentinel
import subsampling.misc
import subsampling.syntactic_complexity

subsampling.sentinel.sentinel_src_metric_model_score(
    subsampling.sentinel.get_sentinel_src_metric_model(
        "Prosho/sentinel-src-mqm-wmt1723"
    ),
    scorer_name="sentinel-src-mqm-wmt1723",
    data=data_all,
    use_tgt_lang_token=True,
)
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
    data_all, scorer_name="oracle-tgt", use_tgt_lang=True
)
subsampling.misc.apply_oracle_with_fixed_scores(
    data_all, scorer_name="oracle-src", use_tgt_lang=False
)

# %%
import copy

def compute_eightpack(data_all: difficulty_sampling.data.Data, key: str, suffix: str = ""):
    PROP = 0.25
    data_all = copy.deepcopy(data_all)
    for lp in data_all.lp2src_data_list.keys():
        data_all.lp2src_data_list[lp] = sorted(
            data_all.lp2src_data_list[lp],
            key=lambda line: np.average([line["scores"][sys][key] for sys in line["scores"].keys()]),
            reverse=False,
        )[:int(len(data_all.lp2src_data_list[lp]) * PROP)]
    plot_eightpack(data_all, suffix=suffix)

compute_eightpack(data_all, key="oracle-tgt", suffix="_oracletgt")
compute_eightpack(data_all, key="oracle-src", suffix="_oraclesrc")
compute_eightpack(data_all, key="sentinel-src-mqm-wmt1723", suffix="_sentinel")
compute_eightpack(data_all, key="syntactic_complexity", suffix="_syntax")
compute_eightpack(data_all, key="ext_artcrowd|XCOMET-QE-XXL", suffix="_crowd")
compute_eightpack(data_all, key="LLM-as-a-Judge (Command-A)", suffix="_llmjudge")

