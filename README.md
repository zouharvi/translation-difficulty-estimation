# Estimating Machine Translation Difficulty 
[![Paper](https://img.shields.io/badge/📜%20paper-481.svg)](TODO)
&nbsp;
[![Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Collection-FCD21D)](https://huggingface.co/collections/Prosho/translation-difficulty-estimators-6816665c008e1d22426eb6c4)

This repository contains the code for the paper [_Estimating Machine Translation Difficulty_](TODO) by Lorenzo Proietti<sup>\*</sup>, Stefano Perrella<sup>\*</sup>, Vilém Zouhar<sup>\*</sup>, Roberto Navigli, Tom Kocmi.

> **Abstract:**
> Machine translation quality has began achieving near-perfect translations in some setups.
> These high-quality outputs make it difficult to distinguish between state-of-the-art models and to identify areas for future improvement.
> Automatically identifying texts where machine translation systems struggle holds promise for developing more discriminative evaluations and guiding future research.
> We formalize the task of translation difficulty estimation, defining a text's difficulty based on the expected quality of its translations.
> We introduce a new metric to evaluate difficulty estimators and use it to assess both baselines and novel approaches.
> Finally, we demonstrate the practical utility of difficulty estimators by using them to construct more challenging machine translation benchmarks. 
> Our results show that dedicated models (dubbed Sentinel-src) outperform both heuristic-based methods (e.g. word rarity or syntactic complexity) and LLM-as-a-judge approaches.
> We release two improved models for difficulty estimation, Sentinel-src-24 and Sentinel-src-25, which can be used to scan large collections of texts and select those most likely to challenge contemporary machine translation systems.

## Trained difficulty estimation models

For trained Sentinel difficulty estimation models, see the associated [HuggingFace collection](https://huggingface.co/collections/Prosho/translation-difficulty-estimators-6816665c008e1d22426eb6c4).

## Replicating experiments

To access data loaders and evaluators, you need to install the local package, which will also make sure all the dependencies are installed:
```
pip3 install -e .
```

Then, in Python, you can use the loader, which is unified across all datasets and automatically fetches data.
The first time you try to load the data, it will take long (30s-5min). Then, everything should be cached and instant:

```python
import difficulty_estimation
data = difficulty_estimation.utils.load_data_wmt_all()
data.keys()
> dict_keys([('wmt23', 'cs-uk'), ('wmt23', 'de-en'), ('wmt23', 'en-cs'), ('wmt23', 'en-de'), ('wmt23', 'en-ja'), ('wmt23', 'en-zh'), ('wmt23', 'he-en'), ('wmt23', 'ja-en'), ('wmt23', 'zh-en'), ('wmt24', 'cs-uk'), ('wmt24', 'en-cs'), ('wmt24', 'en-es'), ('wmt24', 'en-hi'), ('wmt24', 'en-is'), ('wmt24', 'en-ja'), ('wmt24', 'en-ru'), ('wmt24', 'en-uk'), ('wmt24', 'en-zh'), ('wmt24', 'ja-zh'), ('wmt22', 'cs-uk'), ('wmt22', 'en-cs'), ('wmt22', 'en-de'), ('wmt22', 'en-hr'), ('wmt22', 'en-ja'), ('wmt22', 'en-ru'), ('wmt22', 'en-uk'), ('wmt22', 'en-zh'), ('wmt22', 'ru-en'), ('wmt22', 'sah-ru'), ('wmt22', 'zh-en'), ('wmt21.tedtalks', 'en-de'), ('wmt21.tedtalks', 'en-ru'), ('wmt21.tedtalks', 'zh-en'), ('wmt21.news', 'en-cs'), ('wmt21.news', 'en-de'), ('wmt21.news', 'en-is'), ('wmt21.news', 'en-ja'), ('wmt21.news', 'en-ru'), ('wmt21.news', 'zh-en'), ('wmt20', 'zh-en'), ('wmt20', 'en-de'), ('wmt19', 'kk-en'), ('wmt19', 'de-en'), ('wmt19', 'gu-en'), ('wmt19', 'lt-en')])
```

Each dataset has some number of source items and each item contains a few keys.
Each row thus contains a unique source and multiple scored translations.
```python
len(data[("wmt24", "en-cs")])
> 572

data[("wmt24", "en-cs")][0].keys()
> dict_keys(['i', 'src', 'ref', 'tgt', 'cost', 'domain', 'doc', 'scores'])
```

Run scripts in `experiments/` for main results, for example`python3 experiments/01-eval_all.py`.
Further documentation WIP.

## Misc.

If you use this work, please cite:
```bibtex
@misc{proietti2025estimating,
    author={Lorenzo Proietti, Stefano Perrella, Vilém Zouhar, Roberto Navigli, Tom Kocmi},
    title={Estimating Machine Translation Difficulty},
    year={2025},
    url={TODO}
}
```