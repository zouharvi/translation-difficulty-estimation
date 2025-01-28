# difficulty-sampling

Intro TODO.

## Instructions

To access data loaders and evaluators, you need to install the local package, which will also make sure all the dependencies are installed:
```
pip3 install -e .
```


Then, in Python, you can use the loader, which is unified across all datasets and automatically fetches data.
The first time you try to load the data, it will take long (30s-5min). Then, everything should be cached and instant:

```python
import difficulty_sampling
data = difficulty_sampling.utils.load_data_wmt_all()
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

Evaluation for now piggy-backs on top of what subset2evaluate is doing.
This might change later.
See `experiments/vilem/01-example.py` for an example how to create a manual difficulty sampler and evaluate it.

## Contributing

- Do not commit data or executed Jupyter notebooks into this repository.
- Add your code only to `experiments/` and ideally separate based on topic, such as `experiments/sentinel/` i.e. no new top-level directories.
- If you have any code dependencies, do not upload your `requirements.txt` but add the specific ones to `pyproject.toml`.
- Use preferably interactive Python ([works great in VSCode!](https://code.visualstudio.com/docs/python/jupyter-support-py)) over Jupyter notebooks, which are more difficult to version.