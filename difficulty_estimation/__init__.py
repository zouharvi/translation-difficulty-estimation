# default imports
# flake8: noqa F401
import pathlib
import os

wmt24_from_en_lps_esa = [
    "en-cs",
    "en-es",
    "en-hi",
    "en-is",
    "en-ja",
    "en-ru",
    "en-uk",
    "en-zh",
]
wmt24_lps_esa = wmt24_from_en_lps_esa + ["cs-uk"]

wmt24_from_en_lps_mqm = ["en-de", "en-es"]
wmt24_lps_mqm = wmt24_from_en_lps_mqm + ["ja-zh"]

ROOT = pathlib.Path(__file__).parent.parent.absolute()
WMT24_GENMT_DATA = ROOT / "data" / "esa_wmt24_annotations" / "wmt24_esa.jsonl"

if not WMT24_GENMT_DATA.exists():
    URL = "https://github.com/wmt-conference/wmt24-news-systems/raw/refs/heads/main/jsonl/wmt24_esa.jsonl"
    import urllib.request
    import os
    os.makedirs(WMT24_GENMT_DATA.parent, exist_ok=True)
    print("Downloading WMT24 data")
    urllib.request.urlretrieve(URL, WMT24_GENMT_DATA)


import mt_metrics_eval.data
try:
    mt_metrics_eval.data.EvalSet("wmt24pp", "en-de_DE", False)
except ValueError:
    import subprocess
    print("Downloading MTME data")
    subprocess.run(
        ["python3", "-m", "mt_metrics_eval.mtme", "--download"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )

os.makedirs(ROOT / "generated", exist_ok=True)