# default imports
# flake8: noqa F401
import pathlib

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
