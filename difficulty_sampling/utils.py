"""
We piggy-back on top of subset2evaluate for now. This might change later.
"""
import subset2evaluate.utils

load_data_wmt_all = subset2evaluate.utils.load_data_wmt_all
load_data_wmt = subset2evaluate.utils.load_data_wmt

tgt2lp = {
    "chinese": "en-zh",
    "czech": "en-cs",
    "hindi": "en-hi",
    "icelandic": "en-is",
    "japanese": "en-ja",
    "russian": "en-ru",
    "spanish": "en-es",
    "ukrainian": "en-uk",
}