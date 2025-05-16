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
difficulty2color = {
    "easy": "#a1d48d",
    "mixed": "#ffd27d",
    "hard": "#ea848d",
}
COLORS = [
    "#bc272d",  # red
    "#50ad9f",  # green
    "#0000a2",  # blue
    "#e0c016",  # yellow
    "#6a5371",  # purple
]

def matplotlib_default():
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    # use serif
    plt.rcParams.update({"font.family": "serif"})

    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["axes.prop_cycle"] = plt.cycler(color=COLORS)
    mpl.rcParams["legend.fancybox"] = False
    mpl.rcParams["legend.edgecolor"] = "None"
    mpl.rcParams["legend.fontsize"] = 9
    mpl.rcParams["legend.borderpad"] = 0.1


def turn_off_spines(which=['top', 'right']):
    import matplotlib.pyplot as plt

    ax = plt.gca()
    ax.spines[which].set_visible(False)