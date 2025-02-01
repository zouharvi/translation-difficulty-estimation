# %%

import difficulty_sampling.utils

data = difficulty_sampling.utils.load_data_wmt(year="wmt24", langs="en-es", normalize=False)

# %%

import subset2evaluate
import subset2evaluate.select_subset

data_y_all = {}
# "local_cometsrc_cons"
for method in ["cometsrc_diversity", "cometsrc_avg", "cometsrc_var", "cometsrc_diffdisc", "cometsrc_diffdisc_direct"]:
    data_y_all[method] = subset2evaluate.select_subset.basic(data, method=method)

# %%
import matplotlib.pyplot as plt
import numpy as np


data_y1 = np.array([np.average([v["human"]  for v in l["scores"].values()]) for l in data])
len_flat = len(data_y1)
data_y2 = np.array([np.average([v["human"]  for v in l["scores"].values()]) for l in data_y_all["cometsrc_diversity"]][:int(len_flat*0.50)])
data_y3 = np.array([np.average([v["human"]  for v in l["scores"].values()]) for l in data_y_all["cometsrc_diversity"]][:int(len_flat*0.25)])

plt.hist(
    [data_y1, data_y2, data_y3],
    density=True,
    bins=np.arange(20, 100, 10),
    label=["All", "Selected 50%", "Selected 25%"],
)

plt.legend()
plt.title("Selection with COMETsrc - diversity")
plt.show()

# %%



data_y1 = np.array([np.average([v["human"]  for v in l["scores"].values()]) for l in data])
len_flat = len(data_y1)
data_y2 = np.array([np.average([v["human"]  for v in l["scores"].values()]) for l in data_y_all["cometsrc_avg"]][:int(len_flat*0.50)])
data_y3 = np.array([np.average([v["human"]  for v in l["scores"].values()]) for l in data_y_all["cometsrc_avg"]][:int(len_flat*0.25)])

plt.hist(
    [data_y1, data_y2, data_y3],
    density=True,
    bins=np.arange(20, 100, 10),
    label=["All", "Selected 50%", "Selected 25%"],
)

plt.legend()
plt.title("Selection with COMETsrc - avg")
plt.show()