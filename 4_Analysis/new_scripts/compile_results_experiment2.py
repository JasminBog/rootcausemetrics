import pickle
import pandas as pd
import os
import numpy as np

results_microrca = "/home/matilda/PycharmProjects/RCA_metrics /3_Preprocessing_data/MicroRCA_0.4_0.035.csv"
path_restulsAE = "/home/matilda/PycharmProjects/RCA_metrics /5_Insights/store_results_experiments2/"

micro_rca = pd.read_csv(results_microrca, header=None)
micro_rca.columns = ["id", "trueee", "experiment", "rank", "smth", "smth2"]


micro_rca["cause"] = ["" for _ in range(len(micro_rca))]
# micro_rca["cause_ranking"] = np.zeros(micro_rca.shape[0])


groups = micro_rca.groupby(["id", "trueee", "experiment"]).groups

all_content = []
v = []

for file in os.listdir(path_restulsAE):
    service = file.rsplit("_")[0]
    experiment = file.rsplit("_")[1].rsplit(".")[0]
    anom_id = file.rsplit("_")[2]
    other_service = file.rsplit("_")[3].rsplit("_.")[0]
    v.append(int(anom_id))
    with open(path_restulsAE + file, "rb") as file:
        content = pickle.load(file)
        all_content.append(content)

from collections import defaultdict
defdict = defaultdict(list)

for idx, el in enumerate(all_content):
    defdict[(v[idx], el["service_name"], "service_" + el["experiment"] )].append((''.join(el["other_service"]), ''.join([str(el["score"])]), ''.join(el["decison"]), ', '.join(el["error"])))

for keys in list(defdict.keys()):
    micro_rca.loc[groups[keys][0], "cause"] = ''.join(str(defdict[keys]))

micro_rca.to_csv("/home/matilda/PycharmProjects/RCA_metrics /5_Insights/results/results_exp_2.csv", index=False)