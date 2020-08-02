import pickle
import pandas as pd
import os
import numpy as np

results_microrca = "/home/matilda/PycharmProjects/RCA_metrics /3_Preprocessing_data/MicroRCA_0.4_0.035.csv"
path_restulsAE = "/home/matilda/PycharmProjects/RCA_metrics /5_Insights/store_results/"

micro_rca = pd.read_csv(results_microrca, header=None)
micro_rca.columns = ["id", "trueee", "experiment", "rank", "smth", "smth2"]


micro_rca["cause"] = [[] for _ in range(len(micro_rca))]
micro_rca["cause_ranking"] = np.zeros(micro_rca.shape[0])


groups = micro_rca.groupby(["id", "trueee", "experiment"]).groups

all_content = []

for file in os.listdir(path_restulsAE):
    service = file.rsplit("_")[0]
    experiment = file.rsplit("_")[1].rsplit(".")[0]

    with open(path_restulsAE + service + "_" + experiment + ".pickle", "rb") as file:
        content = pickle.load(file)
        all_content.append(content)



def appen_(y, first_index):
    return [x for x in first_index]

for el in all_content:
    first_index = el["error"]
    second_index = el["service_name"]
    third_index = "service_" + el["experiment"]
    fourth_index = el["indecies"]

    for idx in list(first_index.keys()):
        first_index_first = idx
        first_index_value = fourth_index[idx]
        micro_rca.loc[groups[(first_index_first, second_index, third_index)][0], "cause"] =  ', '.join(first_index[idx])
        micro_rca.loc[groups[(first_index_first, second_index, third_index)][0], "cause_ranking"] = int(first_index_value)

# micro_rca.to_csv("/home/matilda/PycharmProjects/RCA_metrics /5_Insights/store_results_experiments2/results_exp_1.csv", index=False)