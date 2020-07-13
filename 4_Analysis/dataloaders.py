import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import pandas as pd
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.utils.data import Dataset


datasets = {}

list_experiments = ["normal_metrics",
                    "service_cpu",
                    "service_memory",
                    "svc_latency"
                    ]

services = ["carts",
            "catalogue",
            "orders",
            "payment",
            "shipping",
            "user",
            ]

input_data_file_location =  "/home/matilda/PycharmProjects/RCA_metrics /2_Copy_Original_data/data/service/1"
response_times = "/home/matilda/PycharmProjects/RCA_metrics /2_Copy_Original_data/data/service/1/normal_metrics_latency_agg_destination_99.csv"
batch_size = 16


current_service = services[0]


experiments_list_sources = {"normal_metrics": "latency_agg_source_99.csv",
                            "service_cpu": "latency_agg_source_99.csv",
                            "service_memory": "latency_agg_source_99.csv",
                            "svc_latency": "latency_agg_source_99.csv",
                            }

experiments_list_destination = {"normal_metrics": "latency_agg_destination_99.csv",
                            "service_cpu": "latency_agg_destination_99.csv",
                            "service_memory": "latency_agg_destination_99.csv",
                            "svc_latency": "latency_agg_destination_99.csv",
                            }
results = {}

for idx1, type_experiment in enumerate(list_experiments):
    for idx, file_name in enumerate(os.listdir(input_data_file_location)):
        if type_experiment + "_" + current_service in file_name and "-db" not in file_name:
            a = pd.read_csv(input_data_file_location + "/" + file_name).iloc[:, 1:]

    if type_experiment == "normal_metrics":
        source = pd.DataFrame(pd.read_csv(input_data_file_location + "/" + type_experiment + "_" + experiments_list_sources[type_experiment]).loc[:, current_service])
        source.columns = ["source_latency"]
        destination = pd.DataFrame(pd.read_csv(input_data_file_location + "/" + type_experiment + "_" +  experiments_list_destination[type_experiment]).loc[:, current_service])
        destination.columns = ["destination_latency"]
        results[type_experiment + "_" + current_service ] = pd.concat([a, source, destination], axis=1)

    else:
        source = pd.DataFrame(pd.read_csv(
            input_data_file_location + "/" + type_experiment + "_" + current_service + "_" +  experiments_list_sources[type_experiment]).loc[:,
                              current_service])
        source.columns = ["source_latency"]
        destination = pd.DataFrame(pd.read_csv(
            input_data_file_location + "/" + type_experiment + "_" + current_service + "_" +  experiments_list_destination[type_experiment]).loc[
                                   :, current_service])
        destination.columns = ["destination_latency"]
        results[type_experiment + "_" + current_service] = pd.concat([a, source, destination], axis=1)


a = results["normal_metrics_carts"]
b = results["service_memory_carts"]
names = a.columns

plt.plot(a.loc[:, "ctn_memory"].values, label="normal")
plt.plot(b.loc[:, "ctn_memory"].values, label="memory leak")
plt.legend()

plt.figure(2)
plt.plot(a.loc[:, "source_latency"].values, label="normal")
plt.plot(b.loc[:, "source_latency"].values, label="memory leak")
plt.legend()

plt.figure(3)
plt.plot(a.loc[:, "destination_latency"].values, label="normal")
plt.plot(b.loc[:, "destination_latency"].values, label="memory leak")
plt.legend()