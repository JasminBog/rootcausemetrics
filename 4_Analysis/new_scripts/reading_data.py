import os
import pandas as pd
import  numpy as np
import torch
import torch.functional as F
from torch.utils.data import TensorDataset, SequentialSampler, RandomSampler
import  matplotlib
matplotlib.use("TkAgg")
import seaborn as sns
import matplotlib.pyplot as plt

def get_normal_files_service(service, percentile, experiment, path):
    path = path + "/" + experiment
    files = os.listdir(path)
    latency_source_read = path + "/" + experiment + "_metrics_latency_agg_source_" + str(percentile) + ".csv"
    latency_destination_read = path + "/" + experiment + "_metrics_latency_agg_destination_" + str(percentile) + ".csv"
    a = [path + "/" +  x for x in files if service in x and "-db" not in x]
    a.append(latency_source_read)
    a.append(latency_destination_read)
    print(a)
    return a


def get_normal_files_service_2(service, percentile, experiment, path, other_service):
    path = path + "/" + experiment
    files = os.listdir(path)
    latency_source_read = path + "/" + experiment + "_metrics_latency_agg_source_" + str(percentile) + ".csv"
    latency_destination_read = path + "/" + experiment + "_metrics_latency_agg_destination_" + str(percentile) + ".csv"
    a = [path + "/" +  x for x in files if other_service in x and "-db" not in x]
    a.append(latency_source_read)
    a.append(latency_destination_read)
    print(a)
    return a




def read_normal_service(service, percentile, experiment, path):
    normal_data_carts = get_normal_files_service(service, percentile, experiment, path)
    print(normal_data_carts[0])
    service_data = pd.read_csv(normal_data_carts[0]).iloc[:, 1:]
    service_source_latency = pd.DataFrame(pd.read_csv(normal_data_carts[1]).iloc[:, 1:].loc[:, service])
    service_source_latency.columns = [service+ "_source_latency"]
    service_destination_latency = pd.DataFrame(pd.read_csv(normal_data_carts[2]).iloc[:, 1:].loc[:, service])
    service_destination_latency.columns = [service + "_destination_latency"]
    return pd.concat([service_data, service_source_latency, service_destination_latency], axis=1)


def read_normal_service_2(service, percentile, experiment, path, other_service):
    normal_data_carts = get_normal_files_service_2(service, percentile, experiment, path, other_service)
    service_data = pd.read_csv(normal_data_carts[0]).iloc[:, 1:]
    # print(abnormal_data_carts[1])
    if other_service == "front-end":
        aa = get_abnormal_files_service_front_end_specific(service, percentile, experiment, path, other_service)
        service_source_latency = pd.DataFrame(pd.read_csv(aa[1]).iloc[:, 1:].loc[:, other_service + "_" + service])
        service_source_latency.columns = [other_service+ "_source_latency"]
        service_source_latency = service_source_latency.fillna(service_source_latency.median())
    else:
        service_source_latency = pd.DataFrame(pd.read_csv(normal_data_carts[1]).iloc[:, 1:].loc[:, other_service])
        service_source_latency.columns = [other_service + "_source_latency"]
        service_source_latency = service_source_latency.fillna(service_source_latency.median())

    service_destination_latency = pd.DataFrame(pd.read_csv(normal_data_carts[2]).iloc[:, 1:].loc[:, other_service])
    service_destination_latency = service_destination_latency.fillna(service_destination_latency.median())
    service_destination_latency.columns = [other_service + "_destination_latency"]
    return pd.concat([service_data, service_source_latency, service_destination_latency], axis=1)


def get_abnormal_files_service(service, percentile, experiment, path):
    path = path + "/" + experiment + "/" + service
    files = os.listdir(path)
    latency_source_read = path + "/service_" + experiment  + "_" + service + "_latency_agg_source_" + str(percentile) + ".csv"
    latency_destination_read = path + "/service_" + experiment + "_" + service + "_latency_agg_destination_" + str(percentile) + ".csv"
    a = [path + "/" + x for x in files if service + ".csv" in x ]
    a.append(latency_source_read)
    a.append(latency_destination_read)
    return a

def get_abnormal_files_service_2(service, percentile, experiment, path, other_service):
    path = path + "/" + experiment + "/" + service
    files = os.listdir(path)
    latency_source_read = path + "/service_" + experiment  + "_" + service + "_latency_agg_source_" + str(percentile) + ".csv"
    latency_destination_read = path + "/service_" + experiment + "_" + service + "_latency_agg_destination_" + str(percentile) + ".csv"
    a = [path + "/" + x for x in files if other_service + ".csv" in x ]
    a.append(latency_source_read)
    a.append(latency_destination_read)
    return a


def get_abnormal_files_service_front_end_specific(service, percentile, experiment, path, other_service):
    path = path + "/" + experiment + "/" + service
    files = os.listdir(path)
    latency_source_read = path + "/service_" + experiment  + "_" + service + "_latency_source_" + str(percentile) + ".csv"
    latency_destination_read = path + "/service_" + experiment + "_" + service + "_latency_destination_" + str(percentile) + ".csv"
    a = [path + "/" + x for x in files if other_service + ".csv" in x ]
    a.append(latency_source_read)
    a.append(latency_destination_read)
    return a



def read_abnormal_service(service, percentile, experiment, path):
    abnormal_data_carts = get_abnormal_files_service(service, percentile, experiment, path)


    service_data = pd.read_csv(abnormal_data_carts[0]).iloc[:, 1:]
    service_source_latency = pd.DataFrame(pd.read_csv(abnormal_data_carts[1]).iloc[:, 1:].loc[:, service])
    service_source_latency.columns = [service+ "_source_latency"]
    service_source_latency = service_source_latency.fillna(service_source_latency.median())
    service_destination_latency = pd.DataFrame(pd.read_csv(abnormal_data_carts[2]).iloc[:, 1:].loc[:, service])
    service_destination_latency = service_destination_latency.fillna(service_destination_latency.median())
    service_destination_latency.columns = [service + "_destination_latency"]
    return pd.concat([service_data, service_source_latency, service_destination_latency], axis=1)


def read_abnormal_service_2(service, percentile, experiment, path, other_service):
    abnormal_data_carts = get_abnormal_files_service_2(service, percentile, experiment, path, other_service)


    service_data = pd.read_csv(abnormal_data_carts[0]).iloc[:, 1:]
    # print(abnormal_data_carts[1])
    if other_service == "front-end":
        aa = get_abnormal_files_service_front_end_specific(service, percentile, experiment, path, other_service)
        service_source_latency = pd.DataFrame(pd.read_csv(aa[1]).iloc[:, 1:].loc[:, other_service + "_" + service])
        service_source_latency.columns = [other_service+ "_source_latency"]
        service_source_latency = service_source_latency.fillna(service_source_latency.median())
    else:
        service_source_latency = pd.DataFrame(pd.read_csv(abnormal_data_carts[1]).iloc[:, 1:].loc[:, other_service])
        service_source_latency.columns = [other_service + "_source_latency"]
        service_source_latency = service_source_latency.fillna(service_source_latency.median())

    service_destination_latency = pd.DataFrame(pd.read_csv(abnormal_data_carts[2]).iloc[:, 1:].loc[:, other_service])
    service_destination_latency = service_destination_latency.fillna(service_destination_latency.median())
    service_destination_latency.columns = [other_service + "_destination_latency"]
    return pd.concat([service_data, service_source_latency, service_destination_latency], axis=1)


# for service in ["carts", "orders", "shipping", "user", "catalogue", "payment"]:
for service in ["carts", "orders", "shipping", "user", "catalogue", "payment"]:
    experiment = "memory" # memory, cpu

    path = "/home/matilda/PycharmProjects/RCA_metrics /2_Copy_Original_data/novel_data"
    # service = "shipping" # set one of carts, orders, shipping, user, catalogue, payment
    percentile = 99 # the percentile for response time
    service_to_store = "/home/matilda/PycharmProjects/RCA_metrics /3_Preprocessing_data/" + experiment + "/" + service + ".csv"


    # service_data = read_normal_service(service, percentile, experiment, path)
    # service_data.to_csv(service_to_store, index=False)
    #
    # abnormal_data = read_abnormal_service(service, percentile, experiment, path)
    # abnormal_data.to_csv(service_to_store, index=False)
    #


for service in ["carts", "orders", "user", "catalogue"]:
    for other_service in ["carts", "orders", "shipping", "user", "catalogue", "payment", "front-end"]:
        experiment = "memory" # memory, cpu

        path = "/home/matilda/PycharmProjects/RCA_metrics /2_Copy_Original_data/novel_data"
        # service = "shipping" # set one of carts, orders, shipping, user, catalogue, payment
        percentile = 99 # the percentile for response time
        service_to_store = "/home/matilda/PycharmProjects/RCA_metrics /3_Preprocessing_data/improve_MicroRCA/" + experiment + "/" + service + "/" + other_service +".csv"

        #
        # service_data = read_normal_service(service, percentile, experiment, path)
        # service_data.to_csv(service_to_store, index=False)

        abnormal_data = read_abnormal_service_2(service, percentile, experiment, path, other_service)
        abnormal_data.to_csv(service_to_store, index=False)

# plt.plot(service_data.loc[:, service+ "_source_latency"], label="source_latency", c="blue")
# plt.plot(service_data.loc[:, service+ "_destination_latency"], label="destination_latency", c="red")
# plt.legend()
# sns.pairplot(abnormal_data.iloc[:, 1:], kind="reg")
# plt.savefig("/home/matilda/PycharmProjects/RCA_metrics /5_Insights/" + experiment +  "_anomaly_pairwise_dependency/" + service + ".png")
# plt.show()
# plt.plot(abnormal_data.loc[:, service+ "_source_latency"], label="source_latency", c="blue")
# plt.plot(abnormal_data.loc[:, service+ "_destination_latency"], label="destination_latency", c="red")
# plt.title(service)
# plt.legend()
# plt.savefig("/home/matilda/PycharmProjects/RCA_metrics /5_Insights/anomaly_response_times/" + service + ".png")
