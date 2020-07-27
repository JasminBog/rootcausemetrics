import copy
import pandas as pd
import numpy as np
import numpy
import torch
import matplotlib
matplotlib.use("TkAgg")

from datetime import datetime
from datetime import timedelta

import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, SequentialSampler, RandomSampler


number_anomalies_points = 7 # number of anomaly points into the seuqnece (plus one)
adjust_minutes_to_have_clear_data = 4 # minutes to filter normal and anomaly
select_fraction_train = 0.7
discard_first_n_samples_train = 60
discard_first_n_samples_test = 120
window_length = 10

service = "user"
experiment = "normal"
path = "/home/matilda/PycharmProjects/RCA_metrics /3_Preprocessing_data/" + experiment + "/" + service + ".csv"

store_path_normal_train = "/home/matilda/PycharmProjects/RCA_metrics /3_Preprocessing_data/ready_data_training/" + service + "/" + experiment + "_train.csv"
store_path_normal_validation = "/home/matilda/PycharmProjects/RCA_metrics /3_Preprocessing_data/ready_data_training/" + service + "/" + experiment + "_validation.csv"


class Service:
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.data = 0
        self.read_data()

    def read_data(self):
        self.data = pd.read_csv(self.path)


def filter_target(lista):
    pom = []
    for x in lista[1:]:
        s = x.replace("Fault start-point: ", "").replace("\n", "")
        f = "%a %b %d %H:%M:%S CEST %Y"
        out = datetime.strptime(s, f)
        pom.append(out)
    return pom


def read_target(service, experiment):
    path = "/home/matilda/PycharmProjects/RCA_metrics /2_Copy_Original_data/novel_data/"
    path = path + "log_filter_" + experiment + ".txt"
    with open(path, "r") as file:
        content = file.readlines()
        for idx, x in enumerate(content):
            if service in x:
                line = idx

        start_line = line
        end_line = line + number_anomalies_points
        return filter_target(content[start_line:end_line])

def create_targets(data_frame, ranges):
    indecies = []
    for idx, x in enumerate(ranges):
        start_time = x
        end_time = x + timedelta(minutes=adjust_minutes_to_have_clear_data)
        pom = data_frame[data_frame.loc[:, "timestamp"]>=start_time]
        pom = pom[pom.loc[:, "timestamp"]<end_time].index
        indecies.append(pom)
    v = []
    for j in indecies:
        for k in j:
            v.append(k)
    return np.array(v)

def noise_removal_train(dataset_csv):
    Q1 = dataset_csv.quantile(0.25); Q3 = dataset_csv.quantile(0.75); IQR = Q3 - Q1
    index_map = (dataset_csv > (Q1 - 1.5 * IQR)) | (dataset_csv < (Q3 + 1.5 * IQR))
    pom = [x for x in index_map.columns if x != "timestamp"]
    index_map = np.where(index_map.loc[:, pom].values.prod(axis=1)>0, True, False)
    return dataset_csv[index_map], (Q1, Q3, IQR)

def noise_removal_test(dataset_csv, Q1, Q3, IQR):
    index_map = (dataset_csv > (Q1 - 1.5 * IQR)) | (dataset_csv < (Q3 + 1.5 * IQR))
    pom = [x for x in index_map.columns if x != "timestamp"]
    index_map = np.where(index_map.loc[:, pom].values.prod(axis=1) > 0, True, False)
    return dataset_csv[index_map], index_map

def minmaxscale_train(dataset_csv):
    mms = MinMaxScaler()
    metrics = mms.fit_transform(dataset_csv)
    return mms, metrics

def minmaxscale_test(dataset_csv, mms):
    return mms.transform(dataset_csv)



def smooth(x, window_len=11, window='hanning'):
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
    if window_len < 3:
        return x
    s = np.r_[x[window_len - 1:0:-1], x, x[-1:-window_len - 1:-1]]
    print(s)
    print(s.shape)
    if window == 'flat':
        w = np.ones(window_len, 'd')
    else:
        w = eval('numpy.' + window + '(window_len)')
    y = np.convolve(w / w.sum(), s, mode='valid')
    return y



def smooth_time_series(data_frame, window_length):
    d = {}
    for feature in range(data_frame.shape[1]):
        d[feature] = smooth(data_frame[:, feature], window_len=window_length, window="hamming")
    return pd.DataFrame(d).values


dataset_normal = Service(path)
normal_data, IQRs = noise_removal_train(dataset_normal.data.iloc[discard_first_n_samples_train:, :])
preserve_col = [x for x in normal_data.columns if x != "timestamp"]
scaler, normal_data = minmaxscale_train(normal_data.loc[:, preserve_col])

train_index = int(normal_data.shape[0]*select_fraction_train)
dataset_normal_data_train = normal_data[:train_index, :]
dataset_normal_data_train = pd.DataFrame(dataset_normal_data_train)
dataset_normal_data_train.columns = preserve_col
dataset_normal_data_train.to_csv(store_path_normal_train, index=False)


dataset_normal_data_validation = normal_data[train_index:, :]
dataset_normal_data_validation = pd.DataFrame(dataset_normal_data_validation)
dataset_normal_data_validation.columns = preserve_col
dataset_normal_data_validation.to_csv(store_path_normal_validation, index=False)
# dataset_normal_data_validation = smooth_time_series(dataset_normal_data_validation, window_length=window_length)

experiment = "cpu"
path = "/home/matilda/PycharmProjects/RCA_metrics /3_Preprocessing_data/" + experiment + "/" + service + ".csv"

store_path_test_descritpion = "/home/matilda/PycharmProjects/RCA_metrics /3_Preprocessing_data/ready_data_training/" + service + "/" + experiment + "_test.csv"
store_path_test_labels = "/home/matilda/PycharmProjects/RCA_metrics /3_Preprocessing_data/ready_data_training/" + service + "/" + experiment + "_labels.csv"

dataset_abnormal = Service(path)
dataset_abnormal.data.timestamp = dataset_abnormal.data.timestamp.apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S") + timedelta(hours=2))
ranges_target = read_target(service, experiment)


abnormal_data, indecies = noise_removal_test(dataset_abnormal.data.iloc[:, 1:], IQRs[0], IQRs[1], IQRs[2])
targets = copy.copy(dataset_abnormal.data)
targets["target"] = np.zeros(targets.shape[0], dtype="int")
targets = targets.loc[indecies, :].loc[:, ["timestamp", "target"]]
ones_indecies = create_targets(targets, ranges_target)
targets.target.iloc[ones_indecies] = np.ones(ones_indecies.shape[0], dtype="int")


abnormal_data = minmaxscale_test(abnormal_data.loc[:, preserve_col], scaler)
# abnormal_data = abnormal_data[discard_first_n_samples_test:, :]
# abnormal_data = smooth_time_series(abnormal_data, window_length=window_length)
abnormal_data = abnormal_data[discard_first_n_samples_test:, :]
p1 = pd.DataFrame(abnormal_data)
p1.columns = preserve_col
p1.to_csv(store_path_test_descritpion, index=False)
targets = targets.iloc[discard_first_n_samples_test:, :]
targets.to_csv(store_path_test_labels, index=False)


# 1) Finish the data preprocessing for test data; Labeling of the test data; Segmentation
# 2) Implement train and test function (Explain to Lilly what is happening here and what are the most crucial parts and further details)
# 3) Implement evaluation function (Important to know how good are our models)
# 4) Implement the output inspection function (RCA)
# 5) Jasmin: Carts, orders; Lilly: User and Catalogue
# Jasmin: Granger ;


############# Investigate normal data #############
# service = "orders"
# anomaly = "normal"
# path = "/home/matilda/PycharmProjects/RCA_metrics /3_Preprocessing_data/" + anomaly + "/" + service + ".csv"
# dataset = Service(path)

# for idx, metric in enumerate(dataset.data.columns[1:-2]):
#     plt.figure(idx)
#     plt.plot(minmax_scale(dataset.data.loc[:, metric].values), label=metric)
#     plt.plot(minmax_scale(dataset.data.loc[:, service + "_source_latency"].values), label="source response time", alpha=0.2)
#     plt.plot(minmax_scale(dataset.data.loc[:, service + "_destination_latency"].values), label="destionation response time", alpha=0.2)
#     plt.title(metric)
#     plt.legend()
# # dataset.path = path

############# Investigate anomaly data #############
service = "orders"
anomaly = "cpu"
path = "/home/matilda/PycharmProjects/RCA_metrics /3_Preprocessing_data/" + anomaly + "/" + service + ".csv"
dataset = Service(path)
metric = "node_cpu"


value = 521
for idx, metric in enumerate(preserve_col):
    plt.subplot(value+idx)
    plt.plot(normal_data[:, idx], label=metric)
    plt.plot(normal_data[:, -2], label="source response time", alpha=0.2)
    plt.plot(normal_data[:, -1], label="destionation response time", alpha=0.2)
    plt.title(metric)
    plt.legend()

plt.figure(2)
# targets
for idx, metric in enumerate(preserve_col):
    plt.subplot(value+idx+1)
    plt.scatter(np.arange(0, abnormal_data[:, idx].shape[0]), abnormal_data[:, idx],  label=metric, c=np.where(targets.iloc[:, 1]==1, "red", "blue"))
    # plt.scatter(np.arange(0, abnormal_data[:, idx].shape[0]), abnormal_data[:, -2],  label=metric, c=np.where(targets.iloc[:, 1]==1, "red", "blue"), alpha=0.2)
    # plt.scatter(np.arange(0, abnormal_data[:, idx].shape[0]), abnormal_data[:, -1],  label=metric, c=np.where(targets.iloc[:, 1]==1, "red", "blue"), alpha=0.2)
    plt.title(metric)
    plt.legend()
# # dataset.path = path