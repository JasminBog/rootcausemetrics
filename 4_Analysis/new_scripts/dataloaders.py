import torch
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler, TensorDataset


service_ = "orders"
experiment = "cpu"
experiments_feature_sets = {"cpu": [0, 2, 3, 5, 6, 7], "memory":[0, 2, 3, 5, 6, 7]}



path_train = "/home/matilda/PycharmProjects/RCA_metrics /3_Preprocessing_data/ready_data_training/" + service_ + "/normal_train.csv"
path_validation = "/home/matilda/PycharmProjects/RCA_metrics /3_Preprocessing_data/ready_data_training/" + service_ + "/normal_validation.csv"
path_test = "/home/matilda/PycharmProjects/RCA_metrics /3_Preprocessing_data/ready_data_training/" + service_ + "/" + experiment + "_test.csv"
path_labels = "/home/matilda/PycharmProjects/RCA_metrics /3_Preprocessing_data/ready_data_training/" + service_ + "/" + experiment + "_labels.csv"

batch_size = 512

features = list(pd.read_csv(path_train).columns[experiments_feature_sets[experiment]])
# print()
print(pd.read_csv(path_validation).columns[experiments_feature_sets[experiment]])

train = pd.read_csv(path_train).values
validation = pd.read_csv(path_validation).values
test = pd.read_csv(path_test).values
labels = pd.read_csv(path_labels).loc[:, "target"]

train = train[:, experiments_feature_sets[experiment]]
validation = validation[:, experiments_feature_sets[experiment]]
test_ = test[:, experiments_feature_sets[experiment]]

train_tensor = TensorDataset(torch.tensor(train, dtype=torch.float32))
train_sampler = RandomSampler(train_tensor)
train_dataloader = DataLoader(train_tensor, sampler=train_sampler, batch_size=batch_size)


validation_tensor = TensorDataset(torch.tensor(validation, dtype=torch.float32))
validation_sampler = SequentialSampler(validation_tensor)
validation_dataloader = DataLoader(validation_tensor, sampler=validation_sampler, batch_size=batch_size)

test_tensor = TensorDataset(torch.tensor(test_, dtype=torch.float32))
test_sampler = SequentialSampler(test_tensor)
test_dataloader = DataLoader(test_tensor, sampler=test_sampler, batch_size=batch_size)



pomaaaa = pd.read_csv(path_test)
plt.plot(pomaaaa.loc[:, "ctn_memory"])