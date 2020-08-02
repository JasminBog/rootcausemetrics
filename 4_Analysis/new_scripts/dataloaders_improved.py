import numpy as np
import torch
import pandas as pd


# import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler, TensorDataset


service_ = "catalogue"
experiment = "memory"
other_service = "catalogue"
# experiments_feature_sets = {"cpu": [0, 2, 3, 5, 6, 7], "memory":[0, 2, 3, 5, 6, 7]}

experiments_feature_sets = {"cpu": [0, 1, 2, 3, 5, 6, 7], "memory":[0, 1, 2, 3, 5, 6, 7]}

# path_train = "/home/matilda/PycharmProjects/RCA_metrics /3_Preprocessing_data/ready_imrpovedMicroRCA/"+ experiment + "/" + service_ + "/" + other_service + "_train.csv"
# path_validation = "/home/matilda/PycharmProjects/RCA_metrics /3_Preprocessing_data/ready_imrpovedMicroRCA/"+ experiment + "/" + service_ + "/" + other_service + "_valid.csv"

path_train = "/home/matilda/PycharmProjects/RCA_metrics /3_Preprocessing_data/ready_data_training/" + service_ + "/normal_train.csv"
path_validation = "/home/matilda/PycharmProjects/RCA_metrics /3_Preprocessing_data/ready_data_training/" + service_ + "/normal_validation.csv"

path_test_1 = "/home/matilda/PycharmProjects/RCA_metrics /3_Preprocessing_data/ready_imrpovedMicroRCA/" + experiment + "/" + service_ + "/" + other_service + "_test_1.csv"
path_labels_1 = "/home/matilda/PycharmProjects/RCA_metrics /3_Preprocessing_data/ready_imrpovedMicroRCA/" + experiment + "/" + service_ + "/" + other_service + "_labels_1.csv"
path_test_2 = "/home/matilda/PycharmProjects/RCA_metrics /3_Preprocessing_data/ready_imrpovedMicroRCA/" + experiment + "/" + service_ + "/" + other_service + "_test_2.csv"
path_labels_2 = "/home/matilda/PycharmProjects/RCA_metrics /3_Preprocessing_data/ready_imrpovedMicroRCA/" + experiment + "/" + service_ + "/" + other_service + "_labels_2.csv"
path_test_3 = "/home/matilda/PycharmProjects/RCA_metrics /3_Preprocessing_data/ready_imrpovedMicroRCA/" + experiment + "/" + service_ + "/" + other_service + "_test_3.csv"
path_labels_3 = "/home/matilda/PycharmProjects/RCA_metrics /3_Preprocessing_data/ready_imrpovedMicroRCA/" + experiment + "/" + service_ + "/" + other_service + "_labels_3.csv"
path_test_4 = "/home/matilda/PycharmProjects/RCA_metrics /3_Preprocessing_data/ready_imrpovedMicroRCA/" + experiment + "/" + service_ + "/" + other_service + "_test_4.csv"
path_labels_4 = "/home/matilda/PycharmProjects/RCA_metrics /3_Preprocessing_data/ready_imrpovedMicroRCA/" + experiment + "/" + service_ + "/" + other_service + "_labels_4.csv"
path_test_5 = "/home/matilda/PycharmProjects/RCA_metrics /3_Preprocessing_data/ready_imrpovedMicroRCA/" + experiment + "/" + service_ + "/" + other_service + "_test_5.csv"
path_labels_5 = "/home/matilda/PycharmProjects/RCA_metrics /3_Preprocessing_data/ready_imrpovedMicroRCA/" + experiment + "/" + service_ + "/" + other_service + "_labels_5.csv"
path_test_6 = "/home/matilda/PycharmProjects/RCA_metrics /3_Preprocessing_data/ready_imrpovedMicroRCA/" + experiment + "/" + service_ + "/" + other_service + "_test_6.csv"
path_labels_6 = "/home/matilda/PycharmProjects/RCA_metrics /3_Preprocessing_data/ready_imrpovedMicroRCA/" + experiment + "/" + service_ + "/" + other_service + "_labels_6.csv"



batch_size = 2048

features = list(pd.read_csv(path_train).columns[experiments_feature_sets[experiment]])
# print()
print(pd.read_csv(path_validation).columns[experiments_feature_sets[experiment]])

train = pd.read_csv(path_train)



train = train.values
validation = pd.read_csv(path_validation).values

test1 = pd.read_csv(path_test_1).values
labels1 = pd.read_csv(path_labels_1).loc[:, "target"]
test2 = pd.read_csv(path_test_2).values
labels2 = pd.read_csv(path_labels_2).loc[:, "target"]
test3 = pd.read_csv(path_test_3).values
labels3 = pd.read_csv(path_labels_3).loc[:, "target"]
test4 = pd.read_csv(path_test_4).values
labels4 = pd.read_csv(path_labels_4).loc[:, "target"]
test5 = pd.read_csv(path_test_5).values
labels5 = pd.read_csv(path_labels_5).loc[:, "target"]
test6 = pd.read_csv(path_test_6).values
labels6 = pd.read_csv(path_labels_6).loc[:, "target"]


path_test_normal = "/home/matilda/PycharmProjects/RCA_metrics /3_Preprocessing_data/ready_imrpovedMicroRCA/"+ experiment + "/" + service_ + "/" + other_service + "_train.csv"


test_normal = pd.read_csv(path_test_normal).values
labels_normal = pd.read_csv(path_labels_6).loc[:, "target"]
labels_normal = labels_normal - labels_normal



train = train[:, experiments_feature_sets[experiment]]
validation = validation[:, experiments_feature_sets[experiment]]
test_1 = test1[:, experiments_feature_sets[experiment]]
test_2 = test2[:, experiments_feature_sets[experiment]]
test_3 = test3[:, experiments_feature_sets[experiment]]
test_4 = test4[:, experiments_feature_sets[experiment]]
test_5 = test5[:, experiments_feature_sets[experiment]]
test_6 = test6[:, experiments_feature_sets[experiment]]
test_normal = test_normal[:, experiments_feature_sets[experiment]]


train_tensor = TensorDataset(torch.tensor(train, dtype=torch.float32))
train_sampler = RandomSampler(train_tensor)
train_dataloader = DataLoader(train_tensor, sampler=train_sampler, batch_size=batch_size)

validation_tensor = TensorDataset(torch.tensor(validation, dtype=torch.float32))
validation_sampler = SequentialSampler(validation_tensor)
validation_dataloader = DataLoader(validation_tensor, sampler=validation_sampler, batch_size=batch_size)


test_normal = TensorDataset(torch.tensor(test_normal, dtype=torch.float32))
test_normal_sam = SequentialSampler(test_normal)
test_normal_dataloader = DataLoader(test_normal, sampler=test_normal_sam, batch_size=batch_size)


test_tensor_1 = TensorDataset(torch.tensor(test_1, dtype=torch.float32))
test_sampler_1 = SequentialSampler(test_tensor_1)
test_dataloader_1 = DataLoader(test_tensor_1, sampler=test_sampler_1, batch_size=batch_size)

test_tensor_2 = TensorDataset(torch.tensor(test_2, dtype=torch.float32))
test_sampler_2 = SequentialSampler(test_tensor_2)
test_dataloader_2 = DataLoader(test_tensor_2, sampler=test_sampler_2, batch_size=batch_size)

test_tensor_3 = TensorDataset(torch.tensor(test_3, dtype=torch.float32))
test_sampler_3 = SequentialSampler(test_tensor_3)
test_dataloader_3 = DataLoader(test_tensor_3, sampler=test_sampler_3, batch_size=batch_size)

test_tensor_4 = TensorDataset(torch.tensor(test_4, dtype=torch.float32))
test_sampler_4 = SequentialSampler(test_tensor_4)
test_dataloader_4 = DataLoader(test_tensor_4, sampler=test_sampler_4, batch_size=batch_size)



test_tensor_5 = TensorDataset(torch.tensor(test_5, dtype=torch.float32))
test_sampler_5 = SequentialSampler(test_tensor_5)
test_dataloader_5 = DataLoader(test_tensor_5, sampler=test_sampler_5, batch_size=batch_size)


test_tensor_6 = TensorDataset(torch.tensor(test_6, dtype=torch.float32))
test_sampler_6 = SequentialSampler(test_tensor_6)
test_dataloader_6 = DataLoader(test_tensor_6, sampler=test_sampler_6, batch_size=batch_size)





# pomaaaa = pd.read_csv(path_test)
# plt.plot(pomaaaa.loc[:, "ctn_memory"])