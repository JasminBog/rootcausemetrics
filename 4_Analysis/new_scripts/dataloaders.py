import torch
import pandas as pd


# import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler, TensorDataset


service_ = "orders"
experiment = "memory"
# experiments_feature_sets = {"cpu": [0, 2, 3, 5, 6, 7], "memory":[0, 2, 3, 5, 6, 7]}

experiments_feature_sets = {"cpu": [0, 1, 2, 3, 5, 6, 7], "memory":[0, 1, 2, 3, 5, 6, 7]}

path_train = "/home/matilda/PycharmProjects/RCA_metrics /3_Preprocessing_data/ready_data_training/" + service_ + "/normal_train.csv"
path_validation = "/home/matilda/PycharmProjects/RCA_metrics /3_Preprocessing_data/ready_data_training/" + service_ + "/normal_validation.csv"
path_test = "/home/matilda/PycharmProjects/RCA_metrics /3_Preprocessing_data/ready_data_training/" + service_ + "/" + experiment + "_test.csv"
path_labels = "/home/matilda/PycharmProjects/RCA_metrics /3_Preprocessing_data/ready_data_training/" + service_ + "/" + experiment + "_labels.csv"

batch_size = 512

features = list(pd.read_csv(path_train).columns[experiments_feature_sets[experiment]])
# print()
print(pd.read_csv(path_validation).columns[experiments_feature_sets[experiment]])

train = pd.read_csv(path_train)

train = train.values
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




from collections import defaultdict
anom_indecies = defaultdict(list)
normal_indecies = []
key = 1
controlVar = False


for x in range(test_.shape[0]-2):

    if labels.values[x] == 1 and controlVar==False:
        controlVar = True

    if labels.values[x] == 1 and controlVar == True:
        anom_indecies[key].append(x)
    else:
        normal_indecies.append(x)

    if labels.values[x+1] == 0 and controlVar==True:
        controlVar = False
        key +=1

test_tensor = TensorDataset(torch.tensor(test_, dtype=torch.float32))
test_sampler = SequentialSampler(test_tensor)
test_dataloader = DataLoader(test_tensor, sampler=test_sampler, batch_size=batch_size)


test_tensor_1 = TensorDataset(torch.tensor(test_[anom_indecies[1]], dtype=torch.float32))
test_sampler_1 = SequentialSampler(test_tensor_1)
test_dataloader_1 = DataLoader(test_tensor_1, sampler=test_sampler_1, batch_size=batch_size)


test_tensor_2 = TensorDataset(torch.tensor(test_[anom_indecies[2]], dtype=torch.float32))
test_sampler_2 = SequentialSampler(test_tensor_2)
test_dataloader_2 = DataLoader(test_tensor_2, sampler=test_sampler_2, batch_size=batch_size)


test_tensor_3 = TensorDataset(torch.tensor(test_[anom_indecies[3]], dtype=torch.float32))
test_sampler_3 = SequentialSampler(test_tensor_3)
test_dataloader_3 = DataLoader(test_tensor_3, sampler=test_sampler_3, batch_size=batch_size)


test_tensor_4 = TensorDataset(torch.tensor(test_[anom_indecies[4]], dtype=torch.float32))
test_sampler_4 = SequentialSampler(test_tensor_4)
test_dataloader_4 = DataLoader(test_tensor_4, sampler=test_sampler_4, batch_size=batch_size)


test_tensor_5 = TensorDataset(torch.tensor(test_[anom_indecies[5]], dtype=torch.float32))
test_sampler_5 = SequentialSampler(test_tensor_5)
test_dataloader_5 = DataLoader(test_tensor_5, sampler=test_sampler_5, batch_size=batch_size)

test_tensor_6 = TensorDataset(torch.tensor(test_[anom_indecies[6]], dtype=torch.float32))
test_sampler_6 = SequentialSampler(test_tensor_6)
test_dataloader_6 = DataLoader(test_tensor_6, sampler=test_sampler_6, batch_size=batch_size)

test_loaders = [test_dataloader_1, test_dataloader_2, test_dataloader_3, test_dataloader_4, test_dataloader_5, test_dataloader_6]











# pomaaaa = pd.read_csv(path_test)
# plt.plot(pomaaaa.loc[:, "ctn_memory"])