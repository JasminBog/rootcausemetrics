import numpy as np
import torch
import pandas as pd

import pickle
import torch.nn as nn
import numpy as np
import torch.optim as optim
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as statistic

from autoencoder_networks_improved import AutoEncoder


# import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler, TensorDataset

from collections import defaultdict


def weights_init(m):
    torch.nn.init.xavier_uniform(m.weight.data)
    torch.nn.init.xavier_uniform(m.bias.data)


def one_epoch_train(train_dataloader, model):
    error = []
    tr_loss = 0
    number_samples = 0
    for idx, batch in enumerate(train_dataloader):
        x_1 = batch[0]
        x_1.to("cuda")
        optimizer.zero_grad()
        output = model(x_1)
        l1_regularization = torch.tensor(0.0)
        lambda_ = 0.6
        for param in model.parameters():
            l1_regularization += torch.norm(param, 1) ** 2

        error = criterion(x_1, output)
        tr_loss += error.item() + lambda_ * l1_regularization
        # tr_loss += error.item()
        error.backward()
        optimizer.step()
        number_samples += batch[0].size(0)
    print(error.detach())
    return error / number_samples, model


def train_whole(number_epochs, train_dataloader, model):
    for epoch in range(number_epochs):
        error, model = one_epoch_train(train_dataloader, model)
        print("Finish epoch: {} with error".format(epoch, error))
    return error, model


def test_preds(model, test_dataloader):
    predictions = []
    real_values = []
    for idx, batch in enumerate(test_dataloader):
        x = batch[0]
        x.to("cuda")
        with torch.no_grad():
            output = model(x)
            predictions += list(output.cpu().numpy())
            real_values += list(x.cpu().numpy())
    return predictions, real_values


def calculate_gaussian_properties(validation, prediction):
    mean = np.mean(validation, axis=0)
    std = np.std(validation)
    vector_probs = 1 - statistic.norm.cdf(prediction, loc=mean, scale=std)
    plt.figure(10)
    plt.scatter(np.arange(len(vector_probs)), vector_probs)
    plt.xlabel("test points")
    plt.ylabel("probability estimates")
    return vector_probs


def evaluate():
    return 0


path_microrca = "/home/matilda/PycharmProjects/RCA_metrics /3_Preprocessing_data/MicroRCA_0.4_0.035.csv"
micro_rca = pd.read_csv(path_microrca, header=None)
hidden_dimension = 3
number_epochs = 2000
learning_rate = 0.0001

batch_size = 2048


import os
finished_results = os.listdir("/home/matilda/PycharmProjects/RCA_metrics /5_Insights/store_results_experiments2/")

rrr = []
ots = ["carts", "orders", "shipping", "user", "catalogue", "payment", "front-end"]
for indeksot in range(micro_rca.shape[0]):

    anomaly_id = micro_rca.iloc[indeksot, 0]
    service_ = micro_rca.iloc[indeksot, 1]
    if "cpu" in micro_rca.iloc[indeksot, 2]:
        experiment = "cpu"
    else:
        experiment = "memory"


    lista_better = eval(micro_rca.iloc[indeksot, 4])

    for other_service in lista_better:



        other_service = other_service[0]

        if service_ + "_" + experiment + "_" + str(anomaly_id) + "_" + other_service + "_.pickle" in finished_results:
            print(service_ + "_" + experiment + "_" + str(anomaly_id) + "_" + other_service + "_.pickle")
            break


        # s_feature_sets = {"cpu": [0, 2, 3, 5, 6, 7], "memory":[0, 2, 3, 5, 6, 7]}
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


        features = list(pd.read_csv(path_train).columns[experiments_feature_sets[experiment]])


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

        test_loaders = [test_dataloader_1, test_dataloader_2, test_dataloader_3, test_dataloader_4, test_dataloader_5, test_dataloader_6]


        input_dimension = train.shape[1]

        model = AutoEncoder(input_size=input_dimension, hidden_size=hidden_dimension)
        # model.apply(weights_init)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.99), lr=learning_rate)

        # Train model
        errors, model = train_whole(number_epochs, train_dataloader, model)


        indecies = []
        dictionart = defaultdict()

        predictions_validation, real_values_validation = test_preds(model, validation_dataloader)

        error_validation = np.power(np.array(predictions_validation) - np.array(real_values_validation), 2).mean(
            axis=1)

        # Evaluate model

        predictions_test, real_values_test = test_preds(model, test_loaders[anomaly_id])

        error_test = np.power(np.array(predictions_test) - np.array(real_values_test), 2).mean(axis=1)


        probabilities_normal = calculate_gaussian_properties(error_validation, error_test)

        decision = ""
        if np.where(probabilities_normal >= 0.5, 1, 0).sum() >= len(probabilities_normal) // 2:
            decision = "NORMAL"
        else:
            decision = "ANOMALY"

        indecies = []
        dictionart = defaultdict()

        q = [features[x] for x in np.argsort(np.power(np.array(predictions_test) - np.array(real_values_test), 2).T.max(axis=1))]
        z = []

        for t in np.arange(len(q) - 1, -1, -1):
            z.append(q[t])
            if "ctn_" + experiment == q[t]:
                ind = t

        d = {}

        with open("/home/matilda/PycharmProjects/RCA_metrics /5_Insights/store_results_experiments2/"+service_ + "_" +experiment + "_" + str(anomaly_id) + "_" + other_service + "_.pickle", "wb") as file:
            d["service_name"] = service_
            d["experiment"] = experiment
            d["other_service"] = other_service
            d["decison"] = decision
            d["indecies"]  = np.abs(ind - len(q))
            d["error"] = z
            d["score"] = np.mean(probabilities_normal)
            pickle.dump(d, file)

        rrr.append(d)