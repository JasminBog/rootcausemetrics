import torch.nn as nn
import numpy as np
import torch.optim as optim
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import seaborn as sns

from dataloaders import *
from autoencoder_networks_improved import AutoEncoder

def one_epoch_train(train_dataloader, model):
    error = []
    tr_loss = 0
    number_samples = 0
    for idx, batch in enumerate(train_dataloader):
        x_1 = batch[0]
        x_1.to("cuda")
        optimizer.zero_grad()
        output = model(x_1)
        error = criterion(x_1, output)
        tr_loss += error.item()
        error.backward()
        optimizer.step()
        number_samples += batch[0].size(0)
    print(error.detach())
    return error/number_samples, model

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

def evaluate():
    return 0

input_dimension = train.shape[1]
hidden_dimension = 3
number_epochs = 1000
learning_rate = 0.0001


model = AutoEncoder(input_size=input_dimension, hidden_size=hidden_dimension)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.99), lr=learning_rate)


errors, model = train_whole(number_epochs, train_dataloader, model)
predictions_validation, real_values_validation = test_preds(model, validation_dataloader)


indecies = []
dictionart = defaultdict()

for idx, loader in enumerate(test_loaders):
    plt.figure(idx)
    predictions_test, real_values_test = test_preds(model, loader)

    q = [features[x] for x in np.argsort(np.power(np.array(predictions_test) - np.array(real_values_test), 2).T.max(axis=1))]
    z = []
    for t in np.arange(len(q)-1, -1, -1):
        z.append(q[t])
        if "ctn_" + experiment == q[t]:
            ind = t

    dictionart[idx] = z
    indecies.append(np.abs(ind-len(q)))

    # # plt_figure_res = "/home/matilda/PycharmProjects/RCA_metrics /5_Insights/results/" + experiment + "_" + service_ + "_results_validation_test.png"
    # plt.subplot(121)
    # validation = np.power(np.array(predictions_validation) - np.array(real_values_validation), 2).mean(axis=1)
    # plt.hist(validation,  bins=100)
    # plt.title("VALIDATION")
    # plt.subplot(122)
    # test = np.power(np.array(predictions_test) - np.array(real_values_test), 2).mean(axis=1)
    # plt.hist(test, bins=100)
    # plt.title("TEST")
    # # plt.savefig(plt_figure_res)
    # # plt.plot(test_[:, 0])
    # # plt.plot(test)
    #
    # plt.figure(figsize=(15, 5))
    sns.heatmap(np.power(np.array(predictions_test) - np.array(real_values_test), 2).T)
    plt.yticks(np.arange(len(features)), features, rotation=0)

import pickle

d = {}

with open("/home/matilda/PycharmProjects/RCA_metrics /5_Insights/store_results/"+service_ + "_" +experiment + ".pickle", "wb") as file:
    d["service_name"] = service_
    d["experiment"] = experiment
    d["error"] = dictionart
    d["indecies"] = indecies
    pickle.dump(d, file)
# figure_name = "/home/matilda/PycharmProjects/RCA_metrics /5_Insights/results/" + experiment + "_" + service_ + ".png"
# plt.savefig(figure_name)