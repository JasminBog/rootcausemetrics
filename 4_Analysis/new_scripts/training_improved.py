import torch.nn as nn
import numpy as np
import torch.optim as optim
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as statistic

from dataloaders_improved import *
from autoencoder_networks_improved import AutoEncoder


def weights_init(m):
    if isinstance(m, nn.Conv2d):
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
        l1_regularization  = torch.tensor(0.0)
        lambda_ = 0.6
        for param in model.parameters():
            l1_regularization += torch.norm(param, 1) ** 2

        error = criterion(x_1, output)
        tr_loss += error.item() + lambda_*l1_regularization
        # tr_loss += error.item()
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

def calculate_gaussian_properties(validation, prediction):
    mean = np.mean(validation, axis=0)
    std  = np.std(validation)
    vector_probs = 1 - statistic.norm.cdf(prediction, loc=mean, scale=std)
    plt.figure(10)
    plt.scatter(np.arange(len(vector_probs)), vector_probs)
    plt.xlabel("test points")
    plt.ylabel("probability estimates")
    return vector_probs


def evaluate():
    return 0

input_dimension = train.shape[1]
hidden_dimension = 3
number_epochs = 2000
learning_rate = 0.0001


model = AutoEncoder(input_size=input_dimension, hidden_size=hidden_dimension)
model.apply(weights_init)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.99), lr=learning_rate)

# Train model
errors, model = train_whole(number_epochs, train_dataloader, model)


# Evaluate model
predictions_validation, real_values_validation = test_preds(model, validation_dataloader)
predictions_test, real_values_test = test_preds(model, test_dataloader_1)


## CALCULATE ERRORS
error_validation = np.power(np.array(predictions_validation) - np.array(real_values_validation), 2).mean(axis=1)
error_test = np.power(np.array(predictions_test) - np.array(real_values_test), 2).mean(axis=1)


probabilities_normal = calculate_gaussian_properties(error_validation, error_test)

if np.where(probabilities_normal >= np.mean(probabilities_normal), 1, 0).sum() >= len(probabilities_normal)//2:
    print("NORMAL")
else:
    print("ANOMALY")

plt.figure(11)
# plt_figure_res = "/home/matilda/PycharmProjects/RCA_metrics /5_Insights/results/" + experiment + "_" + service_ + "_results_validation_test.png"
plt.subplot(121)
validation = error_validation
plt.hist(validation,  bins=100)
plt.title("VALIDATION")
plt.subplot(122)
test = error_test
plt.hist(test, bins=100)
plt.title("TEST")
# plt.savefig(plt_figure_res)
# plt.plot(test_[:, 0])
# plt.plot(test)

plt.figure(figsize=(15, 5))
sns.heatmap(np.power(np.array(predictions_test) - np.array(real_values_test), 2).T)
plt.yticks(np.arange(len(features)), features, rotation=0)


# figure_name = "/home/matilda/PycharmProjects/RCA_metrics /5_Insights/results/" + experiment + "_" + service_ + ".png"
# plt.savefig(figure_name)