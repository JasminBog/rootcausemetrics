import torch.nn
import torch.nn.functional as F
import torch.optim as optim


class AutoEncoder(torch.nn):
    def __init__(self, nn, input_size, hidden_size):
        super.__init__(self)
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        x = F.tanh(self.linear1(x))
        x = F.softmax(self.linear2(x))
        x = F.tanh(self.linear3(x))
        return x
