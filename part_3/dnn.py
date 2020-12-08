import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dense1 = torch.nn.Linear(30, 1)
        self.dense2 = torch.nn.Linear(30, 30)
        self.dense3 = torch.nn.Linear(30, 30)
        self.dense4 = torch.nn.Linear(1060, 30)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        out = self.relu(self.dense4(x))
        out = self.relu(self.dense3(out))
        out = self.relu(self.dense2(out))
        out = self.sigmoid(self.dense1(out))
        return out








