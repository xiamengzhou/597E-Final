import torch

class LRModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = torch.nn.Linear(1060, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        out = self.sigmoid(self.dense(x))
        return out



