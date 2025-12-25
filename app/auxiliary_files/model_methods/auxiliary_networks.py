import torch.nn as nn


# for debugging
class PrintLayer(nn.Module):
    def __init__(self, name):
        super(PrintLayer, self).__init__()
        self.name = name

    def forward(self, x):
        print(self.name + ': -> ', x.shape)
        return x
