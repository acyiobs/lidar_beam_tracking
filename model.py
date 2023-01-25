import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class GruModel(nn.Module):
    def __init__(self, num_classes, num_layers=1, hidden_size=64, embed_size=64):
        super(GruModel, self).__init__()
        self.embed = torch.nn.Linear(216, embed_size)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.gru = torch.nn.GRU(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers, dropout=0.8)
        self.fc = torch.nn.Linear(hidden_size, num_classes)
        self.name = 'GruModel'
        self.dropout1 = nn.Dropout(0.5)

    def initHidden(self, batch_size):
        return torch.zeros((self.num_layers, batch_size, self.hidden_size))

    def forward(self, x, h):
        y = self.embed(x)
        y = self.dropout1(y)
        y, h = self.gru(y, h)
        y = self.fc(y)
        return y, h

