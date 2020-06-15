import torch
import torch.nn as nn
from torch.nn.functional import softmax, relu, sigmoid
import math

class MS_task(nn.Module):
    def __init__(self, input_ch=512):
        super(MS_task, self).__init__()
        self.conv = nn.Conv2d(input_ch, 100, kernel_size=1, padding=0)
        self.fc = (nn.Linear(7 * 7 * 100, 49))

    def forward(self, x):
        x = sigmoid(x)
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x