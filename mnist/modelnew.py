import os.path

import torch
import torch.nn as nn
from layersnew import *
import torch.nn.functional as F
import re
import torch.utils.checkpoint as cp
from collections import OrderedDict
from torch.hub import load_state_dict_from_url
from torch import Tensor
from typing import Any, List, Tuple
import xlsxwriter
import openpyxl

class Net(nn.Module):
    def  __init__(
            self,
            args=None,
    ):
        super(Net, self).__init__()

        self.drop_rate = args.drop_rate
        self.device = args.device

        self.conv1 = tdConv(nn.Conv2d(1, 256, kernel_size=3, padding=1, bias=False))
        self.BN1 = tdNorm(newBatchNorm(256))

        self.spike1 = LIFSpike()
        self.conv2 = tdConv(nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False))
        self.BN2 = tdNorm(newBatchNorm(256))

        self.spike2 = LIFSpike()
        self.conv3 = tdConv(nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False))
        self.BN3 = tdNorm(newBatchNorm(256))

        self.spike3 = LIFSpike()
        self.pool1 = tdConv(nn.AvgPool2d(kernel_size=2, stride=2))
        # self.pool1 = tdConv(nn.MaxPool2d(kernel_size=2, stride=2))

        self.conv4 = tdConv(nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False))
        self.BN4 = tdNorm(newBatchNorm(256))

        self.spike4 = LIFSpike()
        self.conv5 = tdConv(nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False))
        self.BN5 = tdNorm(newBatchNorm(256))

        self.spike5 = LIFSpike()
        self.conv6 = tdConv(nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False))
        self.BN6 = tdNorm(newBatchNorm(256))

        self.spike6 = LIFSpike()
        self.pool2 = tdConv(nn.AvgPool2d(kernel_size=2, stride=2))

        self.fc1 = tdConv(nn.Linear(256 * 7 * 7, 128 * 4 * 4, bias=False))
        self.fc2 = tdConv(nn.Linear(128 * 4 * 4, 10, bias=False))
        self.fcspike1 = LIFOutSpike()
        self.fcspike2 = LIFOutSpike()

        self.lateralfc1 = tdConv(nn.Conv1d(1, 1, kernel_size=5, stride=1, padding=2, groups=1, bias=False))
        self.lateralfc2 = tdConv(nn.Conv1d(1, 1, kernel_size=5, stride=1, padding=2, groups=1, bias=False))

    def dropout_layer(self, x):
        d = torch.mean(x, dim=2)
        p = (d - d.min()) / (d.max() - d.min())
        p = 1 - (1 - p) * self.drop_rate / (1 - p.mean())
        p = torch.clamp(p, min=0., max=1.)
        d = torch.bernoulli(p)
        d = d.div(torch.where(p > 0, p, torch.tensor(1.).to(x.device)))
        d = torch.stack((d,) * steps, 2).to(x.device)
        return x * d

    def forward(self, x: Tensor, batch_idx = 1, epoch = 1):
        x = self.conv1(x)
        x = self.BN1(x)
        x = self.spike1(x)
        x = self.conv2(x)
        x = self.BN2(x)
        x = self.spike2(x)
        x = self.conv3(x)
        x = self.BN3(x)
        x = self.spike3(x)
        x = self.pool1(x)
        x = self.conv4(x)
        x = self.BN4(x)
        x = self.spike4(x)
        x = self.conv5(x)
        x = self.BN5(x)
        x = self.spike5(x)
        x = self.conv6(x)
        x = self.BN6(x)
        x = self.spike6(x)
        x = self.pool2(x)
        x = x.view(x.shape[0], -1, x.shape[4])

        if self.training:
            x = torch.cat((x, x), 0)
        x = self.fc1(x)
        x = self.fcspike1(x, self.lateralfc1(x.unsqueeze(1)).squeeze(1))
        if self.drop_rate > 0 and self.training:
            x = self.dropout_layer(x)
        x = self.fc2(x)
        x = self.fcspike2(x, self.lateralfc2(x.unsqueeze(1)).squeeze(1))
        out = torch.sum(x, dim=2) / steps
        if self.training:
            return out[:len(out) // 2], out[len(out) // 2:]
        else:
            return out