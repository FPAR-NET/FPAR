import torch
import resnetMod
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from MyConvLSTMCell import *
from MS_task import *

DEVICE = "cuda"

class attentionModel(nn.Module):
    def __init__(self, num_classes=61, mem_size=512):

        super(attentionModel, self).__init__()
        self.num_classes = num_classes
        self.resNet = resnetMod.resnet34(True, True)
        self.mem_size = mem_size
        self.weight_softmax = self.resNet.fc.weight
        self.ms_task = MS_task(512)
        self.lstm_cell = MyConvLSTMCell(512, mem_size)
        self.avgpool = nn.AvgPool2d(7)
        self.dropout = nn.Dropout(0.7)  
        self.fc = nn.Linear(mem_size, self.num_classes)
        self.classifier = nn.Sequential(self.dropout, self.fc)

    def forward(self, inputVariable, stage):
        feats_ms = []

        
        inputFrameVariable, inputFlowVariable = inputVariable

        state = (Variable(torch.zeros((inputFrameVariable.size(1), self.mem_size, 7, 7)).to(DEVICE)),
                 Variable(torch.zeros((inputFrameVariable.size(1), self.mem_size, 7, 7)).to(DEVICE)))

        for t in range(inputFrameVariable.size(0)):
            # backbone resnet get as input both rgb and flow data
            logit, feature_conv, feature_convNBN = self.resNet((inputFrameVariable[t], inputFlowVariable[t]))
            bz, nc, h, w = feature_conv.size()
            feature_conv1 = feature_conv.view(bz, nc, h * w)

            if stage == 2:
                feats_ms.append(self.ms_task(feature_conv.view(bz, nc, h, w)))  # features into ms_task (7x7x512)

            probs, idxs = logit.sort(1, True)
            class_idx = idxs[:, 0]
            cam = torch.bmm(self.weight_softmax[class_idx].unsqueeze(1), feature_conv1)
            attentionMAP = F.softmax(cam.squeeze(1), dim=1)
            attentionMAP = attentionMAP.view(attentionMAP.size(0), 1, 7, 7)
            attentionFeat = feature_convNBN * attentionMAP.expand_as(
                feature_conv)  # the features coming from the layer4 are weighted by the attention map


            state = self.lstm_cell(attentionFeat, state)

        feats1 = self.avgpool(state[1]).view(state[1].size(0), -1)
        feats = self.classifier(feats1)

        if stage == 2:
            feats_ms = torch.stack(feats_ms, 0).permute(1, 0, 2)
            feats_ms = feats_ms.view(bz, inputFrameVariable.size(0), h * w) 

        return feats, feats1, feats_ms
