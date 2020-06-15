from __future__ import print_function, division
#from twoStreamModel import *
from objectAttentionModelConvLSTM import *
#from flow_resnet import *
from spatial_transforms import (Compose, ToTensor, CenterCrop, Scale, Normalize, MultiScaleCornerCrop,
                                RandomHorizontalFlip)
from makeDataset import *
#from my_MakeDatasetFlow import *
#from makeDatasetFrame import *
import argparse
import sys
import time
import tqdm
import os

DEVICE = 'cuda'

#dict_path = "/content/gdrive/My Drive/FINAL_LOGS/two_stream_300flow_16frames/model_twoStream_state_dict.pth"
dict_path = "/content/gdrive/My Drive/Lorenzo/ego-rnn-two-in-one/results/rgb_16frames_two_in_one_no_ms/model_rgb_state_dict.pth"
#dict_path = "/content/gdrive/My Drive/FINAL_LOGS/200+150epochs_RGB_16frames/test_2/model_rgb_state_dict.pth"
#dict_path = "/content/gdrive/My Drive/Lorenzo/ego-rnn-ss-task/rgb_16frames_regression_kl/model_rgb_state_dict.pth"
#dict_path = "/content/gdrive/My Drive/FINAL_LOGS/flow300/model_flow_state_dict.pth"

normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
spatial_transform = Compose([Scale(256), CenterCrop(224), ToTensor(), normalize])

dataset = makeDataset("/content", ["S2"], spatial_transform=spatial_transform, seqLen=16, phase="train")

loader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                               shuffle=False, num_workers=2, pin_memory=True)

#model = twoStreamAttentionModel(flowModel="", frameModel="", stackSize=5, memSize=512, num_classes=61)
model = attentionModel(num_classes=61, mem_size=512)
#model = flow_resnet34(True, channels=2 * 5, num_classes=61)

model.load_state_dict(torch.load(dict_path), strict=True) 
model.to(DEVICE)
model.train(False)

valSamples = dataset.__len__()

feats_dict = {i:[] for i in range(61)}
preds = []
true = []

numCorr = 0


#two stream
'''
for j, (inputFrame, inputMMaps, inputFlow, target) in (enumerate(loader)):
    print(j)
    inputVariableFlow = Variable(inputFlow.to(DEVICE))
    inputVariableFrame = Variable(inputFrame.permute(1, 0, 2, 3, 4).to(DEVICE))
    labelVariable = Variable(target.to(DEVICE))
    feats, output_label = model(inputVariableFlow, inputVariableFrame)
    _, predicted = torch.max(output_label.data, 1)
    numCorr += (predicted == target.to(DEVICE)).sum()

    #feats_dict[target[0].item()].append(feats.cpu().tolist())

    preds.append(predicted.item())
    true.append(target.cpu().item())
'''

#two in one
for j, (inputFrame, inputMMaps, inputFlow, target) in (enumerate(loader)):
    print(j)
    #inputVariableFlow = Variable(inputFlow.to(DEVICE))
    inputFlow = inputFlow.view(
                (inputFlow.shape[0], int(inputFlow.shape[1] / 2), 2, inputFlow.shape[2], inputFlow.shape[3]))
    inputVariableFlow = Variable(
                inputFlow.permute(1, 0, 2, 3, 4).to(DEVICE))  # sequence length as first dimension

    inputVariableFrame = Variable(inputFrame.permute(1, 0, 2, 3, 4).to(DEVICE))
    labelVariable = Variable(target.to(DEVICE))
    output_label, feats, _ = model((inputVariableFrame, inputVariableFlow), 2)
    _, predicted = torch.max(output_label.data, 1)
    numCorr += (predicted == target.to(DEVICE)).sum()

    #feats_dict[target[0].item()].append(feats.cpu().tolist())

    preds.append(predicted.item())
    true.append(target.cpu().item())

#ms task
'''
for j, (inputs, inputsF, target) in enumerate(loader):
    print(j)

    inputVariable = Variable(inputs.permute(1, 0, 2, 3, 4).to(DEVICE))
    labelVariable = Variable(target.to(DEVICE))
    output_label, feats, _  = model(inputVariable, stage=2)

    _, predicted = torch.max(output_label.data, 1)
    numCorr += (predicted == target.to(DEVICE)).sum()  
    feats_dict[target[0].item()].append(feats.cpu().tolist())

    preds.append(predicted.item())
    true.append(target.cpu().item())
'''

#rgb
'''
for j, (inputs, inputsF, target) in enumerate(loader):
    print(j)

    inputVariable = Variable(inputs.permute(1, 0, 2, 3, 4).to(DEVICE))
    labelVariable = Variable(target.to(DEVICE))
    output_label, feats  = model(inputVariable)

    _, predicted = torch.max(output_label.data, 1)
    numCorr += (predicted == target.to(DEVICE)).sum()  
    feats_dict[target[0].item()].append(feats.cpu().tolist())
'''

#flow
'''
for j, (inputs, target) in enumerate(loader):
    inputVariable = Variable(inputs.to(DEVICE), volatile=True)
    labelVariable = Variable(target.to(DEVICE))
    output_label, feats = model(inputVariable)
    _, predicted = torch.max(output_label.data, 1)
    numCorr += (predicted == target.to(DEVICE)).sum()

    feats_dict[target[0].item()].append(feats.cpu().tolist())
'''

val_accuracy = (numCorr.data.item() / valSamples) * 100

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap

path = "/content/processed_frames/S2"
labels = sorted(os.listdir(path))

values = []
for val in feats_dict.values():
    values.extend(val)

values = [val[0] for val in values]

labs = []
for item in feats_dict.items():
    labs.append(labels[item[0]])
    for _ in range(len(item[1])-1):
        labs.append("")


df = pd.DataFrame(values).transpose()

plt.figure(figsize=(20, 20))
plt.matshow(df.corr())

tick_marks = [i for i in range(len(labs))]

plt.yticks(tick_marks, labs, rotation='horizontal')
plt.tick_params(axis='y', which='major', labelsize=2.0)
plt.tick_params(axis='x', which='major', labelsize=5.0)

plt.savefig("img.png", dpi=500)

N = 256
vals = np.ones((N, 4))


# colormaps

#vals[:, 0] = np.linspace(1, 240/256, N)
#vals[:, 1] = np.linspace(1, 164/256, N)
#vals[:, 2] = np.linspace(1, 94/256, N)

#vals[:, 0] = np.linspace(1, 39/256, N)
#vals[:, 1] = np.linspace(1, 70/256, N)
#vals[:, 2] = np.linspace(1, 83/256, N)

vals[:, 0] = np.linspace(1, 111/256, N)
vals[:, 1] = np.linspace(1, 83/256, N)
vals[:, 2] = np.linspace(1, 125/256, N)

newcmp = ListedColormap(vals)


plt.figure(figsize=(20, 20))
plt.matshow(confusion_matrix(true, preds, normalize="true"), cmap=newcmp)

tick_marks = [i for i in range(len(labels))]
numbered_labels = [label+" ["+str(n)+"]" for n, label in enumerate(labels)]
numbers = [f"[{i}]" for i in range(len(labels))]

plt.yticks(tick_marks, numbered_labels, rotation='horizontal')
plt.xticks(tick_marks, numbers, rotation='horizontal')

plt.tick_params(axis='y', which='major', labelsize=2.0)
plt.tick_params(axis='x', which='major', labelsize=2.0)
plt.grid(color='grey', linestyle='-', linewidth=0.2)


plt.savefig("img.png", dpi=500)
