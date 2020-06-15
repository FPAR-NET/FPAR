import numpy as np
#from torchvision import transforms
import spatial_transforms as transforms
import cv2
from objectAttentionModelConvLSTM import *
from attentionMapModel import attentionMap
from PIL import Image
import os
import imageio

DEVICE = "cuda"

# Model definition
num_classes = 61  # Classes in the pre-trained model
mem_size = 512

#model_state_dict = '/content/drive/My Drive/Lorenzo/ego-rnn-two-in-one/results/16frames_two_in_one/rgb_16frames_two_in_one_3/model_rgb_state_dict.pth'  # Weights of the pre-trained model
model_state_dict = "/content/drive/My Drive/Lorenzo/ego-rnn-two-in-one/results/rgb_16frames_two_in_one_conv1_unfreeze_3stages_kl/model_rgb_state_dict.pth"

model = attentionModel(num_classes=num_classes, mem_size=mem_size)
model.load_state_dict(torch.load(model_state_dict), strict=True)

model.train(False)
for params in model.parameters():
    params.requires_grad = False

model = model.to(DEVICE)

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

preprocess = transforms.Compose([
    transforms.Scale(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize])

in_path = "/content/drive/My Drive/Lorenzo/ego-rnn-two-in-one/get_modulation/frames"
out_path = "/content/drive/My Drive/Lorenzo/ego-rnn-two-in-one/get_modulation/out2"

inputFlow = []
inputFrame = []

frames = []
for i in range(1, len(os.listdir(in_path+"/rgb"))+1):
    print(i)

    # FLOW X
    fl_name = in_path + '/X/flow_x_' + str(int(round(i))).zfill(5) + '.png'  # zfill used to add leading zeros
    img = Image.open(fl_name)  # load single optical x frame
    inputFlow.append(preprocess(img.convert('L'), inv=False, flow=True))

    # FLOW Y
    fl_name = in_path + '/Y/flow_y_' + str(int(round(i))).zfill(5) + '.png'
    img = Image.open(fl_name)  # load single optical y frame
    inputFlow.append(preprocess(img.convert('L'), inv=False, flow=True))

    # RGB FRAME
    fl_name = in_path + '/rgb/' + 'rgb' + str(int(np.floor(i))).zfill(4) + ".png"
    img = Image.open(fl_name)  # load single rgb frame
    inputFrame.append(preprocess(img.convert('RGB')))


inputFlow = torch.stack(inputFlow, 0).squeeze(1)  # flow
inputFrame = torch.stack(inputFrame, 0)  # frame

inputFlow = inputFlow.unsqueeze(0)
inputFrame = inputFrame.unsqueeze(0)

inputFrame = Variable(inputFrame.permute(1, 0, 2, 3, 4).to(DEVICE))
    
inputFlow = inputFlow.view(
                    (inputFlow.shape[0], int(inputFlow.shape[1] / 2), 2, inputFlow.shape[2], inputFlow.shape[3]))
inputFlow = Variable(
                    inputFlow.permute(1, 0, 2, 3, 4).to(DEVICE))  # sequence length as first dimension

model((inputFrame, inputFlow), 3);

pre, post, beta, gamma = model.resNet.get_modulation_data()

def make_gif(path, out_path):
  kargs = {'duration': 0.05}                                   # duration of single frame

  frames=[]
  for frame in sorted(os.listdir(path)):
    frames.append(imageio.imread(os.path.join(path, frame)))

  imageio.mimsave(out_path, frames, **kargs)                   # create gif

import matplotlib.pyplot as plt

os.makedirs(out_path+"/gif/")

for channel in range(64):
  path = out_path + f"/{channel}"
  os.makedirs(path)
  os.makedirs(path+"/pre")
  os.makedirs(path+"/post")
  os.makedirs(path+"/beta")
  os.makedirs(path+"/gamma")

  print(channel)
  for frame in range(20):
    plt.imsave(path + f"/pre/{frame}.png", pre[frame][0][channel], cmap="Greys")
    plt.imsave(path + f"/post/{frame}.png", post[frame][0][channel], cmap="Greys")
    plt.imsave(path + f"/beta/{frame}.png", beta[frame][0][channel], cmap="Greys")
    plt.imsave(path + f"/gamma/{frame}.png", gamma[frame][0][channel], cmap="Greys")


  make_gif(path + f"/pre/", out_path+f"/gif/{channel}_pre.gif")
  make_gif(path + f"/post/", out_path+f"/gif/{channel}_post.gif")
  make_gif(path + f"/beta/", out_path+f"/gif/{channel}_beta.gif")
  make_gif(path + f"/gamma/", out_path+f"/gif/{channel}_gamma.gif")
