import numpy as np
from torchvision import transforms
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
model_state_dict = 'results/test_2/model_rgb_state_dict.pth'  # Weights of the pre-trained model

model = attentionModel(num_classes=num_classes, mem_size=mem_size)
model.load_state_dict(torch.load(model_state_dict))
model_backbone = model.resNet
attentionMapModel = attentionMap(model_backbone).to(DEVICE)
attentionMapModel.train(False)
for params in attentionMapModel.parameters():
    params.requires_grad = False


normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

preprocess1 = transforms.Compose([
    transforms.Scale(256),
    transforms.CenterCrop(224),
])

preprocess2 = transforms.Compose([
    transforms.ToTensor(),
    normalize])


fl_name_in = './attention_test/frames/'
fl_name_out = './attention_test/attentioned/'

# load video and get frames #
'''
cam = cv2.VideoCapture("./attention_test/videotest.mp4")            # load video 

currentframe = 0
while (True):
    ret, frame = cam.read()
    if ret:
        name = fl_name_in + "/" + str(currentframe).zfill(3) + '.jpg'
        cv2.imwrite(name, frame)
        currentframe += 1
    else:
        break
'''


frames = []
for img_name in sorted(os.listdir(fl_name_in)):
    img_pil = Image.open(os.path.join(fl_name_in, img_name))                    # open image
    img_pil1 = preprocess1(img_pil)                                             # apply preprocessing (scaling and cropping)
    img_size = img_pil1.size
    size_upsample = (img_size[0], img_size[1])
    img_tensor = preprocess2(img_pil1)                                          # convert to tensor
    img_variable = Variable(img_tensor.unsqueeze(0).to(DEVICE))                 # send to gpu
    img = np.asarray(img_pil1)                                                  # keep original image (transformed)
    attentionMap_image = attentionMapModel(img_variable, img, size_upsample)    # compute attention map and stack it on original image
    cv2.imwrite(os.path.join(fl_name_out, img_name), attentionMap_image)        # save image

    frames.append(imageio.imread(os.path.join(fl_name_out, img_name)))          # keep image, read with imageio

kargs = {'duration': 0.05}                                                      # duration of single frame
imageio.mimsave(fl_name_out + "/frames.gif", frames, **kargs)                   # create gif
