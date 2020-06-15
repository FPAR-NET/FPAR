import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

# Attention map based on layer4 weights of the most probable class
# The output from the layer 4 (for a batch of 1) is (1 x 512 x 7 x 7) 

class attentionMap(nn.Module):
    def __init__(self, backbone):
        super(attentionMap, self).__init__()
        self.backbone = backbone
        self.backbone.train(False)
        self.params = list(self.backbone.parameters())
        self.weight_softmax = self.params[-2]                   # take layer4 weights
        
    def forward(self, img_variable, img, size_upsample):
        logit, feature_conv, _ = self.backbone(img_variable)    # take label prediction

        bz, nc, h, w = feature_conv.size()                      # get the layer 4 output size
        feature_conv = feature_conv.view(bz, nc, h*w)           # flatten the 7 x 7 to 49

        h_x = F.softmax(logit, dim=1).data                      # softmax on logits (results of last dense layer)
        probs, idx = h_x.sort(1, True)                          # sort according to probability

        # torch.bmm is a batch matrix multiplication, it takes tensor1 (b×n×m) tensor2(b×m×p) and produce (b×n×p) where b is the batch size

        # self.weight_softmax are the weights coming from layer4 to the last layer (avgpooling has no weights)
        # self.weight_softmax[idx[:, 0]] selects only the weights going to the node with higher probability

        # now we need to multiply the weights with the results of the layer4, to do so we use bmm that multiplies batches of tensors
        # so if the batch is 32 it means that feature_conv is a (32 x 512 x 49) and self.weight_softmax[idx[:, 0]] is (32 x 512)
        # so we need to reshape self.weight_softmax[idx[:, 0]] to get a (32 x 512 x 1) so  we use unsqueeze
        
        cam_img = torch.bmm(self.weight_softmax[idx[:, 0]].unsqueeze(1), feature_conv).squeeze(1)   # the result is a 7x7 matrix that represent the attention map
        cam_img = F.softmax(cam_img, 1).data                    # softmax on attention map
        cam_img = cam_img.cpu().numpy()                         # move out data from gpu
        cam_img = cam_img.reshape(h, w)                         # reshape in a 7x7 matrix
        cam_img = cam_img - np.min(cam_img)                     # min max normalization
        cam_img = cam_img / np.max(cam_img)
        cam_img = np.uint8(255 * cam_img)                       # to visualize the pixels values need to be from 0 to 255
        output_cam = cv2.resize(cam_img, size_upsample)         # upsample to match original dimension 224 x 224
        img = cv2.cvtColor(np.uint8(img), cv2.COLOR_RGB2BGR)    
        heatmap = cv2.applyColorMap(output_cam, cv2.COLORMAP_JET)
        result = heatmap * 0.3 + img * 1                        # weights of cam and image
        return result
