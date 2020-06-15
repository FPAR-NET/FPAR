import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import random
import glob
import sys


def gen_split(root_dir, users, stackSize=5):
    DatasetF = []   # path to rgb frame
    DatasetM = []   # path to motion frame (IDT output)
    Labels = []     # labels
    NumFrames = []  # number of frames

    root_dir = os.path.join(root_dir, 'processed_frames')
    for usr in users:                               #/processed_frames/S1
        dir1 = os.path.join(root_dir, usr)
        classId = 0
        for target in sorted(os.listdir(dir1)):     #/processed_frames/S1/close_chocolate
            trg = os.path.join(dir1, target)
            if target.startswith("."):
                continue
            insts = sorted(os.listdir(trg))        

            if insts != []:
                for inst in insts:                  #/processed_frames/S1/close_chocolate/1
                    if inst.startswith("."):
                        continue

                    inst_dir = os.path.join(trg, inst)
                    numFrames = len(glob.glob1(os.path.join(inst_dir, "rgb"), '*[0-9].png'))

                    # some elements of the dataset have missing frames
                    if len(glob.glob1(os.path.join(inst_dir, "rgb"), '*[0-9].png')) != len(
                            glob.glob1(os.path.join(inst_dir, "mmaps"), "*[0-9].png")):
                        print(f"skipped {inst_dir}, different frame number")
                        continue

                    if numFrames >= stackSize:  # stacksize is the number of optical flow images used in the flow model
                        DatasetF.append(os.path.join(inst_dir, "rgb"))      # /processed_frames/S1/close_chocolate/1/rgb
                        DatasetM.append(os.path.join(inst_dir, "mmaps"))    # /processed_frames/S1/close_chocolate/1/mmaps

                        Labels.append(classId)
                        NumFrames.append(numFrames)

            classId += 1

    return DatasetF, DatasetM, Labels, NumFrames


class makeDataset(Dataset):
    def __init__(self, root_dir, users, spatial_transform=None, stackSize=5, phase='train', seqLen=7):
        self.imagesF, self.imagesM, self.labels, self.numFrames = gen_split(
            root_dir=root_dir, users=users, stackSize=stackSize
        )

        self.spatial_transform = spatial_transform
        self.stackSize = stackSize
        self.seqLen = seqLen
        self.phase = phase

    def __len__(self):
        return len(self.imagesM)

    def __getitem__(self, idx):
        vid_nameF = self.imagesF[idx]
        vid_nameM = self.imagesM[idx]

        label = self.labels[idx]
        numFrame = self.numFrames[idx]

        self.spatial_transform.randomize_parameters()

        inpF = []
        inpM = []

        for i in np.linspace(1, numFrame, self.seqLen,
                             endpoint=False):                # sampling, if numFrame>seqLen some frames are sampled more than once
            fl_name = vid_nameF + '/' + 'rgb' + str(int(np.floor(i))).zfill(4) + ".png"
            img = Image.open(fl_name)  
            inpF.append(self.spatial_transform(img.convert('RGB')))

            fl_name = vid_nameM + '/' + 'map' + str(int(np.floor(i))).zfill(4) + ".png"
            img = Image.open(fl_name)
            inpM.append(self.spatial_transform(img))

        inpSeqF = torch.stack(inpF, 0)  # frame
        inpSeqM = torch.stack(inpM, 0)  # motion map

        return inpSeqF, inpSeqM, label
