import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import random
import glob
import sys


def gen_split(root_dir, users, stackSize=5):
    DatasetX = []  # path to optical x flow
    DatasetY = []  # path to optical y flow
    Labels = []  # labels
    NumFrames = []  # number of frames

    root_dir = os.path.join(root_dir, 'flow_x_processed')
    for usr in users:                                          # /flow_x_processed/S1
        dir1 = os.path.join(root_dir, usr)
        classId = 0
        for target in sorted(os.listdir(dir1)):                # /flow_x_processed/S1/close_chocolate
            trg = os.path.join(dir1, target)
            if target.startswith("."):
                continue
            insts = sorted(os.listdir(trg)) 

            if insts != []:
                for inst in insts:                             # /flow_x_processed/S1/close_chocolate/1
                    if inst.startswith("."):
                        continue

                    inst_dir = os.path.join(trg, inst)
                    numFrames = len(glob.glob1(inst_dir,
                                               '*[0-9].png'))  # there are some duplicates e.g. [(flow_x_00007(1).png] that needs to be ignored)

                    # some elements of the dataset have missing frames
                    if len(glob.glob1(os.path.join(inst_dir.replace('flow_x_processed', 'processed_frames'), "rgb"),
                                      '*[0-9].png')) != len(
                            glob.glob1(os.path.join(inst_dir.replace('flow_x_processed', 'processed_frames'), "mmaps"),
                                       "*[0-9].png")):
                        print(f"skipped {inst_dir}, different frame number")
                        continue

                    if numFrames >= stackSize:                 # stack size is the number of optical flow images used in the flow model
                        DatasetX.append(inst_dir)              # /flow_x_processed/S1/close_chocolate/1
                        DatasetY.append(inst_dir.replace('flow_x_processed',
                                                         'flow_y_processed'))  # /flow_y_processed/S1/close_chocolate/1

                        Labels.append(classId)
                        NumFrames.append(numFrames)

            classId += 1


    return DatasetX, DatasetY, Labels, NumFrames


class makeDataset(Dataset):
    def __init__(self, root_dir, users, spatial_transform=None, stackSize=5, phase='train', seqLen=7):

        self.imagesX, self.imagesY, self.labels, self.numFrames = gen_split(
            root_dir=root_dir, users=users, stackSize=stackSize
        )



        self.spatial_transform = spatial_transform
        self.stackSize = stackSize
        self.seqLen = seqLen
        self.phase = phase

    def __len__(self):
        return len(self.imagesX)


    def __getitem__(self, idx):
        vid_nameX = self.imagesX[idx]
        vid_nameY = self.imagesY[idx]


        label = self.labels[idx]
        numFrame = self.numFrames[idx]

        self.spatial_transform.randomize_parameters()

        if numFrame <= self.stackSize:
            startFrame = 1
        else:  # flow data
            if self.phase == 'train':
                startFrame = random.randint(1,
                                            numFrame - self.stackSize)  # if the number of frames is more that stack size, start from a "middle" point
            else:
                startFrame = np.ceil((numFrame - self.stackSize) / 2)
        inpSeq = []  # returned optical frames
        for k in range(self.stackSize):
            i = k + int(startFrame)  # starting index
            fl_name = vid_nameX + '/flow_x_' + str(int(round(i))).zfill(5) + '.png'  # zfill used to add leading zeros
            img = Image.open(fl_name)  # load single optical x frame
            inpSeq.append(self.spatial_transform(img.convert('L'), inv=True, flow=True))
            fl_name = vid_nameY + '/flow_y_' + str(int(round(i))).zfill(5) + '.png'
            img = Image.open(fl_name)  # load single optical y frame
            inpSeq.append(self.spatial_transform(img.convert('L'), inv=False, flow=True))
        inpSeqSegs = torch.stack(inpSeq, 0).squeeze(
            1)  # flow               


        return inpSeqSegs, label
