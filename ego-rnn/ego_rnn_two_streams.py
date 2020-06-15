from __future__ import print_function, division
from spatial_transforms import (Compose, ToTensor, CenterCrop, Scale, Normalize, MultiScaleCornerCrop,
                                RandomHorizontalFlip)
import torch.nn as nn
from twoStreamModel import *
from torch.autograd import Variable
from torch.utils.data.sampler import WeightedRandomSampler
from makeDataset import *
import argparse

import sys
import time
import os

DEVICE = 'cuda'     # gpu acceleration

# version is a name for the run
# a different folder will be created for every version
def main_run(version, flowModel, rgbModel, stackSize, seqLen, memSize, trainDatasetDir, outDir,
             trainBatchSize, valBatchSize, lr1, numEpochs, decay_step, decay_factor):
    
    num_classes = 61     # gtea61 dataset
    model_folder = os.path.join("./", outDir, version)

    # Create the dir
    print(f"Checking directory {model_folder}")
    if os.path.exists(model_folder):
        print('Dir {} exists!'.format(model_folder))
        sys.exit()
    print(f"Creating directory{model_folder}")
    os.makedirs(model_folder)

    # Log files
    print(f"Creating log files")
    train_log_loss = open((model_folder + '/train_log_loss.txt'), 'w')
    train_log_acc = open((model_folder + '/train_log_acc.txt'), 'w')
    val_log_loss = open((model_folder + '/val_log_loss.txt'), 'w')
    val_log_acc = open((model_folder + '/val_log_acc.txt'), 'w')

    # ImageNet mean and std
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Train val partitioning
    train_usr = ["S1", "S3", "S4"]
    val_usr = ["S2"]


    normalize = Normalize(mean=mean, std=std)

    spatial_transform = Compose([Scale(256), RandomHorizontalFlip(), MultiScaleCornerCrop([1, 0.875, 0.75, 0.65625], 224),
                                 ToTensor(), normalize])
    # train dataset
    print(f"Defining train dataset")
    vid_seq_train = makeDataset(trainDatasetDir, train_usr, spatial_transform,
                               stackSize=stackSize, seqLen=seqLen)

    train_loader = torch.utils.data.DataLoader(vid_seq_train, batch_size=trainBatchSize,
                            shuffle=True, num_workers=4, pin_memory=True)
    

    # val dataset
    print(f"Defining validation dataset")
    vid_seq_val = makeDataset(trainDatasetDir, val_usr,
                                   spatial_transform=Compose([Scale(256), CenterCrop(224), ToTensor(), normalize]),
                                   stackSize=stackSize, phase="val", seqLen=seqLen)
    
    val_loader = torch.utils.data.DataLoader(vid_seq_val, batch_size=valBatchSize,
                                shuffle=False, num_workers=2, pin_memory=True)
    
    valSamples = vid_seq_val.__len__()


    # model
    print("Building model")
    model = twoStreamAttentionModel(flowModel=flowModel, frameModel=rgbModel, stackSize=stackSize, memSize=memSize,         # see twoStreamModel.py
                                    num_classes=num_classes)
    
    print("Setting trainable parameters")
    for params in model.parameters():           # initially freeze all layers
        params.requires_grad = False

    model.train(False)
    train_params = []

    for params in model.classifier.parameters():    # unfreeze classifier layer (the layer that joins the two models outputs)
        params.requires_grad = True
        train_params += [params]

    for params in model.frameModel.lstm_cell.parameters():  # unfreeze lstm layer of the frame model
        train_params += [params]
        params.requires_grad = True

    for params in model.frameModel.resNet.layer4[0].conv1.parameters():     #unfreeze layer 4
        params.requires_grad = True
        train_params += [params]

    for params in model.frameModel.resNet.layer4[0].conv2.parameters():
        params.requires_grad = True
        train_params += [params]

    for params in model.frameModel.resNet.layer4[1].conv1.parameters():
        params.requires_grad = True
        train_params += [params]

    for params in model.frameModel.resNet.layer4[1].conv2.parameters():
        params.requires_grad = True
        train_params += [params]

    for params in model.frameModel.resNet.layer4[2].conv1.parameters():
        params.requires_grad = True
        train_params += [params]
    #
    for params in model.frameModel.resNet.layer4[2].conv2.parameters():
        params.requires_grad = True
        train_params += [params]
    #
    for params in model.frameModel.resNet.fc.parameters():              # unfreeze last fully connected layer of frame model 
        params.requires_grad = True                                     # (I still don't know why, because in the joining of the two models, this layer is skipped)
        train_params += [params]                                        

    base_params = []
    for params in model.flowModel.layer4.parameters():              # unfreeze layer 4 of flow model
        base_params += [params]
        params.requires_grad = True

    print("Moving model to GPU")
    model.to(DEVICE)

    trainSamples = vid_seq_train.__len__()
    min_accuracy = 0

    print("Defining loss function, optimizer and scheduler")
    loss_fn = nn.CrossEntropyLoss()     # loss function
    optimizer_fn = torch.optim.SGD([    # optimizer
        {'params': train_params},
        {'params': base_params, 'lr': 1e-4},  # 1e-4
    ], lr=lr1, momentum=0.9, weight_decay=5e-4)

    #scheduler
    optim_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_fn, step_size=decay_step, gamma=decay_factor)
    train_iter = 0

    print("Training begun")
    # TRAIN PROCEDURE
    for epoch in range(numEpochs):
        optim_scheduler.step()
        epoch_loss = 0
        numCorrTrain = 0
        iterPerEpoch = 0
        model.classifier.train(True)
        model.flowModel.layer4.train(True)


        start = time.time()
        for j, (inputFrame, inputMMaps, inputFlow, targets) in enumerate(train_loader):
            
            print(f"step {j} / {int(np.floor(trainSamples/trainBatchSize))}")
            
            train_iter += 1
            iterPerEpoch += 1
            optimizer_fn.zero_grad()                                                # put gradients to zero
            inputVariableFlow = Variable(inputFlow.to(DEVICE))
            inputVariableFrame = Variable(inputFrame.permute(1, 0, 2, 3, 4).to(DEVICE))
            labelVariable = Variable(targets.to(DEVICE))
            #print("predict")
            output_label = model(inputVariableFlow, inputVariableFrame)         # predict
            loss = loss_fn(F.log_softmax(output_label, dim=1), labelVariable)   # compute loss
            #print("backprop")
            loss.backward()                                                     
            optimizer_fn.step()
            #print("accuracy")
            _, predicted = torch.max(output_label.data, 1)                  
            numCorrTrain += (predicted == targets.to(DEVICE)).sum()             # counting number of correct predictions
            epoch_loss += loss.data.item()  

        
        avg_loss = epoch_loss / iterPerEpoch                                    # computing average per epoch loss
        trainAccuracy = (numCorrTrain.item() / trainSamples) * 100
        print('Average training loss after {} epoch = {} '.format(epoch + 1, avg_loss))
        print('Training accuracy after {} epoch = {}% '.format(epoch + 1, trainAccuracy))
        train_log_loss.write('Training loss after {} epoch = {}\n'.format(epoch + 1, avg_loss))             # log file
        train_log_acc.write('Training accuracy after {} epoch = {}\n'.format(epoch + 1, trainAccuracy))     # log file
        print(f"Elapsed : {time.time()-start}")

        # VALIDATION
        if (epoch + 1) % 5 == 0:
            model.train(False)
            val_loss_epoch = 0
            val_iter = 0
            numCorr = 0
            for j, (inputFrame, inputMMaps, inputFlow, targets) in enumerate(val_loader):
                if j % 1 == 0:
                    print(f"step {j} / {int(np.floor(vid_seq_val.__len__()/valBatchSize))}")

                val_iter += 1
                inputVariableFlow = Variable(inputFlow.to(DEVICE))
                inputVariableFrame = Variable(inputFrame.permute(1, 0, 2, 3, 4).to(DEVICE))
                labelVariable = Variable(targets.to(DEVICE))
                output_label = model(inputVariableFlow, inputVariableFrame)
                loss = loss_fn(F.log_softmax(output_label, dim=1), labelVariable)
                val_loss_epoch += loss.data.item()
                _, predicted = torch.max(output_label.data, 1)
                numCorr += (predicted == labelVariable.data).sum()
            val_accuracy = (numCorr.item() / valSamples) * 100
            avg_val_loss = val_loss_epoch / val_iter
            print('Val Loss after {} epochs, loss = {}'.format(epoch + 1, avg_val_loss))
            print('Val Accuracy after {} epochs = {}%'.format(epoch + 1, val_accuracy))
            val_log_loss.write('Val Loss after {} epochs = {}\n'.format(epoch + 1, avg_val_loss))       # log file
            val_log_acc.write('Val Accuracy after {} epochs = {}%\n'.format(epoch + 1, val_accuracy))   # log file
            if val_accuracy > min_accuracy:
                save_path_model = (model_folder + '/model_twoStream_state_dict.pth')                    # every epoch, check if the val accuracy is improved, if so, save that model
                torch.save(model.state_dict(), save_path_model)                                         # in that way, even if the model overfit, you will get always the best model
                min_accuracy = val_accuracy                                                             # in this way you don't have to care too much about the number of epochs

    train_log_loss.close()
    train_log_acc.close()
    val_log_acc.close()
    val_log_loss.close()

!rm "-r" "results/two_stream_5stacksize_7frames"

version = "two_stream_7frames_300flow"            # progress are saved inside this folder
rgbModel = "/content/drive/My Drive/Lorenzo/ego-rnn-latest/rgb_7frames/rgb_7frames_2/model_rgb_state_dict.pth"
flowModel = "/content/drive/My Drive/Lorenzo/ego-rnn-latest/flow300/model_flow_state_dict.pth"
trainDatasetDir = "/content"
outDir = "results"    # root of "version" folder
stackSize = 5                   # number of flow frame processed in the flow model
seqLen = 7                    # number of rgb frames processed in the frame model
trainBatchSize = 32
valBatchSize = 32
numEpochs = 250
lr1 = 1e-2
decay_step = 1
decay_factor = 0.99
memSize=512

main_run(version, flowModel, rgbModel, stackSize, seqLen, memSize, trainDatasetDir, outDir,
             trainBatchSize, valBatchSize, lr1, numEpochs, decay_step, decay_factor)
