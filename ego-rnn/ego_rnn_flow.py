import os
from __future__ import print_function, division
from flow_resnet import *
from spatial_transforms import (Compose, ToTensor, CenterCrop, Scale, Normalize, MultiScaleCornerCrop,
                                RandomHorizontalFlip)
import torch.nn as nn
from torch.autograd import Variable
from makeDatasetFlow import *
import argparse
import sys
import time

DEVICE = 'cuda'  # gpu acceleration

def main_run(version, train_data_dir, outDir, stackSize, trainBatchSize, valBatchSize,
             numEpochs, lr1, stepSize, decayRate, seqLen):

    num_classes = 61

    model_folder = os.path.join("./", outDir, version)

    if os.path.exists(model_folder):
        print('Directory {} exists!'.format(model_folder))
        sys.exit()
    os.makedirs(model_folder)

    train_log_loss = open((model_folder + '/train_log_loss.txt'), 'w')
    train_log_acc = open((model_folder + '/train_log_acc.txt'), 'w')
    val_log_loss = open((model_folder + '/val_log_loss.txt'), 'w')
    val_log_acc = open((model_folder + '/val_log_acc.txt'), 'w')

    # Train val partitioning
    train_usr = ["S1", "S3", "S4"]
    val_usr = ["S2"]

    # Data loader
    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # imagenet mean and std

    spatial_transform = Compose(
        [Scale(256), RandomHorizontalFlip(), MultiScaleCornerCrop([1, 0.875, 0.75, 0.65625], 224),
         ToTensor(), normalize])

    vid_seq_train = makeDataset(train_data_dir, train_usr, stackSize=stackSize,
                                spatial_transform=spatial_transform, seqLen=seqLen)

    train_loader = torch.utils.data.DataLoader(vid_seq_train, batch_size=trainBatchSize, 
                                               shuffle=True, sampler=None, num_workers=4, pin_memory=True)

    vid_seq_val = makeDataset(train_data_dir, val_usr, stackSize=stackSize,
                              spatial_transform=Compose([Scale(256), CenterCrop(224), ToTensor(), normalize]),
                              seqLen=seqLen)

    val_loader = torch.utils.data.DataLoader(vid_seq_val, batch_size=valBatchSize,
                                             shuffle=False, num_workers=2, pin_memory=True)
    valInstances = vid_seq_val.__len__()

    trainInstances = vid_seq_train.__len__()
    print('Number of samples in the dataset: training = {} | validation = {}'.format(trainInstances, valInstances))

    model = flow_resnet34(True, channels=2 * stackSize, num_classes=num_classes)
    model.train(True)
    train_params = list(model.parameters())

    model.cuda()

    loss_fn = nn.CrossEntropyLoss()

    optimizer_fn = torch.optim.SGD(train_params, lr=lr1, momentum=0.9, weight_decay=5e-4)

    optim_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_fn, milestones=stepSize, gamma=decayRate)

    train_iter = 0
    min_accuracy = 0

    for epoch in range(numEpochs):
        start_time = time.time()
        optim_scheduler.step()
        epoch_loss = 0
        numCorrTrain = 0
        trainSamples = 0
        iterPerEpoch = 0
        model.train(True)
        for i, (inputs, targets) in enumerate(train_loader):
            train_iter += 1
            iterPerEpoch += 1
            optimizer_fn.zero_grad()
            inputVariable = Variable(inputs.to(DEVICE))
            labelVariable = Variable(targets.to(DEVICE))
            trainSamples += inputs.size(0)
            output_label, _ = model(inputVariable)
            loss = loss_fn(output_label, labelVariable)
            loss.backward()
            optimizer_fn.step()
            _, predicted = torch.max(output_label.data, 1)
            numCorrTrain += (predicted == targets.to(DEVICE)).sum()  

            epoch_loss += loss.data.item()
        avg_loss = epoch_loss / iterPerEpoch
        trainAccuracy = (numCorrTrain.data.item() / trainSamples) * 100
        print('Train: Epoch = {} | Loss = {} | Accuracy = {}'.format(epoch + 1, avg_loss, trainAccuracy))
        train_log_loss.write(
            'Training loss after {} epoch = {}\n'.format(epoch + 1, avg_loss))  
        train_log_acc.write('Training accuracy after {} epoch = {}\n'.format(epoch + 1, trainAccuracy)) 
        print("--- time elapsed: %.2s seconds ---" % (time.time() - start_time))
        if (epoch + 1) % 5 == 0:
            model.train(False)
            val_loss_epoch = 0
            val_iter = 0
            val_samples = 0
            numCorr = 0
            for j, (inputs, targets) in enumerate(val_loader):
                val_iter += 1
                val_samples += inputs.size(0)
                inputVariable = Variable(inputs.to(DEVICE), volatile=True)
                labelVariable = Variable(targets.to(DEVICE))
                output_label, _ = model(inputVariable)
                val_loss = loss_fn(output_label, labelVariable)
                val_loss_epoch += val_loss.data.item()
                _, predicted = torch.max(output_label.data, 1)
                numCorr += (predicted == targets.to(DEVICE)).sum()
            val_accuracy = (numCorr.data.item() / val_samples) * 100
            avg_val_loss = val_loss_epoch / val_iter
            print('Validation: Epoch = {} | Loss = {} | Accuracy = {}'.format(epoch + 1, avg_val_loss, val_accuracy))
            val_log_loss.write('Val Loss after {} epochs = {}\n'.format(epoch + 1, avg_val_loss))  
            val_log_acc.write('Val Accuracy after {} epochs = {}%\n'.format(epoch + 1, val_accuracy)) 
            if val_accuracy > min_accuracy:
                save_path_model = (
                        model_folder + '/model_flow_state_dict.pth')  
                torch.save(model.state_dict(),
                           save_path_model)  
                min_accuracy = val_accuracy 
        

    # closing files
    train_log_loss.close()
    train_log_acc.close()
    val_log_acc.close()
    val_log_loss.close()

def __main__():
    version = "flow_300_stack10"
    train_data_dir = "/content/"
    outDir = "results"

    stackSize = 5 #5
    trainBatchSize = 32  # 32
    valBatchSize = 32  # 32
    numEpochs = 750
    lr1 = 1e-2  # 1e-2
    stepSize = [150, 300, 500]
    decayRate = 0.5  # 0.5
    seqLen = 16  # 7

    main_run(version=version,
             train_data_dir=train_data_dir,
             outDir=outDir,
             stackSize=stackSize,
             trainBatchSize=trainBatchSize,
             valBatchSize=valBatchSize,
             numEpochs=numEpochs,
             lr1=lr1,
             stepSize=stepSize,
             decayRate=decayRate,
             seqLen=seqLen)

if __name__ == "__main__":
    __main__()
