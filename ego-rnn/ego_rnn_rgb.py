from __future__ import print_function, division
from objectAttentionModelConvLSTM import *
from spatial_transforms import (Compose, ToTensor, CenterCrop, Scale, Normalize, MultiScaleCornerCrop,
                                RandomHorizontalFlip)
from makeDatasetFrame import *
import argparse
import sys

DEVICE = "cuda"
VAL_FREQUENCY = 5

def main_run(version, stage, train_data_dir, stage1_dict, out_dir, seqLen, trainBatchSize,
             valBatchSize, numEpochs, lr1, decay_factor, decay_step, mem_size):
    num_classes = 61

    model_folder = os.path.join("./", out_dir, version)

    if os.path.exists(model_folder):
        print('Directory {} exists!'.format(model_folder))
        sys.exit()
    os.makedirs(model_folder)

    train_log_loss = open((model_folder + '/train_log_loss.txt'), 'w')
    train_log_acc  = open((model_folder + '/train_log_acc.txt'), 'w')
    val_log_loss   = open((model_folder + '/val_log_loss.txt'), 'w')
    val_log_acc    = open((model_folder + '/val_log_acc.txt'), 'w')

    # Train val partitioning
    train_usr = ["S1", "S3", "S4"]
    val_usr = ["S2"]

    # Data loader
    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    spatial_transform = Compose(
        [Scale(256), RandomHorizontalFlip(), MultiScaleCornerCrop([1, 0.875, 0.75, 0.65625], 224),
         ToTensor(), normalize])

    vid_seq_train = makeDataset(train_data_dir, train_usr,
                                spatial_transform=spatial_transform, seqLen=seqLen)

    train_loader = torch.utils.data.DataLoader(vid_seq_train, batch_size=trainBatchSize,
                                               shuffle=True, num_workers=4, pin_memory=True)

    vid_seq_val = makeDataset(train_data_dir, val_usr,
                              spatial_transform=Compose([Scale(256), CenterCrop(224), ToTensor(), normalize]),
                              seqLen=seqLen, phase="test")

    val_loader = torch.utils.data.DataLoader(vid_seq_val, batch_size=valBatchSize,
                                             shuffle=False, num_workers=2, pin_memory=True)

    train_params = []

    # stage 1: train only lstm
    if stage == 1:

        model = attentionModel(num_classes=num_classes, mem_size=mem_size)
        model.train(False)
        for params in model.parameters():
            params.requires_grad = False

    # stage 2: train lstm, layer4, spatial attention and final fc
    else:
        model = attentionModel(num_classes=num_classes, mem_size=mem_size)
        model.load_state_dict(torch.load(stage1_dict))  
        model.train(False)
        for params in model.parameters():
            params.requires_grad = False
        #
        for params in model.resNet.layer4[0].conv1.parameters():
            params.requires_grad = True
            train_params += [params]

        for params in model.resNet.layer4[0].conv2.parameters():
            params.requires_grad = True
            train_params += [params]

        for params in model.resNet.layer4[1].conv1.parameters():
            params.requires_grad = True
            train_params += [params]

        for params in model.resNet.layer4[1].conv2.parameters():
            params.requires_grad = True
            train_params += [params]

        for params in model.resNet.layer4[2].conv1.parameters():
            params.requires_grad = True
            train_params += [params]
        #
        for params in model.resNet.layer4[2].conv2.parameters():
            params.requires_grad = True
            train_params += [params]
        #
        for params in model.resNet.fc.parameters():  # fully connected layer
            params.requires_grad = True
            train_params += [params]

        model.resNet.layer4[0].conv1.train(True)
        model.resNet.layer4[0].conv2.train(True)
        model.resNet.layer4[1].conv1.train(True)
        model.resNet.layer4[1].conv2.train(True)
        model.resNet.layer4[2].conv1.train(True)
        model.resNet.layer4[2].conv2.train(True)
        model.resNet.fc.train(True)

    for params in model.lstm_cell.parameters():  # for both stages we train the lstm
        params.requires_grad = True
        train_params += [params]

    for params in model.classifier.parameters():  # for both stages we train the last classifier (after the lstm and avg pooling)
        params.requires_grad = True
        train_params += [params]

    model.lstm_cell.train(True)

    model.classifier.train(True)
    model.cuda()

    loss_fn = nn.CrossEntropyLoss()

    optimizer_fn = torch.optim.Adam(train_params, lr=lr1, weight_decay=4e-5, eps=1e-4)

    optim_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_fn, milestones=decay_step,
                                                           gamma=decay_factor)

    train_iter = 0
    min_accuracy = 0

    for epoch in range(numEpochs):
        optim_scheduler.step()
        epoch_loss = 0
        numCorrTrain = 0
        trainSamples = 0
        iterPerEpoch = 0
        model.lstm_cell.train(True)
        model.classifier.train(True)
        if stage == 2:
            model.resNet.layer4[0].conv1.train(True)
            model.resNet.layer4[0].conv2.train(True)
            model.resNet.layer4[1].conv1.train(True)
            model.resNet.layer4[1].conv2.train(True)
            model.resNet.layer4[2].conv1.train(True)
            model.resNet.layer4[2].conv2.train(True)
            model.resNet.fc.train(True)
        for i, (inputs, inputsF, targets) in enumerate(train_loader):
            train_iter += 1
            iterPerEpoch += 1
            optimizer_fn.zero_grad()
            inputVariable = Variable(inputs.permute(1, 0, 2, 3, 4).to(DEVICE))
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
        trainAccuracy = (numCorrTrain.data.item() / trainSamples)

        train_log_loss.write('Training loss after {} epoch = {}\n'.format(epoch + 1, avg_loss))  
        train_log_acc.write('Training accuracy after {} epoch = {}\n'.format(epoch + 1, trainAccuracy))  
        print('Train: Epoch = {} | Loss = {} | Accuracy = {}'.format(epoch + 1, avg_loss, trainAccuracy))

        if (epoch + 1) % VAL_FREQUENCY == 0:
            model.train(False)
            val_loss_epoch = 0
            val_iter = 0
            val_samples = 0
            numCorr = 0
            for j, (inputs, inputsF, targets) in enumerate(val_loader):
                val_iter += 1
                val_samples += inputs.size(0)
                inputVariable = Variable(inputs.permute(1, 0, 2, 3, 4).to(DEVICE))
                labelVariable = Variable(targets.to(DEVICE))
                output_label, _ = model(inputVariable)
                val_loss = loss_fn(output_label, labelVariable)
                val_loss_epoch += val_loss.data.item()
                _, predicted = torch.max(output_label.data, 1)
                numCorr += (predicted == targets.to(DEVICE)).sum()  
            val_accuracy = (numCorr.data.item() / val_samples)
            avg_val_loss = val_loss_epoch / val_iter
            print('Val: Epoch = {} | Loss {} | Accuracy = {}'.format(epoch + 1, avg_val_loss, val_accuracy))
            val_log_loss.write('Val Loss after {} epochs = {}\n'.format(epoch + 1, avg_val_loss))  
            val_log_acc.write('Val Accuracy after {} epochs = {}%\n'.format(epoch + 1, val_accuracy))  
            if val_accuracy > min_accuracy:
                save_path_model = (
                        model_folder + '/model_rgb_state_dict.pth')  
                torch.save(model.state_dict(),
                           save_path_model) 
                min_accuracy = val_accuracy  

    train_log_loss.close()
    train_log_acc.close()
    val_log_acc.close()
    val_log_loss.close()

def __main__():
    version = "rgb_16frames_noCAM"
    trainDatasetDir = "/content/"
    outDir = "results"onli
    stage1Dict = "./" + outDir + "/" + version + "_1/model_rgb_state_dict.pth" 

    # STAGE 1 PARAMETERS
    ST1_seqLen = 16  # 7
    ST1_trainBatchSize = 32  # 32
    ST1_valBatchSize = 32  # 32
    ST1_numEpochs = 200  # 200
    ST1_lr1 = 1e-3  # 1e-3
    ST1_stepSize = [25, 75, 150]  # [25, 75, 150]
    ST1_decayRate = 0.1  # 0.1
    ST1_memSize = 512  # 512

    # STAGE 2 PARAMETERS
    ST2_seqLen = 16  # 7
    ST2_trainBatchSize = 32  # 32
    ST2_valBatchSize = 32  # 32
    ST2_numEpochs = 150  # 150
    ST2_lr1 = 1e-4  # 1e-4
    ST2_stepSize = [25, 75]  # [25, 75]
    ST2_decayRate = 0.1  # 0.1
    ST2_memSize = 512  # 512


    # STAGE 1
    main_run(version + "_1",
             stage=1,
             train_data_dir=trainDatasetDir,
             stage1_dict=stage1Dict,
             out_dir=outDir,
             seqLen=ST1_seqLen,
             trainBatchSize=ST1_trainBatchSize,
             valBatchSize=ST1_valBatchSize,
             numEpochs=ST1_numEpochs,
             lr1=ST1_lr1,
             decay_factor=ST1_decayRate,
             decay_step=ST1_stepSize,
             mem_size=ST1_memSize)
    
    # STAGE 2
    main_run(version + "_2",
             stage=2,
             train_data_dir=trainDatasetDir,
             stage1_dict=stage1Dict,
             out_dir=outDir,
             seqLen=ST2_seqLen,
             trainBatchSize=ST2_trainBatchSize,
             valBatchSize=ST2_valBatchSize,
             numEpochs=ST2_numEpochs,
             lr1=ST2_lr1,
             decay_factor=ST2_decayRate,
             decay_step=ST2_stepSize,
             mem_size=ST2_memSize)

__main__()
