from opts import parse_opts
from dataset.dataset import *
from model.model import *
from torch.utils.data import Dataset, DataLoader
import torch
import os 
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from utils import *
import time
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__=="__main__":
    opt = parse_opts()
    print(opt)

    print("Preprocessing train data ...")
    train_data   = globals()['{}_test'.format(opt.dataset)](split = opt.split, train = 1, opt = opt)
    print("Length of train data = ", len(train_data))

    print("Preprocessing validation data ...")
    val_data   = globals()['{}_test'.format(opt.dataset)](split = opt.split, train = 2, opt = opt)
    print("Length of validation data = ", len(val_data))

    if opt.modality=='RGB': opt.input_channels = 3
    elif opt.modality=='Flow': opt.input_channels = 2

    print("Preparing datatloaders ...")
    train_dataloader = DataLoader(train_data, batch_size = opt.batch_size, shuffle=True)
    val_dataloader   = DataLoader(val_data, batch_size = opt.batch_size, shuffle=True)
    print("Length of train datatloader = ",len(train_dataloader))
    print("Length of validation datatloader = ",len(val_dataloader))  

    if opt.modality == 'RGB':
      model = SpatialNet(opt.n_classes, opt.n_segments)
    elif opt.modality == 'Flow':
      model = TemporalNet(opt.n_classes, opt.n_segments)
    model.to(device)
    

    if opt.resume_path1:
        print('loading checkpoint {}'.format(opt.resume_path1))
        checkpoint = torch.load(opt.resume_path1)
        
        opt.begin_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
    

    if opt.pretrain_path: 
        opt.weight_decay = 1e-5
        opt.learning_rate = 0.001

    if opt.nesterov: dampening = 0
    else: dampening = opt.dampening
        
    print("lr = {} \t momentum = {} \t dampening = {} \t weight_decay = {}, \t nesterov = {}"
                .format(opt.learning_rate, opt.momentum, dampening, opt. weight_decay, opt.nesterov))
    print("LR patience = ", opt.lr_patience)
    
    
    optimizer = optim.SGD(
        model.parameters(),
        lr=opt.learning_rate,
        momentum=opt.momentum,
        dampening=dampening,
        weight_decay=opt.weight_decay,
        nesterov=opt.nesterov) 
    criterion = nn.CrossEntropyLoss()

    
    if opt.resume_path1 != '':
        optimizer.load_state_dict(torch.load(opt.resume_path1)['optimizer'])

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=opt.lr_patience)
    
    #tensor board
    log_path = os.path.join(opt.result_path, opt.dataset)
    writer_path = os.path.join(log_path, '{}_{}_{}_train_batch{}_varLR'
                            .format(opt.dataset, opt.split, opt.modality, opt.batch_size))
    writer = SummaryWriter(writer_path)

    #begin training phase
    for epoch in range(opt.begin_epoch, opt.n_epochs + 1):
        
        model.train()
        
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        accuracies = AverageMeter()

        end_time = time.time()
        for i, (inputs, targets) in enumerate(train_dataloader):
            data_time.update(time.time() - end_time)
        
            #targets = targets.cuda(non_blocking=True)
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            acc = calculate_accuracy(outputs, targets)

            losses.update(loss.item(), inputs.size(0))
            accuracies.update(acc, inputs.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end_time)
            end_time = time.time()

            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch,
                      i + 1,
                      len(train_dataloader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      acc=accuracies))
        print("\nLearning rate:", optimizer.param_groups[0]['lr'], '\n')

        writer.add_scalar('Train/Loss', losses.avg, epoch)
        writer.add_scalar('Train/Accuracy', accuracies.avg, epoch)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
        
        if epoch % opt.checkpoint == 0:
            save_file_path = os.path.join(log_path, '{}_{}_{}_train_batch{}_varLR{}.pth'
                              .format(opt.dataset, opt.split, opt.modality, opt.batch_size, epoch))
            states = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
            print(save_file_path)
            torch.save(states, save_file_path)
        
        #Test on validation set
        model.eval()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        accuracies = AverageMeter()
        for i, (inputs, targets) in enumerate(val_dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            data_time.update(time.time() - end_time)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            acc = calculate_accuracy(outputs, targets)

            losses.update(loss.item(), inputs.size(0))
            accuracies.update(acc, inputs.size(0))

            batch_time.update(time.time() - end_time)
            end_time = time.time()

            print('Val_Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                    epoch,
                    i + 1,
                    len(val_dataloader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    acc=accuracies))
        writer.add_scalar('Validation/Accuracy', accuracies.avg, epoch)
        print("")
        if opt.reduce_lr:
          scheduler.step(losses.avg)
          