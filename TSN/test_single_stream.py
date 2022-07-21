from dataset.dataset import *
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from dataset.preprocess_data import *
from PIL import Image, ImageFilter
import argparse
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from opts import parse_opts
import time
import torch.utils
import sys
from utils import *
from model.model import *
from sklearn.metrics import confusion_matrix
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__=="__main__":
    # print configuration options
    opt = parse_opts()
    print(opt)
    
    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)

    print("Preprocessing validation data ...")
    data   = globals()['{}_test'.format(opt.dataset)](split = opt.split, train = 0, opt = opt)
    print("Length of validation data = ", len(data))
    
    if opt.modality=='RGB': opt.input_channels = 3
    elif opt.modality=='Flow': opt.input_channels = 2

    print("Preparing datatloaders ...")
    val_dataloader = DataLoader(data, batch_size = 1, shuffle=False, num_workers = opt.n_workers, pin_memory = True, drop_last=False)
    print("Length of validation datatloader = ",len(val_dataloader))
    
    # Loading model and checkpoint
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
    
    accuracies = AverageMeter()
    clip_accuracies = AverageMeter()
    
    #Path to store results
    result_path = "{}/{}/".format(opt.result_path, opt.dataset)
    if not os.path.exists(result_path):
        os.makedirs(result_path)    

    ground_truth = []
    pred = []
    model.eval()

    for i, (inputs, label) in enumerate(val_dataloader):        
        #targets = targets.cuda(non_blocking=True)
         #output shape: batch x n_times x n_classes
        outputs = torch.zeros(opt.batch_size, opt.test_times, opt.n_classes, device = label.device)
        inputs = inputs.to(device)

        for t in range(opt.test_times):
          idx = [(k*opt.test_times + t) for k in range(opt.n_segments)]
          output_ = model(inputs[:, :, idx, :, :])
          outputs[:, t, :] = output_  

        outputs = torch.mean(outputs, dim = 1)
        pred5 = np.array(outputs.topk(5, 1, True)[1].cpu().data[0])
            
        acc = float(pred5[0] == label[0])
        ground_truth.append(label[0])
        pred.append(pred5[0])
                        
        accuracies.update(acc, 1)            
        
        line = "Video[" + str(i) + "] : \t top5 " + str(pred5) + "\t top1 = " + str(pred5[0]) +  "\t true = " +str(int(label[0])) + "\t video = " + str(accuracies.avg)
        print(line)

    
    confusion = confusion_matrix(ground_truth, pred)
    print("Video accuracy = ", accuracies.avg)
    print(confusion)
    line = "Video accuracy = " + str(accuracies.avg) + '\n'
    