import numpy as np
import time    
import sys
import copy
#from Function_self import Function
sys.path.append("./scripts/Train/auto_threshold/")
sys.path.append("./distiller/")
import train_model
sys.setrecursionlimit(1000000)
sys.path.append("../pytorch-video-recognition-master/")
import logging
import argparse
from distiller.data_loggers import *
from torch.utils.data import DataLoader
import distiller.apputils.image_classifier as ic
import torch
import distiller
from distiller.quantization.range_linear import PostTrainLinearQuantizer
from distiller.quantization.range_linear import RangeLinearQuantWrapper
import distiller.apputils as apputils
from distiller.apputils.image_classifier import test
import torch.nn as nn
import os
import shutil
import C3D_model_th
import timeit
from datetime import datetime
import socket
#import os
import glob
from tqdm import tqdm
import math
import torch.onnx
#from tensorboardX import SummaryWriter
from torchsummary import summary
from torch import optim
#from torch.utils.data import DataLoader
from torch.autograd import Variable

from dataloaders.dataset import VideoDataset
import get_logger
from scheduler_add_policy import scheduler_add_policy





torch.cuda.empty_cache()
parser = argparse.ArgumentParser()
parser.add_argument('--compress',dest='compress',type=str,nargs='?',action='store')
parser.add_argument('--batch_size',dest='batch_size',type=int,default=1)
#parser.add_argument('--policy',dest='policy',action='store_true')
parser.add_argument('--name',dest='name',type=str)
parser.add_argument('--device',dest='device',type=str,nargs='?')
parser.add_argument('--narrow_inputs',dest='narrow_inputs',action='store_true')
parser.add_argument('--extract',dest='extract',action='store_true')
parser.add_argument('--dequant',dest='dequant',action='store_true')
parser.add_argument('--resume-from',dest='resumed_checkpoint_path',default='',\
    type=str, metavar='PATH',help='path to latest checkpoint. Use to resume paused training session.')
distiller.quantization.add_post_train_quant_args(parser)



# Check complete args
args = parser.parse_args()


# coding:utf-8
import configparser
cfgpath = os.path.join(args.init)
conf = configparser.ConfigParser()



if args.name ==None or args.device ==None or args.batch_size==None:#only compress
	print('*'*45+'complete args: compress,name,batch_size,device,narrow_inputs'+'*'*45)
	os._exit()
##############################################################

postquant_pth = 'Extract/prune_high_ele/run/run_4/models/C3D-ucf101_epoch-10.pth.tar' 
device = torch.device(args.device)
print("Device being used:", device)

# Pruning 30 epoches
nEpochs =400  # Number of epochs for training
resume_epoch = 0  # Default is 0, change if want to resume
useTest = True # See evolution of the test set when training
nTestInterval = 1 # Run on test set every nTestInterval epochs
save_epoch = 30 # Store a model every snapshot epochs
lr = 1e-4 # Learning rate

batch_size = args.batch_size
dataset = 'ucf101' # Options: hmdb51 or ucf101

if dataset == 'hmdb51':
    num_classes=51
elif dataset == 'ucf101':
    num_classes = 101
else:
    print('We only implemented hmdb and ucf datasets.')
    raise NotImplementedError

save_dir_root = os.path.join('/workspace/Pytorch/distiller-master/','Extract',args.name)
exp_name = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]

if resume_epoch != 0:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) if runs else 0
else:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0

save_dir = os.path.join(save_dir_root, 'run', 'run_' + str(run_id))
modelName = 'C3D' # Options: C3D or R2Plus1D or R3D
saveName = modelName + '-' + dataset

msglogger = apputils.config_pylogger('logging.conf',experiment_name=args.name,output_dir=save_dir)
tflogger = TensorBoardLogger(msglogger.logdir)
tflogger.log_gradients = True
pylogger = PythonLogger(msglogger)
#if args.extract != None:
if args.compress:
    shutil.copy(args.compress,save_dir)
shutil.copy(sys.argv[0],save_dir)
shutil.copy('./scripts/Train/auto_threshold/C3D_model_th.py', save_dir)
shutil.copy('./scripts/Train/auto_threshold/train_model.py', save_dir)

msglogger = get_logger.get_logger(save_dir+'/get_logger.log')

msglogger.info('*'*45+'command line :')
msglogger.info(sys.argv)


#**************************************************************************************************
"""
Args:
    num_classes (int): Number of classes in the data
    num_epochs (int, optional): Number of epochs to train for.
"""

if modelName == 'C3D':
    model = C3D_model_th.C3D(num_classes=num_classes,pretrained=True)
    train_params = [
        {'params': C3D_model_th.get_1x_lr_params(model), 'lr': lr},
        {'params': C3D_model_th.get_10x_lr_params(model), 'lr': lr * 10},
        {'params': C3D_model_th.get_10x_lr_Threshold(model), 'lr': lr * 100}

        ]
else:
    print('We only implemented C3D and R2Plus1D models.')
    raise NotImplementedError



optimizer = optim.SGD(train_params, lr=lr, momentum=0.9, weight_decay=5e-4)
# optimizer = optim.Adam(train_params, lr=lr)
 
print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
#***********************************************************************************************
if torch.cuda.device_count() > 1:
    print("Use", torch.cuda.device_count(), 'gpus')
    model = nn.DataParallel(model)
for state in optimizer.state.values():
    for k, v in state.items():
        if torch.is_tensor(v):
            state[k] = v.cuda()
model.to(device)

compression_scheduler = None

from checkpoint_self import load_checkpoint_self

if args.resumed_checkpoint_path:
    checkpoint = torch.load(args.resumed_checkpoint_path, map_location=lambda storage, loc: storage)
    resume_epoch = checkpoint['epoch'] -1

if args.compress:
    source = args.compress
    compression_scheduler=distiller.file_config(model,optimizer, \
        args.compress, compression_scheduler,resumed_epoch=resume_epoch if args.resumed_checkpoint_path else None)
    compression_scheduler.append_float_weight_after_quantizer()
if args.resumed_checkpoint_path:
    model.load_state_dict(checkpoint['state_dict'])

# sys.exit()
if compression_scheduler ==None:
    print('ERROR --------------------------No compress------------------------')

try:
    train_model.train_model(args=args, model=model,optimizer=optimizer, dataset=dataset, save_dir=save_dir, saveName=saveName, num_classes=num_classes, lr=lr,
        nEpochs=nEpochs, resume_epoch=resume_epoch,batch_size = batch_size,save_epoch=save_epoch, useTest=useTest, test_interval=nTestInterval,
        device = device, compression_scheduler=compression_scheduler,msglogger=msglogger, tflogger=tflogger,pylogger=pylogger,modelName=modelName,postquant_pth=postquant_pth)

except KeyboardInterrupt:
    msglogger.info('-' * 89)
    msglogger.info('Exiting from train_model early')#

