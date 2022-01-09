
import sys
#from Function_self import Function
sys.path.append("./scripts/Train/auto_threshold/")
sys.path.append("./distiller/")
import train_model
sys.setrecursionlimit(1000000)
sys.path.append("../pytorch-video-recognition-master/")
import argparse
from distiller.data_loggers import *
from torch.utils.data import DataLoader
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

import glob

import torch.onnx
from torch import optim

import get_logger

import json


torch.cuda.empty_cache()
parser = argparse.ArgumentParser()
distiller.quantization.add_post_train_quant_args(parser)

parser.add_argument('--init',dest='init',type=str,nargs='?',action='store')

# Check complete args
args = parser.parse_args()


# coding:utf-8
import configparser
cfgpath = os.path.join(args.init)
conf = configparser.ConfigParser()
conf.read(cfgpath,encoding="utf-8")

conf_name = (\
    conf.get('fine','prefix') + '_' + conf.get('set', 'user') + '_lambda'+conf.get('fine','lambda')+'_bias'+conf.get('fine', 'bias')+ '_threshold_lr_x' + conf.get('fine', 'threshold_lr_x') + '_threshold_lr_factor' + \
    conf.get('fine', 'threshold_lr_factor')+'_init_threshold' + str(json.loads(conf.get('fine', 'init_threshold'))[0]) \
    + '_reset_th_' + str(conf.getboolean('set', 'resumed_reset_th' )) + '_compress_' + conf.get('set', 'compress').replace('/','_') \
        ).replace('.','_').replace('[','_').replace(']','_').replace(',','_')
conf_compress = str(conf.get('set','compress'))
conf_batch_size = int(conf.get('set', 'batch_size'))

##############################################################
postquant_pth = 'Extract/prune_high_ele/run/run_4/models/C3D-ucf101_epoch-10.pth.tar' 
device = torch.device('cuda:0')
# device = torch.device('cuda:'+conf.get('set', 'device_ids').lstrip('[').rstrip(']'))
print("Device being used:", device)

# Pruning 30 epoches
nEpochs =200  # Number of epochs for training
resume_epoch = 0  # Default is 0, change if want to resume
useTest = True # See evolution of the test set when training
nTestInterval = 1 # Run on test set every nTestInterval epochs
save_epoch = 30 # Store a model every snapshot epochs
lr = 1e-4 # Learning rate

batch_size = conf_batch_size
dataset = 'ucf101' # Options: hmdb51 or ucf101

if dataset == 'hmdb51':
    num_classes=51
elif dataset == 'ucf101':
    num_classes = 101
else:
    print('We only implemented hmdb and ucf datasets.')
    raise NotImplementedError

save_dir_root = os.path.join('Extract',conf_name)
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

msglogger = apputils.config_pylogger('logging.conf',experiment_name=conf_name,output_dir=save_dir)
tflogger = TensorBoardLogger(msglogger.logdir)
tflogger.log_gradients = True
pylogger = PythonLogger(msglogger)
#if args.extract != None:
if conf_compress:
    shutil.copy(conf_compress,save_dir)
shutil.copy(sys.argv[0],save_dir)
shutil.copy('./scripts/Train/auto_threshold/C3D_model_th.py', save_dir)
shutil.copy('./scripts/Train/auto_threshold/train_model.py', save_dir)
shutil.copy(cfgpath, save_dir)

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
    model = C3D_model_th.C3D(num_classes=num_classes,pretrained=True, conf=conf)
    train_params = [
        {'params': C3D_model_th.get_1x_lr_params(model), 'lr': lr},
        {'params': C3D_model_th.get_10x_lr_params(model), 'lr': lr * 10},
        {'params': C3D_model_th.get_10x_lr_Threshold(model), 'lr': lr * conf.getint('fine', 'threshold_lr_x')}

        ]
else:
    print('We only implemented C3D and R2Plus1D models.')
    raise NotImplementedError

optimizer = optim.SGD(train_params, lr=lr, momentum=0.9, weight_decay=5e-4)
# optimizer = optim.Adam(train_params, lr=lr)
 
print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
#***********************************************************************************************
if torch.cuda.device_count() > 0:
    # print("Use", torch.cuda.device_count(), 'gpus')
    print("Use ", conf.get('set', 'device_ids'), ' gpus')
    model = nn.DataParallel(model, device_ids=json.loads(conf.get('set', 'device_ids')))
for state in optimizer.state.values():
    for k, v in state.items():
        if torch.is_tensor(v):
            state[k] = v.cuda()
model.to(device)

compression_scheduler = None

from checkpoint_self import load_checkpoint_self

if 'resumed_checkpoint_path' in conf['set']:
    print("****** resumed_checkpoint_path from {}".format(conf.get('set', 'resumed_checkpoint_path')))
    checkpoint = torch.load(conf.get('set', 'resumed_checkpoint_path'), map_location=lambda storage, loc: storage)
    resume_epoch = checkpoint['epoch'] + 1
    

if conf_compress:
    source = conf_compress
    compression_scheduler=distiller.file_config(model,optimizer, \
        conf_compress, compression_scheduler,resumed_epoch=resume_epoch if 'resumed_checkpoint_path' in conf else None)
    compression_scheduler.append_float_weight_after_quantizer()
if 'resumed_checkpoint_path' in conf['set']:
    model.load_state_dict(checkpoint['state_dict'])
    if conf.getboolean('set', 'resumed_reset_th' ):
        for layer in range(8):
            model.state_dict()['module.Threshold'][layer] = json.loads(conf.get('fine', 'init_threshold'))[layer]
    if not conf.getboolean('set', 'resumed_reset_optim' ):
        optimizer.load_state_dict(checkpoint['opt_dict'])


# sys.exit()
if compression_scheduler ==None:
    print('ERROR --------------------------No compress------------------------')

try:
    train_model.train_model(conf=conf, model=model,optimizer=optimizer, dataset=dataset, save_dir=save_dir, saveName=saveName, num_classes=num_classes, lr=lr,
        nEpochs=nEpochs, resume_epoch=resume_epoch,batch_size = batch_size,save_epoch=save_epoch, useTest=useTest, test_interval=nTestInterval,
        device = device, compression_scheduler=compression_scheduler,msglogger=msglogger, tflogger=tflogger,pylogger=pylogger,modelName=modelName,postquant_pth=postquant_pth)

except KeyboardInterrupt:
    msglogger.info('-' * 89)
    msglogger.info('Exiting from train_model early')#

