import numpy as np
import time    
import sys
from fnmatch import fnmatch

import torch.autograd as autograd
sys.setrecursionlimit(1000000)
sys.path.append("../pytorch-video-recognition-master/")
sys.path.append("./scripts/Train/auto_threshold/")

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

#import C3D_model
import timeit
from datetime import datetime
import socket
#import os

from tqdm import tqdm
import math
import torch.onnx
#from tensorboardX import SummaryWriter
from torch import optim
#from torch.utils.data import DataLoader
from torch.autograd import Variable
import to_csv
from dataloaders.dataset import VideoDataset

grads = {}
 
def save_grad(name):
    def hook(grad):
        grads[name] = grad
    return hook


activation = {}
def get_activation(name):
     def hook(model,input,output):
         activation[name] = output.detach()
     return hook
'''def hook(model,input,output):
    activation[str(model)] = output.detach()
    #return hook
'''

def remove_file(filename):
    if os.path.exists(filename):
        os.remove(filename)
def get_tdvd_proportion(model, nsamples):
    return [x/nsamples for x in model.module.tdvd_proportion]

def train_model(conf,model,optimizer,dataset, save_dir, saveName, num_classes, lr,
                nEpochs,resume_epoch,batch_size, save_epoch, useTest, test_interval,
                device, compression_scheduler,msglogger, tflogger,pylogger,modelName,postquant_pth):
    
    criterion = nn.CrossEntropyLoss()  # standard crossentropy loss for classification

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10,
                                                   verbose=True, factor=0.5, min_lr=1e-8)
    criterion.to(device)

    print('Training model on {} dataset...'.format(dataset))
    num_workers = conf.getint('set', 'num_workers' )
    train_dataloader = DataLoader(VideoDataset(dataset=dataset, split='train',clip_len=16), batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    val_dataloader   = DataLoader(VideoDataset(dataset=dataset, split='val',  clip_len=16), batch_size=batch_size, num_workers=num_workers)
    test_dataloader  = DataLoader(VideoDataset(dataset=dataset, split='test', clip_len=16), batch_size=batch_size, num_workers=num_workers)

    trainval_loaders = {'train': train_dataloader, 'val': val_dataloader}
    trainval_sizes = {x: len(trainval_loaders[x].dataset) for x in ['train', 'val']}
    test_size = len(test_dataloader.dataset)
    parameters ={}
    for net,param in model.named_parameters():
        print(net,param.shape)

    epoch_acc_old = 0
    epoch_old = 0
    for epoch in range(resume_epoch, nEpochs):

        Threshold_requires_grad = conf.getboolean('fine', 'requires_grad' ) and epoch >= conf.getint('fine', 'epoch_start_th_learn')
        model.module.Threshold.requires_grad = Threshold_requires_grad

        epoch_start_time=time.time()
        if compression_scheduler:
            compression_scheduler.on_epoch_begin(epoch)
        for phase in ['train', 'val']:
            if phase == 'val' and epoch < conf.getint('set', 'epoch_start_valtest'):
                break
            start_time = timeit.default_timer()
            running_loss = 0.0
            running_corrects = 0.0
            run_number = 0
            minibatch_id = 0
            if phase == 'train':
                msglogger.info("epoch {}; lr0 {}; lr1 {}; lr2 {};".format(epoch,optimizer.state_dict()['param_groups'][0]['lr'],
                    optimizer.state_dict()['param_groups'][1]['lr'], optimizer.state_dict()['param_groups'][2]['lr']))
                model.train()
            else:
                model.eval()

            for inputs, labels in tqdm(trainval_loaders[phase]):
                scale_threshold = []
                scale_threshold.append(model.module.inputs_quant.scale.item())
                scale_threshold.append(model.module.relu1.fake_q.scale.item())
                scale_threshold.append(model.module.relu2.fake_q.scale.item())
                scale_threshold.append(model.module.relu3.fake_q.scale.item())
                scale_threshold.append(model.module.relu4.fake_q.scale.item())
                scale_threshold.append(model.module.relu5.fake_q.scale.item())
                scale_threshold.append(model.module.relu6.fake_q.scale.item())
                scale_threshold.append(model.module.relu7.fake_q.scale.item())
                model.module.scale = scale_threshold
                torch.cuda.empty_cache()
                inputs = Variable(inputs, requires_grad=False).to(device)# speed up
                labels = Variable(labels).to(device)
                optimizer.zero_grad()
                #print('inputs.device:{:s}:{:s}',inputs.device,labels.device)
                if phase == 'train':
                    if compression_scheduler:
                        compression_scheduler.on_minibatch_begin(
                        epoch,minibatch_id=minibatch_id,minibatches_per_epoch=
                        trainval_sizes[phase]/batch_size,optimizer=optimizer)

                    outputs = model(inputs)
                else:
                    with torch.no_grad():
                        if conf.getboolean('set', 'extract' ) and minibatch_id == 0 and epoch % save_epoch == 0:
                            for net,param in model.named_parameters():
                                print(net,param.shape)
                                parameters[net]=param
                        torch.cuda.empty_cache()
                        outputs = model(inputs)

                probs = nn.Softmax(dim=1)(outputs)
                # probs = outputs
                preds = torch.max(probs, 1)[1]
                loss_weight = criterion(outputs, labels)
                sp2th_weight = torch.tensor([2, 1, 1, 1, 1, 1, 1, 1]).to(device)
                dropout_th = (20*torch.sigmoid(model.module.Threshold) + 1) * sp2th_weight
                loss_th = conf.getfloat('fine', 'lambda')/(torch.norm(dropout_th) + conf.getfloat('fine', 'bias'))
                loss = (loss_weight + loss_th)
                # loss = loss_weight
                
                log_threshold_training = minibatch_id % (trainval_sizes[phase]//batch_size//10) ==0

                if phase == 'train':
                    if compression_scheduler:        
                        agg_loss = compression_scheduler.before_backward_pass(
                        epoch,minibatch_id=minibatch_id,minibatches_per_epoch=trainval_sizes[phase]/batch_size,
                        loss=loss,return_loss_components=True,optimizer=optimizer)
                        loss = agg_loss.overall_loss
                if phase == 'train':
                    loss.backward()
                    torch.cuda.synchronize()
                    if compression_scheduler:
                        compression_scheduler.before_parameter_optimization(epoch,minibatch_id=minibatch_id,minibatches_per_epoch=trainval_sizes[phase]/batch_size,optimizer=optimizer)
                    optimizer.step()
                    dropout_th /= sp2th_weight
                    if log_threshold_training:

                        msglogger.info("\n\repoch: {}, loss_weight: {}, loss_th: {}, loss: {}".format(epoch, loss_weight, loss_th, loss))
                        msglogger.info('\n\roriginal dropout_th: {}'.format(dropout_th.cpu().detach().numpy()))
                        msglogger.info('\n\roriginal Threshold: {} \n\rmodel.module.Threshold.grad: {}'.format(model.module.Threshold.cpu().detach().numpy(), model.module.Threshold.grad.cpu().detach().numpy()))

                running_loss += loss.item() * inputs.size(0)
                if phase == 'train':
                    if compression_scheduler:
                        compression_scheduler.on_minibatch_end(epoch,minibatch_id=minibatch_id,
                        minibatches_per_epoch=trainval_sizes[phase]/batch_size,optimizer=optimizer)
                running_corrects += torch.sum(preds == labels.data)

                run_number += 1
                minibatch_id = minibatch_id + 1
                if conf.getboolean('set', 'extract' ):
                    assert(batch_size==1)
                    tdvd_nsamples = minibatch_id
                    to_csv.to_csv(save_dir + '/tdvd_range_'+ str(model.module.tdvd_range)+'_scale_factor_'+str(model.module.scale_factor)+'.csv', \
                        tdvd_nsamples, get_tdvd_proportion(model, tdvd_nsamples))
            epoch_loss = running_loss / trainval_sizes[phase]
            epoch_acc = running_corrects.double() / trainval_sizes[phase]

            
            print("[{}] Epoch: {}/{} Loss: {} Acc: {}".format(phase, epoch+1, nEpochs, epoch_loss, epoch_acc))
            stop_time = timeit.default_timer()
            print("Execution time: " + str(stop_time - start_time) + "\n")
            save_list = []
            save_list.append(epoch_acc)
            save_list.extend( (dropout_th.cpu().detach().numpy()).tolist() )
            save_list.append(epoch_loss)
            save_list.append(loss_weight)
            save_list.append(loss_th)
            save_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
            save_list.append(optimizer.state_dict()['param_groups'][1]['lr'])
            save_list.append(optimizer.state_dict()['param_groups'][2]['lr'])
            save_list.extend(scale_threshold)
            to_csv.to_csv(save_dir+'/'+phase+'_acc_loss.csv', epoch, save_list)
            msglogger.info(' scale_threshold: {}'.format(scale_threshold))
        lr0_old = optimizer.state_dict()['param_groups'][0]['lr']
        lr2_old = optimizer.state_dict()['param_groups'][2]['lr']
        scheduler.step(epoch_loss)
        if optimizer.state_dict()['param_groups'][0]['lr'] != lr0_old:
            optimizer.state_dict()['param_groups'][2]['lr'] = lr2_old * conf.getfloat('fine', 'threshold_lr_factor')

        ##################################################################################
        ## Save best
        torch_version_1_6 = float((torch.__version__)[0:3]) >= 1.6
        if epoch_acc > epoch_acc_old:
            if os.path.exists(os.path.join(save_dir, 'models'))==False:
              os.makedirs(os.path.join(save_dir, 'models')) 
            remove_file(os.path.join(save_dir, 'models', '_epoch-' + str(epoch_old) + '_best.pth.tar'))
            if torch_version_1_6:
                torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'opt_dict': optimizer.state_dict(),
                }, os.path.join(save_dir, 'models', '_epoch-' + str(epoch) + '_best.pth.tar'), _use_new_zipfile_serialization=False)
            else:
                torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'opt_dict': optimizer.state_dict(),
                }, os.path.join(save_dir, 'models', '_epoch-' + str(epoch) + '_best.pth.tar'))                            
            epoch_acc_old = epoch_acc
            epoch_old = epoch
        # save last epoch
        remove_file(os.path.join(save_dir, 'models', '_epoch-' + str(epoch-1) + '_last.pth.tar'))
        if torch_version_1_6:
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'opt_dict': optimizer.state_dict(),
            }, os.path.join(save_dir, 'models', '_epoch-' + str(epoch) + '_last.pth.tar'), _use_new_zipfile_serialization=False)
        else:
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'opt_dict': optimizer.state_dict(),
            }, os.path.join(save_dir, 'models', '_epoch-' + str(epoch) + '_last.pth.tar'))

        ##################################################################################
        ## TEST 
        if useTest and epoch % test_interval == (test_interval - 1) and epoch >= conf.getint('set', 'epoch_start_valtest'):
            model.eval()
            start_time = timeit.default_timer()

            running_loss = 0.0
            running_corrects = 0.0

            for inputs, labels in tqdm(test_dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.no_grad():
                    outputs = model(inputs)
                probs = nn.Softmax(dim=1)(outputs)
                # probs = outputs
                preds = torch.max(probs, 1)[1]
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / test_size
            epoch_acc = running_corrects.double() / test_size

            msglogger.info("[test] Epoch: {}/{} Loss: {} Acc: {}".format(epoch+1, nEpochs, epoch_loss, epoch_acc))
            stop_time = timeit.default_timer()
            print("Execution time: " + str(stop_time - start_time) + "\n")
            save_list = []
            save_list.append(epoch_acc)
            save_list.extend( (dropout_th.cpu().detach().numpy()).tolist() )
            save_list.append(epoch_loss)
            save_list.append(loss_weight)
            save_list.append(loss_th)
            save_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
            save_list.append(optimizer.state_dict()['param_groups'][1]['lr'])
            save_list.append(optimizer.state_dict()['param_groups'][2]['lr'])
            save_list.extend(scale_threshold)
            to_csv.to_csv(save_dir+'/test_acc_loss.csv', epoch, save_list)

        ###################################################################################
        ## Summary
        msglogger.info('-'*89)
        msglogger.info('|end of epoch {:3d}|time {:5.2f}s | valid loss {:5.3f}|'
                       'valid ppl {:8.2f}'.format(epoch,(time.time() - epoch_start_time),
                       epoch_loss,math.exp(epoch_loss)))
        msglogger.info('-'*89)
        if compression_scheduler:
            distiller.log_weights_sparsity(model,epoch,loggers=[tflogger,pylogger])
        
        if compression_scheduler:
            compression_scheduler.on_epoch_end(epoch,optimizer,metrics={'min':epoch_loss,'max':epoch_acc})

