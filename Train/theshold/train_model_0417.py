import numpy as np
import time    
import sys
from fnmatch import fnmatch
from Function_self import Function_self
sys.setrecursionlimit(1000000)
sys.path.append("../pytorch-video-recognition-master/")
sys.path.append("./scripts/Train/theshold/")
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
#import C3D_model
import timeit
from datetime import datetime
import socket
#import os
import glob
from tqdm import tqdm
import math
import torch.onnx
#from tensorboardX import SummaryWriter
from torch import optim
#from torch.utils.data import DataLoader
from torch.autograd import Variable

from dataloaders.dataset import VideoDataset
Function_self = Function_self()

activation = {}
def get_activation(name):
     def hook(model,input,output):
         activation[name] = output.detach()
     return hook
'''def hook(model,input,output):
    activation[str(model)] = output.detach()
    #return hook
'''


def train_model(args,model,optimizer,dataset, save_dir, saveName, num_classes, lr,
                nEpochs,resume_epoch,batch_size, save_epoch, useTest, test_interval,
                device, compression_scheduler,msglogger, tflogger,pylogger,modelName,postquant_pth):
    
    criterion = nn.CrossEntropyLoss()  # standard crossentropy loss for classification
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10,
                                          gamma=0.1)  # the scheduler divides the lr by 10 every 10 epochs
    criterion.to(device)
    #model.to(device)
    log_dir = os.path.join(save_dir, 'models', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    #writer = SummaryWriter(log_dir=log_dir)
    
    print('Training model on {} dataset...'.format(dataset))
    train_dataloader = DataLoader(VideoDataset(dataset=dataset, split='train',clip_len=16), batch_size=batch_size, shuffle=True, num_workers=0)
    
    val_dataloader   = DataLoader(VideoDataset(dataset=dataset, split='val',  clip_len=16), batch_size=batch_size, num_workers=0)
    test_dataloader  = DataLoader(VideoDataset(dataset=dataset, split='test', clip_len=16), batch_size=batch_size, num_workers=0)

    trainval_loaders = {'train': train_dataloader, 'val': val_dataloader}
    trainval_sizes = {x: len(trainval_loaders[x].dataset) for x in ['train', 'val']}
    test_size = len(test_dataloader.dataset)
    cnt_output = 0
    parameters ={}


    for epoch in range(resume_epoch, nEpochs):
        
        #for name,param in model.state_dict().items():
            #print('Before epoch,name:%s ; param.dim %s'%(name,param.dim()))
        epoch_start_time=time.time()
        if compression_scheduler:
            print('compression is valid')
            compression_scheduler.on_epoch_begin(epoch)
        # each epoch has a training and validation step
        for phase in ['train', 'val']:
            print('phase: {:s} size {:10d} ',phase,trainval_sizes[phase])
            if phase == 'train' and (args.qe_stats_file or args.dequant):
                continue
            start_time = timeit.default_timer()

            # reset the running loss and corrects
            running_loss = 0.0
            running_corrects = 0.0
            run_number = 0
            minibatch_id = 0
            # set model to train() or eval() mode depending on whether it is trained
            # or being validated. Primarily affects layers such as BatchNorm or Dropout.
            if phase == 'train':
                # scheduler.step() is to be called once every epoch during training
                scheduler.step()
                model.train()
            else:
                model.eval()

            for inputs, labels in tqdm(trainval_loaders[phase]):
                torch.cuda.empty_cache()
                # print('start train batch:'+str(timeit.default_timer()))
                #if args.narrow_inputs : # Because YAML quantize_inputs True 8
                    #print(inputs.shape)
                    #print('dataloader:',inputs[0][0][0][0][0])
                    #inputs = Function_self.narrow(inputs)
                # move inputs and labels to the device the training is taking place on
                #inputs = torch.ones(1,3,16,112,112)*20
                #Verify_Conv.Conv3d(inputs)
                #inputs = Variable(inputs, requires_grad=True).to(device)
                inputs = Variable(inputs, requires_grad=False).to(device)# speed up
                labels = Variable(labels).to(device)
                optimizer.zero_grad()
                #print('inputs.device:{:s}:{:s}',inputs.device,labels.device)
                if phase == 'train':
                    if compression_scheduler:
                        compression_scheduler.on_minibatch_begin(
                        epoch,minibatch_id=minibatch_id,minibatches_per_epoch=
                        trainval_sizes[phase]/batch_size,optimizer=optimizer)
                    # pack_inputs = (inputs, False, 1.0)
                    #if epoch > resume_epoch or run_number > trainval_sizes[phase]/batch_size*1/8:
                    model.Switch, model.scale = True, [1,0.0646,0.0351,0.0329,0.0303,0.0392]
                    #else:
                        #model.Switch, model.scale = False, [1,0.0646,0.0351,0.0329,0.0303,0.0392]
                    outputs = model(inputs)
                else:
                    with torch.no_grad():
                        if (args.extract or args.dequant) and minibatch_id == 0 and epoch % save_epoch == 0:
                            if args.dequant: # dequant config
                                mode = 'dequant' 
                                extract_dir = 'Extract/prune_quant_extract/run/run_6/'
                                file_epoch = 45
                                scale_inputs = 0.3818 #input_quant
                                scale_conv4a_weights = 289.3234
                                scale_pool3 = 0.0320 #relu 4
                                scale_conv4a = 0.0320 #relu 5
                                scale_relu5 = 0.0446 #relu 5
                            else:
                                mode = 'extract'
                                extract_dir = save_dir
                                file_epoch = epoch
                                scale_inputs = 1
                                scale_conv4a_weights = 1 ## Every layer is different
                                scale_pool3 = 1
                                scale_conv4a = 1
                                scale_relu5 = 1

                            layers = list(model._modules.items())
                            for [name,layer] in layers:
                                if fnmatch(name,'fc*')==False:
                                    layer.register_forward_hook(get_activation(name))
                            print("Finish layer for ",name)    
                            Function_self.tensor_to_file(extract_dir=extract_dir,name='Activation_'+str(file_epoch)+'_inputs',tensor=inputs,type='act', mode=mode, scale=scale_inputs)
                            #Function_self.tensor_vision (name=os.path.join(save_dir)+'/inputs_vision',tensor=inputs)
                            
                            # extract all weights of all layers
                            for net,param in model.named_parameters():
                                print(net,param.shape)
                                parameters[net]=param
                                #if fnmatch(net,'conv*float_weight'): 
                                    #Function_self.tensor_to_file(extract_dir=extract_dir,name='Weight_'+str(file_epoch)+'_'+net,tensor=param,type='wei',mode=mode, scale=scale_conv4a_weights)
                                #Function_self.tensor_vision (name=os.path.join(save_dir)+'/conv4a.weight_vision',tensor=parameters['conv4a.weight'])
                            
                        # pack_inputs = (inputs, False, 1.0)   
                        if epoch >= 0:
                            model.Switch, model.scale = True, [1,0.0646,0.0351,0.0329,0.0303,0.0392]
                        else:
                            model.Switch, model.scale = False, [1,0.0646,0.0351,0.0329,0.0303,0.0392]
                        outputs = model(inputs)
                        if (args.extract or args.dequant) and minibatch_id == 0 and epoch % save_epoch == 0:
                            
                            # All activations of all layers
                            #for [name,layer] in layers:
                                #if fnmatch(name,'fc*')==False and fnmatch(name,'dropout*')==False and fnmatch(name,'relu9*')==False and fnmatch(name,'relu10*')==False:
                                    #Function_self.tensor_to_file(extract_dir=extract_dir,name='Activation_'+str(file_epoch)+'_'+name,tensor=activation[name],type='act', mode='extract', scale=1)
                            if args.dequant:
                                print('Finish Dequant')
                                sys.exit()
                            #Function_self.tensor_vision (name=os.path.join(save_dir)+'/pool3_vision',tensor=activation['pool3'])
                            #if minibatch_id < 3:
                                #os.system('pause')
                probs = nn.Softmax(dim=1)(outputs)
                preds = torch.max(probs, 1)[1]

                loss = criterion(outputs, labels)
                if phase == 'train':
                    if compression_scheduler:        
                        agg_loss = compression_scheduler.before_backward_pass(
                        epoch,minibatch_id=minibatch_id,minibatches_per_epoch=trainval_sizes[phase]/batch_size,
                        loss=loss,return_loss_components=True,optimizer=optimizer)
                        loss = agg_loss.overall_loss
                if phase == 'train': 
                    loss.backward()
                    if compression_scheduler:
                        compression_scheduler.before_parameter_optimization(epoch,minibatch_id=minibatch_id,minibatches_per_epoch=trainval_sizes[phase]/batch_size,optimizer=optimizer)
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                if phase == 'train':
                    if compression_scheduler:
                        compression_scheduler.on_minibatch_end(epoch,minibatch_id=minibatch_id,
                        minibatches_per_epoch=trainval_sizes[phase]/batch_size,optimizer=optimizer)
                running_corrects += torch.sum(preds == labels.data)
                
                if phase == 'val' or (phase == 'train'and run_number%1==0 ):
                    # print('*'*8, 'trainval_sizes:', trainval_sizes[phase])
                    # print('*'*8, 'running_number :', run_number)
                    # print('*'*8, 'val phase running_corrects:', running_corrects)
                    # print('*'*8, 'val phase batch_corrects:', torch.sum(preds == labels.data))
                run_number += 1
                #if phase == 'train':
                minibatch_id = minibatch_id + 1
                    #if minibatch_id %1000 == 0:
                        #print('finish epoch :10d', epoch)
                #print('end train batch:'+str(timeit.default_timer()))
            epoch_loss = running_loss / trainval_sizes[phase]
            epoch_acc = running_corrects.double() / trainval_sizes[phase]


            print("[{}] Epoch: {}/{} Loss: {} Acc: {}".format(phase, epoch+1, nEpochs, epoch_loss, epoch_acc))
            stop_time = timeit.default_timer()
            print("Execution time: " + str(stop_time - start_time) + "\n")
        #end: phase

        ##################################################################################
        ## Save module
        if epoch % save_epoch == 0: # 0, save_epoch
            if os.path.exists(os.path.join(save_dir, 'models'))==False:
              os.makedirs(os.path.join(save_dir, 'models')) 
            
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'opt_dict': optimizer.state_dict(),
            }, os.path.join(save_dir, 'models', saveName + '_epoch-' + str(epoch) + '.pth.tar'))
            apputils.save_checkpoint(epoch+1,'my_model',model,optimizer=None,name='quantized_'+saveName+'_epoch-'+str(epoch),dir=os.path.join(save_dir,'models'))
            #torch.save(model,os.path.join(save_dir, 'models', 'whole_'+saveName + '_epoch-' + str(epoch) + '.pth.tar'))
            #apputils.save_checkpoint(epoch=1,arch='c3d',model=model,optimizer=optimizer,scheduler=compression_scheduler,
            #extras=None,is_best=False,name='test_checkpoint',dir='.')
            #model,compression_scheduler,optimizer,start_epoch=apputils.load_checkpoint(model,'test_checkpoint_checkpoint.pth.tar',
            #optimizer=optimizer,model_device='cuda:3')

            print("Save model at {}\n".format(os.path.join(save_dir, 'models', saveName + '_epoch-' + str(epoch) + '.pth.tar')))
   
        ##################################################################################
        ## TEST 
        if useTest and epoch % test_interval == (test_interval - 1):
            model.eval()
            start_time = timeit.default_timer()

            running_loss = 0.0
            running_corrects = 0.0

            for inputs, labels in tqdm(test_dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.no_grad():
                    # pack_inputs = (inputs, False, 1.0)
                    if epoch >= 0:
                        model.Switch, model.scale = True, [1,0.0646,0.0351,0.0329,0.0303,0.0392]
                    else:
                        model.Switch, model.scale = False, [1,0.0646,0.0351,0.0329,0.0303,0.0392]
                    outputs = model(inputs)
                probs = nn.Softmax(dim=1)(outputs)
                preds = torch.max(probs, 1)[1]
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / test_size
            epoch_acc = running_corrects.double() / test_size

            msglogger.info("[test] Epoch: {}/{} Loss: {} Acc: {}".format(epoch+1, nEpochs, epoch_loss, epoch_acc))
            stop_time = timeit.default_timer()
            print("Execution time: " + str(stop_time - start_time) + "\n")


        ###################################################################################
        ## Summary
        msglogger.info('-'*89)
        msglogger.info('|end of epoch {:3d}|time {:5.2f}s | valid loss {:5.3f}|'
                       'valid ppl {:8.2f}'.format(epoch,(time.time() - epoch_start_time),
                       epoch_loss,math.exp(epoch_loss)))
        msglogger.info('-'*89)
        if compression_scheduler:
            #for name,param in model.state_dict().items():
                #print('After epoch,name:%s ; param.dim %s'%(name,param))
            distiller.log_weights_sparsity(model,epoch,loggers=[tflogger,pylogger])
        
        if compression_scheduler:
            compression_scheduler.on_epoch_end(epoch,optimizer,metrics={'min':epoch_loss,'max':epoch_acc})

    #writer.close()
    #end:epoch
