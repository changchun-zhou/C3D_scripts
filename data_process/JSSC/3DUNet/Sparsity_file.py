import cv2
import os
import torch 
import numpy as np
import sys
import matplotlib.pyplot as plt
sys.path.append("./scripts/data_process/Sparsity")
import tensor_to_file_act
# import statistical_distribution_origin
from matplotlib.pyplot import MultipleLocator
import argparse
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--name", type=str, default="")
parser.add_argument("--batch", type=int, default=0)
args = parser.parse_args()

# extract_src_dir = 'Extract/ref_modifyth_init0_lr0.01*2_2_0_decay0.5_0.5_0_10epoch_extract/run/run_0/extract'
# extract_src_dir = '../3D-Unet/3DUnetCNN/examples/brats2020/extract/root_lambda0.5_bias0_threshold_lr_x100_compress_._scripts_yaml_quant_prune_extract_resumed.yaml_extract_True/run/run_5/extract/Activation_epoch_43_batch_4.pth.tar'
extract_src_dir = \
'../3DUnetCNN/examples/brats2020/model.module.encoder.layers[0].blocks[0].conv1_relu.pth'
'../3DUnetCNN/examples/brats2020/extract/norm_asym_root_lambda0_bias0.01_threshold_lr_x50_compress_._scripts_yaml_quant_prune_extract_asym_resumed.yaml_extract_True/run/run_1/extract/Activation_epoch_9_batch_8.pth.tar'

# dequant_dir = 'Extract/Extract/0_data_analysis'
# file_epoch = 30
# threshold = 2
batch =args.batch

act_dict = {}
act_dict = {
            # 'inputs_quant':{'shape':[batch,   4, 112, 112, 112], 'scale': 6.55, 'zero_point': 0}, # zp 57
            'module.encoder.layers.0.blocks.0.conv1.relu' :{'shape':[batch, 32,  112,  112,  112], 'scale': 7.04, 'zero_point': 0}, # zero_point 64
            'module.encoder.layers.0.blocks.0.conv2.relu' :{'shape':[batch, 32,  112,  112,  112], 'scale': 16.77 , 'zero_point': 0}, # zero_point 64
            'module.encoder.layers.0.blocks.1.conv1.relu' :{'shape':[batch, 32,  112,  112,  112], 'scale': 4.4 , 'zero_point': 0}, # 64
            'module.encoder.layers.0.blocks.1.conv2.relu' :{'shape':[batch, 32,  112,  112,  112], 'scale': 3.58 , 'zero_point': 0},
            'module.encoder.layers.1.blocks.0.conv1.relu' :{'shape':[batch, 64,   56,   56,   56], 'scale': 4.17 , 'zero_point': 0},
            'module.encoder.layers.1.blocks.0.conv2.relu' :{'shape':[batch, 64,   56,   56,   56], 'scale': 3.57 , 'zero_point': 0},
            'module.encoder.layers.1.blocks.1.conv1.relu' :{'shape':[batch, 64,   56,   56,   56], 'scale': 4.43 , 'zero_point': 0},

            'module.encoder.layers.1.blocks.1.conv2.relu' :{'shape':[batch, 32,  112,  112,  112], 'scale': 9.04, 'zero_point': 0},
            'module.encoder.layers.2.blocks.0.conv1.relu' :{'shape':[batch, 32,  112,  112,  112], 'scale': 3.2, 'zero_point': 0},
            'module.encoder.layers.2.blocks.0.conv2.relu' :{'shape':[batch, 32,  112,  112,  112], 'scale': 3.2, 'zero_point': 0},
            'module.encoder.layers.2.blocks.1.conv1.relu' :{'shape':[batch, 32,  112,  112,  112], 'scale': 3.8, 'zero_point': 0},
            'module.encoder.layers.2.blocks.1.conv2.relu' :{'shape':[batch, 32,  112,  112,  112], 'scale': 10.5, 'zero_point': 0},
            'module.encoder.layers.3.blocks.0.conv1.relu' :{'shape':[batch, 32,  112,  112,  112], 'scale': 3.8, 'zero_point': 0},
            'module.encoder.layers.3.blocks.0.conv2.relu' :{'shape':[batch, 32,  112,  112,  112], 'scale': 7.3, 'zero_point': 0},
            'module.encoder.layers.3.blocks.1.conv1.relu' :{'shape':[batch, 32,  112,  112,  112], 'scale': 1.9, 'zero_point': 0},
            'module.encoder.layers.3.blocks.1.conv2.relu' :{'shape':[batch, 32,  112,  112,  112], 'scale': 0.9, 'zero_point': 0},
            'module.encoder.layers.4.blocks.0.conv1.relu' :{'shape':[batch, 32,  112,  112,  112], 'scale': 4.2, 'zero_point': 0},
            'module.encoder.layers.4.blocks.0.conv2.relu' :{'shape':[batch, 32,  112,  112,  112], 'scale': 18.3, 'zero_point': 0},
            'module.encoder.layers.4.blocks.1.conv1.relu' :{'shape':[batch, 32,  112,  112,  112], 'scale': 3.8, 'zero_point': 0},
            'module.encoder.layers.4.blocks.1.conv2.relu' :{'shape':[batch, 32,  112,  112,  112], 'scale': 1.7, 'zero_point': 0},

            'module.decoder.layers.0.blocks.0.conv1.relu' :{'shape':[batch, 32,  112,  112,  112], 'scale': 4.4, 'zero_point': 0},
            'module.decoder.layers.0.blocks.0.conv2.relu' :{'shape':[batch, 32,  112,  112,  112], 'scale': 18.4, 'zero_point': 0},
            'module.decoder.layers.1.blocks.0.conv1.relu' :{'shape':[batch, 32,  112,  112,  112], 'scale': 2.1, 'zero_point': 0},
            'module.decoder.layers.1.blocks.0.conv2.relu' :{'shape':[batch, 32,  112,  112,  112], 'scale': 28.5, 'zero_point': 0},
            'module.decoder.layers.2.blocks.0.conv1.relu' :{'shape':[batch, 32,  112,  112,  112], 'scale': 3.8, 'zero_point': 0},
            'module.decoder.layers.2.blocks.0.conv2.relu' :{'shape':[batch, 32,  112,  112,  112], 'scale': 21.6, 'zero_point': 0},
            'module.decoder.layers.3.blocks.0.conv1.relu' :{'shape':[batch, 32,  112,  112,  112], 'scale': 5.0, 'zero_point': 0},
            'module.decoder.layers.3.blocks.0.conv2.relu' :{'shape':[batch, 32,  112,  112,  112], 'scale': 14.7, 'zero_point': 0},
            'module.decoder.layers.4.blocks.0.conv1.relu' :{'shape':[batch, 32,  112,  112,  112], 'scale': 4.9, 'zero_point': 0},
            'module.decoder.layers.4.blocks.0.conv2.relu' :{'shape':[batch, 32,  112,  112,  112], 'scale': 16.1, 'zero_point': 0}

            }
Number_Conv = 0
plt.figure(figsize=(15, 5))
plt.suptitle(args.name + ", batch = " + str(batch) + ', in ' + extract_src_dir,fontsize=6)
import datetime   
date_str = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
plt.grid(axis='y')  
# plt.title(" statistical distribution of " + name) 
plt.xlabel('Value', fontsize=16)
plt.ylabel('Proportion /%', fontsize=16)
# plt.xticks(np.arange(-10,10,1), fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

bar_width = 0.02

# def count_nonzero(tensor, scale, value):
    
#     tensor = tensor.permute(2, 0, 1, 3, 4)
#     front, back = tensor[:-1], tensor[1:]
#     diff = back - front

#     tensor_quant = (tensor * scale).round()
#     diffbase_quant = (torch.cat([tensor[0].unsqueeze(0), diff], dim=0)*scale).round()

#     return (diffbase_quant==value).nonzero().size()[0]


tensor = torch.load(extract_src_dir)
# for net in tensor:
#     print(net, tensor[net].size())


# print("scale_threshold: ", tensor['scale_threshold'])
for net, param in act_dict.items():
    # if net == 'module.encoder.layers.1.blocks.1.conv1.relu' :
    #     break
        axis_y = []
        axis_x = np.delete(np.linspace(-80, 20, 41), 0)
        # tensor_tmp = torch.index_select(tensor[net],0,torch.LongTensor([0,2]) ).permute(2, 0, 1, 3, 4)
        tensor_tmp = tensor.permute(2, 0, 1, 3, 4)
        # tensor['scale_threshold'][Number_Conv] = 12
        print('max: {}, min: {}'.format(torch.max(tensor_tmp), torch.min(tensor_tmp)))
        front, back = tensor_tmp[:-1], tensor_tmp[1:]
        diff = back - front

        # tensor_quant = (tensor_tmp * tensor['scale_threshold'][Number_Conv] - tensor['zero_point_threshold'][Number_Conv]).round()
        tensor_quant = (tensor_tmp* 12).round()
        # print((tensor_quant<0).nonzero().size()[0]/tensor[net].numel()*100)
        # diffbase_quant = (torch.cat([tensor_tmp[0].unsqueeze(0), diff], dim=0)*tensor['scale_threshold'][Number_Conv]- tensor['zero_point_threshold'][Number_Conv]).round()
        diffbase_quant = (torch.cat([tensor_tmp[0].unsqueeze(0), diff], dim=0)*12).round()
        for x in axis_x:
            # axis_y.append( count_nonzero(tensor[net], param['scale'], x)/tensor[net].numel()*100 )
            axis_y.append( (tensor_quant==x).nonzero().size()[0]/tensor_tmp.numel()*100)
        plt.bar(bar_width*1*(Number_Conv - 4) + np.array(axis_x) ,np.array(axis_y),label="Conv "+str(Number_Conv+1) , width=bar_width, color=(0.03*(Number_Conv+1), 0.03*(Number_Conv+1), 0.03*(Number_Conv+1)) )
        del tensor_tmp
        # del tensor_quant
        del diffbase_quant
        del diff 
        del front
        del back
        torch.cuda.empty_cache()
        print(Number_Conv)
        Number_Conv += 1
        break


plt.legend(fontsize=14, labels=['conv','conv','conv','conv',  'conv','conv'])   #显示标签
plt.gca().yaxis.set_major_locator(MultipleLocator(1))
plt.gca().xaxis.set_major_locator(MultipleLocator(1))
# plt.ylim(0, 70 )
# plt.xlim(0, 10)
plt.savefig(os.path.join(extract_src_dir)+'_statistical_distribution_bar_'+date_str+ '_' + args.name +'.svg', format='svg')






