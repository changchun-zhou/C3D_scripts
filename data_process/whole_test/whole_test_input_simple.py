''' Generate very simple test data for Sagitta
flag 01010101
data 1111111
'''


import torch 
import numpy as np 
import os
import random
import math

# import heapq
# from Function_self import Function_self
def WRITE_BACK( File, NumCol, NumChar, data, cnt_wr, temp ):
    if NumChar == 4:
        data = hex(data & 0xffff) # signed
    elif NumChar == 2: 
        data = hex(data & 0xff) # signed
    else:
        data = data
    # data = hex(data) # signed
    temp = str(data).lstrip('0x').rstrip('L').zfill(NumChar) + temp
    cnt_wr += 1

    if cnt_wr % (NumCol/NumChar) == 0:
        if NumChar == 1:
            temp = hex(int(temp,2)).lstrip('0x').rstrip("L").zfill(32)
        File.write(temp + '\n')
        temp = ''
    return cnt_wr, temp

def tensor_to_file_act(extract_dir, dequant_dir, name,tensor, type, mode,scale, theshold):

    print("activation theshold:", name, theshold)
    array_shape = (tensor.cpu()).numpy()
    shape = array_shape.shape
    str_row = ''
    cnt_col = 0

    cnt_element = 0
    cnt_element_flag = 0
    cnt_wr_data = 0
    cnt_wr_flag = 0
    temp_data = ''
    temp_flag = ''
    fp_data_wr = open(os.path.join(dequant_dir)+'/'+'dataact_L00'+'.txt','w') # activation for delta
    fp_flag_wr = open(os.path.join(dequant_dir)+'/'+'flagact_L00'+'.txt','w') 
    Num_patch =  (math.ceil(shape[3]/16.0))* (math.ceil(shape[4]/16.0))
    Num_frame = shape[2]
    Num_block = math.ceil(shape[1]/32)
    for patch in range(Num_patch):
        for frame in range(Num_frame):
        # for frame in range(7):
            for block in range(Num_block):
                for H in range(16):
                    for W in range(16):
                        for chn in range(32):
                            tmp = abs( round(random.gauss(0, 0.230)) ) # pool1
                            if abs(tmp) >= theshold :
                                cnt_element += 1
                                flag = 1
                                cnt_wr_data, temp_data= WRITE_BACK(fp_data_wr, 32, 2, tmp, cnt_wr_data, temp_data)
                            else:
                                flag = 0
                            cnt_wr_flag, temp_flag = WRITE_BACK(fp_flag_wr, 128,1, flag, cnt_wr_flag, temp_flag)
                            cnt_element_flag += 1
    print("Activation Sparsity", 1 - float(cnt_element)/(Num_patch*Num_frame*Num_block*16*16*32))
    print("config CFGCCU_num_patch = ", Num_patch -1)
    print("config CFGCCU_num_frame = ", Num_frame -1)
    print("config CFGCCU_num_block = ", Num_block -1)
    print('config CFGGB_num_alloc/total_flgact = ', math.ceil(Num_patch*Num_frame*Num_block*16*16*32/(128*512)),'rest:',Num_patch*Num_frame*Num_block*16*16*32%(128*512))
    print('config CFGGB_num_alloc/total_act    = ', math.ceil(cnt_element/(16*512)),'rest:',cnt_element%(16*512))
    print('config CFGGB_num_loop_wei = ', Num_frame)
    for zero_pad in range( (16*512) - cnt_element%(16*512) ):
        tmp = 0
        cnt_wr_data, temp_data= WRITE_BACK(fp_data_wr, 32, 2, tmp, cnt_wr_data, temp_data)
    for zero_pad_flag in range( (128*512) - cnt_element_flag%(128*512) ):
        flag = 0
        cnt_wr_flag, temp_flag = WRITE_BACK(fp_flag_wr, 128,1, flag, cnt_wr_flag, temp_flag)

def tensor_to_file_wei(extract_dir,dequant_dir,name,tensor, type, mode,scale, theshold):

        array_shape = (tensor.cpu()).numpy()
        shape = array_shape.shape
        str_row = ''
        cnt_col = 0



        cnt_element = 0
        cnt_element_flag = 0
        cnt_wr_data = 0
        cnt_wr_flag = 0
        cnt_wr_addrwei = 0
        temp_addrwei = ''
        temp_data = ''
        temp_flag = ''

        cnt_sparsity = 0

        fp_data_wr = open(os.path.join(dequant_dir)+'/'+'datawei_L00'+'.txt','w') # activation for delta
        fp_flag_wr = open(os.path.join(dequant_dir)+'/'+'flagwei_L00'+'.txt','w')
        fp_addrwei_wr = open(os.path.join(dequant_dir)+'/'+'addrwei_L00'+'.txt','w')
        for patch in range(int(shape[0]/16.0)): # ftrgrp
            addr_element =0
            # cnt_wr_addrwei, temp_addrwei= WRITE_BACK(fp_addrwei_wr, 32, 4, addr_element, cnt_wr_addrwei, temp_addrwei)
            for weight in range(16): # [73, 67, 17, 4, 83, 25, 52, 126, 37, 41, 68, 127, 123, 49, 49, 36, 120, 13, 44, 90, 30, 42, 42, 42, 63, 5, 5, 28, 28, 21, 21, 114, 54, 54, 75, 23, 19, 16, 110, 87, 91, 12, 70, 53, 58, 69, 31, 31, 31, 8, 38, 34, 85, 92, 105, 100, 32, 39, 39, 39, 80, 20, 111, 97, 15, 35, 64, 27, 46, 26, 26, 2, 14, 43, 43, 71, 0, 3, 3, 57, 33, 106, 106, 24, 81, 82, 78, 47, 59, 103, 65, 65, 10, 10, 10, 74, 56, 56, 56, 60, 122, 101, 1, 94, 6, 66, 93, 9, 48, 29, 7, 22, 115, 112, 109, 108, 96, 40, 119, 51, 124, 62, 55, 11, 11, 18, 77, 61]
                for frame in range(shape[2]):
                # for block in range(int(shape[1]/32)):
                    for H in range(shape[3]):
                        for W in range(shape[4]):
                            cnt_wr_addrwei, temp_addrwei= WRITE_BACK(fp_addrwei_wr, 32, 4, int(addr_element/16), cnt_wr_addrwei, temp_addrwei)
                            for chn in range(shape[1]):
                                # Sort_index = weight + 16*patch
                                # Sort_index = CntWeiNotZero_FtrGrp_Sort_index[weight + 16*patch]
                                tmp = abs( round(random.normalvariate(0, 0.230)) )# conv2
                                if tmp != 0 :
                                    cnt_element += 1
                                    cnt_sparsity += 1
                                    addr_element += 1
                                    flag = 1
                                    cnt_wr_data, temp_data= WRITE_BACK(fp_data_wr, 32, 2, tmp, cnt_wr_data, temp_data)
                                else:
                                    flag = 0
                                cnt_wr_flag, temp_flag = WRITE_BACK(fp_flag_wr, 128,1, flag, cnt_wr_flag, temp_flag)
                                cnt_element_flag += 1
                for chn in range(shape[1]): # pad to 28 weight flag
                    flag = chn % 2
                    cnt_wr_flag, temp_flag = WRITE_BACK(fp_flag_wr, 128,1, flag, cnt_wr_flag, temp_flag)
                    cnt_element_flag += 1

            for zero_pad in range( (16*512) - cnt_element%(16*512) ):
                tmp = 0
                cnt_wr_data, temp_data= WRITE_BACK(fp_data_wr, 32, 2, tmp, cnt_wr_data, temp_data)
            cnt_element += (16*512) - cnt_element%(16*512)
            for zero_pad_flag in range( (128*512) - cnt_element_flag%(128*512) ):
                flag = 0
                cnt_wr_flag, temp_flag = WRITE_BACK(fp_flag_wr, 128,1, flag, cnt_wr_flag, temp_flag)
            cnt_element_flag += (128*512) - cnt_element_flag%(128*512)
        print("Weight Sparsity", 1 - float(cnt_sparsity)/(shape[4]*shape[3]*shape[2]*shape[1]*shape[0]))
        print("NumBlk_weiaddr = ", int(shape[0]/16.0))
        print("config CFGCCU_num_ftrgrp = ", int(shape[0]/16.0) -1)
        print('config CFGGB_num_alloc/total_flgwei = ', math.ceil(shape[4]*shape[3]*shape[2]*shape[1]*shape[0]/(128*512)),'rest:',shape[4]*shape[3]*shape[2]*shape[1]*shape[0]%(128*512))
        print('config CFGGB_num_alloc/total_wei = ', math.ceil(cnt_element/(16*512)),'rest:',cnt_element%(16*512))
        print('config CFGGB_num_loop_act = ', int(shape[0]/16.0))

random.seed(000)
net = 'conv2'
tensor = torch.ones([8,  64, 16,  14,  14])
scale = 1
tensor_to_file_act(extract_dir='', dequant_dir='./gen_input_data', \
        name='Activation_'+str(0)+'_'+net,tensor=tensor,type='act',mode='dequant', scale=scale, theshold=1) # >= theshold

tensor = torch.ones([32, 64, 3, 3, 3 ])
tensor_to_file_wei(extract_dir='', dequant_dir='./gen_input_data',\
        name='Weight_'+str(0)+'_'+net,tensor=tensor,type='wei',mode='dequant', scale=scale, theshold=1)

