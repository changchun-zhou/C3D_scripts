from itertools import groupby
import torch 
import numpy as np 
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

sparsity = [0, 33, 40, 60, 74, 85, 97]
energy = [0.24, 0.33, 0.372, 0.508, 1.157, 2.703, 5.832]



def  statistical_distribution(extract_dir, dequant_dir, name,tensor, type, mode,scale, theshold, Number_Conv, density_wei, MACs, date_str, factor_sparsity, file_name, y_max):

    print("activation theshold:", name)
    # fp_flag_bin_wr = open(os.path.join(dequant_dir)+'/'+name+'_flag_bin.dat','w')
    fp_float_rd = open(os.path.join(extract_dir)+'/'+name+'_float.dat','r')

    array_shape = (tensor.cpu()).numpy()
    shape = array_shape.shape
    list_origin =[]
    list_delta =[]
    list_delta_all=[]

    height_array = range(shape[3])
    if type == 'wei':
        height_array = height_array[::-1] # reverse for weight

    # array_origin = [[[[[ 0 for x in range(shape[4])] for y in range(shape[3]) ] for z in range(shape[2]) ] for chn in range(shape[1]) ]for m in range(shape[0])]
    array_origin = [[[[[ 0 for x in range(shape[4]+64)] for y in range(shape[3]+64) ] for z in range(shape[2]) ] for chn in range(shape[1]) ]for m in range(shape[0])]
    array_delta_all = [[[[[ 0 for x in range(shape[4]+64)] for y in range(shape[3]+64) ] for z in range(shape[2]) ] for chn in range(shape[1]) ]for m in range(shape[0])]
    
    for batch in range(shape[0]):
        for frame in range(shape[2]):
            for height in height_array:
                for width in range(shape[4]):
                    for channel in range(shape[1]): ### NOT SUPPORT CONV1: Channel = 3
                        if mode == 'extract': 
                            fp_float_wr.write(str(array_delta_all[batch][channel][frame][height][width])+'\n')
                        elif mode == 'dequant':
                            float_rd = fp_float_rd.readline().rstrip('\n')
                            # if float(float_rd) < 0:
                            #     print(name, ' float_rd : ', float(float_rd))
                            array_origin[batch][channel][frame][height][width] = abs(round(float(float_rd)*scale)) #float-scale-int
                            list_origin.append(array_origin[batch][channel][frame][height][width])

                            if frame >= 1 and mode == 'dequant' and type=='act': # act delta 
                                array_delta_all[batch][channel][frame][height][width] = array_origin[batch][channel][frame][height][width] - array_origin[batch][channel][frame-1][height][width] #theshold=0
                                list_delta.append(array_delta_all[batch][channel][frame][height][width])
                            else:
                                array_delta_all[batch][channel][frame][height][width] = array_origin[batch][channel][frame][height][width]
                            list_delta_all.append(array_delta_all[batch][channel][frame][height][width])
    
    print("number of statistical samples", shape[0])
    print("0 portion of list_origin", float(list_origin.count(0))/len(list_origin)*100)
    print("1 portion of list_origin", float(list_origin.count(1))/len(list_origin)*100)
    print("-1 portion of list_origin", float(list_origin.count(-1))/len(list_origin)*100)

    print("0 portion of delta_all", float(list_delta_all.count(0))/len(list_delta_all)*100)
    print("1 portion of delta_all", float(list_delta_all.count(1))/len(list_delta_all)*100)
    print("-1 portion of delta_all", float(list_delta_all.count(-1))/len(list_delta_all)*100)
    print("length of origin", len(list_origin), "length of delta_all", len(list_delta_all),"length of delta", len(list_delta))
    print("origin mean", np.mean(np.array(list_origin)), "origin std", np.std(np.array(list_origin)))
    print("delta_all mean", np.mean(np.array(list_delta_all)), "delta_all std", np.std(np.array(list_delta_all)))
    print("delta mean", np.mean(np.array(list_delta)), "delta std", np.std(np.array(list_delta)))


    # draw
    # dict_list = {'origin': list_origin, 'delta_all': list_delta_all, 'delta': list_delta}
    # plt.figure(figsize=(30, 30))
    # plt.subplot(2, 4, Number_Conv+1)
    # dict_list = {'delta_all': list_delta_all}
    dict_list = {'origin': list_origin}
    plt.grid(axis='y')  
    # plt.title(" statistical distribution of " + name) 
    plt.xlabel('Threshold', fontsize=16)
    plt.ylabel('Proportion /%', fontsize=16)
    plt.xticks(np.arange(-10,10,1), fontsize=12)
    plt.yticks(np.arange(0,100,5), fontsize=12)

    position = 0
    bar_width = 0.1
    for name_view, list_view in dict_list.items():
        x=[]
        y=[]
        step = 1

        sparsity_th = np.zeros(10)
        rate_cut_th = []
        for k, g in groupby(sorted(list_view), key=lambda x: x//step):
            len_g = len(list(g))
            if k*step >= -10 and k*step <= 10 and k*step != 0:
                # print("loop: ",k,len(list(g)))
                portion = float(len_g)/float(len(list_view))*100
                print('{}-{}:{:.2f}'.format(k*step,(k+1)*step,portion))
                x.append(k*step+ bar_width*(position-1))
                y.append(portion)
            # for th in range(10):
            #     if k*step >= -th and k*step <= th:
            #         sparsity_th[th] += float(len_g)/float(len(list_view))*100

        plt.bar(bar_width*1*(Number_Conv - 4) + np.array(x) ,np.array(y),label="Conv "+str(Number_Conv+1) , width=bar_width, color=(0.1*(Number_Conv+1), 0.1*(Number_Conv+1), 0.1*(Number_Conv+1)) )
        
        # plt.bar(bar_width*4*Number_Conv - bar_width*4*4 - bar_width*2.5, float(list_origin.count(0))/len(list_origin)*100, label="sparisty of origin in Conv " + str(Number_Conv), width = bar_width, color =(0, 1.0/(Number_Conv+1), 0) )

        position += 1
    y_max = max(y) if max(y) > y_max else y_max
    plt.legend(fontsize=14)   #显示标签
    plt.gca().yaxis.set_major_locator(MultipleLocator(5))
    plt.ylim(0, round(y_max) // 5 *5 + 5 )
    plt.xlim(0, 10)
    plt.savefig(os.path.join(extract_dir)+'/statistical_distribution_bar_'+date_str+ '_' + file_name +'.svg', format='svg')
    return y_max
