import random
import pandas as pd
from datetime import datetime
import numpy as np
import os
#创建train_acc.csv和var_acc.csv文件，记录loss和accuracy
# df = pd.DataFrame(columns=['time','step','train Loss','training accuracy'])#列名

#初始化train数据
def to_csv(CSV_PATH, t_step, list_write):
    
    if not os.path.exists(CSV_PATH):
        data = pd.DataFrame(columns=['time','step','accuracy',\
            'threshold0', 'threshold1','threshold2','threshold3','threshold4','threshold5','threshold6', 'threshold7', \
            'epoch_loss', 'loss_weight', 'loss_th',\
            'lr0', 'lr1', 'lr_th',\
            'scale0', 'scale1', 'scale2', 'scale3','scale4', 'scale5','scale6', 'scale7'])
        data.to_csv(CSV_PATH,mode='a',index=False)
    
    time = "%s"%datetime.now()#获取当前时间
    step = "Step[%d]"%t_step
    for i in range(len(list_write)):
        list_write[i] = "%.7f"%list_write[i]
    list = [time,step] + list_write
    #由于DataFrame是Pandas库中的一种数据结构，它类似excel，是一种二维表，所以需要将list以二维列表的形式转化为DataFrame
    data = pd.DataFrame([list])
    data.to_csv(CSV_PATH,mode='a',header=False,index=False)#mode设为a,就可以向csv文件追加数据了

