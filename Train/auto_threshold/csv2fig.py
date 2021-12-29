#可视化
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#读取csv中指定列的数据
def csv2fig(CSV_PATH, step, FIG_PATH)
    data = pd.read_csv(CSV_PATH)
    # data_loss = data[['train Loss']] #class 'pandas.core.frame.DataFrame'
    # data_acc = data[['training accuracy']]
    data_loss = data.iloc[:,3] #即全部行，前两列的数据; 逗号前是行，逗号后是列的范围
    data_acc = data.iloc[:,2]

    x = np.arange(0,step,1)
    y1 =np.array(data_loss)#将DataFrame类型转化为numpy数组
    y2 = np.array(data_acc)
    #绘图
    plt.plot(x,y1,label="loss")
    plt.plot(x,y2,label="accuracy")
    plt.title("loss & accuracy") 
    plt.xlabel('step')
    plt.ylabel('probability')
    plt.legend()   #显示标签
    plt.savefig(FIG_PATH)
    #plt.show()
if __name__ == '__main__':
    csv2fig()
