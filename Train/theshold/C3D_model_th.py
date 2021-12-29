import torch
import torch.nn as nn
from mypath import Path
from copy import deepcopy
import copy
import torchsnooper
import timeit
from torch.autograd import Variable
class C3D(nn.Module):
    """
    The C3D network.
    """

    def __init__(self, num_classes, pretrained=False):
        super(C3D, self).__init__()
        self.Threshold = nn.Parameter(torch.Tensor([6, 1, 1, 1,  1, 2, 2, 2]), requires_grad = True)
        self.Threshold.data.fill_(48.) # 1/4
        self.register_parameter('Threshold',self.Threshold)

        # self.Threshold0 = nn.Parameter(torch.Tensor(1), requires_grad = True)
        # self.Threshold1 = nn.Parameter(torch.Tensor(1), requires_grad = True)
        # self.Threshold2 = nn.Parameter(torch.Tensor(1), requires_grad = True)
        # self.Threshold3 = nn.Parameter(torch.Tensor(1), requires_grad = True)

        # self.Threshold0.data.fill_(30.) # 1/4
        # self.Threshold1.data.fill_(30.) # 1/4
        # self.Threshold2.data.fill_(30.) # 1/4
        # self.Threshold3.data.fill_(30.) # 1/4
        # self.register_parameter('Threshold',self.Threshold0)
        # self.register_parameter('Threshold',self.Threshold1)
        # self.register_parameter('Threshold',self.Threshold2)
        # self.register_parameter('Threshold',self.Threshold3)

        self.Multiple = nn.Parameter(torch.tensor([10.]), requires_grad = True)
        self.Switch, self.scale = False, [1,1,1,1,1,1,1,1]
        self.epoch = 0

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, num_classes)

        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        self.relu5 = nn.ReLU()
        self.relu6 = nn.ReLU()
        self.relu7 = nn.ReLU()
        self.relu8 = nn.ReLU()
        self.relu9 = nn.ReLU()
        self.relu10 = nn.ReLU()

        self.__init_weight()

        if pretrained:
            self.__load_pretrained_weights()
        # self.threshold()
    # def forward(self, x):
    # @torchsnooper.snoop()

    # batch_num, c_num, f_num, h, w= tensor_in.size()
    # def threshold(self, tensor_in, scale, Threshold= 1):
    #     tensor = tensor_in * scale
    #     x = tensor - Threshold
    #     x = nn.ReLU()(x)
    #     y =-(tensor + Threshold)
    #     y = nn.ReLU()(y)
    #     tensor_out = (x - y)/scale
    #     return tensor_out
    def threshold(self, tensor_in, scale, Threshold= 1, Save_File = False): 
        
        # # Threshold = 5*torch.sigmoid(Threshold) # Restrict to 0-20

        batch_num, c_num, f_num, h, w= tensor_in.size()

        Range = 256

        Threshold = nn.ReLU()(Threshold)

        # Threshold = Range + nn.ReLU()(Range - Threshold)

        # Save
        if self.epoch >= 0 and Save_File:
            th_in_act = open('./Extract/0_data_analysis/th_in_act_0608.txt', 'a+')

            tensor_array = tensor_in.cpu().detach().numpy()
            th_in_act.write(str(tensor_array[0][0][0]))
            th_in_act.write('\n'*4)

        tensor_in = tensor_in.permute(2, 0, 1, 3, 4) #(f_num, batch_num, c_num, h, w)
        front, back = tensor_in[:-1], tensor_in[1:]
        diff = scale * (back - front)

        # print('threshold: ', Threshold)
        # print('scale: ', scale)
        x = diff - Threshold
        x = nn.ReLU()(x)
        y =-(diff + Threshold)
        y = nn.ReLU()(y)
        diff_th = x - y
        if self.epoch >= 0 and Save_File:
            th_diff_act = open('./Extract/0_data_analysis/th_diff_act_0608.txt', 'a+')

            diff_array = diff_th.cpu().detach().numpy()
            th_diff_act.write(str(diff_array[0][0][0]))
            th_diff_act.write('\n'*4)
        # back = front + diff_th/scale

        # restore back
        # print('diff_th_quant: ', diff_th[0][0][0][0][0]/scale)
        # print('diff_th_quant: ', diff_th[1][0][0][0][0]/scale)
        # print('diff_th_quant: ', diff_th[2][0][0][0][0]/scale)
        back[0] = tensor_in[0] + diff_th[0]/scale
        for f_i in range(1, f_num-1):
            back[f_i] = back[f_i-1] + diff_th[f_i]/scale

        tensor_out = torch.cat([tensor_in[0].unsqueeze(0), back], dim=0)
        tensor_out = tensor_out.permute(1, 2, 0, 3, 4)

        return tensor_out
    # def threshold(self, tensor_in, scale, threshold = 1): 

    #     tensor_in = tensor_in.permute(2, 0, 1, 3, 4) #(f_num, batch_num, c_num, h, w)
    #     front, back = tensor_in[:-1], tensor_in[1:]
    #     diff = scale * (back - front)
    #     mask = torch.square(diff)>threshold
    #     diff *= mask
    #     back = front + diff/scale
    #     tensor_out = torch.cat([tensor_in[0].unsqueeze(0), back], dim=0)
    #     tensor_out = tensor_out.permute(1, 2, 0, 3, 4)

    #     return tensor_out
    def forward(self,x):
        Layer_Depth = 4
        
        Switch, scale= self.Switch, self.scale
        # print("inputs\n",x)
        # print("\n<<<<<<< Before TDVD >>>>>>>>>>>>>>>\n")
        # batch_num, c_num, f_num, h, w= x.size()
        # for c_i in range(3):
        #     for h_i in range(1):
        #         for w_i in range(1):
        #             for f_i in range(f_num):
        #                 print('{:.1f}'.format(x[0][c_i][f_i][h_i][w_i]), end = " ")
        #     print("\n")
        # if Switch:
        #     x = self.threshold(x, scale[0], self.Threshold[0])
            # print("threshold: ", self.Threshold[0])
        # print("<<<<<<< After TDVD >>>>>>>>>>>>>>>\n")
        # batch_num, c_num, f_num, h, w= x.size()
        # for c_i in range(3):
        #     for h_i in range(1):
        #         for w_i in range(1):
        #             for f_i in range(f_num):
        #                 print('{:.1f}'.format(x[0][c_i][f_i][h_i][w_i]), end = " ")
        #     print("\n")

        x = self.relu1(self.conv1(x))
        x = self.pool1(x)

        if Switch:
            x = self.threshold(x, scale[1], self.Threshold[1])

        x = self.relu2(self.conv2(x))
        x = self.pool2(x)

        # if Switch:
        #     x = self.threshold(x, scale[2], self.Threshold[2])
        # print('finish threshold:'+str(timeit.default_timer()))
        torch.cuda.empty_cache()
        x = self.relu3(self.conv3a(x))
        
        # if Switch:
        #     x = self.threshold(x, scale[3], self.Threshold[3])

        x = self.relu4(self.conv3b(x))
        x = self.pool3(x)


        # x = self.threshold(x, scale[4], self.Threshold[4])

        x = self.relu5(self.conv4a(x))

 
        # x = self.threshold(x, scale[5], self.Threshold[5])
        x = self.relu6(self.conv4b(x))
        x = self.pool4(x)

        # x = self.threshold(x, scale[6], self.Threshold[6])
        x = self.relu7(self.conv5a(x))


        # x = self.threshold(x, scale[7], self.Threshold[7])
        x = self.relu8(self.conv5b(x))
        x = self.pool5(x)


        x = x.view(-1, 8192)
        x = self.relu9(self.fc6(x))
        x = self.dropout1(x)
        x = self.relu10(self.fc7(x))
        x = self.dropout2(x)

        logits = self.fc8(x)
        # print('finish inference:'+str(timeit.default_timer()))
        torch.cuda.empty_cache()
        return logits



    def __load_pretrained_weights(self):
        """Initialiaze network."""
        corresp_name = {
                        # Conv1
                        "features.0.weight": "conv1.weight",
                        "features.0.bias": "conv1.bias",
                        # Conv2
                        "features.3.weight": "conv2.weight",
                        "features.3.bias": "conv2.bias",
                        # Conv3a
                        "features.6.weight": "conv3a.weight",
                        "features.6.bias": "conv3a.bias",
                        # Conv3b
                        "features.8.weight": "conv3b.weight",
                        "features.8.bias": "conv3b.bias",
                        # Conv4a
                        "features.11.weight": "conv4a.weight",
                        "features.11.bias": "conv4a.bias",
                        # Conv4b
                        "features.13.weight": "conv4b.weight",
                        "features.13.bias": "conv4b.bias",
                        # Conv5a
                        "features.16.weight": "conv5a.weight",
                        "features.16.bias": "conv5a.bias",
                         # Conv5b
                        "features.18.weight": "conv5b.weight",
                        "features.18.bias": "conv5b.bias",
                        # fc6
                        "classifier.0.weight": "fc6.weight",
                        "classifier.0.bias": "fc6.bias",
                        # fc7
                        "classifier.3.weight": "fc7.weight",
                        "classifier.3.bias": "fc7.bias",
                        }
        p_dict = torch.load(Path.model_dir())
        s_dict = self.state_dict()
        for name in p_dict:
            if name not in corresp_name:
                continue
            s_dict[corresp_name[name]] = p_dict[name]
        self.load_state_dict(s_dict)

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def get_1x_lr_params(model):
    """
    This generator returns all the parameters for conv and two fc layers of the net.
    """
    b = [model.conv1, model.conv2, model.conv3a, model.conv3b, model.conv4a, model.conv4b,
         model.conv5a, model.conv5b, model.fc6, model.fc7]
    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k

def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last fc layer of the net.
    """
    b = [model.fc8]
    for j in range(len(b)):
        for k in b[j].parameters():
            if k.requires_grad:
                yield k
def get_10x_lr_Threshold(model):
    """
    This generator returns all the parameters for the last fc layer of the net.
    """
    yield model.Threshold

if __name__ == "__main__":
    inputs = torch.rand(1, 3, 16, 112, 112)
    Switch = False
    scale = 1.0
    net = C3D(num_classes=101, pretrained=True)

    outputs = net.forward(inputs, scale, Switch)
    print(outputs.size())
