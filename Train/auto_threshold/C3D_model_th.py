import torch
import torch.nn as nn
from mypath import Path
from copy import deepcopy
import copy
import torchsnooper
import timeit
from torch.autograd import Variable
import json
import numpy as np

class C3D(nn.Module):
    """
    The C3D network.
    """
    # test sftptest ew
    def __init__(self, num_classes, pretrained=False, conf=None):
        super(C3D, self).__init__()
        self.scale = [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]

        self.Threshold = nn.Parameter(torch.Tensor(json.loads(conf.get('fine', 'init_threshold'))), requires_grad = conf.getboolean('fine', 'requires_grad' ) )

        self.register_parameter('Threshold',self.Threshold)
        self.tdvd_range = 10
        self.tdvd_proportion = ( np.zeros([(self.tdvd_range*2 + 1)*8]) ).tolist()
        self.scale_factor = 1
        self.tdvd_permute_dim = (2, 0, 1, 3, 4)

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
        self.relu11 = nn.ReLU()
        self.relu12 = nn.ReLU()
        self.relu13 = nn.ReLU()
        self.relu14 = nn.ReLU()
        self.__init_weight()

        if pretrained:
            self.__load_pretrained_weights()

    def diffbase(self, tensor_in):
        return torch.cat([(tensor_in.permute(self.tdvd_permute_dim))[0].unsqueeze(0), (tensor_in.permute(self.tdvd_permute_dim))[1:] - (tensor_in.permute(self.tdvd_permute_dim))[0]], dim=0)
    def reverseint(self, tensor):
        return tensor.round()
    def stat_tdvd_proportion(self,x, layer_idx):
        for value in range(-self.tdvd_range, self.tdvd_range + 1):
            self.tdvd_proportion[ value + self.tdvd_range + (self.tdvd_range*2 + 1)* layer_idx] += (self.reverseint(self.diffbase(x)*self.scale[layer_idx]*self.scale_factor)==value).nonzero().size()[0]/x.numel()*100

    def threshold(self, tensor_in, Threshold= 1, scale = 1.0):

        Threshold = Threshold/scale

        tensor_in = tensor_in.permute(2, 0, 1, 3, 4)
        front, back = tensor_in[: -1], tensor_in[1:]
        diff = back - front
        
        diff = torch.nn.Hardshrink(1.0)(diff/Threshold)*(Threshold.item())

        back = front + diff
        tensor_out = torch.cat([tensor_in[0].unsqueeze(0), back], dim=0)
        tensor_out = tensor_out.permute(1, 2, 0, 3, 4)

        return tensor_out

    def forward(self,x):

        scale = self.scale
        sp2th_amp = torch.tensor([20, 20, 20, 20, 20, 20, 20, 20]).to(x.device)
        sp2th_min = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1]).to(x.device)
        threshold = torch.sigmoid(self.Threshold) * sp2th_amp + sp2th_min
        
        self.stat_tdvd_proportion(x, 0)
        x = self.threshold(x,  threshold[0], scale[0])

        x = self.relu1(self.conv1(x))
        x = self.pool1(x)
        self.stat_tdvd_proportion(x, 1)
        x = self.threshold(x,  threshold[1], scale[1])

        x = self.relu2(self.conv2(x))
        x = self.pool2(x)

        self.stat_tdvd_proportion(x, 2)
        x = self.threshold(x, threshold[2], scale[2])

        torch.cuda.empty_cache()
        x = self.relu3(self.conv3a(x))
        self.stat_tdvd_proportion(x, 3)
        x = self.threshold(x,  threshold[3], scale[3])

        x = self.relu4(self.conv3b(x))
        x = self.pool3(x)

        self.stat_tdvd_proportion(x, 4)
        x = self.threshold(x,  threshold[4], scale[4])

        x = self.relu5(self.conv4a(x))

        self.stat_tdvd_proportion(x, 5)
        x = self.threshold(x, threshold[5], scale[5])
        x = self.relu6(self.conv4b(x))
        x = self.pool4(x)
        self.stat_tdvd_proportion(x, 6)
        x = self.threshold(x,  threshold[6], scale[6])
        x = self.relu7(self.conv5a(x))

        self.stat_tdvd_proportion(x, 7)
        x = self.threshold(x, threshold[7], scale[7])
        x = self.relu8(self.conv5b(x))
        x = self.pool5(x)


        x = x.view(-1, 8192)
        x = self.relu9(self.fc6(x))
        x = self.dropout1(x)
        x = self.relu10(self.fc7(x))
        x = self.dropout2(x)

        logits = self.fc8(x)

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
