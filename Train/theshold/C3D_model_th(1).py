import torch
import torch.nn as nn
from mypath import Path
from copy import deepcopy
import copy
import torchsnooper

class C3D(nn.Module):
    """
    The C3D network.
    """

    def __init__(self, num_classes, pretrained=False):
        super(C3D, self).__init__()
        self.Switch, self.scale = False, 1.0

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
        # self.theshold()
    # def forward(self, x):
    # @torchsnooper.snoop()
    def theshold(self, tensor_in, scale):
        # batch_num, c_num, f_num, h, w= tensor_in.size()[0]
        batch_num, c_num, f_num, h, w= tensor_in.size()
        print('theshold size:',batch_num, c_num, f_num, h, w)
        tensor_in = tensor_in.permute(2, 0, 1, 3, 4)
        # tensor_out = deepcopy(tensor_in)
        front, back = tensor_in[:-1], tensor_in[1:]
        diff = scale * torch.abs(front - back)
        # comp_tensor = torch.ones(size=(batch_num, c_num, 1, h, w)) * 2
        # comp_tensor = comp_tensor.cuda(1)
        # diff = torch.cat([comp_tensor, diff], dim=2)
        # tensor_out[torch.where(diff <= 1)[0]] = 0
        select_indices = torch.where(diff <= 0.0001)[0]
        back[select_indices] = front[select_indices]
        tensor_out = torch.cat([tensor_in[0].unsequeeze(0), back], dim=0)
        tensor_out = tensor_out.permute(1, 2, 0, 3, 4)
        return tensor_out

        # for batch in range(tensor_in.shape[0]):
        #     for channel in range(tensor_in.shape[1]):
        #         for frame in range(1, tensor_in.shape[2]):
        #             for height in range(tensor_in.shape[3]):
        #                 for width in range(tensor_in.shape[4]):
        #                     if scale*abs(tensor_in[batch][channel][frame][height][width] - 
        #                        tensor_in[batch][channel][frame-1][height][width]) <= 1 : # Theshold
        #                        tensor_out[batch][channel][frame][height][width] = 0 # set to zero

        #                        tensor_out[batch][channel][frame][height][width] = 
        #                         tensor_in[batch][channel][frame-1][height][width] # set to origin

        # return  tensor_out
    def forward(self,x):
        Switch, scale= self.Switch, self.scale
        # if Switch :
        #     x = theshold(x, scale)

        x = self.relu1(self.conv1(x))
        x = self.pool1(x)

        if Switch :
            x = self.theshold(x, scale)

        x = self.relu2(self.conv2(x))
        x = self.pool2(x)

        # if Switch :
        #     x = theshold(x, scale)

        x = self.relu3(self.conv3a(x))

        # if Switch :
        #     x = theshold(x, scale)

        x = self.relu4(self.conv3b(x))
        x = self.pool3(x)

        # if Switch :
        #     x = theshold(x, scale)

        x = self.relu5(self.conv4a(x))

        # if Switch :
        #     x = theshold(x, scale)

        x = self.relu6(self.conv4b(x))
        x = self.pool4(x)

        # if Switch :
        #     x = theshold(x, scale)

        x = self.relu7(self.conv5a(x))

        # if Switch :
        #     x = theshold(x, scale)

        x = self.relu8(self.conv5b(x))
        x = self.pool5(x)

        # if Switch :
        #     x = theshold(x, scale)

        x = x.view(-1, 8192)
        x = self.relu9(self.fc6(x))
        x = self.dropout1(x)
        x = self.relu10(self.fc7(x))
        x = self.dropout2(x)

        logits = self.fc8(x)

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

if __name__ == "__main__":
    inputs = torch.rand(1, 3, 16, 112, 112)
    Switch = False
    scale = 1.0
    net = C3D(num_classes=101, pretrained=True)

    outputs = net.forward(inputs, scale, Switch)
    print(outputs.size())