import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from layers import *

class adjust_net(nn.Module):
    def __init__(self,num_input_channels=3,num_output_channels=1):
        super(adjust_net,self).__init__()
        self.input_channel=num_input_channels
        self.output_channel=num_output_channels
        self.convs = OrderedDict() 
        self.convs[("conv", 1)] = ConvBlock(self.input_channel, 32)
        self.convs[("conv", 2)] = ConvBlock(32,32)
        self.convs[("conv", 3)] = ConvBlock(32,32)
        self.convs[("conv", 4)] = nn.Conv2d(32, self.output_channel, kernel_size=1)
        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.Tanh=nn.Tanh()

    def forward(self, input_features):
        adjust_L=self.convs[("conv", 1)](input_features)
        adjust_L=self.convs[("conv", 2)](adjust_L)
        adjust_L=self.convs[("conv", 3)](adjust_L)
        adjust_L=self.convs[("conv", 4)](adjust_L)
        outputs = self.Tanh(adjust_L)
        
        return outputs

if __name__ == "__main__":

    torch.backends.cudnn.benchmark = True

    model = adjust_net().cuda()
    model.eval()

    tgt_img = torch.randn(4, 1, 256, 320).cuda()
    output=model(tgt_img)
    print(output.shape())
