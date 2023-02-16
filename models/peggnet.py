import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

class Mish(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(F.softplus(x)))
        return x

class Conv_Bn_Activation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activation, bn=True, bias=False):
        super().__init__()
        pad = (kernel_size - 1) // 2

        self.conv = nn.ModuleList()
        if bias:
            self.conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad))
        else:
            self.conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, bias=False))
        
        if bn:
            self.conv.append(nn.BatchNorm2d(out_channels))
         
        if activation == "mish":
            self.conv.append(Mish())
        elif activation == "relu":
            self.conv.append(nn.ReLU(inplace=True))
        else:
            print("activate error !!! {} {} {}".format(sys._getframe().f_code.co_filename,
                                                       sys._getframe().f_code.co_name, sys._getframe().f_lineno))
                                                       

    def forward(self, x):
        for l in self.conv:
            x = l(x)
        return x

class ResBlock(nn.Module):
    """
    Sequential residual blocks each of which consists of
    two convolution layers.
    Args:
        ch (int): number of input and output channels.
        nblocks (int): number of residual blocks.
        shortcut (bool): if True, residual tensor addition is enabled.
    """

    def __init__(self, in_ch, out_ch, nblocks=1, shortcut=True):
        super().__init__()
        self.shortcut = shortcut
        self.module_list = nn.ModuleList()
        for i in range(nblocks):
            resblock_one = nn.ModuleList()
            resblock_one.append(Conv_Bn_Activation(in_ch, out_ch, 1, 1, 'mish'))
            resblock_one.append(Conv_Bn_Activation(out_ch, in_ch, 3, 1, 'mish'))
            self.module_list.append(resblock_one)

    def forward(self, x):
        for module in self.module_list:
            h = x
            for res in module:
                h = res(h)
            x = x + h if self.shortcut else h
        return x

class SPP(nn.Module):
    def __init__(self):
        super().__init__()

        self.maxpool1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=5 // 2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=9, stride=1, padding=9 // 2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=13, stride=1, padding=13 // 2)
    
    def forward(self, x):
        m1 = self.maxpool1(x)
        m2 = self.maxpool2(x)
        m3 = self.maxpool3(x)
        out = torch.cat([m3, m2, m1, x], dim=1)
        return out

class PixelShuffle(nn.Module):
    def __init__(self, upscale_factor):
        super().__init__()
        self.upscale_factor = upscale_factor
    
    def forward(self, x):
        return F.pixel_shuffle(x, self.upscale_factor)


class PEGG_NET(nn.Module):
    def __init__(self, input_channels=1):
        super().__init__()
        
        # encoder network
        self.en_conv1 = Conv_Bn_Activation(input_channels, 32, 3, 1, 'mish')
        self.en_resblock1 = ResBlock(in_ch=32, out_ch=32, nblocks=1, shortcut=True)
        self.en_conv2 = Conv_Bn_Activation(32, 64, 3, 2, 'mish')
        
        self.en_resblock2 = ResBlock(in_ch=64, out_ch=64, nblocks=1, shortcut=True)
        self.en_conv3 = Conv_Bn_Activation(64, 128, 3, 2, 'mish')

        self.en_resblock3 = ResBlock(in_ch=128, out_ch=128, nblocks=1, shortcut=True)
        self.en_conv4 = Conv_Bn_Activation(128, 256, 3, 2, 'mish')

        self.en_conv5 = Conv_Bn_Activation(256, 256, 3, 1, 'mish')
        self.en_conv6 = Conv_Bn_Activation(256, 128, 1, 1, 'mish')
        self.en_spp = SPP()                                   

        # decoder network
        self.de_conv1 = Conv_Bn_Activation(512, 256, 1, 1, 'mish')  # concat with en_conv4 before pixel shuffling
        self.de_pixshuffle1 = PixelShuffle(2)  # out_channels = in_channels // scale**2

        self.de_conv2 = Conv_Bn_Activation(128, 128, 1, 1, 'mish')  # convolve and concat with en_conv3 layer
        self.de_pixshuffle2 = PixelShuffle(2)

        self.de_conv3 = Conv_Bn_Activation(64, 64, 1, 1, 'mish')  # convolve and concat with en_conv2 layer
        self.de_pixshuffle3 = PixelShuffle(2)

        self.de_conv4 = Conv_Bn_Activation(32, 32, 1, 1, 'relu')    # convolve and concat with en_conv1_layer

        # self.de_convt3 = nn.ConvTranspose2d(16, 16, kernel_size=9, stride=2, padding=5, output_padding=2)  # upsample image to x2 size
        # self.de_bn3 = nn.BatchNorm2d(16)
        # self.de_relu3 = nn.ReLU(inplace=True)

        # Output layers
        self.pos_output = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=(3-1)//2)
        self.cos_output = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=(3-1)//2)
        self.sin_output = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=(3-1)//2)
        self.width_output = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=(3-1)//2)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)


    def forward(self, input):
        # encoder
        # print("size of input: ", torch.Tensor.size(input))
        en_x1 = self.en_conv1(input)
        en_r1 = self.en_resblock1(en_x1)
        en_x2 = self.en_conv2(en_r1)

        en_r2 = self.en_resblock2(en_x2)
        en_x3 = self.en_conv3(en_r2)

        en_r3 = self.en_resblock3(en_x3)
        en_x4 = self.en_conv4(en_r3)

        en_x5 = self.en_conv5(en_x4)
        en_x6 = self.en_conv6(en_x5)
        en_spp = self.en_spp(en_x6)

        #decoder
        # print("size of en_spp: ",torch.Tensor.size(en_spp))
        de_x1 = self.de_conv1(en_spp)
        # print("size of de_x1: ",torch.Tensor.size(de_x1))
        de_x1 = torch.cat([de_x1, en_x4], dim=1)
        # print("size of de_x1: ",torch.Tensor.size(de_x1))
        de_ps1 = self.de_pixshuffle1(de_x1)
        # print("size of de_ps1: ",torch.Tensor.size(de_ps1))

        de_x2 = self.de_conv2(de_ps1)
        # print("size of de_x2: ",torch.Tensor.size(de_x2))
        de_x2 = torch.cat([de_x2, en_x3], dim=1)
        # print("size of de_x2: ",torch.Tensor.size(de_x2))
        de_ps2 = self.de_pixshuffle2(de_x2)
        # print("size of de_ps2: ",torch.Tensor.size(de_ps2))

        de_x3 = self.de_conv3(de_ps2)
        # print("size of de_x3: ",torch.Tensor.size(de_x3))
        de_x3 = torch.cat([de_x3, en_x2], dim=1)
        de_ps3 = self.de_pixshuffle3(de_x3)
        
        de_x4 = self.de_conv4(de_ps3)
        out = torch.cat([de_x4, en_x1], dim=1)
        
        # de_xt3 = self.de_convt3(de_x2)
        # out = self.de_relu3(self.de_bn3(de_xt3))
        
        # output layer
        pos_output = self.pos_output(out)
        cos_output = self.cos_output(out)
        sin_output = self.sin_output(out)
        width_output = self.width_output(out)

        return pos_output, cos_output, sin_output, width_output
    

    def compute_loss(self, xc, yc):
        y_pos, y_cos, y_sin, y_width = yc
        pos_pred, cos_pred, sin_pred, width_pred = self(xc)

        p_loss = F.smooth_l1_loss(pos_pred, y_pos)
        cos_loss = F.smooth_l1_loss(cos_pred, y_cos)
        sin_loss = F.smooth_l1_loss(sin_pred, y_sin)
        width_loss = F.smooth_l1_loss(width_pred, y_width)

        # p_loss = F.mse_loss(pos_pred, y_pos)
        # cos_loss = F.mse_loss(cos_pred, y_cos)
        # sin_loss = F.mse_loss(sin_pred, y_sin)
        # width_loss = F.mse_loss(width_pred, y_width)

        return {
            'loss': p_loss + cos_loss + sin_loss + width_loss,
            'losses': {
                'p_loss': p_loss,
                'cos_loss': cos_loss,
                'sin_loss': sin_loss,
                'width_loss': width_loss
            },
            'pred': {
                'pos': pos_pred,
                'cos': cos_pred,
                'sin': sin_pred,
                'width': width_pred
            }
        }


