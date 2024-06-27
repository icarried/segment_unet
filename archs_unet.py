import torch
import torch.nn.functional as F
import torch.nn as nn
__all__ = ['unet']
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class conv3x3(nn.Module):    # 3x3 conv, BN, ReLU, 3x3 conv, BN, ReLU
    def __init__(self,ch_in,ch_out):
        super(conv3x3,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x
class conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(conv1x1, self).__init__()
        # 定义1x1卷积层
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv1x1(x)

class up_conv(nn.Module):   # 3x3 conv, stride=2, BN, ReLU
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(ch_in, ch_out, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class unet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(unet, self).__init__()
        ch = [4, 8, 16, 32, 64]
        input_channels = in_channels
        num_classes = out_channels

        self.encoder1 = conv3x3(input_channels, ch[0])
        self.encoder2 = conv3x3(ch[0], ch[1])
        self.encoder3 = conv3x3(ch[1], ch[2])
        self.encoder4 = conv3x3(ch[2], ch[3])
        self.encoder5 = conv3x3(ch[3], ch[4])

        self.decoder4 = conv3x3(ch[4]+ch[3], ch[3])
        self.decoder3 = conv3x3(ch[3]+ch[2], ch[2])
        self.decoder2 = conv3x3(ch[2]+ch[1], ch[1])
        self.decoder1 = conv3x3(ch[1]+ch[0], ch[0])

        self.soft = conv1x1(ch[0], num_classes)

    def forward(self, x):
        out = self.encoder1(x)
        t1 = out
        out = F.relu(F.max_pool2d(out, 2, 2))

        out = self.encoder2(out)
        t2 = out
        out = F.relu(F.max_pool2d(out, 2, 2))

        out = self.encoder3(out)
        t3 = out
        out = F.relu(F.max_pool2d(out, 2, 2))

        out = self.encoder4(out)
        t4 = out
        out = F.relu(F.max_pool2d(out, 2, 2))

        out = self.encoder5(out)

        
        # t2 = out
        out = F.relu(F.interpolate(out, scale_factor=(2, 2), mode='bilinear'))
        out = torch.cat((out, t4), dim=1)
        out = self.decoder4(out)
        out = F.relu(F.interpolate(out, scale_factor=(2, 2), mode='bilinear'))
        out = torch.cat((out, t3), dim=1)
        out = self.decoder3(out)
        out = F.relu(F.interpolate(out, scale_factor=(2, 2), mode='bilinear'))
        out = torch.cat((out, t2), dim=1)
        out = self.decoder2(out)
        out = F.relu(F.interpolate(out, scale_factor=(2, 2), mode='bilinear'))
        out = torch.cat((out, t1), dim=1)
        out = self.decoder1(out)
        # print(out.shape)
        out = self.soft(out)
        return out

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()

        # 3x3卷积路径
        self.branch = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # 残差连接
        self.short_cut = nn.Sequential()
        if in_channels != out_channels:
            self.short_cut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # 1x1卷积路径
        branch = self.branch(x)

        # 残差连接
        res_out = branch + self.short_cut(x)

        return res_out

class resunet(nn.Module):
    def __init__(self, num_classes, input_channels=3):
        super(resunet, self).__init__()
        ch = [4, 4, 4, 4, 4]

        self.encoder1 = ResBlock(input_channels, ch[0])
        self.encoder2 = ResBlock(ch[0], ch[1])
        self.encoder3 = ResBlock(ch[1], ch[2])
        self.encoder4 = ResBlock(ch[2], ch[3])
        self.encoder5 = ResBlock(ch[3], ch[4])

        self.decoder4 = ResBlock(ch[4]+ch[3], ch[3])
        self.decoder3 = ResBlock(ch[3]+ch[2], ch[2])
        self.decoder2 = ResBlock(ch[2]+ch[1], ch[1])
        self.decoder1 = ResBlock(ch[1]+ch[0], ch[0])

        self.soft = conv1x1(ch[0], num_classes)

    def forward(self, x):
        out = self.encoder1(x)
        t1 = out
        out = F.relu(F.max_pool2d(out, 2, 2))

        out = self.encoder2(out)
        t2 = out
        out = F.relu(F.max_pool2d(out, 2, 2))

        out = self.encoder3(out)
        t3 = out
        out = F.relu(F.max_pool2d(out, 2, 2))

        out = self.encoder4(out)
        t4 = out
        out = F.relu(F.max_pool2d(out, 2, 2))

        out = self.encoder5(out)

        
        # t2 = out
        out = F.relu(F.interpolate(out, scale_factor=(2, 2), mode='bilinear'))
        out = torch.cat((out, t4), dim=1)
        out = self.decoder4(out)
        out = F.relu(F.interpolate(out, scale_factor=(2, 2), mode='bilinear'))
        out = torch.cat((out, t3), dim=1)
        out = self.decoder3(out)
        out = F.relu(F.interpolate(out, scale_factor=(2, 2), mode='bilinear'))
        out = torch.cat((out, t2), dim=1)
        out = self.decoder2(out)
        out = F.relu(F.interpolate(out, scale_factor=(2, 2), mode='bilinear'))
        out = torch.cat((out, t1), dim=1)
        out = self.decoder1(out)
        # print(out.shape)
        out = self.soft(out)
        return out

    def forward(self, g, x):
        g1 = self.W_g(g)
        g1 = self.dwconv_g(g1)
        x1 = self.W_x(x)
        x1 = self.dwconv_l(x1)
        psi = self.relu(g1 + x1)
        # x = nn.Sequential(ConvNeXt_Block(dim=768, drop_rate=0, layer_scale_init_value=1e-6))
        psi = self.dwconv(psi)

        psi = self.psi(psi)

        return x * psi * g


def print_network(model, model_name):
    """Print out the network information."""
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print(model_name)
    print("The number of parameters: {}".format(num_params))


if __name__ == '__main__':
    input_tensor = torch.randn(4, 1, 256, 256)
    model = unet(num_classes=1, input_channels=1)
    model_name = "unet"
    output_tensor = model(input_tensor)
    print_network(model, model_name)
    print("Input Tensor Shape:", input_tensor.shape)
    print("Output Tensor Shape:", output_tensor.shape)