import torch 
import torch.nn as nn 
import torch.nn.functional as F


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )
    
    def forward(self, x):
        x = self.conv_block(x)
        return x 



class YoloV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            ConvLayer(3, 64, 7, 2),
            nn.MaxPool2d(2, 2)
        )
        self.layer2 = nn.Sequential(
            ConvLayer(64, 192, 3),
            nn.MaxPool2d(2, 2)
        )
        self.layer3 = nn.Sequential(
            ConvLayer(192, 128, 1),
            ConvLayer(128, 256, 3),
            ConvLayer(256, 256, 1),
            ConvLayer(256, 512, 3),
            nn.MaxPool2d(2, 2)
        )
        self.layer4x = self._make_layer4x()
        


        
    
    def _make_layer4x(self):
        layer_4x = nn.Sequential(
            ConvLayer(512, 256, 1),
            ConvLayer(256, 512, 3),
            ConvLayer(512, 512, 1),
            ConvLayer(512, 1024, 3),
            nn.MaxPool2d(2, 2)
        )
        return nn.ModuleList([layer_4x for _ in range(4)])
model = YoloV1()
print(model._make_layer4x())

