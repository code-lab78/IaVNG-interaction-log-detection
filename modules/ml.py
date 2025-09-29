
import os
import torch
import torch.nn as nn


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, input_w, input_h, n_layer=3, n_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layers = torch.nn.ModuleList()
        for idx in range(n_layer):
            layer = self.make_layer(16*(2**idx), stride=2 if idx != 0 else 1)
            self.layers.append(layer)
        self.avg_pool = nn.AvgPool2d(8)

        out = self.conv(torch.zeros(1, 3, input_w, input_h))
        out = self.bn(out)
        out = self.relu(out)
        for layer in self.layers:
            out = layer(out)
        out = self.avg_pool(out)
        n_fc_weights = int(out.view(-1).shape[0])
        
        self.fc = nn.Linear(n_fc_weights, n_classes)


    def make_layer(self, out_channels, n_block=2, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(ResidualBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, n_block):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)


    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        for layer in self.layers:
            out = layer(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out).softmax(1)
        return out
