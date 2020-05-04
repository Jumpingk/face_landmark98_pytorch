'''Without BN, the start learning rate should be 0.01
The input of all models is 224x224
(c) DongQi Wang
'''
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class AlexNet(nn.Module):
    def __init__(self, num_classes):
        super(AlexNet, self).__init__()
        self.alex_net = models.AlexNet(num_classes=num_classes)
        self.base_net = self.alex_net.features
        self.pooling  = nn.AvgPool2d(3)
        self.fc = nn.Linear(1024, num_classes)
    def forward(self, x):  # input [batch_size, 3, 224, 224] 
        x = self.base_net(x)  # ==> [batch_size, 256, 6, 6] 
        x = self.pooling(x)  # ==> [batch_size, 256, 2, 2]
        print(x.size())
        exit()
        x = x.view(x.size(0), -1)  # ==> [batch_size, 1024]
        x = self.fc(x)  # ==> [batch_size, num_classes]
        return x


class ImprovedAlexNet(nn.Module):
    def __init__(self, num_classes):
        super(ImprovedAlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5),
            nn.ReLU(inplace=True),  # 令inplace=True: 原地计算，节省运算内存
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Linear(2304, num_classes)
    def forward(self, x):  # input [batch_size, 3, 224, 224] 
        x = self.features(x)  # ==> [batch_size, 256, 3, 3]
        x = x.view(x.size(0), -1)  # ==> [batch_size, 2304]
        x = F.dropout(x,inplace=True)
        x = self.fc(x)  # ==> [batch_size, num_classes]
        return x


class MobileNetV2(nn.Module):
    def __init__(self,num_classes):
        super(MobileNetV2,self).__init__()
        self.pretrain_net = models.mobilenet_v2(pretrained=True)
        self.base_net = self.pretrain_net.features
        self.pooling  = nn.AvgPool2d(3)
        self.fc = nn.Linear(5120, num_classes)
    def forward(self,x):  # input [batch_size, 3, 224, 224]
        x = self.base_net(x)  # [batch_size, 3, 224, 224] ==> [batch_size, 1280, 7, 7]
        x = self.pooling(x)  # [batch_size, 1280, 7, 7] ==> [batch_size, 1280, 2, 2]
        x = x.view(x.size(0), -1)  # [batch_size, 1280, 2, 2] ==> [batch_size, 5120]   ## 5120 = 1280*2*2
        x = self.fc(x)  # [batch_size, 5120]  ==> [batch_size， num_classes]
        return x


class SqueezeNet(nn.Module):
    def __init__(self,num_classes):
        super(SqueezeNet,self).__init__()
        self.pretrain_net = models.squeezenet1_1(pretrained=False)  # 修改
        self.base_net = self.pretrain_net.features
        self.pooling  = nn.AvgPool2d(3)
        self.fc = nn.Linear(8192,num_classes)
    def forward(self,x):
        x = self.base_net(x)  # [batch_size, 3, 224, 224] ==> [batch_size, 512, 13, 13]
        x = self.pooling(x)  # ==> [batch_size, 512, 4, 4] 
        x = x.view(x.size(0), -1)  # ==> [batch_size, 8192]
        x = self.fc(x)  # [batch_size, 8192] ==> [batch_size, num_classes]
        return x

