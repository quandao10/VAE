import torch
from torch.nn import Module
import torch.nn.functional as F
import torch.nn as nn


# class EncoderBlock(Module):
#     def __init__(self, channels, kernel_size=3, halve_size=False):
#         super(EncoderBlock, self).__init__()
#         self.havel_size = halve_size
#
#         if halve_size:
#             half_channels = int(channels / 2)
#             self.conv2d_1 = nn.Conv2d(half_channels, channels, kernel_size, stride=2, padding=1)
#             self.downsample = nn.AvgPool2d(3, stride=1, padding=1)
#             self.conv2d_1x1 = nn.Conv2d(half_channels, channels, 1, stride=2, padding=0)
#             self.batchnorm_1 = nn.BatchNorm2d(half_channels)
#         else:
#             self.conv2d_1 = nn.Conv2d(channels, channels, kernel_size, stride=1, padding=1)
#             self.batchnorm_1 = nn.BatchNorm2d(channels)
#
#         self.conv2d_2 = nn.Conv2d(channels, channels, kernel_size, stride=1, padding=1)
#         self.batchnorm_2 = nn.BatchNorm2d(channels)
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#
#     def shortcut(self, x, z):
#         if self.havel_size:
#             haft_x = self.downsample(x)
#             haft_x = self.conv2d_1x1(haft_x)
#             return haft_x + z
#         else:
#             return x + z
#
#     def forward(self, x):
#         z = self.batchnorm_1(x)
#         z = F.relu(z)
#         z = self.conv2d_1(z)
#         z = self.batchnorm_2(z)
#         z = F.relu(z)
#         z = self.conv2d_2(z)
#         z = self.shortcut(x, z)
#         return z


class Encoder(Module):
    def __init__(self, noise_dims, training=True):
        super(Encoder, self).__init__()
        self.training = training
        self.conv1 = nn.Conv2d(3, 64, 3, stride=2, padding=1)
        self.batch_conv1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.batch_conv2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.batch_conv3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, 3, stride=2, padding=1)
        self.batch_conv4 = nn.BatchNorm2d(512)
        self.linear_var = nn.LazyLinear(noise_dims)
        self.batchnorm_linear = nn.BatchNorm1d(noise_dims)
        self.linear_mean = nn.LazyLinear(noise_dims)

    def forward(self, x):
        for i in range(1, 5):
            x = getattr(self, 'conv' + str(i))(x)
            x = getattr(self, 'batch_conv' + str(i))(x)
            x = F.leaky_relu(x)
            x = F.dropout(x, p=0.2, training=self.training)
        x = x.view(x.size(0), -1)
        mean = F.leaky_relu(self.batchnorm_linear(self.linear_mean(x)))
        log_var = F.leaky_relu(self.batchnorm_linear(self.linear_var(x)))
        std = torch.exp(log_var*0.5)
        return mean, std

# model = Encoder()