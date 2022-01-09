import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, noise_dim, training=True):
        super(Decoder, self).__init__()
        self.training = training
        self.linear = nn.Linear(noise_dim, 4 * 4 * 512)
        self.tconv2d_1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.batchnorm_1 = nn.BatchNorm2d(256)
        self.tconv2d_2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.batchnorm_2 = nn.BatchNorm2d(128)
        self.tconv2d_3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.batchnorm_3 = nn.BatchNorm2d(64)
        self.tconv2d_4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.batchnorm_4 = nn.BatchNorm2d(32)
        self.tconv2d_5 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)
        self.batchnorm_5 = nn.BatchNorm2d(3)

    def forward(self, x):
        x = self.linear(x)
        x = x.view(x.size(0), -1, 4, 4)
        for i in range(1, 6):
            x = getattr(self, 'tconv2d_' + str(i))(x)
            x = getattr(self, 'batchnorm_' + str(i))(x)
            x = F.leaky_relu(x)
            x = F.dropout(x, p=0.2, training=self.training)
        x = torch.tanh(x)
        return x


# model = Decoder(100).to("cuda")
# input = torch.randn(2, 100).to("cuda")
# output = model(input)
# print(output.shape)
