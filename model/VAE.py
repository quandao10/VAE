import torch
from model.Encoder import Encoder
from model.Decoder import Decoder
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, noise_dim, training=True):
        super(VAE, self).__init__()
        self.training = training
        self.encoder = Encoder(noise_dims=noise_dim, training=self.training)
        self.decoder = Decoder(noise_dim=noise_dim, training=self.training)

    def forward(self, x):
        mean, std = self.encoder(x)
        normal_dist = torch.distributions.Normal(mean, std)
        latent_z = normal_dist.rsample()
        recovered_x = self.decoder(latent_z)
        return recovered_x, normal_dist
