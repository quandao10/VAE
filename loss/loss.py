import torch


def loss(img, recovered_img, normal_dist):
    mse_loss = torch.nn.MSELoss()(img, recovered_img) * 1e4
    standard_norm = torch.distributions.Normal(torch.zeros_like(normal_dist.loc),
                                               torch.ones_like(normal_dist.scale))
    kl_loss = torch.distributions.kl_divergence(normal_dist, standard_norm).sum(-1).mean()
    return mse_loss + kl_loss, mse_loss, kl_loss
