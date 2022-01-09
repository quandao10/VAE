import torch
import neptune.new as neptune
import numpy as np
import torchvision.utils as vutils
from neptune.new.types import File

run = neptune.init(
    project="quan-ml/VAE",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI1NDgzYmY3Zi1kMjZjLTRkNjUtYWY2Ny0wODAwZDBjNjkwNGUifQ==",
)  # your credentials


def train(vae, train_loader, criterion, optimizer, args, fixed_noise, device):
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=100, verbose=True)
    run["parameters"] = vars(args)
    iters = 0
    vae.train()
    for epoch in range(args.epochs):
        for i, data in enumerate(train_loader, 0):
            data = data.to(device)
            optimizer.zero_grad()
            recovered_img, normal_dist = vae(data)
            loss, mse_loss, kl_loss = criterion(data, recovered_img, normal_dist)
            loss.backward()
            scheduler.step(loss)
            optimizer.step()

            run["total loss"].log(loss.item())
            run["mse loss"].log(mse_loss.item())
            run["kl loss"].log(kl_loss.item())

            if iters % 100 == 0:
                print('[{}/ {}] [{}/ {}] \t Iteration: {}, Loss: {}'.format(epoch, args.epochs, i, len(train_loader),
                                                                            iters, loss.item()))
                with torch.no_grad():
                    vae.decoder.training = False
                    gen_img = vae.decoder(fixed_noise)
                    gen_img = gen_img.cpu()
                run["images"].log(
                    File.as_image(np.transpose(vutils.make_grid(gen_img, padding=2, normalize=True), [1, 2, 0])))
            iters += 1
        torch.save(vae.state_dict(), 'checkpoint/vae.pth')
