import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize, CenterCrop, Resize
import argparse
from dataset.dataset import FaceData
from model.VAE import VAE
from loss.loss import loss
from train.train import train


def get_args():
    parser = argparse.ArgumentParser("Train a model")

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--exp_id", type=str, default="exp_1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise_dims", type=int, default=100)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = Compose([
        Resize(128),
        CenterCrop(128),
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_dataset = FaceData(root_dir="img_align_celeba", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    fixed_noise = torch.randn(64, args.noise_dims, device=device)

    vae = VAE(args.noise_dims).to(device)
    criterion = loss
    optimizer = torch.optim.Adam(vae.parameters(), lr=args.lr, betas=(args.momentum, 0.999))

    train(vae, train_loader, criterion, optimizer, args, fixed_noise, device)

