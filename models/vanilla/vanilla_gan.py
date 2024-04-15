import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
import time
import psutil
import sys

# Helper functions for MMD and EMD
def gaussian_kernel(x, y, sigma=1.0):
    return torch.exp(-torch.norm(x - y, dim=1) ** 2 / (2 * sigma ** 2))

def mmd(x, y, kernel=gaussian_kernel):
    m = x.size(0)
    n = y.size(0)
    xx = kernel(x, x).mean()
    yy = kernel(y, y).mean()
    xy = kernel(x, y).mean()
    return xx + yy - 2 * xy

def emd(x, y):
    return torch.nn.functional.pairwise_distance(x, y, p=2).mean()

# Setting up TensorBoard
writer = SummaryWriter('runs/GAN_MNIST')

# Initialize Discriminator and Generator
class Discriminator(nn.Module):
    def __init__(self, img_dim):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(img_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.disc(x)

class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, img_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.gen(x)

device = "cuda" if torch.cuda.is_available() else "cpu"
disc = Discriminator(784).to(device)
gen = Generator(64, 784).to(device)
opt_disc = optim.Adam(disc.parameters(), lr=3e-4)
opt_gen = optim.Adam(gen.parameters(), lr=3e-4)
criterion = nn.BCELoss()
loader = DataLoader(datasets.MNIST(root="dataset/", transform=transforms.ToTensor(), download=True),
                    batch_size=32, shuffle=True)

# Training Loop
for epoch in range(100):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.view(-1, 784).to(device)
        noise = torch.randn(real.size(0), 64).to(device)
        fake = gen(noise)

        # Discriminator loss
        lossD = (criterion(disc(real), torch.ones_like(disc(real))) +
                 criterion(disc(fake.detach()), torch.zeros_like(disc(fake)))) / 2
        opt_disc.zero_grad()
        lossD.backward()
        opt_disc.step()

        # Generator loss
        lossG = criterion(disc(fake), torch.ones_like(disc(fake)))
        opt_gen.zero_grad()
        lossG.backward()
        opt_gen.step()

        # Log losses to TensorBoard
        writer.add_scalar('Loss/Discriminator', lossD.item(), epoch * len(loader) + batch_idx)
        writer.add_scalar('Loss/Generator', lossG.item(), epoch * len(loader) + batch_idx)

        # Clear line before printing
        sys.stdout.write("\r")
        sys.stdout.flush()
        # Print progress, updating the same line
        print(f'Epoch [{epoch+1}/100], Batch {batch_idx+1}/{len(loader)}, Loss D: {lossD.item():.4f}, Loss G: {lossG.item():.4f}', end='')

    # Calculate and log MMD and EMD at the end of each epoch
    mmd_score = mmd(real, fake.detach())
    emd_score = emd(real.flatten(start_dim=1), fake.detach().flatten(start_dim=1))
    writer.add_scalar('MMD Score', mmd_score.item(), epoch)
    writer.add_scalar('EMD Score', emd_score.item(), epoch)

    # Log images at the end of each epoch
    with torch.no_grad():
        fake_images = gen(torch.randn(32, 64).to(device)).reshape(-1, 1, 28, 28)
        img_grid = vutils.make_grid(fake_images, normalize=True)
        writer.add_image('Generated Images', img_grid, epoch)

    # Move to the next line after each epoch
    print()

# Close TensorBoard writer
writer.close()