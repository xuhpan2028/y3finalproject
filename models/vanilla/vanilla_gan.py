import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard
import matplotlib.pyplot as plt
import os
import numpy as np
import torchvision.utils as vutils
import time
import subprocess
import psutil


output_file = "info.txt"


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
    


def gaussian_kernel(x, y, sigma=1.0):
    return torch.exp(-torch.norm(x - y) ** 2 / (2 * sigma ** 2))

def mmd(x, y, kernel=gaussian_kernel):
    m = x.size(0)
    n = y.size(0)
    xx = kernel(x, x).sum() / (m * (m - 1))
    yy = kernel(y, y).sum() / (n * (n - 1))
    y = y.view(-1, 784)  # Reshape y to have the same shape as x
    xy = kernel(x, y).sum() / (m * n)
    return xx + yy - 2 * xy

def emd(x, y):
    y = y.view(-1, 784)  # Reshape y to have the same shape as x
    return torch.nn.functional.pairwise_distance(x, y, p=2).mean()



# Hyperparameters etc.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
lr = 3e-4
z_dim = 64
image_dim = 28 * 28 * 1  # 784
batch_size = 32
num_epochs = 5
start_time = time.time()

# Initialize Discriminator and Generator
disc = Discriminator(image_dim).to(device)
gen = Generator(z_dim, image_dim).to(device)
fixed_noise = torch.randn((batch_size, z_dim)).to(device)
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
dataset = datasets.MNIST(root="dataset/", transform=transform, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
opt_disc = optim.Adam(disc.parameters(), lr=lr)
opt_gen = optim.Adam(gen.parameters(), lr=lr)
criterion = nn.BCELoss()
writer_fake = SummaryWriter(f"logs/fake")
writer_real = SummaryWriter(f"logs/real")
step = 0

# Initialize CPU and GPU performance monitoring
cpu_usage = []
gpu_usage = []

# Training loop
f = open("models/vanilla/info.txt", "w")
f.write("Epoch\tMMD Score\tEMD Score\tCPU Usage (%)\tGPU Memory Usage (GB)\n")
for epoch in range(num_epochs):
    start_epoch_time = time.time()
    for batch_idx, (real, _) in enumerate(loader):
        real = real.view(-1, 784).to(device)
        batch_size = real.shape[0]

        ### Train Discriminator
        noise = torch.randn(batch_size, z_dim).to(device)
        fake = gen(noise)
        disc_real = disc(real).view(-1)
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake).view(-1)
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        lossD = (lossD_real + lossD_fake) / 2
        disc.zero_grad()
        lossD.backward(retain_graph=True)
        opt_disc.step()

        ### Train Generator
        output = disc(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()

        if batch_idx == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)} \
                Loss D: {lossD:.4f}, loss G: {lossG:.4f}", end="\t"
            )

    # Generate and save images after each epoch
    with torch.no_grad():
        fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
        img_grid = vutils.make_grid(fake, normalize=True)
        # vutils.save_image(img_grid, f"{save_dir}/epoch_{epoch}.png")
        
    #     # Optionally display the image using matplotlib
    #     plt.figure(figsize=(8,8))
    #     plt.axis("off")
    #     plt.title(f"Generated Images After Epoch {epoch}")
    #     plt.imshow(np.transpose(img_grid.cpu().numpy(), (1,2,0)))
    #     plt.show()

    # Log generated images for TensorBoard
    data = real.reshape(-1, 1, 28, 28)
    img_grid_fake = vutils.make_grid(fake, normalize=True)
    img_grid_real = vutils.make_grid(data, normalize=True)

    writer_fake.add_image("Mnist Fake Images", img_grid_fake, global_step=step)
    writer_real.add_image("Mnist Real Images", img_grid_real, global_step=step)
    step += 1

    epoch_time = time.time() - start_epoch_time
    print(f"Epoch [{epoch}/{num_epochs}] Processing Time: {epoch_time:.2f} seconds")

    mmd_score = mmd(real, fake)
    emd_score = emd(real, fake)
    

    # Monitor CPU and GPU performance
    cpu_usage = psutil.cpu_percent()
    gpu_usage = torch.cuda.memory_allocated() / 1024 ** 3  

    f.write(f"{mmd_score}\t{emd_score}\t{cpu_usage}\t{gpu_usage}\n")
f.close()