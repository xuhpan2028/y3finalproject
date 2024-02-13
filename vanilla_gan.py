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

def monitor_gpu_performance(output_file):
    # Command to run nvidia-smi and capture GPU performance metrics
    command = "nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.total,memory.free,memory.used,temperature.gpu --format=csv"
    
    # Run the command and capture the output
    output = subprocess.check_output(command, shell=True)
    
    # Decode the output from bytes to string
    output = output.decode("utf-8")
    
    # Write the output to the specified file (append mode)
    with open(output_file, "a") as f:
        f.write(output)

output_file = "gpu_performance.txt"


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

# Hyperparameters etc.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
lr = 3e-4
z_dim = 64
image_dim = 28 * 28 * 1  # 784
batch_size = 32
num_epochs = 100
start_time = time.time()

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

# Directory for saving generated images
save_dir = "./generated_images"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

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
        vutils.save_image(img_grid, f"{save_dir}/epoch_{epoch}.png")
        
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
    monitor_gpu_performance(output_file)

