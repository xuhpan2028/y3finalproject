import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from model import Discriminator, Generator, initialize_weights
import subprocess
import os
import torchvision.utils as vutils
from scipy.stats import entropy
from scipy.linalg import sqrtm
import torch.nn.functional as F
import pynvml
import numpy as np
import optuna


def gaussian_kernel(x, y, sigma=1.0):
    return torch.exp(-torch.norm(x - y) ** 2 / (2 * sigma ** 2))

def mmd(x, y, kernel=gaussian_kernel):
    m = x.size(0)
    n = y.size(0)
    xx = kernel(x, x).sum() / (m * (m - 1))
    yy = kernel(y, y).sum() / (n * (n - 1))
    xy = kernel(x, y).sum() / (m * n)
    return xx + yy - 2 * xy

# Hyperparameters
device = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 5e-5
BATCH_SIZE = 64
IMAGE_SIZE = 64
CHANNELS_IMG = 1
Z_DIM = 128
NUM_EPOCHS = 100
FEATURES_CRITIC = 64
FEATURES_GEN = 64
CRITIC_ITERATIONS = 5
WEIGHT_CLIP = 0.01

transforms = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)])
])

dataset = datasets.MNIST(root="dataset/", transform=transforms, download=True)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
critic = Discriminator(CHANNELS_IMG, FEATURES_CRITIC).to(device)
initialize_weights(gen)
initialize_weights(critic)

opt_gen = optim.RMSprop(gen.parameters(), lr=LEARNING_RATE)
opt_critic = optim.RMSprop(critic.parameters(), lr=LEARNING_RATE)

writer = SummaryWriter("runs/wgan")
step = 0

gen.train()
critic.train()

pynvml.nvmlInit()
num_gpus = pynvml.nvmlDeviceGetCount()

for epoch in range(NUM_EPOCHS):
    for batch_idx, (data, _) in tqdm(enumerate(loader)):
        data = data.to(device)
        cur_batch_size = data.shape[0]

        # Train Critic
        for _ in range(CRITIC_ITERATIONS):
            noise = torch.randn(cur_batch_size, Z_DIM, 1, 1).to(device)
            fake = gen(noise)
            critic_real = critic(data).reshape(-1)
            critic_fake = critic(fake).reshape(-1)
            loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake))
            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()
            for p in critic.parameters():
                p.data.clamp_(-WEIGHT_CLIP, WEIGHT_CLIP)

        # Train Generator
        gen_fake = critic(fake).reshape(-1)
        loss_gen = -torch.mean(gen_fake)
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        if batch_idx % 100 == 0:
            # Logging
            writer.add_scalar("Loss/Critic", loss_critic.item(), global_step=step)
            writer.add_scalar("Loss/Generator", loss_gen.item(), global_step=step)

            with torch.no_grad():
                fake_images = gen(torch.randn(32, Z_DIM, 1, 1).to(device))
                img_grid = vutils.make_grid(fake_images, padding=2, normalize=True)
                writer.add_image("Generated Images", img_grid, global_step=step)

            mmd_score = mmd(data, fake)
            emd_score = F.pairwise_distance(data, fake, p=2).mean()
            writer.add_scalar("Metric/MMD", mmd_score.item(), global_step=step)
            writer.add_scalar("Metric/EMD", emd_score.item(), global_step=step)

            for i in range(num_gpus):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                writer.add_scalar(f"GPU_{i}/Utilization", utilization.gpu, global_step=step)
                writer.add_scalar(f"GPU_{i}/Memory_Used", memory_info.used / 1024**2, global_step=step)
                writer.add_scalar(f"GPU_{i}/Temperature", temperature, global_step=step)

            step += 1

pynvml.nvmlShutdown()
writer.close()
