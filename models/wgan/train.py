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
from torch.nn.functional import adaptive_avg_pool2d
import torch.nn.functional as F
import pynvml
import pyemd
import numpy as np
from scipy.stats import entropy
from scipy.linalg import sqrtm
from torch.nn.functional import adaptive_avg_pool2d
import torch.nn.functional as F

img_dir = "models/wgan/generated_images"
os.makedirs(img_dir, exist_ok=True)

output_file = "models/wgan/info.txt"

def gaussian_kernel(x, y, sigma=1.0):
    return torch.exp(-torch.norm(x - y) ** 2 / (2 * sigma ** 2))

def mmd(x, y, kernel=gaussian_kernel):
    m = x.size(0)
    n = y.size(0)
    xx = kernel(x, x).sum() / (m * (m - 1))
    yy = kernel(y, y).sum() / (n * (n - 1))
    xy = kernel(x, y).sum() / (m * n)
    return xx + yy - 2 * xy


# Hyperparameters etc
device = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 5e-5
BATCH_SIZE = 64
IMAGE_SIZE = 64
CHANNELS_IMG = 1
Z_DIM = 128
NUM_EPOCHS = 5
FEATURES_CRITIC = 64
FEATURES_GEN = 64
CRITIC_ITERATIONS = 5
WEIGHT_CLIP = 0.01

transforms = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]
        ),
    ]
)

dataset = datasets.MNIST(root="dataset/", transform=transforms, download=True)
# comment mnist and uncomment below if you want to train on CelebA dataset
# dataset = datasets.ImageFolder(root="celeb_dataset", transform=transforms)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# initialize gen and disc/critic
gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
critic = Discriminator(CHANNELS_IMG, FEATURES_CRITIC).to(device)
initialize_weights(gen)
initialize_weights(critic)

# initializate optimizer
opt_gen = optim.RMSprop(gen.parameters(), lr=LEARNING_RATE)
opt_critic = optim.RMSprop(critic.parameters(), lr=LEARNING_RATE)

# for tensorboard plotting
fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)
writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")
step = 0

gen.train()
critic.train()

# Initialize NVML
pynvml.nvmlInit()

# Get the number of available GPUs
num_gpus = pynvml.nvmlDeviceGetCount()

# Open the output file
with open(output_file, "w") as f:
    f.write("Epoch, Batch, MMD, EMD, GPU Utilization, GPU Memory Used, GPU Temperature\n")

    for epoch in range(NUM_EPOCHS):
        # Target labels not needed! <3 unsupervised
        for batch_idx, (data, _) in enumerate(tqdm(loader)):
            data = data.to(device)
            cur_batch_size = data.shape[0]

            # Train Critic: max E[critic(real)] - E[critic(fake)]
            for _ in range(CRITIC_ITERATIONS):
                noise = torch.randn(cur_batch_size, Z_DIM, 1, 1).to(device)
                fake = gen(noise)
                critic_real = critic(data).reshape(-1)
                critic_fake = critic(fake).reshape(-1)
                loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake))
                critic.zero_grad()
                loss_critic.backward(retain_graph=True)
                opt_critic.step()

                # clip critic weights between -0.01, 0.01
                for p in critic.parameters():
                    p.data.clamp_(-WEIGHT_CLIP, WEIGHT_CLIP)

            # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
            gen_fake = critic(fake).reshape(-1)
            loss_gen = -torch.mean(gen_fake)
            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            # Print losses occasionally and print to tensorboard
            if batch_idx % 100 == 0 and batch_idx > 0:
                gen.eval()
                critic.eval()
                print(
                    f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} \
                      Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}"
                )

                with torch.no_grad():
                    fake = gen(noise)
                    # take out (up to) 32 examples
                    img_grid_real = torchvision.utils.make_grid(
                        data[:32], normalize=True
                    )
                    img_grid_fake = torchvision.utils.make_grid(
                        fake[:32], normalize=True
                    )

                    writer_real.add_image("Real", img_grid_real, global_step=step)
                    writer_fake.add_image("Fake", img_grid_fake, global_step=step)

                    gen.eval()  # Set the generator to evaluation mode
                    fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)  # Generate fixed noise
                    # Generate a batch of images with the generator
                    fake_images = gen(fixed_noise)
                    # Create a grid of images
                    img_grid = vutils.make_grid(fake_images, padding=2, normalize=True)
                    # Save the grid of images to a file
                    vutils.save_image(img_grid, os.path.join(img_dir, f"epoch_{epoch+1}.png"))
                    gen.train()

                step += 1
                gen.train()
                critic.train()

                # Calculate MMD, EMD, and Inception Score
                mmd_score = mmd(data, fake)
                emd_score = torch.nn.functional.pairwise_distance(data, fake, p=2).mean()

                # Get GPU metrics
                gpu_metrics = []
                for i in range(num_gpus):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    gpu_metrics.append((utilization.gpu, memory_info.used, temperature))

                # Write metrics to file
                f.write(f"{epoch}, {batch_idx}, {mmd_score}, {emd_score}, {gpu_metrics}\n")

# Shutdown NVML
pynvml.nvmlShutdown()
