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
import optuna


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
# opt_disc = optim.Adam(disc.parameters(), lr=3e-4)
# opt_gen = optim.Adam(gen.parameters(), lr=3e-4)
criterion = nn.BCELoss()
loader = DataLoader(datasets.MNIST(root="dataset/", transform=transforms.ToTensor(), download=True),
                    batch_size=32, shuffle=True)


# Define the objective function for Optuna
def objective(trial):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Suggest hyperparameters
    lr_disc = trial.suggest_float("lr_disc", 1e-5, 1e-3, log=True)
    lr_gen = trial.suggest_float("lr_gen", 1e-5, 1e-3, log=True)

    # Initialize models and optimizers
    disc = Discriminator(784).to(device)
    gen = Generator(64, 784).to(device)
    opt_disc = optim.Adam(disc.parameters(), lr=lr_disc)
    opt_gen = optim.Adam(gen.parameters(), lr=lr_gen)
    criterion = nn.BCELoss()

    # Data loading
    loader = DataLoader(datasets.MNIST(root="dataset/", transform=transforms.ToTensor(), download=True), batch_size=32, shuffle=True)

    # Setup TensorBoard
    writer = SummaryWriter(f'runs/GAN_MNIST_{trial.number}')

    # Training loop
    for epoch in range(10):  # Reduced number of epochs for faster trials
        for batch_idx, (real, _) in enumerate(loader):
            real = real.view(-1, 784).to(device)
            noise = torch.randn(real.size(0), 64).to(device)
            fake = gen(noise)

            # Train Discriminator
            lossD = (criterion(disc(real), torch.ones_like(disc(real))) +
                     criterion(disc(fake.detach()), torch.zeros_like(disc(fake)))) / 2
            opt_disc.zero_grad()
            lossD.backward()
            opt_disc.step()

            # Train Generator
            lossG = criterion(disc(fake), torch.ones_like(disc(fake)))
            opt_gen.zero_grad()
            lossG.backward()
            opt_gen.step()

        # Log losses to TensorBoard
        writer.add_scalar('Loss/Discriminator', lossD.item(), epoch)
        writer.add_scalar('Loss/Generator', lossG.item(), epoch)

    writer.close()
    
    print(f"  Loss D: {lossD.item()}", end = " ")
    print(f"  Loss G: {lossG.item()}")
    return lossD.item() + lossG.item()  # You might choose a different metric for the study's objective

# Run Optuna study
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)

# Print best hyperparameters
print("Best trial:")
trial = study.best_trial
print(f"  Value (Loss D + Loss G): {trial.value}")
for key, value in trial.params.items():
    print(f"    {key}: {value}")