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
writer = SummaryWriter('runs/deeper_nn')

# Initialize Discriminator and Generator
class Discriminator(nn.Module):
    def __init__(self, img_dim):
        super().__init__()
        self.disc = nn.Sequential(
            # First layer
            nn.Linear(img_dim, 256),
            nn.LeakyReLU(0.2),

            # Second layer
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            
            # Third layer
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),

            # Output layer
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.disc(x)

class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.gen = nn.Sequential(
            # First layer
            nn.Linear(z_dim, 128),
            nn.LeakyReLU(0.2),
            
            # Second layer
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),

            # Third layer
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),

            # Output layer
            nn.Linear(512, img_dim),
            nn.Tanh(),
        )


    def forward(self, x):
        return self.gen(x)





def objective(trial):

    device = "cuda" if torch.cuda.is_available() else "cpu"


    # Suggest hyperparameters
    lr_disc = trial.suggest_float("lr_disc", 1e-5, 1e-3, log=True)
    lr_gen = trial.suggest_float("lr_gen", 1e-5, 1e-3, log=True)



    disc = Discriminator(784).to(device)
    gen = Generator(64, 784).to(device)
    opt_disc = optim.Adam(disc.parameters(), lr=lr_disc)
    opt_gen = optim.Adam(gen.parameters(), lr=lr_gen)
    criterion = nn.BCELoss()
    loader = DataLoader(datasets.MNIST(root="dataset/", transform=transforms.ToTensor(), download=True),
                        batch_size=32, shuffle=True)

    # Setup TensorBoard
    writer = SummaryWriter(f'runs/deeper_nn/trial{trial.number}')

    # Training Loop
    for epoch in range(10):
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

    # Close TensorBoard writer
    writer.close()
    return lossD.item() + lossG.item()


# Run Optuna study
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)

# Print best hyperparameters
print("Best trial:")
trial = study.best_trial
print(f"  Value (Loss D + Loss G): {trial.value}")
for key, value in trial.params.items():
    print(f"    {key}: {value}")