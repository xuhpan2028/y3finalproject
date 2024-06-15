import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import sys

sys.path.append('/homes/hp921/y3finalproject')
from utils import *

# Define the Discriminator with Q-network and Generator classes
class Discriminator(nn.Module):
    def __init__(self, img_dim, c_dim):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(img_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        self.q_net = nn.Sequential(
            nn.Linear(img_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, c_dim)
        )

    def forward(self, x):
        validity = self.disc(x)
        latent_code = self.q_net(x)
        return validity, latent_code

class Generator(nn.Module):
    def __init__(self, z_dim, c_dim, img_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim + c_dim, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, img_dim),
            nn.Tanh(),
        )

    def forward(self, z, c):
        x = torch.cat([z, c], dim=1)
        return self.gen(x)

# Hyperparameters
z_dim = 100
c_dim = 10
img_dim = 28*28  # 784
batch_size = 64
learning_rate = 3e-4
epochs = 50

# Data loading
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

mnist = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
dataloader = DataLoader(mnist, batch_size=batch_size, shuffle=True)

# Initialize models
generator = Generator(z_dim, c_dim, img_dim)
discriminator = Discriminator(img_dim, c_dim)

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate)
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate)

# Loss functions
adversarial_loss = nn.BCELoss()
continuous_loss = nn.MSELoss()

writer = SummaryWriter('/homes/hp921/y3finalproject/runs/infogan_mnist')


# Training loop
for epoch in range(epochs):
    for idx, (imgs, _) in enumerate(dataloader):
        batch_size = imgs.size(0)
        imgs = imgs.view(batch_size, -1)

        # Labels
        real = torch.ones(batch_size, 1)
        fake = torch.zeros(batch_size, 1)

        # Sample noise and latent code
        z = torch.randn(batch_size, z_dim)
        c = torch.randn(batch_size, c_dim)

        # Generate images
        gen_imgs = generator(z, c)

        # Train Discriminator
        real_loss = adversarial_loss(discriminator(imgs)[0], real)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach())[0], fake)
        d_loss = (real_loss + fake_loss) / 2

        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        # Train Generator
        validity, q_latent_code = discriminator(gen_imgs)
        g_adv_loss = adversarial_loss(validity, real)
        g_cont_loss = continuous_loss(q_latent_code, c)
        g_loss = g_adv_loss + 0.1 * g_cont_loss

        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()

        if idx % 100 == 0:
            print(f"Epoch [{epoch}/{epochs}] Batch {idx}/{len(dataloader)} \
                  Loss D: {d_loss.item()}, loss G: {g_loss.item()}")

print("Training completed.")