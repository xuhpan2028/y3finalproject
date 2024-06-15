import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define the Discriminator and Generator classes
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

# Initialize the Generator and Discriminator
z_dim = 64
img_dim = 784
device = "cuda" if torch.cuda.is_available() else "cpu"
gen = Generator(z_dim, img_dim).to(device)
disc = Discriminator(img_dim).to(device)

# Load the saved state dictionaries
gen.load_state_dict(torch.load('savedmodel/generator_final.pth', map_location=device))
disc.load_state_dict(torch.load('savedmodel/discriminator_final.pth', map_location=device))

# Set models to evaluation mode
gen.eval()
disc.eval()

# Load the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

mnist = datasets.MNIST(root='dataset', train=True, transform=transform, download=True)
loader = torch.utils.data.DataLoader(mnist, batch_size=1000, shuffle=True)

# Get real data
dataiter = iter(loader)
real_images, _ = next(dataiter)
real_images = real_images.view(-1, 784).to(device)
real_images = real_images.cpu().detach().numpy()

# Generate fake data
z = torch.randn(1000, z_dim).to(device)
fake_images = gen(z)
fake_images = fake_images.cpu().detach().numpy()

# Plot PDF of real vs generated data
plt.figure(figsize=(10, 6))
sns.kdeplot(real_images.flatten(), label='Real Data', fill=True, alpha=0.5)
sns.kdeplot(fake_images.flatten(), label='Generated Data', fill=True, alpha=0.5)
plt.title('PDF of Real vs Generated Data')
plt.xlabel('Pixel Intensity')
plt.ylabel('Density')
plt.legend()
plt.savefig('pdf_real_vs_generated.png')
