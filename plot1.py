import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize Conditional Discriminator and Generator
class Discriminator(nn.Module):
    def __init__(self, img_dim, num_classes, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, embed_dim)
        self.disc = nn.Sequential(
            nn.Linear(img_dim + embed_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, labels):
        embedded_labels = self.embedding(labels)
        x = torch.cat([x, embedded_labels], dim=1)
        return self.disc(x)

class GeneratorCGAN(nn.Module):
    def __init__(self, z_dim, img_dim, num_classes, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, embed_dim)
        self.gen = nn.Sequential(
            nn.Linear(z_dim + embed_dim, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, img_dim),
            nn.Tanh(),
        )

    def forward(self, x, labels):
        embedded_labels = self.embedding(labels)
        x = torch.cat([x, embedded_labels], dim=1)
        return self.gen(x)

class GeneratorWGAN(nn.Module):
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
z_dim = 64
num_classes = 10  # For MNIST
img_dim = 784  # 28*28
embed_dim = 10  # Dimension of the embedding vectors
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CGAN Generator
gen_cgan = GeneratorCGAN(z_dim, img_dim, num_classes, embed_dim).to(device)
gen_cgan.load_state_dict(torch.load('savedmodel/mnist_cgan_g.pth', map_location=device))
gen_cgan.eval()

# Load WGAN Generator
gen_wgan = GeneratorWGAN(z_dim, img_dim).to(device)
gen_wgan.load_state_dict(torch.load('savedmodel/wgan_generator_final.pth', map_location=device))
gen_wgan.eval()

# Load the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

mnist = datasets.MNIST(root='dataset', train=True, transform=transform, download=True)
loader = torch.utils.data.DataLoader(mnist, batch_size=1000, shuffle=True)

# Get real data
dataiter = iter(loader)
real_images, real_labels = next(dataiter)
real_images = real_images.view(-1, 784).to(device)
real_images_np = real_images.cpu().detach().numpy()

# Generate fake data for CGAN for a specific class (e.g., class 0)
class_label = 0
z = torch.randn(1000, z_dim).to(device)
labels = torch.full((1000,), class_label, dtype=torch.long).to(device)
fake_images_cgan = gen_cgan(z, labels)
fake_images_cgan_np = fake_images_cgan.cpu().detach().numpy()

# Generate fake data for WGAN
z = torch.randn(1000, z_dim).to(device)
fake_images_wgan = gen_wgan(z)
fake_images_wgan_np = fake_images_wgan.cpu().detach().numpy()

# Plot PDF of real vs generated data
plt.figure(figsize=(10, 6))
sns.kdeplot(real_images_np.flatten(), label='Real Data', fill=True, alpha=0.5)
sns.kdeplot(fake_images_cgan_np.flatten(), label='Generated Data CGAN', fill=True, alpha=0.5)
sns.kdeplot(fake_images_wgan_np.flatten(), label='Generated Data WGAN', fill=True, alpha=0.5)
plt.title('PDF of Real vs Generated Data (CGAN and WGAN)')
plt.xlabel('Pixel Intensity')
plt.ylabel('Density')
plt.legend()
plt.savefig('pdf_real_vs_generated_cgan_wgan.png')
plt.show()