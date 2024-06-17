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

class Generator(nn.Module):
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

device = "cuda" if torch.cuda.is_available() else "cpu"
z_dim = 64
num_classes = 10  # For MNIST
img_dim = 784  # 28*28
embed_dim = 10  # Dimension of the embedding vectors

gen = Generator(z_dim, img_dim, num_classes, embed_dim).to(device)
disc = Discriminator(img_dim, num_classes, embed_dim).to(device)

# Load the saved state dictionaries
gen.load_state_dict(torch.load('savedmodel/mnist_cgan_g.pth', map_location=device))
disc.load_state_dict(torch.load('savedmodel/mnist_cgan_d.pth', map_location=device))

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
real_images, real_labels = next(dataiter)

# Filter real images for class 0
real_images_0 = real_images[real_labels == 0]
real_images_0 = real_images_0.view(-1, 784).to(device)
real_images_0 = real_images_0.cpu().detach().numpy()

# Generate fake data for class 0
class_label = 0
z = torch.randn(1000, z_dim).to(device)
labels = torch.full((1000,), class_label, dtype=torch.long).to(device)
fake_images = gen(z, labels)
fake_images = fake_images.cpu().detach().numpy()

# Plot PDF of real vs generated data
plt.figure(figsize=(10, 6))
sns.kdeplot(real_images_0.flatten(), label='Real Data (0)', fill=True, alpha=0.5)
sns.kdeplot(fake_images.flatten(), label='Generated Data (0)', fill=True, alpha=0.5)
plt.title(f'PDF of Real vs Generated Data for Class {class_label}')
plt.xlabel('Pixel Intensity')
plt.ylabel('Density')
plt.legend()
plt.savefig('pdf_real_vs_generated_cgan_0.png')
plt.show()