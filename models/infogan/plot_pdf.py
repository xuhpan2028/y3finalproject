import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

# Adjust these dimensions to match the saved model
z_dim = 100  
c_dim = 10
img_dim = 28 * 28  # 784
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize the models
gen = Generator(z_dim, c_dim, img_dim).to(device)
disc = Discriminator(img_dim, c_dim).to(device)

# Load the saved state dictionaries
gen.load_state_dict(torch.load('savedmodel/infogan_mnist_gen.pth', map_location=device))
disc.load_state_dict(torch.load('savedmodel/infogan_mnist_disc.pth', map_location=device))

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
real_images = real_images.view(-1, img_dim).to(device)
real_images = real_images.cpu().detach().numpy()

# Generate fake data
z = torch.randn(1000, z_dim).to(device)
c = torch.randn(1000, c_dim).to(device)
fake_images = gen(z, c)
fake_images = fake_images.cpu().detach().numpy()

# Plot PDF of real vs generated data
plt.figure(figsize=(10, 6))
sns.kdeplot(real_images.flatten(), label='Real Data', fill=True, alpha=0.5)
sns.kdeplot(fake_images.flatten(), label='Generated Data', fill=True, alpha=0.5)
plt.title('PDF of Real vs Generated Data')
plt.xlabel('Pixel Intensity')
plt.ylabel('Density')
plt.legend()
plt.savefig('pdf_real_vs_generated_infogan.png')
plt.show()