import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import seaborn as sns

# Define paths and parameters
save_path = 'savedmodel/'
z_dim = 64

# Define the same Generator model structure
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

# Load the trained generator model
device = "cuda" if torch.cuda.is_available() else "cpu"
gen = Generator(z_dim, 784).to(device)
gen.load_state_dict(torch.load(f'{save_path}wgan_generator_final.pth', map_location=device))
gen.eval()

# Data loading and transform
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
mnist = datasets.MNIST(root='mnist_data', train=True, transform=transform, download=True)
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
plt.savefig('pdf_real_vs_generated_wgan.png')
plt.show()