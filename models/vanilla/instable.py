import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter

# Hyperparameters
batch_size = 100
learning_rate = 3e-4
learning_rate_D = 3e-4
num_epochs = 1000
latent_size = 64

# Data loading
transform = transforms.Compose([
    transforms.ToTensor()
    # transforms.Normalize(mean=(0.5,), std=(0.5,))
])

mnist = dsets.MNIST(root='./dataset', train=True, transform=transform, download=True)
data_loader = torch.utils.data.DataLoader(dataset=mnist, batch_size=batch_size, shuffle=True)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(784, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.layer3 = nn.Linear(256, 1)
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        return torch.sigmoid(out)

# Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(latent_size, 256),
            nn.LeakyReLU(0.2)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(256, 784),
            nn.Tanh()
        )
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        return out

# Initialize models
D = Discriminator()
G = Generator()

# Loss and optimizer
criterion = nn.BCELoss()
d_optimizer = optim.Adam(D.parameters(), lr=learning_rate)
g_optimizer = optim.Adam(G.parameters(), lr=learning_rate)

# TensorBoard Summary Writer
writer = SummaryWriter('runs/vanilla_instable')

# Training
for epoch in range(num_epochs):
    for i, (images, _) in enumerate(data_loader):
        # Train Discriminator
        images = images.view(batch_size, -1)
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)

        outputs = D(images)
        d_loss_real = criterion(outputs, real_labels)
        real_score = outputs

        z = torch.randn(batch_size, latent_size)
        fake_images = G(z)
        outputs = D(fake_images)
        d_loss_fake = criterion(outputs, fake_labels)
        fake_score = outputs

        d_loss = d_loss_real + d_loss_fake
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # Train Generator
        z = torch.randn(batch_size, latent_size)
        fake_images = G(z)
        outputs = D(fake_images)

        g_loss = criterion(outputs, real_labels)

        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        if (i+1) % 200 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(data_loader)}], '
                  f'D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}, '
                  f'D(x): {real_score.mean().item():.4f}, D(G(z)): {fake_score.mean().item():.4f}')
    
    # Log images to TensorBoard
    with torch.no_grad():
        z = torch.randn(batch_size, latent_size)
        fake_images = G(z)
        fake_images = fake_images.view(fake_images.size(0), 1, 28, 28)
        img_grid = vutils.make_grid(fake_images, normalize=True)
        writer.add_image('Generated Images', img_grid, epoch)

writer.close()