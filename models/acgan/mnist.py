import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import sys
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torchvision.utils import make_grid

sys.path.append('/homes/hp921/y3finalproject')
from utils import *

# Initialize Discriminator and Generator
class Discriminator(nn.Module):
    def __init__(self, img_dim, num_classes):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(img_dim, 128),
            nn.LeakyReLU(0.1),
        )
        # Binary classification for real/fake
        self.real_fake = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        # Auxiliary classifier for predicting class
        self.aux_classifier = nn.Sequential(
            nn.Linear(128, num_classes),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.disc(x)
        validity = self.real_fake(x)
        label = self.aux_classifier(x)
        return validity, label

class Generator(nn.Module):
    def __init__(self, z_dim, img_dim, num_classes):
        super().__init__()
        self.label_embedding = nn.Embedding(num_classes, num_classes)
        self.gen = nn.Sequential(
            nn.Linear(z_dim + num_classes, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, img_dim),
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        # Concatenate noise and label embedding
        label_embedding = self.label_embedding(labels)
        x = torch.cat((noise, label_embedding), dim=1)
        return self.gen(x)

# Hyperparameters
img_dim = 784  # Example for MNIST images (28*28)
z_dim = 64  # Dimension of the noise vector
num_classes = 10
batch_size = 64
lr = 3e-4
num_epochs = 50

# Data preparation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

train_data = datasets.MNIST(
    root='dataset', train=True, transform=transform, download=True
)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# Initialize models and optimizers
D = Discriminator(img_dim, num_classes).to('cuda')
G = Generator(z_dim, img_dim, num_classes).to('cuda')

optim_D = optim.Adam(D.parameters(), lr=lr)
optim_G = optim.Adam(G.parameters(), lr=lr)

adversarial_loss = nn.BCELoss()
auxiliary_loss = nn.CrossEntropyLoss()

writer = SummaryWriter('/homes/hp921/y3finalproject/runs/acgan_mnist')

# Training loop
for epoch in range(num_epochs):
    for batch_idx, (real_images, labels) in enumerate(train_loader):
        batch_size = real_images.size(0)
        
        # Adversarial ground truths
        valid = torch.ones(batch_size, 1).to('cuda')
        fake = torch.zeros(batch_size, 1).to('cuda')

        # Configure input
        real_images = real_images.view(batch_size, -1).to('cuda')
        real_images, labels = real_images.to('cuda'), labels.to('cuda')

        # -----------------
        #  Train Generator
        # -----------------

        optim_G.zero_grad()

        # Sample noise and labels as generator input
        z = torch.randn(batch_size, z_dim).to('cuda')
        gen_labels = torch.randint(0, num_classes, (batch_size,)).to('cuda')

        # Generate a batch of images
        gen_images = G(z, gen_labels)

        # Loss measures generator's ability to fool the discriminator
        validity, pred_labels = D(gen_images)
        g_loss_adv = adversarial_loss(validity, valid)
        g_loss_aux = auxiliary_loss(pred_labels, gen_labels)
        g_loss = g_loss_adv + g_loss_aux

        g_loss.backward()
        optim_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optim_D.zero_grad()

        # Loss for real images
        real_pred, real_aux = D(real_images)
        d_real_loss_adv = adversarial_loss(real_pred, valid)
        d_real_loss_aux = auxiliary_loss(real_aux, labels)

        # Loss for fake images
        fake_pred, fake_aux = D(gen_images.detach())
        d_fake_loss_adv = adversarial_loss(fake_pred, fake)
        d_fake_loss_aux = auxiliary_loss(fake_aux, gen_labels)

        # Total discriminator loss
        d_loss_adv = (d_real_loss_adv + d_fake_loss_adv) / 2
        d_loss_aux = (d_real_loss_aux + d_fake_loss_aux) / 2
        d_loss = d_loss_adv + d_loss_aux

        d_loss.backward()
        optim_D.step()

        
        

    # Log images to TensorBoard
    if epoch % 10 == 0:
        with torch.no_grad():
            sample_noise = torch.randn(16, z_dim).to('cuda')
            sample_labels = torch.arange(0, 10).to('cuda')
            sample_labels = torch.cat([sample_labels, sample_labels[:6]]).to('cuda')
            generated_images = G(sample_noise, sample_labels).view(-1, 1, 28, 28)
            grid = make_grid(generated_images, nrow=4, normalize=True)
            writer.add_image('generated_images', grid, epoch)

    print(
            f"Epoch [{epoch}/{num_epochs}] \
                Loss D: {d_loss.item():.4f}, loss G: {g_loss.item():.4f}"
        )

# Save models
torch.save(G.state_dict(), 'savedmodel/acgan_mnist_gen.pth')
torch.save(D.state_dict(), 'savedmodel/acgan_mnist_disc.pth')

writer.close()