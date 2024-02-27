import torch
from model import Discriminator, Generator
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm  # Import tqdm for the progress bar

# Hyperparameters
LEARNING_RATE = 0.0002
BETAS = (0.5, 0.999)
BATCH_SIZE = 64
IMAGE_SIZE = 28
NUM_EPOCHS = 50

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=BETAS)
optimizer_D = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=BETAS)

adversarial_loss = torch.nn.BCEWithLogitsLoss()
auxiliary_loss = torch.nn.CrossEntropyLoss()

# Load data
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = datasets.MNIST('.', download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

for epoch in range(NUM_EPOCHS):
    epoch_loss_D, epoch_loss_G = 0, 0  # For averaging losses over the epoch
    with tqdm(enumerate(dataloader), total=len(dataloader)) as pbar:
        for i, (imgs, labels) in pbar:

            batch_size = imgs.shape[0]

            # Real images
            real_imgs = imgs.to(device)
            labels = labels.to(device)

            # Ground truths for real and fake images
            valid = torch.ones(batch_size, 1, requires_grad=False).to(device)
            fake = torch.zeros(batch_size, 1, requires_grad=False).to(device)

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Loss for real images
            real_pred, real_aux = discriminator(real_imgs, labels)
            d_real_loss = (adversarial_loss(real_pred, valid) + auxiliary_loss(real_aux, labels)) / 2

            # Generate a batch of images
            noise = torch.randn(batch_size, 100, requires_grad=False).to(device)
            gen_labels = torch.randint(0, 10, (batch_size,)).to(device)
            gen_imgs = generator(noise, gen_labels)

            # Loss for fake images
            fake_pred, fake_aux = discriminator(gen_imgs.detach(), gen_labels)
            d_fake_loss = (adversarial_loss(fake_pred, fake) + auxiliary_loss(fake_aux, gen_labels)) / 2

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Generate a batch of images
            gen_imgs = generator(noise, gen_labels)

            # Loss for fake images, but try to fool the discriminator
            validity, pred_label = discriminator(gen_imgs, gen_labels)
            g_loss = (adversarial_loss(validity, valid) + auxiliary_loss(pred_label, gen_labels)) / 2

            g_loss.backward()
            optimizer_G.step()

            # Update progress bar
            pbar.set_description(
                f"Epoch {epoch+1}/{NUM_EPOCHS} [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item():.6f}] [G loss: {g_loss.item():.6f}]"
            )
            epoch_loss_D += d_loss.item()
            epoch_loss_G += g_loss.item()

    # Average losses for the current epoch
    avg_epoch_loss_D = epoch_loss_D / len(dataloader)
    avg_epoch_loss_G = epoch_loss_G / len(dataloader)
    print(f"Epoch {epoch+1} completed. Avg D Loss: {avg_epoch_loss_D:.6f}, Avg G Loss: {avg_epoch_loss_G:.6f}")
