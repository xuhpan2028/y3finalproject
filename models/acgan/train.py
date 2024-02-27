import torch
from model import Discriminator, Generator  # Ensure these are your correct import paths
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm  # Import tqdm for the progress bar
import psutil

def gaussian_kernel(x, y, sigma=1.0):
    return torch.exp(-torch.norm(x - y) ** 2 / (2 * sigma ** 2))

def mmd(x, y, kernel=gaussian_kernel):
    # Flatten the input tensors to shape [batch_size, -1]
    x = x.view(x.size(0), -1)  # Reshape x to have shape [batch_size, 784]
    y = y.view(y.size(0), -1)  # Reshape y to have shape [batch_size, 784]

    m = x.size(0)
    n = y.size(0)
    xx = kernel(x, x).sum() / (m * (m - 1))
    yy = kernel(y, y).sum() / (n * (n - 1))
    xy = kernel(x, y).sum() / (m * n)
    return xx + yy - 2 * xy


def emd(x, y):
    # Flatten both tensors to have shape [-1, 784], assuming x and y are batches of 1x28x28 images
    x_flat = x.view(x.shape[0], -1)  # Reshape x to [batch_size, 784]
    y_flat = y.view(y.shape[0], -1)  # Reshape y to [batch_size, 784]

    # Compute the pairwise distance between the flattened images in the batch
    # Note: pairwise_distance expects inputs of shape [batch_size, vector_size]
    distance = torch.nn.functional.pairwise_distance(x_flat, y_flat, p=2)
    
    # Return the mean of the distances
    return distance.mean()

# Hyperparameters
LEARNING_RATE = 0.0002
BETAS = (0.5, 0.999)
BATCH_SIZE = 64
IMAGE_SIZE = 28
NUM_EPOCHS = 50

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize generator and discriminator
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=BETAS)
optimizer_D = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=BETAS)

# Loss functions
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


log_file = "training_log.txt"  # Define the log file name

# Open the log file
file =  open(log_file, "w")
file.write("Epoch, Batch, D Loss, G Loss, MMD Score, EMD Score, CPU Usage (%), GPU Memory Allocated (bytes)\n")

for epoch in range(NUM_EPOCHS):
    epoch_loss_D, epoch_loss_G = 0, 0  # For averaging losses over the epoch
    with tqdm(enumerate(dataloader), total=len(dataloader)) as pbar:
        for i, (imgs, labels) in pbar:
            batch_size = imgs.shape[0]

            # Move data to the correct device
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

    
    
    mmd_score = mmd(real_imgs, gen_imgs).item()
    emd_score = emd(real_imgs, gen_imgs).item()


    # Monitor performance
    cpu_usage = psutil.cpu_percent()
    gpu_usage = torch.cuda.memory_allocated()

    # Logging
    file.write(f"{epoch+1}, {i}, {d_loss.item():.6f}, {g_loss.item():.6f}, {mmd_score:.6f}, {emd_score:.6f}, {cpu_usage}, {gpu_usage}\n")
