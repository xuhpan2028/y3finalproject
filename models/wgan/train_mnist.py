import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import subprocess
from scipy.stats import wasserstein_distance  # For EMD calculation

save_path = 'savedmodel/'

# Helper functions for MMD and EMD
def gaussian_kernel(x, y, sigma=1.0):
    x = x.unsqueeze(1)
    y = y.unsqueeze(0)
    return torch.exp(-torch.sum((x - y) ** 2, dim=2) / (2 * sigma ** 2))

def compute_mmd(x, y, sigma=1.0):
    K_xx = gaussian_kernel(x, x, sigma).mean()
    K_yy = gaussian_kernel(y, y, sigma).mean()
    K_xy = gaussian_kernel(x, y, sigma).mean()
    return K_xx + K_yy - 2 * K_xy

def compute_emd(x, y):
    x = x.cpu().detach().numpy()
    y = y.cpu().detach().numpy()
    emd_score = 0
    for i in range(x.shape[1]):
        emd_score += wasserstein_distance(x[:, i], y[:, i])
    return emd_score / x.shape[1]

def get_gpu_usage():
    result = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', '--format=csv,nounits,noheader'],
        encoding='utf-8')
    memory_used, memory_total, utilization = map(int, result.strip().split(', '))
    return memory_used, memory_total, utilization

# Setting up TensorBoard
writer = SummaryWriter('runs/wgan_test1')

# Initialize Critic and Generator
class Critic(nn.Module):
    def __init__(self, img_dim):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(img_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.critic(x)

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


device = "cuda" if torch.cuda.is_available() else "cpu"
critic = Critic(784).to(device)
gen = Generator(64, 784).to(device)
lr_critic = 3e-4
lr_gen = 3e-4
opt_critic = optim.RMSprop(critic.parameters(), lr=lr_critic)
opt_gen = optim.RMSprop(gen.parameters(), lr=lr_gen)

# Clip value for critic weights
clip_value = 0.01

# Training loop
transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, ), (0.5, ))]
)

# Data loading
loader = DataLoader(datasets.MNIST(root="dataset/", transform=transforms, download=True), batch_size=32, shuffle=True)

n_critic = 5  # Number of critic iterations per generator iteration

for epoch in range(1000):  # Reduced number of epochs for faster trials
    print(f"Starting epoch {epoch}")
    for batch_idx, (real, _) in enumerate(loader):
        real = real.view(-1, 784).to(device)
        noise = torch.randn(real.size(0), 64).to(device)
        fake = gen(noise)

        # Train Critic
        for _ in range(n_critic):
            critic_real = critic(real).mean()
            critic_fake = critic(fake.detach()).mean()
            loss_critic = -(critic_real - critic_fake)
            opt_critic.zero_grad()
            loss_critic.backward()
            opt_critic.step()
            
            # Clip critic weights
            for p in critic.parameters():
                p.data.clamp_(-clip_value, clip_value)

        # Train Generator
        fake = gen(noise)
        loss_gen = -critic(fake).mean()
        opt_gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

    # Calculate MMD and EMD scores
    mmd_score = compute_mmd(real, fake)
    emd_score = compute_emd(real, fake)

    # Get GPU usage
    memory_used, memory_total, utilization = get_gpu_usage()

    print(f"Epoch: {epoch}, Loss Critic: {loss_critic.item()}, Loss Generator: {loss_gen.item()}, MMD: {mmd_score}, EMD: {emd_score}, GPU Memory: {memory_used}/{memory_total} MiB, GPU Utilization: {utilization}%")
    # Log losses and scores to TensorBoard
    writer.add_scalar('Loss/Critic', loss_critic.item(), epoch)
    writer.add_scalar('Loss/Generator', loss_gen.item(), epoch)
    writer.add_scalar('Score/MMD', mmd_score, epoch)
    writer.add_scalar('Score/EMD', emd_score, epoch)
    writer.add_scalar('GPU/Memory_Used', memory_used, epoch)
    writer.add_scalar('GPU/Memory_Total', memory_total, epoch)
    writer.add_scalar('GPU/Utilization', utilization, epoch)

    with torch.no_grad():
        fake = gen(noise).view(-1, 1, 28, 28)  # Reshape to (B, C, H, W)
        writer.add_images('Generated Images', fake, epoch)



# Save final model checkpoints
torch.save(gen.state_dict(), f'{save_path}wgan_generator_final.pth')
torch.save(critic.state_dict(), f'{save_path}wgan_critic_final.pth')


writer.close()

print(f"  Loss Critic: {loss_critic.item()}", end = " ")
print(f"  Loss Generator: {loss_gen.item()}")
print(f"  MMD: {mmd_score}", end = " ")
print(f"  EMD: {emd_score}")
print(f"  GPU Memory Used: {memory_used}/{memory_total} MiB", end = " ")
print(f"  GPU Utilization: {utilization}%")
