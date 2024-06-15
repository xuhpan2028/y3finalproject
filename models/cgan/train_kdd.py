import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
import os
sys.path.append('/homes/hp921/y3finalproject')
from utils import *
from torch.utils.tensorboard import SummaryWriter

train_url = 'https://raw.githubusercontent.com/merteroglu/NSL-KDD-Network-Instrusion-Detection/master/NSL_KDD_Train.csv'
test_url = 'https://raw.githubusercontent.com/merteroglu/NSL-KDD-Network-Instrusion-Detection/master/NSL_KDD_Test.csv'

col_names = ["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label"]

df = pd.read_csv(train_url,header=None, names = col_names)
df_test = pd.read_csv(test_url, header=None, names = col_names)

categorical_columns=['protocol_type', 'service', 'flag']

df_categorical_values = df[categorical_columns]
testdf_categorical_values = df_test[categorical_columns]

# df_categorical_values.head()
df_categorical_values_enc=df_categorical_values.apply(LabelEncoder().fit_transform)
df[categorical_columns] = df[categorical_columns].apply(LabelEncoder().fit_transform)

# test set
testdf_categorical_values_enc=testdf_categorical_values.apply(LabelEncoder().fit_transform)

df['label'] = df['label'].apply(lambda x: 0 if x == 'normal' else 1)

# Separate normal and anomaly data
normal_data_df = df[df['label'] == 1].drop(columns=['label']).reset_index(drop=True)
anomaly_data_df = df[df['label'] == 0].drop(columns=['label']).reset_index(drop=True)

# Convert DataFrame to NumPy array
normal_data = normal_data_df.values.astype(np.float32)
anomaly_data = anomaly_data_df.values.astype(np.float32)

# Normalize data to the range [-1, 1] using MinMaxScaler
scaler = MinMaxScaler(feature_range=(-1, 1))
normal_data = scaler.fit_transform(normal_data)
anomaly_data = scaler.transform(anomaly_data)

# Convert to PyTorch tensor
normal_data = torch.tensor(normal_data)
anomaly_data = torch.tensor(anomaly_data)

# Hyperparameters
batch_size = 64
latent_dim = 76
embedding_dim = 16  # Embedding dimension for the label
learning_rate = 0.0008614867655382943
learning_rate_D = 4.032045351608014e-05
num_epochs = 100

# DataLoader
data_loader = torch.utils.data.DataLoader(normal_data, batch_size=batch_size, shuffle=True)

class Discriminator(nn.Module):
    def __init__(self, img_dim, embedding_dim):
        super().__init__()
        self.label_embedding = nn.Embedding(2, embedding_dim)
        self.disc = nn.Sequential(
            nn.Linear(img_dim + embedding_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        label_embeddings = self.label_embedding(labels)
        x = torch.cat((x, label_embeddings), dim=1)
        return self.disc(x)

class Generator(nn.Module):
    def __init__(self, z_dim, img_dim, embedding_dim):
        super().__init__()
        self.label_embedding = nn.Embedding(2, embedding_dim)
        self.gen = nn.Sequential(
            nn.Linear(z_dim + embedding_dim, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, img_dim),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        label_embeddings = self.label_embedding(labels)
        x = torch.cat((noise, label_embeddings), dim=1)
        return self.gen(x)

save_path = 'savedmodel/'
os.makedirs(save_path, exist_ok=True)

# Setting up TensorBoard
writer = SummaryWriter('/homes/hp921/y3finalproject/runs/cgan_KDD')

device = "cuda" if torch.cuda.is_available() else "cpu"
print("using ", device)

# Define your models, optimizer, and loss function
generator = Generator(latent_dim, normal_data.shape[1], embedding_dim).to(device)
discriminator = Discriminator(normal_data.shape[1], embedding_dim).to(device)

criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate)
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate_D)

# Training loop
for epoch in range(1000):  # Reduced number of epochs for faster trials
    for batch_idx, real_data in enumerate(data_loader):
        real = real_data.to(device)
        real_labels = torch.ones(real.size(0), dtype=torch.long).to(device)
        fake_labels = torch.zeros(real.size(0), dtype=torch.long).to(device)
        
        noise = torch.randn(real.size(0), latent_dim).to(device)
        fake = generator(noise, fake_labels)

        # Train Discriminator
        disc_real = discriminator(real, real_labels)
        disc_fake = discriminator(fake.detach(), fake_labels)

        lossD_real = criterion(disc_real, torch.ones_like(disc_real))
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        lossD = (lossD_real + lossD_fake) / 2

        optimizer_D.zero_grad()
        lossD.backward()
        optimizer_D.step()

        # Train Generator
        output = discriminator(fake, fake_labels)
        lossG = criterion(output, torch.ones_like(output))

        optimizer_G.zero_grad()
        lossG.backward()
        optimizer_G.step()

    # Calculate MMD and EMD scores
    mmd_score = compute_mmd(real, fake)
    emd_score = compute_emd(real, fake)

    # Get GPU usage
    memory_used, memory_total, utilization = get_gpu_usage()

    print(f"Epoch: {epoch}, Loss D: {lossD.item()}, Loss G: {lossG.item()}, MMD: {mmd_score}, EMD: {emd_score}, GPU Memory: {memory_used}/{memory_total} MiB, GPU Utilization: {utilization}%")
    # Log losses and scores to TensorBoard
    writer.add_scalar('Loss/Discriminator', lossD.item(), epoch)
    writer.add_scalar('Loss/Generator', lossG.item(), epoch)
    writer.add_scalar('Score/MMD', mmd_score, epoch)
    writer.add_scalar('Score/EMD', emd_score, epoch)
    writer.add_scalar('GPU/Memory_Used', memory_used, epoch)
    writer.add_scalar('GPU/Memory_Total', memory_total, epoch)
    writer.add_scalar('GPU/Utilization', utilization, epoch)

    print(f'Epoch [{epoch+1}/1000]  Loss D: {lossD.item()}, Loss G: {lossG.item()}')

# Save final model checkpoints
torch.save(generator.state_dict(), f'{save_path}kdd_vanilla_gen.pth')
torch.save(discriminator.state_dict(), f'{save_path}kdd_vanilla_dis.pth')