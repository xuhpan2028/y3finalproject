import optuna
import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
import os
sys.path.append('/homes/hp921/y3finalproject')
from utils import compute_mmd, compute_emd, get_gpu_usage
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

# DataLoader
batch_size = 64
data_loader = torch.utils.data.DataLoader(normal_data, batch_size=batch_size, shuffle=True)

class Discriminator(nn.Module):
    def __init__(self, img_dim):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(img_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.disc(x)

class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, img_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.gen(x)

def objective(trial):
    latent_dim = trial.suggest_int('latent_dim', 50, 200)
    learning_rate_G = trial.suggest_loguniform('learning_rate_G', 1e-5, 1e-2)
    learning_rate_D = trial.suggest_loguniform('learning_rate_D', 1e-5, 1e-2)
    
    # Define your models, optimizer, and loss function
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = Generator(latent_dim, normal_data.shape[1]).to(device)
    discriminator = Discriminator(normal_data.shape[1]).to(device)
    
    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate_G)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate_D)
    
    num_epochs = 100  # Fixed number of epochs

    # Training loop
    for epoch in range(num_epochs):
        for batch_idx, real_data in enumerate(data_loader):
            real = real_data.to(device)
            noise = torch.randn(real.size(0), latent_dim).to(device)
            fake = generator(noise)

            # Train Discriminator
            disc_real = discriminator(real)
            disc_fake = discriminator(fake.detach())

            lossD_real = criterion(disc_real, torch.ones_like(disc_real))
            lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            lossD = (lossD_real + lossD_fake) / 2

            optimizer_D.zero_grad()
            lossD.backward()
            optimizer_D.step()

            # Train Generator
            output = discriminator(fake)
            lossG = criterion(output, torch.ones_like(output))

            optimizer_G.zero_grad()
            lossG.backward()
            optimizer_G.step()
    
    # Calculate MMD score
    mmd_score = compute_mmd(real, fake)
    return lossG.item() + 0.2 * lossD.item() + mmd_score

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

print('Best trial:')
trial = study.best_trial

print(f'  Value: {trial.value}')
print('  Params: ')
for key, value in trial.params.items():
    print(f'    {key}: {value}')