import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import torch
import torch.nn as nn
import os

# Define the discriminator model
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

# Function to load and preprocess data
def load_and_preprocess_data():
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

    df_test = pd.read_csv(test_url, header=None, names = col_names)

    # Preprocess the test data
    categorical_columns=['protocol_type', 'service', 'flag']
    df_categorical_values = df_test[categorical_columns]
    df_categorical_values_enc = df_categorical_values.apply(LabelEncoder().fit_transform)
    df_test[categorical_columns] = df_categorical_values_enc
    df_test['label'] = df_test['label'].apply(lambda x: 0 if x == 'normal' else 1)

    test_labels = df_test['label'].values
    test_data = df_test.drop(columns=['label']).values.astype(np.float32)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    test_data = scaler.fit_transform(test_data)

    return torch.tensor(test_data), test_labels

# Load models and evaluate
def evaluate_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("using ", device)

    # Load pre-trained models
    img_dim = 41  # Number of features in the dataset
    latent_dim = 100
    save_path = 'savedmodel/'

    discriminator = Discriminator(img_dim).to(device)
    discriminator.load_state_dict(torch.load(f'{save_path}kdd_vanilla_dis.pth'))
    discriminator.eval()

    test_data, test_labels = load_and_preprocess_data()
    test_data = test_data.to(device)

    # Discriminator predictions
    with torch.no_grad():
        predictions = discriminator(test_data).cpu().numpy()

    # Convert predictions to binary labels (0 or 1)
    predicted_labels = (predictions > 0.5).astype(int).flatten()

    # Calculate accuracy
    accuracy = np.mean(predicted_labels == test_labels)
    print(f'Accuracy: {accuracy * 100:.2f}%')

if __name__ == "__main__":
    evaluate_model()