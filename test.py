import torch
import torch.nn as nn
from torchviz import make_dot

class Discriminator(nn.Module):
    def __init__(self, img_dim):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Linear(img_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.disc(x)

# Example usage
img_dim = 784  # Example image dimension
model = Discriminator(img_dim)

# Create a dummy input with a larger batch size
batch_size = 8
x = torch.randn(batch_size, img_dim)

# Generate the computation graph
y = model(x)
dot = make_dot(y, params=dict(model.named_parameters()))

# Render and save the graph to a file
dot.render("discriminator_model", format="png")