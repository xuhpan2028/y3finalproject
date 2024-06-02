import torch
import subprocess
from scipy.stats import wasserstein_distance  # For EMD calculation


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
