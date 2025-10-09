import torch 
from dataclasses import dataclass, fields
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

@dataclass
class AnnuityConfig:
    name: str = "annuity"
    # for processes x and y 
    alpha: float = 0.39
    beta: float = 0.0785
    nu: float = 0.0201
    eta: float = 0.0135
    rho: float = -0.0645
    startx: float = 0.0
    starty: float = 0.0

def _to_tensor(value, device, dtype):
    if isinstance(value, float): 
        return torch.tensor(value, device=device, dtype=dtype)
    elif isinstance(value, (list, tuple)) and all(isinstance(v, float) for v in value):
        return torch.tensor(value, device=device, dtype=dtype)
    elif isinstance(value, np.ndarray):
        return torch.tensor(value, device=device, dtype=dtype)
    else:
        return value

class TwoFactorOrnsteinUhlenbeckModel:
    def __init__(self, config, seed, device='cpu', dtype=torch.float32):
        self.rng = torch.Generator().manual_seed(seed)
        self.device = device
        self.dtype = dtype
        for f in fields(config):
            value = getattr(config, f.name)
            value = _to_tensor(value, device=self.device, dtype=self.dtype)
            setattr(self, f.name, value)

    def _single_increment(self, start_point, dt, reversion, volatility, normal_draws):
        return start_point*torch.exp(-reversion*dt) + volatility*torch.sqrt((1-torch.exp(-2*reversion*dt))/(2*reversion))*normal_draws

    def coupled_increment(self, start_x, start_y, dt):
        normal_1 = torch.randn(start_x.shape) 
        normal_2 = torch.randn(start_y.shape) 
        # x and y 
        x = self._single_increment(start_x, dt, reversion=self.alpha, volatility=self.nu, normal_draws=normal_1)
        y = self._single_increment(start_y, dt, reversion=self.beta, volatility=self.nu, normal_draws=self.rho*normal_1 + torch.sqrt(1-self.rho**2)*normal_2)
        return x, y


if __name__ == "__main__":
    config = AnnuityConfig()
    device = 'cpu'
    dtype = torch.float32
    n_samples = 100 
    model = TwoFactorOrnsteinUhlenbeckModel(config=config, seed=42, device=device, dtype=dtype)
    start_x = start_y = torch.zeros(n_samples, device=device, dtype=dtype)
    x, y = model.coupled_increment(start_x, start_y, dt=1)
    sns.histplot(x.cpu().numpy(), kde=True, color='blue', label='x', stat='density')
    sns.histplot(y.cpu().numpy(), kde=True, color='orange', label='y', stat='density')
    plt.legend()
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title('Histogram of x and y')
    plt.savefig('ornstein_uhlenbeck_histogram.png')
