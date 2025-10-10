import torch 
import numpy as np
import pandas as pd
from dataclasses import dataclass, fields
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
parent_dir = Path(__file__).parent

@dataclass
class MortalityConfig:
    name: str = "mortality"
    gamma: float = -2.0014
    c: float = -0.062
    d: float = 0.9825
    sigma_K: float = 2.4180
    sigma_k: float = 1.2720
    multi_pop = pd.read_csv(parent_dir/"multi_pop_k.txt", sep='\t')
    # take equilibrium value here 
    k_start: float = (c/(1 - d))
    K_start: float = multi_pop['K.t'].mean()

def to_tensor(value, device, dtype):
    if isinstance(value, float): 
        return torch.tensor(value, device=device, dtype=dtype)
    elif isinstance(value, (list, tuple)) and all(isinstance(v, float) for v in value):
        return torch.tensor(value, device=device, dtype=dtype)
    elif isinstance(value, np.ndarray):
        return torch.tensor(value, device=device, dtype=dtype)
    else:
        return value

class MortalityModel:
    def __init__(self, config, device='cpu', dtype=torch.float32, seed=42):
        self.device = device
        self.dtype = dtype
        self.config = config
        self.rng = torch.Generator(device=device).manual_seed(seed)
        for f in fields(config):
            value = getattr(config, f.name)
            value = to_tensor(value, device=self.device, dtype=self.dtype)
            setattr(self, f.name, value)
        self.k_equilibrium = self.c / (1 - self.d)
    
    def simulate(self, start_k, start_K, t, n_samples):
        # simulate t steps from start point 
        assert t >= 0, "Time t must be non-negative"
        if t == 0:
            return start_k.repeat(n_samples), start_K.repeat(n_samples)
        else:
            # simulate k_t
            mean_k = self.c/(1 - self.d) + (self.d**t)*(start_k - self.c/(1 - self.d))
            var_k = (self.sigma_k**2) * (1 - (self.d**(2*t))) / (1 - self.d**2)
            normal_draws = torch.randn(n_samples, generator=self.rng)
            k_t = mean_k + torch.sqrt(var_k) * normal_draws            
            # simulate K_t
            mean_K = self.gamma * t + start_K
            var_K = (self.sigma_K**2) * t
            normal_draws_K = torch.randn(n_samples, generator=self.rng)
            K_t = mean_K + torch.sqrt(var_K) * normal_draws_K
            return k_t, K_t 
    
    def simulate_steps(self, start_k, start_K, t, n_samples):
        # simulate t steps from start point 
        assert t >= 0, "Time t must be non-negative"
        k_list = [torch.tensor(start_k, device=self.device, dtype=self.dtype).expand(n_samples, )]
        K_list = [torch.tensor(start_K, device=self.device, dtype=self.dtype).expand(n_samples, )]
        k = k_list[-1]
        K = K_list[-1]
        for _ in range(t):
            k, K = self.simulate(start_K=K, start_k=k, t=1, n_samples=n_samples)
            k_list.append(k)
            K_list.append(K)
        return torch.stack(k_list, dim=1), torch.stack(K_list, dim=1)


if __name__ == "__main__":
    MC = MortalityConfig()
    model = MortalityModel(MC, device='cpu')
    k_t, K_t = model.simulate(MC.k_start, MC.K_start,t=10, n_samples=1000)
    plt.figure(figsize=(10,6))
    sns.histplot(k_t.cpu().numpy(), kde=True, bins=30, color='blue', label='k_t')
    sns.histplot(K_t.cpu().numpy(), kde=True, bins=30, color='orange', label='K_t')
    plt.legend()
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of k_t and K_t')
    plt.savefig('k_t_K_t.png')
    # new simulation steps 
    k, K = model.simulate_steps(MC.k_start, MC.K_start, t=100, n_samples=10)
    print(k.shape, K.shape)
    plt.figure(figsize=(10,6))
    for i in range(10): 
        if i == 0:
            plt.plot(k[i].cpu().numpy(), label='k', color='blue')
            plt.plot(K[i].cpu().numpy(), label='K', color='orange')
        else:
            plt.plot(k[i].cpu().numpy(), color='blue')
            plt.plot(K[i].cpu().numpy(), linestyle='--', color='orange')            
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.xlim(0, 100)
    plt.title('Simulated Paths of k and K')
    plt.legend()
    plt.savefig('k_K_paths.png')