import torch 
import numpy as np
import pandas as pd
from dataclasses import dataclass, fields, field 
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
parent_dir = Path(__file__).parent

@dataclass
class MortalityConfig:
    name: str = "mortality"
    gamma: float = -0.00022
    c: float = -0.0620
    d: float = 0.9825
    sigma_K: float = 2.4180
    sigma_k: float = 1.2720
    multi_pop = pd.read_csv(parent_dir/"multi_pop_k.txt", sep='\t')
    # take the  equilibrium start value here 
    k_start: float = (c/(1 - d))
    K_start: float = multi_pop['K.t'].mean()
    # 
    multi_pop: pd.DataFrame = field(default_factory=lambda: pd.read_csv(parent_dir/"multi_pop.txt", sep='\t'))
    a: np.ndarray = field(init=False)
    b: np.ndarray = field(init=False)
    A: np.ndarray = field(init=False)
    B: np.ndarray = field(init=False)
    age: np.ndarray = field(init=False)

    def __post_init__(self):
        self.a = self.multi_pop['a.x'].values
        self.b = self.multi_pop['b.x'].values
        self.A = self.multi_pop['A.x'].values
        self.B = self.multi_pop['B.x'].values
        self.age = self.multi_pop['x'].values
    

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
        self.rng = self.set_seed(seed)
        for f in fields(config):
            value = getattr(config, f.name)
            value = to_tensor(value, device=self.device, dtype=self.dtype)
            setattr(self, f.name, value)

    def set_seed(self, seed):
        self.rng = torch.Generator(device=self.device).manual_seed(seed)
    
    def simulate(self, start_k, start_K, t, n_samples, age):
        assert t >= 0, "Time t must be non-negative"
        # simulate to the t-th time step from the start point 
        # age starts at 1. index=0 means age 1. idx = age - 1
        idx = age - 1
        a = self.a[idx]
        b = self.b[idx]
        A = self.A[idx]
        B = self.B[idx]
        if t == 0:
            k = start_k.repeat(n_samples)
            K = start_K.repeat(n_samples)
            # reading mortality parameters from multi pop 
            m = torch.exp(a + b * k + A + B * K)
            return start_k.repeat(n_samples), start_K.repeat(n_samples), m 
        else:
            # simulate k_t
            mean_k = self.c/(1 - self.d) + (self.d**t)*(self.k_start - self.c/(1 - self.d))
            var_k = (self.sigma_k**2) * (1 - (self.d**(2*t))) / (1 - self.d**2)
            normal_draws = torch.randn(n_samples, generator=self.rng, device=self.device, dtype=self.dtype)
            k = mean_k + torch.sqrt(var_k) * normal_draws            
            # simulate K_t
            mean_K = self.gamma * t + start_K
            var_K = (self.sigma_K**2) * t
            normal_draws_K = torch.randn(n_samples, generator=self.rng, device=self.device, dtype=self.dtype)
            K = mean_K + torch.sqrt(var_K) * normal_draws_K
            #
        m = torch.exp(a + b * k + A + B * K)
        return k, K, m

    def simulate_steps(self, start_k, start_K, t, n_samples, age):
        # we include the first step 
        assert age >= 1, "Age must be at least 1"
        # simulate t steps from start point 
        assert t >= 0, "Time t must be non-negative"
        start_k = start_k.expand(n_samples, )
        start_K = start_K.expand(n_samples, )
        idx = age - 1
        m = torch.exp(self.a[idx] + self.b[idx] * start_k + self.A[idx] + self.B[idx] * start_K)
        K_list = [start_K]
        k_list = [start_k]
        m_list = [m]
        for _ in range(t):
            idx = age - 1
            a = self.a[idx]
            b = self.b[idx]
            A = self.A[idx]
            B = self.B[idx]
            k, K = self.simulate(start_K=start_K, start_k=start_k, t=1, n_samples=n_samples, age=65)[:2]
            m = torch.exp(a + b * k + A + B * K)
            k_list.append(k)
            K_list.append(K)
            m_list.append(m)
            age += 1            
        return torch.stack(k_list, dim=1), torch.stack(K_list, dim=1), torch.stack(m_list, dim=1)
    

if __name__ == "__main__":
    multi_pop = pd.read_csv(parent_dir/"multi_pop.txt", sep='\t')
    print(multi_pop.head())
    print(multi_pop.tail())
    MC = MortalityConfig()
    model = MortalityModel(MC, device='cpu')
    k_t, K_t, m_t = model.simulate(MC.k_start, MC.K_start,t=10, n_samples=1000, age=65)
    print(m_t)
    plt.figure(figsize=(10,6))
    sns.histplot(k_t.cpu().numpy(), kde=True, bins=30, color='blue', label='k_t')
    sns.histplot(K_t.cpu().numpy(), kde=True, bins=30, color='orange', label='K_t')
    plt.legend()
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of k_t and K_t')
    plt.savefig('k_t_K_t.png')
    # histogram of mortality rates
    plt.figure(figsize=(10,6))
    sns.histplot(m_t.cpu().numpy(), kde=True, bins=30, color='green', label='m_t')
    plt.legend()
    plt.xlabel('Mortality Rate')
    plt.ylabel('Frequency')
    plt.title('Histogram of Mortality Rates')
    plt.savefig('m_t.png')

    ## 
    age = 1
    time_steps = 10
    k, K, m = model.simulate_steps(MC.k_start, MC.K_start, t=time_steps, n_samples=10, age=age)
    print(k.shape, K.shape)
    plt.figure(figsize=(10,6))
    for i in range(10):
        plt.plot(k[i].cpu().numpy(), label=f'k')
        plt.plot(K[i].cpu().numpy(), label=f'K', linestyle='--')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Simulated Paths of k and K')
    # plt.legend()
    plt.savefig('k_K_paths.png')
    # mortality paths
    plt.figure(figsize=(10,6))
    for i in range(10):
        plt.plot(m[i].cpu().numpy(), label=f'm', color='green')
    plt.xlabel('Time')
    plt.ylabel('Mortality Rate')
    plt.title('Simulated Paths of Mortality Rates')
    # x_tick_labels = [age + i for i in range(time_steps + 1)]
    plt.xticks(ticks=range(0, time_steps + 1, 5),labels=[age + i for i in range(0, time_steps + 1, 5)])
    plt.xlim(0, time_steps)
    # plt.legend()
    plt.savefig('mortality_paths.png')