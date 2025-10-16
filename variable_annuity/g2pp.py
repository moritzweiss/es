import torch 
from dataclasses import dataclass, fields
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

@dataclass
class InterestRateConfig:
    name: str = "g2pp"
    # for processes x and y 
    alpha: float = 0.39
    beta: float = 0.0785
    nu: float = 0.0201
    eta: float = 0.0135
    rho: float = -0.0645
    startx: float = 0.0
    starty: float = 0.0
    # for the G2++ model
    beta0: float = 1.54617/100
    beta1: float = 2.47537/100
    beta2: float = -26.22674/100
    beta3: float = 26.04199/100
    theta1: float = 4.80792/100
    theta2: float = 5.83035/100     

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
        self.device = device
        self.dtype = dtype
        for f in fields(config):
            value = getattr(config, f.name)
            value = _to_tensor(value, device=self.device, dtype=self.dtype)
            setattr(self, f.name, value)
        self.set_seed(seed)

    def set_seed(self, seed):
        self.rng = torch.Generator(device=self.device).manual_seed(seed)

    def _single_increment(self, start_point, dt, reversion, volatility, normal_draws):
        return start_point*torch.exp(-reversion*dt) + volatility*torch.sqrt((1-torch.exp(-2*reversion*dt))/(2*reversion))*normal_draws

    def coupled_increment(self, start_x, start_y, dt):
        normal_1 = torch.randn(start_x.shape, generator=self.rng, device=self.device, dtype=self.dtype) 
        normal_2 = torch.randn(start_y.shape, generator=self.rng, device=self.device, dtype=self.dtype) 
        # x and y 
        x = self._single_increment(start_x, dt, reversion=self.alpha, volatility=self.nu, normal_draws=normal_1)
        y = self._single_increment(start_y, dt, reversion=self.beta, volatility=self.nu, normal_draws=self.rho*normal_1 + torch.sqrt(1-self.rho**2)*normal_2)
        return x, y

class G2PPModel:
    def __init__(self, config, seed, device='cpu', dtype=torch.float32):
        self.rng = torch.Generator().manual_seed(seed)
        self.device = device
        self.dtype = dtype
        for f in fields(config):
            value = getattr(config, f.name)
            value = _to_tensor(value, device=self.device, dtype=self.dtype)
            setattr(self, f.name, value)
        # some attributes are not needer for OU, but we pass them anyway 
        self.coupled_ou = TwoFactorOrnsteinUhlenbeckModel(config, seed, device, dtype)
        self.coupled_ou.set_seed(seed)
    
    def set_seed(self, seed):
        self.rng = torch.Generator(device=self.device).manual_seed(seed)
        self.coupled_ou.set_seed(seed)

    def f(self, t):
        assert t >= 0
        if t == 0:
            return self.beta0 
        else:
            factor = lambda t, a, b: torch.exp(-t/a)*(t+b)/b
            first_term = self.beta0
            second_term = self.beta1*(self.theta1/t)*(1 - torch.exp(-t/self.theta1))
            third_term = self.beta2*(self.theta1/t)*(1 - factor(t, self.theta1, self.theta1))
            fourth_term = self.beta3*(self.theta2/t)*(1 - factor(t, self.theta1, self.theta2))
            return first_term + second_term + third_term + fourth_term
    
    def g(self, kappa, t):
        return torch.sqrt((1 - torch.exp(-kappa*t)))*(1/kappa)

    def psi(self, t):
        term1 = self.f(t) 
        term2 = (self.nu**2/2)*self.g(self.alpha, t)**2
        term3 = (self.eta**2/2)*self.g(self.beta, t)**2
        term4 = self.rho*self.nu*self.eta*self.g(self.alpha, t)*self.g(self.beta, t)
        return term1 + term2 + term3 + term4
    
    def simulate(self, dt, x_start, y_start, t_start):
        # simulate the process at time t_start + dt given the values of x and y at time t_start
        x, y = self.coupled_ou.coupled_increment(x_start, y_start, dt)        
        return self.psi(t_start+dt) + x + y, x, y
    
    def simulate_path(self, n_steps, x_start, y_start, t_start, dt):
        # x,y,z at time t_start
        x_path = [x_start]
        y_path = [y_start]
        # function psi at time t_start 
        r_path = [self.simulate(0, x_start, y_start, t_start)[0]]
        for step in range(1, n_steps+1):
            r, x, y = self.simulate(dt, x_path[-1], y_path[-1], t_start + step*dt)
            x_path.append(x)
            y_path.append(y)
            r_path.append(r)
        x_path = torch.stack(x_path, dim=1)
        y_path = torch.stack(y_path, dim=1)
        r_path = torch.stack(r_path, dim=1)
        return r_path, x_path, y_path


if __name__ == "__main__":
    config = InterestRateConfig()
    device = 'cpu'
    dtype = torch.float32
    n_samples = int(1e4) 
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
    model = G2PPModel(config=config, seed=42, device=device, dtype=dtype)
    r = model.simulate(dt=100, x_start=start_x, y_start=start_y, t_start=0)
    plt.figure()
    sns.histplot(r.cpu().numpy(), kde=True, color='green', label='r', stat='density')
    plt.legend()
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title(f'Histogram of interest rate at time t={100}')
    plt.savefig('g2pp_histogram.png')
    n_steps = 101 
    r, x, y = model.simulate_path(n_steps=n_steps, x_start=start_x, y_start=start_y, t_start=0, dt=1)
    plt.figure()
    plt.plot(r[0:10, :].cpu().numpy().T)
    plt.xlabel('Time step')
    plt.ylabel('Interest rate')
    plt.xlim(0, n_steps-1)
    plt.title('Sample paths of interest rate')
    plt.savefig('g2pp_sample_paths.png')
    plt.figure()
    plt.plot(x[0:10, :].cpu().numpy().T)
    plt.xlabel('Time step')
    plt.ylabel('Process x')
    plt.xlim(0, n_steps-1)
    plt.title('Sample paths of process x')
    plt.savefig('g2pp_process_x_sample_paths.png')
    plt.figure()
    plt.plot(y[0:10, :].cpu().numpy().T)
    plt.xlabel('Time step')         
    plt.ylabel('Process y')
    plt.xlim(0, n_steps-1)
    plt.title('Sample paths of process y')
    plt.savefig('g2pp_process_y_sample_paths.png')