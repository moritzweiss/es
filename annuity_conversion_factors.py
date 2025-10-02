import numpy as np
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt



@dataclass
class Parameters:
    dt: float = 1.0
    K0: float = 0.0
    k0: float = 0.0
    x0: float = 0.0
    y0: float = 0.0
    # Mortality parameters (Appendix 3)
    theta: float = -2.0014
    gamma: float = -2.0014
    sigma_eps: float = np.sqrt(2.4180)
    c: float = -0.0620
    d: float = 0.9825
    sigma_delta: float = np.sqrt(1.2720)
    # Interest‑rate parameters (Table 1)
    alpha: float = 0.3912
    beta: float = 0.0785
    nu: float = 0.0201
    eta: float = 0.0135
    rho: float = -0.6450

    def to_tensor(self, device: str = 'cpu', dtype: torch.dtype = torch.float32):
        kwargs = {}
        for f, val in self.__dict__.items():
            if isinstance(val, torch.Tensor):
                print(f'this is already a tensor: {val}')
            kwargs[f] = torch.tensor(val, device=device, dtype=dtype)
        return Parameters(**kwargs)


def simulate_risk_factors(params, k0=0, K0=0, r0=0, x0=0, y0=0, psi0=0, device='cpu', T=10, n_samples: int = 32, only_return_final=True):
    '''
    bla bla bla
    '''
    
    K = torch.full((n_samples, T), K0, device=device, dtype=torch.float32)
    k = torch.full((n_samples, T), k0, device=device, dtype=torch.float32)
    x = torch.full((n_samples, T), x0, device=device, dtype=torch.float32)
    y = torch.full((n_samples, T), y0, device=device, dtype=torch.float32)
    r = torch.full((n_samples, T), r0, device=device, dtype=torch.float32)


    for t in range(1,T):
        eps = torch.randn(n_samples, device=device)* params.sigma_eps
        delta = torch.randn(n_samples, device=device)* params.sigma_delta
        # correllations of W1 and W2 are missing. just leave it for now 
        W1 = torch.randn(n_samples, device=device)
        W2 = torch.randn(n_samples, device=device)
        K[:, t] = K[:, t-1] + params.gamma + eps
        k[:, t] = params.c + params.d * k[:, t-1] + delta
        x[:, t] = x[:, t-1] + (-params.alpha * x[:, t-1]) + params.nu * W1
        y[:, t] = y[:, t-1] + (-params.beta * y[:, t-1]) + params.eta * (params.rho * W1 + torch.sqrt(1 - params.rho**2) * W2)
        # just set psi to zero for now
        r[:, t] = x[:, t] + y[:, t] 

    if only_return_final:
        K = K[:, -1]
        k = k[:, -1]
        x = x[:, -1]
        y = y[:, -1]
        r = r[:, -1]

    return K, k, x, y, r 

def simulate_payoff(K, k, x, y, r, params,  max_age=5):
    '''
    This function simulates the payoff of a variable annuity based on the risk factors K, k, x, y, and r.    
    '''

    n_samples = K.shape[0]
    discount = torch.ones(n_samples, device=K.device, dtype=torch.float32) 
    discounted_sum  = torch.zeros(n_samples, device=K.device, dtype=torch.float32)
    for t in range(1, max_age):
        eps = torch.randn(n_samples, device=K.device) * params.sigma_eps
        delta = torch.randn(n_samples, device=K.device) * params.sigma_delta
        W1 = torch.randn(n_samples, device=K.device)
        W2 = torch.randn(n_samples, device=K.device)
        # interest rate 
        r = x + y 
        # calculate m 
        mgx = 0.1 + 0.1 * K 
        max = 0.1 + 0.1 * k 
        m = mgx*mgx 
        # calc
        discount = discount*torch.exp(-(r+m))
        discounted_sum += discount 
        # new risk factors         
        K = K + params.gamma + eps
        k = params.c + params.d * k + delta
        x = x + (-params.alpha * x) + params.nu * W1
        y = y + (-params.beta * y) + params.eta * (params.rho * W1 + torch.sqrt(1 - params.rho**2) * W2)
    
    return discounted_sum




def simulate_mortality_factors(t:int, n_samples:int, params: Parameters) -> np.ndarray:
    '''    
    K_t = K_{t-1} + gamma + eps_t
    kt = c + d k_{t-1} + delta_t 
    this simulation starts from t=0 and runs up to t 
    '''
    eps = np.random.normal(loc=0.0, scale=params.sigma_eps, size=n_samples)
    Kt = t*params.gamma + t*eps 
    delta = np.random.normal(loc=0.0, scale=params.sigma_delta, size=n_samples)
    kt = params.c + params.d * params.k0 + delta
    return Kt

def simulate_exemplary_model(
    n_scenarios: int,
    dt: float = 1.0,
    K0: float = 0.0,
    k0: float = 0.0,
    x0: float = 0.0,
    y0: float = 0.0,
    # Mortality parameters (Appendix 3)
    theta: float = -2.0014,
    sigma_eps: float = np.sqrt(2.4180),
    c: float = -0.0620,
    d: float = 0.9825,
    sigma_delta: float = np.sqrt(1.2720),
    # Interest‑rate parameters (Table 1)
    alpha: float = 0.3912,
    beta: float = 0.0785,
    nu: float = 0.0201,
    eta: float = 0.0135,
    rho: float = -0.6450
) -> np.ndarray:
    """
    Simulate n_scenarios of (K_T, k_T, x_T, y_T) over a single time step of length dt.
    Returns an array of shape (n_scenarios, 4).
    """
    # Draw independent normals for mortality innovations
    eps = np.random.normal(loc=0.0, scale=sigma_eps, size=n_scenarios)
    delta = np.random.normal(loc=0.0, scale=sigma_delta, size=n_scenarios)

    # Draw correlated normals for Vasicek (x, y)
    W1 = np.random.normal(size=n_scenarios)
    W2 = np.random.normal(size=n_scenarios)

    # Update mortality factors
    K_T = K0 + theta * dt + eps
    k_T = c + d * k0 + delta

    # Update interest‑rate factors via Euler discretization
    x_T = x0 + (-alpha * x0) * dt + nu * np.sqrt(dt) * W1
    y_T = (
        y0
        + (-beta * y0) * dt
        + eta * np.sqrt(dt) * (rho * W1 + np.sqrt(1 - rho**2) * W2)
    )

    # Stack into (n,4)
    return np.vstack([K_T, k_T, x_T, y_T]).T


class Net(nn.Module):
    def __init__(self, n_assets=20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_assets, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x)




# Example usage:
if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cuda")


    model = Net(n_assets=5).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=100)

    losses = []
    batch_size = int(100)
    gradient_steps = int(1e2)
    lrs = []
    model.train()
    params = Parameters()
    params = params.to_tensor(device=device, dtype=torch.float32)
    batch_size = int(2**9)
    for epoch in range(gradient_steps):
        K, k, x, y, r  = simulate_risk_factors(T=2, n_samples=batch_size, params=params, device='cuda', only_return_final=True)
        input_data = torch.stack([K, k, x, y, r], dim=1)
        output_data = simulate_payoff(K=K, k=k, x=x, y=y, r=r, params=params, max_age=4)
        predictions = model(input_data)
        loss = criterion(predictions, output_data)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        lrs.append(optimizer.param_groups[0]['lr'])
        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
    
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')    
    plt.xlim(0, gradient_steps)
    plt.title('Training Loss - Annuity Conversion Factor')
    plt.savefig("training_loss_annuity_conversion_factor.png")


    # params = Parameters()
    # params = params.to_tensor(device='cuda', dtype=torch.float32)
    # stacked = torch.stack([K, k, x, y, r], dim=1)
    # print(f'shape of input tensor is {stacked.shape}')
    # # print(stacked)
    # Y=simulate_payoff(K=K, k=k, x=x, y=y, r=r, params=params, max_age=4)
    # print(f'shape of output tensor is {Y.shape}')
    # print(Y)