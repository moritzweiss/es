import torch

def simulate_XY(
    N,
    q0=torch.log(torch.tensor(100.0)),
    m=0.05,
    sigma_S=0.18,
    tau=1.0,
    r0=0.025,
    gamma=0.02,
    zeta=0.25,
    sigma_r=0.01,
    lambd=0.02,
    mu_x=0.01,
    kappa=0.07,
    sigma_mu=0.0012,
    rho12=-0.30,
    rho13=0.06,
    rho23=-0.04,
    b=10.792,
    T=15.0,
    K=50,
    device='cpu'
):
    """
    Vectorized simulation of (X, Y) for the GMIB example in PyTorch.
    Returns:
        X: Tensor of shape (N, 3) containing (q_tau, r_tau, mu_tau)
        Y: Tensor of shape (N,) containing discounted payoffs
    """
    # Move constants to tensor
    q0 = torch.tensor(q0, device=device)
    m = torch.tensor(m, device=device)
    sigma_S = torch.tensor(sigma_S, device=device)
    tau = torch.tensor(tau, device=device)
    r0 = torch.tensor(r0, device=device)
    gamma = torch.tensor(gamma, device=device)
    zeta = torch.tensor(zeta, device=device)
    sigma_r = torch.tensor(sigma_r, device=device)
    lambd = torch.tensor(lambd, device=device)
    mu_x = torch.tensor(mu_x, device=device)
    kappa = torch.tensor(kappa, device=device)
    sigma_mu = torch.tensor(sigma_mu, device=device)
    rho12 = torch.tensor(rho12, device=device)
    rho13 = torch.tensor(rho13, device=device)
    rho23 = torch.tensor(rho23, device=device)
    b = torch.tensor(b, device=device)
    T = torch.tensor(T, device=device)
    
    # Precompute constants
    gamma_bar = gamma - lambd * sigma_r / zeta
    dt = T - tau
    
    # Correlation matrix and Cholesky
    corr = torch.tensor([[1.0, rho12, rho13],
                         [rho12, 1.0, rho23],
                         [rho13, rho23, 1.0]], device=device)
    L = torch.linalg.cholesky(corr)
    
    # Stage 1: simulate X_tau under P
    Z1 = torch.randn(N, 3, device=device)
    Z1_corr = Z1 @ L.T
    
    alpha = m - 0.5 * sigma_S**2
    q_tau = q0 + alpha * tau + sigma_S * torch.sqrt(tau) * Z1_corr[:, 0]
    
    r_tau = (
        r0 * torch.exp(-zeta * tau)
        + gamma * (1 - torch.exp(-zeta * tau))
        + sigma_r * torch.sqrt((1 - torch.exp(-2*zeta*tau)) / (2*zeta)) * Z1_corr[:, 1]
    )
    
    mu_tau = (
        mu_x * torch.exp(kappa * tau)
        + sigma_mu * torch.sqrt((torch.exp(2*kappa*tau) - 1) / (2*kappa)) * Z1_corr[:, 2]
    )
    
    X = torch.stack([q_tau, r_tau, mu_tau], dim=1)
    
    # Stage 2: simulate to T under Q
    Z2 = torch.randn(N, 3, device=device)
    Z2_corr = Z2 @ L.T
    
    # q_T
    q_T = q_tau + (r_tau - 0.5 * sigma_S**2) * dt + sigma_S * torch.sqrt(dt) * Z2_corr[:, 0]
    
    # r_T under Q
    mean_r_T = r_tau * torch.exp(-zeta * dt) + gamma_bar * (1 - torch.exp(-zeta * dt))
    var_r_T = (sigma_r**2) / (2*zeta) * (1 - torch.exp(-2*zeta * dt))
    r_T = mean_r_T + torch.sqrt(var_r_T) * Z2_corr[:, 1]
    
    # mu_T
    mean_mu_T = mu_tau * torch.exp(kappa * dt)
    var_mu_T = (sigma_mu**2) / (2*kappa) * (torch.exp(2*kappa * dt) - 1)
    mu_T = mean_mu_T + torch.sqrt(var_mu_T) * Z2_corr[:, 2]
    
    # Affine discount factor functions
    def Br(u): 
        return (1 - torch.exp(-zeta * u)) / zeta
    def Bmu(u): 
        return (torch.exp(kappa * u) - 1) / kappa
    def A_affine(u):
        br = Br(u)
        bm = Bmu(u)
        dt_u = u
        term_r = (sigma_r**2 / zeta**2) * (dt_u - 2*br + (1 - torch.exp(-2*zeta*dt_u)) / (2*zeta))
        term_mu = (sigma_mu**2 / kappa**2) * (dt_u - 2*bm + (torch.exp(2*kappa*dt_u) - 1) / (2*kappa))
        term_cross = (2 * rho23 * sigma_r * sigma_mu / (zeta * kappa)) * (
            bm - dt_u + br - (1 - torch.exp(-(zeta - kappa)*dt_u)) / (zeta - kappa)
        )
        return torch.exp(gamma_bar * (br - dt_u) + 0.5 * (term_r + term_mu + term_cross))
    def F_affine(u, r0_, mu0_):
        return A_affine(u) * torch.exp(-Br(u) * r0_ - Bmu(u) * mu0_)
    
    # Discount factor from tau to T
    F_tau = F_affine(dt, r_tau, mu_tau)
    
    # Compute annuity factor A = sum_{k=1}^K F_affine(k, r_T, mu_T)
    ks = torch.arange(1, K+1, device=device, dtype=torch.float32)
    Br_k = Br(ks)
    Bmu_k = Bmu(ks)
    A_k = A_affine(ks)
    rT_expanded = r_T.unsqueeze(1)
    muT_expanded = mu_T.unsqueeze(1)
    F_k = A_k.unsqueeze(0) * torch.exp(-Br_k.unsqueeze(0) * rT_expanded - Bmu_k.unsqueeze(0) * muT_expanded)
    A_sum = F_k.sum(dim=1)
    
    S_T = torch.exp(q_T)
    Y = F_tau * torch.maximum(S_T, b * A_sum)
    
    return X, Y

# Example usage:
if __name__ == "__main__":
    X_sample, Y_sample = simulate_XY(10000, device='cpu')
    print("X_sample shape:", X_sample.shape)
    print("Y_sample shape:", Y_sample.shape)
