import numpy as np

def simulate_X(N, q0, m, sigma_S, tau, 
               r0, gamma, zeta, sigma_r, 
               mu_x, kappa, sigma_mu, 
               rho12, rho13, rho23):
    """
    Vectorized simulation of X = (q_tau, r_tau, mu_x+tau) for N paths.
    """
    
    # Build correlation matrix and its Cholesky factor
    corr = np.array([[1, rho12, rho13],
                     [rho12, 1, rho23],
                     [rho13, rho23, 1]])
    L = np.linalg.cholesky(corr)
    
    # Simulate independent standard normals and introduce correlations
    Z = np.random.normal(size=(N, 3))               
    Z_corr = Z @ L.T                                
    
    # Compute each component vectorized
    q_tau = q0 + (m - 0.5 * sigma_S**2) * tau + sigma_S * np.sqrt(tau) * Z_corr[:, 0]    
    r_tau = r0 * np.exp(-zeta * tau) + gamma * (1 - np.exp(-zeta * tau))  + sigma_r * np.sqrt((1 - np.exp(-2*zeta*tau)) / (2*zeta)) * Z_corr[:, 1]    
    mu_tau = mu_x * np.exp(kappa * tau) + sigma_mu * np.sqrt((np.exp(2*kappa*tau) - 1) / (2*kappa)) * Z_corr[:, 2]
    
    
    return q_tau, r_tau, mu_tau

# Example usage:
params = {
    'N': 100000,
    'q0': np.log(100),
    'm': 0.05,
    'sigma_S': 0.18,
    'tau': 1.0,
    'r0': 0.025,
    'gamma': 0.02,
    'zeta': 0.25,
    'sigma_r': 0.01,
    'mu_x': 0.01,
    'kappa': 0.07,
    'sigma_mu': 0.0012,
    'rho12': -0.30,
    'rho13': 0.06,
    'rho23': -0.04,
}

q_tau_samples, r_tau_samples, mu_tau_samples = simulate_X(**params)

# Verify shapes
print("Shapes:", q_tau_samples.shape, r_tau_samples.shape, mu_tau_samples.shape)
