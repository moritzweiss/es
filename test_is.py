# %%
from max_call_option import MaxCallConfig, DataSampler, PortfolioConfig
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from expected_shortfall import expected_shortfall


#%%
config = PortfolioConfig()
sampler = DataSampler(config, device=torch.device("cuda:0"), dtype=torch.float32,  seed=42)

alpha = 0.99
sampling_alpha = 0.8

n_samples = int(1e6)
X, _ = sampler.sampleX(importance_sampling=False, alpha=sampling_alpha, n_samples=n_samples)
Y = sampler.sampleY(initial_value=X)
Y, _ = torch.sort(Y, dim=-1, descending=True)

X_is, weights = sampler.sampleX(importance_sampling=True, alpha=sampling_alpha, n_samples=n_samples)
Y_is = sampler.sampleY(initial_value=X_is)
Y_is, idx = torch.sort(Y_is, dim=-1, descending=True)
weights = weights[idx]

weights = weights / weights.sum(dim=-1, keepdim=True)

print(f'mean payoff without IS: {Y.mean()}')
print(f'mean payoff with IS: {(Y_is*weights).sum()}')


ex, j = expected_shortfall(Y, alpha=alpha, make_decreasing=False)
ex_is, j_is = expected_shortfall(losses=Y_is, alpha=alpha, make_decreasing=False, sample_weights=weights, normalize=False)

print(f'number of samples in upper tail with IS: {j_is/1000}, without IS: {j/1000}')

print(f"Expected Shortfall without IS: {ex:.4f}")
print(f"Expected Shortfall with IS: {ex_is:.4f}")

print(f'value at risk without IS: {Y[j]:.4f}')
print(f'value at risk with IS: {Y_is[j_is]:.4f}')


Y = Y.cpu().numpy() if torch.is_tensor(Y) else Y
Y_is = Y_is.cpu().numpy() if torch.is_tensor(Y_is) else Y_is
plt.figure(figsize=(10, 6))
# plt.xlim([0,20])
# plt.ylim([0, 0.5])
# sns.kdeplot(Y, bw_adjust=0.2, label='Y', fill=False, alpha=0.5)
# sns.kdeplot(Y_is, bw_adjust=0.2, label='Y with IS', fill=False, alpha=0.5)
plt.hist(Y, bins=200, alpha=0.5, label='Y', density=True)
plt.hist(Y_is, bins=200, alpha=0.5, label='Y with IS', density=True)
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()
plt.title('Density Plot of Y ')



# %%
config = MaxCallConfig()
sampler = DataSampler(config, device=torch.device("cuda:1"), dtype=torch.float32,  seed=42)


alpha = 0.99
sampling_alpha = 0.99

n_samples = int(1e6)
X, _ = sampler.sampleX(importance_sampling=False, alpha=sampling_alpha, n_samples=n_samples)
Y = sampler.sampleY(initial_value=X)
Y, _ = torch.sort(Y, dim=-1, descending=True)

X_is, weights = sampler.sampleX(importance_sampling=True, alpha=sampling_alpha, n_samples=n_samples)
Y_is = sampler.sampleY(initial_value=X_is)
Y_is, idx = torch.sort(Y_is, dim=-1, descending=True)
weights = weights[idx]

weights = weights / weights.sum(dim=-1, keepdim=True)

print(f'mean payoff without IS: {Y.mean()}')
print(f'mean payoff with IS: {(Y_is*weights).sum()}')


ex, j = expected_shortfall(Y, alpha=alpha, make_decreasing=False)
ex_is, j_is = expected_shortfall(losses=Y_is, alpha=alpha, make_decreasing=False, sample_weights=weights, normalize=False)

print(f'number of samples in upper tail with IS: {j_is/1000}, without IS: {j/1000}')

print(f"Expected Shortfall without IS: {ex:.4f}")
print(f"Expected Shortfall with IS: {ex_is:.4f}")

print(f'value at risk without IS: {Y[j]:.4f}')
print(f'value at risk with IS: {Y_is[j_is]:.4f}')


Y = Y.cpu().numpy() if torch.is_tensor(Y) else Y
Y_is = Y_is.cpu().numpy() if torch.is_tensor(Y_is) else Y_is
plt.figure(figsize=(10, 6))
plt.xlim([0,20])
plt.ylim([0, 0.5])
# sns.kdeplot(Y, bw_adjust=0.2, label='Y', fill=False, alpha=0.5)
# sns.kdeplot(Y_is, bw_adjust=0.2, label='Y with IS', fill=False, alpha=0.5)
plt.hist(Y, bins=200, alpha=0.5, label='Y', density=True)
plt.hist(Y_is, bins=200, alpha=0.5, label='Y with IS', density=True)
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()
plt.title('Density Plot of Y ')


# %%
