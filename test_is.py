# %%
from max_call_option import MaxCallConfig, DataSampler
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from expected_shortfall import expected_shortfall

# %%
config = MaxCallConfig()
sampler = DataSampler(config, device=torch.device("cuda:0"), dtype=torch.float32,  seed=42)

alpha = 1 - 1e-6
n_samples = int(1e6)
X, _ = sampler.sampleX(importance_sampling=False, alpha=alpha, n_samples=n_samples)
Y = sampler.sampleY(initial_value=X)
Y, _ = torch.sort(Y, dim=-1, descending=True)

X_is, _ = sampler.sampleX(importance_sampling=True, alpha=alpha, n_samples=n_samples)
Y_is = sampler.sampleY(initial_value=X_is)
Y_is, _ = torch.sort(Y_is, dim=-1, descending=True)

ex, j = expected_shortfall(Y, alpha=alpha, is_decreasing=True)
ex_is, j_is = expected_shortfall(Y_is, alpha=alpha, is_decreasing=True)

print(j, j_is)

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
