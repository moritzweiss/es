import torch


class GBM():

    def __init__(self, drift, vola, corr, rng, A=None):
        device = drift.device
        assert drift.ndim == 1, "drift must be a 1-dimensional tensor"
        d = drift.shape[0]
        # build correlation matrix 
        assert corr.ndim == 0, 'only working for simple correlation structure'
        assert drift.shape == (d,)
        assert vola.shape  == (d,)
        self.n_assets = d
        self.drift = drift
        self.vola = vola
        self.corr = corr
        self.rng = rng
        if A is None:
            corr_matrix = torch.full((d,d), corr, device=device)
            corr_matrix.fill_diagonal_(1.0)
            A = torch.linalg.cholesky(corr_matrix)
            A = torch.diag(vola) @ A
            self.A = A
        else:
            self.A = A

                
    def simulate_increment(self, n_samples, time_increment, initial_value, Z=None, drift=None):
        # we offer the option to receive Z from outside for importance sampling
        # also offer option to overwrite default drift 
        if Z is None:
            Z = torch.randn(n_samples, self.n_assets, generator=self.rng, device=self.drift.device)
        else:
            assert Z.shape == (n_samples, self.n_assets)
        if drift is None:
            drift_term = (self.drift- 0.5*self.vola**2) * time_increment
        else:
            drift_term = (drift - 0.5*self.vola**2) * time_increment
        diffusion = Z @ self.A.T * torch.sqrt(time_increment)
        return initial_value*torch.exp(drift_term + diffusion)

    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sns
    print("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    drift = torch.tensor([0.01, 0.02], device=device)
    vola = torch.tensor([0.1, 0.2], device=device)
    corr = torch.tensor(0.9, device=device)
    initial_value = torch.tensor([100.0, 100.0], device=device)
    time_increment = torch.tensor(1/252, device=device)
    rng = torch.Generator(device=device).manual_seed(42)
    gbm = GBM(drift, vola, corr, rng)
    out, _ = gbm.simulate_increment(n_samples=int(1e6),initial_value=initial_value, time_increment=time_increment)    
    print(out.shape)
    out_np = out.cpu().numpy()
    sns.kdeplot(out_np[:, 0], fill=True)
    plt.xlabel('First Asset Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of First Asset Value')
    plt.savefig('plots/gbm_first_asset_histogram.png')