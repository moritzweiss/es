import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from dataclasses import dataclass, field, fields 
from typing import List
from simulate_gbm import GBM 
import numpy as np
from helper_functions import estimates
import pandas as pd

@dataclass
class PortfolioConfig:
    interest_rate: float = 0.01
    final_time: float = 1/3
    intermediate_time: float = 1/52
    initial_value: float = 100.0
    corr: float = 0.30
    n_assets : int = 20
    drift: List[float] = field(default_factory=lambda: [(2.5 + ((i % 10 + 1) / 2)) / 100 for i in range(20)])
    vola: List[float] = field(default_factory=lambda: [(14 + (i % 10 + 1)) / 100 for i in range(20)])
    strike: float = 100.0
    n_calls: int = 10
    n_puts: int = 10

def is_weights(Z: torch.Tensor, v: torch.Tensor):
    '''
    computes the importance sampling weights for samples Z shifted by v. the incoming samples are Z+v. 
    Z: (n_samples, n_assets)
    v: (n_assets,)
    returns: (n_samples,)
    Z is a shifted normal with distribution N(v, I)
    the RD derivative is phi(z)/phi_v(z) = exp(-x.v+0.5*(v.v)^2)
    using those weights we can compute expectations with respect to the original distribution N(0,I) by multiplying with those weights
    the weights may be normalized furthere by dividing by N or by normalizing 
    '''
    assert Z.dim() == 2
    assert v.dim() == 1
    assert Z.shape[1] == v.shape[0]
    return torch.exp(-(Z*v).sum(dim=1) + 0.5*(v*v).sum())

class DataSampler():

    def __init__(self, config, device, dtype, n_samples, seed):        

        self.device = device
        self.dtype = dtype
        for f in fields(config):
            value = getattr(config, f.name)
            tensor_value = self._to_tensor(value)
            setattr(self, f.name, tensor_value)

        rng = torch.Generator(device=device)
        rng.manual_seed(seed)
        self.rng = rng
        # build A here explicitely we need it for computing the shift for IS 
        A = self._cholesky_factor(corr=self.corr, n_assets=self.n_assets, vola=self.vola)
        self.A = A
        self.GBM = GBM(drift=self.drift, vola=self.vola, corr=self.corr, rng=rng, A=A)
        self.n_samples = n_samples
    
    def _cholesky_factor(self,corr, n_assets, vola):
        corr_matrix = torch.full((n_assets,n_assets), corr, device=self.device)
        corr_matrix.fill_diagonal_(1.0)
        A= torch.linalg.cholesky(corr_matrix)
        A = torch.diag(vola) @ A
        return A 
    
    def _to_tensor(self, value):
        if isinstance(value, float): 
            return torch.tensor(value, device=self.device, dtype=self.dtype)
        elif isinstance(value, (list, tuple)) and all(isinstance(v, float) for v in value):
            return torch.tensor(value, device=self.device, dtype=self.dtype)
        elif isinstance(value, np.ndarray):
            return torch.tensor(value, device=self.device, dtype=self.dtype)
        else:
            return value

    @torch.no_grad()        
    def sampleX(self, importance_sampling=False):
        '''
        returns X and sampling weights 
        '''
        if importance_sampling:                        
            v = self.shift_vector()
        else:
            v = torch.zeros(self.n_assets, device=self.device, dtype=self.dtype)
        Z = torch.randn(self.n_samples, self.n_assets, device=self.device, generator=self.rng)
        Z += v
        weights = is_weights(Z, v)
        if not importance_sampling:
            assert torch.allclose(weights, torch.ones_like(weights)), "Weights should be all ones when importance_sampling is False"
        return self.GBM.simulate_increment(n_samples=self.n_samples, initial_value=self.initial_value, time_increment=self.intermediate_time, Z=Z), weights, Z 

    def shift_vector(self, alpha=0.95):         
         v = torch.tensor([1.0]*self.n_calls+[-1.0]*self.n_puts, device=self.device, dtype=self.dtype)
         quantile = torch.distributions.Normal(0,1).icdf(torch.tensor(alpha, device=self.device, dtype=self.dtype))            
         v = self.A.T @ v
         return (quantile*v) / torch.linalg.vector_norm(v) 

    @torch.no_grad()             
    def sampleY(self, initial_value):
        dt = self.final_time - self.intermediate_time
        # is this correct? we should sample under the risk neutral measure! added additional drift option to GBM! 
        Y = self.GBM.simulate_increment(n_samples=self.n_samples, initial_value=initial_value, time_increment=dt, drift=self.interest_rate)
        assert Y.shape == (self.n_samples, self.n_assets)
        Y = torch.relu(Y[:,:self.n_calls]-self.strike) + torch.relu(self.strike-Y[:,self.n_calls:])
        Y = Y.sum(dim=1)
        Y = torch.exp(-self.interest_rate*(self.final_time-self.intermediate_time))*Y
        return Y.unsqueeze(1)


class DeepNeuralNet(nn.Module):
    def __init__(self, n_features=20, hidden_layer=128):
        super().__init__()
        self.net = nn.Sequential(
            # big gain from using wider neural network with two hidden layers ! 
            nn.Linear(n_features, hidden_layer),
            nn.BatchNorm1d(hidden_layer),
            nn.GELU(),
            # nn.Tanh(),
            nn.Linear(hidden_layer, hidden_layer),
            nn.BatchNorm1d(hidden_layer),
            nn.GELU(),
            # nn.Tanh(),
            nn.Linear(hidden_layer, 1))

    def forward(self, x):
        return self.net(x)


class LinearModel(nn.Module):
    def __init__(self, n_features=20):
        super().__init__()
        self.net = nn.Linear(n_features, 1)

    def forward(self, x):
        return self.net(x)

class PolynomialModel(nn.Module):
    def __init__(self, n_features, mean, std):
        super().__init__()
        self.n_input_features = n_features
        self.n_out_features = int(n_features+(n_features*(n_features+1))/2)
        self.linear = nn.Linear(self.n_out_features, 1, bias=True)
        self.mean = mean
        self.std = std

    def forward(self, x):
        assert x.dim() == 2
        assert x.shape[1] == self.n_input_features
        # create quadratic features
        batch_size = x.shape[0]
        iu = torch.triu_indices(self.n_input_features, self.n_input_features, offset=0, device=x.device)
        # standardize quadratic features
        x_norm = (x-self.mean)/self.std
        # x_norm = x 
        pairs = x_norm.unsqueeze(2) * x_norm.unsqueeze(1)  # (batch_size, n_input_features, n_input_features)
        quad_features = pairs[:, iu[0], iu[1]]  # (batch_size, n_out_features) 
        # end of quatratic features
        all_features = torch.cat([x, quad_features], dim=1)  # (batch_size, n_out_features)
        assert all_features.shape == (batch_size, self.n_out_features)
        # apply linear layer with bias 
        return self.linear(all_features)

def solve_lstsq(X: torch.Tensor, Y: torch.Tensor, cond: float):
    """
    Ordinary least squares with intercept using torch.linalg.lstsq(driver='gelsd').
    X: (N, p) feature matrix (no intercept yet)
    Y: (N, 1)
    Returns:
      beta: (p,) coefficients
      bias: scalar intercept
    """
    N, p = X.shape
    device, dtype = X.device, X.dtype

    # work in CPU/float64 for LAPACK gelsd
    Xd = X.detach().to("cpu", torch.float64)
    Yd = Y.detach().to("cpu", torch.float64)

    ones = torch.ones(N, 1, device=Xd.device, dtype=Xd.dtype)
    X_aug = torch.cat([Xd, ones], dim=1)  # add intercept column

    res = torch.linalg.lstsq(X_aug, Yd, driver="gelsd", rcond=cond)
    theta = res.solution                   # (p+1, 1)

    beta = theta[:-1, 0].to(device, dtype) # (p,)
    bias = theta[-1, 0].to(device, dtype)  # scalar
    return beta, bias, res

class PolyClosedForm(nn.Module):
    def __init__(self, n_features: int, mean: torch.Tensor, std: torch.Tensor, cond: float):
        super().__init__()
        self.n_input_features = n_features
        self.n_out_features = int(n_features + (n_features * (n_features + 1)) // 2)
        self.mean = mean
        self.std = std
        self.linear = nn.Linear(self.n_out_features, 1, bias=True)
        self.cond = cond 

    def _transform(self, X: torch.Tensor):
        x_norm = (X-self.mean)/self.std
        iu = torch.triu_indices(self.n_input_features, self.n_input_features, offset=0, device=X.device)
        pairs = x_norm.unsqueeze(2) * x_norm.unsqueeze(1)  # (batch_size, n_input_features, n_input_features)
        quad_features = pairs[:, iu[0], iu[1]]  # (batch_size, n_out_features)                 
        X = torch.cat([X, quad_features], dim=1)
        return X

    @torch.no_grad()
    def fit_from_batch(self, X: torch.Tensor, Y: torch.Tensor):
        # 
        Z = self._transform(X)
        beta, bias, res = solve_lstsq(Z, Y, cond=self.cond)
        self.linear.weight.copy_(beta.view(1, -1))
        self.linear.bias.copy_(bias.view(()))
        info = {"rank": int(res.rank), "cond": (res.singular_values[0] / res.singular_values[-1]).item() if res.singular_values.numel() > 1 else float("inf")}
        return info
    
    def forward(self, x):
        Z = self._transform(x)
        return self.linear(Z)


def model_registry(name, config, mean=None, std=None, cond=1e-2, hidden_layer=32):
    if name == "DNN":
        return DeepNeuralNet(n_features=config.n_assets, hidden_layer=32)
    elif name == "Linear":
        return LinearModel(n_features=config.n_assets)
    elif name == "Poly":
        return PolynomialModel(n_features=config.n_assets, mean=mean, std=std)
    elif name == "PolyClosedForm":
        return PolyClosedForm(n_features=config.n_assets, mean=mean, std=std, cond=cond)
    else:
        raise ValueError(f"Unknown model name: {name}")


if __name__ == "__main__":
    # this is all still a bit hack. can modify this stuff later on. 
    # for the neural network IS or no IS does not seem to matter that much 
    # for the other models it helps 
    # faster convergence for RELU or GELU rather than TANH 
    # batch normalization helps with tanh()-activation 
    # relu with batch normalization works better than without batch normalization  
    # training parameters 
    # reference value is 105.59 for 99%ES. the nn+relu combination gets very close to this! 
    device = torch.device("cuda")
    batch_size = int(2**18)
    # initial value of learning rate is important for NN training. initial larger is better.  
    # minimum loss is somewhere around 1103
    # ES estimates are around 105 +- 1 with neural networks 
    # learning_rate = 1e-3
    learning_rate = 1e-3
    gradient_steps = int(2e3)
    train_seed = 0 
    importance_sampling = False
    # evaluation parameters †
    eval_seed = 100 
    # more eval samples worked better for the neural net ! probably some randomness involved here. 
    # can increaase for final result 
    # it seems that IS makes NN peformance worse or at least does not improve it
    n_eval_samples = int(3e6)
    n_eval_samples = int(1e6)
    alpha = 0.99

    # get the config 
    PC = PortfolioConfig()

    # to collect results 
    results = []

    # get mean and std for a large batch of X. this is used for poly regression. 
    DS = DataSampler(config=PC, device=device, dtype=torch.float32, n_samples=int(1e6), seed=train_seed)
    X, weights, Z = DS.sampleX(importance_sampling=importance_sampling)
    X_mean = X.mean(dim=0, keepdim=True)
    X_std = X.std(dim=0, keepdim=True)          
    print("\nFeature normalization:")
    print(f"X mean: {X_mean}")
    print(f"X std: {X_std}") 

    models = ["Linear", "Poly", "DNN"]
    # models = ["DNN"]
    for model_name in models:
        print('\n-------------------------------\n')
        print(f"Training model: {model_name}")
        # setting up data sampler, model, loss, optimizer 
        # mean and std is only used for the polynomial regression 
        DS = DataSampler(config=PC, device=device, dtype=torch.float32, n_samples=batch_size, seed=train_seed)
        # closed form ridge is not working very well, or at least not better than gradient descent 
        # mean and std is passed to polynomial regression forward loop 
        model = model_registry(name=model_name, config=PC, mean=X_mean, std=X_std, cond=1e-10, hidden_layer=128).to(device)
        model.train()
        criterion = nn.MSELoss()
        optimizer= optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=100, min_lr=1e-6)
        # for logging 
        losses = []
        lrs = []
        # training loop
        if model_name == "PolyClosedForm":
            # this doesnt work well 
            DS.n_samples = int(1e6)
            X, _, _ = DS.sampleX(importance_sampling=importance_sampling)
            Y = DS.sampleY(initial_value=X)
            model.fit_from_batch(X=X, Y=Y)
        else:
            for epoch in range(gradient_steps):
                # Z are the standard normal samples before the IS or GBM transformation 
                X, weights, Z = DS.sampleX(importance_sampling=importance_sampling)
                # X = normalize(X, mean=X_mean, std=X_std)
                strike = 100.0
                # X = torch.log(X/strike) works worse 
                Y = DS.sampleY(initial_value=X)
                optimizer.zero_grad()
                outputs = model(X)
                mse_loss = criterion(outputs, Y)
                # use l1 penalty with polynomial regression
                if model_name == "Poly":
                    lambda_reg = 10.0 # choice for L1 penalty 
                    # lambda_reg = 5.0
                    l1_penalty = lambda_reg * model.linear.weight.abs().sum()
                    # l1_penalty = lambda_reg * torch.linalg.vector_norm(model.linear.weight, ord=2)
                    # ridge penalty works worse than l1 penalty 
                    loss = mse_loss + l1_penalty    
                else:
                    loss = mse_loss
                loss.backward()
                optimizer.step()
                scheduler.step(loss)
                losses.append(loss.item())
                lrs.append(optimizer.param_groups[0]['lr'])
                if (epoch+1) % 100 == 0 or epoch == 0:
                    print(f"Epoch [{epoch+1}/{gradient_steps}], Loss: {loss.item():.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
                    if model_name == "Poly":
                        print(f'l1/mse%: {100*l1_penalty/mse_loss}')

        plt.figure()
        plt.plot(losses)
        plt.xlabel("gradient step")
        plt.ylabel("mean squared error loss")
        plt.xlim(0, gradient_steps)
        plt.title(f"portfolio of options, batch size={batch_size}")
        plt.savefig(f"plots/portfolio_training_loss_{model_name}.png")

        # evaluation. should reset seed here. just resetting the data sampler for now. 
        DS = DataSampler(config=PC, device=device, dtype=torch.float32, n_samples=n_eval_samples, seed=eval_seed)
        X, sampling_weights, Z = DS.sampleX(importance_sampling=importance_sampling)
        # DNN and Linear Model work well with raw features X. Normalize quadratic terms for Poly-Regression 
        # moneyness 
        # works worse 
        # X = torch.log(X/strike) 
        outputs = model(X)
        outputs = outputs.flatten()
        # weights are always provided. in case of no IS, the shift vector is zero and weights are all one 
        # we normalize by number of samples. but could also normalize by sum(sampling_weights) and normalize weights to one 
        sampling_weights = sampling_weights.flatten() 
        # sampling_weights = sampling_weights / len(sampling_weights)  
        sampling_weights = sampling_weights/ sum(sampling_weights)
        # get auxilary variables Y and Z 
        Y = DS.sampleY(initial_value=X)
        Y = Y.flatten()
        Z = DS.sampleY(initial_value=X)
        Z = Z.flatten()
        # compute ES and relative errors. TODO: write more tests for those values. 
        es, err1, err2, _, _ = estimates(predictions=outputs, y=Y, z=Z, alpha=alpha, sampling_weights=sampling_weights)
        print(f"expected shortfall estimate: {es.item():.4f}")
        print(f"first error estimate: {err1.item():.4f}")
        print(f"second error estimate: {err2.item():.4f}")
        # logging 
        results.append({
            "model": model_name,
            "expected_shortfall": es.item(),
            "norm": err1.item(),
            "tail_norm": err2.item()
        })
    # writing results into csv or latex table 
    df = pd.DataFrame(results)
    df = df.set_index("model")
    df.to_csv("portfolio_options_results.csv", float_format="%.2f")
    to_latex = {"expected_shortfall": r"$\mathrm{ES}_{\alpha}(\hat{f}(X))$",
                "norm": r"$\frac{\|\hat f - \bar f\|_{L^2(\nu)}}{\mathrm{ES}_{\alpha}(\hat{f}(X))}$",
                "tail_norm": r"$\frac{\|\hat f - \bar f\|_{L^2(\hat \nu_\alpha)}}{\mathrm{ES}_{\alpha}(\hat{f}(X))}$"}
    df = df.rename(columns=to_latex)
    colfmt = "l" + "c" * len(df.columns) 
    if importance_sampling:
        caption = "Portfolio of call and put options with importance sampling."
    else:
        caption = "Portfolio of call and put options."
    latex_str = df.to_latex(float_format="%.4f", escape=False, index_names=False, column_format=colfmt)            
    # add centering to the table 
    latex_str = ("\\begin{table}[ht]\n"
    "\\centering\n" +
    latex_str +
    f"\\caption{{{caption}}}\n"
    "\\end{table}\n")
    if importance_sampling:
        file_name = f'results_with_is.tex'
    else:
        file_name = 'results.tex'
    with open(file_name, "w") as f:
        f.write(latex_str)