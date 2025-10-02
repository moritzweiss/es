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
class MaxCallConfig:
    interest_rate: float = 0.00          
    final_time: float = 1/3             
    intermediate_time: float = 1/52
    initial_value: float = 10.0       
    corr: float = 0.30                  
    n_assets: int = 100                  
    vola: List[float] = field(default_factory=lambda: [( 10 + (i+1)/2 ) / 100 for i in range(100)])
    drift: List[float] = field(default_factory=lambda: [0.0]*100)  
    strike: float = 16.4

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

    def __init__(self, config, device, dtype, seed):        

        self.device = device
        self.dtype = dtype
        for f in fields(config):
            value = getattr(config, f.name)
            tensor_value = self._to_tensor(value)
            setattr(self, f.name, tensor_value)
        self.rng = torch.Generator(device=device)
        self.set_seed(seed)
        A = self._cholesky_factor(corr=self.corr, n_assets=self.n_assets, vola=self.vola)
        self.A = A
        self.GBM = GBM(drift=self.drift, vola=self.vola, corr=self.corr, rng=self.rng, A=A)
    
    def set_seed(self, seed):
        self.rng.manual_seed(seed)
    
    def _cholesky_factor(self,corr, n_assets, vola):
        corr_matrix = torch.full((n_assets,n_assets), corr, device=self.device)
        corr_matrix.fill_diagonal_(1.0)
        A = torch.linalg.cholesky(corr_matrix)
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
    def sampleX(self, n_samples, importance_sampling=False, alpha=0.95):
        '''
        returns X and sampling weights 
        '''
        Z = torch.randn(n_samples, self.n_assets, device=self.device, generator=self.rng)
        if importance_sampling:                        
            v = self.shift_vector(alpha=alpha)
            Z += v
            weights = is_weights(Z, v)
        else:
            # equal weight option 
            weights = torch.ones(n_samples, device=self.device)
        return self.GBM.simulate_increment(n_samples=n_samples, initial_value=self.initial_value, time_increment=self.intermediate_time, Z=Z), weights.flatten()

    @torch.no_grad()
    def shift_vector(self, alpha):         
         v = torch.tensor([1.0]*self.n_assets, device=self.device, dtype=self.dtype)
         quantile = torch.distributions.Normal(0,1).icdf(torch.tensor(alpha, device=self.device, dtype=self.dtype))            
         v = self.A.T @ v
         return (quantile*v) / torch.linalg.vector_norm(v) 

    @torch.no_grad()             
    def sampleY(self, initial_value):
        n_samples = initial_value.shape[0]
        dt = self.final_time - self.intermediate_time
        Y = self.GBM.simulate_increment(n_samples=n_samples, initial_value=initial_value, time_increment=dt)
        assert Y.shape == (n_samples, self.n_assets)
        return torch.relu(torch.max(Y, dim=1)[0] - self.strike).flatten()

class PortfolioDataSampler(DataSampler):
    """DataSampler specialized for portfolio of call and put options"""
    
    def __init__(self, config, device, dtype, n_samples, seed):
        # Initialize parent class but override n_samples behavior
        super().__init__(config, device, dtype, seed)
        self.n_samples = n_samples
        
    @torch.no_grad()        
    def sampleX(self, importance_sampling=False, alpha=0.95):
        '''
        returns X, sampling weights, and Z (pre-transformation samples)
        '''
        if importance_sampling:                        
            v = self.shift_vector_portfolio(alpha=alpha)
        else:
            v = torch.zeros(self.n_assets, device=self.device, dtype=self.dtype)
        Z = torch.randn(self.n_samples, self.n_assets, device=self.device, generator=self.rng)
        Z += v
        weights = is_weights(Z, v)
        if not importance_sampling:
            assert torch.allclose(weights, torch.ones_like(weights)), "Weights should be all ones when importance_sampling is False"
        X = self.GBM.simulate_increment(n_samples=self.n_samples, initial_value=self.initial_value, time_increment=self.intermediate_time, Z=Z)
        return X, weights, Z 

    def shift_vector_portfolio(self, alpha=0.95):         
        v = torch.tensor([1.0]*self.n_calls+[-1.0]*self.n_puts, device=self.device, dtype=self.dtype)
        quantile = torch.distributions.Normal(0,1).icdf(torch.tensor(alpha, device=self.device, dtype=self.dtype))            
        v = self.A.T @ v
        return (quantile*v) / torch.linalg.vector_norm(v) 

    @torch.no_grad()             
    def sampleY(self, initial_value):
        dt = self.final_time - self.intermediate_time
        # Sample under the risk neutral measure
        Y = self.GBM.simulate_increment(n_samples=self.n_samples, initial_value=initial_value, time_increment=dt, drift=self.interest_rate)
        assert Y.shape == (self.n_samples, self.n_assets)
        Y = torch.relu(Y[:,:self.n_calls]-self.strike) + torch.relu(self.strike-Y[:,self.n_calls:])
        Y = Y.sum(dim=1)
        Y = torch.exp(-self.interest_rate*(self.final_time-self.intermediate_time))*Y
        return Y.unsqueeze(1)

class TailSampler:
    def __init__(self, data_sampler, top=0.10, ratio=0.7):
        self.data_sampler = data_sampler
        self.top = top 
        self.ratio = ratio

    @torch.no_grad()
    def sampleX(self, n_samples):
        importance_sampling=False
        required_tail_samples = int(self.ratio * n_samples)        
        n = 0 
        tail_samples = []
        while n < required_tail_samples:
            X, _ = self.data_sampler.sampleX(n_samples=n_samples, importance_sampling=importance_sampling, alpha=None)
            Y = self.data_sampler.sampleY(initial_value=X)
            threshold = torch.quantile(Y, 1-self.top)
            mask = Y >= threshold
            tail_samples.append(X[mask])
            n += X[mask].shape[0]
        X_tail = torch.cat(tail_samples, dim=0)[:required_tail_samples]
        X_regular, _ = self.data_sampler.sampleX(n_samples=n_samples, importance_sampling=importance_sampling, alpha=None)
        n_non_tail_samples = n_samples - required_tail_samples
        X_non_tail = X_regular[:n_non_tail_samples]
        X_combined = torch.cat([X_tail, X_non_tail], dim=0)
        return X_combined
    
class DeepNeuralNet(nn.Module):
    # network architecture has an effect on the results 
    # deeper not necessarily better
    # estimate in the tail are not better than baselines with importance sampling 
    # small network is close to linear regression but worse than polynomial regression
    # larger network performs worse for the tail estimates 
    def __init__(self, n_features=20, hidden_layer=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            # droput makes it worse
            # nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            # nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            # nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x).flatten()

class PolynomialNet(nn.Module):
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
        batch_size = x.shape[0]
        iu = torch.triu_indices(self.n_input_features, self.n_input_features, offset=0, device=x.device)
        x_norm = (x-self.mean)/self.std
        pairs = x_norm.unsqueeze(2) * x_norm.unsqueeze(1)  
        quad_features = pairs[:, iu[0], iu[1]]  
        all_features = torch.cat([x, quad_features], dim=1)  
        assert all_features.shape == (batch_size, self.n_out_features)
        return self.linear(all_features)

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

class Transform():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def normalize(self, X):
        return (X - self.mean) / self.std
    
    def unnormalize(self, X):
        return X * self.std + self.mean

class Regression():
    ''''
    poly transformation is faster on GPU but can run out of memory
    if memory is enough set transform_on_compute_device to False 
    the svd solution method is only implemented on CPU in torch
    compute_device is set to CPU to allow SVD solution method 
    '''
    def __init__(self, poly=False, tol=1e-2, compute_device=torch.device("cpu")):
        if poly:
            assert compute_device.type == "cpu", "Polynomial regression only implemented on CPU"
        self.poly = poly
        self.tol = tol
        self.compute_device = compute_device
        self.transform_on_compute_device = True

    def transform(self, X):
        # the transformation is faster on GPU. but need more memory for this. can check other adas. 
        if self.transform_on_compute_device:
            X = X.to(self.compute_device)
        if self.poly:
            n_features = X.shape[1]
            iu = torch.triu_indices(n_features, n_features, offset=0, device=X.device)
            pairs = X.unsqueeze(2) * X.unsqueeze(1)  
            quad_features = pairs[:, iu[0], iu[1]]  
            X = torch.cat([torch.ones(X.shape[0], 1, device=X.device), X, quad_features], dim=1)  
        else:
            X = torch.cat([torch.ones(X.shape[0], 1, device=X.device), X], dim=1)
        return X

    def fit(self, X, Y):
        X = self.transform(X)
        X = X.to(self.compute_device)
        Y = Y.to(self.compute_device)
        if self.compute_device.type == "cpu":
            # svd decomposition only works on CPU
            out = torch.linalg.lstsq(X, Y, rcond=self.tol, driver='gelsd')
        else:
            # gels only one available on GPU. assumes that A is full rank. rcond is not used. 
            out = torch.linalg.lstsq(X, Y, driver='gels')      
        self.params = out[0]
    
    def predict(self, X):
        device = X.device
        X = self.transform(X)
        X = X.to(self.compute_device)
        prediction = X @ self.params
        prediction = prediction.to(device)
        return prediction
    
    def __call__(self, X):
        return self.predict(X)

def model_registry(name, config, mean=None, std=None, hidden_layer=32):
    """Model registry for both max call and portfolio problems"""
    if name == "DNN" or name == "NN":
        return DeepNeuralNet(n_features=config.n_assets, hidden_layer=hidden_layer)
    elif name == "Linear":
        return LinearModel(n_features=config.n_assets)
    elif name == "Poly":
        if hasattr(config, 'n_calls'):  # Portfolio case
            return PolynomialModel(n_features=config.n_assets, mean=mean, std=std)
        else:  # Max call case
            return PolynomialNet(n_features=config.n_assets, mean=mean, std=std)
    else:
        raise ValueError(f"Unknown model name: {name}")

def run_max_call_experiment():
    """Run max call option experiment"""
    model_names = ["Linear", "Poly", "NN"]
    results = []
    train_seed = 0
    eval_seed = 100
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    eval_samples = int(1e6)
    importance_sampling = True
    alpha = 0.99
    sampling_alpha = 0.99
    
    MC = MaxCallConfig()
    DS = DataSampler(config=MC, device=device, dtype=dtype, seed=train_seed)
    X, _ = DS.sampleX(importance_sampling=importance_sampling, alpha=sampling_alpha, n_samples=int(1e5))
    Y = DS.sampleY(initial_value=X)
    TR = Transform(mean=torch.mean(Y), std=torch.std(Y))

    # training loop                    
    for model_name in model_names:
        print(f"Training {model_name} for max call...")
        DS.set_seed(train_seed)
        if model_name in ["Linear", "Poly"]:            
            batch_size = int(1e5)
            compute_device = torch.device("cpu")
            X , _ = DS.sampleX(importance_sampling=False, alpha=sampling_alpha, n_samples=batch_size)            
            Y = DS.sampleY(initial_value=X)            
            poly = True if model_name == "Poly" else False
            model = Regression(poly=poly, tol=1e-2, compute_device=compute_device)
            model.fit(X, Y)
        elif model_name == "NN":
            batch_size = int(2**19)
            gradient_steps = int(1e3)
            learning_rate = 1e-3
            model = DeepNeuralNet(n_features=MC.n_assets, hidden_layer=128).to(device)
            model.train()
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=250, min_lr=1e-5)
            
            for epoch in range(gradient_steps):
                X, weights = DS.sampleX(importance_sampling=importance_sampling, alpha=sampling_alpha, n_samples=batch_size)
                Y = DS.sampleY(initial_value=X)
                outputs = model(X).flatten()
                loss = criterion(outputs, Y)
                optimizer.zero_grad()  
                loss.backward()
                optimizer.step()
                scheduler.step(loss)
                
                if epoch % 500 == 0:
                    print(f"epoch {epoch}, loss: {loss.item():.4f}, lr: {optimizer.param_groups[0]['lr']:.6f}")
                
        print(f"evaluating {model_name}")
        if hasattr(model, 'eval'):
            model.eval()
        DS.set_seed(eval_seed)
        X, sampling_weights = DS.sampleX(importance_sampling=importance_sampling, alpha=sampling_alpha, n_samples=eval_samples)
        outputs = model(X)
        sampling_weights = sampling_weights/len(sampling_weights)
        Y = DS.sampleY(initial_value=X)
        Z = DS.sampleY(initial_value=X)

        out = estimates(outputs, Y, Z, sampling_weights=sampling_weights, alpha=alpha)
        results.append({
            "model": model_name,
            "es": out[0].item(),
            "diff_nu/es": out[1].item(),
            "diff_tail/es": out[2].item(),
            "diff_nu/true_f": out[3].item(),
            "diff_tail/true_f_tail": out[4].item()
        })
        
    return results

def run_portfolio_experiment():
    """Run portfolio of call and put options experiment"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = int(2**18)
    learning_rate = 1e-3
    gradient_steps = int(2e3)
    train_seed = 0 
    importance_sampling = False
    eval_seed = 100 
    n_eval_samples = int(1e6)
    alpha = 0.99

    PC = PortfolioConfig()
    results = []

    # Get mean and std for normalization
    DS = PortfolioDataSampler(config=PC, device=device, dtype=torch.float32, n_samples=int(1e6), seed=train_seed)
    X, weights, Z = DS.sampleX(importance_sampling=importance_sampling)
    X_mean = X.mean(dim=0, keepdim=True)
    X_std = X.std(dim=0, keepdim=True)          
    print(f"\nFeature normalization:")
    print(f"X mean shape: {X_mean.shape}, X std shape: {X_std.shape}") 

    models = ["Linear", "Poly", "DNN"]
    for model_name in models:
        print(f'\n-------------------------------\n')
        print(f"Training model: {model_name}")
        
        DS = PortfolioDataSampler(config=PC, device=device, dtype=torch.float32, n_samples=batch_size, seed=train_seed)
        model = model_registry(name=model_name, config=PC, mean=X_mean, std=X_std, hidden_layer=128).to(device)
        model.train()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=100, min_lr=1e-6)
        
        losses = []
        for epoch in range(gradient_steps):
            X, weights, Z = DS.sampleX(importance_sampling=importance_sampling)
            Y = DS.sampleY(initial_value=X)
            optimizer.zero_grad()
            outputs = model(X)
            mse_loss = criterion(outputs, Y)
            
            if model_name == "Poly":
                lambda_reg = 10.0
                l1_penalty = lambda_reg * model.linear.weight.abs().sum()
                loss = mse_loss + l1_penalty    
            else:
                loss = mse_loss
                
            loss.backward()
            optimizer.step()
            scheduler.step(loss)
            losses.append(loss.item())
            
            if (epoch+1) % 100 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1}/{gradient_steps}], Loss: {loss.item():.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
                if model_name == "Poly":
                    print(f'l1/mse%: {100*l1_penalty/mse_loss:.2f}')

        # Evaluation
        DS = PortfolioDataSampler(config=PC, device=device, dtype=torch.float32, n_samples=n_eval_samples, seed=eval_seed)
        X, sampling_weights, Z = DS.sampleX(importance_sampling=importance_sampling)
        outputs = model(X).flatten()
        sampling_weights = sampling_weights.flatten() / sampling_weights.sum()
        
        Y = DS.sampleY(initial_value=X).flatten()
        Z_eval = DS.sampleY(initial_value=X).flatten()
        
        es, err1, err2, _, _ = estimates(predictions=outputs, y=Y, z=Z_eval, alpha=alpha, sampling_weights=sampling_weights)
        print(f"expected shortfall estimate: {es.item():.4f}")
        print(f"first error estimate: {err1.item():.4f}")
        print(f"second error estimate: {err2.item():.4f}")
        
        results.append({
            "model": model_name,
            "expected_shortfall": es.item(),
            "norm": err1.item(),
            "tail_norm": err2.item()
        })
    
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run experiments')
    parser.add_argument('--experiment', choices=['max_call', 'portfolio', 'both'], 
                       default='max_call', help='Which experiment to run')
    args = parser.parse_args()
    
    if args.experiment == 'max_call' or args.experiment == 'both':
        print("Running Max Call Option Experiment...")
        max_call_results = run_max_call_experiment()
        print("\nMax Call Results:")
        for result in max_call_results:
            print(result)
    
    if args.experiment == 'portfolio' or args.experiment == 'both':
        print("\nRunning Portfolio Option Experiment...")
        portfolio_results = run_portfolio_experiment()
        print("\nPortfolio Results:")
        for result in portfolio_results:
            print(result)

# Original max call experiment code (keeping for backward compatibility)
if False:  # Set to True to run original max call code
    # Original code has been refactored into structured functions above
    pass