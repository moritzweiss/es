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
from expected_shortfall import expected_shortfall
from helper_functions import norm_estimate, compute_estimates, write_to_latex_table, DeepNeuralNet, Regression



@dataclass
class PortfolioConfig:
    name: str = "portfolio"
    interest_rate: float = 0.01
    final_time: float = 1/3
    intermediate_time: float = 1/52
    initial_value: float = 100.0
    corr: float = 0.30
    # no correlation to test IS sampling better 
    # corr: float = 0.0
    n_assets : int = 20
    drift: List[float] = field(default_factory=lambda: [(2.5 + ((i % 10 + 1) / 2)) / 100 for i in range(20)])
    vola: List[float] = field(default_factory=lambda: [(14 + (i % 10 + 1)) / 100 for i in range(20)])
    strike: float = 100.0
    n_calls: int = 10
    n_puts: int = 10

@dataclass
class MaxCallConfig:
    name: str = "max_call"
    interest_rate: float = 0.00          
    final_time: float = 1/3             
    intermediate_time: float = 1/52
    initial_value: float = 10.0       
    corr: float = 0.30                  
    n_assets: int = 100                  
    vola: List[float] = field(default_factory=lambda: [( 10 + (i+1)/2 ) / 100 for i in range(100)])
    drift: List[float] = field(default_factory=lambda: [0.0]*100)  
    strike: float = 16.4                

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
        # Simulate asset values at intermediate time
        X = self.GBM.simulate_increment(n_samples=n_samples, initial_value=self.initial_value, time_increment=self.intermediate_time, Z=Z)
        return X, weights.flatten()

    @torch.no_grad()
    def shift_vector(self, alpha):    
        if self.name == "portfolio":     
             assert self.n_calls + self.n_puts == self.n_assets
             v = torch.tensor([1.0]*self.n_calls + [-1.0]*self.n_puts, device=self.device, dtype=self.dtype)
        elif self.name == "max_call":
             v = torch.tensor([1.0]*self.n_assets, device=self.device, dtype=self.dtype)
        else:
             raise ValueError("unknown config name")
        quantile = torch.distributions.Normal(0,1).icdf(torch.tensor(alpha, device=self.device, dtype=self.dtype))            
        v = self.A.T @ v
        return (quantile*v) / torch.linalg.vector_norm(v) 

    @torch.no_grad()             
    def sampleY(self, initial_value):
        if self.name == "portfolio":
            n_samples = initial_value.shape[0]
            dt = self.final_time - self.intermediate_time
            # Sample under risk-neutral measure with interest rate as drift
            Y = self.GBM.simulate_increment(n_samples=n_samples, initial_value=initial_value, time_increment=dt, drift=self.interest_rate)
            assert Y.shape == (n_samples, self.n_assets)
            calls = torch.relu(Y[:,:self.n_calls] - self.strike).sum(dim=1)
            puts = torch.relu(self.strike - Y[:,self.n_calls:]).sum(dim=1)
            # Discount back to present value
            discount_factor = torch.exp(-self.interest_rate * dt)
            return ((calls + puts) * discount_factor).flatten()
        elif self.name == "max_call":
            n_samples = initial_value.shape[0]
            dt = self.final_time - self.intermediate_time
            # For max call, use the default drift from config (typically 0 for this problem)
            Y = self.GBM.simulate_increment(n_samples=n_samples, initial_value=initial_value, time_increment=dt)
            assert Y.shape == (n_samples, self.n_assets)
            return torch.relu(torch.max(Y, dim=1)[0] - self.strike).flatten()
        else:
            raise ValueError("unknown config name")
    


if __name__ == "__main__":
    import random
    
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    
    experiment_type = "max_call"  
    # experiment_type = "portfolio"
    
    if experiment_type == "portfolio":
        config = PortfolioConfig()
    else:
        config = MaxCallConfig()
    
    model_names = ["Linear", "Poly", "NN"]
    # model_names = ["Linear", "Poly"]
    results = []
    train_seed = 0
    eval_seed = 100
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    eval_samples = int(2e6)
    importance_sampling = True

    alpha = 0.99
    # lower sampling alpha works better for portfolio 0.8 works fine for portfolio 
    sampling_alpha = 0.9
    # for the max call sampling alpha =0.9 or 0.99 works well
    # sampling_alpha = 0.99

    name = f"with_is_{sampling_alpha}" if importance_sampling else "no_is"
    name = f"{experiment_type}_{name}"
    
    print(f"Running experiment: {config.name}")
    DS = DataSampler(config=config, device=device, dtype=dtype, seed=train_seed)
    

    # training loop                    
    for model_name in model_names:
        print(f"Training {model_name} for {config.name}...")
        DS.set_seed(train_seed)
        if model_name in ["Linear", "Poly"]:
            batch_size = int(1e5) 
            compute_device = torch.device("cpu")
            X , _ = DS.sampleX(importance_sampling=importance_sampling, alpha=sampling_alpha, n_samples=batch_size)            
            Y = DS.sampleY(initial_value=X)            
            poly = True if model_name == "Poly" else False
            model = Regression(poly=poly, tol=1e-2, compute_device=compute_device)
            model.fit(X, Y)
        elif model_name == "NN":
            # Set torch seed for reproducible model initialization
            torch.manual_seed(train_seed)
            if torch.cuda.is_available():   
                torch.cuda.manual_seed(train_seed)
                torch.cuda.manual_seed_all(train_seed)
                
            if experiment_type == "max_call":
                gradient_steps = int(1e3)
                # gradient_steps = 2
            elif experiment_type == "portfolio":
                # portfolio of puts and calls
                gradient_steps = int(2e3)
            else:
                raise ValueError("unknown experiment type")
            batch_size = int(2**19)
            learning_rate = 1e-3
            # smaller network for portfolio
            model = DeepNeuralNet(n_features=config.n_assets, experiment_type=experiment_type).to(device)
            model.train()
            criterion = nn.MSELoss()
            optimizer= optim.Adam(model.parameters(), lr=learning_rate)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=250, min_lr=1e-5)
            losses = []
            lrs = []
            for epoch in range(gradient_steps):
                X, weights = DS.sampleX(importance_sampling=importance_sampling, alpha=sampling_alpha, n_samples=batch_size)
                Y = DS.sampleY(initial_value=X)
                predictions = model(X)
                predictions = predictions.flatten()
                loss = criterion(predictions, Y)
                optimizer.zero_grad()  
                loss.backward()
                optimizer.step()
                scheduler.step(loss)
                losses.append(loss.item())
                lrs.append(optimizer.param_groups[0]['lr'])
                if epoch % 500 == 0:
                    print(f"epoch {epoch}, loss: {loss.item()}, lr: {optimizer.param_groups[0]['lr']}")

            plt.figure()
            plt.plot(losses)
            plt.xlabel("gradient step")
            plt.ylabel("mean squared error loss")
            plt.xlim(0, gradient_steps) 
            plt.title(f"{experiment_type}_batch size={batch_size}")
            plt.savefig(f"plots/{name}_{model_name}.png")

            plt.figure()
            plt.plot(lrs)
            plt.xlabel("gradient step")
            plt.ylabel("learning rate")
            plt.xlim(0, gradient_steps)
            plt.title(f"{experiment_type}_batch size={batch_size}")
            plt.savefig(f"plots/{name}_{model_name}_lr.png")
            
        else:
            raise ValueError("unknown model name")
                
        print(f"evaluating {model_name}")        
        model.eval() if model_name == "NN" else None


        # evaluation
        out, to_latex = compute_estimates(DS=DS, model=model, eval_seed=eval_seed, eval_samples=eval_samples, alpha=alpha, sampling_alpha=sampling_alpha, importance_sampling=importance_sampling, on_cpu=True)

        results.append(out)
    
    # latex 
    df = pd.DataFrame(results)
    df.index = model_names
    df.rename(columns=to_latex, errors="raise", inplace=True)    
    # df.to_csv(f"results/{name}.csv", float_format="%.2f")
    write_to_latex_table(df=df, experiment_type=experiment_type, importance_sampling=importance_sampling, sampling_alpha=sampling_alpha)

    

    