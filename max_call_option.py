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
from helper_functions import norm_estimate

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
    
class DeepNeuralNet(nn.Module):
    # network architecture has an effect on the results 
    # deeper not necessarily better
    # estimate in the tail are not better than baselines with importance sampling 
    # small network is close to linear regression but worse than polynomial regression
    # larger network performs worse for the tail estimates 
    def __init__(self, n_features=20, experiment_type="max_call"):
        super().__init__()

        if experiment_type == "max_call":
            self.net = nn.Sequential(
                nn.Linear(n_features, 256),
                nn.BatchNorm1d(256),
                nn.GELU(),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.GELU(),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.GELU(),
                nn.Linear(128, 1)
            )
        
        elif experiment_type == "portfolio":
            hidden_layer = 128
            self.net = nn.Sequential(
            nn.Linear(n_features, hidden_layer),
            nn.BatchNorm1d(hidden_layer),
            nn.GELU(),
            # nn.Tanh(),
            nn.Linear(hidden_layer, hidden_layer),
            nn.BatchNorm1d(hidden_layer),
            nn.GELU(),
            # nn.Tanh(),
            nn.Linear(hidden_layer, 1)
            )

    def forward(self, x):
        return self.net(x).flatten()


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
        # the transformation is faster on GPU. but need more memory for this. we can check other adas. 
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

def evaluation_samples(DS, model, eval_seed, eval_samples, alpha, importance_sampling, on_cpu=False):
    samples_per_batch = int(1e6)
    n_batches = int(eval_samples/samples_per_batch)
    DS.set_seed(eval_seed)
    all_X = []
    all_Y = []
    all_Z = []
    all_outputs = []
    all_sampling_weights = []
    for _ in range(n_batches):
        X, sampling_weights = DS.sampleX(importance_sampling=importance_sampling, alpha=alpha, n_samples=samples_per_batch)
        outputs = model(X)
        Y = DS.sampleY(initial_value=X)
        Z = DS.sampleY(initial_value=X)        
        if on_cpu:
            X, Y = X.detach().cpu(), Y.detach().cpu()
            Z, sampling_weights = Z.detach().cpu(), sampling_weights.detach().cpu()
            outputs = outputs.detach().cpu()
        all_X.append(X)
        all_Y.append(Y)
        all_Z.append(Z)
        all_outputs.append(outputs)
        all_sampling_weights.append(sampling_weights)
    all_X = torch.cat(all_X, dim=0)
    all_Y = torch.cat(all_Y, dim=0)
    all_Z = torch.cat(all_Z, dim=0)
    all_sampling_weights = torch.cat(all_sampling_weights, dim=0)
    # normalize sampling weights 
    all_sampling_weights = all_sampling_weights / all_sampling_weights.sum()
    all_outputs = torch.cat(all_outputs, dim=0)
    # Sort all arrays in descending order of predictions
    sort_idx = torch.argsort(all_outputs, descending=True)
    all_X = all_X[sort_idx]
    all_Y = all_Y[sort_idx]
    all_Z = all_Z[sort_idx]
    all_outputs = all_outputs[sort_idx]
    all_sampling_weights = all_sampling_weights[sort_idx]
    return all_X, all_Y, all_Z, all_outputs, all_sampling_weights

if __name__ == "__main__":
    import random
    
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    
    # experiment_type = "max_call"  
    experiment_type = "portfolio"
    
    if experiment_type == "portfolio":
        config = PortfolioConfig()
    else:
        config = MaxCallConfig()
    
    model_names = ["Linear", "Poly", "NN"]
    results = []
    train_seed = 0
    eval_seed = 100
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    eval_samples = int(2e6)
    importance_sampling = True

    alpha = 0.99
    sampling_alpha = 0.8

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


        # eval with importance sampling for norm tail estimates and for expected shortfall
        # can choose alpha or sampling_alpha to collect the samples 
        X, Y, Z, predictions, sampling_weights = evaluation_samples(DS=DS, model=model, eval_seed=eval_seed, eval_samples=eval_samples, alpha=alpha, importance_sampling=True, on_cpu=True)
        es, j = expected_shortfall(losses=predictions, sample_weights=sampling_weights, alpha=alpha, normalize=False, make_decreasing=False)
        diff_nu_tail, true_f_tail = norm_estimate(predictions, Y, Z, j, alpha, sampling_weights, tail_estimate=True)

        # clear memory 
        X = X.detach()
        Y = Y.detach()
        Z = Z.detach()
        predictions = predictions.detach()
        sampling_weights = sampling_weights.detach()
        del X, Y, Z, sampling_weights
        torch.cuda.empty_cache()
        print("alloc:", torch.cuda.memory_allocated()/1e6, "MB")
        print("reserved:", torch.cuda.memory_reserved()/1e6, "MB")

          # eval without importance sampling for regular norm estimate (not in the tail)
        X, Y, Z, predictions, sampling_weights = evaluation_samples(DS=DS, model=model, eval_seed=eval_seed, eval_samples=eval_samples, alpha=alpha, importance_sampling=False, on_cpu=True) 
        diff_nu, true_f = norm_estimate(predictions, Y, Z, None, alpha, sampling_weights, tail_estimate=False)

        results.append({
                "model": model_name,
                "es": es.item(),
                "diff_nu/es": (diff_nu/es).item(),
                "diff_tail/es": (diff_nu_tail/es).item(),
                "diff_nu/true_f": (diff_nu/true_f).item(),
                "diff_tail/true_f_tail": (diff_nu_tail/true_f_tail).item()})
    
    # csv 
    df = pd.DataFrame(results)
    df = df.set_index("model")
    df.to_csv(f"results/{name}.csv", float_format="%.2f")

    # latex 
    to_latex = {"es": r"$\mathrm{ES}_{\alpha}(\hat{f}(X))$",
                "diff_nu/es": r"$\frac{\|\hat f - \bar f\|_{L^2(\nu)}}{\mathrm{ES}_{\alpha}(\hat{f}(X))}$",
                "diff_tail/es": r"$\frac{\|\hat f - \bar f\|_{L^2(\hat \nu_\alpha)}}{\mathrm{ES}_{\alpha}(\hat{f}(X))}$",
                "diff_nu/true_f": r"$\frac{\|\hat f - \bar f\|_{L^2(\nu)}}{\|\bar f\|_{L^2(\nu)}}$",
                "diff_tail/true_f_tail": r"$\frac{\|\hat f - \bar f\|_{L^2(\hat \nu_\alpha)}}{\|\bar f\|_{L^2(\hat \nu_\alpha)}}$"} 
    df = df.rename(columns=to_latex)
    colfmt = "l" + "c" * len(df.columns) 
    
    # removing under scors 
    label = "portfolio" if experiment_type == "portfolio" else "max call"   
    if importance_sampling:
        caption = f"{label} with importance sampling."
    else:
        caption = f"{label} without importance sampling."
    latex_str = df.to_latex(float_format="%.4f", escape=False, index_names=False, column_format=colfmt)            
    # add centering to the table 
    latex_str = ("\\begin{table}[ht]\n"
    "\\centering\n" +
    latex_str +
    f"\\caption{{{caption}}}\n"
    "\\end{table}\n")
    file_name = f"results/{name}.tex" 
    with open(file_name, "w") as f:
        f.write(latex_str)

    #  html 
    table_html = df.to_html(float_format="%.4f", escape=False, index_names=False)
    html = f"""<!doctype html>
    <html lang="en">
    <head>
    <meta charset="utf-8">
    <title>{name}</title>
    <style>
        body{{font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; padding: 1rem 2rem;}}
        table{{border-collapse: collapse; width: 100%;}}
        th, td{{border: 1px solid #ddd; padding: 0.5rem; text-align: center;}}
        th{{background: #f6f6f6;}}
        caption{{caption-side: bottom; padding-top: .5rem; font-style: italic;}}
    </style>
    <script>
        window.MathJax = {{
        tex: {{ inlineMath: [['$', '$'], ['\\\\(', '\\\\)']] }},
        svg: {{ fontCache: 'global' }}
        }};
    </script>
    <script id="MathJax-script" async
            src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    </head>
    <body>
    <h2>{name}</h2>
    {table_html.replace("<table", f"<table><caption>{caption}</caption>", 1)}
    </body>
    </html>
    """
    file_name = f"results/{name}.html" if importance_sampling else f"results/{name}.html"
    with open(file_name, "w", encoding="utf-8") as f:
        f.write(html)
    
    # save to markdown 
    md_table = df.to_markdown(floatfmt=".4f", index=True)  
    file_name = f"results/{name}.md" if importance_sampling else f"results/{name}.md"
    with open(file_name, "w", encoding="utf-8") as f:
        f.write(md_table)