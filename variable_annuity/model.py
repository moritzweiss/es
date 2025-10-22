from mortality import MortalityConfig, MortalityModel
from g2pp import G2PPModel, InterestRateConfig
import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import random
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from expected_shortfall import expected_shortfall
from helper_functions import norm_estimate

class DataSampler():

    def __init__(self, mortality_config, rates_config, device, dtype, seed):                
        self.mortality_model = MortalityModel(config=mortality_config, device=device, dtype=dtype, seed=seed)
        self.g2pp = G2PPModel(config=rates_config, device=device, dtype=dtype, seed=seed)
        self.t = 1 
        self.x = 65
        self.omega = 90
        self.device = device
        self.dtype = dtype
 
    def set_seed(self, seed):
        self.mortality_model.set_seed(seed)
    
    @torch.no_grad()        
    def sampleX(self, n_samples, importance_sampling=False, alpha=None):
        '''
        returns X and sampling weights 
        '''
        k, K, m = self.mortality_model.simulate(start_k=self.mortality_model.k_start, start_K=self.mortality_model.K_start, t=self.t, n_samples=n_samples, age=self.x)   
        x_start = torch.zeros(n_samples, device=self.device, dtype=self.dtype)
        y_start = torch.zeros(n_samples, device=self.device, dtype=self.dtype)
        r, x, y = self.g2pp.simulate(dt=self.t, x_start=x_start, y_start=y_start, t_start=0)
        X = torch.stack([k, K, x, y], dim=1)
        # no importance sampling implemented yet
        return X, None
    
    @torch.no_grad()
    def sampleY(self, X):
        '''
        returns Y given X
        '''
        n_samples = X.shape[0]
        # we include the first step 
        max_steps = self.omega - self.x  - 1 
        k, K = X[:, 0], X[:, 1]
        k, K, m = self.mortality_model.simulate_steps(start_k=k, start_K=K, t=max_steps, n_samples=n_samples, age=self.x)
        r, x, y = self.g2pp.simulate_path(n_steps=max_steps, x_start=X[:, 2], y_start=X[:, 3], t_start=self.t, dt=1)
        Y = r+m  
        Y = torch.cumsum(Y, dim=1)
        Y = torch.sum(torch.exp(-Y), dim=1)
        return Y

class DeepNeuralNet(torch.nn.Module):
    def __init__(self, n_features, experiment_type):
        super(DeepNeuralNet, self).__init__()
        if experiment_type == "max_call":
            hidden_layers = [128, 128, 64]
        elif experiment_type == "portfolio":
            hidden_layers = [64, 64]
        elif experiment_type == "variable_annuity":
            # best one [256, 256]
            hidden_layers = [256, 256]
        else:
            raise ValueError("unknown experiment type")
        layers = []
        input_size = n_features
        for hidden_size in hidden_layers:
            layers.append(torch.nn.Linear(input_size, hidden_size))
            layers.append(torch.nn.BatchNorm1d(hidden_size))
            layers.append(torch.nn.GELU())
            input_size = hidden_size
        layers.append(torch.nn.Linear(input_size, 1))
        self.network = torch.nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x).squeeze(-1)


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
    for _ in range(n_batches):
        X, _ = DS.sampleX(importance_sampling=importance_sampling, alpha=alpha, n_samples=samples_per_batch)
        outputs = model(X)
        Y = DS.sampleY(X)
        Z = DS.sampleY(X)
        if on_cpu:
            X, Y, Z, outputs = X.detach().cpu(), Y.detach().cpu(), Z.detach().cpu(), outputs.detach().cpu()
        all_X.append(X)
        all_Y.append(Y)
        all_Z.append(Z)
        all_outputs.append(outputs)
    all_X = torch.cat(all_X, dim=0)
    all_Y = torch.cat(all_Y, dim=0)
    all_Z = torch.cat(all_Z, dim=0)
    # normalize sampling weights 
    all_outputs = torch.cat(all_outputs, dim=0)
    # Sort all arrays in descending order of predictions
    sort_idx = torch.argsort(all_outputs, descending=True)
    all_X = all_X[sort_idx]
    all_Y = all_Y[sort_idx]
    all_Z = all_Z[sort_idx]
    all_outputs = all_outputs[sort_idx]
    # no importance sampling implemented yet
    return all_X, all_Y, all_Z, all_outputs, None

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    
    n_samples = int(1e5)
    MC = MortalityConfig()
    G2PP = InterestRateConfig()
    DS = DataSampler(mortality_config=MC, rates_config=G2PP, device=torch.device("cpu"), dtype=torch.float32, seed=0)

    # experiment_type = "max_call"  
    experiment_type = "variable_annuity"
    config = MortalityConfig()
    
    model_names = ["Linear", "Poly", "NN"]
    # model_names = ["Linear", "Poly"]
    results = []
    train_seed = 0
    eval_seed = 100
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    eval_samples = int(2e6)

    alpha = 0.99
    sampling_alpha = 0.99 
    importance_sampling = False 

    name = f"with_is_{sampling_alpha}" if importance_sampling else "no_is"
    name = f"{experiment_type}_{name}"
    
    print(f"Running experiment: {config.name}")
    DS = DataSampler(mortality_config=config, rates_config=G2PP, device=device, dtype=dtype, seed=train_seed)
    
    # training loop                    
    for model_name in model_names:
        print(f"Training {model_name} for {config.name}...")
        DS.set_seed(train_seed)
        if model_name in ["Linear", "Poly"]:
            batch_size = int(1e5) 
            compute_device = torch.device("cpu")
            X, _ = DS.sampleX(importance_sampling=importance_sampling, alpha=sampling_alpha, n_samples=batch_size)            
            Y = DS.sampleY(X)
            poly = True if model_name == "Poly" else False
            # here the conditioning number can be very smalll 
            model = Regression(poly=poly, tol=1e-5, compute_device=compute_device)
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
            elif experiment_type == "variable_annuity":
                gradient_steps = int(2e3)
            else:
                raise ValueError("unknown experiment type")
            # batch_size = int(2**19)
            batch_size = int(2**18)
            learning_rate = 1e-3
            # smaller network for portfolio
            if experiment_type == "variable_annuity":
                n_features = 4
            else:
                n_features = config.n_assets
            model = DeepNeuralNet(n_features=n_features, experiment_type=experiment_type).to(device)
            model.train()
            criterion = nn.MSELoss()
            optimizer= optim.Adam(model.parameters(), lr=learning_rate)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=250, min_lr=1e-5)
            losses = []
            lrs = []
            for epoch in range(gradient_steps):
                X, weights = DS.sampleX(importance_sampling=importance_sampling, alpha=sampling_alpha, n_samples=batch_size)
                Y = DS.sampleY(X=X)
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
        X, Y, Z, predictions, sampling_weights = evaluation_samples(DS=DS, model=model, eval_seed=eval_seed, eval_samples=eval_samples, alpha=alpha, importance_sampling=False, on_cpu=True)
        es, j = expected_shortfall(losses=predictions, sample_weights=sampling_weights, alpha=alpha, normalize=False, make_decreasing=False)
        diff_nu_tail, true_f_tail, error_bound_tail = norm_estimate(predictions, Y, Z, j, alpha, sampling_weights, tail_estimate=True)

        # clear memory 
        X = X.detach()
        Y = Y.detach()
        Z = Z.detach()
        predictions = predictions.detach()
        sampling_weights = sampling_weights.detach() if sampling_weights is not None else None
        del X, Y, Z, sampling_weights
        torch.cuda.empty_cache()
        print("alloc:", torch.cuda.memory_allocated()/1e6, "MB")
        print("reserved:", torch.cuda.memory_reserved()/1e6, "MB")

        # eval without importance sampling for regular norm estimate (not in the tail)
        X, Y, Z, predictions, sampling_weights = evaluation_samples(DS=DS, model=model, eval_seed=eval_seed, eval_samples=eval_samples, alpha=alpha, importance_sampling=False, on_cpu=True) 
        diff_nu, true_f, error_bound = norm_estimate(predictions, Y, Z, None, alpha, sampling_weights, tail_estimate=False)

        results.append({
        "model": model_name,
        "es": es.item(),
        "diff_nu/es": (diff_nu/es).item(),
        "diff_nu_cb/es": (error_bound/es).item(),
        "diff_tail/es": (diff_nu_tail/es).item(),
        "diff_tail_cb/es": (error_bound_tail/es).item(),
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
                "diff_nu_cb/es": r"$\frac{95\%\text{CB}\|\hat f - \bar f\|_{L^2(\nu)}}{\mathrm{ES}_{\alpha}(\hat{f}(X))}$",
                "diff_nu/true_f": r"$\frac{\|\hat f - \bar f\|_{L^2(\nu)}}{\|\bar f\|_{L^2(\nu)}}$",
                "diff_tail_cb/es": r"$\frac{95\%\text{CB}\|\hat f - \bar f\|_{L^2(\hat \nu_\alpha)}}{\mathrm{ES}_{\alpha}(\hat{f}(X))}$",
                "diff_tail/true_f_tail": r"$\frac{\|\hat f - \bar f\|_{L^2(\hat \nu_\alpha)}}{\|\bar f\|_{L^2(\hat \nu_\alpha)}}$"}     
    df = df.rename(columns=to_latex)
    colfmt = "l" + "c" * len(df.columns) 

    
    # removing under scors 
    label = "portfolio" if experiment_type == "portfolio" else "max call"   
    if importance_sampling:
        caption = f"{label} with importance sampling."
        ref = f'table:{experiment_type}_with_is'
    else:
        caption = f"{label} without importance sampling."
        ref = f'table:{experiment_type}_no_is'
    latex_str = df.to_latex(float_format="%.4f", escape=False, index_names=False, column_format=colfmt)            
    # add centering to the table 
    latex_str = ("\\begin{table}[htbp]\n"
    "\\centering\n" +
    latex_str +
    f"\\caption{{{caption}}}\n"
    f"\\label{{{ref}}}\n"
    "\\end{table}\n")
    file_name = f"results/{name}.tex" 
    with open(file_name, "w") as f:
        f.write(latex_str)