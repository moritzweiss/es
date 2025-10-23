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
from helper_functions import norm_estimate, compute_estimates, write_to_latex_table, DeepNeuralNet, Regression

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
        sampling_weights = torch.ones(n_samples, device=self.device, dtype=self.dtype)
        return X, sampling_weights
    
    @torch.no_grad()
    def sampleY(self, initial_value):
        '''
        returns Y given X
        '''
        n_samples = initial_value.shape[0]
        # we include the first step 
        max_steps = self.omega - self.x  - 1 
        k, K = initial_value[:, 0], initial_value[:, 1]
        k, K, m = self.mortality_model.simulate_steps(start_k=k, start_K=K, t=max_steps, n_samples=n_samples, age=self.x)
        r, x, y = self.g2pp.simulate_path(n_steps=max_steps, x_start=initial_value[:, 2], y_start=initial_value[:, 3], t_start=self.t, dt=1)
        Y = r+m  
        Y = torch.cumsum(Y, dim=1)
        Y = torch.sum(torch.exp(-Y), dim=1)
        return Y


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
    # eval_samples = int(1e6)
    # eval_samples = int(2e6)
    eval_samples = int(5e6)

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
                # this is just for debugging
                # gradient_steps = 10
            else:
                raise ValueError("unknown experiment type")
            batch_size = int(2**19)
            # batch_size = int(1e5)
            learning_rate = 1e-2
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
            plt.ylim(0, 50)
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


        out, to_latex = compute_estimates(DS=DS, model=model, eval_seed=eval_seed, eval_samples=eval_samples, alpha=alpha, sampling_alpha=sampling_alpha, importance_sampling=importance_sampling)

        results.append(out)
    
    # save to latex table 
    df = pd.DataFrame(results)
    df.index = model_names
    df.rename(columns=to_latex, inplace=True, errors="raise")
    write_to_latex_table(df, experiment_type=experiment_type, importance_sampling=importance_sampling, sampling_alpha=sampling_alpha)