# testing importance sampling weights here 
import sys
import os
import torch
top_level_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if top_level_dir not in sys.path:
    sys.path.insert(0, top_level_dir)
import portfolio_call_and_put_options as pco
from portfolio_call_and_put_options import is_weights

if __name__ == "__main__":
    n_samples = int(1e6)   
    n_assets = 20 
    # 
    A = torch.linalg.cholesky(0.5*torch.ones(n_assets, n_assets) + 0.5*torch.eye(n_assets))
    quantile = torch.distributions.Normal(0,1).icdf(torch.tensor(0.95))            
    v = A.T @ torch.tensor([1.0]*10+[-1.0]*10)
    v = quantile*v / torch.norm(v) 
    # why are weights not close to one with this weight choice? are they too large? because tails are too heavy! 
    # v = torch.tensor([-1]*10 + [1]*10, dtype=torch.float32) 
    # 
    Z = torch.randn(n_samples, n_assets, generator=torch.Generator().manual_seed(0))
    Z += v 
    w = is_weights(Z, v)
    assert w.shape == (n_samples,)
    w = w/n_samples
    print(w.sum().item())  

    