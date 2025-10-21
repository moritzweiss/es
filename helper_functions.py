import torch
from typing import Callable
from expected_shortfall import expected_shortfall   

@torch.no_grad()
def norm_estimate(prediction,y,z,j, alpha, sampling_weights=None, tail_estimate=False):
    '''
    # input is assumed in descending order of predictions
    computes F^\nu_N = y*z + prediction*(prediction - y - z). is an estimate of |hat{f} - f|^2
    computes C^nu_N = YZ. estimate of |f|^2
    '''
    device = prediction.device
    mc_values = (y * z + prediction * (prediction - y - z))
    if sampling_weights is None:
        N = prediction.shape[0]
        sampling_weights = torch.ones(N, device=device)/N        
    if tail_estimate:
        # Fix: Store original sum before modifying weights
        original_sum_j = sampling_weights[:j].sum()
        sampling_weights = sampling_weights.clone()  # Work on a copy
        sampling_weights[:j] = sampling_weights[:j]/(1 - alpha)
        sampling_weights[j]  = (1 - original_sum_j/(1 - alpha))
        sampling_weights[j+1:] = 0.0
    sq_norm = (mc_values*sampling_weights).sum()    
    sq_variance = (sampling_weights*(mc_values - sq_norm)**2).sum()
    # 
    norm = torch.sign(sq_norm)*torch.sqrt(torch.sign(sq_norm)*sq_norm)
    q_a = torch.distributions.Normal(0,1).icdf(torch.tensor(0.95))
    # compute the error bound over the tail of the distribution
    # Fix: Avoid division by zero when j == 0
    # if j > 0:
    #     error_bound = norm + q_a*torch.sqrt(sq_variance/j)
    # else:
    #     raise ValueError("Index j must be greater than 0 to compute error bound.")
    # norm 
    if tail_estimate:
        if j > 0:
            error_bound = norm + q_a*torch.sqrt(sq_variance/j)
        else:
            raise ValueError("Index j must be greater than 0 to compute error bound.")
    else:
        error_bound = norm + q_a*torch.sqrt(sq_variance/prediction.shape[0])

    x = (y*z*sampling_weights).sum()
    x = torch.sqrt(torch.sign(x)*x)
    assert error_bound >= norm - 1e-6, "Error bound should be larger than norm estimate"
    return norm, x, error_bound

@torch.no_grad()
def estimates(predictions: torch.Tensor, y: torch.Tensor, z: torch.Tensor, sampling_weights: torch.Tensor = None, alpha: float = 0.95) -> torch.Tensor:
    assert predictions.ndim == 1 
    assert y.ndim == 1
    assert z.ndim == 1
    idx = torch.argsort(predictions, descending=True)
    y = y[idx]
    z = z[idx]
    predictions = predictions[idx]
    if sampling_weights is not None:
        sampling_weights = sampling_weights[idx]
    es, j = expected_shortfall(losses=predictions, alpha=alpha, sample_weights=sampling_weights, normalize=False, make_decreasing=True)
    norm_diff_nu, norm_true_nu, _ = norm_estimate(predictions, y, z, j, alpha, sampling_weights, tail_estimate=False)
    norm_diff_tail, norm_true_tail, error_bound_tail = norm_estimate(predictions, y, z, j, alpha, sampling_weights, tail_estimate=True)
    # TODO: confidence bounds 
    return es, norm_diff_nu/es, norm_diff_tail/es, norm_diff_nu/norm_true_nu, norm_diff_tail/norm_true_tail 