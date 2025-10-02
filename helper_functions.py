import torch
from typing import Callable
from expected_shortfall import expected_shortfall   

@torch.no_grad()
def norm_estimate(prediction,y,z,j, alpha, sampling_weights=None, tail_estimate=False):
    '''
    computes F^\nu_N = y*z + prediction*(prediction - y - z). is an estimate of |hat{f} - f|^2
    computes C^nu_N = YZ. estimate of |f|^2
    '''
    device = prediction.device
    if sampling_weights == None:
        N = prediction.shape[0]
        sampling_weights = torch.ones(N, device=device)/N
    mc_values = (y * z + prediction * (prediction - y - z))
    if tail_estimate:
        first_term = (mc_values[:j]*sampling_weights[:j]).sum()/(1-alpha)
        second_term = (1 - sampling_weights[:j].sum()/(1-alpha))*mc_values[j]
        sq_norm = first_term + second_term
    else:
        sq_norm = (mc_values*sampling_weights).sum()    
    # 
    norm = torch.sign(sq_norm)*torch.sqrt(torch.sign(sq_norm)*sq_norm)
    x = (y*z*sampling_weights).sum()
    x = torch.sqrt(torch.sign(x)*x)
    return norm, x

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
    es, j = expected_shortfall(losses=predictions, alpha=alpha, sample_weights=sampling_weights, normalization=False, is_decreasing=True)
    norm_diff_nu, norm_true_nu = norm_estimate(predictions, y, z, j, alpha, sampling_weights, tail_estimate=False)
    norm_diff_tail, norm_true_tail = norm_estimate(predictions, y, z, j, alpha, sampling_weights, tail_estimate=True)
    # TODO: confidence bounds 
    return es, norm_diff_nu/es, norm_diff_tail/es, norm_diff_nu/norm_true_nu, norm_diff_tail/norm_true_tail 