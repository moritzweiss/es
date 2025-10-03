import torch
from typing import Optional, Tuple 

@torch.no_grad()
def expected_shortfall(
    losses: torch.Tensor,
    alpha: float = 0.95,
    sample_weights: Optional[torch.Tensor] = None, 
    normalization = False,
    is_decreasing = False ) -> Tuple[torch.tensor, int]:
    """
    This implements the estimator: ES_hat = (1/(1-alpha)) [ sum_{i=1}^{j-1} w_(i) L_(i) + ( 1 - (sum_{i=1}^{j-1} w_(i)) / (1-alpha) ) * L_(j) ]
    where L_(1) ≥ ... ≥ L_(N) are order stats and j is the first index whose cumulative weight exceeds (1-alpha). 

    Args:
        losses: Tensor of losses (higher = worse). Should be one dimensional. 
        alpha: Confidence level in (0,1). ES averages the worst (1-alpha) tail.
        sample_weights: Optional non-negative weights with the same shape as `losses` (broadcastable). 
        Typical use: importance-sampling weights.
        If None, uses equal weights 1/N.
    """
    # should weights always be normalized ? 
    # shuld is use (1-alpha)*sum(weights) as threshold ? 

    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0,1).")
    assert losses.dim() == 1
    N = losses.numel()
    assert N > 0
    assert losses.shape == (N,)

    if sample_weights is None:
        sample_weights = torch.full_like(losses, 1.0 / N)
    else:
        assert sample_weights.shape == losses.shape
        assert torch.all(sample_weights>=0), "weights must be positive"        
        if normalization:
            sample_weights = sample_weights / sample_weights.sum()
        else:
            assert torch.isclose(sample_weights.sum(), torch.tensor(1.0, dtype=sample_weights.dtype, device=sample_weights.device), rtol=1e-2), "weights must not sum to one"
    if not is_decreasing:
        losses, idx = torch.sort(losses, dim=-1, descending=True)
        sample_weights = torch.gather(input=sample_weights,dim=0,index=idx)

    cumulative_weights = torch.cumsum(sample_weights, dim=0)  
    threshold = (1.0 - alpha)
    j = torch.searchsorted(cumulative_weights, torch.tensor(threshold, dtype=cumulative_weights.dtype, device=cumulative_weights.device), right=True).clamp_max(N - 1)  
    j = int(j.item())

    if j==0:
        return losses[0], j 

    coeff = 1.0 - cumulative_weights[j-1]/threshold
    es = (sample_weights[:j]*losses[:j]).sum()/threshold + coeff*losses[j]

    return es, j 


if __name__ == "__main__":
    torch.manual_seed(0)
    normalization = True  
    # a) standard 
    losses = torch.tensor([10., 8., 8., 8., 5., 4., 3., 2., 1., 0.])
    alpha = 0.7
    es = losses[:3].sum()/3
    out, j = expected_shortfall(losses=losses, alpha=alpha, normalization=normalization)
    assert torch.isclose(out, es, rtol=1e-5)
    assert j == 3 
    # b) an example with ties 
    alpha = 0.75
    j = 2 # 0 based indexing shifts by -1
    es = losses[:j].sum()/(10*(1-alpha)) + (1 - j/(10*(1-alpha))) * losses[j]
    es_out, j_out = expected_shortfall(losses=losses, alpha=alpha, normalization=normalization)
    assert torch.isclose(es_out, es, rtol=1e-5)    
    assert j_out == j
    # c) example with weights, but not normalized  
    weights = torch.tensor([10., 9., 8., 7., 6., 5., 4., 3., 2., 1.])
    alpha = 0.7
    es_w, j_w = expected_shortfall(
        losses=losses,
        alpha=alpha,
        sample_weights=weights,
        normalization=False,        
        is_decreasing=True         
    )
    assert torch.isclose(es_w, torch.tensor(10, dtype=torch.float32), rtol=1e-6, atol=0.0)
    assert abs(j_w-0) < 1e-1
    # d) example with normalized weights 
    weights = torch.tensor([0.20, 0.18, 0.15, 0.12, 0.10, 0.08, 0.07, 0.05, 0.03, 0.02])
    assert torch.isclose(weights.sum(), torch.tensor(1.0))  
    alpha = 0.7
    es_w, j_w = expected_shortfall(losses=losses,alpha=alpha, sample_weights=weights, normalization=False, is_decreasing=True)
    # j=2 therefore j=1 in python. the below only concerns the first value. 
    first_term = (losses[0]*weights[0])/(1-alpha)
    second_term = (1-weights[0]/(1-alpha))*losses[1]
    es_expected=first_term+second_term
    assert j_w == 1
    assert torch.isclose(es_w, es_expected, rtol=1e-6, atol=0.0)
    # everything works as expected !  