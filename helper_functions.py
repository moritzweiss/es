import torch
from typing import Callable
from expected_shortfall import expected_shortfall   
import torch.nn as nn

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

def compute_estimates(DS, model, eval_samples, eval_seed, alpha, sampling_alpha, importance_sampling, on_cpu=True):
    all_X, all_Y, all_Z, all_outputs, all_sampling_weights = evaluation_samples(DS, model, eval_seed, eval_samples, sampling_alpha, importance_sampling, on_cpu)
    # compute quantities 
    es, j = expected_shortfall(losses=all_outputs, alpha=alpha, sample_weights=all_sampling_weights, normalize=False, make_decreasing=False)
    diff_norm, true_f_norm, confidence_bound = norm_estimate(all_outputs, all_Y, all_Z, j, alpha, all_sampling_weights, tail_estimate=False)
    diff_norm_tail, true_f_norm_tail, confidence_bound_tail = norm_estimate(all_outputs, all_Y, all_Z, j, alpha, all_sampling_weights, tail_estimate=True)

    # note that: alternatively one could not sample in the tail for estimates under nu_X
    # and sample in the tail for estimates under the tail distribution
    # X = X.detach()
    # Y = Y.detach()
    # Z = Z.detach()
    # predictions = predictions.detach()
    # sampling_weights = sampling_weights.detach()
    # del X, Y, Z, sampling_weights
    # torch.cuda.empty_cache()
    # print("alloc:", torch.cuda.memory_allocated()/1e6, "MB")
    # print("reserved:", torch.cuda.memory_reserved()/1e6, "MB")

    results = { 
        'es': es.item(),
        'relative_error_nu/es': (diff_norm/es).item(),
        'confidence_bound_nu/es': (confidence_bound/es).item(),
        'relative_error_tail/es': (diff_norm_tail/es).item(),
        'confidence_bound_tail/es': (confidence_bound_tail/es).item(),
        'relative_error_true_nu/true_f': (diff_norm/true_f_norm).item(),
        'relative_error_true_tail/true_f_tail': (diff_norm_tail/true_f_norm_tail).item()
        }
    
    to_latex = {"es": r"$\mathrm{ES}_{\alpha}(\hat{f}(X))$",
                # under nu
                "relative_error_nu/es": r"$\frac{\|\hat f - \bar f\|_{L^2(\nu)}}{\mathrm{ES}_{\alpha}(\hat{f}(X))}$",
                "confidence_bound_nu/es": r"$\frac{95\%\text{CB}\|\hat f - \bar f\|_{L^2(\nu)}}{\mathrm{ES}_{\alpha}(\hat{f}(X))}$",
                # under hat(nu_alpha)
                "relative_error_tail/es": r"$\frac{\|\hat f - \bar f\|_{L^2(\hat \nu_\alpha)}}{\mathrm{ES}_{\alpha}(\hat{f}(X))}$",
                "confidence_bound_tail/es": r"$\frac{95\%\text{CB}\|\hat f - \bar f\|_{L^2(\hat \nu_\alpha)}}{\mathrm{ES}_{\alpha}(\hat{f}(X))}$",
                # error relative to true f
                "relative_error_true_nu/true_f": r"$\frac{\|\hat f - \bar f\|_{L^2(\nu)}}{\|\bar f\|_{L^2(\nu)}}$",
                "relative_error_true_tail/true_f_tail": r"$\frac{\|\hat f - \bar f\|_{L^2(\hat \nu_\alpha)}}{\|\bar f\|_{L^2(\hat \nu_\alpha)}}$"}     

    return results, to_latex

def write_to_latex_table(df, experiment_type, importance_sampling, sampling_alpha=None):
    # experiment type has underscores 
    label = experiment_type.replace("_", " ")
    if importance_sampling:
        name = f"{experiment_type}_importance_sampling_{sampling_alpha}"
        caption = f"{label} with importance sampling."
        ref = f"table:{experiment_type}_importance_sampling"
    else:
        name = f"{experiment_type}"
        caption = f"{label} without importance sampling."
        ref = f"table:{experiment_type}_standard_sampling"

    colfmt = "l" + "c" * len(df.columns) 
    latex_str = df.to_latex(float_format="%.4f", escape=False, index_names=False, column_format=colfmt)            
    latex_str = ("\\begin{table}[htbp]\n"
    "\\centering\n" +
    latex_str +
    f"\\caption{{{caption}}}\n"
    f"\\label{{{ref}}}\n"
    "\\end{table}\n")
    file_name = f"results/{name}.tex" 
    with open(file_name, "w") as f:
        f.write(latex_str)

def write_to_html_and_markdown():
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


# ML models 
class DeepNeuralNet(nn.Module):
    # network architecture has an effect on the results 
    # deeper not necessarily better
    # estimate in the tail are not better than baselines with importance sampling 
    # small network is close to linear regression but worse than polynomial regression
    # larger network performs worse for the tail estimates 
    def __init__(self, n_features=20, experiment_type="max_call"):
        super().__init__()

        assert experiment_type in ["max_call", "portfolio"], "Experiment type not recognized."

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

    def fit(self, X, Y, sample_weights=None):
        X = self.transform(X)
        X = X.to(self.compute_device)
        Y = Y.to(self.compute_device)
        
        if sample_weights is not None:
            # Weighted least squares: solve (X^T W X) β = X^T W Y
            # where W is diagonal matrix of weights
            sample_weights = sample_weights.to(self.compute_device)
            # Normalize weights to have mean 1 for proper scaling
            sample_weights = sample_weights / sample_weights.mean()
            sqrt_weights = torch.sqrt(sample_weights).unsqueeze(1)
            X_weighted = X * sqrt_weights
            Y_weighted = Y * sqrt_weights.squeeze()
            
            if self.compute_device.type == "cpu":
                out = torch.linalg.lstsq(X_weighted, Y_weighted, rcond=self.tol, driver='gelsd')
            else:
                out = torch.linalg.lstsq(X_weighted, Y_weighted, driver='gels')
        else:
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