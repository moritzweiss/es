import torch
import matplotlib.pyplot as plt
from dataclasses import dataclass
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


@dataclass
class ModelParams:
    '''
    same order as in the paper 
    '''
    tau: float = 1.0
    x: float = 55.0
    T: float = 15.0
    b: float = 10.792
    q_0: float = 4.605
    m : float = 0.05
    sigma_S: float = 0.18
    r_0: float = 0.025
    zeta: float = 0.25
    gamma: float = 0.02
    sigma_r: float = 0.01
    lambd: float = 0.02
    mu_x: float = 0.01
    kappa: float = 0.07
    sigma_mu: float = 0.0012
    rho12: float = -0.3
    rho13: float = 0.06
    rho23: float = -0.04
    
    @property
    def gamma_bar(self):
        return self.gamma - self.lambd * self.sigma_r / self.zeta
        
    def to_tensor(self, device: str = 'cpu', dtype: torch.dtype = torch.float32):
        kwargs = {}
        for f, val in self.__dict__.items():
            if isinstance(val, torch.Tensor):
                print('this is a tensor')
                print(val)
            kwargs[f] = torch.tensor(val, device=device, dtype=dtype)
        return ModelParams(**kwargs)


def ou_increment(ru: torch.tensor, rate:torch.tensor, mean:torch.tensor, vol:torch.tensor, dt:torch.tensor, Z:torch.tensor):
    '''

    ru: initial value of OU process

    the model is: d rt = rate(mean - rt) dt + vol dW_t

    see Glassermann page 109 
    
    '''
    assert dt > 0, "dt must be positive"
    assert Z.dim() == 1, "Z must be a 1-dimensional tensor"

    normal_mean = ru*torch.exp(-rate*dt) + mean * (1 - torch.exp(-rate*dt)) 
    normal_std = vol * torch.sqrt((1 - torch.exp(-2*rate*dt)) / (2*rate))
    xt = normal_mean + normal_std * Z

    return xt

def arith_bm_increment(qu: torch.tensor, mean:torch.tensor, vol:torch.tensor, dt:torch.tensor, Z:torch.tensor):
    '''
    qu: initial value 
    mu: drift term
    sigma: volatility term
    dt: time increment
    Z: standard normal random variable

    The model is: d X_t = mean dt + vol dW_t
    '''

    assert dt > 0, "dt must be positive"
    assert Z.dim() == 1, "Z must be a 1-dimensional tensor"

    normal_mean = mean * dt
    normal_std = vol * torch.sqrt(dt) * Z
    xt = qu + normal_mean + normal_std

    return xt

def multivariate_normal(corr: torch.tensor, N: int):    
    assert corr.dim() == 2 and corr.size(0) == corr.size(1), "corr must be a square matrix"
    assert corr.size(0) == 3, "corr must be a 3x"
    L = torch.linalg.cholesky(corr)
    Z = torch.randn(N, corr.size(0))  
    Z = Z @ L.T  
    return Z

class MultivariateNormal:
    def __init__(self, L, device: str = 'cpu', dtype: torch.dtype = torch.float32):
        self.L = L.to(device=device, dtype=dtype) 

    def sample(self, N: int):
        Z = torch.randn(N, self.L.size(0), device=self.L.device, dtype=self.L.dtype)
        return Z @ self.L.T

class Simulator:
    def __init__(self, params: ModelParams, device: str = 'cpu', dtype: torch.dtype = torch.float32):
        self.params = params
        self.tparams = params.to_tensor(device=device, dtype=dtype)                
        corr=torch.tensor([[1.0, params.rho12, params.rho13],[params.rho12, 1.0, params.rho23], [params.rho13, params.rho23, 1.0]], device=device, dtype=dtype)
        L = torch.linalg.cholesky(corr)
        self.mulivariate_normal = MultivariateNormal(L=L, device=device, dtype=dtype)
        self.affine_coefficients = AffineCoefficients(params=self.tparams)
    
    def simulate(self, samples: int = 1000):            
        cp = self.tparams

        # simulate up to tau under P 
        Z = self.mulivariate_normal.sample(N=samples)
        qtau = arith_bm_increment(qu=cp.q_0, mean=(cp.m-0.5*(cp.sigma_S**2)), vol=cp.sigma_S, dt=cp.tau, Z=Z[:, 0] )
        rtau = ou_increment(ru=cp.r_0, rate=cp.zeta, mean=cp.gamma, vol=cp.sigma_r, dt=cp.tau, Z=Z[:, 1])    
        # just setting mean = 0 here. the update still works as expected. 
        mutau = ou_increment(ru=cp.mu_x, rate=-cp.kappa, mean=torch.tensor(0.0), vol=cp.sigma_mu, dt=cp.tau, Z=Z[:, 2])

        # simulate from tau to T under Q 
        Z = self.mulivariate_normal.sample(N=samples)
        # freeze interest rate at rtau
        qT = arith_bm_increment(qu=qtau, mean=(rtau - 0.5 * cp.sigma_S**2), vol=cp.sigma_S, dt=cp.T-cp.tau, Z=Z[:, 0])        
        gamma_bar = cp.gamma - cp.lambd * cp.sigma_r / cp.zeta
        rT = ou_increment(ru=rtau, rate=cp.zeta, mean=gamma_bar, vol=cp.sigma_r, dt=cp.T-cp.tau, Z=Z[:, 1])
        # just setting mean = 0 here. the update still works as expected. 
        muT = ou_increment(ru=mutau, rate=-cp.kappa, mean=torch.tensor(0.0), vol=cp.sigma_mu, dt=cp.T-cp.tau, Z=Z[:, 2])

        # compute final payoff in Y 
        all_terms = [self.affine_coefficients.F(t=cp.T, k=k, rt=rtau, muxt=muT) for k in range(1, 51)]
        all_terms = torch.stack(all_terms, dim=1)  
        sum_of_terms = torch.sum(all_terms, dim=1)  
        Y = self.affine_coefficients.F(t=cp.tau, k=cp.T-cp.tau, rt=rtau, muxt=mutau)*torch.maximum(torch.exp(qT), cp.b*sum_of_terms)

        # X = torch.stack([qT, rT, muT], dim=1)
        X = torch.stack([qtau, rtau, mutau], dim=1)
        Y = Y.unsqueeze(1)

        output = dict(X=X, Y=Y, qT=qT, rT=rT, muT=muT, qtau=qtau, rtau=rtau, mutau=mutau)

        return output                 
            

class AffineCoefficients:
    def __init__(self, params):
        self.params = params
        self.zeta = params.zeta
        self.kappa = params.kappa       
        self.lambd = params.lambd      
        self.gamma = params.gamma
        self.sigma_r = params.sigma_r         
        self.sigma_mu = params.sigma_mu
        self.rho23 = params.rho23   
        self.gamma_bar = params.gamma_bar
        self.b = params.b   
        self.tau = params.tau
        self.T = params.T
    
    def Br(self, t,T):
        deltat  = T - t 
        return (1 - torch.exp(-self.zeta * deltat)) / self.zeta

    def Bmu(self, t,T):
        deltat  = T - t 
        return (torch.exp(self.kappa * deltat) - 1) / self.kappa
    
    def A(self, t,T):
        ''''
        check the corresponding paper. the parameter maps are:
        alpha = zeta 
        delta = sigma_mu
        '''
        term1 = self.gamma_bar * (self.Br(t,T) - T + t)
        # split up term2 
        term2 = T - t - 2 * self.Br(t,T) + (1 - torch.exp(-2 * self.zeta * (T - t))) / (2 * self.zeta)
        term2 = (self.sigma_r**2 / self.zeta**2) * term2
        # split up term3
        # the exponential here blows up 
        term3 = T - t - 2*self.Bmu(t,T) + (torch.exp(2 * self.kappa * (T - t)) - 1)/ (2 * self.kappa)
        # as a hack just drop the later term 
        # term3 = T - t 
        # - 2*self.Bmu(t,T) + (torch.exp(self.kappa * (T - t)) - 1)/ (2 * self.kappa)
        term3 = (self.sigma_mu**2 / self.kappa**2) * term3
        # split up term 4 
        term4 = self.Bmu(t,T) - T + t +self.Br(t,T) - (1 - torch.exp(-(self.zeta - self.kappa) * (T - t))) / (self.zeta - self.kappa)
        term4 = (2*self.rho23 * self.sigma_r * self.sigma_mu / (self.zeta * self.kappa)) * term4

        return torch.exp(term1 + 0.5 * (term2 + term3 + term4))
    
    def F(self, t,k,rt,muxt):
        '''
        F(t,k,rt,muxt) = A(t,t+k) * exp(-Br(t,t+k) * rt - Bmu(t,t+k) * muxt)
        '''
        return self.A(t, t+k) * torch.exp(-self.Br(t, t+k) * rt - self.Bmu(t, t+k) * muxt)
        

class Net(nn.Module):
    def __init__(self, n_assets=20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_assets, 16),
            nn.BatchNorm1d(16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.BatchNorm1d(16),
            nn.Tanh(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x)


params = ModelParams()
SI = Simulator(params=params, device="cuda", dtype=torch.float32)
out  = SI.simulate(samples=int(1e4))
qtau = out['qtau']
qT = out['qT']

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(qtau.cpu().numpy(), bins=50, alpha=0.7, color='blue')
plt.title('Histogram of qtau')
plt.xlabel('qtau')
plt.ylabel('Frequency')
plt.savefig("histogram_qtau.png")

plt.subplot(1, 2, 2)
plt.hist(qT.cpu().numpy(), bins=50, alpha=0.7, color='green')
plt.title('Histogram of qT')
plt.xlabel('qT')
plt.ylabel('Frequency')
plt.savefig("histogram_qT.png")

# plt.tight_layout()
# plt.show()



if False:
    # Find normalization constants for y
    params = ModelParams()
    SI = Simulator(params=params, device="cuda", dtype=torch.float32)
    x_sample, y_sample = SI.simulate(samples=int(1e4))
    y_mean = y_sample.mean().item()
    y_std = y_sample.std().item()
    print(f"Normalization constants for y: mean={y_mean}, std={y_std}")

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net(n_assets=3).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, min_lr=1e-7)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1 )

    losses = []
    # batch_size = int(2**13)
    batch_size = int(2**10)
    gradient_steps = int(1e3)
    lrs = []
    model.train()
    params = ModelParams()
    SI = Simulator(params=params, device=device, dtype=torch.float32)
    # batch_size = int(2**12)
    batch_size = int(2**14)
    # gradient_steps = int(1)
    gradient_steps = int(100)
    for epoch in range(gradient_steps):
        x, y = SI.simulate(samples=batch_size)
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("NaN or Inf detected in input data, skipping this batch.")
            continue
        if torch.isnan(y).any() or torch.isinf(y).any():
            print("NaN or Inf detected in target data, skipping this batch.")
            continue
        threshold = 1e6
        if (x.abs() > threshold).any() or (y.abs() > threshold).any():
            print(f"Warning: Input or target data exceeds threshold at epoch {epoch}")
            continue
        x = (x - x.mean(0)) / (x.std(0) + 1e-6)
        y = torch.log1p(y)
        # y = (y - y.mean(0)) / (y.std(0) + 1e-6)
        # y = (y - 105) / (59 + 1e-6)
        predictions = model(x)    
        loss = criterion(predictions, y)
        # if loss > 10000:
        #     print(f'high losses detected')
        #     print(torch.max(x), torch.min(x), torch.max(y), torch.min(y))
        losses.append(loss.cpu().item())
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        # scheduler.step(loss)
        scheduler.step(loss)
        lrs.append(optimizer.param_groups[0]['lr'])
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}, Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        

            
    plt.plot(losses)
    plt.xlabel("gradient step")
    plt.ylabel("mean squared error loss")
    plt.xlim(0, gradient_steps)
    plt.ylim(min(losses), losses[0]+0.1*losses[0])
    plt.title(f"variable annuity, batch size={batch_size}")
    plt.savefig("training_loss_variable_annuity.png")


    with torch.no_grad():
        min_weight = float('inf')
        max_weight = float('-inf')
        for name, param in model.named_parameters():
            if param.requires_grad:
                min_weight = min(min_weight, param.min().item())
                max_weight = max(max_weight, param.max().item())
        print(f"Model weights: min={min_weight}, max={max_weight}")

    # plt.figure()
    # plt.plot(lrs)
    # plt.xlabel("gradient step")
    # plt.ylabel("learning rate")
    # plt.xlim(0, gradient_steps)
    # plt.title(f"variable annuities, batch size={batch_size}")
    # plt.savefig("learning_rate_variable_annuities.png")

    # if __name__ == "__main__":    
    #     params = ModelParams()
    #     prams = params.to_tensor(device='cpu', dtype=torch.float32)
    #     SI = Simulator(params=params, device='cpu', dtype=torch.float32)
    #     X,Y = SI.simulate(samples=8)
    #     print("X shape:", X.shape)
    #     print("Y shape:", Y.shape)
    #     print("X:", X)
    #     print("Y:", Y)