import itertools
import time 
import numpy as np 
from sklearn.linear_model import LinearRegression 
import torch 
from torch import nn 
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from simulate_gbm import default_config
import scipy
from sklearn.linear_model import Ridge

t_float_type = torch.float32
np_float_type = np.float32

torch.manual_seed(4)

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
print(device)


class Model(nn.Module):
    def __init__(self, n_assets, nodes=128):
        # can also write this as a sequential model 
        super(Model, self).__init__()
        # self.layers = nn.Sequential()
        self.linear1 = nn.Linear(n_assets, nodes)
        self.linear2 = nn.Linear(nodes, nodes)
        self.linear3 = nn.Linear(nodes, nodes)
        self.linear4 = nn.Linear(nodes, 1)
        self.activation = nn.Tanh()
        self.batch_norm_1 = nn.BatchNorm1d(nodes)
        self.batch_norm_2 = nn.BatchNorm1d(nodes)
        self.batch_norm_3 = nn.BatchNorm1d(nodes)

    def forward(self, x): 
        # first hidden layer 
        x = self.linear1(x)
        x = self.batch_norm_1(x)
        x = self.activation(x)
        # second hidden layer 
        x = self.linear2(x)
        x = self.batch_norm_2(x)
        x = self.activation(x)
        # third hidden layer 
        x = self.linear3(x)
        x = self.batch_norm_3(x)
        x = self.activation(x)        
        # output layer 
        x = self.linear4(x)
        return x 


class RegressionNN(nn.Module):
    def __init__(self, n_assets):
        super(RegressionNN, self).__init__()
        self.linear = nn.Linear(5151, 1, bias=False)

    def forward(self, x):
        return self.linear(x)


default_config = {'correlation': 0.3, 'n_assets': 100, 'initial_price': 10, 'train_samples': int(5e5), \
                    'mc_samples': int(6e7), 'poly':False, 'payoff': 'binary', 'seed':1234, \
                    't':1/52, 'T':1/3, 'binary_scalar':10, 'model': 'nn', 'importance_sampling': False}

# payoff options: binary and max call 


# neural network training and testing 
class NNTrainTest():
    def __init__(self, config):
        # move everything into a config file
        # only set attributed which are needed later one  
        self.train_samples = config['train_samples']
        self.n_assets = config['n_assets']
        self.initial_price = torch.tensor(config['initial_price'], device=device, dtype=t_float_type)
        self.t = torch.tensor(config['t'], device=device, dtype=t_float_type)
        self.T = torch.tensor(config['T'], device=device, dtype=t_float_type)
        self.correlation = 0.3
        if self.n_assets == 10:
            self.vol = torch.tensor([(0.1+(n+90)/200) for n in range(1, 11)], device=device, dtype=t_float_type)
        else:
            self.vol = torch.tensor([(0.1+n/200) for n in range(1, self.n_assets+1)], device=device, dtype=t_float_type)

        # create covariance matrix 
        covariance_matrix = self.correlation*torch.ones((self.n_assets, self.n_assets), device=device, dtype=t_float_type)
        covariance_matrix.fill_diagonal_(1.0)
        covariance_matrix = torch.diag(self.vol) @ covariance_matrix @ torch.diag(self.vol)
        self.cov = covariance_matrix
        self.A = torch.linalg.cholesky(self.cov)
        # correlation matrix for normal distribution 
        self.corr = self.correlation*torch.ones((self.n_assets, self.n_assets), device=device, dtype=t_float_type)
        self.corr.fill_diagonal_(1.0)
        self.corr_A = torch.linalg.cholesky(self.corr)

        # mean 
        loc = torch.zeros(self.n_assets, device=device, dtype=t_float_type)

        # normal distribution  
        self.normal_distribution = torch.distributions.multivariate_normal.MultivariateNormal(loc=loc, scale_tril=self.A)
        self.standard_normal = torch.distributions.normal.Normal(torch.tensor([0.0], device=device), torch.tensor([1.0], device=device))

        # model 
        self.model = config['model']

        # strike price for binary payoff
        if self.n_assets == 10:
            self.strike_price = torch.tensor(14.46, dtype=t_float_type, device=device)
        else:
            self.strike_price = torch.tensor(16.3, dtype=t_float_type, device=device)
        self.binary_scalar = torch.tensor(10.0, dtype=t_float_type, device=device)

        #
        self.distribution_shift = True

        self.payoff = config['payoff']

        # 
        self.importance_sampling = config['importance_sampling']
        quantile = torch.tensor(scipy.stats.norm.ppf(0.99), device=device, dtype=t_float_type)
        print(f'normal quantile: {quantile}')
        v = torch.ones(self.n_assets, device=device, dtype=t_float_type)
        self.shift_vector = (self.corr_A.T @ v)*quantile/torch.linalg.norm(self.corr_A.T @ v, ord=2)


    def simulate_bm(self, n_samples, t, importance_sampling):
        # n_samples int, t torch tensor 
        # check why self made multivariate normal works worse than torch multivariate normal ? 
        if importance_sampling:
            return torch.sqrt(t)*self.vol*torch.matmul(self.standard_normal.sample((n_samples, self.n_assets)).squeeze(-1)+self.shift_vector, torch.transpose(self.corr_A, 0,1))
        else:
            return torch.sqrt(t)*self.vol*torch.matmul(self.standard_normal.sample((n_samples, self.n_assets)).squeeze(-1), torch.transpose(self.corr_A, 0,1))
        # return torch.sqrt(t)*self.normal_distribution.sample((n_samples,)) 
    
    def simulate_V(self, n_samples):
        dt = self.T - self.t
        return self.simulate_bm(n_samples=n_samples, t=dt, importance_sampling=False) - 0.5*(self.vol**2)*dt  

    def simulate_X(self, n_samples):
        return torch.log(self.initial_price) + self.simulate_bm(n_samples=n_samples, t=self.t, importance_sampling=self.importance_sampling) - 0.5*(self.vol**2)*self.t  

    def compute_Y(self, X, V):
        # print('min debug')
        if self.payoff == 'max_call':
            return torch.maximum(torch.exp(torch.max(X+V, axis=1)[0])-self.strike_price, torch.tensor(0.0))
        if self.payoff == 'binary':
            return self.binary_scalar*(torch.exp(torch.max(X+V, axis=1)[0])-self.strike_price > 0)

    def train(self, batch_size=2**13, lr=1e-3, n_gradient_steps=int(1e3), cond=1e-5):
        if self.model == 'nn' or self.model == 'p_reg_nn':
            if self.model == 'p_reg_nn':
                self.model_inst = RegressionNN(n_assets=self.n_assets).to(device)            
            else:
                self.model_inst = Model(n_assets=self.n_assets).to(device)            
            optimizer = torch.optim.Adam(self.model_inst.parameters(), lr=lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', verbose=True, patience=3, min_lr=1e-6)
            running_loss = 0 
            mse_loss = nn.MSELoss()
            for n in range(n_gradient_steps):
                optimizer.zero_grad()
                X = self.simulate_X(n_samples=batch_size)
                V = self.simulate_V(n_samples=batch_size)
                Y = self.compute_Y(X=X, V=V).reshape((batch_size, 1))
                if self.model == 'p_reg_nn':
                    with torch.no_grad():
                        X = self.poly_torch(X, degree=2, normalize=True)
                prediction = self.model_inst.forward(X)
                loss = mse_loss(Y, prediction)        
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if n % 1000 == 0 and n >1:
                    scheduler.step(running_loss/1000)
                    print(int(n/1000))
                    print(f'average loss per {1000} batches: {running_loss/1000}')
                    running_loss = 0     
        with torch.no_grad():
            if self.model == 'reg' or self.model == 'p_reg':
                if self.model == 'reg':
                    X = self.simulate_X(n_samples=self.train_samples)
                    V = self.simulate_V(n_samples=self.train_samples)
                    Y = self.compute_Y(X=X, V=V).reshape((self.train_samples, 1))                
                    del V
                    Y = Y.cpu().numpy()
                    torch.cuda.empty_cache()
                    X = self.poly_torch(X, degree=1)
                    X = X.cpu().numpy()
                if self.model == 'p_reg':
                    n_batches = 5
                    batch_size = int(self.train_samples/n_batches)
                    X_list = []
                    Y_list = []
                    for _ in range(n_batches):
                        X = self.simulate_X(n_samples=batch_size)
                        V = self.simulate_V(n_samples=batch_size)
                        Y = self.compute_Y(X=X, V=V).reshape((batch_size, 1))                
                        Y = Y.cpu().numpy()
                        X = self.poly_torch(X, degree=2, normalize=False) 
                        X = X.cpu().numpy()
                        X_list.append(X)
                        Y_list.append(Y)
                    X = np.vstack(X_list)
                    del X_list
                    Y = np.vstack(Y_list)
                    del Y_list

                    # clf = Ridge(alpha=1.0, fit_intercept=False)
                    # clf.fit(X, Y)                    
                    # self.model_inst = torch.tensor(clf.coef_.T, device=device)
                    # return None 


                print(f'mean of Y is {np.mean(Y)}')
                # cond = np.finfo(np_float_type).eps
                # cond = np.finfo(np_float_type).eps
                # cond = 1e-5
                # cond = 1e-3
                print(f'condition number is {cond}')
                out = scipy.linalg.lstsq(X, Y, cond=cond, lapack_driver='gelsd', overwrite_a=True, overwrite_b=True)
                print('fitted')
                print(f'singular values: {out[3][0]}, {out[3][-1]}')
                self.model_inst = torch.tensor(out[0], device=device, dtype=t_float_type)
                del X, Y
                return  None 

    def normalize(self, x, bias=True):
        if bias:
            torch.sub(x[:, 1:], x[:, 1:].mean(0), out=x[:, 1:])
            torch.div(x[:, 1:], x[:, 1:].std(0), out=x[:, 1:])
            return x
        else:
            return (x - x.mean(0))/x.std(0)

    def poly_torch(self, x, degree,normalize=False):
        device = x.device
        n_samples = x.shape[0]
        N = x.shape[1]
        if degree == 1:
            y = torch.column_stack((torch.ones(n_samples, device=x.device, dtype=x.dtype), x))
            if normalize:
                return self.normalize(y)
            else:
                return y 
        if degree == 2:
            y = torch.empty((n_samples, N+1+int(N*(N+1)/2)), device=x.device, dtype=x.dtype)        
            y[:, 0] = torch.ones(n_samples, device=device)
            y[:, 1:N+1] = x
            offset = N+1
            end = 0 
            for n in range(N):
                length = N - n 
                start = end 
                end = start + length
                y[:, offset+start:offset+end] = x[:, n:N]*x[:, :N-n]
                torch.multiply(x[:, n:N], x[:, :N-n], out=y[:, offset+start:offset+end])
            # del x    
            if normalize:
                return self.normalize(y)
            else:
                return y
    
    def poly_np(self, x):
        # with normalisation 
        n_samples = x.shape[0]
        N = x.shape[1]        
        y = np.empty((n_samples, N+1+int(N*(N+1)/2)))
        y[:, 0] = np.ones(n_samples)
        y[:, 1:N+1] = x
        offset = N+1
        end = 0 
        for n in range(N):
            length = N - n 
            start = end 
            end = start + length
            y[:, offset+start:offset+end] = x[:, n:N]*x[:, :N-n]
            np.multiply(x[:, n:N], x[:, :N-n], out=y[:, offset+start:offset+end])
        del x 
        y[:, 1:] = (y[:, 1:] - y[:, 1:].mean(0))/ y[:, 1:].std(0)
        return y        


    def polynomial_features(self, X, degree):
        # very slow 
        x = list(itertools.combinations_with_replacement(range(100),2))
        products = torch.zeros((X.shape[0], len(x)+1))
        products[:, 0] = torch.ones(products.shape[0])
        for n, (i,j) in enumerate(x):
            products[:, n+1] = X[:,i]*X[:, j]   
        del X
        return  

    def test(self, n_mc_samples=int(6e8), batch_size=int(1e6)):
        if self.model == 'nn': 
            self.model_inst.eval()
        with torch.no_grad():
            running_error = [] 
            running_variance = []
            N = int(n_mc_samples/batch_size)
            print(N)
            for n in range(N):
                U = self.simulate_V(n_samples=batch_size)
                V = self.simulate_V(n_samples=batch_size)
                X = self.simulate_X(n_samples=batch_size)
                Y = self.compute_Y(X=X, V=V)
                Z = self.compute_Y(X=X, V=U)
                del U, V 
                # to do add regression code 
                if self.model == 'reg':
                    X = self.poly_torch(X , degree=1)
                    prediction = X @ self.model_inst                    
                if self.model == 'p_reg':                    
                    X = self.poly_torch(X, degree=2, normalize=False)
                    prediction = X @ self.model_inst
                    # prediction = self.model_inst.predict(X) 
                if self.model == 'nn':                    
                    prediction = self.model_inst.forward(X)
                if self.model == 'p_reg_nn':
                    X = self.poly_torch(X, degree=2, normalize=True)
                    prediction = self.model_inst.forward(X)
                prediction = prediction.reshape(-1)
                error = Y*Z + prediction*(prediction-Y-Z)        
                variance = Y*Z
                running_error.append(error.cpu().numpy())
                running_variance.append(variance.cpu().numpy())
                # if n%10 == 0:
                    # print(n)
        running_error = np.hstack(running_error)
        running_variance = np.hstack(running_variance)
        error = np.mean(running_error)
        standard_error = np.std(running_error)/np.sqrt(n_mc_samples)
        variance = np.mean(running_variance)
        if error < 0:
            return -np.sqrt(-error)/np.sqrt(variance), standard_error
        else:
            return np.sqrt(error/variance), standard_error

    def poly_feature_np(self, x):
        out = []
        N = x.shape[1]
        for n in range(N):
            out.append(x[:, :N-n]*x[:, n:])
        out = np.hstack(out)
        return None 

if __name__ == "__main__":
    config = default_config.copy()
    config['payoff'] = 'binary'
    config['model'] = 'reg'
    config['n_assets'] = 100
    config['train_samples'] = int(5e5)
    config['importance_sampling'] = True
    Test = NNTrainTest(config=config)
    print(Test.importance_sampling)
    start = time.time() 
    Test.train(batch_size=int(2**13), n_gradient_steps=int(3e4), cond=1e-5)
    print(time.time()-start)
    out = Test.test(n_mc_samples=int(6e7), batch_size=int(2e5))
    print(out)