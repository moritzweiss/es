from max_call_option import DataSampler, MaxCallConfig
import torch

DS = DataSampler(config=MaxCallConfig(), dtype=torch.float32, device="cuda:1", n_samples=int(1e6), seed=0)
x, weights, z = DS.sampleX()
Y = DS.sampleY(initial_value=x)

mean = x.mean(0) # the mean should be around 10
std = x.std(0) # the std should be around 

print("mean:", mean[-1] )
print("std:", std[-1] )


# both are correct 
