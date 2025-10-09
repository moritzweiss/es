source("functions.R")

#get true solution from large nested simulation
#you probably can't run all of this in one go on a normal laptop --> do bit by bit
temp = do_nmc(n = 10**10, n_outer = 10**5, n_inner = 10**5)
true_solution = temp$a_outer

####
#### run MC, LSMC and NN approaches
####

#first define matrices with results
number_of_neurons = c(1,2,3,4,5,8,12,15,20)
res_neural = matrix(NA, nrow=n_runs, ncol = length(number_of_neurons))
res_LSMC = numeric(n_runs)
res_MC = numeric(n_runs)
time_neural = res_neural
time_LSMC = res_LSMC
time_MC = res_MC

#run all approaches 100 times
#might take a long time --> do bit by bit
n_runs = 100
for(i in 1:n_runs){
  print(i)
  
  # run small nested Monte-Carlo
  start = Sys.time()
  simulation = do_nmc(10000, 1000, 10)
  end = Sys.time()
  time_MC[i] = difftime(end, start, units = "secs")
  new_sim = array(rnorm(10**6*4), c(10**6,4))
  
  #run LSMC based on previous MC
  start = Sys.time()
  LSMC = do_LSMC(simulation, new_sim)
  end = Sys.time()
  time_LSMC[i] = difftime(end, start, units = "secs")
  
  res_MC[i] = quantile_error(true_solution, simulation$a_outer, 0.9)
  res_LSMC[i] = quantile_error(true_solution, LSMC, 0.9)
  
  #run neural network for different numbers of neurons in the hidden layer
  for(j in 1:ncol(res_neural)){
    start = Sys.time()
    neural = do_neural(simulation, new_sim, epochs = 100, act_fun = NA, learning_rate = NA, first_layer=number_of_neurons[j], loss_fun = "mse", verbose = 0)
    end = Sys.time()
    time_neural[i,j] = difftime(end, start, units = "secs")
    
    res_neural[i,j] = quantile_error(true_solution, neural, 0.9)
  }
}
