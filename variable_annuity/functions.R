library(dplyr)
library("StMoMo")
library(keras)
library(tensorflow)

#define function that computes the error between true solution and estimation on the alpha-quantile
quantile_error = function(true_solution, estimation, alpha){
  return((quantile(estimation, alpha) - quantile(true_solution$a_outer, alpha))/quantile(true_solution$a_outer, alpha)*100)
}

#G2++:
lambda_s = 0 #0.04
sigma = 0.1
lambda_x = -0.0033
lambda_y = 0.0255
a = 0.3912
b = 0.0785
nu = 0.021
eta = 0.0135
rho_r = -0.6450

beta0 = 1.54617/100
beta1 = 2.47537/100
beta2 = -26.22674/100
beta3 = 26.04199/100
theta1 = 4.80792/100
theta2 = 5.83035/100

rho_1 = -0.15
t_rho_2 = 0.15

#multipopulation Li-Lee:
mort = data.frame(theta = -2.0014, sigma2_e = 2.418, c = -0.06202683, a = 0.9825, sigma2_c = 1.272)
mort_age = read.table("multi_pop.txt", header = TRUE)
mort_time = read.table("multi_pop_k.txt", header = TRUE)

g = function(k, t){
  return(1/k*(1-exp(-k*t)))
}

f = function(t){
  return(beta0 + beta1*theta1/t*(1- exp(-t/theta1)) + beta2*theta1/t*(1-exp(-t/theta1)*(1+t/theta1)) + beta3*theta2/t*(1-exp(-t/theta2)*(1+t/theta2)))
}
psi = function(t){
  return(f(t) + nu**2*g(a,t)**2 + eta**2*g(b,t)**2 + rho_r*nu*eta*g(a,t)*g(b,t))
}

# nested simulations with specified number of inner and outer scenarios --> n = n_outer*n_inner
do_nmc =function (n, n_outer, n_inner){
  future_years = age.max-age
  X = array(rnorm(4*n*future_years), c(n, future_years,4)) #[,,1] = interest rate x, [,,2] = interest rate y, [,,3] = mortality Europe, [,,3] = mortality Switzerland,
  outer_scenarios = array(rnorm(n_outer*4), c(n_outer,4))
  X[,1,1] = rep(outer_scenarios[,1], each = n_inner)
  X[,1,2] = rep(outer_scenarios[,2], each = n_inner)
  X[,1,3] = rep(outer_scenarios[,3], each = n_inner)
  X[,1,4] = rep(outer_scenarios[,4], each = n_inner)
  
  value = 0
  x = 0
  y = 0
  integ_r = 0
  sum_mu = 0
  K.t = mort_time$K.t[nrow(mort_time)]
  k.t = mort_time$k.t[nrow(mort_time)]
  for(i in 1:future_years){
    x = a*x + nu*X[,i,1]
    y = b*y + eta*X[,i,2]
    r = x+y+psi(i)
    integ_r = integ_r + r
    mu = exp(mort_age$A.x[(age-age.min)+i] + mort_age$B.x[(age-age.min)+i]*K.t 
             + mort_age$a.x[(age-age.min)+i] + mort_age$b.x[(age-age.min)+i]*k.t)
    sum_mu = sum_mu + mu
    value = value + exp(-integ_r - sum_mu)
    K.t = K.t + mort$theta + sqrt(mort$sigma2_e)*X[,i,3]
    K.t = mort$c + mort$a*k.t + sqrt(mort$sigma2_c)*X[,i,4]
  }
  return(list("a_outer"= tapply(value, rep(1:n_outer, each=n_inner), mean), "X"= outer_scenarios))
}

# functions to create design matrix Phi
create_matrix_Phi = function(degree, X, progress = FALSE){
  n = nrow(X)
  #determine the basis functions depending on the specified degree
  basis = 0:case_when(degree == 0 ~ 0,
                      degree == 1 ~ 4,
                      degree == 2 ~ 14)
  m = length(basis)
  Phi = array(dim = c(n,m))
  for(i in 1:m){
    if(progress){print(paste(i, "/", m))}
    Phi[,i] = basis_fun(X, basis[i])
  }
  return(Phi)
}

basis_fun =function(x, alpha){
  return(case_when( alpha == 0 ~ 1,
                    alpha == 1 ~ x[,1],
                    alpha == 2 ~ x[,2],
                    alpha == 3 ~ x[,3],
                    alpha == 4 ~ x[,4],
                    alpha == 5 ~ x[,1]**2,
                    alpha == 6 ~ x[,2]**2,
                    alpha == 7 ~ x[,3]**2,
                    alpha == 8 ~ x[,4]**2,
                    alpha == 9 ~ x[,1]*x[,2],
                    alpha == 10 ~ x[,1]*x[,3],
                    alpha == 11 ~ x[,1]*x[,4],
                    alpha == 12 ~ x[,2]*x[,3],
                    alpha == 13 ~ x[,2]*x[,4],
                    alpha == 14 ~ x[,3]*x[,4]))
}

do_LSMC = function(simulation, new_sim, degree=2){
  #Phi is the design matrix. beta can be calculated in closed form
  Phi = create_matrix_Phi(degree, simulation$X)
  beta = c(solve(t(Phi)%*%Phi)%*%(t(Phi)%*%simulation$a_outer))
  
  new_Phi = create_matrix_Phi(degree, new_sim)
  LSMC = new_Phi%*%beta
  return(LSMC)
}

do_neural = function(simulation, new_sim, act_fun = NA, first_layer = 10, batch_size = 128, epochs = 100, learning_rate = NA, loss_fun = "mse", verbose = 0){
  input = as.matrix(simulation$X)
  output = as.matrix(simulation$a_outer)
  
  #define quadratic activation function if required
  activation_quadratic = function(x){return(x^2)}
  my_norm_out = function(z){
    return(z*sd(output) + mean(output))
  }
  
  if(is.na(learning_rate)){
    learning_rate = learning_rate_schedule_exponential_decay( initial_learning_rate = 0.003, 
                                                              decay_steps = epochs * nrow(input) / batch_size, 
                                                              decay_rate = 0.94)
  }
  
  constant = TRUE
  while(constant){ #restart the neural network if there's no improvement in the first epochs
    model = keras_model_sequential()
    if(is.na(act_fun)){
      model %>% 
      layer_dense(units = first_layer, activation = activation_quadratic, input_shape = c(4)) %>%
      layer_dense(units = 1, activation = "linear")%>%
        layer_lambda(my_norm_out)
    } else {
      model %>% 
      layer_dense(units = first_layer, activation = act_fun) %>%
      layer_dense(units = 1, activation = "linear")%>%
      layer_lambda(my_norm_out)
    }
  
  
    model %>% compile(
      optimizer = optimizer_adam(learning_rate = learning_rate),
      loss = loss_fun,
      metrics = list("mean_absolute_percentage_error")
    )
    
    history = model %>% keras::fit(input, output, epochs = 6, batch_size = batch_size, verbose = verbose)
    constant = (history$metrics$loss[4] - history$metrics$loss[6])/history$metrics$loss[1] < 0.0001
  }
  history = model %>% keras::fit(input, output, epochs = epochs, batch_size = batch_size, verbose = verbose)
  
  pred_input = as.matrix(new_sim)
  predictions <- model %>% predict(pred_input, verbose = 0)
  
  return(predictions)
}