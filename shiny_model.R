#install.packages("shiny")
library(mvtnorm, stats)

#feed forward algorithm with trained parameters


                                                                                                                                                                                                                                                                                                                                                           [0.00607883, 0.00607883, 0.10941893]], dtype=float32)
calc_single_Z <- function(last_Z, cur_F, T_j, G, Q, num_samples ){
  set.seed(271)
  w_t <- rep(mu, each=num_samples)  + rvmt(n = num_samples, mean = 0, sigma = Q, df = Inf)
  current_Z <- F %*% last_Z + G %*% T_j + w_t
}


linear_acne_model <- function(A, Y_t, T_jt, G_j, M, b, Q, R, alpha){
  Z_t_mean <- A %*% Y_t + G_j %*% T_jt
  Z_t <- pvmnorm(Z_t_mean, Q)
  S_t_mean <- M %*% Z_t + b
  S_t_alpha <- alpha
  S_t_beta < - alpha / S_t_mean #fix this line
  S_t <- dgamma(x = None, rate = alpha, scale  = 1/(S_t_mean * alpha) )
  
  }


#function call
