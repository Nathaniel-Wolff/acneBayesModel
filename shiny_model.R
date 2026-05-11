library(mvtnorm)

#feed foward algorithm with trained parameters
linear_acne_model <- function(A, Y_t, T_jt, G_j, M, b, Q, R) {
  Z_t_mean <- A %*% Y_t + G_j %*% T_jt
  Z_t <- pvmnorm(Z_t_mean, Q)
  
  
  }

