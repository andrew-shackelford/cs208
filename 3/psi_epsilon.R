library("PSIlence")

epsilonGlobal <- 1
deltaGlobal <- 1e-9

# output dataframe
out <- rep(0, 500)

for (k in 1:500) {
  # for each k, calculate inverse matrix
  init <- rep(c(1/k, 0), k )
  params <- matrix(init, nrow=k, ncol=2, byrow=TRUE)
  inverse <- PSIlence:::update_parameters(params=params, hold=0, eps=epsilonGlobal, del=deltaGlobal)
  
  # since no parameters are held, each output row is the same, so just pick one
  out[k] <- inverse[1][1]
}

write.csv(out, file='psi_epsilon.csv')