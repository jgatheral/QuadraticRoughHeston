#####################################################################################
# Jim Gatheral, June 2021
# Negative H added May 2023
# Definitions changed March 2025
#####################################################################################

library(gsl)
library(MittagLeffleR)

#######################################################
# Gamma kernel functions
#######################################################

# Gamma kernel
kGamma <- function(params)Vectorize(function(tau){
  al <- params$al
  H <- al - 1/2
  lam <- params$lam
  eta <- params$eta
  return(eta*sqrt(2*H)*tau^{al-1}*exp(-lam*tau))
})

# Gamma variance K00
K00 <- function(params) Vectorize(function(tau){
  al <- params$al
  H <- al - 1/2
  H2 <- 2*H
  lam <- params$lam
  eta <- params$eta
  prefactor <- H2/((2*lam)^H2)
  bkt <- gamma(H2)- gamma_inc(H2,2*lam*tau)
  res2 <- eta^2*tau^(2*H)
  res <- ifelse(lam>0,prefactor * bkt,res2)
  return(res)
})

# K11
K11 <- function(params)Vectorize(function(tau){
  al <- params$al
  H <- al - 1/2
  H2 <- 2*H
  lam <- params$lam
  eta <- params$eta
  
  prefactor <- eta^2*H2/((2*lam)^H2)
  bkt <- gamma_inc(H2,2*lam*tau)- gamma_inc(H2,4*lam*tau)
  res2 <- eta^2*tau^(2*H)*(2^H2-1)
  res <- ifelse(lam>0,prefactor * bkt,res2)
  return(res)
})

# Kjj
Kjj <- function(params)function(tau) Vectorize(
  function(j) {
    kGamma2 <-function(s){(kGamma(params)(s+j*tau))^2}
    res <- integrate(kGamma2,lower=0,upper=tau)$value
    return(res)
  })


# K0
K0 <- function(params) Vectorize(function(dt){
  
  al <- params$al
  H <- al - 1/2
  lam <- params$lam
  eta <- params$eta
  etaTilde <- eta*sqrt(2*H)
  
  prefactor <- etaTilde/(lam^al)
  bkt <- gamma(al)- gamma_inc(al,lam*dt)
  res2 <- etaTilde/al*dt^(al)
  res <- ifelse(lam>0,prefactor * bkt,res2)
  return(res)
  
})

K1 <- function(params) function(dt){
  
  al <- params$al
  H <- al - 1/2
  lam <- params$lam
  eta <- params$eta
  
  prefactor <- eta*sqrt(2*H)/(lam^al)
  bkt <- gamma_inc(al,lam*dt)- gamma_inc(al,2*lam*dt)
  res2 <- sqrt(2*H)/al*dt^(al)*(2^al-1)
  res <- ifelse(lam>0,prefactor * bkt,res2)
  return(res)
  
}

# Gamma covariance
K0j <- function(params,j)function(tau){
  
  gp <- kGamma(params)
  integr <- function(s){gp(s)*gp(s+k*t)}
  res <- integrate(integr, lower=0,upper=t)$value
  return(res)
  
}

# Gamma first order covariance
K01 <- function(params)function(t){
  
  gp <- kGamma(params)
  eps <- 0
  integr <- function(s){gp(s)*gp(s+t)}
  res <- integrate(integr, lower=0,upper=t)$value
  return(res)
  
}

# Resolvent kernel of kGamma^2
bigK <- function(params)function(tau){
  al <- params$al
  H <- al - 1/2
  H.2 <- 2*al-1
  lam <- params$lam
  eta <- params$eta
  etaHat2 <- eta^2*H.2*gamma(H.2)
  tau.2H <- tau^(H.2)
  res <- etaHat2*exp(-2*lam*tau) * tau^(H.2-1)*mlf(etaHat2*tau.2H,H.2,H.2)
  return(res)    
}

# bigK0
bigK0<- function(params)Vectorize(function(tau){
  
  bigKp <- bigK(params)
  integ <- function(s){bigKp(s)}
  res <- ifelse(tau>0,integrate(integ,lower=0,upper=tau)$value,0)
  return(res)    
})

  


