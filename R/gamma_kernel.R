#######################################################
# Gamma kernel functions
#######################################################

library(gsl)
library(MittagLeffleR)


# Gamma kernel
kGamma <- function(params) {
  Vectorize(function(tau) {
    al <- params$al
    lam <- params$lam
    nu <- params$nu
    res <- (nu / gamma(al)) * tau^(al - 1) * exp(-lam * tau)
    return(res)
  })
}

# Integral \int_0^tau kGamma(s)^2 ds
K00 <- function(params) {
  Vectorize(function(tau) {
    al <- params$al
    H2 <- 2 * al - 1
    lam <- params$lam
    nu <- params$nu
    prefactor <- (nu / gamma(al))^2
    diff_gamma <- gamma(H2) - gamma_inc(H2, 2 * lam * tau)
    res <- prefactor * ifelse(lam > 0, diff_gamma / (2 * lam)^H2, tau^H2 / H2)
    return(res)
  })
}

# Integral \int_0^tau kGamma(s + j * tau)^2 ds
Kjj <- function(params) {
  function(tau) {
    Vectorize(
      function(j) {
        kGamma2 <- function(s) {
          (kGamma(params)(s + j * tau))^2
        }
        res <- integrate(kGamma2, lower = 0, upper = tau)$value
        return(res)
      }
    )
  }
}

# Integral \int_0^tau kGamma(s) ds
K0 <- function(params) {
  Vectorize(function(tau) {
    al <- params$al
    lam <- params$lam
    nu <- params$nu
    prefactor <- (nu / gamma(al)) / (lam^al)
    bkt <- gamma(al) - gamma_inc(al, lam * tau)
    res2 <- (nu / gamma(al)) / al * tau^(al)
    res <- ifelse(lam > 0, prefactor * bkt, res2)
    return(res)
  })
}

# Resolvent kernel of kGamma^2
bigK <- function(params) {
  function(tau) {
    al <- params$al
    H2 <- 2 * al - 1
    lam <- params$lam
    nu <- params$nu
    nuHat2 <- nu^2 * gamma(H2) / gamma(al)^2
    res <- nuHat2 * exp(-2 * lam * tau) * tau^(H2 - 1) * mlf(nuHat2 * tau^H2, H2, H2)
    return(res)
  }
}

# Integral \int_0^tau bigK(s) ds
bigK0 <- function(params) {
  Vectorize(function(tau) {
    bigKp <- bigK(params)
    integ <- function(s) {
      bigKp(s)
    }
    res <- ifelse(tau > 0, integrate(integ, lower = 0, upper = tau)$value, 0)
    return(res)
  })
}
