library(statmod)

yFromXi <- function(params, xi) {
  al <- params$al
  H <- al - 1 / 2
  eta <- params$eta
  lam <- params$lam

  # Compute nodes and weights
  gj <- gauss.quad(n = 10, kind = "jacobi", alpha = 2 * H - 1, beta = 0)

  y.raw <- function(u) {
    # Function to be integrated
    f <- function(x) {
      prefactor <- (u / 2)^(2 * H) * eta^2 * 2 * H
      prefactor * xi(u / 2 * (1 + x)) * exp(-lam * u * (1 - x))
    }

    # Approximate the integral
    int.u <- ifelse(u > 0, sum(gj$weights * f(gj$nodes)), 0)
    yu.2 <- xi(u) - params$c - int.u
    return(sqrt(pmax(yu.2, 0)))
  }

  return(Vectorize(y.raw))
}
