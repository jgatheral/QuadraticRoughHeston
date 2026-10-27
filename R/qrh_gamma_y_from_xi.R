library(statmod)

yFromXi <- function(params, xi, n_quad = 20) {
  al <- params$al
  H <- al - 1 / 2
  nu <- params$nu
  lam <- params$lam

  # Compute nodes and weights
  gj <- gauss.quad(n = n_quad, kind = "jacobi", alpha = 2 * H - 1, beta = 0)

  y.raw <- function(u) {
    # Function to be integrated
    f <- function(x) {
      prefactor <- (u / 2)^(2 * H) * nu^2 / gamma(al)^2
      prefactor * xi(u / 2 * (1 + x)) * exp(-lam * u * (1 - x))
    }

    # Approximate the integral
    int.u <- ifelse(u > 0, sum(gj$weights * f(gj$nodes)), 0)
    yu.2 <- xi(u) - params$c - int.u
    return(sqrt(yu.2))
  }

  return(Vectorize(y.raw))
}
