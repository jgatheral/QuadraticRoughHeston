#############################################################################
# Forward variance curves from variance swap data
#
# Inputs throughout:
#   expiries  increasing expiries T_1 < ... < T_n (years)
#   w.in      total variances w(T_i) = T_i * (variance swap rate)
#   eps       bid-offer half-spread in volatility units (0 = fit exactly)
#
# With eps > 0 each input may be moved by up to eps in volatility,
#   sqrt(w.out/T) ~ sqrt(w.in/T) + err,  |err| <= eps,
# to make the resulting forward curve smoother.
#
# Three constructions, all returning a list with at least
#   xi.curve  the forward variance curve, a vectorized function of t
#   fit.errs  sqrt(w.out/T) - sqrt(w.in/T), the volatility adjustments
#   w.out     the total variances actually fitted
#
#   xi.curve         piecewise constant forward variance
#   xi.curve.smooth  maximally smooth (Filipovic-Willems), may go negative
#   xi.curve.mono    monotone (PCHIP) interpolation of w: never negative
#############################################################################

# Total variances moved by err (volatility units), and d w.out / d err.
adjust.w <- function(expiries, w.in, err) w.in + 2 * sqrt(w.in * expiries) * err
dw.derr <- function(expiries, w.in) 2 * sqrt(w.in * expiries)

vol.errs <- function(expiries, w.in, w.out) sqrt(w.out / expiries) - sqrt(w.in / expiries)

# Box-constrained minimization of obj over err in [-eps, eps]^n.
fit.errs.optim <- function(obj, grad, n, eps) {
  if (eps <= 0) return(rep(0, n))
  fit <- optim(rep(0, n), obj, grad, method = "L-BFGS-B",
               lower = rep(-eps, n), upper = rep(eps, n))
  if (fit$convergence != 0) {
    warning("Forward variance fit did not converge: ", fit$message,
            ". Returning the fitted curve.", call. = FALSE)
  }
  fit$par
}

#############################################################################
# Piecewise constant forward variance
#############################################################################

# Matrix D with xi = D %*% w: xi_1 = w_1/T_1, xi_i = (w_i - w_{i-1})/(T_i - T_{i-1}).
fwd.diff.matrix <- function(expiries) {
  n <- length(expiries)
  D <- diag(1 / diff(c(0, expiries)), n)
  if (n > 1) D[cbind(2:n, 1:(n - 1))] <- -diag(D)[2:n]
  D
}

# Objective for smoothing the piecewise constant curve: squared moves in w
# plus squared slopes of the forward curve between buckets.
obj.w <- function(expiries, w.in) {
  D <- fwd.diff.matrix(expiries)
  S <- diff(D) / diff(expiries)          # slopes of xi between buckets
  J <- dw.derr(expiries, w.in)
  obj <- function(err.vec) {
    w1 <- adjust.w(expiries, w.in, err.vec)
    1e3 * (sum((w1 - w.in)^2) + sum((S %*% w1)^2))
  }
  attr(obj, "gradient") <- function(err.vec) {
    w1 <- adjust.w(expiries, w.in, err.vec)
    2e3 * J * drop((w1 - w.in) + crossprod(S, S %*% w1))
  }
  obj
}

xi.curve <- function(expiries, w.in, eps = 0) {
  n <- length(w.in)
  obj <- obj.w(expiries, w.in)
  err <- fit.errs.optim(obj, attr(obj, "gradient"), n, eps)
  w.out <- adjust.w(expiries, w.in, err)
  xi.vec <- drop(fwd.diff.matrix(expiries) %*% w.out)

  # xi_i on (T_{i-1}, T_i]; the last value beyond T_n.
  xi.curve.out <- function(t) {
    xi.vec[pmin(findInterval(t, expiries, left.open = TRUE) + 1, n)]
  }
  list(xi.vec = xi.vec, xi.curve = xi.curve.out,
       fit.errs = vol.errs(expiries, w.in, w.out), w.out = w.out)
}

#############################################################################
# Maximally smooth forward variance: Filipovic and Willems (2018),
# as implemented by Rick Cao (2019).
#
# w(t) = sum_j Z_j phi(t, T_j), with kernel
#   phi(t, tau) = 1 - m^3/6 + t tau (2 + m)/2,   m = min(t, tau),
# and Z = A^{-1} w.out, A_ij = phi(T_i, T_j). This is the interpolant of
# minimal norm; xi = w' is piecewise quadratic, constant beyond T_n.
#############################################################################

phi.kernel <- function(t, tau) {
  m <- outer(t, tau, pmin)
  1 - m^3 / 6 + outer(t, tau) * (2 + m) / 2
}

phi.kernel.deriv <- function(t, tau) {        # d phi / dt
  m <- outer(t, tau, pmin)
  rep(tau, each = length(t)) * (1 + m) - m^2 / 2
}

xi.curve.smooth <- function(expiries, w.in, xi = TRUE, eps = 0) {
  n <- length(expiries)
  A.inv <- solve(phi.kernel(expiries, expiries))
  J <- dw.derr(expiries, w.in)
  obj <- function(err) {                 # squared norm of the interpolant
    v <- adjust.w(expiries, w.in, err)
    drop(crossprod(v, A.inv %*% v))
  }
  grad <- function(err) {
    v <- adjust.w(expiries, w.in, err)
    2 * J * drop(A.inv %*% v)
  }
  err <- fit.errs.optim(obj, grad, n, eps)
  w.out <- adjust.w(expiries, w.in, err)
  Z <- drop(A.inv %*% w.out)

  w.curve <- function(t) drop(phi.kernel(t, expiries) %*% Z)
  fwd.curve <- function(t) drop(phi.kernel.deriv(t, expiries) %*% Z)
  list(xi.curve = if (xi) fwd.curve else w.curve,
       w.curve = w.curve,
       fit.errs = vol.errs(expiries, w.in, w.out), w.out = w.out)
}

#############################################################################
# Monotone forward variance: piecewise cubic Hermite (PCHIP) interpolation
# of w through (0, 0), (T_1, w_1), ..., (T_n, w_n).
# The slope at each interior T_i is the weighted harmonic mean of the two
# adjacent piecewise constant forwards (Fritsch-Butland), which keeps the
# interpolant increasing when w is increasing, so xi = w' >= 0.
# At the ends the slopes are the first and last piecewise constant forwards,
# and xi is extrapolated flat beyond T_n. xi is continuous.
# With eps > 0, w.in is first adjusted as in xi.curve.
#############################################################################

pchip.slopes <- function(x, y) {
  h <- diff(x)
  s <- diff(y) / h                     # piecewise constant forwards
  n <- length(s)
  m <- c(s[1], numeric(n - 1), s[n])
  if (n > 1) {
    i <- 2:n                           # interior knots x[i]
    w1 <- 2 * h[i] + h[i - 1]
    w2 <- h[i] + 2 * h[i - 1]
    same.sign <- s[i - 1] * s[i] > 0
    m[i] <- ifelse(same.sign, (w1 + w2) / (w1 / s[i - 1] + w2 / s[i]), 0)
  }
  m
}

xi.curve.mono <- function(expiries, w.in, eps = 0) {
  n <- length(w.in)
  obj <- obj.w(expiries, w.in)
  err <- fit.errs.optim(obj, attr(obj, "gradient"), n, eps)
  w.out <- adjust.w(expiries, w.in, err)
  if (any(diff(w.out) <= 0))
    stop("total variance is not increasing in T (calendar arbitrage): ",
         "increase eps")

  x <- c(0, expiries)
  y <- c(0, w.out)
  m <- pchip.slopes(x, y)
  f <- splinefunH(x, y, m)             # extrapolates linearly beyond T_n
  list(xi.curve = function(t) f(t, deriv = 1),
       w.curve = function(t) f(t),
       fit.errs = vol.errs(expiries, w.in, w.out), w.out = w.out)
}

#############################################################################
# Object wrapper around xi.curve.smooth, kept for compatibility with the
# lectures that use it:
#   obj <- CurveSmoothBuilder(expiries, w.in, eps); obj$fitCurve()
#   xi <- obj$getForwardVarCurve()
#############################################################################

CurveSmoothBuilder <- function(expiries, w.in, eps = 0) {
  fit <- NULL
  me <- list(
    fitCurve = function() {
      fit <<- xi.curve.smooth(expiries, w.in, eps = eps)
      invisible(NULL)
    },
    getForwardVarCurve = function() fit$xi.curve,
    getTotalVarCurve = function(...) fit$w.curve,
    getFitErrors = function() fit$fit.errs,
    getWOut = function() fit$w.out
  )
  class(me) <- append(class(me), "CurveSmoothBuilder")
  me
}
