######################################################################
#
# Jim Gatheral, March 2025
#
######################################################################


######################################################################
#  QRH.sim implements the QRH scheme.
######################################################################
QRH.sim <- function(params, xi, n_quad = 20) {
  function(paths, steps, expiries, output = "all", delvix = 1 / 12, nvix = 10,
           seed = NULL) {
    if (!is.null(seed)) set.seed(seed)
    library(gsl)
    c <- params$c
    H <- params$al - 1 / 2

    v0 <- xi(0)
    ey <- yFromXi(params, xi, n_quad)
    bigK0p <- bigK0(params)
    K00p <- K00(params)
    K0p <- K0(params)

    sim <- function(expiry) {
      dt <- expiry / steps
      K0del <- K0p(dt)
      K00del <- K00p(dt)
      bigK0del <- bigK0p(dt)
      tj <- (1:steps) * dt
      yj <- ey(tj)
      K00j <- c(0, K00p(tj))
      bstar <- sqrt(diff(K00j) / dt)
      chi <- array(0, dim = c(steps, paths))
      v <- rep(v0, paths)
      Y <- rep(ey(0), paths)
      yhat <- rep(yj[1], paths)
      rho.uchi <- K0del / sqrt(K00del * dt)
      beta.uchi <- K0del / dt
      X <- numeric(paths)
      w <- numeric(paths)
      for (j in 1:steps) {
        Z.eps <- rnorm(paths)
        Z.chi <- rnorm(paths)
        alp <- 1 / (2 * H + 1)
        varu <- bigK0del * (alp * yhat^2 + (1 - alp) * Y^2 +
          c)
        vbar <- varu / K00del
        sig.chi <- sqrt(vbar * dt)
        sig.eps <- sqrt(vbar * K00del * (1 - rho.uchi^2))
        chi[j, ] <- sig.chi * Z.chi
        eps <- sig.eps * Z.eps
        u <- beta.uchi * chi[j, ] + eps
        Y <- yhat + u
        vf <- Y^2 + c
        dw <- (v + vf) / 2 * dt
        w <- w + dw
        X <- X - dw / 2 - chi[j, ]
        if (j < steps) {
          btilde <- rev(bstar[2:(j + 1)])
          yhat <- yj[j + 1] + as.numeric(btilde %*% chi[1:j, , drop = FALSE])
        }
        v <- vf
      }
      vix2 <- 0
      ds <- delvix / nvix
      if ((output == "vix") || (output == "all")) {
        for (k in 1:nvix) {
          tk <- expiry + k * ds
          Ku <- c(K00p(tk), K00p(tk - tj))
          ck.vec <- sqrt(-diff(Ku) / dt)
          dyTu <- as.numeric(ck.vec %*% chi)
          yTu <- ey(tk) + dyTu
          vix2 <- vix2 + (yTu^2 + c) * (1 + bigK0p((nvix -
            k) * ds)) / nvix
        }
        vix2 <- vix2 + v * (1 + bigK0p(delvix)) / (2 * nvix) -
          (yTu^2 + c) / (2 * nvix)
        vix <- sqrt(vix2)
      }
      res.sim <- switch(output,
        v = v,
        X = X,
        w = w,
        vix = vix,
        all = list(v = v, X = X, w = w, vix = vix)
      )
      return(res.sim)
    }
    sim.out <- NULL
    if (output != "all") {
      sim.out <- do.call(rbind, lapply(expiries, sim))
    }
    # "all" returns one named list of output vectors per expiry.
    else {
      for (j in 1:length(expiries)) {
        sim.out[[j]] <- sim(expiries[j])
      }
    }
    return(sim.out)
  }
}


######################################################################
# QRH.blip runs the QRH scheme with both the base and blipped forward
# volatility curves, enabling the computation of the SSR.
# h is the signed BLIP coefficient: h(T) = h * sqrt(T). A positive .01
# subtracts .01 * sqrt(T) * kernel(u) from y0(u), as in Python.
#######################################################################

QRH.blip <- function(params, xi, h = 0.01, n_quad = 20) {
  function(paths, steps, expiries, seed = NULL) {
    if (!is.null(seed)) set.seed(seed)
    library(gsl)
    c <- params$c
    H <- params$al - 1 / 2

    kp <- kGamma(params)
    bigK0p <- bigK0(params)
    K00p <- K00(params)
    K0p <- K0(params)
    nn <- length(expiries)
    res.matrix <- array(NA, dim = c(nn * 8, paths))

    v0 <- xi(0)
    ey <- yFromXi(params, xi, n_quad)

    for (i in 1:nn) {
      expiry <- expiries[i]
      dt <- expiry / steps
      K0del <- K0p(dt)
      K00del <- K00p(dt)
      bigK0del <- bigK0p(dt)
      tj <- (1:steps) * dt
      K00j <- c(0, K00p(tj))
      bstar <- sqrt(diff(K00j) / dt)
      rho.uchi <- K0del / sqrt(K00del * dt)
      beta.uchi <- K0del / dt

      Y <- rep(ey(0), paths)
      w <- numeric(paths)
      v <- rep(v0, paths)
      Chi <- numeric(paths) # Chi is int_0^t \sqrt(V_s) dW_s
      yj <- ey(tj)
      yhat <- rep(yj[1], paths)
      chi <- array(0, dim = c(steps, paths))

      ey.h <- function(s) {
        ey(s) - h * sqrt(expiry) * kp(s)
      } # Blip the forward y curve by h * sqrt(expiry) * kernel(s).

      Y.h <- rep(ey(0), paths)
      w.h <- numeric(paths)
      v.h <- rep(v0, paths)
      Chi.h <- numeric(paths)
      yj.h <- ey.h(tj)
      yhat.h <- rep(yj.h[1], paths)
      chi.h <- array(0, dim = c(steps, paths))

      for (j in 1:steps) {
        Z.eps <- rnorm(paths)
        Z.chi <- rnorm(paths)
        alp <- 1 / (2 * H + 1)
        varu <- bigK0del * (alp * yhat^2 + (1 - alp) * Y^2 + c)
        vbar <- varu / K00del
        sig.chi <- sqrt(vbar * dt)
        sig.eps <- sqrt(vbar * K00del * (1 - rho.uchi^2))
        chi[j, ] <- sig.chi * Z.chi
        eps <- sig.eps * Z.eps
        u <- beta.uchi * chi[j, ] + eps
        Y <- yhat + u
        vf <- Y^2 + c
        dw <- (v + vf) / 2 * dt
        w <- w + dw
        Chi <- Chi + chi[j, ] # This is int_0^T \sqrt{V_t} \dW_t
        if (j < steps) {
          btilde <- rev(bstar[2:(j + 1)])
          yhat <- yj[j + 1] + as.numeric(btilde %*% chi[1:j, , drop = FALSE])
        }
        v <- vf

        # Repeat for blipped curve
        alp <- 1 / (2 * H + 1)
        varu.h <- bigK0del * (alp * yhat.h^2 + (1 - alp) * Y.h^2 + c)
        vbar.h <- varu.h / K00del
        sig.chi <- sqrt(vbar.h * dt)
        sig.eps <- sqrt(vbar.h * K00del * (1 - rho.uchi^2))
        chi.h[j, ] <- sig.chi * Z.chi
        eps <- sig.eps * Z.eps
        u <- beta.uchi * chi.h[j, ] + eps
        Y.h <- yhat.h + u
        vf.h <- Y.h^2 + c
        dw.h <- (v.h + vf.h) / 2 * dt
        w.h <- w.h + dw.h
        Chi.h <- Chi.h + chi.h[j, ]
        if (j < steps) {
          yhat.h <- yj.h[j + 1] + as.numeric(btilde %*% chi.h[1:j, , drop = FALSE])
        }
        v.h <- vf.h
      }

      res.matrix[8 * (i - 1) + 1, ] <- Y
      res.matrix[8 * (i - 1) + 2, ] <- w
      res.matrix[8 * (i - 1) + 3, ] <- v
      res.matrix[8 * (i - 1) + 4, ] <- Chi
      res.matrix[8 * (i - 1) + 5, ] <- Y.h
      res.matrix[8 * (i - 1) + 6, ] <- w.h
      res.matrix[8 * (i - 1) + 7, ] <- v.h
      res.matrix[8 * (i - 1) + 8, ] <- Chi.h
    }
    return(res.matrix)
  }
}
