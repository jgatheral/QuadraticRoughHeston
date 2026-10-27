# Shared helpers for the concise R notebooks. Source with R/ as working directory.
source("gamma_kernel.R")
source("qrh_gamma_y_from_xi.R")
source("qrh_simulation.R")
source("black_scholes.R")
source("fwd_var_curve.R")
source("fukasawa_robust.R")

calibrations <- read.csv("../data/calibration_params.csv", colClasses = c(date = "character"))
paper.dates <- calibrations$date
paper.colors <- c("#1e732b", "#6b8203", "#b38a16", "#ed8e6e")

paper.inputs <- function(date, eps = .03) {
  spx <- read.csv(paste0("../data/", date, "_spx_vol.csv"))
  vix <- read.csv(paste0("../data/", date, "_vix_vol.csv"))
  swaps <- varSwap.Robust(spx)
  xi <- xi.curve.smooth(swaps$expiries, swaps$vs.mid * swaps$expiries, eps = eps)$xi.curve
  params <- as.list(calibrations[calibrations$date == date, c("al", "lam", "nu", "c")])
  list(spx = spx, vix = vix, params = params, xi = xi)
}

paper.save <- function(draw, output, name, width = 8, height = 5) {
  dir.create(output, recursive = TRUE, showWarnings = FALSE)
  for (suffix in c("pdf", "eps")) {
    path <- file.path(output, paste0(name, ".", suffix))
    if (suffix == "pdf") {
      pdf(path, width = width, height = height)
    } else {
      postscript(path,
        width = width, height = height, horizontal = FALSE,
        onefile = FALSE, paper = "special"
      )
    }
    tryCatch(draw(), finally = dev.off())
  }
}

paper.forward <- function(eps = .03) {
  u <- seq(0, .2, length.out = 101)
  curves <- data.frame(u = u)
  for (date in paper.dates) curves[[date]] <- paper.inputs(date, eps)$xi(u)
  curves
}

paper.plot.forward <- function(curves) {
  matplot(curves$u, curves[paper.dates],
    type = "l", lty = 1, lwd = 2,
    col = paper.colors, xlab = "Maturity u", ylab = expression(xi[0](u)), ylim = c(0, .15)
  )
  legend("topright", paper.dates, col = paper.colors, lty = 1, lwd = 2)
}

# Price OTM options using the sample forward, as in Python/Black.
paper.iv <- function(spot, k, t) {
  forward <- mean(spot)
  strikes <- forward * exp(k)
  vapply(strikes, function(strike) {
    if (strike >= forward) {
      BSImpliedVolCall(forward, strike, t, 0, mean(pmax(spot - strike, 0)))
    } else {
      BSImpliedVolPut(forward, strike, t, 0, mean(pmax(strike - spot, 0)))
    }
  }, numeric(1))
}

paper.smiles <- function(paths, steps, seed = 1234, eps = .03) {
  panels <- list()
  for (date in paper.dates) {
    input <- paper.inputs(date, eps)
    expiries <- head(sort(unique(input$vix$Texp)), 4)
    set.seed(seed)
    sim <- QRH.sim(input$params, input$xi, n_quad = 40)(
      paths, steps, expiries, output = "all", nvix = 40, delvix = 30 / 365)
    for (asset in c("SPX", "VIX")) {
      quotes <- input[[tolower(asset)]]
      for (i in seq_along(expiries)) {
        t <- expiries[i]
        d <- quotes[quotes$Texp == t & !is.na(quotes$Bid) & !is.na(quotes$Ask), ]
        spot <- if (asset == "SPX") exp(sim[[i]]$X) else sim[[i]]$vix
        d$k <- log(d$Strike / d$Fwd)
        d$model <- paper.iv(spot, d$k, t)
        d$date <- date
        d$asset <- asset
        panels[[length(panels) + 1]] <- d
      }
    }
  }
  do.call(rbind, panels)
}

paper.plot.smiles <- function(data) {
  old <- par(mfrow = c(8, 4), mar = c(3, 3, 2, 1), mgp = c(1.7, .6, 0))
  on.exit(par(old))
  for (date in paper.dates) {
    expiries <- sort(unique(data$Texp[data$date == date]))
    for (pair in list(1:2, 3:4)) {
      for (asset in c("SPX", "VIX")) {
        for (i in pair) {
          t <- expiries[i]
          d <- data[data$date == date & data$asset == asset & data$Texp == t, ]
          d <- d[order(d$k), ]
          plot(d$k, d$Bid,
            col = "red", pch = 24, cex = .4,
            ylim = range(c(d$Bid, d$Ask, d$model), finite = TRUE),
            main = paste(date, asset, "T =", round(t, 3)),
            xlab = "Log-strike", ylab = "Implied vol."
          )
          points(d$k, d$Ask, col = "blue", pch = 25, cex = .4)
          lines(d$k, d$model, col = "green4")
          abline(v = 0, col = "gray", lty = 2)
        }
      }
    }
  }
}

# Signed finite difference and ATM-forward digital skew, matching Python.
paper.ssr <- function(paths, steps, seed = 1234, eps = .03, h = 0.01) {
  expiries <- 2^((-13:0) / 2)
  rows <- list()
  for (date in paper.dates) {
    input <- paper.inputs(date, eps)
    set.seed(seed)
    values <- vapply(expiries, function(t) {
      z <- QRH.blip(input$params, input$xi, h, n_quad = 40)(paths, steps, t)
      spot <- exp(-z[4, ] - z[2, ] / 2)
      shifted <- exp(-z[8, ] - z[6, ] / 2)
      sigma <- paper.iv(spot, 0, t)
      beta <- (paper.iv(shifted, 0, t) - sigma) / (h * sqrt(t))
      d2 <- -sigma * sqrt(t) / 2
      skew <- (pnorm(d2) - mean(spot >= mean(spot))) / (sqrt(t) * dnorm(d2))
      beta / skew
    }, numeric(1))
    stopifnot(all(is.finite(values)))
    rows[[date]] <- data.frame(
      date = date, expiry = c(0, expiries),
      ssr = c(input$params$al + 1, values)
    )
  }
  do.call(rbind, rows)
}

paper.plot.ssr <- function(data) {
  plot(range(data$expiry), range(data$ssr), type = "n", xlab = "Expiry T", ylab = "SSR")
  for (i in seq_along(paper.dates)) {
    d <- data[data$date == paper.dates[i], ]
    lines(d$expiry, d$ssr, type = "b", pch = 20, cex = .5, col = paper.colors[i])
  }
  legend("topright", paper.dates, col = paper.colors, lty = 1, pch = 20)
}
