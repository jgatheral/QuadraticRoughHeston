# Magic strike nonparametric estimates of option strips

#############################################################################
# Robust variance swap
#############################################################################

# Piecewise cubic Hermite interpolation, matching SciPy's PCHIP.
pchip <- function(x, y) {
  n <- length(x)
  h <- diff(x)
  stopifnot(n >= 2, length(y) == n, all(h > 0))
  slope <- diff(y) / h
  if (n == 2) return(splinefunH(x, y, rep(slope, 2)))
  derivative <- numeric(n)
  for (i in 2:(n - 1)) {
    if (slope[i - 1] * slope[i] > 0) {
      w1 <- 2 * h[i] + h[i - 1]
      w2 <- h[i] + 2 * h[i - 1]
      derivative[i] <- (w1 + w2) / (w1 / slope[i - 1] + w2 / slope[i])
    }
  }
  edge <- function(h1, h2, m1, m2) {
    d <- ((2 * h1 + h2) * m1 - h1 * m2) / (h1 + h2)
    if (sign(d) != sign(m1)) return(0)
    if (sign(m1) != sign(m2) && abs(d) > 3 * abs(m1)) return(3 * m1)
    d
  }
  derivative[1] <- edge(h[1], h[2], slope[1], slope[2])
  derivative[n] <- edge(h[n - 1], h[n - 2], slope[n - 1], slope[n - 2])
  splinefunH(x, y, derivative)
}

varSwap.Robust <- function(ivolData, slices = NULL) {
  ivolData <- ivolData[complete.cases(ivolData), ]
  bidVols <- as.numeric(ivolData$Bid)
  askVols <- as.numeric(ivolData$Ask)
  expDates <- sort(unique(ivolData$Texp))
  nSlices <- length(expDates)
  if (!is.null(slices)) {
    nSlices <- length(slices)
  } else {
    slices <- 1:nSlices
  }
  vs.mid <- numeric(nSlices)
  vs.bid <- numeric(nSlices)
  vs.ask <- numeric(nSlices)

  varswap <- function(kIn, vol.series, include, slice) {
    t <- expDates[slice]
    volIn <- vol.series[include]
    sigIn <- volIn * sqrt(t)
    zm.In <- -kIn / sigIn - sigIn / 2
    yIn <- pnorm(zm.In)
    ord.yIn <- order(yIn)
    sigIn.y <- sigIn[ord.yIn]
    sigIn.0 <- sigIn.y[1]
    sigIn.1 <- tail(sigIn.y, 1)

    wbar.flat <- integrate(pchip(sort(yIn), sigIn.y^2), min(yIn), max(yIn),
                          subdivisions = 50, rel.tol = 1.49e-8, abs.tol = 1.49e-8,
                          stop.on.error = FALSE)$value
    res.mid <- wbar.flat
    z.minus <- (zm.In[ord.yIn])[1]
    res.lh <- sigIn.0^2 * pnorm(z.minus)
    z.plus <- tail(zm.In[ord.yIn], 1)
    res.rh <- sigIn.1^2 * pnorm(-z.plus)

    res.vs <- res.mid + res.lh + res.rh
    return(res.vs)
  } # End of varswap function

  for (i in seq_along(slices)) {
    slice <- slices[i]
    t <- expDates[slice]
    texp <- ivolData$Texp
    bidVol <- bidVols[texp == t]
    askVol <- askVols[texp == t]
    midVol <- (bidVol + askVol) / 2
    f <- (ivolData$Fwd[texp == t])[1]
    k <- log(ivolData$Strike[texp == t] / f)
    include <- !is.na(midVol)
    kIn <- k[include]
    vs.mid[i] <- varswap(kIn, midVol, include, slice) / t
    vs.bid[i] <- varswap(kIn, bidVol, include, slice) / t
    vs.ask[i] <- varswap(kIn, askVol, include, slice) / t
  }
  return(list(
    expiries = expDates[slices], vs.mid = vs.mid, vs.bid = vs.bid,
    vs.ask = vs.ask
  ))
}
