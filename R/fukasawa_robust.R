# Magic strike nonparametric estimates of option strips

#############################################################################
# Robust variance swap
#############################################################################

varSwap.Robust <- function(ivolData, slices = NULL) {
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
    y.min <- min(yIn)
    y.max <- max(yIn)
    sigIn.0 <- sigIn.y[1]
    sigIn.1 <- tail(sigIn.y, 1)

    integVar <- function(yOut) {
      stinterp(x = sort(yIn), y = sigIn.y^2, yOut)$y
    }

    wbar.flat <- integrate(integVar, lower = y.min, upper = y.max)$value
    res.mid <- wbar.flat
    z.minus <- (zm.In[ord.yIn])[1]
    res.lh <- sigIn.0^2 * pnorm(z.minus)
    z.plus <- tail(zm.In[ord.yIn], 1)
    res.rh <- sigIn.1^2 * pnorm(-z.plus)

    res.vs <- res.mid + res.lh + res.rh
    return(res.vs)
  } # End of varswap function

  for (slice in slices) {
    t <- expDates[slice]
    texp <- ivolData$Texp
    bidVol <- bidVols[texp == t]
    askVol <- askVols[texp == t]
    midVol <- (bidVol + askVol) / 2
    f <- (ivolData$Fwd[texp == t])[1]
    k <- log(ivolData$Strike[texp == t] / f)
    include <- !is.na(midVol)
    kIn <- k[include]
    vs.mid[slice] <- varswap(kIn, midVol, include, slice) / t
    vs.bid[slice] <- varswap(kIn, bidVol, include, slice) / t
    vs.ask[slice] <- varswap(kIn, askVol, include, slice) / t
  }
  return(list(
    expiries = expDates, vs.mid = vs.mid, vs.bid = vs.bid,
    vs.ask = vs.ask
  ))
}
