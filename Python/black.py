import numpy as np
from scipy.stats import norm
from scipy.optimize import root_scalar


def black_price(K, T, F, vol, r=0.0, opttype=1):
    w = T * vol**2
    d1 = np.log(F / K) / w**0.5 + 0.5 * w**0.5
    d2 = d1 - w**0.5
    price = np.exp(-r * T) * (F * norm.cdf(opttype * d1) - K * norm.cdf(opttype * d2))
    return opttype * price


def black_delta(K, T, F, vol, r=0.0, opttype=1):
    w = T * vol**2
    d1 = np.log(F / K) / w**0.5 + 0.5 * w**0.5
    return np.exp(-r * T) * opttype * norm.cdf(opttype * d1)


def black_gamma(K, T, F, vol, r=0.0):
    w = T * vol**2
    d1 = np.log(F / K) / w**0.5 + 0.5 * w**0.5
    return np.exp(-r * T) * norm.pdf(d1) / (F * vol * T**0.5)


def black_vega(K, T, F, vol, r=0.0):
    w = T * vol**2
    d1 = np.log(F / K) / w**0.5 + 0.5 * w**0.5
    return np.exp(-r * T) * F * norm.pdf(d1) * T**0.5


@np.vectorize
def black_impvol_brentq(K, T, F, value, r=0.0, opttype=1):
    if (K <= 0) or (T <= 0) or (F <= 0) or (value <= 0):
        return np.nan

    try:
        result = root_scalar(
            f=lambda vol: black_price(K, T, F, vol, r, opttype) - value,
            bracket=[1e-10, 5.0],
            method="brentq",
        )
        if result.converged:
            return result.root
        else:
            return np.nan
    except ValueError:
        return np.nan


def black_impvol(K, T, F, value, r=0.0, opttype=1, TOL=1e-10, MAX_ITER=1000):
    K = np.atleast_1d(K)
    value = np.atleast_1d(value)
    opttype = np.full_like(K, opttype)

    if K.shape != value.shape:
        raise ValueError("K and value must have the same shape.")

    if np.abs(opttype).any() != 1:
        raise ValueError("opttype must be either 1 or -1.")

    F = float(F)
    T = float(T)
    r = float(r)

    if T <= 0 or F <= 0:
        return np.full_like(K, np.nan)

    low = 1e-10 * np.ones_like(K)
    high = 5.0 * np.ones_like(K)
    mid = 0.5 * (low + high)
    for _ in range(MAX_ITER):
        price = black_price(K, T, F, mid, r, opttype)
        diff = (price - value) / value

        if np.all(np.abs(diff) < TOL):
            return mid

        mask = diff > 0
        high[mask] = mid[mask]
        low[~mask] = mid[~mask]
        mid = 0.5 * (low + high)
    raise ValueError("Implied volatility did not converge")
