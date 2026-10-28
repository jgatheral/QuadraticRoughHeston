import matplotlib.pyplot as plt
import numpy as np
from scipy import special
from scipy.integrate import quad
from scipy.interpolate import PchipInterpolator
from scipy.linalg import hankel, svd
from scipy.optimize import minimize
from scipy.special import gamma
from scipy.stats import norm

from black import black_otm_impvol_mc


def kernel(t, H: float, lam: float):
    """
    Kernel function.
    """
    if lam < 0:
        raise ValueError("lambda must be non-negative")
    return t ** (H - 0.5) * np.exp(-lam * t) / gamma(H + 0.5)


def kernel_exp_approx(
    H: float, lam: float, a: float, b: float, N: int, eps: float = 1e-3
):
    """
    This function implements a numerical approximation technique that represents
    a kernel function as a weighted sum of exponentials. The method constructs
    a Hankel matrix from kernel evaluations, performs SVD decomposition, and
    solves a least squares problem to find optimal weights and decay rates.

    Parameters
    ----------
    H : float
        The Hurst parameter for the kernel function.
    lam : float
        The lambda parameter for the kernel function.
    a : float
        Left endpoint of the approximation interval.
    b : float
        Right endpoint of the approximation interval.
    N : int
        Number of discretization points (2*N+1 points will be used).
    eps : float, optional
        Tolerance parameter for truncating singular values (default: 1e-3).
        Controls the approximation accuracy.

    Returns
    -------
    c_optim_tilde : ndarray
        Optimized weight coefficients for the exponential approximation,
        transformed to the interval [a,b].
    gammas_tilde : ndarray
        Decay rates for the exponential terms, transformed to the interval [a,b].

    Notes
    -----
    The approximation has the form:
        kernel(x) ≈ sum_i c_optim_tilde[i] * exp(-gammas_tilde[i] * x)

    The method follows these steps:
    1. Evaluate kernel on uniform grid and construct Hankel matrix
    2. Perform SVD and truncate based on tolerance eps
    3. Extract roots from the truncated singular vector
    4. Solve least squares problem for optimal weights
    5. Transform coefficients to the target interval [a,b]

    References
    ----------
    Beylkin, G., & Monzon, L. (2005). On approximation of functions by
    exponential sums. Applied and Computational Harmonic Analysis, 19(1), 17-48.
    """
    Ns = np.arange(2 * N + 1)  # {0,...,2N}

    ### Step 1
    # evaluate the kernel on nodes and construct Hankel matrix
    vals = a + (b - a) * Ns / (2 * N)
    kernel_vals = kernel(vals, H=H, lam=lam)
    hankel_matrix = hankel(kernel_vals[: N + 1], kernel_vals[N:])

    # sanity check: shape of the Hankel matrix must be (N+1, N+1)
    assert hankel_matrix.shape == (N + 1, N + 1)

    ### Step 2
    U, S, _ = svd(hankel_matrix)
    # find smallest index m such that sigma_m <= eps * ||kernel_vals||_2
    m = np.argwhere(eps * np.linalg.norm(kernel_vals) >= S)[0, 0]
    U_m = U[:, m]

    ### Step 3
    # Retain real roots in [0, 1).
    roots_m = np.roots(U_m[::-1])
    roots_m = np.real(roots_m[np.isreal(roots_m)])
    roots_01 = roots_m[(roots_m >= 0) & (roots_m < 1)]
    roots_01 = np.sort(roots_01)

    ### Step 4 - least squares for weights
    def fun(c):
        crho = np.sum(c[:, None] * np.power(roots_01[:, None], Ns[None, :]), axis=0)
        l2_error = np.sum((kernel_vals - crho) ** 2)
        return l2_error

    c_optim = minimize(fun, x0=1 * np.ones(m)).x
    c_optim = np.real(c_optim)

    ### Step 5
    gammas = -2 * N * np.log(roots_01)

    # transform to interval [a,b]
    gammas_tilde = gammas / (b - a)
    c_optim_tilde = c_optim * np.exp(gammas * a / (b - a))

    return c_optim_tilde, gammas_tilde


def gauss_legendre(a: float, b: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the Gauss-Legendre quadrature points and weights on the interval [a, b].

    Parameters
    ----------
    a : float
        Lower bound of the integration interval.
    b : float
        Upper bound of the integration interval.
    n : int
        Number of quadrature points.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing two 1-D arrays:
        - Quadrature points on [a, b].
        - Quadrature weights on [a, b].
    """
    knots, weights = np.polynomial.legendre.leggauss(n)
    knots_a_b = 0.5 * (b - a) * knots + 0.5 * (b + a)
    weights_a_b = 0.5 * (b - a) * weights
    return knots_a_b, weights_a_b


def plot_ivols_mc(
    ivol_data,
    slices=None,
    mc_matrix=None,
    plot=True,
    colnum=None,
    return_fig=False,
    return_error_metrics=False,
):
    """Plot implied vols."""

    bid_vols = ivol_data["Bid"].astype(float)
    ask_vols = ivol_data["Ask"].astype(float)
    exp_dates = np.unique(ivol_data["Texp"])

    if slices is not None:
        slices = list(slices)
    else:
        slices = list(range(len(exp_dates)))

    selected_exp_dates = exp_dates[slices]
    n_slices = len(slices)

    if colnum is None:
        colnum = int(np.sqrt(n_slices * 2))

    rows = int(np.round(colnum / 2))
    columns = int(np.round(colnum))

    while rows * columns < n_slices:
        rows += 1

    atm_vols = np.zeros(n_slices)
    atm_skew = np.zeros(n_slices)
    atm_curv = np.zeros(n_slices)
    atm_vols_mc = np.zeros(n_slices)
    atm_skew_mc = np.zeros(n_slices)
    atm_err_mc = np.zeros(n_slices)
    if return_error_metrics:
        rmse_bid_ask_mc = np.full(n_slices, np.nan)
        mae_bid_ask_mc = np.full(n_slices, np.nan)
        median_bid_ask_mc = np.full(n_slices, np.nan)

    fig, axes = plt.subplots(rows, columns, figsize=(15, 10))
    axes = np.atleast_1d(axes).flatten()

    for i, slice_idx in enumerate(slices):
        t = exp_dates[slice_idx]
        texp = ivol_data["Texp"]
        bid_vol = bid_vols[texp == t]
        ask_vol = ask_vols[texp == t]
        mid_vol = (bid_vol + ask_vol) / 2
        f = ivol_data["Fwd"][texp == t].iloc[0]
        k = np.log(ivol_data["Strike"][texp == t] / f)
        include = ~np.isnan(bid_vol) & (bid_vol > 0)
        kmin = np.min(k[include])
        kmax = np.max(k[include])
        ybottom = 0.6 * np.min(bid_vol[include])
        ytop = 1.2 * np.max(ask_vol[include])
        # Pad by 4% on each side, like R's default axis expansion.
        kpad = 0.04 * (kmax - kmin)
        xrange = [kmin - kpad, kmax + kpad]
        yrange = [ybottom, ytop]

        if plot:
            ax = axes[i]
            ax.plot(k, bid_vol, "ro", markersize=3, label="Bid Vol")
            ax.plot(k, ask_vol, "bo", markersize=3, label="Ask Vol")
            ax.axvline(0, linestyle="--", color="gray")
            ax.axhline(0, linestyle="--", color="gray")
            ax.set_xlim(xrange)
            ax.set_ylim(yrange)
            ax.set_title(f"T = {t:.5f}", fontweight="bold")
            ax.set_xlabel("Log-Strike")
            ax.set_ylabel("Implied Vol.")

        if mc_matrix is not None:

            def vol_mc(k):
                return black_otm_impvol_mc(S=mc_matrix[slice_idx, :], k=k, T=t)

            atm_vols_mc[i] = np.asarray(vol_mc(0.0)).item()
            sig_mc = atm_vols_mc[i] * np.sqrt(t)
            sig_mc_right = np.asarray(vol_mc(sig_mc / 10.0)).item()
            sig_mc_left = np.asarray(vol_mc(-sig_mc / 10.0)).item()
            atm_skew_mc[i] = (sig_mc_right - sig_mc_left) / (2.0 * sig_mc / 10.0)

            def err_mc(k):
                return black_otm_impvol_mc(
                    S=mc_matrix[slice_idx, :], k=k, T=t, mc_error=True
                )["error_95"]

            atm_err_mc[i] = np.asarray(err_mc(0)).item()
            if return_error_metrics:
                bid_ask_width = ask_vol - bid_vol
                vol_model = vol_mc(k)
                normalized_errors = (vol_model - mid_vol) / bid_ask_width
                valid = (
                    np.isfinite(normalized_errors)
                    & np.isfinite(bid_ask_width)
                    & (bid_ask_width > 0)
                )
                if np.any(valid):
                    errors = normalized_errors[valid]
                    rmse_bid_ask_mc[i] = np.sqrt(np.mean(errors**2))
                    mae_bid_ask_mc[i] = np.mean(np.abs(errors))
                    median_bid_ask_mc[i] = np.median(np.abs(errors))
            if plot:
                k_vals = np.linspace(kmin, kmax, 100)
                vol_mc_vals = np.array([vol_mc(k) for k in k_vals])
                ax.plot(k_vals, vol_mc_vals, "green", linewidth=2, label="MC")

        k_in = k[~np.isnan(mid_vol)]
        vol_in = mid_vol[~np.isnan(mid_vol)]
        vol_interp = PchipInterpolator(k_in, vol_in)
        atm_vols[i] = vol_interp(0)
        sig = atm_vols[i] * np.sqrt(t)
        atm_skew[i] = (vol_interp(sig / 10) - vol_interp(-sig / 10)) / (
            2 * sig / 10
        )
        atm_curv[i] = (
            vol_interp(sig / 10) + vol_interp(-sig / 10) - 2 * atm_vols[i]
        ) / (2 * (sig / 10) ** 2)

    if plot:
        for i in range(n_slices, len(axes)):
            fig.delaxes(axes[i])
        plt.tight_layout()
        plt.show()

    res = {
        "expiries": selected_exp_dates,
        "atm_vols": atm_vols,
        "atm_skew": atm_skew,
        "atm_curv": atm_curv,
        "atm_vols_mc": atm_vols_mc,
        "atm_skew_mc": atm_skew_mc,
        "atm_err_mc": atm_err_mc,
    }
    if return_error_metrics:
        res["rmse_bid_ask_mc"] = rmse_bid_ask_mc
        res["mae_bid_ask_mc"] = mae_bid_ask_mc
        res["median_bid_ask_mc"] = median_bid_ask_mc
    if return_fig:
        res["fig"] = fig
    return res


def var_swap_robust(ivol_data, slices=None):
    """Estimate variance swaps for selected expiry indices, in selection order."""
    ivol_data = ivol_data.dropna()
    bid_vols = ivol_data["Bid"].astype(float)
    ask_vols = ivol_data["Ask"].astype(float)
    exp_dates = np.sort(np.unique(ivol_data["Texp"]))
    n_slices = len(exp_dates)

    if slices is not None:
        slices = list(slices)
        n_slices = len(slices)
    else:
        slices = range(n_slices)

    vs_mid = np.zeros(n_slices)
    vs_bid = np.zeros(n_slices)
    vs_ask = np.zeros(n_slices)

    def varswap(k_in, vol_series, slice_idx):
        t = exp_dates[slice_idx]
        sig_in = vol_series * np.sqrt(t)
        zm_in = -k_in / sig_in - sig_in / 2
        y_in = norm.cdf(zm_in)
        ord_y_in = np.argsort(y_in)
        sig_in_y = sig_in[ord_y_in]
        y_min = np.min(y_in)
        y_max = np.max(y_in)
        sig_in_0 = sig_in_y[0]
        sig_in_1 = sig_in_y[-1]

        wbar_flat = quad(PchipInterpolator(np.sort(y_in), sig_in_y**2), y_min, y_max)[0]
        res_mid = wbar_flat
        z_minus = zm_in[ord_y_in][0]
        res_lh = sig_in_0**2 * norm.cdf(z_minus)
        z_plus = zm_in[ord_y_in][-1]
        res_rh = sig_in_1**2 * norm.cdf(-z_plus)

        res_vs = res_mid + res_lh + res_rh
        return res_vs

    for i, slice_idx in enumerate(slices):
        t = exp_dates[slice_idx]
        texp = ivol_data["Texp"]
        bid_vol = bid_vols[texp == t].to_numpy()
        ask_vol = ask_vols[texp == t].to_numpy()
        mid_vol = (bid_vol + ask_vol) / 2
        F = ivol_data["Fwd"][texp == t].iloc[0]  # forward price
        k = np.log(ivol_data["Strike"][texp == t].to_numpy() / F)  # log-fwd moneyness
        vs_mid[i] = varswap(k, mid_vol, slice_idx) / t
        vs_bid[i] = varswap(k, bid_vol, slice_idx) / t
        vs_ask[i] = varswap(k, ask_vol, slice_idx) / t

    return {
        "expiries": exp_dates[list(slices)],
        "vs_mid": vs_mid,
        "vs_bid": vs_bid,
        "vs_ask": vs_ask,
    }

def y0_from_xi0(u, xi_smooth, w_jac, x_jac, params):
    """
    Compute y0(u) from initial variance curve xi0(u).

    Parameters
    ----------
    u : array_like
        Time points where u >= 0.

    Returns
    -------
    ndarray
        y0(u) values at u.
    """
    u = np.atleast_1d(np.asarray(u, dtype=float))
    integral = np.zeros_like(u)
    mask = u > 0
    H = params["al"] - 0.5

    integral[mask] = (
        w_jac[:, None]
        * xi_smooth(0.5 * u[mask][None, :] * (1 + x_jac[:, None]))
        * np.exp(-params["lam"] * u[mask][None, :] * (1 - x_jac[:, None]))
    ).sum(axis=0)
    integral[mask] *= (params["nu"] / special.gamma(params["al"])) ** 2 * (
        0.5 * u[mask]
    ) ** (2.0 * H)
    return (xi_smooth(u) - integral - params["c"]) ** 0.5
