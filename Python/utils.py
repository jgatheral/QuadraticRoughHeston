import numpy as np
from scipy.stats import norm
from scipy.integrate import quad
from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")


def plot_ivols_mc(ivol_data, slices=None, mc_matrix=None, plot=True, colnum=None):
    """Plot implied vols."""

    bid_vols = ivol_data["Bid"].astype(float)
    ask_vols = ivol_data["Ask"].astype(float)
    exp_dates = np.unique(ivol_data["Texp"])
    n_slices = len(exp_dates)

    if slices is not None:
        n_slices = len(slices)
    else:
        slices = range(n_slices)

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
    atm_curv_mc = np.zeros(n_slices)
    atm_err_mc = np.zeros(n_slices)

    fig, axes = plt.subplots(rows, columns, figsize=(15, 10))
    axes = axes.flatten()

    for i, slice in enumerate(slices):
        t = exp_dates[slice]
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
        xrange = [kmin, kmax]
        yrange = [ybottom, ytop]

        if plot:
            ax = axes[i]
            ax.plot(k, bid_vol, "ro", markersize=3, label="Bid Vol")
            ax.plot(k, ask_vol, "bo", markersize=3, label="Ask Vol")
            ax.axvline(0, linestyle="--", color="gray")
            ax.axhline(0, linestyle="--", color="gray")
            ax.set_xlim(xrange)
            ax.set_ylim(yrange)
            ax.set_title(f"T = {t:.5f}")
            ax.set_xlabel("Log-Strike")
            ax.set_ylabel("Implied Vol.")

        if mc_matrix is not None:
            spots = mc_matrix[slice, :]
            s0 = np.mean(spots)

            def vol_mc(k):
                return bs_out(spots, t, s0 * np.exp(k))["BSV"]

            atm_vols_mc[slice] = vol_mc(0)
            sig_mc = vol_mc(0) * np.sqrt(t)
            atm_skew_mc[slice] = (vol_mc(sig_mc / 10) - vol_mc(-sig_mc / 10)) / (
                2 * sig_mc / 10
            )
            atm_curv_mc[slice] = (
                vol_mc(sig_mc / 10) + vol_mc(-sig_mc / 10) - 2 * vol_mc(0)
            ) / (2 * (sig_mc / 10) ** 2)

            def err_mc(k):
                return bs_out(spots, t, s0 * np.exp(k))["err"]

            atm_err_mc[slice] = err_mc(0)
            if plot:
                k_vals = np.linspace(kmin, kmax, 100)
                vol_mc_vals = [vol_mc(k) for k in k_vals]
                ax.plot(k_vals, vol_mc_vals, "orange", linewidth=2, label="MC")

        k_in = k[~np.isnan(mid_vol)]
        vol_in = mid_vol[~np.isnan(mid_vol)]
        vol_interp = PchipInterpolator(k_in, vol_in)
        atm_vols[slice] = vol_interp(0)
        sig = atm_vols[slice] * np.sqrt(t)
        atm_skew[slice] = (vol_interp(sig / 10) - vol_interp(-sig / 10)) / (
            2 * sig / 10
        )
        atm_curv[slice] = (
            vol_interp(sig / 10) + vol_interp(-sig / 10) - 2 * atm_vols[slice]
        ) / (2 * (sig / 10) ** 2)

    if plot:
        for i in range(n_slices, len(axes)):
            fig.delaxes(axes[i])
        plt.tight_layout()
        plt.show()

    return {
        "expiries": exp_dates,
        "atm_vols": atm_vols,
        "atm_skew": atm_skew,
        "atm_curv": atm_curv,
        "atm_vols_mc": atm_vols_mc,
        "atm_skew_mc": atm_skew_mc,
        "atm_err_mc": atm_err_mc,
    }


def var_swap_robust(ivol_data, slices=None):
    """Robust estimation of variance swap quotes."""
    ivol_data = ivol_data.dropna()
    bid_vols = ivol_data["Bid"].astype(float)
    ask_vols = ivol_data["Ask"].astype(float)
    exp_dates = np.sort(np.unique(ivol_data["Texp"]))
    n_slices = len(exp_dates)

    if slices is not None:
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

        def integ_var(y_out):
            interp = PchipInterpolator(np.sort(y_in), sig_in_y**2)
            return interp(y_out)

        wbar_flat = quad(integ_var, y_min, y_max)[0]
        res_mid = wbar_flat
        z_minus = zm_in[ord_y_in][0]
        res_lh = sig_in_0**2 * norm.cdf(z_minus)
        z_plus = zm_in[ord_y_in][-1]
        res_rh = sig_in_1**2 * norm.cdf(-z_plus)

        res_vs = res_mid + res_lh + res_rh
        return res_vs

    for slice_idx in slices:
        t = exp_dates[slice_idx]
        texp = ivol_data["Texp"]
        bid_vol = bid_vols[texp == t].to_numpy()
        ask_vol = ask_vols[texp == t].to_numpy()
        mid_vol = (bid_vol + ask_vol) / 2
        F = ivol_data["Fwd"][texp == t].iloc[0]  # forward price
        k = np.log(ivol_data["Strike"][texp == t].to_numpy() / F)  # log-fwd moneyness
        vs_mid[slice_idx] = varswap(k, mid_vol, slice_idx) / t
        vs_bid[slice_idx] = varswap(k, bid_vol, slice_idx) / t
        vs_ask[slice_idx] = varswap(k, ask_vol, slice_idx) / t

    return {
        "expiries": exp_dates,
        "vs_mid": vs_mid,
        "vs_bid": vs_bid,
        "vs_ask": vs_ask,
    }


def gamma_swap_robust(ivol_data, slices=None):
    """Robust estimation of gamma swap quotes."""
    bid_vols = ivol_data["Bid"].astype(float)
    ask_vols = ivol_data["Ask"].astype(float)
    exp_dates = np.sort(np.unique(ivol_data["Texp"]))
    n_slices = len(exp_dates)

    if slices is not None:
        n_slices = len(slices)
    else:
        slices = range(n_slices)

    gs_mid = np.zeros(n_slices)
    gs_bid = np.zeros(n_slices)
    gs_ask = np.zeros(n_slices)

    def gammaswap(k_in, vol_series, include, slice_idx):
        t = exp_dates[slice_idx]
        vol_in = vol_series[include]
        sig_in = vol_in * np.sqrt(t)
        zm_in = -k_in / sig_in - sig_in / 2
        zp_in = zm_in + sig_in
        y_in = norm.cdf(zp_in)
        ord_y_in = np.argsort(y_in)
        sig_in_y = sig_in[ord_y_in]
        y_min = np.min(y_in)
        y_max = np.max(y_in)
        sig_in_0 = sig_in_y[0]
        sig_in_1 = sig_in_y[-1]

        def integ_var(y_out):
            interp = PchipInterpolator(np.sort(y_in), sig_in_y**2)
            return interp(y_out)

        res_mid = quad(integ_var, y_min, y_max)[0]
        z_minus = zp_in[ord_y_in][0]
        res_lh = sig_in_0**2 * norm.cdf(z_minus)
        z_plus = zp_in[ord_y_in][-1]
        res_rh = sig_in_1**2 * norm.cdf(-z_plus)

        res_vs = res_mid + res_lh + res_rh
        return res_vs

    for slice_idx in slices:
        t = exp_dates[slice_idx]
        texp = ivol_data["Texp"]
        bid_vol = bid_vols[texp == t]
        ask_vol = ask_vols[texp == t]
        mid_vol = (bid_vol + ask_vol) / 2
        f = ivol_data["Fwd"][texp == t].iloc[0]
        k = np.log(ivol_data["Strike"][texp == t] / f)
        include = ~np.isnan(mid_vol)
        k_in = k[include]
        gs_mid[slice_idx] = gammaswap(k_in, mid_vol, include, slice_idx) / t
        gs_bid[slice_idx] = gammaswap(k_in, bid_vol, include, slice_idx) / t
        gs_ask[slice_idx] = gammaswap(k_in, ask_vol, include, slice_idx) / t

    return {"expiries": exp_dates, "gs_mid": gs_mid, "gs_bid": gs_bid, "gs_ask": gs_ask}
