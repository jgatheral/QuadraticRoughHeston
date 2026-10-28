import numpy as np
from scipy.integrate import quad

from black import black_otm_impvol_mc
from utils import kernel, kernel_exp_approx


def cov_delta_X_delta_W(delta, H, lam):
    """
    Compute the covariance matrix between delta_X and delta_W.

        delta_Xk = int_{t_{k-1}}^{t_k} K(t_k - s) dW_s, where K is the kernel function,
        delta_Wk = W_{t_k} - W_{t_{k-1}}.

    Parameters
    ----------
    delta : float
        Time step size.
    H : float
        The Hurst parameter for the kernel function.
    lam : float
        The lambda parameter for the kernel function.

    Returns
    -------
    ndarray
        2x2 covariance matrix [[var_dX, cov_dX_dW], [cov_dX_dW, var_dW]].
    """
    var_dX = delta * quad(lambda s: kernel(delta * s, H, lam) ** 2, 0, 1)[0]
    cov_dX_dW = delta * quad(lambda s: kernel(delta * s, H, lam), 0, 1)[0]
    var_dW = delta
    return np.array(
        [
            [var_dX, cov_dX_dW],
            [cov_dX_dW, var_dW],
        ]
    )


def simulate_qrh_approx(
    params,
    T,
    k,
    n_steps,
    n_mc,
    seed=None,
    return_paths=False,
    eps_ssr=0.0,
    hybrid=True,
    eps_kernel=1e-4,
    N_kernel=250,
    endpoints_kernel=(None, None),
):
    """
    Simulate the QRH process using the sum of exponentials approximation.

    Parameters
    ----------
    params : dict
        Model parameters with keys 'H', 'lam', 'rho', 'p' and time-dependent
        callables 'a0', 'a1', 'a2', 'b0', 'x0'.
    T : float
        Time to maturity.
    k : array_like
        Log-forward moneyness for implied volatility computation.
    n_steps : int
        Number of time steps.
    n_mc : int
        Number of Monte Carlo paths.
    seed : int, optional
        Random seed for reproducibility.
    return_paths : bool, default=False
        If True, return (V, S) paths; otherwise return implied volatilities.
    eps_ssr : float, optional
        Small shift for SSR computation.
    hybrid : bool, default=True
        If True, use hybrid scheme for variance process else use multifactor scheme.
    eps_kernel : float, optional
        Tolerance for truncating singular values in kernel approximation.
    N_kernel : int, optional
        Number of discretization points for kernel approximation.

    Returns
    -------
    ndarray, tuple or dict
        Implied volatilities if return_paths=False, else (V, S) arrays.
        A nonzero eps_ssr instead returns SSR and ATM volatility/skew estimates.
    """
    left_kernel, right_kernel = endpoints_kernel
    if left_kernel is None:
        left_kernel = T / n_steps
    if right_kernel is None:
        right_kernel = T

    if seed is not None:
        np.random.seed(seed)

    # if True, we compute SSR
    compute_ssr = eps_ssr != 0.0

    # Extract model parameters
    H = params["H"]
    lam = params["lam"]
    rho = params["rho"]
    rho_perp = np.sqrt(1 - rho**2)
    # a0, a1, a2 are functions of time
    a0 = params["a0"]
    a1 = params["a1"]
    a2 = params["a2"]
    b0 = params["b0"]
    p = params["p"]
    x0 = params["x0"]

    # Time discretization
    dT = T / n_steps

    # Get exponential kernel approximation parameters
    c_kernel, gamma_kernel = kernel_exp_approx(
        H=H, lam=lam, a=left_kernel, b=right_kernel, N=N_kernel, eps=eps_kernel
    )
    print(f"Using N={len(c_kernel)} exponentials for kernel approximation.")

    # Initialize
    X = np.zeros((n_steps + 1, n_mc))  # Rough process
    V = np.zeros((n_steps + 1, n_mc))  # Variance process
    log_S = np.zeros((n_steps + 1, n_mc))  # Log-spot process
    U = np.zeros((n_steps + 1, n_mc, len(c_kernel)))  # OU auxiliary processes
    # initial values
    V[0, :] = a0(0.0) + a1(0.0) * (X[0, :] - a2(0.0)) ** 2
    mask_init = x0(0.0) - X[0, :] > 0
    V[0, mask_init] += (
        b0(0.0) * (X[0, mask_init] < x0(0.0)) * (x0(0.0) - X[0, mask_init]) ** p
    )
    if compute_ssr:
        X_ssr = np.zeros_like(X)
        V_ssr = np.zeros_like(V)
        log_S_ssr = np.zeros_like(log_S)
        U_ssr = np.zeros_like(U)
        # initial values
        U_ssr[0, :, :] = eps_ssr * rho
        X_ssr[0, :] = eps_ssr * rho * np.sum(c_kernel)
        V_ssr[0, :] = a0(0.0) + a1(0.0) * (X_ssr[0, :] - a2(0.0)) ** 2
        mask_init_ssr = x0(0.0) - X_ssr[0, :] > 0
        V_ssr[0, mask_init_ssr] += (
            b0(0.0)
            * (X_ssr[0, mask_init_ssr] < x0(0.0))
            * (x0(0.0) - X_ssr[0, mask_init_ssr]) ** p
        )

    # Compute correlation structure between dX and dW
    cov_mat = cov_delta_X_delta_W(delta=dT, H=H, lam=lam)
    std_X = cov_mat[0, 0] ** 0.5
    std_W = cov_mat[1, 1] ** 0.5
    cov_XW = cov_mat[0, 1]
    rho_XW = cov_XW / (std_X * std_W)
    rho_XW_perp = np.sqrt(1 - rho_XW**2)

    for i in range(n_steps):
        # Construct correlated increments

        if hybrid:
            normal = np.random.normal(size=(2, n_mc))
            dX = std_X * normal[0, :]
            dW = std_W * (rho_XW * normal[0, :] + rho_XW_perp * normal[1, :])
        else:
            dW = np.sqrt(dT) * np.random.normal(size=n_mc)
        dW_perp = np.sqrt(dT) * np.random.normal(size=n_mc)

        # Update U with implicit-explicit scheme
        U[i + 1, :] = (U[i, :] + np.sqrt(V[i, :, None]) * dW[:, None]) / (
            1 + gamma_kernel[None, :] * dT
        )
        if compute_ssr:
            U_ssr[i + 1, :] = (
                U_ssr[i, :] + np.sqrt(V_ssr[i, :, None]) * dW[:, None]
            ) / (1 + gamma_kernel[None, :] * dT)

        # Update X
        if hybrid:
            X[i + 1, :] = (
                np.sum(
                    c_kernel[None, :]
                    * np.exp(-gamma_kernel[None, :] * dT)
                    * U[i, :, :],
                    axis=1,
                )
                + np.sqrt(V[i, :]) * dX
            )
            if compute_ssr:
                X_ssr[i + 1, :] = (
                    np.sum(
                        c_kernel[None, :]
                        * np.exp(-gamma_kernel[None, :] * dT)
                        * U_ssr[i, :, :],
                        axis=1,
                    )
                    + np.sqrt(V_ssr[i, :]) * dX
                )
        else:
            X[i + 1, :] = np.sum(c_kernel[None, :] * U[i + 1, :, :], axis=1)
            if compute_ssr:
                X_ssr[i + 1, :] = np.sum(c_kernel[None, :] * U_ssr[i + 1, :, :], axis=1)

        # Update V
        t_i1 = (i + 1) * dT
        V[i + 1, :] = a0(t_i1) + a1(t_i1) * (X[i + 1, :] - a2(t_i1)) ** 2
        mask = x0(t_i1) - X[i + 1, :] > 0
        V[i + 1, mask] += (
            b0(t_i1) * (X[i + 1, mask] < x0(t_i1)) * (x0(t_i1) - X[i + 1, mask]) ** p
        )
        if compute_ssr:
            V_ssr[i + 1, :] = a0(t_i1) + a1(t_i1) * (X_ssr[i + 1, :] - a2(t_i1)) ** 2
            mask_ssr = x0(t_i1) - X_ssr[i + 1, :] > 0
            V_ssr[i + 1, mask_ssr] += (
                b0(t_i1)
                * (X_ssr[i + 1, mask_ssr] < x0(t_i1))
                * (x0(t_i1) - X_ssr[i + 1, mask_ssr]) ** p
            )

        # Update log S with trapezoidal rule for integral in V
        log_S[i + 1, :] = (
            log_S[i, :]
            - 0.5 * (V[i, :] + V[i + 1, :]) * 0.5 * dT
            + np.sqrt(V[i, :]) * (rho * dW + rho_perp * dW_perp)
        )
        if compute_ssr:
            log_S_ssr[i + 1, :] = (
                log_S_ssr[i, :]
                - 0.5 * (V_ssr[i, :] + V_ssr[i + 1, :]) * 0.5 * dT
                + np.sqrt(V_ssr[i, :]) * (rho * dW + rho_perp * dW_perp)
            )
    S = np.exp(log_S)

    if compute_ssr:
        atm_impvol, atm_skew = black_otm_impvol_mc(
            S=S[-1, :], k=np.zeros(1), T=T, return_atm_skew=True
        )
        S_ssr = np.exp(log_S_ssr[-1, :])
        atm_impvol_ssr, _ = black_otm_impvol_mc(
            S=S_ssr, k=np.zeros(1), T=T, return_atm_skew=True
        )
        ssr = ((atm_impvol_ssr - atm_impvol) / eps_ssr) / atm_skew
        return {
            "ssr": ssr,
            "atm_impvol": atm_impvol,
            "atm_impvol_ssr": atm_impvol_ssr,
            "atm_skew": atm_skew,
        }

    if return_paths:
        return V, S
    else:
        # return implied volatilities
        return np.asarray(black_otm_impvol_mc(S=S[-1, :], k=k, T=T))
