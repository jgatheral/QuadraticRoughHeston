from collections.abc import Callable

import numpy as np
from numfracpy import Mittag_Leffler_two
from scipy import integrate
from scipy.special import gamma, gammainc, roots_jacobi
from scipy.stats import norm

from black import black_otm_impvol_mc


class QuadraticRoughHeston:
    """
    Quadratic Rough Heston model implementation.

    Parameters
    ----------
    xi0 : callable
        Initial forward variance curve function.
    c : float
        Constant variance floor parameter.
    nu : float
        Volatility of volatility parameter.
    lam : float
        Mean reversion rate parameter.
    al : float
        Roughness parameter, must be in (0.5, 1).
    n_quad : int, default 20
        Number of quadrature points for numerical integration.

    Attributes
    ----------
    H : float
        Hurst parameter, equal to al - 0.5.
    nu_hat : float
        Scaled volatility parameter.
    """

    def __init__(
        self,
        xi0: Callable[[float], float],
        c: float,
        nu: float,
        lam: float,
        al: float,
        n_quad: int = 20,
    ):
        if not (0.5 < al < 1):
            raise ValueError("'al' must be between 0.5 and 1.")

        if not c > 0:
            raise ValueError("'c' must be positive.")

        if not nu > 0:
            raise ValueError("'nu' must be positive.")

        self.xi0 = xi0
        self.c = c
        self.nu = nu
        self.lam = lam
        self.al = al
        self.H = self.al - 0.5
        self.n_quad = n_quad
        self.nu_hat = self.nu * gamma(2.0 * self.H) ** 0.5 / gamma(self.al)

    def kernel(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the gamma kernel function.

        Parameters
        ----------
        x : array_like
            Input values where x >= 0.

        Returns
        -------
        ndarray
            Gamma kernel values at x.
        """
        return (self.nu / gamma(self.al)) * x ** (self.al - 1) * np.exp(-self.lam * x)

    def y0(self, u):
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
        u = np.atleast_1d(u)
        integral = np.zeros_like(u)
        mask = u > 0
        x_jac, w_jac = roots_jacobi(n=self.n_quad, alpha=2.0 * self.H - 1.0, beta=0.0)
        integral[mask] = (
            w_jac[:, None]
            * self.xi0(0.5 * u[mask][None, :] * (1 + x_jac[:, None]))
            * np.exp(-self.lam * u[mask][None, :] * (1 - x_jac[:, None]))
        ).sum(axis=0)
        integral[mask] *= (self.nu / gamma(self.al)) ** 2 * (0.5 * u[mask]) ** (
            2.0 * self.H
        )
        return (self.xi0(u) - integral - self.c) ** 0.5

    def y0_shifted(self, u: np.ndarray, h: float) -> np.ndarray:
        """
        Compute shifted y0(u) from initial variance curve xi0(u) for SSR computation.

        Parameters
        ----------
        u : array_like
            Time points where u >= 0.

        Returns
        -------
        ndarray
            Shifted y0(u) values at u.
        """
        return self.y0(u) - h * self.kernel(u)

    def resolvent_kernel(self, x):
        """
        Compute the resolvent kernel at x.

        Parameters
        ----------
        x : float or array_like
            Input values where x >= 0.

        Returns
        -------
        float or ndarray
            Resolvent kernel values using Mittag-Leffler function.
        """
        return (
            self.nu_hat**2
            * np.exp(-2.0 * self.lam * x)
            * x ** (2.0 * self.H - 1.0)
            * Mittag_Leffler_two(
                self.nu_hat**2 * x ** (2.0 * self.H),
                2.0 * self.H,
                2.0 * self.H,
            )
        )

    def integral_bigK0(self, x):
        r"""
        Compute integral of resolvent kernel from 0 to x.

        Parameters
        ----------
        x : float
            Upper integration limit, must be non-negative.

        Returns
        -------
        float
            Integral values at x.
        """
        if x > 0:
            return integrate.quad(lambda x: self.resolvent_kernel(x), 0.0, x)[0]
        elif x == 0:
            return 0.0
        else:
            raise ValueError("'x' must be non-negative.")

    def integral_K00(self, x, quad_scipy=False):
        r"""
        Compute integral of squared gamma kernel from 0 to x.

        Parameters
        ----------
        x : array_like
            Integration limits, must be non-negative.
        quad_scipy : bool, default False
            If True, use SciPy quadrature. Otherwise, use analytical formula.

        Returns
        -------
        ndarray
            Integral values at x.

        Notes
        -----
        In SciPy, gammainc is the regularized lower incomplete gamma function.
        """
        if quad_scipy:
            return integrate.quad(lambda s: self.kernel(s) ** 2, 0.0, x)[0]
        else:
            x = np.atleast_1d(np.asarray(x))
            if self.lam == 0.0:
                return (
                    (self.nu / gamma(self.al)) ** 2
                    * x ** (2.0 * self.H)
                    / (2.0 * self.H)
                )
            mask = x > 0
            res = np.empty_like(x)
            res[mask] = (
                gamma(2.0 * self.H)
                * gammainc(2.0 * self.H, 2 * self.lam * x[mask])
                / (2.0 * self.lam) ** (2.0 * self.H)
            )
            res[~mask] = x[~mask] ** (2.0 * self.H) / (2.0 * self.H)
            res *= (self.nu / gamma(self.al)) ** 2
            return res

    def integral_K0(self, x, quad_scipy=False):
        r"""
            Compute integral of gamma kernel from 0 to x.

        Parameters
        ----------
        x : array_like
            Integration limits, must be non-negative.
        quad_scipy : bool, default False
            If True, use SciPy quadrature. Otherwise, use analytical formula.

        Returns
        -------
        ndarray
            Integral values at x.
        """
        if quad_scipy:
            return integrate.quad(lambda s: self.kernel(s), 0.0, x)[0]
        else:
            x = np.atleast_1d(np.asarray(x))
            if self.lam == 0.0:
                return (self.nu / gamma(self.al)) * x**self.al / self.al
            mask = x > 0
            res = np.empty_like(x)
            res[mask] = (
                gamma(self.H + 0.5)
                * gammainc(self.H + 0.5, self.lam * x[mask])
                / self.lam ** (self.H + 0.5)
            )
            res[~mask] = x[~mask] ** self.al / self.al
            res *= self.nu / gamma(self.al)
            return res

    def simulate(
        self,
        paths,
        steps,
        spx_expiries=None,
        vix_expiries=None,
        output=None,
        delvix=1.0 / 12.0,
        nvix=10,
        h_ssr=None,
        seed=None,
    ):
        """
        Simulate sample paths for the Quadratic Rough Heston model.

        Parameters
        ----------
        paths : int
            Number of Monte Carlo paths.
        steps : int
            Number of time steps per path.
        spx_expiries : array_like or None
            Expiry times for SPX outputs.
        vix_expiries : array_like or None
            Expiry times for VIX outputs.
        output : {'all', 'spx', 'vix'} or None, default None
            Which outputs to return. If None, inferred from expiries.
        delvix : float, default 1/12
            VIX averaging window length.
        nvix : int, default 10
            Number of quadrature points for VIX calculation.
        h_ssr : callable or None, optional
            Function for shifted y0 (for SSR). Takes expiry as argument.
        seed : int or None, optional
            Random seed for reproducibility.

        Returns
        -------
        dict
            Dictionary with keys 'spx' and/or 'vix', each mapping expiry
            times to simulation results. If expiries are identical,
            computations are shared to avoid duplication.
        """
        if not paths > 0:
            raise ValueError("'paths' must be postive.")
        if not steps > 0:
            raise ValueError("'steps' must be positive.")
        if not delvix > 0:
            raise ValueError("'delvix' must be positive.")
        if not nvix > 0:
            raise ValueError("'nvix' must be positive.")
        if spx_expiries is None and vix_expiries is None:
            raise ValueError(
                "At least one of spx_expiries or vix_expiries must be provided."
            )
        if seed is not None:
            np.random.seed(seed)

        # Prepare random numbers
        # Z_eps = np.random.normal(size=(steps, paths))
        # Z_chi = np.random.normal(size=(steps, paths))
        v0 = self.xi0(0.0)
        y0_0 = (self.xi0(0.0) - self.c) ** 0.5
        alp = 1.0 / (2.0 * self.H + 1.0)

        def sim(expiry):
            """
            Simulate single expiry for Quadratic Rough Heston model.

            Parameters
            ----------
            expiry : float
                Time to expiration.

            Returns
            -------
            dict
                Simulation results with keys depending on output parameter:
                - 'X': SPX log price increments
                - 'vix': VIX values
                - 'v': variance process (if output='all')
                - 'w': integrated variance (if output='all')
                - 'chi': volatility shocks (if output='all')
                - '_h' variants if h_ssr is provided
            """
            print(f"expiry: {expiry:.4f}")
            dt = expiry / steps
            K0del = self.integral_K0(dt).item()
            K00del = self.integral_K00(dt).item()
            bigK0del = self.integral_bigK0(dt)
            tj = np.arange(1, steps + 1) * dt
            yj = self.y0(tj)
            K00j = np.zeros(steps + 1)
            K00j[1:] = self.integral_K00(tj)
            bstar = np.sqrt(np.diff(K00j) / dt)
            chi = np.zeros((steps, paths))
            v = np.full(paths, v0)
            Y = np.full(paths, y0_0)
            yhat = np.full(paths, yj[0])
            rho_uchi = K0del / (K00del * dt) ** 0.5
            beta_uchi = K0del / dt
            X = np.zeros(paths)
            w = np.zeros(paths)

            if h_ssr is not None:
                chi_h = np.zeros((steps, paths))
                v_h = np.full(paths, v0)
                Y_h = np.full(paths, y0_0)
                yj_h = self.y0_shifted(tj, h_ssr(expiry))
                yhat_h = np.full(paths, yj_h[0])
                X_h = np.zeros(paths)
                w_h = np.zeros(paths)

            for j in range(steps):
                Z_eps = np.random.normal(size=(paths))
                Z_chi = np.random.normal(size=(paths))
                vbar = bigK0del * (alp * yhat**2 + (1 - alp) * Y**2 + self.c) / K00del
                sig_chi = np.sqrt(vbar * dt)
                sig_eps = np.sqrt(vbar * K00del * (1.0 - rho_uchi**2))
                chi[j, :] = sig_chi * Z_chi
                eps = sig_eps * Z_eps
                u = beta_uchi * chi[j, :] + eps
                Y = yhat + u
                vf = Y**2 + self.c
                dw = (v + vf) / 2 * dt
                w += dw
                X -= 0.5 * dw + chi[j, :]
                if j < steps - 1:
                    btilde = bstar[1 : j + 2][::-1]
                    yhat = yj[j + 1] + np.tensordot(btilde, chi[: j + 1, :], axes=1)
                v = vf

                if h_ssr is not None:
                    vbar_h = (
                        bigK0del
                        * (alp * yhat_h**2 + (1 - alp) * Y_h**2 + self.c)
                        / K00del
                    )
                    sig_chi_h = np.sqrt(vbar_h * dt)
                    sig_eps_h = np.sqrt(vbar_h * K00del * (1.0 - rho_uchi**2))
                    chi_h[j, :] = sig_chi_h * Z_chi
                    eps_h = sig_eps_h * Z_eps
                    u_h = beta_uchi * chi_h[j, :] + eps_h
                    Y_h = yhat_h + u_h
                    vf_h = Y_h**2 + self.c
                    dw_h = (v_h + vf_h) / 2 * dt
                    w_h += dw_h
                    X_h -= 0.5 * dw_h + chi_h[j, :]
                    if j < steps - 1:
                        btilde = bstar[1 : j + 2][::-1]
                        yhat_h = yj_h[j + 1] + np.tensordot(
                            btilde, chi_h[: j + 1, :], axes=1
                        )
                    v_h = vf_h

            if output in ["vix", "spx-vix", "all"]:
                vix2 = 0.0
                ds = delvix / nvix
                for k in range(nvix):
                    tk = expiry + (k + 1.0) * ds
                    Ku = np.concatenate(
                        (self.integral_K00(tk), self.integral_K00(tk - tj))
                    )
                    ck_vec = np.sqrt(-np.diff(Ku) / dt)
                    dyTu = np.dot(ck_vec, chi)
                    yTu = self.y0(tk) + dyTu
                    vix2 += (
                        (yTu**2 + self.c)
                        * (1.0 + self.integral_bigK0((nvix - k - 1) * ds))
                        / nvix
                    )
                vix2 += v * (1.0 + self.integral_bigK0(delvix)) / (2 * nvix) - (
                    yTu**2 + self.c
                ) / (2.0 * nvix)
                vix = np.sqrt(vix2)

            res_sim = {}

            if output in ["spx", "spx-vix", "all"]:
                res_sim["X"] = X
            if output in ["vix", "spx-vix", "all"]:
                res_sim["vix"] = vix
            if output == "all":
                res_sim["v"] = v
                res_sim["w"] = w
                res_sim["chi"] = chi

            if h_ssr is not None:
                res_sim.update({"v_h": v_h, "X_h": X_h, "w_h": w_h})

            return res_sim

        results = {}

        # Convert to tuple for comparison and set logic
        spx_expiries_tuple = tuple(spx_expiries) if spx_expiries is not None else ()
        vix_expiries_tuple = tuple(vix_expiries) if vix_expiries is not None else ()

        if (
            spx_expiries_tuple
            and vix_expiries_tuple
            and spx_expiries_tuple == vix_expiries_tuple
        ):
            # Only compute once if expiries are the same
            print("Simulating SPX and VIX expiries...")
            sim_out = {expiry: sim(expiry) for expiry in spx_expiries_tuple}
            results["spx"] = {k: {"X": v["X"]} for k, v in sim_out.items()}
            results["vix"] = {k: {"vix": v["vix"]} for k, v in sim_out.items()}
            if output == "all":
                results["all"] = sim_out
        else:
            if spx_expiries is not None:
                print("Simulating SPX expiries...")
                if output != "all":
                    output = "spx"
                results["spx"] = {expiry: sim(expiry) for expiry in spx_expiries_tuple}
            if vix_expiries is not None:
                print("Simulating VIX expiries...")
                if output != "all":
                    output = "vix"
                results["vix"] = {expiry: sim(expiry) for expiry in vix_expiries_tuple}

        return results

    def compute_ssr(self, expiries, paths, steps, h_ssr, seed=None):
        "Compute SSR for expiries"

        sim = self.simulate(
            paths=paths,
            steps=steps,
            spx_expiries=expiries,
            output="spx",
            h_ssr=h_ssr,
            seed=seed,
        )

        # simulate MC paths (normal and shifted)
        S_matrix = np.array([np.exp(sim["spx"][T]["X"]) for T in expiries])
        S_h_matrix = np.array([np.exp(sim["spx"][T]["X_h"]) for T in expiries])

        # compute regression coefficient beta
        atm_impvols_mc = np.array(
            [
                black_otm_impvol_mc(S_matrix[idx, :], k=0.0, T=T)
                for idx, T in enumerate(expiries)
            ]
        ).flatten()
        atm_impvols_h_mc = np.array(
            [
                black_otm_impvol_mc(S_h_matrix[idx, :], k=0.0, T=T)
                for idx, T in enumerate(expiries)
            ]
        ).flatten()
        beta = (atm_impvols_h_mc - atm_impvols_mc) / h_ssr(expiries)

        # compute atm skew
        w = expiries * atm_impvols_mc**2
        atm_digit = (S_matrix >= S_matrix.mean(axis=1, keepdims=True)).mean(axis=1)
        atm_skew = norm.cdf(-0.5 * w**0.5) - atm_digit
        atm_skew /= expiries**0.5 * norm.pdf(-0.5 * w**0.5)

        # compute ssr
        ssr = beta / atm_skew

        return ssr
