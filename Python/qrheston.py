from typing import Callable

import numpy as np
from numfracpy import Mittag_Leffler_two
from scipy import integrate
from scipy.special import gamma, gammainc, roots_jacobi


class QuadraticRoughHeston:
    """Quadratic Rough Heston model."""

    def __init__(
        self,
        xi0: Callable[[float], float],
        c: float,
        nu: float,
        lam: float,
        al: float,
        n_quad: int = 20,
    ) -> float:
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
        """Gamma kernel."""
        return (self.nu / gamma(self.al)) * x ** (self.al - 1) * np.exp(-self.lam * x)

    def y0(self, u):
        """Compute y0(u) from xi0(u)."""
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
        """Compute shifted or blipped y0(u)."""
        return self.y0(u) - h * self.kernel(u)

    def resolvent_kernel(self, x):
        """Compute the resolvent kernel at x."""
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
        Compute the integral \int_0^x resolvent_kernel(s) ds.
        """
        if x > 0:
            return integrate.quad(lambda x: self.resolvent_kernel(x), 0.0, x)[0]
        elif x == 0:
            return 0.0
        else:
            raise ValueError("'x' must be non-negative.")

    def integral_K00(self, x, quad_scipy=False):
        r"""
        Compute the integral K00 at x. It corresponds to the integral
        \int_0^x k(s)^2 ds where k(s) is the gamma kernel function at s.
        Note: in SciPy, gammainc is the regularized lower incomplete gamma function.
        """
        if quad_scipy:
            return integrate.quad(lambda s: self.kernel(s) ** 2, 0.0, x)[0]
        else:
            x = np.atleast_1d(np.asarray(x))
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
        Compute the integral K0 at x. It corresponds to the integral \int_0^x k(s) ds
        where k(s) is the gamma kernel function at s.
        """
        if quad_scipy:
            return integrate.quad(lambda s: self.kernel(s), 0.0, x)[0]
        else:
            x = np.atleast_1d(np.asarray(x))
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
        expiries,
        output="all",
        delvix=1.0 / 12.0,
        nvix=10,
        h_ssr=None,
    ):
        if output not in ["all", "spx", "vix"]:
            raise ValueError("'output' must be one of ['all', 'spx', 'vix'].")
        if not paths > 0:
            raise ValueError("'paths' must be postive.")
        if not steps > 0:
            raise ValueError("'steps' must be positive.")
        if not delvix > 0:
            raise ValueError("'delvix' must be positive.")
        if not nvix > 0:
            raise ValueError("'nvix' must be positive.")
        if not isinstance(expiries, (list, np.ndarray)):
            raise ValueError("'expiries' must be a list or numpy array.")
        # if h_ssr is not None:
        #     h_ssr = np.atleast_1d(np.asarray(h_ssr))
        #     if h_ssr.shape[0] == 1:
        #         h_ssr = np.full_like(expiries, h_ssr[0])
        #     if h_ssr.shape[0] != len(expiries):
        #         raise ValueError(
        #             "'h_ssr' must be a scalar or have the same length as 'expiries'."
        #         )
        # print(h_ssr)

        Z_eps = np.random.normal(size=(steps, paths))
        Z_chi = np.random.normal(size=(steps, paths))
        v0 = self.xi0(0.0)
        y0_0 = (self.xi0(0.0) - self.c) ** 0.5
        alp = 1.0 / (2.0 * self.H + 1.0)

        def sim(expiry):
            dt = expiry / steps
            K0del = float(self.integral_K0(dt))
            K00del = float(self.integral_K00(dt))
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
                vbar = bigK0del * (alp * yhat**2 + (1 - alp) * Y**2 + self.c) / K00del
                sig_chi = np.sqrt(vbar * dt)
                sig_eps = np.sqrt(vbar * K00del * (1.0 - rho_uchi**2))
                chi[j, :] = sig_chi * Z_chi[j, :]
                eps = sig_eps * Z_eps[j, :]
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
                    chi_h[j, :] = sig_chi_h * Z_chi[j, :]
                    eps_h = sig_eps_h * Z_eps[j, :]
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

            if output in ["vix", "all"]:
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

            if output in ["all", "spx"]:
                res_sim["X"] = X
            if output in ["all", "vix"]:
                res_sim["vix"] = vix
            if output == "all":
                res_sim["v"] = v
                res_sim["w"] = w

            if h_ssr is not None:
                res_sim.update({"v_h": v_h, "X_h": X_h, "w_h": w_h})

            return res_sim

        sim_out = {expiry: sim(expiry) for expiry in expiries}
        return sim_out
