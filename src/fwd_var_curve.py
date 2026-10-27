import numpy as np
from scipy.linalg import inv
from scipy.optimize import minimize


def obj_w(expiries, w_in):
    def objective(err_vec):
        w_in_1 = w_in + 2 * np.sqrt(w_in) * err_vec * np.sqrt(expiries)
        xi_vec = np.concatenate(
            ([w_in_1[0] / expiries[0]], np.diff(w_in_1) / np.diff(expiries))
        )
        dxi_dt = np.diff(xi_vec) / np.diff(expiries)
        w_out = (
            np.concatenate(([0], np.cumsum(xi_vec[1:] * np.diff(expiries))))
            + xi_vec[0] * expiries[0]
        )
        res = np.sum((w_in - w_out) ** 2) + np.sum(dxi_dt**2)
        return res * 1e3

    return objective


def xi_curve(expiries, w_in, eps=0):
    n = len(w_in)
    if eps > 0:
        res_optim = minimize(
            obj_w(expiries, w_in),
            np.zeros(n),
            method="L-BFGS-B",
            bounds=[(-eps, eps)] * n,
        )
        err_vec = res_optim.x
        w_in_1 = w_in + 2 * np.sqrt(w_in) * err_vec * np.sqrt(expiries)
    else:
        w_in_1 = w_in
    xi_vec_out = np.concatenate(
        ([w_in_1[0] / expiries[0]], np.diff(w_in_1) / np.diff(expiries))
    )

    def xi_curve_raw(t):
        if t <= expiries[-1]:
            return xi_vec_out[np.sum(expiries < t)]
        else:
            return xi_vec_out[-1]

    xi_curve_out = np.vectorize(xi_curve_raw, otypes=[float])
    fit_errs = np.sqrt(w_in_1 / expiries) - np.sqrt(w_in / expiries)

    return {
        "xi_vec": xi_vec_out,
        "xi_curve": xi_curve_out,
        "fit_errs": fit_errs,
        "w_out": w_in_1,
    }


def xi_curve_smooth(expiries, w_in, xi=True, eps=0.0):
    def phi(tau):
        def func(x):
            min_val = np.minimum(x, tau)
            return 1 - min_val**3 / 6 + x * tau * (2 + min_val) / 2

        return func

    def phi_deri(tau):
        def func(x):
            min_val = np.minimum(x, tau)
            return tau - min_val**2 / 2 + tau * min_val

        return func

    n = len(expiries)
    A = np.array([[phi(expiries[i])(expiries[j]) for j in range(n)] for i in range(n)])
    A_inv = inv(A)

    def obj_1(err_vec):
        v = w_in + 2 * np.sqrt(w_in) * err_vec * np.sqrt(expiries)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            value = v.T @ A_inv @ v
        return float(value) if np.isfinite(value) else np.finfo(float).max

    if eps > 0:
        # R's optim() uses a central finite-difference step of 1e-3 by default.
        # That relatively large step matters here because A is very ill-conditioned;
        # SciPy's much smaller default step produces noisy gradients and a different
        # fit.  Match the other implementation's bounds handling and L-BFGS-B
        # defaults so the two ports follow the same optimization path closely.
        def obj_1_grad(err_vec):
            step = 1e-3
            grad = np.empty(n)
            for i in range(n):
                err_hi = err_vec.copy()
                err_lo = err_vec.copy()
                err_hi[i] = min(eps, err_vec[i] + step)
                err_lo[i] = max(-eps, err_vec[i] - step)
                grad[i] = (obj_1(err_hi) - obj_1(err_lo)) / (
                    err_hi[i] - err_lo[i]
                )
            return grad

        res_optim = minimize(
            obj_1,
            np.zeros(n),
            method="L-BFGS-B",
            jac=obj_1_grad,
            bounds=[(-eps, eps)] * n,
            options={
                "maxcor": 5,
                "ftol": 1e7 * np.finfo(float).eps,
                "gtol": 0,
                "maxiter": 100,
            },
        )
        err_vec = res_optim.x
    else:
        err_vec = np.zeros(n)
    w_in_1 = w_in + 2 * np.sqrt(w_in) * err_vec * np.sqrt(expiries)
    # L-BFGS-B can leave IEEE status flags set; the product itself is finite for
    # valid inputs, so isolate it from those stale floating-point warnings.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        Z = A_inv @ w_in_1

    def curve_raw(x):
        sum_curve = sum(Z[i] * phi(expiries[i])(x) for i in range(n))
        sum_curve_deri = sum(Z[i] * phi_deri(expiries[i])(x) for i in range(n))
        return sum_curve_deri if xi else sum_curve

    xi_curve_out = np.vectorize(curve_raw, otypes=[float])
    fit_errs = np.sqrt(w_in_1 / expiries) - np.sqrt(w_in / expiries)

    return {"xi_curve": xi_curve_out, "fit_errs": fit_errs, "w_out": w_in_1}
