import numpy as np
import scipy.special as sp
from scipy import integrate, linalg


def vix2_approx_coefficients(qrh, expiry, steps, n_quad_vix=10, delta_vix=1 / 12):
    # pre-compute quantities on the integral grid "u_quad"

    Delta = expiry / steps
    u_quad = np.linspace(0.0, delta_vix, n_quad_vix + 1)

    def func_a(k, u):
        return np.sqrt(
            (qrh.integral_K00(u - (k - 1) * Delta) - qrh.integral_K00(u - k * Delta))
            / Delta
        )

    func_a_quad = np.array(
        [func_a(k, expiry + u_quad) for k in np.arange(1, steps + 1)]
    )
    weights_quad = np.array([1.0 + qrh.integral_bigK0(delta_vix - u) for u in u_quad])
    y0_quad = qrh.y0(expiry + u_quad)
    tab_xi = (
        np.array(
            [
                integrate.quad(lambda u: qrh.xi0(u), (k - 1) * Delta, k * Delta)[0]
                for k in range(1, steps + 1)
            ]
        )
        / Delta
    )

    # Compute scalar a
    a_quad = (y0_quad**2 + qrh.c) * weights_quad
    a_scalar = 0.5 * np.sum(a_quad[:-1] + a_quad[1:]) / n_quad_vix

    # Compute vector b
    b_vec = np.zeros(steps)
    for i in range(steps):
        b_vec_quad = y0_quad * func_a_quad[i, :] * weights_quad
        b_vec[i] = 0.5 * np.sum(b_vec_quad[:-1] + b_vec_quad[1:]) * 2.0 / n_quad_vix
    b_vec *= np.sqrt(Delta * tab_xi)  # rescale variance for chi

    # Compute matrix C
    c_mat = np.zeros((steps, steps))
    for i in range(steps):
        for j in range(steps):
            if i <= j:
                c_quad = func_a_quad[i, :] * func_a_quad[j, :] * weights_quad
                c_mat[i, j] = 0.5 * np.sum(c_quad[:-1] + c_quad[1:]) / n_quad_vix
            else:
                c_mat[i, j] = c_mat[j, i]
            # rescale variance for chi
            c_mat[i, j] *= Delta * np.sqrt(tab_xi[i] * tab_xi[j])

    return a_scalar, b_vec, c_mat


def log_charfunc_quadratic_normal(t, a, b, C):
    """
    Log of the characteristic function of Gaussian quadratic form
    S = a + b^T*Z + Z^T*C*Z.

    Parameters
    ----------
    t : float or array_like
        Argument of the characteristic function.
    a : float
        Scalar offset term.
    b : array_like, shape (n,)
        Linear coefficient vector.
    C : array_like, shape (n, n)
        Quadratic coefficient matrix (covariance matrix).

    Returns
    -------
    complex or ndarray
        Log characteristic function value log E[exp(itS)] where Z ~ N(0, I).
    """
    # TODO: vectorize for t array
    id = np.eye(len(b))
    shift_id = id - 2j * t * C
    inv = linalg.solve(shift_id, id)
    det = linalg.det(shift_id)
    return -0.5 * np.log(det) + 1j * a * t - 0.5 * t**2 * b.T @ inv @ b


def charfunc_quadratic_normal(t, a, b, C):
    return np.exp(log_charfunc_quadratic_normal(t, a, b, C))


def log_charfunc_quadratic_normal_svd(t, a, b, U, D):
    """
    Log characteristic function of Gaussian quadratic form using SVD decomposition.

    Parameters
    ----------
    t : float or array_like
        Argument of the characteristic function.
    a : float
        Scalar offset term.
    b : array_like, shape (n,)
        Linear coefficient vector.
    U : array_like, shape (n, n)
        Orthogonal matrix from SVD of covariance matrix C.
    D : array_like, shape (n,)
        Eigenvalues from SVD of covariance matrix C.

    Returns
    -------
    complex or ndarray
        Log characteristic function value E[exp(itS)] where S = a + b^T*Z + Z^T*C*Z
        and C = U @ diag(D) @ U^T.
    """
    det_shift_id = np.prod(1.0 - 2j * t * D)
    U_t_b = U.T @ b
    return (
        -0.5 * np.log(det_shift_id)
        + 1j * a * t
        - 0.5 * t**2 * np.sum(U_t_b**2 / (1.0 - 2j * t * D))
    )


def charfunc_quadratic_normal_svd(t, a, b, U, D):
    return np.exp(log_charfunc_quadratic_normal_svd(t, a, b, U, D))


def _exp_diff(logA, logB):
    """Compute exp(logA) - exp(logB) in a numerically stable way."""
    m = max(logA, logB)
    if m < -700:  # to avoid underflow in exp
        return 0.0
    if logA >= logB:
        return np.exp(logA) * (-np.expm1(logB - logA))
    else:
        return np.exp(logB) * np.expm1(logA - logB)


def integral_vix_fut_new(a, b, C, U, D, upper=1.0e4):
    r"""
    We use that
    int_0^\infty (1 - phi(it)) / t^{3/2} dt
    =
    int_0^\infty (1 - e^{-ct}) / t^{3/2} dt
    + int_0^\infty (e^{-ct} - phi(it)) / t^{3/2} dt
    =
    2\sqrt{pi}c + 2 int_0^\infty (e^{-cu^2} - phi(iu^2)) / u^{2} du

    with c = factor = a - sum_i b_tilde_i^2 / lambda_i / 4
    and b_tilde = U^T b, lambda_i = D_i
    """

    U_t_b = U.T @ b
    factor = a - np.sum(U_t_b**2 / D) / 4

    def integrand(u):
        # log_cf_i = log_charfunc_quadratic_normal_svd(1j * u**2, a=a, b=b, U=U, D=D)
        log_cf_i = log_charfunc_quadratic_normal(1j * u**2, a=a, b=b, C=C)
        return _exp_diff(-factor * u**2, log_cf_i) / u**2

    integral = (
        np.sqrt(factor)
        + (1 / np.sqrt(np.pi)) * integrate.quad(integrand, 0.0, upper)[0]
    )
    return integral


def pdf_vix2(x, a, b, C=None, U=None, D=None, upper=1e4):
    """
    Probability density function of Gaussian quadratic form via Fourier inversion.

    Parameters
    ----------
    x : float
        Point at which to evaluate the PDF.
    a : float
        Scalar offset term.
    b : array_like, shape (n,)
        Linear coefficient vector.
    C : array_like, shape (n, n), optional
        Quadratic coefficient matrix (covariance matrix).
        If provided, uses standard computation.
    U : array_like, shape (n, n), optional
        Orthogonal matrix from SVD of covariance matrix C.
        Must be provided together with D for SVD computation.
    D : array_like, shape (n,), optional
        Eigenvalues from SVD of covariance matrix C.
        Must be provided together with U for SVD computation.
    upper : float, optional
        Upper integration limit. Default is 1e4.

    Returns
    -------
    float
        PDF value at x for S = a + b^T*Z + Z^T*C*Z where Z ~ N(0, I).

    Notes
    -----
    Either C or both U and D must be provided. If both are given, C takes precedence.
    """
    if C is not None:

        def integrand(t):
            return np.real(np.exp(-1j * t * x) * charfunc_quadratic_normal(t, a, b, C))
    elif U is not None and D is not None:

        def integrand(t):
            return np.real(
                np.exp(-1j * t * x) * charfunc_quadratic_normal_svd(t, a, b, U, D)
            )
    else:
        raise ValueError("Either C or both U and D must be provided")

    return integrate.quad(integrand, 0.0, upper)[0] / np.pi


def pdf_vix2_svd(x, a, b, U, D, upper=1e4):
    """
    Probability density function of Gaussian quadratic form via Fourier inversion
    using SVD.

    Parameters
    ----------
    x : float
        Point at which to evaluate the PDF.
    a : float
        Scalar offset term.
    b : array_like, shape (n,)
        Linear coefficient vector.
    U : array_like, shape (n, n)
        Orthogonal matrix from SVD of covariance matrix C.
    D : array_like, shape (n,)
        Eigenvalues from SVD of covariance matrix C.
    upper : float, optional
        Upper integration limit. Default is 1e4.

    Returns
    -------
    float
        PDF value at x for S = a + b^T*Z + Z^T*C*Z where C = U @ diag(D) @ U^T.
    """

    def integrand(t):
        return np.real(
            np.exp(-1j * t * x) * charfunc_quadratic_normal_svd(t, a, b, U, D)
        )

    return integrate.quad(integrand, 0.0, upper)[0] / np.pi


def pdf_vix(x, a, b, C=None, U=None, D=None):
    """
    Probability density function of VIX (square root of quadratic form).

    Parameters
    ----------
    x : float
        Point at which to evaluate the PDF.
    a : float
        Scalar offset term.
    b : array_like, shape (n,)
        Linear coefficient vector.
    C : array_like, shape (n, n), optional
        Quadratic coefficient matrix (covariance matrix).
        If provided, uses standard computation.
    U : array_like, shape (n, n), optional
        Orthogonal matrix from SVD of covariance matrix C.
        Must be provided together with D for SVD computation.
    D : array_like, shape (n,), optional
        Eigenvalues from SVD of covariance matrix C.
        Must be provided together with U for SVD computation.

    Returns
    -------
    float
        PDF value at x for sqrt(S) where S = a + b^T*Z + Z^T*C*Z.

    Notes
    -----
    Either C or both U and D must be provided. If both are given, C takes precedence.
    """
    if C is not None:
        return 2.0 * x * pdf_vix2(x**2, a, b, C)
    elif U is not None and D is not None:
        return 2.0 * x * pdf_vix2_svd(x**2, a, b, U, D)
    else:
        raise ValueError("Either C or both U and D must be provided")


def pdf_cos_method(x, N, f_cf, lower, upper):
    """
    Compute a PDF using the COS (cosine) method.

    Parameters
    ----------
    x : array_like
        1-D array of points where the PDF is evaluated.
    N : int
        Number of cosine series terms (truncation parameter).
    f_cf : callable
        Characteristic function phi(u) of the distribution. Must accept a
        numpy array of real frequencies and return complex values.
    lower, upper : float
        Integration interval [lower, upper] used for the COS expansion.

    Returns
    -------
    pdf : ndarray
        1-D array of estimated PDF values corresponding to x.
    """
    k = np.arange(N)
    x = np.atleast_1d(np.asarray(x))
    factor = np.exp(-1j * k * np.pi * lower / (upper - lower))
    f_cf_k = np.array([f_cf(k_i * np.pi / (upper - lower)) for k_i in k])
    F_k = 2 / (upper - lower) * np.real(f_cf_k * factor)
    cos_k = np.cos(np.pi * k[None, :] * (x[:, None] - lower) / (upper - lower))
    F_k[0] *= 0.5  # first term is halved
    return np.sum(F_k[None, :] * cos_k, axis=1)


def pdf_vix2_cos(x, N, a, b, U, D, lower=-10.0, upper=10.0):
    """
    Probability density function of Gaussian quadratic form using COS method.

    Parameters
    ----------
    x : array_like
        Points at which to evaluate the PDF.
    N : int
        Number of cosine series terms.
    a : float
        Scalar offset term.
    b : array_like, shape (n,)
        Linear coefficient vector.
    U : array_like, shape (n, n)
        Orthogonal matrix from SVD of covariance matrix C.
    D : array_like, shape (n,)
        Eigenvalues from SVD of covariance matrix C.
    lower : float, optional
        Lower integration limit for COS method. Default is -10.0.
    upper : float, optional
        Upper integration limit for COS method. Default is 10.0.

    Returns
    -------
    ndarray
        PDF values at x for S = a + b^T*Z + Z^T*C*Z where C = U @ diag(D) @ U^T.
    """

    def _f_cf(u):
        return charfunc_quadratic_normal_svd(u, a=a, b=b, U=U, D=D)

    return pdf_cos_method(x=x, N=N, f_cf=_f_cf, lower=lower, upper=upper)


def pdf_vix_cos(x, N, a, b, U, D, lower=-10.0, upper=10.0):
    """
    Probability density function of VIX using COS method.

    Parameters
    ----------
    x : array_like
        Points at which to evaluate the PDF.
    N : int
        Number of cosine series terms.
    a : float
        Scalar offset term.
    b : array_like, shape (n,)
        Linear coefficient vector.
    U : array_like, shape (n, n)
        Orthogonal matrix from SVD of covariance matrix C.
    D : array_like, shape (n,)
        Eigenvalues from SVD of covariance matrix C.
    lower : float, optional
        Lower integration limit for COS method. Default is -10.0.
    upper : float, optional
        Upper integration limit for COS method. Default is 10.0.

    Returns
    -------
    ndarray
        PDF values at x for sqrt(S) where S = a + b^T*Z + Z^T*C*Z.
    """
    return (
        2.0 * x * pdf_vix2_cos(x**2, N=N, a=a, b=b, U=U, D=D, lower=lower, upper=upper)
    )


def cumulants_quadratic_gaussian(a, b, C, upto=6):
    """
    Compute cumulants of a Gaussian quadratic form S = a + b^T Z + Z^T C Z.

    Parameters
    ----------
    a : float
        Scalar offset.
    b : array_like, shape (n,)
        Linear term.
    C : array_like, shape (n, n)
        Symmetric matrix of the quadratic term.
    upto : int, default 6
        Highest cumulant order to return.

    Returns
    -------
    list
        kappa with length up to + 1, where kappa[r] = κ_r for r >= 1 (kappa[0] = None).
    """
    lam = np.linalg.eigvalsh(C)  # stable for symmetric C
    kappa = [None] * (upto + 1)
    kappa[1] = a + lam.sum()
    kappa[2] = np.dot(b, b) + 2 * np.sum(lam**2)
    for r in range(3, upto + 1):
        kappa[r] = (2 ** (r - 1)) * sp.factorial(r - 1) * np.sum(lam**r)
    return kappa


def moments_from_cumulants(kappa):
    """
    Convert cumulants to raw moments (explicit up to order 4).

    Parameters
    ----------
    kappa : sequence
        Cumulants with kappa[r] = κ_r; index 0 unused.

    Returns
    -------
    list
        mu with mu[r] = E[X^r] for r = 1..n (mu[0] = None).
    """
    n = len(kappa) - 1
    mu = [None] * (n + 1)
    if n >= 1:
        mu[1] = kappa[1]
    if n >= 2:
        mu[2] = kappa[2] + kappa[1] ** 2
    if n >= 3:
        mu[3] = kappa[3] + 3 * kappa[2] * kappa[1] + kappa[1] ** 3
    if n >= 4:
        mu[4] = (
            kappa[4]
            + 4 * kappa[3] * kappa[1]
            + 3 * kappa[2] ** 2
            + 6 * kappa[2] * kappa[1] ** 2
            + kappa[1] ** 4
        )
    return mu
