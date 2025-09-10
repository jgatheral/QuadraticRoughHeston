import numpy as np
from scipy import integrate, linalg


def charfunc_quadratic_normal(t, a, b, C):
    """
    Characteristic function of Gaussian quadratic form S = a + b^T*Z + Z^T*C*Z.

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
        Characteristic function value E[exp(itS)] where Z ~ N(0, I).
    """
    # TODO: vectorize for t array
    id = np.eye(len(b))
    shift_id = id - 2j * t * C
    inv = linalg.solve(shift_id, id)
    det = linalg.det(shift_id)
    factor = np.exp(1j * a * t) * det ** (-0.5)
    return factor * np.exp(-0.5 * t**2 * b.T @ inv @ b)


def charfunc_quadratic_normal_svd(t, a, b, U, D):
    """
    Characteristic function of Gaussian quadratic form using SVD decomposition.

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
        Characteristic function value E[exp(itS)] where S = a + b^T*Z + Z^T*C*Z
        and C = U @ diag(D) @ U^T.
    """
    # TODO: vectorize for t array
    # inverse_shift_id = U @ np.diag(1.0 / (1.0 - 2j * t * D)) @ U.T
    det_shift_id = np.prod(1.0 - 2j * t * D)
    U_t_b = U.T @ b
    factor = np.exp(1j * a * t) * det_shift_id ** (-0.5)
    return factor * np.exp(-0.5 * t**2 * np.sum(U_t_b**2 / (1.0 - 2j * t * D)))


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
