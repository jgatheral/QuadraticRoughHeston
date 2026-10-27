import numpy as np

from fwd_var_curve import xi_curve


def test_forward_variance_intervals():
    curve = xi_curve(np.array([0.5, 1.0, 2.0]), np.array([0.02, 0.05, 0.09]))
    times = np.array([0.0, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0])
    np.testing.assert_allclose(
        curve["xi_curve"](times), [0.04, 0.04, 0.06, 0.06, 0.04, 0.04, 0.04]
    )
    np.testing.assert_allclose(curve["fit_errs"], 0.0, atol=1e-14)
