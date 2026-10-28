import numpy as np
import pytest
import pandas as pd

from utils import gauss_legendre, var_swap_robust


def test_quadrature_integrates_polynomial():
    nodes, weights = gauss_legendre(1.0, 3.0, 3)
    assert np.dot(weights, nodes**4) == pytest.approx((3**5 - 1) / 5)


def test_variance_swap_selected_expiries():
    data = pd.DataFrame([
        {"Texp": t, "Fwd": 100.0, "Strike": strike, "Bid": vol - .01, "Ask": vol + .01}
        for t, vol in [(0.1, .2), (0.3, .3), (0.7, .4)]
        for strike in [80, 90, 100, 110, 120]
    ])
    result = var_swap_robust(data, slices=[2, 0])
    np.testing.assert_allclose(result["expiries"], [0.7, 0.1])
    np.testing.assert_allclose(result["vs_mid"], [.4**2, .2**2])
    np.testing.assert_allclose(result["vs_bid"], [.39**2, .19**2])
    np.testing.assert_allclose(result["vs_ask"], [.41**2, .21**2])
