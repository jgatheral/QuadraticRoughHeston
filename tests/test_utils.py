import numpy as np
import pytest

from utils import gauss_legendre


def test_quadrature_integrates_polynomial():
    nodes, weights = gauss_legendre(1.0, 3.0, 3)
    assert np.dot(weights, nodes**4) == pytest.approx((3**5 - 1) / 5)
