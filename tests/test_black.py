import numpy as np
import pytest

from black import black_impvol, black_impvol_brentq, black_price


def test_put_call_parity():
    strikes = np.array([80.0, 100.0, 120.0])
    call = black_price(strikes, T=0.5, F=100, vol=0.2, r=0.03)
    put = black_price(strikes, T=0.5, F=100, vol=0.2, r=0.03, opttype=-1)
    np.testing.assert_allclose(
        call - put, np.exp(-0.03 * 0.5) * (100 - strikes), atol=1e-12
    )


@pytest.mark.parametrize("solver", [black_impvol, black_impvol_brentq])
@pytest.mark.parametrize("opttype", [-1, 1])
def test_implied_volatility_round_trip(solver, opttype):
    strikes = np.array([90.0, 100.0, 110.0])
    vols = np.array([0.2, 0.3, 0.4])
    prices = black_price(strikes, 0.5, 100, vols, r=0.03, opttype=opttype)
    actual = solver(strikes, 0.5, 100, prices, r=0.03, opttype=opttype)
    np.testing.assert_allclose(actual, vols, rtol=2e-6)


@pytest.mark.parametrize("opttype", [0, 2, 0.5, [1, 2], [-1, 0]])
def test_invalid_option_type(opttype):
    with pytest.raises(ValueError, match="opttype"):
        black_impvol([90, 110], 0.5, 100, [12.0, 3.0], opttype=opttype)


def test_integer_strikes_at_invalid_expiry():
    result = black_impvol([90, 110], 0, 100, [10, 0])
    assert np.isnan(result).all()
