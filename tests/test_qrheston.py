import numpy as np
import pytest

from qrheston import QuadraticRoughHeston


@pytest.fixture
def model():
    return QuadraticRoughHeston(xi0=lambda t: 0.09, c=0.005, nu=0.4, lam=5, al=0.65)


def test_simulation(model):
    paths, steps, expiry = 64, 8, 0.1
    sim = model.simulate(
        paths, steps, spx_expiries=[expiry], vix_expiries=[expiry],
        output="all", nvix=2, h_ssr=lambda t: 0.0, seed=1234,
    )["all"][expiry]
    for key in ("X", "v", "w", "vix"):
        assert sim[key].shape == (paths,)
        assert np.isfinite(sim[key]).all()
    assert sim["chi"].shape == (steps, paths)
    assert (sim["v"] >= model.c).all()
    assert (sim["w"] > 0).all()
    assert (sim["vix"] > 0).all()
    np.testing.assert_allclose(sim["X"], -0.5 * sim["w"] - sim["chi"].sum(axis=0))
    for key in ("X", "v", "w"):
        np.testing.assert_array_equal(sim[key], sim[f"{key}_h"])


def test_ssr(model):
    expiries = np.array([0.05, 0.1])
    kwargs = dict(
        expiries=expiries, paths=512, steps=8,
        h_ssr=lambda t: 0.01 * np.sqrt(t), seed=1234,
    )
    ssr = model.compute_ssr(**kwargs)
    assert ssr.shape == expiries.shape
    assert np.isfinite(ssr).all()
    np.testing.assert_array_equal(ssr, model.compute_ssr(**kwargs))


@pytest.mark.parametrize("method", ["y0", "integral_K0", "integral_K00"])
def test_integer_times(model, method):
    evaluate = getattr(model, method)
    np.testing.assert_allclose(evaluate([0, 1]), evaluate([0.0, 1.0]))


def test_joint_expiries_infer_output(model):
    kwargs = dict(paths=16, steps=4, spx_expiries=[0.1], vix_expiries=[0.1],
                  nvix=2, seed=1234)
    inferred = model.simulate(**kwargs)
    explicit = model.simulate(**kwargs, output="spx-vix")
    for asset, key in [("spx", "X"), ("vix", "vix")]:
        np.testing.assert_array_equal(inferred[asset][0.1][key], explicit[asset][0.1][key])
