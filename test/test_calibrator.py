
import numpy as np
import pytest

from types import SimpleNamespace
from unittest.mock import Mock

from synpop.calibration.calibrator import HIPF


@pytest.fixture
def make_hipf():
    def _make(
        weights,
        sizes,
        n=100,
        nu=160,
        x_h=None,
        h_m_ar=None,
        x_p=None,
        p_m_ar=None,
    ):
        weights = np.asarray(weights, dtype=float)
        sizes = np.asarray(sizes, dtype=float)
        count = len(weights)

        model = HIPF.__new__(HIPF)

        model.data = SimpleNamespace(
            h_sizes=sizes,
            total_households=n,
            total_individuals=nu,
            x_h=np.asarray(
                x_h if x_h is not None else np.empty((count, 0))
            ),
            h_m_ar=np.asarray(
                h_m_ar if h_m_ar is not None else []
            ),
            x_p=np.asarray(
                x_p if x_p is not None else np.empty((count, 0))
            ),
            p_m_ar=np.asarray(
                p_m_ar if p_m_ar is not None else []
            ),
        )

        model.w = weights.copy()
        model.is_converged = False

        return model

    return _make


class TestHIPF:

    @pytest.mark.parametrize(
        "weights, nu, expected",
        [
            ([50, 50], 160, [40, 60]),
            ([30, 70], 150, [50, 50]),
            ([40, 60], 160, [40, 60]),
        ],
    )
    def test_adjust_weight(
        self, make_hipf, weights, nu, expected
    ):
        model = make_hipf(
            weights=weights,
            sizes=[1, 2],
            n=100,
            nu=nu,
        )

        model._adjust_weight(tolerance=1e-12)

        np.testing.assert_allclose(
            model.w, expected, atol=1e-8
        )
        assert np.sum(model.w) == pytest.approx(100)
        assert np.dot(model.data.h_sizes, model.w) == pytest.approx(nu)

    def test_adjust_weight_with_zero_weights(self, make_hipf):
        model = make_hipf(
            weights=[50, 50, 0],
            sizes=[1, 2, 3],
            n=100,
            nu=160,
        )

        model._adjust_weight(tolerance=1e-12)

        np.testing.assert_allclose(
            model.w, [40, 60, 0], atol=1e-8
        )

    @pytest.mark.parametrize("nu", [100, 200, 250])
    def test_adjust_weight_infeasible_target(
        self, make_hipf, nu
    ):
        model = make_hipf(
            weights=[50, 50],
            sizes=[1, 2],
            n=100,
            nu=nu,
        )

        with pytest.raises(ValueError, match="Target mean"):
            model._adjust_weight(tolerance=1e-12)

    def test_adjust_weight_negative_weights(self, make_hipf):
        model = make_hipf(
            weights=[50, -10],
            sizes=[1, 2],
        )

        with pytest.raises(ValueError, match="Weights must"):
            model._adjust_weight(tolerance=1e-12)

    def test_update_household_marginals(self, make_hipf):
        model = make_hipf(
            weights=[2, 3],
            sizes=[1, 2],
            x_h=[[1, 0], [0, 1]],
            h_m_ar=[4, 6],
        )

        model._update_by_marginals()

        np.testing.assert_allclose(model.w, [4, 6])


    def test_update_individual_marginals(self, make_hipf):
        model = make_hipf(
            weights=[2, 3],
            sizes=[1, 2],
            x_p=[[1], [1]],
            p_m_ar=[10],
        )

        model._update_by_marginals()

        np.testing.assert_allclose(model.w, [4, 4.5])

    def test_fit_converges(self, make_hipf, monkeypatch):
        model = make_hipf(
            weights=[50, 50],
            sizes=[1, 2],
        )

        update_mock = Mock()
        adjust_mock = Mock()
        convergence_mock = Mock(side_effect=[False, True])

        monkeypatch.setattr(
            HIPF, "_update_by_marginals", update_mock
        )
        monkeypatch.setattr(
            HIPF, "_adjust_weight", adjust_mock
        )
        monkeypatch.setattr(
            HIPF, "check_convergence", convergence_mock
        )

        # Supply the error attribute expected by fit().
        monkeypatch.setattr(
            HIPF,
            "mean_relative_error",
            property(lambda self: 0.0),
            raising=False,
        )

        result = model.fit(max_iter=10, tolerance=1e-8)

        assert model.is_converged is True
        assert update_mock.call_count == 2
        assert adjust_mock.call_count == 2
        assert convergence_mock.call_count == 2
        assert result is not None


    def test_fit_max_iterations(self, make_hipf, monkeypatch):
        model = make_hipf(
            weights=[50, 50],
            sizes=[1, 2],
        )

        update_mock = Mock()
        adjust_mock = Mock()

        monkeypatch.setattr(
            HIPF, "_update_by_marginals", update_mock
        )
        monkeypatch.setattr(
            HIPF, "_adjust_weight", adjust_mock
        )
        monkeypatch.setattr(
            HIPF, "check_convergence", Mock(return_value=False)
        )
        monkeypatch.setattr(
            HIPF,
            "mean_relative_error",
            property(lambda self: 0.1),
            raising=False,
        )

        result = model.fit(max_iter=3)

        assert model.is_converged is False
        assert update_mock.call_count == 3
        assert adjust_mock.call_count == 3
        assert result is not None
