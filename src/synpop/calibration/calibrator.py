"""Calibrators for calibrating results from Bayesian network to match marginals"""

import logging

logger = logging.getLogger(__name__)
from synpop.calibration.matrix import CalibrationMatrix, CalibrationResult
from scipy.optimize import minimize
from scipy.special import logsumexp
from abc import ABC, abstractmethod
import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp

# TODO Transform np.array to sparse matrix to improve the computation


class BaseCalibrator(ABC):
    """Reweighting algorithms base."""

    def __init__(self, data: CalibrationMatrix):
        self.data = data
        self.X = data.X
        self.t = data.t
        self.w = np.copy(data.w_initial)
        self.mean_relative_error = 0.0
        self.is_converged = False

    def check_convergence(self, tolerance: float) -> bool:
        relative_errors = np.abs((self.X.T @ self.w - self.t) / self.t)
        self.mean_relative_error = float(np.mean(relative_errors))
        # Using the maximum relative error as the convergence indicator
        return bool(np.max(relative_errors) < tolerance)

    @abstractmethod
    def fit(
        self, max_iter: int = 100, tolerance: float = 1e-8, **kwargs
    ) -> CalibrationResult:
        pass


class GR(BaseCalibrator):

    def __init__(self, data):
        super().__init__(data)
        self.u: np.ndarray | None = None
        self.d = self.w

    def _get_cali_method(self, method, bounds):

        if method == "linear":
            return lambda u: 1 + u, lambda u: 1

        elif method == "raking":
            return lambda u: np.exp(u), lambda u: np.exp(u)

        elif method == "logit":
            L, U = bounds
            if L >= 1 or U <= 1:
                raise ValueError("Lower bound must be < 1 and upper bound > 1")

            A = (U - L) / ((1 - L) * (U - 1))

            def F(u):
                safe_Au = np.clip(A * u, a_min=-700, a_max=700)
                exp_Au = np.exp(safe_Au)
                return (L * (U - 1) + U * (1 - L) * exp_Au) / (U - 1 + (1 - L) * exp_Au)

            def F_prime(u):
                return A * (U - F(u)) * (F(u) - L) / (U - L)

            return F, F_prime
        raise ValueError(
            "The calibration methods should be in 'linear', 'raking', or 'logit'."
        )

    def _update(self, lam, F, F_prime, m_x, learning_rate: float = 0.01):
        u = np.dot(self.X, lam)  # (s,)
        # \phi(\lambda) = \sum_{k \in s} d_k (F(x_k^T \lambda) - 1) x_k (Eq. 4.4)
        phi = np.dot(self.d * (F(u) - 1), self.X)  # (J,)
        diff = m_x - phi
        # \phi'(\lambda) = \sum_{k \in s} d_k F'(x_k^T \lambda) x_k x_k^T (Eq. 4.6)
        w_prime = self.d * F_prime(u)
        phi_prime = np.dot(self.X.T, w_prime[:, np.newaxis] * self.X)
        phi_prime_inverse = np.linalg.pinv(phi_prime)
        # \lambda_{i+1} = \lambda_i + \phi'(\lambda_i)^{-1} (m_x - \phi(\lambda_i)) (Eq. 4.5)
        lam = lam + learning_rate * np.dot(phi_prime_inverse, diff)
        self.u = np.dot(self.X, lam)
        self.w = self.d * F(self.u)
        return lam

    def fit(
        self,
        max_iter: int = 100,
        tolerance: float = 1e-8,
        bounds: tuple = (0.001, 50.0),
        method: str = "raking",
        **kwargs,
    ) -> CalibrationResult:
        """_summary_

        Args:
            method (str, optional): The calibration methods should be in 'linear', 'raking', or 'logit'. Defaults to 'raking'.
            max_iter (int, optional): The maximum number iteration. Defaults to 100.
            bounds (tuple, optional): The bounds for logit method. Defaults to (0.1,5.0).
            tolerance (float, optional): The tolerance for fitting step. Defaults to 1e-8.

        Returns:
            np.ndarray: _description_
        """
        J = self.X.shape[1]  # (s, J)
        lam = np.zeros(J)  # (J,)
        m_x = self.t - np.dot(self.w, self.X)  # (J,)
        F, F_prime = self._get_cali_method(method, bounds)
        for i in range(max_iter):
            lam = self._update(lam, F, F_prime, m_x)

            if self.check_convergence(tolerance):
                logger.info(
                    f"GR converged after {i+1} iterations. Final relative absolute error is: {self.mean_relative_error:.6f}"
                )
                self.is_converged = True
                return CalibrationResult(
                    self.w, self.is_converged, i + 1, self.mean_relative_error
                )

        logger.warning(
            f"GR did not converge after {max_iter} iterations. Final relative absolute error is: {self.mean_relative_error:.6f}"
        )
        return CalibrationResult(
            self.w, self.is_converged, max_iter, self.mean_relative_error
        )


class IPU(BaseCalibrator):
    """
    Ye, X., Konduri, K. C., Pendyala, R. M., Sana, B., & Waddell, P. (2009).
    Methodology to match distributions of both household and person attributes in generation of synthetic populations (Nos. 09–2096).
    Article 09–2096. Transportation Research Board 88th Annual MeetingTransportation Research Board.
    https://trid.trb.org/View/881554

    Args:
        X (np.ndarray): shape(s, J), a matrix of variables.
        t (np.ndarray): shape(J,), a vector of marginals of variables.
    """

    def _update(self):
        for j in range(self.X.shape[1]):
            current_marginals = np.dot(self.X[:, j], self.w)
            rho = np.divide(
                self.t[j],
                current_marginals,
                out=np.ones_like(self.t[j], dtype=float),
                where=current_marginals != 0,
            )
            self.w = np.where(self.X[:, j] == 0, self.w, self.w * rho)

    def fit(self, max_iter: int = 100, tolerance: float = 1e-8, **kwargs):
        for i in range(max_iter):
            self._update()
            if self.check_convergence(tolerance):
                logger.info(
                    f"IPU Converged after {i+1} iterations. Final relative absolute error is: {self.mean_relative_error:.6f}"
                )
                self.is_converged = True
                return CalibrationResult(
                    self.w, self.is_converged, i + 1, self.mean_relative_error
                )

        logger.warning(
            f"IPU did not converge after {max_iter} iterations. Final relative absolute error: {self.mean_relative_error:.6f}"
        )
        return CalibrationResult(
            self.w, self.is_converged, max_iter, self.mean_relative_error
        )


class HIPF(BaseCalibrator):

    def __init__(self, data) -> None:
        """
        Kirill Mueller & Kay W. Axhausen, 2011. "Hierarchical IPF: Generating a synthetic population for Switzerland,"
        ERSA conference papers ersa11p305, European Regional Science Association.
        """
        super().__init__(data)

    def _update_by_marginals(self):
        for i in range(len(self.data.h_m_ar)):
            current_marginal = np.dot(self.data.x_h[:, i], self.w)

            if current_marginal > 0:
                ratio = self.data.h_m_ar[i] / current_marginal
                self.w *= np.power(ratio, self.data.x_h[:, i])

        for j in range(len(self.data.p_m_ar)):
            current_marginal = np.dot(self.data.x_p[:, j], self.w)

            if current_marginal > 0:
                ratio = self.data.p_m_ar[j] / current_marginal
                # TODO Here the can introduce a mastrix to filter person with corresponding category.
                persons_with_trait = self.data.x_p[:, j]
                persons_without_trait = self.data.h_sizes - self.data.x_p[:, j]
                # Average the ratios by the household size (step 4)
                average_ratio = (
                    persons_with_trait * ratio + persons_without_trait
                ) / self.data.h_sizes
                self.w *= average_ratio

    def _adjust_weight(self, tolerance):
        """
        Adjust household weights to match target household and
        individual totals by minimizing relative entropy.
        Solve for t = log(d), where:
            f'_h = c * exp(t * p_h) * f_h
        Args:
            tolerance (_type_): _description_

        Returns:
            _type_: _description_
        """
        p = self.data.h_sizes
        w = self.w
        n = self.data.total_households
        nu = self.data.total_individuals
        if not np.all(np.isfinite(w)) or np.any(w < 0):
            raise ValueError("Weights must be finite and nonnegative.")
        active = w > 0
        if not np.any(active):
            raise ValueError("All household weights are zero.")
        sizes = p[active]
        log_w = np.log(w[active])
        target_mean = nu / n
        if not sizes.min() < target_mean < sizes.max():
            raise ValueError(
                "Target mean must be strictly between the minimum "
                "and maximum supported household sizes."
            )

        def g(t: float) -> float:
            """Adjusted mean household size minus target mean."""
            z = log_w + sizes * t
            probabilities = np.exp(z - logsumexp(z))
            return float(np.dot(sizes, probabilities) - target_mean)

        # Find an interval for Brent's method.
        lower, upper = -1.0, 1.0

        for _ in range(60):
            if g(lower) <= 0 <= g(upper):
                break
            lower *= 2
            upper *= 2
        else:
            raise RuntimeError("Could not bracket the root.")
        # Brent's method
        t = brentq(g, lower, upper, xtol=tolerance)
        z = log_w + sizes * t
        adjusted_w = np.zeros_like(w)
        adjusted_w[active] = n * np.exp(z - logsumexp(z))
        self.w = adjusted_w

    def fit(self, max_iter: int = 100, tolerance: float = 1e-8, **kwargs):
        for i in range(max_iter):
            self._update_by_marginals()
            self._adjust_weight(tolerance)
            if self.check_convergence(tolerance):
                logger.info(
                    f"HIPF Converged after {i+1} iterations. Final relative absolute error is: {self.mean_relative_error:.6f}"
                )
                self.is_converged = True
                return CalibrationResult(
                    self.w, self.is_converged, i + 1, self.mean_relative_error
                )
        logger.warning(
            f"HIPF did not converge after {max_iter} iterations. Final relative absolute error: {self.mean_relative_error:.6f}"
        )
        return CalibrationResult(
            self.w, self.is_converged, max_iter, self.mean_relative_error
        )


class CrossEntropy(BaseCalibrator):
    def __init__(self, data):
        super().__init__(data)
        self.q = self.w / np.sum(self.w)
        self.total_target_weights = self.data.total_households
        self.log_q = np.log(np.clip(self.q, 1e-12, None))

    def _objective(self, lambda_vec, h_m_scaled, p_m_scaled, alpha):
        num_h_constraints = len(h_m_scaled)
        lambda_h = lambda_vec[:num_h_constraints]
        lambda_p = lambda_vec[num_h_constraints:]

        exponent = (self.data.x_h @ lambda_h) + (self.data.x_p @ lambda_p)
        ln_Z = logsumexp(self.log_q + exponent)

        l2_penalty = 0.5 * alpha * np.sum(lambda_vec**2)

        obj_val = (
            ln_Z
            - np.dot(lambda_h, h_m_scaled)
            - np.dot(lambda_p, p_m_scaled)
            + l2_penalty
        )

        log_p_hat = self.log_q + exponent - ln_Z
        p_hat = np.exp(log_p_hat)

        grad_h = (self.data.x_h.T @ p_hat) - h_m_scaled
        grad_p = (self.data.x_p.T @ p_hat) - p_m_scaled
        grad = np.concatenate([grad_h, grad_p]) + alpha * lambda_vec

        return obj_val, grad

    def fit(self, max_iter=100, tolerance=1e-8, alpha=1e-3, **kwargs):
        h_m_scaled = self.data.h_m_ar / self.total_target_weights
        p_m_scaled = self.data.p_m_ar / self.total_target_weights
        initial_lambda = np.zeros(len(h_m_scaled) + len(p_m_scaled))

        result = minimize(
            fun=self._objective,
            x0=initial_lambda,
            args=(h_m_scaled, p_m_scaled, alpha),
            method="BFGS",
            jac=True,
            tol=tolerance,
            options={"maxiter": max_iter},
        )
        self.is_converged = result.success

        if result.success:
            logger.info("CE converged successfully!")
            optimal_lambda = result.x
            lambda_h = optimal_lambda[: len(h_m_scaled)]
            lambda_p = optimal_lambda[len(h_m_scaled) :]

            final_exponent = (self.data.x_h @ lambda_h) + (self.data.x_p @ lambda_p)
            final_ln_Z = logsumexp(self.log_q + final_exponent)

            optimal_p_hat = np.exp(self.log_q + final_exponent - final_ln_Z)
            self.w = optimal_p_hat * self.total_target_weights

            return CalibrationResult(
                self.w, self.is_converged, result.nit, self.mean_relative_error
            )
        else:
            logger.error(f"CE failed to converge: {result.message}")
            return CalibrationResult(
                self.w, self.is_converged, max_iter, self.mean_relative_error
            )


CALIBRATOR = {
    "ipu": IPU,
    "gr": GR,
    "hipf": HIPF,
    "cross_entropy": CrossEntropy,
}
