"""Calibrators for calibrating results from Bayesian network to match marginals"""

import logging

logger = logging.getLogger(__name__)
from synpop.calibration.matrix import CalibrationMatrix, CalibrationResult
from scipy.optimize import minimize
from scipy.special import logsumexp
from abc import ABC, abstractmethod
import numpy as np

# TODO Transform np.array to sparse matrix to improve the computation


class BaseCalibrator(ABC):
    """Reweighting algorithms base."""

    def __init__(self, data: CalibrationMatrix):
        self.data = data
        self.w = np.copy(data.w_initial)
        self.relative_abs_error = 0.0
        self.is_converged = False

    def check_convergence(self, tolerance: float) -> bool:
        t = self.data.t
        X = self.data.X
        current_error = np.sum(np.abs(np.dot(X.T, self.w) / t - 1)) / len(t)
        delta = np.abs(self.relative_abs_error - current_error)
        self.relative_abs_error = current_error
        return bool(delta < tolerance)

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
        error_msg = (
            "The calibration methods should be in 'linear', 'raking', or 'logit'."
        )
        logger.error(error_msg)
        raise

    def _update(self, lam, F, F_prime, m_x, learning_rate: float = 0.01):
        u = np.dot(self.data.X, lam)  # (s,)
        # \phi(\lambda) = \sum_{k \in s} d_k (F(x_k^T \lambda) - 1) x_k (Eq. 4.4)
        phi = np.dot(self.d * (F(u) - 1), self.data.X)  # (J,)
        diff = m_x - phi
        # \phi'(\lambda) = \sum_{k \in s} d_k F'(x_k^T \lambda) x_k x_k^T (Eq. 4.6)
        w_prime = self.d * F_prime(u)
        phi_prime = np.dot(self.data.X.T, w_prime[:, np.newaxis] * self.data.X)
        phi_prime_inverse = np.linalg.pinv(phi_prime)
        # \lambda_{i+1} = \lambda_i + \phi'(\lambda_i)^{-1} (m_x - \phi(\lambda_i)) (Eq. 4.5)
        lam = lam + learning_rate * np.dot(phi_prime_inverse, diff)
        self.u = np.dot(self.data.X, lam)
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
        J = self.data.X.shape[1]  # (s, J)
        lam = np.zeros(J)  # (J,)
        m_x = self.data.t - np.dot(self.w, self.data.X)  # (J,)
        F, F_prime = self._get_cali_method(method, bounds)
        for i in range(max_iter):
            try:
                lam = self._update(lam, F, F_prime, m_x)
            except Exception as e:
                logger.error(e)

            if self.check_convergence(tolerance):
                logger.info(
                    f"GR converged after {i+1} iterations. Final relative absolute error is: {self.relative_abs_error:.6f}"
                )
                self.is_converged = True
                return CalibrationResult(
                    self.w, self.is_converged, i + 1, self.relative_abs_error
                )

        logger.error(
            f"GR did not converge. Final relative absolute error is: {self.relative_abs_error:.6f}"
        )
        return CalibrationResult(
            self.w, self.is_converged, max_iter, self.relative_abs_error
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
        for j in range(self.data.X.shape[1]):
            current_marginals = np.dot(self.data.X[:, j], self.w)
            rho = np.divide(
                self.data.t[j],
                current_marginals,
                out=np.ones_like(self.data.t[j], dtype=float),
                where=current_marginals != 0,
            )
            self.w = np.where(self.data.X[:, j] == 0, self.w, self.w * rho)

    def fit(self, max_iter: int = 100, tolerance: float = 1e-8, **kwargs):
        for i in range(max_iter):
            try:
                self._update()
                if self.check_convergence(tolerance):
                    logger.info(
                        f"IPU Converged after {i+1} iterations. Final relative absolute error is: {self.relative_abs_error:.6f}"
                    )
                    self.is_converged = True
                    return CalibrationResult(
                        self.w, self.is_converged, i + 1, self.relative_abs_error
                    )
            except Exception as e:
                logger.error(f"IPU:{e}")

        logger.error(
            f"IPU did not converge. Final relative absolute error: {self.relative_abs_error:.6f}"
        )
        return CalibrationResult(
            self.w, self.is_converged, max_iter, self.relative_abs_error
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
        # n is the household total number,
        # v is the individuals total number,
        """
        Using Newton-Raphson method

        Args:
            tolerance (_type_): _description_

        Returns:
            _type_: _description_
        """
        coef = (
            self.data.h_sizes
            * (self.data.total_households / self.data.total_individuals)
            - 1
        )

        def newton_raphson(tolerance):
            d = 1
            while True:
                g_d = np.sum(coef * self.w * (d**self.data.h_sizes))
                g_prime_d = np.sum(
                    self.data.h_sizes * coef * self.w * (d ** (self.data.h_sizes - 1))
                )
                if abs(g_prime_d) < 1e-12:
                    logger.warning("Newton-Raphson: Derivative approached zero.")
                    return d
                step = g_d / g_prime_d  # d_k = d_{k-1} + g(d)/g'(d)

                if abs(step) < tolerance:
                    return d
                d -= step

        d = newton_raphson(tolerance)

        try:
            c = self.data.total_individuals / np.sum(
                self.data.h_sizes * self.w * (d**self.data.h_sizes)
            )
            ratio = c * (d**self.data.h_sizes)
            self.w = self.w * ratio
        except Exception:
            logger.error("Failed adjusting the persons-per-household ratio")

    def fit(self, max_iter: int = 100, tolerance: float = 1e-8, **kwargs):
        for i in range(max_iter):
            try:
                self._update_by_marginals()
                self._adjust_weight(tolerance)
            except Exception as e:
                logger.error(f"HIPF:{e}")
            if self.check_convergence(tolerance):
                logger.info(
                    f"HIPF Converged after {i+1} iterations. Final relative absolute error is: {self.relative_abs_error:.6f}"
                )
                self.is_converged = True
                return CalibrationResult(
                    self.w, self.is_converged, i + 1, self.relative_abs_error
                )
        logger.error(
            f"HIPF did not converge. Final relative absolute error: {self.relative_abs_error:.6f}"
        )
        return CalibrationResult(
            self.w, self.is_converged, max_iter, self.relative_abs_error
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
                self.w, self.is_converged, result.nit, self.relative_abs_error
            )
        else:
            logger.error(f"CE failed to converge: {result.message}")
            return CalibrationResult(
                self.w, self.is_converged, max_iter, self.relative_abs_error
            )


CALIBRATOR = {
    "ipu": IPU,
    "gr": GR,
    "hipf": HIPF,
    "cross_entropy": CrossEntropy,
}
