"""Log-likelihood functions for proper likelihood-based calibration."""

from typing import Optional

import numpy as np
import polars as pl
import scipy.stats as sps

from ..alignment import AlignedData
from .likelihood import LogLikelihoodFn


def beta_binomial_loglik_per_rep(
    x_col: str,
    n_col: str,
    alpha: float,
    beta: float,
) -> LogLikelihoodFn:
    """
    Beta-binomial log-likelihood per replicate.

    For the beta-binomial log-likelihood, we begin with a Beta(alpha, beta) prior
    and subsequently observe from the simulation some number of successes (s_x)
    out of some total number of trials (s_n).

    The result is a Beta(s_x+alpha, s_n-s_x+beta) posterior.

    We then compare this to the real data, which has o_x successes (positives)
    in o_n trials (total observations). We use a beta-binomial likelihood:

    p(x|n, a, b) = (n choose x) B(x+a, n-x+b) / B(a, b)

    where:
        x = number of successes from the observed data
        n = number of trials from the observed data
        a = number of successes from the simulation plus alpha from the prior
        b = number of failures from the simulation plus beta from the prior
    and B is the beta function, B(x, y) = Gamma(x)Gamma(y)/Gamma(x+y)

    Returns a function that computes per-replicate sums of log-likelihoods.

    Parameters
    ----------
    x_col : str
        Column name for the number of successes in both simulated and observed data
    n_col : str
        Column name for the number of trials in both simulated and observed data
    alpha : float
        Alpha parameter for Beta prior (must be positive)
    beta : float
        Beta parameter for Beta prior (must be positive)

    Returns
    -------
    LogLikelihoodFn
        Function that returns np.ndarray of shape (R,) with per-replicate log-likelihood sums

    Raises
    ------
    ValueError
        If alpha or beta are not positive, or if replicate_col is None
    """
    if alpha <= 0 or beta <= 0:
        raise ValueError(f"Prior parameters must be positive: alpha={alpha}, beta={beta}")

    def fn(aligned: AlignedData) -> np.ndarray:
        """
        Compute per-replicate log-likelihoods for beta-binomial model.

        Parameters:
            aligned: Aligned data with replicate column

        Returns:
            np.ndarray of shape (R,) with per-replicate log-likelihood sums

        Raises:
            ValueError: If replicate_col is None
        """
        if aligned.replicate_col is None:
            raise ValueError(
                "Likelihood evaluation requires a replicate column. "
                "Ensure your alignment strategy produces replicated data."
            )

        # Extract columns
        s_x = aligned.get_sim_col(x_col)
        s_n = aligned.get_sim_col(n_col)
        o_x = aligned.get_obs_col(x_col).cast(pl.Int32)
        o_n = aligned.get_obs_col(n_col).cast(pl.Int32)

        # Apply prior to get posterior parameters
        posterior_alpha = s_x + alpha
        posterior_beta = s_n - s_x + beta

        # Compute per-row log-likelihoods
        loglik = sps.betabinom.logpmf(k=o_x, n=o_n, a=posterior_alpha, b=posterior_beta)
        loglik_series = pl.Series(name="loglik", values=loglik)

        # Add to dataframe and group by replicate
        df_with_loglik = aligned.data.with_columns(loglik_series)
        grouped = df_with_loglik.group_by(aligned.replicate_col).agg(
            pl.col("loglik").sum().alias("rep_loglik")
        )

        # Return as numpy array, sorted by replicate index
        return grouped.sort(aligned.replicate_col)["rep_loglik"].to_numpy()

    return fn


def binomial_loglik_per_rep(
    p_col: str,
    x_col: str,
    n_col: str,
) -> LogLikelihoodFn:
    """
    Binomial log-likelihood per replicate.

    The binomial log-likelihood assesses the likelihood of observing x successes
    in n trials given a probability p. The "p" values come from the simulation
    data (e.g., disease prevalence). The "x" and "n" values come from the
    observed data.

    Returns a function that computes per-replicate sums of log-likelihoods.

    Parameters
    ----------
    p_col : str
        Column name for the probability in the simulated data
    x_col : str
        Column name for the number of successes in the observed data
    n_col : str
        Column name for the number of trials in the observed data

    Returns
    -------
    LogLikelihoodFn
        Function that returns np.ndarray of shape (R,) with per-replicate log-likelihood sums

    Raises
    ------
    ValueError
        If replicate_col is None
    """

    def fn(aligned: AlignedData) -> np.ndarray:
        """
        Compute per-replicate log-likelihoods for binomial model.

        Parameters:
            aligned: Aligned data with replicate column

        Returns:
            np.ndarray of shape (R,) with per-replicate log-likelihood sums

        Raises:
            ValueError: If replicate_col is None
        """
        if aligned.replicate_col is None:
            raise ValueError(
                "Likelihood evaluation requires a replicate column. "
                "Ensure your alignment strategy produces replicated data."
            )

        # Extract columns
        p = aligned.get_sim_col(p_col)
        x = aligned.get_obs_col(x_col).cast(pl.Int32)
        n = aligned.get_obs_col(n_col).cast(pl.Int32)

        # Compute per-row log-likelihoods
        loglik = sps.binom.logpmf(k=x, n=n, p=p)
        loglik_series = pl.Series(name="loglik", values=loglik)

        # Add to dataframe and group by replicate
        df_with_loglik = aligned.data.with_columns(loglik_series)
        grouped = df_with_loglik.group_by(aligned.replicate_col).agg(
            pl.col("loglik").sum().alias("rep_loglik")
        )

        # Return as numpy array, sorted by replicate index
        return grouped.sort(aligned.replicate_col)["rep_loglik"].to_numpy()

    return fn


def beta_binomial_loglik_factory(
    x_col: str,
    n_col: str,
    prior: Optional[tuple | list] = (1, 1),
) -> LogLikelihoodFn:
    """
    Factory function for beta-binomial log-likelihood with prior validation.

    This is a convenience wrapper around beta_binomial_loglik_per_rep that
    validates the prior tuple format.

    Parameters
    ----------
    x_col : str
        Column name for the number of successes
    n_col : str
        Column name for the number of trials
    prior : tuple | list, optional
        Beta prior parameters (alpha, beta). Default is (1, 1) (uniform prior).

    Returns
    -------
    LogLikelihoodFn
        Configured log-likelihood function

    Raises
    ------
    ValueError
        If prior is not a tuple/list of two positive numbers
    """
    if (
        not isinstance(prior, (tuple, list))
        or len(prior) != 2
        or not all(isinstance(x, (int, float)) and x > 0 for x in prior)
    ):
        raise ValueError(
            "Prior must be a tuple of two positive numbers (alpha, beta). "
            f"Got: {prior}"
        )

    alpha, beta = prior
    return beta_binomial_loglik_per_rep(x_col, n_col, alpha, beta)
