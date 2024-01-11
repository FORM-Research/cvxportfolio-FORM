"""This module contains common desired portfolios."""

import copy
import logging
import warnings

import cvxpy as cp
import numpy as np
import pandas as pd

from abc import ABC, abstractmethod

from .errors import ConvexityError, ConvexSpecificationError, DataError, MissingTimesError, PortfolioOptimizationError
from .estimator import DataEstimator, Estimator
from .returns import CashReturn
from .utils import flatten_heterogeneous_list
from .costs import StocksTransactionCost
from .returns import ReturnsForecast
from .constraints import LongOnly, LeverageLimit
from .risks import FullCovariance

import policies

logger = logging.getLogger(__name__)


class Portfolio(ABC):
    """A portfolio is a collection of assets and a policy for trading them.

    Parameters
    ----------
    assets : list of Asset
        The assets in the portfolio.
    policy : Policy
        The policy for trading the assets.
    cash_key : str, optional
        The name of the cash column in the portfolio's data. Default is 'cash'.
    """

    def __init__(self, assets, cash_key="cash"):
        self.assets = assets
        self.cash_key = cash_key

    def __repr__(self):
        return f"Portfolio({self.assets}, {self.policy})"

    def __str__(self):
        return "Portfolio with %d assets and policy %s" % (len(self.assets), self.policy)

    def __eq__(self, other):
        return self.assets == other.assets and self.policy == other.policy and self.cash_key == other.cash_key

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((tuple(self.assets), self.policy, self.cash_key))

    def __len__(self):
        return len(self.assets)

    def __getitem__(self, key):
        return self.assets[key]

    def __iter__(self):
        return iter(self.assets)

    def __contains__(self, item):
        return item in self.assets

    def _repr_html_(self):
        """Display as HTML in IPython notebook."""
        return f"<h4>{str(self)}</h4>"

    def _repr_latex_(self):
        """Display as LaTeX in IPython notebook."""
        return f"${str(self)}$"

    def _repr_markdown_(self):
        """Display as markdown in IPython notebook."""
        return str(self)

    def _repr_pretty_(self, p, cycle):
        """Display as pretty text in IPython notebook."""
        p.text(str(self))

    @property
    def cash_key(self):
        """The name of the cash column in the portfolio's data."""
        return self._cash_key

    @cash_key.setter
    def cash_key(self, value):
        if not isinstance(value, str):
            raise TypeError("cash_key must be a string")
        self._cash_key = value

    @property
    @abstractmethod
    def policy(self):
        """The policy for trading the assets."""
        pass


class MVO(Portfolio):
    def __init__(
        self,
        assets,
        cash_key,
        multi_period=False,
        use_tx_costs=False,
        gamma_trade=0.0,
        include_cash_return=True,
        benchmark=policies.AllCash,
        **kwargs,
    ):
        super().__init__(assets, cash_key)

        objective = cvx.ReturnsForecast()

        if multi_period:
            # Initialize the policy using MultiPeriodOptimization
            self._policy = policies.MultiPeriodOptimization(
                objective,
                constraints=constraints,
                include_cash_return=include_cash_return,
                benchmark=benchmark,
                **kwargs,
            )
        else:
            # Initialize the policy using SinglePeriodOptimization
            self._policy = policies.SinglePeriodOptimization(
                objective,
                constraints=constraints,
                include_cash_return=include_cash_return,
                benchmark=benchmark,
                **kwargs,
            )

        cvx.MultiPeriodOptimization(
            cvx.ReturnsForecast()
            - gamma_risk * cvx.FactorModelCovariance(num_factors=10)
            - gamma_trade * cvx.StocksTransactionCost(),
            [cvx.LongOnly(), cvx.LeverageLimit(1)],
            planning_horizon=6,
            solver="ECOS",
        )

    @property
    def policy(self):
        return self._policy

    @policy.setter
    def policy(self, multi_period=False, **kwargs):
        if multi_period:
            self._policy = policies.MultiPeriodOptimization(**kwargs)
        else:
            self._policy = policies.SinglePeriodOptimization(**kwargs)

    @property
    def objective(self):
        return self._objective

    @objective.setter
    def objective(self, use_tx_costs=False, gamma_trade=0.0, use_factor_model=False, gamma_risk=0.0):
        if use_tx_costs:
            self._objective = ReturnsForecast() - gamma_trade * StocksTransactionCost()
        elif use_factor_model:
            raise NotImplementedError  # TODO: Implement factor model
            # self._objective = ReturnsForecast() - gamma_risk * FactorModelCovariance(num_factors=10)
        else:
            self._objective = ReturnsForecast()

    @property
    def constraints(self):
        return self._constraints

    @constraints.setter
    def constraints(self, no_shorting=True, leverage_limit=1.0, **kwargs):
        constraints = []
        constraints = FullCovariance()
        if no_shorting:
            constraints.append(LongOnly())
        else:
            constraints.append(LeverageLimit(leverage_limit))
        self._constraints = constraints
