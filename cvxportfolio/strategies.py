"""This module contains common desired strategies."""

import copy
import logging
import warnings

import cvxpy as cp
import numpy as np
import pandas as pd
from functools import reduce
from operator import add

from typing import List, Optional
from abc import ABC, abstractmethod

from .errors import (
    ConvexityError,
    ConvexSpecificationError,
    DataError,
    MissingTimesError,
    PortfolioOptimizationError,
    StrategyError,
)
from .estimator import DataEstimator, Estimator
from .returns import CashReturn
from .utils import flatten_heterogeneous_list
from .costs import StocksTransactionCost
from .returns import ReturnsForecast
from .constraints import LongOnly, LeverageLimit, FullSigmaLimit, ReturnsLimit
from .risks import FullCovariance
from .forecast import HistoricalFactorizedCovariance, HistoricalMeanReturn
from .simulator import MarketSimulator
from .data import BASE_LOCATION
from .result import StrategyResult

from . import policies

logger = logging.getLogger(__name__)

__all__ = [
    "MeanVarianceStrategy",
]


class Strategy(ABC):
    """
    A strategy is a set of rules for asset allocation over time.

    Parameters
    ----------
    assets : list of Asset
        The assets involved in the strategy.
    policy : Policy
        The policy for executing the strategy.
    cash_key : str, optional
        The name of the cash column in the strategy's data. Default is 'cash'.
    """

    def __init__(self, universe: Optional[List[str]] = None, cash_key: Optional[str] = "cash"):
        self._universe = universe if universe is not None else []
        self._cash_key = cash_key

    def __repr__(self):
        return f"Strategy(universe={self.universe}, cash_key={self.cash_key})"

    def __str__(self):
        return "Strategy with %d assets and policy %s" % (len(self.assets), self.policy)

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
    def cash_key(self) -> str:
        return self._cash_key

    @cash_key.setter
    def cash_key(self, value: str):
        if not isinstance(value, str):
            raise TypeError("cash_key must be a string")
        self._cash_key = value

    @property
    def universe(self) -> List[str]:
        return self._universe

    @universe.setter
    def universe(self, value: List[str]):
        if not isinstance(value, list):
            raise TypeError("universe must be a list of strings")
        self._universe = value


class MeanVarianceStrategy(Strategy):
    """
    A strategy class that implements mean-variance optimization.

    Attributes
    ----------
    universe : list
        List of assets involved in the strategy.
    r_hat : Estimator or DataFrame
        Expected returns estimator or DataFrame.
    decay : float
        Decay factor for returns forecasting.
    Sigma : Estimator or DataFrame
        Covariance matrix estimator or DataFrame.
    kelly : bool
        Whether to use the Kelly criterion.
    cash_key : str
        Name of the cash column in the strategy
    """

    def __init__(
        self,
        universe=None,
        r_hat=HistoricalMeanReturn,
        decay=1.0,
        Sigma=HistoricalFactorizedCovariance,
        kelly=False,
        cash_key="USDOLLAR",
    ):
        """
        Initialize a MeanVarianceStrategy instance.

        Parameters
        ----------
        universe : list, optional
            List of assets involved in the strategy.
        r_hat : Estimator or DataFrame, optional
            Expected returns estimator or DataFrame.
        decay : float, optional
            Decay factor for returns forecasting.
        Sigma : Estimator or DataFrame, optional
            Covariance matrix estimator or DataFrame.
        kelly : bool, optional
            Whether to use the Kelly criterion.
        cash_key : str, optional
            Name of the cash column in the strategy data.
        """
        super().__init__(universe, cash_key=cash_key)
        self.returns_forecast = ReturnsForecast(r_hat=r_hat, decay=decay)
        self.covariance_forecast = FullCovariance(Sigma=Sigma(kelly=kelly))
        self.gamma_risk = None
        self.gamma_trade = None
        self._objective_components = []
        self._objective = None
        self.planning_horizon = None
        self.constraints = [LongOnly(), LeverageLimit(1)]
        self.policy = None
        self.simulator = None
        self.results = None

    @property
    def objective(self):
        """The objective function of the portfolio."""
        if self._objective is None:
            if len(self._objective_components) > 1:
                # need to use reduce instead of sum to avoid issues with
                # the fact that python starts cum sum with 0 and it is not
                # a COST instance
                self._objective = reduce(add, self._objective_components)
            elif len(self._objective_components) == 1:
                self._objective = self._objective_components[0]
        return self._objective

    @objective.setter
    def objective(self, component):
        """Add a new component to the objective function."""
        self._objective_components.append(component)
        self._objective = None  # reset cached objective as new item has been added

    @property
    def simulator(self):
        """The market simulator for the strategy."""
        if self._simulator is None:
            raise StrategyError("Simulator not set. A universe must be provided to initialize the simulator.")
        return self._simulator

    @simulator.setter
    def simulator(self, value):
        self._simulator = value

    @property
    def results(self):
        """The results of the strategy backtest."""
        return self._results

    @results.setter
    def results(self, value):
        if value is None:
            self._results = None
        else:
            self._results = StrategyResult(value, self.simulator)

    def add_constraint(self, constraint):
        # TODO: add type check for constraint
        """Add a constraint to the portfolio."""
        self.constraints.append(constraint())

    def set_planning_horizon(self, horizon):
        """Set the planning horizon for the portfolio."""
        self.planning_horizon = horizon
        if horizon <= 1:
            self.policy = policies.SinglePeriodOptimization
        else:
            self.policy = policies.MultiPeriodOptimization

    def set_risk_target(self, risk_target):
        """Set the risk target for the portfolio."""
        self.risk_target = risk_target
        self.objective = self.returns_forecast
        self.constraints += [FullSigmaLimit(risk_target, self.covariance_forecast.Sigma)]

    def set_return_target(self, return_target):
        """Set the return target for the portfolio."""
        self.return_target = return_target
        self.objective = -self.covariance_forecast
        self.constraints += [ReturnsLimit(return_target, self.returns_forecast.r_hat)]

    def add_transaction_penalty(self, gamma_trade, **kwargs):
        """Add a transaction penalty to the portfolio."""
        self.gamma_trade = gamma_trade
        self.objective = -gamma_trade * StocksTransactionCost(**kwargs)

    def set_simulator_from_data(
        self,
        returns,
        volumes=None,
        prices=None,
        copy_dataframes=True,
        trading_frequency=None,
        min_history=pd.Timedelta("365.24d"),
        base_location=BASE_LOCATION,
        grace_period=pd.Timedelta("1d"),
        cash_key="USDOLLAR",
    ):
        self.simulator = MarketSimulator(
            returns,
            volumes=volumes,
            prices=prices,
            copy_dataframes=copy_dataframes,
            trading_frequency=trading_frequency,
            min_history=min_history,
            base_location=base_location,
            grace_period=grace_period,
            cash_key=cash_key,
        )

    def set_simulator_from_universe(self, universe, **kwargs):
        self.simulator = MarketSimulator(universe, **kwargs)

    def backtest(self, start_time=None, end_time=None, **kwargs):
        self.check_reqs()
        _policy = self.policy(
            objective=self.objective, constraints=self.constraints, planning_horizon=self.planning_horizon
        )
        self.results = self.simulator.backtest(_policy, start_time=start_time, end_time=end_time, **kwargs)

    def check_reqs(self):
        # TODO: perform correct checks as property
        """Check that all required parameters are set."""
        # if self.returns_forecast is None:
        #     raise PortfolioOptimizationError("Returns forecast not set")
        # if self.covariance_forecast is None:
        #     raise PortfolioOptimizationError("Covariance forecast not set")
        if self.policy is None:
            raise StrategyError("Policy not set")
        if self.planning_horizon is None:
            raise StrategyError("Planning horizon not set")
        if self.objective is None:
            raise StrategyError("Objective not set")
        if len(self.constraints) == 0:
            raise StrategyError("Constraints not set")
        # if self.constraints is None:
        #     raise PortfolioOptimizationError("Constraints not set")
        # if self.simulator is None:
        #     raise PortfolioOptimizationError("Simulator not set")

    def initialize_default_portfolio(self):
        """Initialize the default policy."""
        self.gamma_risk = 0.5
        self.objective = ReturnsForecast() - self.gamma_risk * FullCovariance(
            HistoricalFactorizedCovariance(kelly=False)
        )
        self.constraints = [LongOnly(), LeverageLimit(1)]
        self.policy = policies.SinglePeriodOptimization(self.objective, self.constraints)
        self.simulator = MarketSimulator(self.universe)
