"""A portfolio is a collection of strategies applied on assets and a cash account.
"""
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


class Portfolio(ABC):
    """
    A portfolio is a set of rules for asset allocation over time.

    Parameters
    ----------
    """

    def __init__(self):
        self.strategies = []

    # TODO: impose cash_key and universe on all strategies
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

    def add_strategies(self, strategies: List):
        self.strategies += strategies

    def backtest(self, start_time=None, end_time=None, **kwargs):
        """Return the results of the portfolio simulation.
        returns results for all strategies.

        Returns
        -------
        """
        self.results = []
        # TODO: allow for parallel backtesting
        for strategy in self.strategies:
            strategy.backtest(start_time=start_time, end_time=end_time, **kwargs)
            self.results.append(strategy.results)

        self.performance = pd.concat([res.full_v for res in self.results], axis=1)
        self.performance.columns = [
            strat.name if strat.name is not None else i for i, strat in enumerate(self.strategies)
        ]

    def get_next_weights(self, t: int) -> np.ndarray:
        """Return the weights for the next period.

        Parameters
        ----------
        t : int
            The current period.

        Returns
        -------
        np.ndarray
            The weights for the next period.
        """
        for strategy in self.strategies:
            current_weights = strategy.full_w.iloc[-1]
            strategy.policy.values_in_time_recursive(
                t=t,
                current_weights=current_weights,
                current_portfolio_value=current_portfolio_value,
                past_returns=past_returns,
                past_volumes=past_volumes,
                current_prices=current_prices,
            )
