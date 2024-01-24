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

    def __init__(self, enable_custom_assets=False):
        self.strategies = []
        self.enable_custom_assets = enable_custom_assets

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
        # TODO: think about name strategies at the portfolio level
        for strategy in strategies:
            if self.enable_custom_assets:
                strategy.enable_custom_assets = True
            self.strategies.append(strategy)

    def add_custom_asset(self, asset):
        """Not implemented yet. Need to think about how to do this and if it is necessary."""
        for strategy in self.strategies:
            strategy.add_custom_asset(asset)

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
