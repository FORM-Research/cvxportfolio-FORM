# Copyright 2023 Enzo Busseti
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module include classes that download, store, and serve market data.

The two main abstractions are :class:`SymbolData` and :class:`MarketData`.
Neither are exposed outside this module. Their derived classes instead are.

If you want to implement the interface to another data source you should
derive from either of those two classes.

This module will be removed or heavily reformatted, most of it is
unused. The only parts that will remain are the FRED and YFinance
interfaces, simplified, and not meant to be accessed directly by users.
"""

import datetime
import logging
import sqlite3
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .utils import (periods_per_year_from_datetime_index, repr_numpy_pandas,
                    resample_returns)

# import yfinance as yf


# from .estimator import DataEstimator

__all__ = ["YahooFinanceSymbolData", "FredSymbolData"]

BASE_LOCATION = Path.home() / "cvxportfolio_data"

class SymbolData:
    """Base class for a single symbol time series data.
    
    The data is either in the form of a Pandas Series or DataFrame
    and has datetime index.
    
    This class needs to be derived. At a minimum,
    one should redefine the ``_download`` method, which
    implements the downloading of the symbol's time series 
    from an external source. The method takes the current (already
    downloaded and stored) data and is supposed to **only append** to it.
    In this way we only store new data and don't modify already downloaded
    data.
    
    Additionally one can redefine the ``_preload`` method, which prepares
    data to serve to the user (so the data is stored in a different format
    than what the user sees.) We found that this separation can be useful.
    
    This class interacts with module-level functions named ``_loader_BACKEND``
    and ``_storer_BACKEND``, where ``BACKEND`` is the name of the storage
    system used. We define ``pickle``, ``csv``, and ``sqlite`` backends. 
    These may have limitations. See their docstrings for more information.
    
    
    :param symbol: the symbol that we download
    :type symbol: str
    :param storage_backend: the storage backend
    :type storage_backend: str
    :param base_storage_location: the location of the storage. We store in a 
        subdirectory named after the class which derives from this.
    :type base_storage_location: pathlib.Path
    """
    # """Base class for Cvxportfolio database interface.
    #
    # Provides a back-end independent way to load and store
    # pandas Series and DataFrames where the first index is a
    # pandas Timestamp (or numpy datetime64). It also provides
    # a systematic way to access external data sources via
    # the network. We specialize it to storing data locally
    # and downloading public time series of financial data.
    # By emulating some of these classes you can interface
    # cvxportfolio with other databases or other data sources.
    #
    # This interface is also used by cvxportfolio to store the
    # data it generates, such as Backtest classes data, or
    # estimators such as factor risk models.
    #
    # Cvxportfolio uses in-memory data whenever possible, and
    # in particular never uses BaseData methods during a backtest.
    # This ensures thread safety and allows us to use simple
    # local databases such as sqlite (or even flat csv files!).
    # Cvxportfolio loads data from the database before (possibly parallel)
    # backtesting, and stores it after backtesting. So, only one process
    # at a time accesses this class' methods. If you write custom callbacks
    # that are invoked during backtests, such as callables inside
    # DataEstimator, you should most probably not use cvxportfolio.data methods
    # inside them.
    #
    # LIMITATIONS:
    #     - columns names should be strings in order to work with all current
    #       data storage backends. If you create a DataFrame from a numpy array
    #       without specifying column names they will default to integers (0, 1, ...).
    #       If you store and load it back you will may get string column names ('0', '1', ...),
    #       depending on the backend.
    #     - the first level of the index should be a pandas Timestamp or equivalently
    #       numpy datetime64
    #     - you can only store sql-friendly data: integers, floats (including `np.nan`),
    #       datetime (including `np.datetime64('NaT')`), and simple alphanumeric strings
    #       (i.e., without commas or quote marks).
    #       If you need to store more complex python objects, such as the string
    #         "{'parameter1':3.0, 'parameter2': pd.Timestamp('2022-01-01')}",
    #       you will have to check that it works with the backend you use
    #      (it probably would not not with csv).
    # """

    def __init__(self, symbol,
                 storage_backend='pickle',
                 base_storage_location=BASE_LOCATION):
        self._symbol = symbol
        self._storage_backend = storage_backend
        self._base_storage_location = base_storage_location
        self._update()
        self._data = self._load()

    @property
    def storage_location(self):
        """Storage location. Directory is created if not existent.
        
        :rtype: pathlib.Path
        """
        loc = self._base_storage_location / f"{self.__class__.__name__}"
        loc.mkdir(parents=True, exist_ok=True)
        return loc

    @property
    def symbol(self):
        """The symbol whose data this instance contains.
        
        :rtype: str
        """
        return self._symbol

    @property
    def data(self):
        """Time series data, updated to the most recent observation.
        
        :rtype: pandas.Series or pandas.DataFrame
        """
        return self._data

    def _load_raw(self):
        """Load raw data from database."""
        # we could implement multiprocess safety here
        loader = globals()['_loader_' + self._storage_backend]
        try:
            return loader(self.symbol, self.storage_location)
        except FileNotFoundError:
            return None

    def _load(self):
        """Load data from database using `self.preload` function to process.
        """
        return self._preload(self._load_raw())

    def _store(self, data):
        """Store data in database."""
        # we could implement multiprocess safety here
        storer = globals()['_storer_' + self._storage_backend]
        storer(self.symbol, data, self.storage_location)

    def _update(self):
        """Update current stored data for symbol."""
        current = self._load_raw()
        updated = self._download(self.symbol, current)
        self._store(updated)

    def _download(self, symbol, current):
        """Download data from external source given already downloaded data.
        
        This method must be redefined by derived classes.
        
        :param symbol: The symbol we download.
        :type symbol: str
        :param current: The data already downloaded. We are supposed to
             **only append** to it. If None, no data is present.
        :type current: pandas.Series or pandas.DataFrame or None
        
        :rtype: pandas.Series or pandas.DataFrame
        
        """
        raise NotImplementedError

    def _preload(self, data):
        """Prepare data to serve to the user.
        
        This method can be redefined by derived classes.
        
        :param data: The data returned by the storage backend.
        :type data: pandas.Series or pandas.DataFrame
        
        :rtype: pandas.Series or pandas.DataFrame
        """
        return data


#
# Yahoo Finance.
#

def _timestamp_convert(unix_seconds_ts):
    """Convert a UNIX timestamp in seconds to pd.Timestamp."""
    return pd.Timestamp(unix_seconds_ts*1E9, tz='UTC')

def _now_timezoned():
    return pd.Timestamp(
        datetime.datetime.now(datetime.timezone.utc).astimezone())

class YahooFinanceSymbolData(SymbolData):
    """Yahoo Finance symbol data."""

    @staticmethod
    def _internal_process(data):
        """Manipulate yfinance data for better storing."""

        # nan-out nonpositive prices
        data.loc[data["Open"] <= 0, 'Open'] = np.nan
        data.loc[data["Close"] <= 0, "Close"] = np.nan
        data.loc[data["High"] <= 0, "High"] = np.nan
        data.loc[data["Low"] <= 0, "Low"] = np.nan
        data.loc[data["Adj Close"] <= 0, "Adj Close"] = np.nan

        # nan-out negative volumes
        data.loc[data["Volume"] < 0, 'Volume'] = np.nan

        intraday_logreturn = np.log(data["Close"]) - np.log(data["Open"])
        close_to_close_logreturn = np.log(data["Adj Close"]).diff().shift(-1)
        open_to_open_logreturn = (
            close_to_close_logreturn + intraday_logreturn -
            intraday_logreturn.shift(-1)
        )
        data["Return"] = np.exp(open_to_open_logreturn) - 1
        del data["Adj Close"]
        # eliminate last period's intraday data
        data.loc[data.index[-1], ["High", "Low",
                                  "Close", "Return", "Volume"]] = np.nan
        return data

    @staticmethod
    def _get_data_yahoo(ticker, start='1900-01-01', end='2100-01-01'):
        """Get 1 day OHLC from Yahoo finance. 
    
        Result is timestamped with the open time (time-zoned) of
        the instrument.
        """

        BASE_URL = 'https://query2.finance.yahoo.com'

        HEADERS = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1)'
            ' AppleWebKit/537.36 (KHTML, like Gecko)'
            ' Chrome/39.0.2171.95 Safari/537.36'}

        # print(HEADERS)
        start = int(pd.Timestamp(start).timestamp())
        end = int(pd.Timestamp(end).timestamp())

        res = requests.get(
            url=f"{BASE_URL}/v8/finance/chart/{ticker}",
            params={'interval': '1d',
                "period1": start,
                "period2": end},
            headers=HEADERS)

        # print(res)

        if res.status_code == 404:
            raise DataError(
                f'Data for symbol {ticker} is not available.'
                + 'Json output:', str(res.json()))

        if res.status_code != 200:
            raise DataError(f'Yahoo finance data download failed. Json:',
                str(res.json()))

        data = res.json()['chart']['result'][0]

        index = pd.DatetimeIndex([
            _timestamp_convert(el)
            for el in data['timestamp']])

        df_result = pd.DataFrame(data['indicators']['quote'][0], index=index)
        df_result['adjclose'] = data['indicators']['adjclose'][0]['adjclose']

        # last timestamp is probably broken (not timed to market open)
        # we set its time to same as the day before, but this is wrong
        # on days of DST switch. It's fine though because that line will be
        # overwritten next update
        if df_result.index[-1].time() != df_result.index[-2].time():
            tm1 = df_result.index[-2].time()
            newlast = df_result.index[-1].replace(
                hour=tm1.hour, minute=tm1.minute, second=tm1.second)
            df_result.index = pd.DatetimeIndex(
                list(df_result.index[:-1]) + [newlast])

        # remove later, for now we match yfinance column names and ordering
        df_result = df_result[['open', 'high', 'low',
                              'close', 'adjclose', 'volume']]
        df_result.columns = ['Open', 'High', 'Low',
                             'Close', 'Adj Close', 'Volume']
        return df_result

    @classmethod
    def _download(cls, symbol, current=None,
                overlap=5, grace_period='5d', **kwargs):
        """Download single stock from Yahoo Finance.

        If data was already downloaded we only download
        the most recent missing portion.

        Args:

            symbol (str): yahoo name of the instrument
            current (pandas.DataFrame or None): current data present locally
            overlap (int): how many lines of current data will be overwritten
                by newly downloaded data
            kwargs (dict): extra arguments passed to yfinance.download

        Returns:
            updated (pandas.DataFrame): updated DataFrame for the symbol
        """
        if overlap < 2:
            raise Exception(
                'There could be issues with DST and Yahoo finance data.')
        if (current is None) or (len(current) < overlap):
            updated = cls._get_data_yahoo(symbol, **kwargs)
            # updated = yf.download(symbol, **kwargs)
            return cls._internal_process(updated)
        else:
            if (_now_timezoned() - current.index[-1]
            # if (pd.Timestamp.now() - current.index[-1]
                ) < pd.Timedelta(grace_period):
                return current
            new = cls._get_data_yahoo(symbol, start=current.index[-overlap])
            # new = yf.download(symbol,  start=current.index[-overlap])
            new = cls._internal_process(new)
            return pd.concat([current.iloc[:-overlap], new])

    @staticmethod
    def _preload(data):
        """Prepare data for use by Cvxportfolio.

        We drop the 'Volume' column expressed in number of stocks and
        replace it with 'ValueVolume' which is an estimate of the (e.g.,
        US dollar) value of the volume exchanged on the day.
        """
        data["ValueVolume"] = data["Volume"] * data["Open"]
        del data["Volume"]
        # remove infty values
        data.iloc[:, :] = np.nan_to_num(
            data.values, copy=True, nan=np.nan, posinf=np.nan, neginf=np.nan)
        # remove extreme values
        # data.loc[data["Return"] < -.99, "Return"] = np.nan
        # data.loc[data["Return"] > .99, "Return"] = np.nan
        return data

#
# FRED.
#

class FredSymbolData(SymbolData):
    """Base class for FRED data access."""

    URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"

    # TODO: implement FRED point-in-time
    # example:
    # https://alfred.stlouisfed.org/graph/alfredgraph.csv?id=CES0500000003&vintage_date=2023-07-06
    # hourly wages time series **as it appeared** on 2023-07-06
    # store using pd.Series() of diff'ed values only.

    def _internal_download(self, symbol):
        return pd.read_csv(
            self.URL + f'?id={symbol}',
            index_col=0, parse_dates=[0])[symbol]

    def _download(self, symbol="DFF", current=None, grace_period='5d'):
        """Download or update pandas Series from FRED.

        If already downloaded don't change data stored locally and only
        add new entries at the end.

        Additionally, we allow for a `grace period`, if the data already
        downloaded has a last entry not older than the grace period, we
        don't download new data.
        """
        if current is None:
            return self._internal_download(symbol)
        else:
            if (pd.Timestamp.today() - current.index[-1]
                ) < pd.Timedelta(grace_period):
                return current

            new = self._internal_download(symbol)
            new = new.loc[new.index > current.index[-1]]

            if new.empty:
                return current

            assert new.index[0] > current.index[-1]
            return pd.concat([current, new])

    def _preload(self, data):
        """Add UTC timezone."""
        data.index = data.index.tz_localize('UTC')
        return data

#
# Sqlite storage backend.
#

def _open_sqlite(storage_location):
    return sqlite3.connect(storage_location/"db.sqlite")

def _close_sqlite(connection):
    connection.close()

def _loader_sqlite(symbol, storage_location):
    """Load data in sqlite format.
    
    Limitations: if your pandas object's index has a name it will be lost,
        the index is renamed 'index'. If you pass timestamp data (including
        the index) it must have explicit timezone.
    """
    try:
        connection = _open_sqlite(storage_location)
        dtypes = pd.read_sql_query(
            f"SELECT * FROM {symbol}___dtypes",
            connection, index_col="index",
            dtype={"index": "str", "0": "str"})

        parse_dates = 'index'
        my_dtypes = dict(dtypes["0"])

        tmp = pd.read_sql_query(
            f"SELECT * FROM {symbol}", connection,
            index_col="index", parse_dates=parse_dates, dtype=my_dtypes)

        _close_sqlite(connection)
        multiindex = []
        for col in tmp.columns:
            if col[:8] == "___level":
                multiindex.append(col)
            else:
                break
        if len(multiindex):
            multiindex = [tmp.index.name] + multiindex
            tmp = tmp.reset_index().set_index(multiindex)
        return tmp.iloc[:, 0] if tmp.shape[1] == 1 else tmp
    except pd.errors.DatabaseError:
        return None

def _storer_sqlite(symbol, data, storage_location):
    """Store data in sqlite format.

    We separately store dtypes for data consistency and safety.

    Limitations: if your pandas object's index has a name it will be lost,
        the index is renamed 'index'. If you pass timestamp data (including
        the index) it must have explicit timezone.

    """
    connection = _open_sqlite(storage_location)
    exists = pd.read_sql_query(
        f"SELECT name FROM sqlite_master WHERE type='table' AND name='{symbol}'",
        connection,
    )
    if len(exists):
        res = connection.cursor().execute(f"DROP TABLE '{symbol}'")
        res = connection.cursor().execute(
            f"DROP TABLE '{symbol}___dtypes'")
        connection.commit()

    if hasattr(data.index, "levels"):
        data.index = data.index.set_names(
            ["index"] +
            [f"___level{i}" for i in range(1, len(data.index.levels))]
        )
        data = data.reset_index().set_index("index")
    else:
        data.index.name = "index"

    if data.index[0].tzinfo is None:
        warnings.warn('Index has not timezone, setting to UTC')
        data.index = data.index.tz_localize('UTC')

    data.to_sql(f"{symbol}", connection)
    pd.DataFrame(data).dtypes.astype("string").to_sql(
        f"{symbol}___dtypes", connection)
    _close_sqlite(connection)


#
# Pickle storage backend.
#

def _loader_pickle(symbol, storage_location):
    """Load data in pickle format."""
    return pd.read_pickle(storage_location / f"{symbol}.pickle")

def _storer_pickle(symbol, data, storage_location):
    """Store data in pickle format."""
    data.to_pickle(storage_location / f"{symbol}.pickle")

#
# Csv storage backend.
#

def _loader_csv(symbol, storage_location):
    """Load data in csv format."""

    index_dtypes = pd.read_csv(
        storage_location / f"{symbol}___index_dtypes.csv",
        index_col=0)["0"]

    dtypes = pd.read_csv(
        storage_location / f"{symbol}___dtypes.csv", index_col=0,
        dtype={"index": "str", "0": "str"})
    dtypes = dict(dtypes["0"])
    new_dtypes = {}
    parse_dates = []
    for i, level in enumerate(index_dtypes):
        if "datetime64[ns" in level: # includes all timezones
            parse_dates.append(i)
    for i, el in enumerate(dtypes):
        if "datetime64[ns" in dtypes[el]:  # includes all timezones
            parse_dates += [i + len(index_dtypes)]
        else:
            new_dtypes[el] = dtypes[el]

    tmp = pd.read_csv(storage_location / f"{symbol}.csv",
        index_col=list(range(len(index_dtypes))),
        parse_dates=parse_dates, dtype=new_dtypes)

    return tmp.iloc[:, 0] if tmp.shape[1] == 1 else tmp


def _storer_csv(symbol, data, storage_location):
    """Store data in csv format."""
    pd.DataFrame(data.index.dtypes if hasattr(data.index, 'levels')
        else [data.index.dtype]).astype("string").to_csv(
        storage_location / f"{symbol}___index_dtypes.csv")
    pd.DataFrame(data).dtypes.astype("string").to_csv(
        storage_location / f"{symbol}___dtypes.csv")
    data.to_csv(storage_location / f"{symbol}.csv")

#
# Market Data
#

class BaseMarketData:
    """Prepare, hold, and serve market data."""

    def serve_data_policy(self, t):
        """Give data to policy at time t.
        
        :param t: Trading time.
        :type t: pandas.Timestamp
        
        :rtype: tuple of pandas.DataFrames and pandas.Series
        """
        raise NotImplementedError

    def serve_data_simulator(self, t):
        """Give data to simulator at time t.
        
        :param t: Trading time.
        :type t: pandas.Timestamp
        
        :rtype: tuple of pandas.DataFrames and pandas.Series
        """
        raise NotImplementedError

    def trading_calendar(self, start_time, end_time):
        """The trading calendar between two times (inclusive).
        
        :rtype: pandas.DatetimeIndex
        """
        raise NotImplementedError

    @property
    def periods_per_year(self):
        """Average trading periods per year.
        
        :rtype: int
        """
        raise NotImplementedError

class InMemoryMarketData(BaseMarketData):
    """Market data that is stored in memory when initialized."""

    def _serve_data_policy(self, t, mask=None):
        """Give data to policy at time t."""
        tidx = self.returns.index.get_loc(t)
        past_returns = pd.DataFrame(self.returns.iloc[:tidx])
        if not (mask is None):
            past_returns = past_returns[mask]
        if not self.volumes is None:
            tidx = self.volumes.index.get_loc(t)
            past_volumes = pd.DataFrame(self.volumes.iloc[:tidx])
            if not (mask is None):
                past_volumes = past_volumes[mask[:-1]]
        else:
            past_volumes = None
        if not self.prices is None:
            current_prices = pd.Series(self.prices.loc[t])
            if not (mask is None):
                current_prices = current_prices[mask[:-1]]
        else:
            current_prices = None

        return past_returns, past_volumes, current_prices

    def _serve_data_simulator(self, t, mask=None):
        """Give data to simulator at time t."""
        tidx = self.returns.index.get_loc(t)
        current_and_past_returns = pd.DataFrame(self.returns.iloc[:tidx+1])
        if not (mask is None):
            current_and_past_returns = current_and_past_returns[mask]
        if not self.volumes is None:
            tidx = self.volumes.index.get_loc(t)
            current_and_past_volumes = pd.DataFrame(self.volumes.iloc[:tidx+1])
            if not (mask is None):
                current_and_past_volumes = current_and_past_volumes[mask[:-1]]
        else:
            current_and_past_volumes = None
        if not (self.prices is None):
            current_prices = pd.Series(self.prices.loc[t])
            if not (mask is None):
                current_prices = current_prices[mask[:-1]]
        else:
            current_prices = None

        return current_and_past_returns, current_and_past_volumes, current_prices
        
    @property
    def universe(self):
        """Full trading universe including cash."""
        return self.returns.columns

    @property
    def periods_per_year(self):
        """Average trading periods per year inferred from the data."""
        return periods_per_year_from_datetime_index(self.returns.index)

    @property
    def min_history(self):
        """Min history expressed in periods."""
        return int(np.round(self.periods_per_year * (
            self._min_history_timedelta / pd.Timedelta('365.24d'))))

    @staticmethod
    def _resample_returns(returns, periods):
        """Resample returns from number of periods to single period."""
        return np.exp(np.log(1 + returns) / periods) - 1

    def _add_cash_column(self, cash_key):
        """Add the cash column to an already formed returns dataframe.
        
        This assumes that the trading periods are about equally spaced.
        If, say, you have trading periods with very different lengths you
        should redefine this method **and** replace the :class:`CashReturn`
        objective term.
        """

        if not cash_key == 'USDOLLAR':
            raise NotImplementedError(
                'Currently the only data pipeline built is for USDOLLAR cash')

        data = FredSymbolData('DFF', base_storage_location=self.base_location)

        cash_returns_per_period = self._resample_returns(
            data.data/100, periods=self.periods_per_year)

        # we merge instead of assigning column because indexes might
        # be misaligned (e.g., with tz-aware timestamps)
        cash_returns_per_period.name = cash_key
        original_returns_index = self.returns.index
        tmp = self.returns.merge(cash_returns_per_period)

        raise Exception

        self.returns[cash_key] = self.returns[cash_key].ffill()

    def _make_internal_dataframes_read_only(self):
        """Makes the internal dataframes read-only."""

        self.returns = ro(self.returns)
        if not self.prices is None:
            self.prices = ro(self.prices)
        if not self.volumes is None:
            self.volumes = ro(self.volumes)

    def _get_backtest_times(self, start_time=None, end_time=None, include_end=True):
        """Get trading calendar from market data."""
        result = self.returns.index
        result = result[result >= self._earliest_backtest_start]
        if start_time:
            result = result[result >= start_time]
        if end_time:
            result = result[(result <= end_time)]
        if not include_end:
            result = result[:-1]
        return result

    @staticmethod
    def _set_read_only(df):
        """Set numpy array contained in dataframe to read only.
        
        This is done on data store internally before it is served to
        the policy or the simulator to ensure data consistency in case
        some element of the pipeline accidentally corrupts the data.
        
        This is enough to prevent direct assignement to the resulting
        dataframe. However it could still be accidentally corrupted by
        assigning to columns or indices that are not present in the
        original. We avoid that case as well by returning a wrapped
        dataframe (which doesn't copy data on creation) in
        serve_data_policy and serve_data_simulator.
        """
        data = df.values
        data.flags.writeable = False
        return pd.DataFrame(data, index=df.index, columns=df.columns)

    # @property
    # def _break_timestamps(self):
    #     """List of timestamps at which a backtest should be broken.
    #
    #     An asset enters into a backtest after having non-NaN returns for
    #     self.min_history periods and exits after having NaN returns for
    #     self.max_contiguous_missing. Defaults values are 252 and 10
    #     respectively.
    #     """
    #     self.entry_dates = defaultdict(list)
    #     self.exit_dates = defaultdict(list)
    #     for asset in self.returns.columns[:-1]:
    #         single_asset_returns = self.returns[asset].dropna()
    #         if len(single_asset_returns) > self.min_history:
    #             self.entry_dates[single_asset_returns.index[self.min_history]].append(
    #                 asset)
    #             exit_date = single_asset_returns.index[-1]
    #             if (self.returns.index[-1] - exit_date) >= pd.Timedelta(self.max_contiguous_missing):
    #                 self.exit_dates[exit_date].append(asset)
    #
    #     _ = sorted(set(self.exit_dates) | set(self.entry_dates))
    #     logging.debug(f'computing break timestamps {_}')
    #     return _

    @property
    def _limited_universes(self):
        """Valid universes for each section, minus cash.

        A backtest is broken into multiple ones that start at each key
        of this, have the universe specified by this, and end at the
        next startpoint.
        """
        result = OrderedDict()
        uni = []
        for ts in self._break_timestamps:
            uni += self.entry_dates[ts]
            uni = [el for el in uni if not el in self.exit_dates[ts]]
            result[ts] = tuple(sorted(uni))
        return result

    @property
    def _earliest_backtest_start(self):
        """Earliest date at which we can start a backtest."""
        return self.returns.iloc[:, :-1].dropna(how='all').index[self.min_history]

    def _get_limited_backtests(self, start_time, end_time):
        """Get start/end times and universes of constituent backtests.

        Each one has constant universe with assets' that meet the
        ``min_history`` requirement and has not disappeared from the
        dataset.
        """

        full_backtest_times = self._get_backtest_times(start_time, end_time)
        brkt = np.array(self._break_timestamps)

        def get_valid_universe_and_its_expiration_for(time):
            try:
                return self._limited_universes[brkt[brkt <= time][-1]], \
                    brkt[brkt > time][0] if len(
                        brkt[brkt > time]) else full_backtest_times[-1]
            except IndexError:
                raise DataError(
                    'There are no assets that meet the required min_history.')

        result = []
        start = full_backtest_times[0]
        while True:

            universe, expiration = get_valid_universe_and_its_expiration_for(
                start)
            if expiration > full_backtest_times[-1]:
                expiration = full_backtest_times[-1]
            result.append({
                'start_time': start,
                'end_time': expiration,
                'universe': list(universe) + [self.cash_key]})

            if expiration == full_backtest_times[-1]:
                return result

            start = expiration

    sampling_intervals = {'weekly': 'W-MON',
                          'monthly': 'MS', 'quarterly': 'QS', 'annual': 'AS'}

    # @staticmethod
    # def _is_first_interval_small(datetimeindex):
    #     """Check if post-resampling the first interval is small.
    #
    #     We have no way of knowing exactly if the first interval
    #     needs to be dropped. We drop it if its length is smaller
    #     than the average of all others, minus 2 standard deviation.
    #     """
    #     first_interval = (datetimeindex[1] - datetimeindex[0])
    #     all_others = (datetimeindex[2:] - datetimeindex[1:-1])
    #     return first_interval < (all_others.mean() - 2 * all_others.std())

    def _downsample(self, interval):
        """_downsample market data."""
        if not interval in self.sampling_intervals:
            raise SyntaxError(
                'Unsopported trading interval for down-sampling.')
        interval = self.sampling_intervals[interval]
        new_returns_index = pd.Series(self.returns.index, self.returns.index
                                      ).resample(interval, closed='left', 
                                                 label='left').first().values
        # print(new_returns_index)
        self.returns = np.exp(np.log(
            1+self.returns).resample(interval, closed='left', label='left'
                                     ).sum(min_count=1))-1
        self.returns.index = new_returns_index

        # last row is always unknown
        self.returns.iloc[-1] = np.nan

        # # we drop the first row if its interval is small
        # if self._is_first_interval_small(self.returns.index):
        #     self.returns = self.returns.iloc[1:]

        # we nan-out the first non-nan element of every col
        for col in self.returns.columns[:-1]:
            self.returns[col].loc[
                    (~(self.returns[col].isnull())).idxmax()
                ] = np.nan

        if self.volumes is not None:
            new_volumes_index = pd.Series(
                self.volumes.index, self.volumes.index
                    ).resample(interval, closed='left', 
                               label='left').first().values
            self.volumes = self.volumes.resample(
                interval, closed='left', label='left').sum(min_count=1)
            self.volumes.index = new_volumes_index

            # last row is always unknown
            self.volumes.iloc[-1] = np.nan

            # # we drop the first row if its interval is small
            # if self._is_first_interval_small(self.volumes.index):
            #     self.volumes = self.volumes.iloc[1:]

            # we nan-out the first non-nan element of every col
            for col in self.volumes.columns:
                self.volumes[col].loc[
                        (~(self.volumes[col].isnull())).idxmax()
                    ] = np.nan

        if self.prices is not None:
            new_prices_index = pd.Series(
                self.prices.index, self.prices.index
                ).resample(
                    interval, closed='left', label='left').first().values
            self.prices = self.prices.resample(
                interval, closed='left', label='left').first()
            self.prices.index = new_prices_index

            # # we drop the first row if its interval is small
            # if self._is_first_interval_small(self.prices.index):
            #     self.prices = self.prices.iloc[1:]

            # we nan-out the first non-nan element of every col
            for col in self.prices.columns:
                self.prices[col].loc[
                        (~(self.prices[col].isnull())).idxmax()
                    ] = np.nan

class UserProvidedMarketData(InMemoryMarketData):

    def __init__(self, returns=None, volumes=None, prices=None):
        pass

    def _check_sizes(self):

        if (not self.volumes is None) and (
                not (self.volumes.shape[1] == self.returns.shape[1] - 1)
                or not all(self.volumes.columns == self.returns.columns[:-1])):
            raise SyntaxError(
                'Volumes should have same columns as returns, minus cash_key.')

        if (not self.prices is None) and (
                not (self.prices.shape[1] == self.returns.shape[1] - 1)
                or not all(self.prices.columns == self.returns.columns[:-1])):
            raise SyntaxError(
                'Prices should have same columns as returns, minus cash_key.')

class MarketData(InMemoryMarketData):
    """Prepare, hold, and serve market data.

    Not meant to be accessed by user. Most of its initialization is
    documented in MarketSimulator.
    """

    def __init__(self,
                 universe=(),
                 returns=None,
                 volumes=None,
                 prices=None,
                 datasource='YFinance',
                 cash_key='USDOLLAR',
                 base_location=BASE_LOCATION,
                 min_history=pd.Timedelta('365.24d'),
                 # TODO change logic for this (it's now this to not drop quarterly data)
                 max_contiguous_missing='370d',
                 trading_frequency=None,
                 copy_dataframes=True,
                 **kwargs,
                 ):

        # drop duplicates and ensure ordering
        universe = sorted(set(universe))

        self.base_location = Path(base_location)
        self.min_history_timedelta = min_history
        self.max_contiguous_missing = max_contiguous_missing
        self.cash_key = cash_key

        if len(universe):
            self._get_market_data(universe, datasource)
            self._add_cash_column(self.cash_key)
            self._remove_missing_recent()
        else:
            if returns is None:
                raise SyntaxError(
                    "If you don't specify a universe you should pass `returns`.")
            self.returns = pd.DataFrame(returns, copy=copy_dataframes)
            self.volumes = volumes if volumes is None else\
                pd.DataFrame(volumes, copy=copy_dataframes)
            self.prices = prices if prices is None else\
                pd.DataFrame(prices, copy=copy_dataframes)
            if cash_key != returns.columns[-1]:
                self._add_cash_column(cash_key)

        if trading_frequency:
            self._downsample(trading_frequency)

        self._set_read_only()
        self._check_sizes()

    def _reduce_universe(self, reduced_universe):
        assert reduced_universe[-1] == self.cash_key
        logging.debug(
            f'Preparing MarketData with reduced_universe {reduced_universe}')
        return MarketData(
            returns=self.returns[reduced_universe],
            volumes=self.volumes[reduced_universe[:-1]
                                 ] if not (self.volumes is None) else None,
            prices=self.prices[reduced_universe[:-1]
                               ] if not (self.prices is None) else None,
            cash_key=self.cash_key,
            copy_dataframes=False)

    @property
    def min_history(self):
        """Min.

        history expressed in periods.
        """
        return int(np.round(self.PPY * (
            self.min_history_timedelta / pd.Timedelta('365.24d'))))

    @property
    def universe(self):
        return self.returns.columns



    @property
    def PPY(self):
        """Periods per year, assumes returns are about equally spaced."""
        return periods_per_year_from_datetime_index(self.returns.index)

    def _check_sizes(self):

        if (not self.volumes is None) and (not (self.volumes.shape[1] == self.returns.shape[1] - 1)
                                           or not all(self.volumes.columns == self.returns.columns[:-1])):
            raise SyntaxError(
                'Volumes should have same columns as returns, minus cash_key.')

        if (not self.prices is None) and (not (self.prices.shape[1] == self.returns.shape[1] - 1)
                                          or not all(self.prices.columns == self.returns.columns[:-1])):
            raise SyntaxError(
                'Prices should have same columns as returns, minus cash_key.')



    def _set_read_only(self):
        """Set numpy array contained in dataframe to read only.

        This is enough to prevent direct assignement to the resulting
        dataframe. However it could still be accidentally corrupted by
        assigning to columns or indices that are not present in the
        original. We avoid that case as well by returning a wrapped
        dataframe (which doesn't copy data on creation) in
        _serve_data_policy and _serve_data_simulator.
        """

        def ro(df):
            data = df.values
            data.flags.writeable = False
            return pd.DataFrame(data, index=df.index, columns=df.columns)

        self.returns = ro(self.returns)

        if not self.prices is None:
            self.prices = ro(self.prices)

        if not self.volumes is None:
            self.volumes = ro(self.volumes)

    def _add_cash_column(self, cash_key):
        """Add the cash column to an already formed returns dataframe.

        This assumes that the trading periods are about equally spaced.
        If, say, you have trading periods with very different lengths you
        should redefine this method **and** replace the :class:`CashReturn`
        objective term.
        """

        if not cash_key == 'USDOLLAR':
            raise NotImplementedError(
                'Currently the only data pipeline built is for USDOLLAR cash')

        data = FredSymbolData('DFF', base_storage_location=self.base_location)
        cash_returns_per_period = resample_returns(
            data.data/100, periods=self.PPY)

        # we merge instead of assigning column because indexes might
        # be misaligned (e.g., with tz-aware timestamps)
        cash_returns_per_period.name = self.cash_key
        original_returns_index = self.returns.index
        tmp = pd.concat([self.returns, cash_returns_per_period], axis=1)
        tmp[cash_key] = tmp[cash_key].ffill()
        self.returns = tmp.loc[original_returns_index]

    DATASOURCES = {'YFinance': YahooFinanceSymbolData, 'FRED': FredSymbolData}

    def _get_market_data(self, universe, datasource):
        database_accesses = {}
        print('Updating data')

        for stock in universe:
            logging.debug(
                f'Getting data for {stock} with {self.DATASOURCES[datasource]}.')
            print('.')
            database_accesses[stock] = self.DATASOURCES[datasource](
                stock, base_storage_location=self.base_location)

        if datasource == 'YFinance':
            self.returns = pd.DataFrame(
                {stock: database_accesses[stock].data['Return'] for stock in universe})
            self.volumes = pd.DataFrame(
                {stock: database_accesses[stock].data['ValueVolume'] for stock in universe})
            self.prices = pd.DataFrame(
                {stock: database_accesses[stock].data['Open'] for stock in universe})
        else:  # only FRED for indexes
            self.prices = pd.DataFrame(
                {stock: database_accesses[stock].data for stock in universe})  # open prices
            self.returns = 1 - self.prices / self.prices.shift(-1)
            self.volumes = None

    def _remove_missing_recent(self):
        """Clean recent data.

        Yfinance has some issues with most recent data; we remove recent
        days if there are NaNs.
        """

        if self.prices.iloc[-5:].isnull().any().any():
            logging.debug(
                'Removing some recent lines because there are missing values.')
            drop_at = self.prices.iloc[-5:].isnull().any(axis=1).idxmax()
            logging.debug(f'Dropping at index {drop_at}')
            self.returns = self.returns.loc[self.returns.index < drop_at]
            if self.prices is not None:
                self.prices = self.prices.loc[self.prices.index < drop_at]
            if self.volumes is not None:
                self.volumes = self.volumes.loc[self.volumes.index < drop_at]

        # for consistency we must also nan-out the last row of returns and volumes
        self.returns.iloc[-1] = np.nan
        if self.volumes is not None:
            self.volumes.iloc[-1] = np.nan

    def _universe_at_time(self, t):
        """Return the valid universe at time t."""
        past_returns = self.returns.loc[self.returns.index < t]
        return self.universe[(past_returns.count() >= self.min_history) &
            (~self.returns.loc[t].isnull())]

    def _get_backtest_times(self, start_time=None, end_time=None, include_end=True):
        """Get trading calendar from market data."""
        result = self.returns.index
        result = result[result >= self._earliest_backtest_start]
        if start_time:
            result = result[result >= start_time]
        if end_time:
            result = result[(result <= end_time)]
        if not include_end:
            result = result[:-1]
        return result


    @property
    def _earliest_backtest_start(self):
        """Earliest date at which we can start a backtest."""
        return self.returns.iloc[:, :-1].dropna(how='all').index[self.min_history]

