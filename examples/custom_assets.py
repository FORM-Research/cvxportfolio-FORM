from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cvxportfolio as cvx
from cvxportfolio.strategies import MeanVarianceStrategy
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Define parameters
universe = ["AAPL", "AMZN", "UBER", "ZM", "CVX", "TSLA", "GM", "ABNB", "CTAS", "GOOG"]


# Here we define a class to forecast expected returns
class CustomMeanReturn:
    """Expected return as mean of recent window of past returns."""

    def __init__(self, window=20):
        self.window = window

    def values_in_time(self, past_returns, **kwargs):
        """This method computes the quantity of interest.

        It has many arguments, we only need to use past_returns
        in this case.

        NOTE: the last column of `past_returns` are the cash returns.
        You need to explicitely skip them otherwise the compiler will
        throw an error.
        """
        return past_returns.iloc[-self.window :, :-1].mean()


# Here we define a class to forecast covariances
class CustomCovariance:
    """Covariance computed on recent window of past returns."""

    def __init__(self, window=20, variances=None):
        self.window = window
        if variances is not None:
            self.custom = True
            # remove tz-awareness
            variances.index = variances.index.tz_localize(None)
            self.variances = variances

    def values_in_time(self, past_returns, **kwargs):
        """This method computes the quantity of interest.

        It has many arguments, we only need to use past_returns
        in this case.

        NOTE: the last column of `past_returns` are the cash returns.
        You need to explicitely skip them otherwise the compiler will
        throw an error.
        """
        if self.custom and "CUSTOM" in past_returns.columns:
            t = past_returns.index[-1]
            custom_var = past_returns["CUSTOM"].var()

            # remove custom from the past_returns
            past_returns = past_returns.drop("CUSTOM", axis=1)
            cov = past_returns.iloc[-self.window :, :-1].cov()
            # add custom back to the past_returns as column and row of zeros
            cov["CUSTOM"] = 0
            cov.loc["CUSTOM"] = 0
            cov.loc["CUSTOM", "CUSTOM"] = custom_var
            return cov
        else:
            return past_returns.iloc[-self.window :, :-1].cov()


def generate_private_asset(reference_df, sparsity="yearly", initial_price=100, volume_range=(100, 1000)):
    """
    Generate a custom private asset with sparse returns and volumes.

    Parameters:
    reference_df (pd.DataFrame): DataFrame with a DatetimeIndex to align with.
    sparsity (float): Fraction of dates to be populated (0 < sparsity <= 1).
    price_range (tuple): Tuple indicating the min and max range for prices.
    volume_range (tuple): Tuple indicating the min and max range for volumes.

    Returns:
    pd.DataFrame: DataFrame for prices.
    pd.DataFrame: DataFrame for volumes.
    """
    rng = np.random.default_rng(12345)

    # Dates from the reference DataFrame
    dates = reference_df.index

    # Determine which dates to populate based on sparsity
    populated_dates = rng.choice(dates, size=int(sparsity * len(dates)), replace=False)
    populated_dates.sort()  # Sort the dates

    # Generate random prices and volumes for the populated dates
    # use GBM to generate prices
    prices = np.zeros(len(populated_dates))
    prices[0] = initial_price
    for i in range(1, len(prices)):
        prices[i] = prices[i - 1] * np.exp(rng.normal(0, 0.01))
    volumes = rng.uniform(volume_range[0], volume_range[1], size=len(populated_dates))
    returns = np.zeros(len(populated_dates))
    returns[1:] = prices[1:] / prices[:-1] - 1
    variances = pd.Series(returns).rolling(window=4).var(ddof=1).values

    # Create DataFrames
    prices_df = pd.DataFrame(index=dates, columns=["CUSTOM"], dtype=float)
    volumes_df = pd.DataFrame(index=dates, columns=["CUSTOM"], dtype=float)
    returns_df = pd.DataFrame(index=dates, columns=["CUSTOM"], dtype=float)
    variances_df = pd.DataFrame(index=dates, columns=["CUSTOM"], dtype=float)

    # Populate the DataFrames
    prices_df.loc[populated_dates, "CUSTOM"] = prices
    volumes_df.loc[populated_dates, "CUSTOM"] = volumes
    returns_df.loc[populated_dates, "CUSTOM"] = returns
    variances_df.loc[populated_dates, "CUSTOM"] = variances

    return prices_df, volumes_df, returns_df, variances_df


# define the hyperparameters
window_mu = 4  # 4 quarters of data
window_sigma = 4  # 4 quarters of data
BASE_LOCATION = BASE_LOCATION = Path.home() / "cvxportfolio_data"
trading_frequency = "quarterly"

portfolio = cvx.Portfolio()

universe = ["AAPL", "AMZN", "UBER", "ZM", "CVX", "TSLA", "GM", "ABNB", "CTAS", "GOOG"]
start_date = "2020-01-01"

# get price data (assuming you don't have it already)
public_data = cvx.data.DownloadedMarketData(
    universe=universe,
    cash_key="USDOLLAR",
    base_location=BASE_LOCATION,
    min_history=pd.Timedelta("365.24d"),
    datasource="YahooFinance",
)
public_assets_prices = public_data.prices
public_assets_volumes = public_data.volumes
public_assets_returns = public_data.returns

# fabricated custom asset data
private_assets_prices, private_assets_volumes, private_assets_returns, private_assets_vars = generate_private_asset(
    public_assets_prices,
    sparsity="quarterly",
)

# merge public and private assets
assets_prices = pd.concat([public_assets_prices, private_assets_prices], axis=1)
assets_volumes = pd.concat([public_assets_volumes, private_assets_volumes], axis=1)
assets_returns = pd.concat([public_assets_returns, private_assets_returns], axis=1)
# put "USDOLLAR" columns last in the DataFrame
assets_returns = assets_returns[assets_returns.columns.drop("USDOLLAR").tolist() + ["USDOLLAR"]]
# temp = np.exp(np.log(1 + assets_returns["CUSTOM"]).resample("QS", closed="left", label="left").sum(min_count=1)) - 1

# create a data object
data = cvx.data.UserProvidedMarketData(
    prices=assets_prices,
    volumes=assets_volumes,
    returns=assets_returns,
)
del data.trading_frequency
# Create an instance of MeanVariancePortfolio
mvo = MeanVarianceStrategy(name="MVO-limit-risk")

# Set the risk and return targets
mvo.set_risk_target(risk_target=0.004, Sigma=CustomCovariance(window_sigma, variances=private_assets_vars))

# Set the planning horizon
mvo.set_planning_horizon(1)

# Add transaction cost penalty
mvo.add_transaction_penalty(gamma_trade=0.5)

# Set up the simulator with required data
mvo.set_simulator_from_data(trading_frequency=trading_frequency, **vars(data))

mvo2 = MeanVarianceStrategy(name="MVO-limit-return")
mvo2.set_return_target(return_target=0.025, r_hat=CustomMeanReturn(window_mu))
mvo2.set_planning_horizon(1)
mvo2.add_transaction_penalty(gamma_trade=0.5)
mvo2.set_simulator_from_data(trading_frequency=trading_frequency, **vars(data))

portfolio.add_strategies([mvo, mvo2])

portfolio.backtest(start_time=start_date)

for res in portfolio.results:
    print(res.result)

fig = go.Figure()
for col in portfolio.performance.columns:
    fig.add_trace(go.Scatter(x=portfolio.performance.index, y=portfolio.performance[col], name=col))

fig.update_layout(title="Multi-Period Optimization", xaxis_title="Date", yaxis_title="Value")
fig.show()

num_strats = len(portfolio.strategies)
fig = make_subplots(rows=num_strats, cols=1, shared_xaxes=True, vertical_spacing=0.02)
for i, res in enumerate(portfolio.results):
    f = res.plot_weights()
    for data in f.data:
        fig.add_trace(data, row=i + 1, col=1)

fig.update_layout(title="Multi-Period Optimization", xaxis_title="Date", yaxis_title="Value")
fig.show()

portfolio.get_next_weights(data.prices.index[-1])


# Print and plot the results
print("\n# MULTI-PERIOD OPTIMIZATION\n", portfolio.results)
