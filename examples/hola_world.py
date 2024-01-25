from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import cvxportfolio as cvx
from cvxportfolio.strategies import MeanVarianceStrategy
import plotly.graph_objects as go
import plotly.express as px

# Define parameters
universe = ["AAPL", "AMZN", "UBER", "ZM", "CVX", "TSLA", "GM", "ABNB", "CTAS", "GOOG"]


# Here we define a class to forecast expected returns
class WindowMeanReturn:
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
class WindowCovariance:
    """Covariance computed on recent window of past returns."""

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
        return past_returns.iloc[-self.window :, :-1].cov()


# define the hyperparameters
window_mu = 252
window_sigma = 252
BASE_LOCATION = BASE_LOCATION = Path.home() / "cvxportfolio_data"
trading_frequency = "quarterly"

portfolio = cvx.Portfolio()

universe = ["AAPL", "AMZN", "UBER", "ZM", "CVX", "TSLA", "GM", "ABNB", "CTAS", "GOOG"]
start_date = "2020-01-01"

# get price data
data = cvx.data.DownloadedMarketData(
    universe=universe,
    cash_key="USDOLLAR",
    base_location=BASE_LOCATION,
    min_history=pd.Timedelta("365.24d"),
    datasource="YahooFinance",
)
del data.trading_frequency


# Create an instance of MeanVariancePortfolio
mvo = MeanVarianceStrategy(name="MVO-limit-risk")

# Set the risk and return targets
mvo.set_risk_target(risk_target=0.005, Sigma=WindowCovariance(window_sigma))

# Set the planning horizon
mvo.set_planning_horizon(1)

# Add transaction cost penalty
mvo.add_transaction_penalty(gamma_trade=0.5)

# Set up the simulator with required data
mvo.set_simulator_from_data(trading_frequency="quarterly", **vars(data))

mvo2 = MeanVarianceStrategy(name="MVO-limit-return")
mvo2.set_return_target(return_target=0.02, r_hat=WindowMeanReturn(window_mu))
mvo2.set_planning_horizon(1)
mvo2.add_transaction_penalty(gamma_trade=0.5)
mvo2.set_simulator_from_data(trading_frequency="quarterly", **vars(data))

portfolio.add_strategies([mvo, mvo2])

portfolio.backtest(start_time=start_date)

for res in portfolio.results:
    print(res.result)

fig = go.Figure()
for col in portfolio.performance.columns:
    fig.add_trace(go.Scatter(x=portfolio.performance.index, y=portfolio.performance[col], name=col))

fig.update_layout(title="Multi-Period Optimization", xaxis_title="Date", yaxis_title="Value")
fig.show()


# Print and plot the results
print("\n# MULTI-PERIOD OPTIMIZATION\n", portfolio.results)
