import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cvxportfolio as cvx
from cvxportfolio.strategies import MeanVarianceStrategy

portfolio = cvx.Portfolio(enable_custom_assets=True)

# Define parameters
universe = ["AAPL", "AMZN", "UBER", "ZM", "CVX", "TSLA", "GM", "ABNB", "CTAS", "GOOG"]

# Create an instance of MeanVariancePortfolio
mvo = MeanVarianceStrategy()

# Set the risk and return targets
mvo.set_risk_target(risk_target=0.005)

# Set the planning horizon
mvo.set_planning_horizon(1)

# Add transaction cost penalty
mvo.add_transaction_penalty(gamma_trade=0.5)

# Set up the simulator with required data
mvo.set_simulator_from_universe(universe=universe, trading_frequency="quarterly")

# Perform the backtest
start_time = "2020-01-01"

mvo2 = MeanVarianceStrategy()
mvo2.set_risk_target(risk_target=0.004)
mvo2.set_planning_horizon(1)
mvo2.add_transaction_penalty(gamma_trade=0.5)
mvo2.set_simulator_from_universe(universe=universe, trading_frequency="quarterly")

portfolio.add_strategies([mvo, mvo2])
asset = {}
asset["house"] = {}
asset["house"]["name"] = "house"
# generate returns from a normal distribution with mean 0.01 and standard deviation 0.02 for all days in mvo.simulator.market_data.returns dataframe
asset["house"]["returns"] = pd.Series(
    np.random.normal(0.01, 0.02, len(mvo.simulator.market_data.returns)), index=mvo.simulator.market_data.returns.index
)
# generate std from a normal distribution with mean 0.1 and standard deviation 0.2 for all days in mvo.simulator.market_data.returns dataframe
asset["house"]["std"] = pd.Series(
    np.random.normal(0.1, 0.2, len(mvo.simulator.market_data.returns)), index=mvo.simulator.market_data.returns.index
)

portfolio.add_custom_asset(asset=asset["house"])

portfolio.backtest(start_time=start_time)

# Print and plot the results
print("\n# MULTI-PERIOD OPTIMIZATION\n", portfolio.results)


# portfolio.results.plot_returns()
# portfolio.results.plot_weights()


# Save the plots if required
if "CVXPORTFOLIO_SAVE_PLOTS" in os.environ:
    plt.savefig("mean_variance_portfolio.png")
else:
    plt.show()
