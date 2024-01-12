import os
import matplotlib.pyplot as plt
import pandas as pd
import cvxportfolio as cvx

# Define parameters
universe = ["AAPL", "AMZN", "UBER", "ZM", "CVX", "TSLA", "GM", "ABNB", "CTAS", "GOOG"]

# Create an instance of MeanVariancePortfolio
portfolio = cvx.MeanVariancePortfolio()

# Set the risk and return targets
portfolio.set_risk_target(risk_target=0.005)

# Set the planning horizon
portfolio.set_planning_horizon(1)

# Add transaction cost penalty
portfolio.add_transaction_penalty(gamma_trade=0.5)

# Set up the simulator with required data
portfolio.set_simulator_from_universe(universe=universe, trading_frequency="quarterly")

# Perform the backtest
start_time = "2020-01-01"
portfolio.backtest(start_time=start_time)

# Print and plot the results
print("\n# MULTI-PERIOD OPTIMIZATION\n", portfolio.results)


portfolio.results.plot_returns()
portfolio.results.plot_weights()

# Save the plots if required
if "CVXPORTFOLIO_SAVE_PLOTS" in os.environ:
    plt.savefig("mean_variance_portfolio.png")
else:
    plt.show()
