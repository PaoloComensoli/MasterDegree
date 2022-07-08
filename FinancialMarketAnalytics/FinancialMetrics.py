import numpy as np
import pandas as pd
from scipy.stats import norm

def annualized_returns(returns):
    return returns.mean() * 252

def annualized_volatility(returns):
    return returns.std() * np.sqrt(252)

def sharpe_ratio(returns):
    sharpe_ratio = returns.mean() / returns.std()
    annualized_sharpe_ration = sharpe_ratio * np.sqrt(252)
    return annualized_sharpe_ration

def portfolio_efficiency(returns):
    return annualized_returns(returns) / annualized_volatility(returns)

def max_drawdown(returns):
    s = (returns + 1).cumprod()
    return np.ptp(s)/s.max()

def VaR(returns):
    VaR_90 = norm.ppf(1-0.9, returns.mean(), returns.std())
    VaR_95 = norm.ppf(1-0.95, returns.mean(), returns.std())
    VaR_99 = norm.ppf(1-0.99, returns.mean(), returns.std())
    return VaR_90, VaR_95, VaR_99

def IR(returns, benchmark):
    difference = returns - benchmark
    volatility = difference.std() * np.sqrt(252)
    information_ratio = difference.mean() / volatility
    return information_ratio


def get_base_metrics(portfolio_name, portfolio_returns):
    portfolio_returns = np.array(portfolio_returns)

    metrics = pd.DataFrame(columns=['Portfolio Title', 'Annualized Returns', 'Annualized Volatility'])
    metrics.loc[0] = [portfolio_name, annualized_returns(portfolio_returns), annualized_volatility(portfolio_returns)]
    return metrics

def get_advanced_metrics(portfolio_name, portfolio_returns, benchmark):
    portfolio_returns = np.array(portfolio_returns)
    benchmark = np.array(benchmark)
    metrics = pd.DataFrame(columns=['Portfolio Title', 'Sharpe Ratio', 'Efficiency', 'MDD', 'Var 90', 'Var 95', 'Var 99', 'IR'])
    VaRs = VaR(portfolio_returns)
    metrics.loc[0] = [
        portfolio_name, 
        sharpe_ratio(portfolio_returns), 
        portfolio_efficiency(portfolio_returns), 
        max_drawdown(portfolio_returns), 
        VaRs[0],
        VaRs[1],
        VaRs[2],
        IR(portfolio_returns, benchmark)
    ]
    return metrics
