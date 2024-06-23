import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import List

logger = logging.getLogger(name=__name__)
logger.setLevel(logging.INFO)
stdout_handler = logging.StreamHandler()
format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
stdout_handler.setFormatter(format)
logger.addHandler(stdout_handler)

# Fixed Variables
YEARS_REMAINING = 110-25
STARTING_EQUITY = 6e4
STARTING_CASH = 6e4
AFTER_TAX_SALARY = 1e5
FIXED_EXPENSES = 5e4  # Maybe just under inflation
ANNUAL_DRAWDOWN = FIXED_EXPENSES + 0
SALARY_GROWTH_RATE = 0.02  # Maybe just under inflation can be done with np.clip to restrict size of the increase
PORTFOLIO_DISTRIBUTION = {
    "cash": 0.1,
    "equity": 0.9
}
SNP_DIV_YIELD = 0.014
INTEREST_RATE = 0.03
TAIL_RISK_COST = 3e4  # 30k AUD drawdown
TAIL_RISK_PROBABILITY = 1/30  # 1 in every 30 year event
RUNS = 1

assert sum(PORTFOLIO_DISTRIBUTION.values()) == 1, logger.error("Porftolio distribution does not sum to 1")

def portfolio_annuity(salary, snp_return_percent, drawdown, tail_risk):
    """
    Calc portfolio value
    """
    cash = STARTING_CASH
    equity = STARTING_EQUITY
    cash_pos = []
    equity_pos = []
    dividend_pos = []
    interest_pos = []
    snp_return_pos = []
    tail_risk_pos = []
    for sal, snp_perc, dd, tail in zip(salary, snp_return_percent, drawdown, tail_risk):
        dividend = equity * SNP_DIV_YIELD
        interest = cash * INTEREST_RATE
        snp_return = equity * snp_perc
        annual_distribution  = sal + dividend + interest - tail - dd
        cash += annual_distribution * PORTFOLIO_DISTRIBUTION["cash"]
        equity += snp_return + (annual_distribution * PORTFOLIO_DISTRIBUTION["equity"])
        cash_pos.append(cash)
        equity_pos.append(equity)
        dividend_pos.append(dividend)
        interest_pos.append(interest)
        snp_return_pos.append(snp_return)
        tail_risk_pos.append(tail)
    cash_pos = np.array(cash_pos)
    equity_pos = np.array(equity_pos)
    dividend_pos = np.array(dividend_pos)
    interest_pos = np.array(interest_pos)
    snp_return_pos = np.array(snp_return_pos)
    tail_risk_pos = np.array(tail_risk_pos)

    return {
        "cash": cash_pos,
        "equity": equity_pos,
        "dividend": dividend_pos,
        "interest": interest_pos,
        "snp_return": snp_return_pos,
        "tail_risk": tail_risk_pos,
    }


def monte_carlo(runs=RUNS):
    """
    Run monte carlo.
    """
    logger.info(f"Running {runs} simulations.")
    np.random.seed(42)
    simiulations: List
    simiulations = []

    for i in range(runs):
        snp_return_percent = np.random.normal(loc=0.12, scale=0.2 , size=YEARS_REMAINING)
        inflation = np.random.normal(loc=0.03, scale=0.035 , size=YEARS_REMAINING)
        drawdown = ANNUAL_DRAWDOWN * np.cumprod(1 + inflation)
        salary_growth = np.repeat(1 + SALARY_GROWTH_RATE, YEARS_REMAINING)
        salary = AFTER_TAX_SALARY * np.cumprod(salary_growth)
        tail_risk = TAIL_RISK_COST * np.random.binomial(n=1, p=TAIL_RISK_PROBABILITY, size=YEARS_REMAINING)
        portfolio = portfolio_annuity(salary=salary, snp_return_percent=snp_return_percent, drawdown=drawdown, tail_risk=tail_risk)

        year = np.array(list(range(YEARS_REMAINING))) + 1
        simiulations.append(
            {
                "simulation": i,
                "year": year,
                "nest_egg": portfolio["cash"] + portfolio["equity"],
                "cash_position": portfolio["cash"],
                "equity_position": portfolio["equity"],
                "salary": salary,
                "salary_growth": salary_growth,
                "div_yield": np.repeat(SNP_DIV_YIELD, YEARS_REMAINING),
                "div_return": portfolio["dividend"],
                "snp_return": portfolio["snp_return"],
                "snp_return_percent": snp_return_percent,
                "interest_return": portfolio["interest"],
                "inflation": inflation,
                "living_expenses": drawdown,
                "tail_risk": portfolio["tail_risk"],
            }
        )
    return np.array(simiulations)


def sim_analysis(sims, column):
    return pd.DataFrame({i["simulation"]: i[column] for i in sims})


if __name__ == "__main__":
    monte_sims = monte_carlo()

    metrics = ["nest_egg", "cash_position", "equity_position", "snp_return_percent", "inflation", "tail_risk"]
    columns = 2
    kicker = 1 if len(metrics) % columns > 0 else 0
    fig, axs = plt.subplots(len(metrics)//columns + kicker, columns)
    for i in range(len(metrics)):
        df = sim_analysis(sims=monte_sims, column=metrics[i])
        for c in df.columns:
            axs[i//2, i%2].plot(df.index, df[c], linestyle='-')
            axs[i//2, i%2].set_title(metrics[i])
            axs[i//2, i%2].grid(True)

    plt.show()




# df.to_csv("nest_egg")
# positively skewed annual return distribution to use a log normal distribution to estimate S&P returns


# Annual Drawdown x inflation
# What are the random variables in my nest egg:
# s&p annual returns
# Inflation
# Short Term Interest Rates
# Tail Risk Scenario (emergencies)


#### S&P Return Distribution for the last 98yrs
# count  98.000000
# mean   12.159490
# std    19.723113
# min   -43.340000
# 25%    -0.845000
# 50%    14.685000
# 75%    26.417500
# max    53.990000

#### Annualised Aus Quarterly Inflation
# count  276.000000
# mean     5.101087
# std      4.464645
# min     -1.300000
# 25%      2.075000
# 50%      3.300000
# 75%      7.600000
# max     23.900000

#### numpy normal distribution (1,1,1e7)
# mean   9.999202e-01
# std    1.000074e+00
# min   -4.269521e+00
# 25%    3.255955e-01
# 50%    9.998640e-01
# 75%    1.674581e+00
# max    6.249556e+00

# from functools import lru_cache

# @lru_cache(maxsize=1)
# def calc_cash(cash, interest=INTEREST_RATE):
#     cash
#     return cash * (1 + interest)
