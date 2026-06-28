# coding=utf-8

"""
Goal: Accurately estimating the performance of a trading strategy.
Authors: Thibaut Théate and Damien Ernst
Institution: University of Liège
"""

###############################################################################
################################### Imports ###################################
###############################################################################

import numpy as np

from tabulate import tabulate
from matplotlib import pyplot as plt



###############################################################################
######################### Class PerformanceEstimator ##########################
###############################################################################

class PerformanceEstimator:
    """
    GOAL: Accurately estimating the performance of a trading strategy, by
          computing many different performance indicators.
    """

    def __init__(self, tradingData):
        self.data = tradingData


    def computePnL(self):
        # Risolto FutureWarning usando .iloc per gli accessi posizionali
        self.PnL = self.data["Money"].iloc[-1] - self.data["Money"].iloc[0]
        return self.PnL
    

    def computeAnnualizedReturn(self):
        cumulativeReturn = self.data['Returns'].cumsum()
        # Risolto外 FutureWarning usando .iloc[-1] invece di [-1]
        cumulativeReturn = cumulativeReturn.iloc[-1]
        
        start = self.data.index[0].to_pydatetime()
        end = self.data.index[-1].to_pydatetime()     
        timeElapsed = end - start
        timeElapsed = timeElapsed.days

        if(cumulativeReturn > -1):
            self.annualizedReturn = 100 * (((1 + cumulativeReturn) ** (365/timeElapsed)) - 1)
        else:
            self.annualizedReturn = -100
        return self.annualizedReturn
    
    
    def computeAnnualizedVolatility(self):
        self.annualizedVolatily = 100 * np.sqrt(252) * self.data['Returns'].std()
        return self.annualizedVolatily
    
    
    def computeSharpeRatio(self, riskFreeRate=0):
        expectedReturn = self.data['Returns'].mean()
        volatility = self.data['Returns'].std()
        
        if expectedReturn != 0 and volatility != 0:
            self.sharpeRatio = np.sqrt(252) * (expectedReturn - riskFreeRate)/volatility
        else:
            self.sharpeRatio = 0
        return self.sharpeRatio
    
    
    def computeSortinoRatio(self, riskFreeRate=0):
        expectedReturn = np.mean(self.data['Returns'])
        negativeReturns = [returns for returns in self.data['Returns'] if returns < 0]
        volatility = np.std(negativeReturns)
        
        if expectedReturn != 0 and volatility != 0:
            self.sortinoRatio = np.sqrt(252) * (expectedReturn - riskFreeRate)/volatility
        else:
            self.sortinoRatio = 0
        return self.sortinoRatio
    
    
    def computeMaxDrawdown(self, plotting=False):
        capital = self.data['Money'].values
        through = np.argmax(np.maximum.accumulate(capital) - capital)
        if through != 0:
            peak = np.argmax(capital[:through])
            self.maxDD = 100 * (capital[peak] - capital[through])/capital[peak]
            self.maxDDD = through - peak
        else:
            self.maxDD = 0
            self.maxDDD = 0
            return self.maxDD, self.maxDDD

        if plotting:
            plt.figure(figsize=(10, 4))
            plt.plot(self.data['Money'], lw=2, color='Blue')
            plt.plot([self.data.iloc[[peak]].index, self.data.iloc[[through]].index],
                     [capital[peak], capital[through]], 'o', color='Red', markersize=5)
            plt.xlabel('Time')
            plt.ylabel('Price')
            plt.savefig(''.join(['Figures/','MaximumDrawDown', '.png']))

        return self.maxDD, self.maxDDD
    

    def computeProfitability(self):
        good = 0
        bad = 0
        profit = 0
        loss = 0
        # Risolto FutureWarning usando .iloc[i] per controllare l'azione
        index = next((i for i in range(len(self.data.index)) if self.data['Action'].iloc[i] != 0), None)
        if index == None:
            self.profitability = 0
            self.averageProfitLossRatio = 0
            return self.profitability, self.averageProfitLossRatio
        money = self.data['Money'].iloc[index]

        # Risolto FutureWarning all'interno del loop usando .iloc
        for i in range(index+1, len(self.data.index)):
            if(self.data['Action'].iloc[i] != 0):
                delta = self.data['Money'].iloc[i] - money
                money = self.data['Money'].iloc[i]
                if(delta >= 0):
                    good += 1
                    profit += delta
                else:
                    bad += 1
                    loss -= delta

        delta = self.data['Money'].iloc[-1] - money
        if(delta >= 0):
            good += 1
            profit += delta
        else:
            bad += 1
            loss -= delta

        self.profitability = 100 * good/(good + bad)
         
        if(good != 0):
            profit /= good
        if(bad != 0):
            loss /= bad
        if(loss != 0):
            self.averageProfitLossRatio = profit/loss
        else:
            self.averageProfitLossRatio = float('Inf')

        return self.profitability, self.averageProfitLossRatio
        

    def computeSkewness(self):
        self.skewness = self.data["Returns"].skew()
        return self.skewness
        
    
    def computePerformance(self):
        self.computePnL()
        self.computeAnnualizedReturn()
        self.computeAnnualizedVolatility()
        self.computeProfitability()
        self.computeSharpeRatio()
        self.computeSortinoRatio()
        self.computeMaxDrawdown()
        self.computeSkewness()

        self.performanceTable = [["Profit & Loss (P&L)", "{0:.0f}".format(self.PnL)], 
                                 ["Annualized Return", "{0:.2f}".format(self.annualizedReturn) + '%'],
                                 ["Annualized Volatility", "{0:.2f}".format(self.annualilyVolatily if hasattr(self, 'annualilyVolatily') else self.annualizedVolatily) + '%'],
                                 ["Sharpe Ratio", "{0:.3f}".format(self.sharpeRatio)],
                                 ["Sortino Ratio", "{0:.3f}".format(self.sortinoRatio)],
                                 ["Maximum Drawdown", "{0:.2f}".format(self.maxDD) + '%'],
                                 ["Maximum Drawdown Duration", "{0:.0f}".format(self.maxDDD) + ' days'],
                                 ["Profitability", "{0:.2f}".format(self.profitability) + '%'],
                                 ["Ratio Average Profit/Loss", "{0:.3f}".format(self.averageProfitLossRatio)],
                                 ["Skewness", "{0:.3f}".format(self.skewness)]]
        
        return self.performanceTable

    def displayPerformance(self, name, market=""):
        self.computePerformance()
        headers = ["Performance Indicator", name]
        tabulation = tabulate(self.performanceTable, headers, tablefmt="fancy_grid")
        print(tabulation)
        
        # Format market prefix safely if provided
        market_prefix = f"{market}_" if market else ""
        
        # Example: Amazon_TDQN_performance.txt
        filename = f"Performances\\{market_prefix}{name}_performance.txt"
        
        with open(filename, "a", encoding="utf-8") as f:
            f.write("\n" + "="*50 + "\n")
            f.write(f"Execution Context: {market or 'Unknown Market'} | Strategy: {name}\n")
            f.write("="*50 + "\n")
            f.write(tabulation)
            f.write("\n")
        print(f"Table successfully saved/appended to {filename}")