# coding=utf-8

"""
Goal: Implementing some data augmentation techniques for the stock market time series.
Authors: Thibaut Théate and Damien Ernst
Institution: University of Liège
"""

###############################################################################
################################### Imports ###################################
###############################################################################

import copy
import numpy as np

from tradingEnv import TradingEnv



###############################################################################
################################ Global variables #############################
###############################################################################

# Default ranges for the parameters of the data augmentation techniques 
shiftRange = [0]
stretchRange = [1]
filterRange = [5]
noiseRange = [0]



###############################################################################
############################# Class DataAugmentation ##########################
###############################################################################

class DataAugmentation:
    """
    GOAL: Implementing some data augmentation techniques for stock time series.
    
    VARIABLES: /
                                
    METHODS:    - __init__: Initialization of some class variables.
                - shiftTimeSeries: Generate a new trading environment by simply
                                   shifting up or down the volume time series.
                - stretching: Generate a new trading environment by stretching
                              or contracting the original price time series.
                - noiseAddition: Generate a new trading environment by adding
                                 some noise to the original time series.
                - lowPassFilter: Generate a new trading environment by filtering
                                 (low-pass) the original time series.
                - generate: Generate a set of new trading environments based on the
                            data augmentation techniques implemented.       
    """
    
    def shiftTimeSeries(self, tradingEnv, shiftMagnitude=0):
        """
        GOAL: Generate a new trading environment by simply shifting up or down
              the volume time series.
        
        INPUTS: - tradingEnv: Original trading environment to augment.
                - shiftMagnitude: Magnitude of the shift.
        
        OUTPUTS: - newTradingEnv: New trading environment generated.
        """

        # Creation of the new trading environment
        newTradingEnv = copy.deepcopy(tradingEnv)

        # Constraint on the shift magnitude
        if shiftMagnitude < 0:
            minValue = np.min(tradingEnv.data['Volume'])
            shiftMagnitude = max(-minValue, shiftMagnitude)
        
        # Shifting of the volume time series
        newTradingEnv.data['Volume'] += shiftMagnitude

        # Return the new trading environment generated
        return newTradingEnv


    def streching(self, tradingEnv, factor=1):
        """
        GOAL: Generate a new trading environment by stretching
              or contracting the original price time series, by 
              multiplying the returns by a certain factor.
        
        INPUTS: - tradingEnv: Original trading environment to augment.
                - factor: Stretching/contraction factor.
        
        OUTPUTS: - newTradingEnv: New trading environment generated.
        """

        # Creation of the new trading environment
        newTradingEnv = copy.deepcopy(tradingEnv)

        # Application of the stretching/contraction operation
        returns = newTradingEnv.data['Close'].pct_change() * factor
        
        # Risoluzione ChainedAssignment/FutureWarning: mazziamo le posizioni delle colonne prima del loop
        col_close = newTradingEnv.data.columns.get_loc('Close')
        col_low = newTradingEnv.data.columns.get_loc('Low')
        col_high = newTradingEnv.data.columns.get_loc('High')
        col_open = newTradingEnv.data.columns.get_loc('Open')

        for i in range(1, len(newTradingEnv.data.index)):
            newTradingEnv.data.iat[i, col_close] = newTradingEnv.data.iat[i-1, col_close] * (1 + returns.iloc[i])
            newTradingEnv.data.iat[i, col_low] = newTradingEnv.data.iat[i, col_close] * tradingEnv.data['Low'].iloc[i]/tradingEnv.data['Close'].iloc[i]
            newTradingEnv.data.iat[i, col_high] = newTradingEnv.data.iat[i, col_close] * tradingEnv.data['High'].iloc[i]/tradingEnv.data['Close'].iloc[i]
            newTradingEnv.data.iat[i, col_open] = newTradingEnv.data.iat[i-1, col_close]

        # Return the new trading environment generated
        return newTradingEnv


    def noiseAddition(self, tradingEnv, stdev=1):
        """
        GOAL: Generate a new trading environment by adding some gaussian
              random noise to the original time series.
        
        INPUTS: - tradingEnv: Original trading environment to augment.
                - stdev: Standard deviation of the generated white noise.
        
        OUTPUTS: - newTradingEnv: New trading environment generated.
        """

        # Creation of a new trading environment
        newTradingEnv = copy.deepcopy(tradingEnv)

        # Ricaviamo gli indici per evitare ChainedAssignmentError e FutureWarning
        col_close = newTradingEnv.data.columns.get_loc('Close')
        col_low = newTradingEnv.data.columns.get_loc('Low')
        col_high = newTradingEnv.data.columns.get_loc('High')
        col_volume = newTradingEnv.data.columns.get_loc('Volume')
        col_open = newTradingEnv.data.columns.get_loc('Open')

        # Generation of the new noisy time series
        for i in range(1, len(newTradingEnv.data.index)):
            # Generation of artificial gaussian random noises
            price = newTradingEnv.data.iat[i, col_close]
            volume = newTradingEnv.data.iat[i, col_volume]
            priceNoise = np.random.normal(0, stdev*(price/100))
            volumeNoise = np.random.normal(0, stdev*(volume/100))

            # Addition of the artificial noise generated usando .iat
            newTradingEnv.data.iat[i, col_close] *= (1 + priceNoise/100)
            newTradingEnv.data.iat[i, col_low] *= (1 + priceNoise/100)
            newTradingEnv.data.iat[i, col_high] *= (1 + priceNoise/100)
            newTradingEnv.data.iat[i, col_volume] *= (1 + volumeNoise/100)
            newTradingEnv.data.iat[i, col_open] = newTradingEnv.data.iat[i-1, col_close]

        # Return the new trading environment generated
        return newTradingEnv


    def lowPassFilter(self, tradingEnv, order=5):
        """
        GOAL: Generate a new trading environment by filtering
              (low-pass filter) the original time series.
        
        INPUTS: - tradingEnv: Original trading environment to augment.
                - order: Order of the filtering operation.
        
        OUTPUTS: - newTradingEnv: New trading environment generated.
        """

        # Creation of a new trading environment
        newTradingEnv = copy.deepcopy(tradingEnv)

        # Application of a filtering (low-pass) operation
        newTradingEnv.data['Close'] = newTradingEnv.data['Close'].rolling(window=order).mean()
        newTradingEnv.data['Low'] = newTradingEnv.data['Low'].rolling(window=order).mean()
        newTradingEnv.data['High'] = newTradingEnv.data['High'].rolling(window=order).mean()
        newTradingEnv.data['Volume'] = newTradingEnv.data['Volume'].rolling(window=order).mean()
        
        # Ricaviamo gli indici numerici delle colonne
        col_close = newTradingEnv.data.columns.get_loc('Close')
        col_low = newTradingEnv.data.columns.get_loc('Low')
        col_high = newTradingEnv.data.columns.get_loc('High')
        col_volume = newTradingEnv.data.columns.get_loc('Volume')
        col_open = newTradingEnv.data.columns.get_loc('Open')

        for i in range(order):
            newTradingEnv.data.iat[i, col_close] = tradingEnv.data['Close'].iloc[i]
            newTradingEnv.data.iat[i, col_low] = tradingEnv.data['Low'].iloc[i]
            newTradingEnv.data.iat[i, col_high] = tradingEnv.data['High'].iloc[i]
            newTradingEnv.data.iat[i, col_volume] = tradingEnv.data['Volume'].iloc[i]
            
        newTradingEnv.data['Open'] = newTradingEnv.data['Close'].shift(1)
        newTradingEnv.data.iat[0, col_open] = tradingEnv.data['Open'].iloc[0]

        # Return the new trading environment generated
        return newTradingEnv


    def generate(self, tradingEnv):
        """
        Generate a set of new trading environments based on the data
        augmentation techniques implemented.
        
        :param: - tradingEnv: Original trading environment to augment.
        
        :return: - tradingEnvList: List of trading environments generated
                                   by data augmentation techniques.
        """

        # Application of the data augmentation techniques to generate the new trading environments
        tradingEnvList = []
        for shift in shiftRange:
            tradingEnvShifted = self.shiftTimeSeries(tradingEnv, shift)
            for stretch in stretchRange:
                tradingEnvStretched = self.streching(tradingEnvShifted, stretch)
                for order in filterRange:
                    tradingEnvFiltered = self.lowPassFilter(tradingEnvStretched, order)
                    for noise in noiseRange:
                        tradingEnvList.append(self.noiseAddition(tradingEnvFiltered, noise))
        return tradingEnvList