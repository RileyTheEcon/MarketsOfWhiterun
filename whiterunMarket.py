# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 19:15:53 2026

@author: RC
"""





# ================================ LIBRARIES ================================ #

import os
import json
#import time
from pathlib import Path
from collections import namedtuple

import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt

#import RileysLibrary as rc
#



# Use named tuple to make references easier
PARAMS = ['gamma', 'omega', 'beta', 'eta']
PREFS  = namedtuple('PREFS', PARAMS)

# Import or set session parameters
if os.path.exists('config.json'):
    with open('config.json') as f:
        CONFIG = json.load(f)
else:
    # ['EXPERIMENT PARAMS']
    VILLAGES   = int(1E3) # number of villages to text
    DAYS       = int(1E3) # number of days to simulate
    
    # ['WORLD PARAMS']
    FARMERS    = 12  # number of farmer per village
    PROD_WHEAT = 3 # amount of wheat produced per production action
    
    # ['MARKET PARAMS']
    PR_BREAD   = 3 # initial price of bread
    PR_WHEAT   = 1 # initial price of wheat
    
    # ['FARMER PARAMS']
    # Farmer status change parameters
    EXHAUSTION = 1 # amount of hunger gained per production action
    STARVATION = 5 # max amount of hunger allowed before starvation
    RECOVERY   = 0 # eating resets hunger if 0, else reduces hunger
    
    # Initial conditions for farmer agent inventories
    BREAD      = 1
    WHEAT      = 0
    GOLD       = 3
    HUNGER     = 0
    
#   end CONFIG

# =========================================================================== #





# ================================ FUNCTIONS ================================ #
# Define console logging object
# --I'm in the middle of a big reformating effort of my library to get it closer to the standards to
# be included into PIP and CONDA-FORGE, so I'm hardcoding stuff here in case a reference has to be
# changed in the library.
class logger :
    def __init__ (
            self,
            writeTo = 'log/'
            ) :
        # Import library
        import logging
        
        # Make logging dir if it doesn't already exist
        Path(writeTo).mkdir(parents=True,exist_ok=True)
        
        # Establish basic log config regular memory dumps
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            handlers=[
                logging.FileHandler(
                    f'{writeTo}{pd.Timestamp("today"):%Y-%m-%d %H_%M}.txt'),
                logging.StreamHandler()  # also prints to console
            ]
        )
        
        # Create logging object
        self.log = logging.getLogger()
        
    #   end __init__
    #
    def print(self, *args, sep=' '):
        # Print text to user, save to log, dump memory
        message = sep.join(str(a) for a in args)
        self.log.info(
            f'[{pd.Timestamp("today"):%Y-%m-%d %H:%M:%S}] ' + message
        )
    #   end print
    #
#   end logger
#   Initialize console log
sessionLog = logger()
###############################
def post_exp_eda1 (
        outcome : pd.DataFrame(),
        title = ''
        ) :
    #TODO: Add plot export functionality
    
    # Get preference parameters for surviving farmers; print means
    params = outcome[PARAMS] # create params subset
    
    # count of farmers in village
    aveCount   = outcome.groupby('trial').count()['farmer'].mean() 
    # cross-village farmer survival rate
    rateFarmer = outcome['status'].sum()/len(outcome) 
    # implied village survival rate
    rateNaive  = (1 - (1 - rateFarmer)**aveCount) 
    # observed village survival rate
    rateTrial  = (outcome.groupby('trial').max()['days']==DAYS).sum()/VILLAGES 
    
    # Create normalized array
    dfNorm = outcome[['status']].copy()
    dfNorm.loc[:,PARAMS] = (
        outcome[PARAMS].div(outcome[PARAMS].sum(axis=1), axis=0)
        ).copy()
    
    sessionLog.print(
        '\n',title,'\n',
        f'farmer survival rate:  {round(100*rateFarmer, 2)}%\n', 
         f'village survival rate: {round(100*rateTrial, 2)}%',
        f'(Naive rate {round(100*rateNaive,2)}%)\n'
        '\n',
        "Param means by status:\n", 
        (outcome[['status']+PARAMS]
            .groupby('status')
            .mean()
            ),
        '\n\n',
        "Param means by status (norm.):\n", 
        (dfNorm
            .groupby('status')
            .mean()
            )
        )
    
    # Plot hist of days survived
    plt.hist(
        outcome['days'], 
        bins=30, 
        color='steelblue', 
        edgecolor='white'
        )
    plt.yscale("log")
    plt.xlabel('Days survived')
    plt.ylabel('Number of villages')
    plt.title('Village survival duration, '+title)
    plt.show();
    
    # Plot parameter distributions
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    axes = axes.ravel()      # or axes.flatten()
    
    for ax, param in zip(axes, params):
        outcome.loc[outcome.status == 0, param].hist(
            bins=30, alpha=0.5, ax=ax, label="status = 0"
        )
        outcome.loc[outcome.status == 1, param].hist(
            bins=30, alpha=0.5, ax=ax, label="status = 1"
        )
    
        ax.set_title(param)
        ax.legend()
        ax.set_yscale('log')
    plt.suptitle(f"{title}")
    plt.tight_layout()
    plt.show();
    
    # Do it again but normalized
    params = dfNorm[PARAMS] # overwrite alias for ease
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    axes = axes.ravel()      # or axes.flatten()
    
    for ax, param in zip(axes, params):
        dfNorm.loc[dfNorm.status == 0, param].hist(
            bins=30, alpha=0.5, ax=ax, label="status = 0"
        )
        dfNorm.loc[dfNorm.status == 1, param].hist(
            bins=30, alpha=0.5, ax=ax, label="status = 1"
        )
    
        ax.set_title(param)
        ax.legend()
        ax.set_yscale('log')
    plt.suptitle(f"{title} (norm.)")
    plt.tight_layout()
    plt.show();
    
####
def agent_quantity (
        agent, 
        price, 
        **kwargs
        ) :
    ''' Provided an agent-object with inventory and utility function, find the
        quantity of a good to transact until marginal utility is negative
    '''
    # Get resource relation -- 1 = increasing ; -1 = decreasing
    dWheat = kwargs.get('wheat', 0)
    dBread = kwargs.get('bread', 0)
    dGold  = kwargs.get('gold',  0)

    #   Define assistant functions
    # Create theoretical inventory, given (q)uantity transacted
    def state (q) :
        return dict(
            wheat = agent.wheat + dWheat*q,
            bread = agent.bread + dBread*q,
            gold  = agent.gold  + dGold*price*q
            )
    #   end state

    # Check if resources are exhausted
    def feasible (q) :
        s = state(q)
        return (s['wheat'] >= 0) and (s['bread'] >= 0) and (s['gold'] >= 0)
    #   end feasible

    quantity = 0  # create in memory
    prevUtil = agent.utility(**state(quantity)) 
    
    #NOTE: The seq'ing here is important: Resource exhaustion must be check
    # before util is calc'ed or else -1 is passed to util which 
    # throws ln(0) error
    #TODO: Change decision behavior at marginUtil == 0
    while feasible(quantity+1) :
        # Calc current utility
        currentUtil = agent.utility(**state(quantity+1))
        
        # Check if marginal util negative
        if currentUtil - prevUtil < 0 : break
        
        # Increment + pass value for next iter
        quantity += 1
        prevUtil  = currentUtil
    #   end while

    return quantity
####


####    AGENTS
# Define a class for Farmer
class agentFarmer :
    def __init__(self, func, **kwargs) :
        self.status = 1
        self.prefs  = PREFS(*func())
        self.bread  = kwargs.get('bread', BREAD)   # initial bread inventory
        self.wheat  = kwargs.get('wheat', WHEAT)   # initial wheat inventory
        self.gold   = kwargs.get('gold',  GOLD)    # initial gold inventory
        self.hunger = kwargs.get('hunger',HUNGER)  # initial hunger status
    #   end init

    def utility (self, **kwargs) :
        # allow saved values to be temporarily overwritten for alter scenario
        gold   = kwargs.get('gold', getattr(self, 'gold'))
        wheat  = kwargs.get('wheat', getattr(self, 'wheat'))
        bread  = kwargs.get('bread', getattr(self, 'bread'))
        hunger = kwargs.get('hunger', getattr(self, 'hunger'))
        
        return (self.prefs.gamma * np.log(gold+1) + 
                self.prefs.omega * np.log(wheat+1) +
                self.prefs.beta  * np.log(bread+1) -
                self.prefs.eta   * hunger
               )
    #   end def utility

#   end class agentFarmer
####

####    MAIN
def main () :
    # Initialize
    sessionLog.print(__doc__)
    
    
#   end 
# =========================================================================== #





# =================================== MAIN ================================== #
if __name__ == "__main__" :
    main()
    
#   endif
# =========================================================================== #




