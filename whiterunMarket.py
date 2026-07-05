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



if os.path.exists('config.json'):
    with open('config.json') as f:
        CONFIG = json.load(f)
else:
    CONFIG = {'example': 'name'}
# =========================================================================== #





# ================================ FUNCTIONS ================================ #
# Define console logging object--in case I haven't pushed library update yet
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
def survivor_prefs(exp):
    """Return a DataFrame of (gamma, omega, beta, eta) for every farmer
    that was still alive at the end of its village's run."""
    records = [
        farmer.prefs._asdict()
        for trial in exp.outcome
        for farmer in trial['farmers'].values()
        if farmer.status > 0
    ]
    return pd.DataFrame(records)
#   end prefs
def village_survival (
        exp
        ) :
    ''' Return a list of village survival times
    '''
    return [v['days'] for v in exp.outcome]
####
def post_exp_eda1 (
        exp,
        title = ''
        ) :
    # Get preference parameters for surviving farmers; print means
    exp.dfPrefs = survivor_prefs(exp)
    sessionLog.print(
        title,"survivors, mean params:\n", 
        exp.dfPrefs.mean()
        )
    
    # Plot parameter distributions
    exp.dfPrefs.hist(bins=30, figsize=(8, 6))
    plt.suptitle(title+' prior — surviving farmers')
    plt.tight_layout()
    plt.show();
    
    # Get list of number of days villages survived
    exp.days = village_survival(exp)
    
    # Plot hist of days survived
    plt.hist(
        exp.days, 
        bins=30, 
        color='steelblue', 
        edgecolor='white'
        )
    plt.xlabel('Days survived')
    plt.ylabel('Number of villages')
    plt.title('Village survival duration, '+title)
    plt.show();
    
####


# EXPERIMENT 1
# Use named tuple to make references easier
Prefs = namedtuple('Prefs', ['gamma', 'omega', 'beta', 'eta'])

# Define a class for Farmer
class farmerObj :
    def __init__(self, func) :
        self.status = 1
        self.prefs  = Prefs(*func())
        self.bread  = 1
        self.wheat  = 0
        self.gold   = 3
        self.hunger = 0
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

#   end class farmerObj



# Define a class for Experiment 1
class experiment1 :
    def __init__(self, func) :
        # Experiment 1 -- Parameters
        self.VILLAGES = int(1E3) # number of villages to text
        self.DAYS     = int(1E3) # number of days to simulate
        self.FARMERS  = 12  # number of farmer per village

        self.PR_BREAD   = 3 # price of bread
        self.PR_WHEAT   = 1 # price of wheat
        self.PROD_WHEAT = 3 # amount of wheat produced per production action
        self.EXHAUSTION = 1 # amount of hunger gained per production action
        self.STARVATION = 5 # max amount of hunger allowed before starvation

        # Save distro function
        self.func = func

    #   end def init
    
    def runSimulation (self) :
        # Iter thru village simulations
        self.outcome = list()
        for v in range(self.VILLAGES) :
            # Initialize counter
            dayCount = 0
            
            # Initialize village
            dictFarmers = {f : farmerObj(self.func) for f in range(self.FARMERS)
                          } # dict of farmers
        
            # Iter thru time frame
            for t in range(self.DAYS) :
                # Stop village if all farmers died
                if not any(farmer.status > 0 for farmer in dictFarmers.values()):
                    break
                
                # Iter thru farmers
                for farmer in dictFarmers.values() :
                    if farmer.status > 0 : # skip dead farmers
                        marginalUtil = {
                            'Idle' : 0
                            } # Idle will always be 0

                        # Action: Buy bread
                        # Can't buy bread if you can't afford it
                        marginalUtil['Buy bread'] = (
                            -1 if farmer.gold < self.PR_BREAD else
                            farmer.utility(
                                bread=farmer.bread+(farmer.gold//self.PR_BREAD),
                                gold=farmer.gold-(farmer.gold//self.PR_BREAD)*self.PR_BREAD
                                ) - farmer.utility() # utility gain
                            )

                        # Action: Produce wheat
                        # Can't produce wheat if you are already starving
                        marginalUtil['Produce wheat'] = (
                            -1 if farmer.hunger >= self.STARVATION  else
                            farmer.utility(
                                wheat=farmer.wheat+self.PROD_WHEAT,
                                hunger=farmer.hunger+self.EXHAUSTION
                                ) - farmer.utility() # utility gain
                            )
                        
                        # Action: Sell wheat
                        # Must have wheat before you can sell it
                        marginalUtil['Sell wheat'] = (
                            -1 if farmer.wheat < 1  else
                            farmer.utility(
                                wheat=0,
                                gold=farmer.gold+(farmer.wheat*self.PR_WHEAT)
                                ) - farmer.utility() # utility gain
                            )
                        
                        # Do action that improves total utility
                        action = max(marginalUtil, key=marginalUtil.get)
                        
                        # Apply action
                        if action=='Buy bread' : 
                            units = farmer.gold//self.PR_BREAD
                            farmer.gold  -= units*self.PR_BREAD
                            farmer.bread += units
                        elif action=='Produce wheat': 
                            farmer.wheat += self.PROD_WHEAT
                            farmer.hunger += self.EXHAUSTION
                        elif action=='Sell wheat': 
                            farmer.gold += farmer.wheat*self.PR_WHEAT
                            farmer.wheat = 0
                        else : pass # "Idle" => do nothing
                        
                        # Farmers eat dinner
                        if farmer.bread>0 :
                            farmer.bread -= 1
                            farmer.hunger = 0
                            #TODO: Test hunger reduction
                        else : # if you don't eat, you get hungry
                            farmer.hunger += 1
                            #TODO: Test idle hunger gain
                        #   end dinner check
                        
                        if farmer.hunger>self.STARVATION : # deactivate hungry farmer
                            farmer.status = 0
                        #   end hunger check
                    #   end if farmer not dead
                #   end farmer iteration
                #TODO: Baker agents
                ''' In a different experiment we can see what happens when we
                    give bakers their own inventories and gold
                '''
                dayCount += 1
            #   end for days -- end of time frame
            self.outcome.append({
                'trial'   : v, # what trial number is it within experiment
                'days'    : dayCount, # total number of days complete
                'farmers' : dictFarmers
                })
            
            sessionLog.print('Village '+str(v)+' complete!')
        #   end for villages -- end of simulations
    #   end def runSim
#   end experiment1

# MAIN
def main () :
    # Initialize
    
    # Experiment 1 A
    sessionLog.print('Experiment 1.A (Log-Normal) start')
    
    m = 0
    v = 1
    
    exp1a = experiment1(lambda : np.random.lognormal(mean=m, sigma=v, size=4))
    exp1a.runSimulation()
    
    post_exp_eda1(exp1a, 'Log-Normal')
    
    
    
    
    
    # Experiment 1 B
    sessionLog.print('Experiment 1.B (Gamma) start')
    
    alpha = 1
    beta  = 1
    
    exp1b = experiment1(lambda : np.random.gamma(alpha,1/beta, size=4))
    exp1b.runSimulation()
    
    post_exp_eda1(exp1b, 'Gamma')
    
    
    
    
    
    # Experiment 1 C
    sessionLog.print('Experiment 1.C (Dirichlet) start')
    
    exp1c = experiment1(lambda : np.random.dirichlet([1.0,1.0,1.0,1.0]))
    exp1c.runSimulation()
    
    post_exp_eda1(exp1c, 'Dirichlet')
    
    
    
    
    
#   end 
# =========================================================================== #





# =================================== MAIN ================================== #
if __name__ == "__main__" :
    sessionLog.print(__doc__)
    
    main()
    
#   endif
# =========================================================================== #




