# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 11:38:06 2026

@author: RC
"""





# ================================ LIBRARIES ================================ #

# import os
# import json
# import time
# from pathlib import Path

import numpy  as np
import pandas as pd
# import matplotlib.pyplot as plt

# import RileysLibrary as rc

import whiterunMarket as wrm
#

sessionLog = wrm.sessionLog # alias
# =========================================================================== #





# ================================ FUNCTIONS ================================ #
####    EXPERIMENT 1
# Define a class for Experiment 1
class experiment1 :
    def __init__(self, func, verbose=True, **kwargs) :
        # Experiment 1 -- Parameters
        #   Include experiment specific parameters specifications, so we can
        #   test alternate specifications within the same session
        
        # ['EXPERIMENT PARAMS']
        self.VILLAGES   = kwargs.get('VILLAGES', wrm.VILLAGES) # number of villages to text
        self.DAYS       = kwargs.get('DAYS', wrm.DAYS) # number of days to simulate
        
        # ['WORLD PARAMS']
        self.FARMERS    = kwargs.get('FARMERS', wrm.FARMERS)  # number of farmer per village
        self.PROD_WHEAT = kwargs.get('PROD_WHEAT',wrm.PROD_WHEAT) # amount of wheat produced per production action
        
        # ['MARKET PARAMS']
        self.PR_BREAD   = kwargs.get('PR_BREAD', wrm.PR_BREAD) # initial price of bread
        self.PR_WHEAT   = kwargs.get('PR_WHEAT', wrm.PR_WHEAT) # initial price of wheat
        
        # ['FARMER PARAMS']
        self.EXHAUSTION = kwargs.get('EXHAUSTION', wrm.EXHAUSTION) # amount of hunger gained per production action
        self.STARVATION = kwargs.get('STARVATION', wrm.STARVATION) # max amount of hunger allowed before starvation
        self.RECOVERY   = kwargs.get('RECOVERY', wrm.RECOVERY) # eating resets hunger if 0, else reduces hunger
        
        # Save distro function to pass to farmer __init__
        self.func = func
        
        # Pass verbose bin to self
        self.verbose = verbose
        
    #   end def init
    
    def runSimulation (self) :
        # Iter thru village simulations
        self.outcome = pd.DataFrame()
        for v in range(self.VILLAGES) :
            # Initialize counter
            dayCount = 0
            
            # Initialize village
            dictFarmers = {f : wrm.agentFarmer(self.func) for f in range(self.FARMERS)
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
                            farmer.hunger = (
                                0 if self.RECOVERY==0 
                                else farmer.hunger - self.RECOVERY
                                ) #TODO: Test hunger reduction
                            
                        else : # if you don't eat, you get hungry
                            farmer.hunger += 1 #TODO: Test idle hunger gain
                            
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
            #   Reformat farmer results to dataframe
            rows = []
            for farmer_id, farmer in dictFarmers.items():
                # Format per-row w row : Dict()
                row = vars(farmer).copy()      # all farmer attributes
                row.pop("prefs")               # remove the nested namedtuple
                row.update(farmer.prefs._asdict())  # expand it
                row["farmer"] = farmer_id
                row['days'] = dayCount
                row['trial'] = v
                
                # Attach row to working-df
                rows.append(row)
                
            #   end for
            # Convert list of dicts to DataFrame
            df = pd.DataFrame(rows)
            df = df[['trial','farmer','status','days','bread','wheat','gold',
                     'hunger','gamma','omega','beta','eta'
                     ]].copy() # I want the columns arranged a certain way
            
            # Save results; notify user
            self.outcome = pd.concat([self.outcome, df], ignore_index=True)
            if self.verbose : sessionLog.print('Village '+str(v)+' complete!')
        #   end for villages -- end of simulations
    #   end def runSim
#   end experiment1

def main () :
    
    ####    EXPERIMENT 1    ####
    # Experiment 1 A
    sessionLog.print('Experiment 1.A (Log-Normal) start')
    
    m = 0
    v = 1
    
    exp1a = experiment1(lambda : np.random.lognormal(mean=m, sigma=v, size=4))
    exp1a.runSimulation()
    sessionLog.print('\n')
    
    
    
    # Experiment 1 B
    sessionLog.print('Experiment 1.B (Gamma) start')
    
    alpha = 1
    beta  = 1
    
    exp1b = experiment1(lambda : np.random.gamma(alpha,1/beta, size=4))
    exp1b.runSimulation()
    sessionLog.print('\n')
    
    
    
    # Experiment 1 C
    sessionLog.print('Experiment 1.C (Dirichlet) start')
    
    exp1c = experiment1(lambda : np.random.dirichlet([1.0,1.0,1.0,1.0]))
    exp1c.runSimulation()
    sessionLog.print('\n')
    
    
    
    # Experiment 1 D
    sessionLog.print('Experiment 1.D (Dirichlet-adjusted) start')
    
    vecParam = [0.198727,  0.102454,  0.637629,  0.061190]
    exp1d = experiment1(
        lambda : np.random.dirichlet(vecParam)
        )
    exp1d.runSimulation()
    sessionLog.print('\n')
    
    #TODO: Retry with normalized uniform distro draws
    
    ####
    
    
    
    
    
    ####    EXPERIMENT RESULTS  ####
    # Experiment 1 Result EDA
    sessionLog.print('Experiment run complete; starting evaluation of results.')
    
    wrm.post_exp_eda1(exp1a.outcome, 'Log-Normal (1)')
    
    
    
    wrm.post_exp_eda1(exp1b.outcome, 'Gamma (1)')
    
    
    
    wrm.post_exp_eda1(exp1c.outcome, 'Dirichlet (1)')
    
    
    
    wrm.post_exp_eda1(exp1d.outcome, 'Dirichlet-adj. (1)')
    
    
    
#   end 
# =========================================================================== #





# =================================== MAIN ================================== #
if __name__ == "__main__" :
    #   Initialize console log
    sessionLog = wrm.logger()
    sessionLog.print(__doc__)
    
    #   Run Script
    main()
    
#   endif
# =========================================================================== #




