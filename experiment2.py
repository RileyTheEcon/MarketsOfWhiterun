# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 12:53:22 2026

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



# =========================================================================== #





# ================================ FUNCTIONS ================================ #
#### EXPERIMENT 2
# Define a class for Experiment 2
class experiment2 :
    def __init__(self, func, **kwargs) :
        # Experiment 2 -- Parameters
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
                        unitsBread = wrm.agent_quantity(
                            farmer, 
                            self.PR_BREAD, 
                            bread=1, gold=-1
                            )
                        marginalUtil['Buy bread'] = (
                            -1 if farmer.gold < self.PR_BREAD else
                            farmer.utility(
                                bread=farmer.bread+unitsBread,
                                gold=farmer.gold-unitsBread*self.PR_BREAD
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
                        unitsWheat = wrm.agent_quantity(
                            farmer, 
                            self.PR_WHEAT, 
                            wheat=-1, gold=1
                            )
                        marginalUtil['Sell wheat'] = (
                            -1 if farmer.wheat < 1  else
                            farmer.utility(
                                wheat=farmer.wheat-unitsWheat,
                                gold=farmer.gold+(unitsWheat*self.PR_WHEAT)
                                ) - farmer.utility() # utility gain
                            )
                        
                        # Do action that improves total utility
                        action = max(marginalUtil, key=marginalUtil.get)
                        
                        # Apply action
                        if action=='Buy bread' : 
                            farmer.gold  -= unitsBread*self.PR_BREAD
                            farmer.bread += unitsBread
                        elif action=='Produce wheat': 
                            farmer.wheat += self.PROD_WHEAT
                            farmer.hunger += self.EXHAUSTION
                        elif action=='Sell wheat':
                            farmer.gold  += unitsWheat*self.PR_WHEAT
                            farmer.wheat -= unitsWheat
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
            df = df[['trial','farmer','status', 'days','bread','wheat','gold',
                     'hunger','gamma','omega','beta','eta'
                     ]].copy() # I want the columns arranged a certain way
            
            # Save results; notify user
            self.outcome = pd.concat([self.outcome, df], ignore_index=True)
            sessionLog.print('Village '+str(v)+' complete!')
        #   end for villages -- end of simulations
    #   end def runSim
#   end experiment2
def main () :
    ####    EXPERIMENT 2
    # Experiment 2 A
    sessionLog.print('Experiment 2.A (Log-Normal) start')
    
    m = 0
    v = 1
    
    exp2a = experiment2(lambda : np.random.lognormal(mean=m, sigma=v, size=4))
    exp2a.runSimulation()
    
    
    
    # Experiment 2 B
    sessionLog.print('Experiment 2.B (Gamma) start')
    
    alpha = 1
    beta  = 1
    
    exp2b = experiment2(lambda : np.random.gamma(alpha,1/beta, size=4))
    exp2b.runSimulation()
    
    
    
    # Experiment 2 C
    sessionLog.print('Experiment 2.C (Dirichlet) start')
    
    exp2c = experiment2(lambda : np.random.dirichlet([1.0,1.0,1.0,1.0]))
    exp2c.runSimulation()
    
    
    
    
    
    ####    EXPERIMENT RESULTS  ####
    
    ####
    
    # Experiment 2 Result EDA
    sessionLog.print('Experiment run complete; starting evaluation of results.')
    
    wrm.post_exp_eda1(exp2a.outcome, 'Log-Normal (2)')
    exp2a.outcome[wrm.PARAMS] = (
        exp2a.outcome[wrm.PARAMS].div(exp2a.outcome[wrm.PARAMS].sum(axis=1), axis=0)
        )
    wrm.post_exp_eda1(exp2a.outcome, 'Log-Normal (2) (norm.)')
    
    
    
    wrm.post_exp_eda1(exp2b.outcome, 'Gamma (2)')
    exp2b.outcome[wrm.PARAMS] = (
        exp2b.outcome[wrm.PARAMS].div(exp2b.outcome[wrm.PARAMS].sum(axis=1), axis=0)
        )
    wrm.post_exp_eda1(exp2b.outcome, 'Gamma (2) (norm.)')
    
    
    
    wrm.post_exp_eda1(exp2c.outcome, 'Dirichlet (2)')
    exp2c.outcome[wrm.PARAMS] = (
        exp2c.outcome[wrm.PARAMS].div(exp2c.outcome[wrm.PARAMS].sum(axis=1), axis=0)
        )
    wrm.post_exp_eda1(exp2c.outcome, 'Dirichlet (2) (norm.)')
    
    ####
#   end 
# =========================================================================== #





# =================================== MAIN ================================== #
if __name__ == "__main__" :
    #   Initialize console log
    sessionLog = wrm.logger()
    sessionLog.print(__doc__)
    
    main()
    
#   endif
# =========================================================================== #




