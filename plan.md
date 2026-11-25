# model predictive control with robust optimization basis:
- no future scenario estimation
- objective: daily peak load (max charging speed across station across 1 day)
- constraints: 
  - all cars need to be charged to required level before departure
  - one car per slot
  - max power any car can take
  - etc.
- decisions: 
  - station wide charging power per 15 min
- uncertainty: 
  - driver may depart earlier or later than stated
  - use per car uncertainty (budget uncertainty)
-  MPC: 
    - simple way, reoptimize whenever current setting changes: any new arrival or any departure that is outside of the uncertainty budget. 
    - driver changes inputs (charge level, departure time)
    

# model predictive control with adaptive optimization basis:
- here and now decisions: charging rates for the current time slot
- wait and see decisions: charging rates for future sessions
- Scenario characterization: 
  - True scenarios are infinite, we build an estimated proxy scenario set that is finite from past days. 
  - What the scenarios are: as an example, if today is tuesday, then the scenario is last tuesday with car arrival and departure times in the remaining part of that day
- first stage: optimize current charging fleet
- second stage: optimize over scenarios of rest of day, simple case could use equal probability. 

