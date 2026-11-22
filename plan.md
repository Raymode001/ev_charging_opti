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
  - driver may depart or arrive earlier or later than stated
  - use per car undertainty (budget uncertainty)
-  MPC: 
    - simple way, reoptimize whenever current setting changes: any new arrival or any departure that is outside of the uncertainty budget. 
    - driver changes inputs (charge level, departure time)
    

