# Reinforcement Learning for the vehicle routing problem with a highly variable customer basis and stochastic demands

This project is part of my PhD project. Please refer to the full paper [here](Off-line approximate dynamic programming for the vehicle routing problem with a highly variable customer basis and stochastic demands). 
In this study, we present a novel stochastic vehicle routing problem and propose a Reinforcement Learning algorithm to solve it.
Accordingly, a set of stochastic customers submit their service request at the beginning of each day 
and the operator must route vehicles (with a limited capacity) to service customers such that
the total collected demands is maximized. We call this problem, 
the VRPVCSD, short for Vehicle Routing Problem with a highly Variable Customer basis and Stochastic Demands.

To solve this problem, we formulate it as a Markov Decision Process and propose a Reinforcement Learning algorithm.
In particular, to overcome the curses of dimensionality, we propose a fixed-size, hand-engineered, observation function to replace 
with the variable-size large state of the system.

The proposed solution method can solve VRPVCSD and VRPSD (Vehicle Routing Problem with Stochastic Demands). 
Instances for both problems are provided in the folder Instances.
This code requires the following python libraries:
- tensorflow 1.14
- numpy 1.18.2
- scipy.spatial.distance 1.5.4

The following is the list of input arguments:
```python
[--model ['VRPVCSD', 'VRPSD']] # routing problem
[--operation ['train', 'test']]
[--c int] # number of customers
[--v int] # number of vehicles
[--q int] # capacity of the vehicle
[--dl float] # duration limit
[--sv int] # stochastic variability 
[--density int] # density class
[--trials int] # number of trials to train
[--base_address 'address'] # an address to store the model 
[--nb int] # the size of target customers 
[--code 'code'] # model name to load and test
[--obs [0, 1]] # whether to use the observation function or not
```

The following code trains the model for 100K trials 
on a small instance with customer density of 0 and vehicle capacity of 25:
```python
python main.py --operation train --density 0 --q 25 --obs 0 --trials 100000 --base_address Models/State/VRPVCSD/
```

