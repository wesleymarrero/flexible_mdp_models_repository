# ================================
# Finite horizon policy evaluation
# ================================

# Loading modules
import numpy as np # array operations

#Finite horizon policy evaluation function
def evaluate_pi(trt, ptrans, healthy, Lterm, trtdisutility, discount):

    # Extrating parameters
    numhealth = ptrans.shape[0] # number of states
    years = ptrans.shape[1] # number of decision epochs

    # Array to store value function
    V_pi = np.full(years, np.nan) # value function following the policy being evaluated
    Q_hh = np.full((numhealth, years), np.nan) # stores intermediate value functions
    Q_hh[np.arange(numhealth) != healthy, ...] = 0  # there is no reward if patient is not healthy

    # Policy evaluation
    for t in reversed(range(years)): 
        # Computes value functions: uses the forward time format
        if t == max(range(years)):  # one decision remaining to be made in planning horizon
            Q_hh[healthy, t] = ptrans[healthy, t, trt[t]]*(1 + discount*Lterm)  # terminal condition (immediate reward is 1 year of perfect health)
        else:
            Q_hh[healthy, t] = ptrans[healthy, t, trt[t]]*(1 + discount*V_pi[t+1])  # backwards induction (immediate reward is 1 year of perfect health)
        Q_hh[np.arange(numhealth) != healthy, t] = 0  # there is no reward if patient is not healthy
        V_pi[t] = np.amax([0, np.sum(Q_hh[:, t])-trtdisutility[trt[t]]]) # subtract treatment disutility from expected qalys

    return V_pi

#Finite horizon policy evaluation in terms of events function
def evaluate_events(trt, ptrans, healthy, event_states):

    # Extrating parameters
    numhealth = ptrans.shape[0] # number of states
    years = ptrans.shape[1] # number of decision epochs

    # Array to store value function
    E_pi = np.full(years, np.nan) # stores expected number of events following the policy being evaluated
    E_time_pi = np.full(years, np.nan)  # stores expected time to adverse events following the policy being evaluated
    evt = np.full((numhealth, years), np.nan) # stores intermediate value functions

    # Policy evaluation
    for t in reversed(range(years)):
        # Computes value functions: uses the forward time format
        if t == max(range(years)):  # one decision remaining to be made in planning horizon
            for hh in range(numhealth):
                evt[hh, t] = ptrans[hh, t, trt[t]]*(event_states[hh]) # no events after the planning horizon
            E_time_pi[t] = 1/(1-ptrans[healthy, t, trt[t]]) # healthy state is a absorbing state in terminal year
        else:
            for hh in range(numhealth):
                evt[hh, t] = ptrans[hh, t, trt[t]]*(event_states[hh]+E_pi[t+1])   # backwards induction
            E_time_pi[t] = 1 + ptrans[healthy, t, trt[t]]*E_time_pi[t+1] # recurrence ralationship
        E_pi[t] = np.sum(evt[:, t]) # expected number of events

    return E_pi, E_time_pi


