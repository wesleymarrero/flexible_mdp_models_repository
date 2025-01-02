# ===================================================
# Estimating value functions using the SBBI algorithm
# ===================================================

# Loading modules
import numpy as np


# Simulation-based Backward Induction Algorithm
def sbbi(tp, healthy, Q_hat_next, Q_hat_rv, Q_hat70_next, discount, obs):

    # Extracting parameters
    sens_sc = len(tp) # number of sensitivity analysis scenarios (assuming best treatment in the future)
    numtrt = tp[0].shape[1]  # number of treatment choices

    # Initializing Q-value observations
    Q = np.full((numtrt, sens_sc + len(Q_hat_rv) + len(Q_hat_next) - 2), np.nan)

    # Generating state transitions
    np.random.seed(obs)  # seeds for pseudo-random number generator
    u = np.random.rand(len(tp)*numtrt, 1)  # generating len(tp)*numtrt uniform random numbers (additinal axis added for comparison with cummulative probabilities)
    prob = np.concatenate([tp[x].T for x in range(sens_sc)]) # combining transition probabilities of scenarios and actions in a single array (every 21 rows is a new scenario)
    prob = prob/prob.sum(axis=1)[:, np.newaxis] # making sure transition probabilities sum up to 1 (at each scenario, decision epoch, and action)
    cum_prob = prob.cumsum(axis=1)  # estimating cummulative probabilities
    h_next = (u < cum_prob).argmax(axis=1).reshape(len(tp), numtrt) # generating next states

    for sc in range(sens_sc): # risk and treatment benefit sensitivity analysis scenarios only (it saves computational time)
        for j in range(numtrt): # each treatment
            # Estimating Q-values (based on future action)
            if h_next[sc, j] == healthy: # only the healthy state has rewards associated
                if sc == 0: # base case risk and treatment benefit
                    # Base case
                    Q[j, sc] = 1 + discount*Q_hat_next[0]

                    # Normal rewards scenario
                    Q[j, sens_sc] = Q_hat_rv[1][0][obs] + discount*Q_hat_rv[1][1][obs]

                    # Random future action scenario
                    Q[j, sens_sc+1] = 1 + discount*Q_hat_next[1]

                    # Worst future action scenario
                    Q[j, sens_sc+2] = 1 + discount*Q_hat_next[2]

                elif sc == 1:  # ages 70-74 scenario
                    if not np.isnan(Q_hat70_next):  # only run if there is a patient with the current id
                        Q[j, sc] = 1 + discount*Q_hat70_next

                elif 2 <= sc < sens_sc: # risk and treatment benefit scenarios
                    Q[j, sc] = 1 + discount*Q_hat_next[0]

                else: # making sure all scenarios are taken into account
                    print("Scenario", sc, "was not taken into account")
                    Q[j, sc] = np.inf # indicator that scenario was not considered
            else: # no reward if patient is not healthy
                if sc == 0:
                    Q[j, sc], Q[j, sens_sc], Q[j, sens_sc+1], Q[j, sens_sc+2] = [0, 0, 0, 0]
                else:
                    Q[j, sc] = 0

    return Q

