# ==================================================================
# Calculating value functions and policies using backwards induction
# ==================================================================

# Loading modules
import numpy as np

# Backwards induction algorithm
def backwards_induction(ptrans, healthy, Lterm, trtdisutility, discount, feasibility):

    # #Line for debugging purposes
    # t=9; h=j=0; ptrans=transitions[0]; trtdisutility=trt_effects[-1]

    # Extracting parameters
    numhealth = ptrans.shape[0]  # number of states
    years = ptrans.shape[1]  # number of non-stationary stages
    numtrt = ptrans.shape[2]  # number of treatment choices

    # Storing MDP calculations
    Q_hh = np.full((numhealth, years, numtrt), np.nan)  # stores intermediate value functions for each trt
    Q_hh[np.arange(numhealth) != healthy, ...] = 0  # there is no reward if patient is not healthy
    Q = np.full((years, numtrt), np.nan) # stores Q-values
    V = np.full(years, np.nan) # stores optimal value function values
    policy = np.full(years, np.nan) # stores optimal treatment decisions

    for t in reversed(range(years)): # number of decisions remaining
        for j in range(numtrt): # each treatment
            # Computes value functions: uses the forward time format
            if t == max(range(years)):  # one decision remaining to be made in planning horizon
                Q_hh[healthy, t, j] = ptrans[healthy, t, j]*(1 + discount*Lterm)  # terminal condition (immediate reward is 1 year of perfect health)
            else:
                Q_hh[healthy, t, j] = ptrans[healthy, t, j]*(1 + discount*V[t+1])  # backwards induction (immediate reward is 1 year of perfect health)

            Q[t, j] = np.amax([0, np.sum(Q_hh[:, t, j])-trtdisutility[j]])  # add treatment disutility to expected qalys

        # Optimal value function and policies
        V[t] = np.amax(Q[t, feasibility[t]])  # maximized qalys over feasible treatments
        policy[t] = np.argmax(Q[t, feasibility[t]])  # optimal treatment

    return Q, V, policy.astype(int)
