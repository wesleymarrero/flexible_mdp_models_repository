# =========================================================
# Algorithms to obtain approximately optimal policies
# =========================================================

# Loading packages
import numpy as np
import time as tm
import pandas as pd

# ----------------------------
# Exact methods
# ----------------------------

# Finite horizon policy evaluation function
def evaluate_pi(trt, ptrans, healthy_list, Lterm, trtdisutility, discount):

    # Extrating parameters
    numhealth = ptrans.shape[0] # number of states
    years = ptrans.shape[1] # number of decision epochs
    diab_stat = ptrans.shape[3] # number of diabetes statuses

    # Array to store value function
    V_pi = np.full((years, diab_stat), np.nan) # value function following the policy being evaluated
    Q_hh = np.full((numhealth, years, diab_stat), np.nan) # stores intermediate value functions
    Q_hh[np.setdiff1d(np.arange(numhealth), healthy_list), ...] = 0  # there is no reward if patient is not healthy

    # Line for debugging purposes
    # trt = np.zeros((years, diab_stat), dtype=int); ptrans = tp.copy()

    # Policy evaluation
    for t in reversed(range(years)):
        for d in range(diab_stat):
            for h_ind, h in enumerate(healthy_list):
                # Computes value functions: uses the forward time format
                if t == max(range(years)):  # one decision remaining to be made in planning horizon
                    Q_hh[h, t, d] = ptrans[h, t, trt[t, d], d]*(1 + discount*Lterm)  # terminal condition (immediate reward is 1 year of perfect health)
                else:
                    Q_hh[h, t, d] = ptrans[h, t, trt[t, d], d]*(1 + discount*V_pi[t+1, h_ind])  # backwards induction (immediate reward is 1 year of perfect health)
            V_pi[t, d] = np.amax([0, np.sum(Q_hh[:, t, d])-trtdisutility[trt[t, d]]]) # subtract treatment disutility from expected qalys

    return V_pi

# Backwards induction algorithm
def backwards_induction(ptrans, healthy_list, Lterm, trtdisutility, discount, feasibility):

    # #Line for debugging purposes
    # t=9; h=j=0; ptrans=transitions[0]; trtdisutility=trt_effects[-1]

    # Extracting parameters
    numhealth = ptrans.shape[0]  # number of states
    years = ptrans.shape[1]  # number of non-stationary stages
    numtrt = ptrans.shape[2]  # number of treatment choices
    diab_stat = ptrans.shape[3]  # number of diabetes statuses

    # Storing MDP calculations
    Q_hh = np.full((numhealth, years, numtrt, diab_stat), np.nan)  # stores intermediate value functions for each trt
    Q_hh[np.setdiff1d(np.arange(numhealth), healthy_list), ...] = 0  # there is no reward if patient is not healthy
    Q = np.full((years, numtrt, diab_stat), np.nan) # stores Q-values
    V = np.full((years, diab_stat), np.nan) # stores optimal value function values
    policy = np.full((years, diab_stat), np.nan) # stores optimal treatment decisions

    for t in reversed(range(years)): # number of decisions remaining
        for d in range(diab_stat):
            for j in range(numtrt):  # each treatment
                for h_ind, h in enumerate(healthy_list):
                    # Computes value functions: uses the forward time format
                    if t == max(range(years)):  # one decision remaining to be made in planning horizon
                        Q_hh[h, t, j] = ptrans[h, t, j, d]*(1 + discount*Lterm)  # terminal condition (immediate reward is 1 year of perfect health)
                    else:
                        Q_hh[h, t, j] = ptrans[h, t, j, d]*(1 + discount*V[t+1, h_ind])  # backwards induction (immediate reward is 1 year of perfect health)

                Q[t, j, d] = np.amax([0, np.sum(Q_hh[:, t, j, d])-trtdisutility[j]])  # add treatment disutility to expected qalys

            # Optimal value function and policies
            V[t, d] = np.amax(Q[t, feasibility[t], d])  # maximized qalys over feasible treatments
            policy[t] = np.argmax(Q[t, feasibility[t]])  # optimal treatment

    return Q, V, policy.astype(int)

# ----------------------------
# SBBI + SBMCC
# ----------------------------

# Simulation-based Backward Induction Algorithm
def sbbi(tp_t, disutility, healthy_list, Q_hat_next, discount, obs): # , time, limit

    # while (tm.time()-time) < limit: # time condition does not seem to work well in parallel environment
    # Extracting parameters
    numtrt = tp_t.shape[1]  # number of treatment choices
    diab_stat = tp_t.shape[2]

    # Initializing Q-value observations
    Q = np.full((numtrt, diab_stat), np.nan)

    # Generating state transitions
    np.random.seed(obs)  # seeds for pseudo-random number generator
    u = np.random.rand(numtrt*diab_stat, 1)  # generating len(tp_t)*numtrt uniform random numbers (additinal axis added for comparison with cummulative probabilities)
    prob = np.concatenate([tp_t[..., x].T for x in range(diab_stat)])  # combining transition probabilities of scenarios and actions in a single array (every 21 rows is a new scenario)
    prob = prob/prob.sum(axis=1)[..., np.newaxis] # making sure transition probabilities sum up to 1 (at each decision epoch, and action)
    cum_prob = prob.cumsum(axis=1)  # estimating cummulative probabilities
    h_next = (u < cum_prob).argmax(axis=1).reshape(diab_stat, numtrt).T # generating next states

    for d in range(diab_stat): # each diabetes status
        for j in range(numtrt): # each treatment
            # Estimating Q-values (based on future action)
            if h_next[j, d] in healthy_list: # only the healthy states have rewards associated
                Q[j, d] = (1-disutility[j]) + discount*Q_hat_next[np.where(h_next[j, d] == healthy_list)[0][0]]
            else: # no reward if patient is not healthy
                Q[j, d] = 0

    return Q

# Simulation-based multiple comparison with a control algorithm
def sbmcc(Q_bar, Q_hat, sigma2_bar, a_ctrl, obs, rep):

    # Extracting parameters
    numtrt = Q_hat.shape[0]
    diab_stat = Q_hat.shape[1]

    # Arrays to store results
    psi = np.full((numtrt, diab_stat), np.nan)

    # Calculating root statistic
    for d in range(diab_stat): # each diabetes status
        for j in range(numtrt):  # each treatment
            psi[j, d] = (Q_bar[a_ctrl[d], d, rep] - Q_bar[j, d, rep] - (Q_hat[a_ctrl[d], d] - Q_hat[j, d])) / \
                     np.sqrt((sigma2_bar[j, d, rep] + sigma2_bar[a_ctrl[d], d, rep]) / obs)

    # Obtaining maximum psi for each diabetes status
    psi_max = np.amax(psi, axis=0)

    return psi_max

# ----------------------------
# MC methods
# ----------------------------

# On-policy first-visit MC control algorithm (assuming epsilon-greedy policy)
def on_policy_mcc(tp, Lterm, disutility, healthy_list, absorving, feasible, discount, time, limit, epsilon=1):

    # Extracting parameters
    S = tp.shape[0]  # number of states
    T = tp.shape[1]  # number of decision epochs
    A = tp.shape[2]  # number of actions
    diab_stat = tp.shape[3] # number of diabetes statuses

    # Initializing parameters
    M = np.zeros((T, A, diab_stat)) # matrix to store number of observations of each action at every decision epoch
    G = np.zeros((T, A, diab_stat)) # matrix to store returns of each action at every decision epoch
    Q_hat = np.zeros((T, A, diab_stat)) # initializing action-value functions (only for healthy states)
    pi = np.ones((T, A, diab_stat))*epsilon/A # assigning epsilon/A probability of selection to all actions
    greedy = [np.argmax(Q_hat[t, feasible[t], d], axis=0) for d in range(diab_stat) for t in range(T)]  # identifying greedy action in each state from feasible actions
    pi[np.tile(np.arange(T), diab_stat), greedy, np.repeat(np.arange(diab_stat), T)] += (1-epsilon)  # increasing the probability of selection of the best action in each state
    seed = 100 # initial seed for pseudo-random number generator
    n = 0  # initial episode counter

    while (tm.time()-time) < limit: # each episode

        np.random.seed(seed); seed += 1 # establishing seed of pseudo-random numbers (for reproducibility)
        s = [np.random.choice(healthy_list)] # assuming patients are healthy at the beginning of the planning horizon
        a = [] # list to store actions at current episode

        for t in range(T): # continue in episode until we reach end of episode (the length of the episodes are determined by the planning horizon)

            # Determining next state and selecting action using epsilon-greedy policy
            if s[-1] in healthy_list: # next state from transition probabilities (patient is healthy)
                np.random.seed(seed); seed += 1  # establishing seed
                h_ind = np.where(s[-1] == np.array(healthy_list))[0][0] # identifying index of healthy state
                a.append(np.random.choice(np.arange(pi.shape[1]), p=pi[t, :, h_ind]))  # selecting next action
                s_next = np.random.choice(np.arange(S), p=tp[:, t, a[-1], h_ind])  # sampling next state
            else: # next state has to be the absorig state (patient is not healthy)
                a.append(0) # no treatment is possible in absorving state
                s_next = absorving # patient remains in absorving state
            s.append(s_next) # appending next state to list of states

        # Initializing return with terminal rewards
        if s[-1] in healthy_list:
            g = Lterm # patients' healthy expected lifetime after planning horizon
        else:
            g = 0 # no reward after planning horizon

        for t in reversed(range(T)): # looping through episode backwards to upate action value functions
            # Increase return only if the patient transitions to a healthy state
            if s[t+1] in healthy_list:
                # Note: since states depend on time, state-action pairs can only be oberved once in each episode
                g = discount*g + (1 - disutility[a[t]]) # calculating return from time t onwards
            else:
                g = 0 # no reward if the patient does not transition to a healthy state

            # Updating only if patient is in a healthy state
            if s[t] in healthy_list:
                h_ind = np.where(s[t] == np.array(healthy_list))[0][0]  # identifying index of healthy state
                M[t, a[t], h_ind] += 1 # updating count of observations
                G[t, a[t], h_ind] += g # updating return matrix with return in the episode
                Q_hat[t, a[t], h_ind] = G[t, a[t], h_ind]/M[t, a[t], h_ind] # updating action-value function

        # Updating epsilon-greedy policy
        epsilon = 1/((n+1)+1)  # updating value of epsilon
        pi = np.ones((T, A, diab_stat))*epsilon/A # assigning epsilon/A probability of selection to all actions
        greedy = [np.argmax(Q_hat[t, feasible[t], d], axis=0) for d in range(diab_stat) for t in range(T)]  # identifying greedy action in each state from feasible actions
        pi[np.tile(np.arange(T), diab_stat), greedy, np.repeat(np.arange(diab_stat), T)] += (1-epsilon)  # increasing the probability of selection of the best action at every decision epoch and diabetes status
        n+=1 # increasing counter of episode number

    return Q_hat, pi

# Off-policy MC control algorithm (assuming epsilon-greedy behavior policy)
def off_policy_mcc(tp, Lterm, disutility, healthy_list, absorving, feasible, discount, epsilon, time, limit):

    # Extracting parameters
    S = tp.shape[0]  # number of states
    T = tp.shape[1]  # number of decision epochs
    A = tp.shape[2]  # number actions
    diab_stat = tp.shape[3]  # number of diabetes statuses

    # Initializing parameters
    C = np.zeros((T, A, diab_stat)) # matrix to store cummulative weights in each state-action pair at every diabetes status
    Q_hat = np.zeros((T, A, diab_stat))  # initializing action-value functions
    b = np.ones((T, A, diab_stat))*epsilon/A  # assigning epsilon/A probability of selection to all actions
    pi_list = [np.argmax(Q_hat[t, feasible[t], d], axis=0) for d in range(diab_stat) for t in range(T)]  # initial index of action that attains the maximum at each state from feasible actions
    b[np.tile(np.arange(T), diab_stat), pi_list, np.repeat(np.arange(diab_stat), T)] += (1-epsilon)  # increasing the probability of selection of the best action in each state
    pi = np.stack(pi_list).reshape(diab_stat, T).T # reshaping list of greedy actions to apropriate dimensions
    seed = 112  # initial seed for pseudo-random number generator

    while (tm.time()-time) < limit:  # each episode

        # Generating initial state
        s = [np.random.choice(healthy_list)]  # assuming patients are healthy at the beginning of the planning horizon
        a = []  # list to store actions at current episode

        for t in range(T): # continue in episode until we reach end of episode (the length of the episodes are determined by the planning horizon)

            # Determining next state and selecting action using epsilon-soft policy
            if s[-1] in healthy_list:  # next state from transition probabilities (patient is healthy)
                np.random.seed(seed); seed += 1  # establishing seed
                h_ind = np.where(s[-1] == np.array(healthy_list))[0][0]  # identifying index of healthy state
                a.append(np.random.choice(np.arange(b.shape[1]), p=b[t, :, h_ind]))  # selecting next action
                s_next = np.random.choice(np.arange(S), p=tp[:, t, a[-1], h_ind])  # sampling next state
            else:  # next state has to be the absorig state (patient is not healthy)
                a.append(0)  # no treatment is possible in absorving state
                s_next = absorving  # patient remains in absorving state
            s.append(s_next)  # appending next state to list of states

        # Initializing return with terminal rewards
        if s[-1] in healthy_list:
            g = Lterm  # patients' healthy expected lifetime after planning horizon
        else:
            g = 0  # no reward after planning horizon
        W = 1  # initial weight
        for t in reversed(range(T)): # looping through episode backwards to upate action value functions
            # Increase return only if the patient transitions to a healthy state
            if s[t+1] in healthy_list:
                # Note: since states depend on time, state-action pairs can only be oberved once in each episode
                g = discount*g + (1 - disutility[a[t]])  # calculating return from time t onwards
            else:
                g = 0  # no reward if the patient does not transition to a healthy state

            # Updating only if patient is in a healthy state
            if s[t] in healthy_list:
                h_ind = np.where(s[t] == np.array(healthy_list))[0][0]  # identifying index of healthy state
                C[t, a[t], h_ind] += W  # updating cummulative weight
                Q_hat[t, a[t], h_ind] += (W/C[t, a[t], h_ind])*(g - Q_hat[t, a[t], h_ind]) # updating action-value function
                pi[t, :] = np.argmax(Q_hat[t, feasible[t], :], axis=0) # updating greedy actions from feasible actions

                # Updating epsilon-greedy policy
                b[t, :, h_ind] = np.ones(A)*epsilon/A  # assigning epsilon/A probability of selection to all actions
                b[t, pi[t, h_ind], h_ind] += (1-epsilon)  # increasing the probability of selection of the best action in current state

                if pi[t, h_ind] == a[t]: # check if current action is equal to the greedy action
                    W *= (1/b[t, a[t], h_ind]) # updating weight
                    if W == 0:  # exiting episode if there is no more weight updating
                        break
                else:
                    break # exit the episode

    return Q_hat, pi

# ----------------------------
# TD methods
# ----------------------------

# Sarsa algorithm (assuming 1/n step-size and epsilon-greedy policy)
def sarsa(tp, Lterm, disutility, healthy_list, absorving, feasible, discount, time, limit, epsilon=1):

    # Extracting parameters
    S = tp.shape[0]  # number of states
    T = tp.shape[1]  # number of decision epochs
    A = tp.shape[2]  # number of actions
    diab_stat = tp.shape[3]  # number of diabetes statuses

    # Initializing parameters
    Q_hat = np.zeros((T, A, diab_stat)) # initializing action-value functions
    pi = np.ones((T, A, diab_stat))*epsilon/A # assigning epsilon/A probability of selection to all actions
    greedy = [np.argmax(Q_hat[t, feasible[t], d], axis=0) for d in range(diab_stat) for t in range(T)]  # identifying greedy action in each state from feasible actions
    pi[np.tile(np.arange(T), diab_stat), greedy, np.repeat(np.arange(diab_stat), T)] += (1-epsilon)  # increasing the probability of selection of the best action in each state
    N_sa = np.zeros((T, A, diab_stat))  # matrix to store number of observations in each state and action pair (for step-size)
    seed = 100 # initial seed for pseudo-random number generator
    n = 0 # initial episode counter

    while (tm.time()-time) < limit: # each episode

        # Selecting current action according to policy of interest
        s_now = np.random.choice(healthy_list)  # assuming patients are healthy at the beginning of the planning horizon
        np.random.seed(seed); seed += 1  # establishing seed
        h_ind = np.where(s_now == np.array(healthy_list))[0][0]  # identifying index of healthy state
        a_now = np.random.choice(np.arange(A), p=pi[0, :, h_ind])  # selecting next action

        for t in range(T): # continue in episode until we reach end of episode (the length of the episodes are determined by the planning horizon)

            # Determining next state and selecting next action according to policy of interest
            if s_now in healthy_list:  # next state from transition probabilities (patient is healthy)
                np.random.seed(seed); seed += 1  # establishing seed
                h_ind = np.where(s_now == np.array(healthy_list))[0][0]  # identifying index of healthy state
                s_next = np.random.choice(np.arange(S), p=tp[:, t, a_now, h_ind])  # sampling next state
                if t == max(range(T)): # no action is needed after the planning horizon
                    a_next = np.nan
                else: # selects next action according to epsion-greedy policy
                    a_next = np.random.choice(np.arange(A), p=pi[t+1, :, h_ind])  # selecting next action
            else:  # next state has to be the absorig state (patient is not healthy)
                s_next = absorving # patient remains in absorving state
                a_next = 0 # no treatment is possible in absorving state

            # Updating only if patient is in a healthy state
            if s_now in healthy_list:
                # Updating estimate of action-value function
                h_ind = np.where(s_now == np.array(healthy_list))[0][0]  # identifying index of healthy state
                N_sa[t, a_now, h_ind] += 1; alpha = 1/N_sa[t, a_now, h_ind] # establishing step-size parameter (using the Harmonic series as the step-size)

                ## Receive reward only if the patient transitions to a healthy state
                if s_next in healthy_list:
                    if t == max(range(T)):
                        Q_hat[t, a_now, h_ind] += alpha*(((1 - disutility[a_now])+discount*Lterm)-Q_hat[t, a_now, h_ind])  # updating action-value function with terminal reward
                    else:
                        Q_hat[t, a_now, h_ind] += alpha*(((1 - disutility[a_now])+discount*Q_hat[t+1, a_next, h_ind]) - Q_hat[t, a_now, h_ind]) # updating action-value function
                else:
                    Q_hat[t, a_now, h_ind] += alpha*(0-Q_hat[t, a_now, h_ind])  # updating action-value function

                # Updating epsilon-greedy policy
                epsilon = 1/((n+1)+1) # updating value of epsilon
                pi[t, :, h_ind] = np.ones(A)*epsilon/A # assigning epsilon/A probability of selection to all actions
                greedy = np.argmax(Q_hat[t, feasible[t], h_ind], axis=0) # identifying greedy action in each state from feasible actions
                pi[t, greedy, h_ind] += (1-epsilon) # increasing the probability of selection of the best action in current state

            # Updating current state and action
            s_now = s_next; a_now = a_next

        # Increasing iteration number
        n += 1

    return Q_hat, pi

# Q-learning algorithm (assuming 1/n step-size and epsilon-greedy behavior policy)
def q_learning(tp, Lterm, disutility, healthy_list, absorving, feasible, discount, epsilon, time, limit):

    # Extracting parameters
    S = tp.shape[0]  # number of states
    T = tp.shape[1]  # number of decision epochs
    A = tp.shape[2]  # number of actions
    diab_stat = tp.shape[3]  # number of diabetes statuses

    # Initializing parameters
    Q_hat = np.zeros((T, A, diab_stat)) # initializing action-value functions
    b = np.ones((T, A, diab_stat))*epsilon/A # assigning epsilon/A probability of selection to all actions
    greedy = [np.argmax(Q_hat[t, feasible[t], d], axis=0) for d in range(diab_stat) for t in range(T)]  # initial index of action that attains the maximum at each state from feasible actions
    b[np.tile(np.arange(T), diab_stat), greedy, np.repeat(np.arange(diab_stat), T)] += (1-epsilon)  # increasing the probability of selection of the best action in each state
    N_sa = np.zeros((T, A, diab_stat))  # matrix to store number of observations in each state and action pair (for step-size)
    seed = 100 # initial seed for pseudo-random number generator

    while (tm.time()-time) < limit: # each episode

        # Generating initial state
        s_now = np.random.choice(healthy_list)  # assuming patients are healthy at the beginning of the planning horizon

        for t in range(T): # continue in episode until we reach end of episode (the length of the episodes are determined by the planning horizon)

            # Selecting current action according to epsilon-greedy policy, determining next state, and selecting next action to greedy policy
            if s_now in healthy_list:  # next state from transition probabilities (patient is healthy)
                np.random.seed(seed); seed += 1  # establishing seed
                h_ind = np.where(s_now == np.array(healthy_list))[0][0]  # identifying index of healthy state
                a_now = np.random.choice(np.arange(b.shape[1]), p=b[t, :, h_ind])  # selecting next action according to behavior policy
                s_next = np.random.choice(np.arange(S), p=tp[:, t, a_now, h_ind])  # sampling next state
                if t == max(range(T)): # no action is needed after the planning horizon
                    a_next = np.nan
                else: # selecting next action according to greedy policy
                    a_next = np.argmax(Q_hat[t+1, :, h_ind]) # selecting next action
            else:  # next state has to be the absorig state (patient is not healthy)
                a_now = 0 # no treatment is possible in absorving state
                s_next = absorving # patient remains in absorving state
                a_next = 0 # no treatment is possible in absorving state

            # Updating only if patient is in a healthy state
            if s_now in healthy_list:
                # Updating estimate of action-value function
                h_ind = np.where(s_now == np.array(healthy_list))[0][0]  # identifying index of healthy state
                N_sa[t, a_now, h_ind] += 1; alpha = 1/N_sa[t, a_now, h_ind] # establishing step-size parameter (using the Harmonic series as the step-size)
                ## Receive reward only if the patient transitions to a healthy state
                if s_next in healthy_list:
                    if t == max(range(T)):
                        Q_hat[t, a_now, h_ind] += alpha*(((1 - disutility[a_now])+discount*Lterm)-Q_hat[t, a_now, h_ind])  # updating action-value function with terminal reward
                    else:
                        Q_hat[t, a_now, h_ind] += alpha*(((1 - disutility[a_now])+discount*Q_hat[t+1, a_next, h_ind])-Q_hat[t, a_now, h_ind])  # updating action-value function
                else:
                    Q_hat[t, a_now, h_ind] += alpha*(0-Q_hat[t, a_now, h_ind])  # updating action-value function
    
                # Updating epsilon-greedy policy
                b[t, :, h_ind] = np.ones(A)*epsilon/A # assigning epsilon/A probability of selection to all actions
                greedy = np.argmax(Q_hat[t, feasible[t], h_ind], axis=0) # identifying greedy action in current state
                b[t, greedy, h_ind] += (1-epsilon) # increasing the probability of selection of the best action in current state

            # Updating current state
            s_now = s_next

    # Identifying approximately optimal policy
    pi = np.stack([np.argmax(Q_hat[t, feasible[t], :], axis=0) for t in range(len(feasible))])  # identifying greedy action in each state from feasible actions

    return Q_hat, pi

# TD-SVP algorithm (assuming 1/n step-size and epsilon-greedy behavior policy)
def td_svp(V, zeta, tp, Lterm, disutility, healthy_list, absorving, feasible, discount, epsilon, time, limit):

    # Extracting parameters
    S = tp.shape[0]  # number of states
    T = tp.shape[1]  # number of decision epochs
    A = tp.shape[2]  # number of actions
    diab_stat = tp.shape[3]  # number of diabetes statuses

    # Initializing parameters
    Q_hat = np.zeros((T, A, diab_stat)) # initializing action-value functions
    b = np.ones((T, A, diab_stat))*epsilon/A # assigning epsilon/A probability of selection to all actions
    greedy = [np.argmax(Q_hat[t, feasible[t], d], axis=0) for d in range(diab_stat) for t in range(T)]  # initial index of action that attains the maximum at each state from feasible actions
    b[np.tile(np.arange(T), diab_stat), greedy, np.repeat(np.arange(diab_stat), T)] += (1-epsilon)  # increasing the probability of selection of the best action in each state
    pi = [[[] for _ in range(T)] for _ in range(diab_stat)] # initializing sets of near-optimal actions
    N_sa = np.zeros((T, A, diab_stat))  # matrix to store number of observations in each state and action pair (for step-size)
    seed = 100 # initial seed for pseudo-random number generator

    # Line for debugging purposes
    # time = tm.time(); limit = 5; V = V_hat_q_learn

    while (tm.time()-time) < limit: # each episode

        # Generating initial state and initial set of near optimal actions
        s_now = np.random.choice(healthy_list)  # assuming patients are healthy at the beginning of the planning horizon
        h_ind = np.where(s_now == np.array(healthy_list))[0][0]  # identifying index of healthy state
        pi[h_ind][0] = np.where(Q_hat[0, :, h_ind] >= (1 - zeta) * V[0, h_ind])[0]  # identifying sets of near-optimal actions
        pi[h_ind][0] = [a for a in pi[h_ind][0] if a in feasible[0]]  # including only feasible actions

        for t in range(T): # continue in episode until we reach end of episode (the length of the episodes are determined by the planning horizon)

            # Selecting current action according to epsilon-greedy policy, determining next state, and selecting next action to greedy policy
            if s_now in healthy_list:  # next state from transition probabilities (patient is healthy)
                np.random.seed(seed); seed += 1  # establishing seed
                h_ind = np.where(s_now == np.array(healthy_list))[0][0]  # identifying index of healthy state
                a_now = np.random.choice(np.arange(b.shape[1]), p=b[t, :, h_ind])  # selecting next action according to behavior policy
                s_next = np.random.choice(np.arange(S), p=tp[:, t, a_now, h_ind])  # sampling next state

                if t == max(range(T)): # no action is needed after the planning horizon
                    a_next = np.nan
                else: # selecting next action according to TD-SVP method
                    pi[h_ind][t + 1] = np.where(Q_hat[t + 1, :, h_ind] >= (1 - zeta) * V[t + 1, h_ind])[0] # identifying sets of near-optimal actions
                    pi[h_ind][t + 1] = [a for a in pi[h_ind][t + 1] if a in feasible[t + 1]]  # including only feasible actions
                    if len(pi[h_ind][t + 1]) > 0: # set of near-optimal actions is not empty
                        a_next = pi[h_ind][t + 1][Q_hat[t+1, pi[h_ind][t + 1], h_ind].argmin()] # selecting next action
                    else: # set is empty
                        a_next = np.argmax(Q_hat[t+1, :, h_ind]) # selecting next action
            else:  # next state has to be the absorig state (patient is not healthy)
                a_now = 0 # no treatment is possible in absorving state
                s_next = absorving # patient remains in absorving state
                a_next = 0 # no treatment is possible in absorving state

            # Updating only if patient is in a healthy state
            if s_now in healthy_list:
                # Updating estimate of action-value function
                h_ind = np.where(s_now == np.array(healthy_list))[0][0]  # identifying index of healthy state
                N_sa[t, a_now, h_ind] += 1; alpha = 1/N_sa[t, a_now, h_ind] # establishing step-size parameter (using the Harmonic series as the step-size)
                ## Receive reward only if the patient transitions to a healthy state
                if s_next in healthy_list:
                    if t == max(range(T)):
                        Q_hat[t, a_now, h_ind] += alpha*(((1-disutility[a_now])+discount*Lterm)-Q_hat[t, a_now, h_ind])  # updating action-value function with terminal reward
                    else:
                        Q_hat[t, a_now, h_ind] += alpha*(((1-disutility[a_now])+discount*Q_hat[t+1, a_next, h_ind])-Q_hat[t, a_now, h_ind])  # updating action-value function
                else:
                    Q_hat[t, a_now, h_ind] += alpha*(0-Q_hat[t, a_now, h_ind])  # updating action-value function

                # Updating epsilon-greedy policy
                b[t, :, h_ind] = np.ones(A)*epsilon/A  # assigning epsilon/A probability of selection to all actions
                greedy = np.argmax(Q_hat[t, feasible[t], h_ind], axis=0)  # identifying greedy action in current state
                b[t, greedy, h_ind] += (1-epsilon)  # increasing the probability of selection of the best action in current state

            # Updating current state
            s_now = s_next

    # Converting set value policies to data frame
    pi = [pd.DataFrame(pi[d]).T for d in range(diab_stat)]

    return Q_hat, pi
