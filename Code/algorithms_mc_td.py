# =========================================================
# Algorithms to obtain approximately optimal policies
# =========================================================

# Loading packages
import numpy as np
import time as tm
import pandas as pd

# ----------------------------
# SBBI + SBMM
# ----------------------------

# Simulation-based Backward Induction Algorithm
def sbbi(tp_t, disutility, healthy, Q_hat_next, discount, obs): # , time, limit

    # Extracting parameters
    numtrt = tp_t.shape[1]  # number of treatment choices

    # Initializing Q-value observations
    Q = np.full(numtrt, np.nan)

    # Generating state transitions
    np.random.seed(obs)  # seeds for pseudo-random number generator
    u = np.random.rand(numtrt, 1)  # generating len(tp_t)*numtrt uniform random numbers (additinal axis added for comparison with cummulative probabilities)
    prob = tp_t.T # transposing transition probabilities (for newaxis operation below)
    prob = prob/prob.sum(axis=1)[:, np.newaxis] # making sure transition probabilities sum up to 1 (at each decision epoch, and action)
    cum_prob = prob.cumsum(axis=1)  # estimating cummulative probabilities
    h_next = (u < cum_prob).argmax(axis=1).reshape(numtrt) # generating next states

    for j in range(numtrt): # each treatment
        # Estimating Q-values (based on future action)
        if h_next[j] == healthy: # only the healthy state has rewards associated
            Q[j] = (1-disutility[j]) + discount*Q_hat_next
        else: # no reward if patient is not healthy
            Q[j] = 0

    return Q

# Simulation-based multiple comparison with a control algorithm
def sbmcc(Q_bar, Q_hat, sigma2_bar, a_ctrl, obs, rep):

    # Extracting parameters
    numtrt = Q_hat.shape[0]

    # Arrays to store results
    psi = np.full(numtrt, np.nan)

    # Calculating root statistic
    for j in range(numtrt):  # each treatment
        psi[j] = (Q_bar[a_ctrl, rep] - Q_bar[j, rep] - (Q_hat[a_ctrl] - Q_hat[j])) / \
                 np.sqrt((sigma2_bar[j, rep] + sigma2_bar[a_ctrl, rep]) / obs)

    # Obtaining maximum psi
    psi_max = np.amax(psi)

    return psi_max

# ----------------------------
# MC methods
# ----------------------------

# On-policy first-visit MC control algorithm (assuming epsilon-greedy policy)
def on_policy_mcc(tp, Lterm, disutility, healthy, absorving, feasible, discount, time, limit, epsilon=1):

    # Extracting parameters
    S = tp.shape[0]  # number of states
    T = tp.shape[1]  # number of decision epochs
    A = tp.shape[2]  # number of actions

    # Initializing parameters
    M = np.zeros((T, A)) # matrix to store number of observations of each action at every decision epoch
    G = np.zeros((T, A)) # matrix to store returns of each action at every decision epoch
    Q_hat = np.zeros((T, A)) # initializing action-value functions (only for healthy states)
    pi = np.ones((T, A))*epsilon/A # assigning epsilon/A probability of selection to all actions
    greedy = [np.argmax(Q_hat[t, feasible[t]], axis=0) for t in range(len(feasible))]  # identifying greedy action in each state from feasible actions
    pi[np.arange(pi.shape[0]), greedy] += (1-epsilon)  # increasing the probability of selection of the best action in each state
    seed = 100 # initial seed for pseudo-random number generator
    n = 0  # initial episode counter

    while (tm.time()-time) < limit: # each episode

        np.random.seed(seed); seed += 1 # establishing seed of pseudo-random numbers (for reproducibility)
        s = [healthy] # assuming patients are healthy at the beginning of the planning horizon
        a = [] # list to store actions at current episode

        for t in range(T): # continue in episode until we reach end of episode (the length of the episodes are determined by the planning horizon)

            # Determining next state and selecting action using epsilon-greedy policy
            if s[-1] == healthy: # next state from transition probabilities (patient is healthy)
                np.random.seed(seed); seed += 1  # establishing seed
                a.append(np.random.choice(np.arange(pi.shape[1]), p=pi[t, :]))  # selecting next action
                s_next = np.random.choice(np.arange(S), p=tp[:, t, a[-1]])  # sampling next state
            else: # next state has to be the absorig state (patient is not healthy)
                a.append(0) # no treatment is possible in absorving state
                s_next = absorving # patient remains in absorving state
            s.append(s_next) # appending next state to list of states

        # Initializing return with terminal rewards
        if s[-1] == healthy:
            g = Lterm  # patients' healthy expected lifetime after planning horizon
        else:
            g = 0 # no reward after planning horizon

        for t in reversed(range(T)): # looping through episode backwards to upate action value functions
            # Increase return only if the patient transitions to a healthy state
            if s[t+1]==healthy:
                # Note: since states depend on time, state-action pairs can only be oberved once in each episode
                g = discount*g + (1 - disutility[a[t]]) # calculating return from time t onwards
            else:
                g = 0 # no reward if the patient does not transition to a healthy state
            M[t, a[t]] += 1 # updating count of observations
            G[t, a[t]] += g # updating return matrix with return in the episode
            Q_hat[t, a[t]] = G[t, a[t]]/M[t, a[t]] # updating action-value function
        epsilon = 1/((n+1)+1)  # updating value of epsilon
        pi = np.ones((T, A))*epsilon/A # assigning epsilon/A probability of selection to all actions
        greedy = [np.argmax(Q_hat[t, feasible[t]], axis=0) for t in range(len(feasible))] # identifying greedy action in each state from feasible actions
        pi[np.arange(len(pi)), greedy] += (1-epsilon)  # increasing the probability of selection of the best action at every decision epoch
        n+=1 # increasing counter of episode number

    return Q_hat, pi

# Off-policy MC control algorithm (assuming epsilon-greedy behavior policy)
def off_policy_mcc(tp, Lterm, disutility, healthy, absorving, feasible, discount, epsilon, time, limit):

    # Extracting parameters
    S = tp.shape[0]  # number of states
    T = tp.shape[1]  # number of decision epochs
    A = tp.shape[2]  # number actions

    # Initializing parameters
    C = np.zeros((T, A)) # matrix to store cummulative weights in each state-action pair
    Q_hat = np.zeros((T, A))  # initializing action-value functions
    b = np.ones((T, A))*epsilon/A  # assigning epsilon/A probability of selection to all actions
    pi = [np.argmax(Q_hat[t, feasible[t]], axis=0) for t in range(len(feasible))]  # initial index of action that attains the maximum at each state from feasible actions
    b[np.arange(b.shape[0]), pi] += (1-epsilon)  # increasing the probability of selection of the best action in each state
    seed = 112  # initial seed for pseudo-random number generator

    while (tm.time()-time) < limit:  # each episode

        # Generating initial state
        s = [healthy]  # assuming patients are healthy at the beginning of the planning horizon
        a = []  # list to store actions at current episode

        for t in range(T): # continue in episode until we reach end of episode (the length of the episodes are determined by the planning horizon)

            # Determining next state and selecting action using epsilon-soft policy
            if s[-1] == healthy:  # next state from transition probabilities (patient is healthy)
                np.random.seed(seed); seed += 1  # establishing seed
                a.append(np.random.choice(np.arange(b.shape[1]), p=b[t, :]))  # selecting next action
                s_next = np.random.choice(np.arange(S), p=tp[:, t, a[-1]])  # sampling next state
            else:  # next state has to be the absorig state (patient is not healthy)
                a.append(0)  # no treatment is possible in absorving state
                s_next = absorving  # patient remains in absorving state
            s.append(s_next)  # appending next state to list of states

        # Initializing return with terminal rewards
        if s[-1] == healthy:
            g = Lterm  # patients' healthy expected lifetime after planning horizon
        else:
            g = 0  # no reward after planning horizon
        W = 1  # initial weight
        for t in reversed(range(T)): # looping through episode backwards to upate action value functions
            # Updating only if the patient transitions to a healthy state
            if s[t+1] == healthy:
                # Note: since states depend on time, state-action pairs can only be oberved once in each episode
                g = discount*g + (1 - disutility[a[t]])  # calculating return from time t onwards
            else:
                g = 0  # no reward if the patient does not transition to a healthy state
            C[t, a[t]] += W  # updating cummulative weight
            Q_hat[t, a[t]] += (W/C[t, a[t]])*(g - Q_hat[t, a[t]]) # updating action-value function
            pi[t] = np.argmax(Q_hat[t, feasible[t]], axis=0) # updating greedy actions from feasible actions

            # Updating epsilon-greedy policy
            b[t, :] = np.ones(A)*epsilon/A  # assigning epsilon/A probability of selection to all actions
            b[t, pi[t]] += (1-epsilon)  # increasing the probability of selection of the best action in current state

            if pi[t] == a[t]: # check if current action is equal to the greedy action
                W *= (1/b[t, a[t]]) # updating weight
                if W == 0:  # exiting episode if there is no more weight updating
                    break
            else:
                break # exit the episode

    return Q_hat, pi

# ----------------------------
# TD methods
# ----------------------------

# Sarsa algorithm (assuming 1/n step-size and epsilon-greedy policy)
def sarsa(tp, Lterm, disutility, healthy, absorving, feasible, discount, time, limit, epsilon=1):

    # Extracting parameters
    S = tp.shape[0]  # number of states
    T = tp.shape[1]  # number of decision epochs
    A = tp.shape[2]  # number of actions

    # Initializing parameters
    Q_hat = np.zeros((T, A)) # initializing action-value functions
    pi = np.ones((T, A))*epsilon/A # assigning epsilon/A probability of selection to all actions
    greedy = [np.argmax(Q_hat[t, feasible[t]], axis=0) for t in range(len(feasible))] # identifying greedy action in each decision epoch
    pi[np.arange(pi.shape[0]), greedy] += (1-epsilon) # increasing the probability of selection of the best action in each state
    N_sa = np.zeros((T, A))  # matrix to store number of observations in each state and action pair (for step-size)
    seed = 100 # initial seed for pseudo-random number generator
    n = 0 # initial episode counter

    while (tm.time()-time) < limit: # each episode

        # Selecting current action according to policy of interest
        s_now = healthy  # assuming patients are healthy at the beginning of the planning horizon
        np.random.seed(seed); seed += 1  # establishing seed
        a_now = np.random.choice(np.arange(A), p=pi[0, :])  # selecting next action

        for t in range(T): # continue in episode until we reach end of episode (the length of the episodes are determined by the planning horizon)

            # Determining next state and selecting next action according to policy of interest
            if s_now == healthy:  # next state from transition probabilities (patient is healthy)
                np.random.seed(seed); seed += 1  # establishing seed
                s_next = np.random.choice(np.arange(S), p=tp[:, t, a_now])  # sampling next state
                if t == max(range(T)): # no action is needed after the planning horizon
                    a_next = np.nan
                else: # selects next action according to epsion-greedy policy
                    a_next = np.random.choice(np.arange(A), p=pi[t+1, :])  # selecting next action
            else:  # next state has to be the absorig state (patient is not healthy)
                s_next = absorving # patient remains in absorving state
                a_next = 0 # no treatment is possible in absorving state

            # Updating estimate of action-value function
            N_sa[t, a_now] += 1; alpha = 1/N_sa[t, a_now] # establishing step-size parameter (using the Harmonic series as the step-size)

            ## Receive reward only if the patient transitions to a healthy state
            if s_next == healthy:
                if t == max(range(T)):
                    Q_hat[t, a_now] += alpha*(((1 - disutility[a_now])+discount*Lterm)-Q_hat[t, a_now])  # updating action-value function with terminal reward
                else:
                    Q_hat[t, a_now] += alpha*(((1 - disutility[a_now])+discount*Q_hat[t+1, a_next]) - Q_hat[t, a_now]) # updating action-value function
            else:
                Q_hat[t, a_now] += alpha*(0-Q_hat[t, a_now])  # updating action-value function

            # Updating epsilon-greedy policy
            epsilon = 1/((n+1)+1) # updating value of epsilon
            pi[t, :] = np.ones(A)*epsilon/A # assigning epsilon/A probability of selection to all actions
            greedy = np.argmax(Q_hat[t, feasible[t]], axis=0) # identifying greedy action in each state from feasible actions
            pi[t, greedy] += (1-epsilon) # increasing the probability of selection of the best action in current state

            # Updating current state and action
            s_now = s_next; a_now = a_next

        # Increasing iteration number
        n += 1

    return Q_hat, pi

# Q-learning algorithm (assuming 1/n step-size and epsilon-greedy behavior policy)
def q_learning(tp, Lterm, disutility, healthy, absorving, feasible, discount, epsilon, time, limit):

    # Extracting parameters
    S = tp.shape[0]  # number of states
    T = tp.shape[1]  # number of decision epochs
    A = tp.shape[2]  # number of actions

    # Initializing parameters
    Q_hat = np.zeros((T, A)) # initializing action-value functions
    b = np.ones((T, A))*epsilon/A # assigning epsilon/A probability of selection to all actions
    greedy = [np.argmax(Q_hat[t, feasible[t]], axis=0) for t in range(len(feasible))]  # identifying greedy action in each state from feasible actions
    b[np.arange(b.shape[0]), greedy] += (1-epsilon) # increasing the probability of selection of the best action in each state
    N_sa = np.zeros((T, A))  # matrix to store number of observations in each state and action pair (for step-size)
    seed = 100 # initial seed for pseudo-random number generator

    while (tm.time()-time) < limit: # each episode

        # Generating initial state
        s_now = healthy  # assuming patients are healthy at the beginning of the planning horizon

        for t in range(T): # continue in episode until we reach end of episode (the length of the episodes are determined by the planning horizon)

            # Selecting current action according to epsilon-greedy policy, determining next state, and selecting next action to greedy policy
            if s_now == healthy:  # next state from transition probabilities (patient is healthy)
                np.random.seed(seed); seed += 1  # establishing seed
                a_now = np.random.choice(np.arange(b.shape[1]), p=b[t, :])  # selecting next action according to behavior policy
                s_next = np.random.choice(np.arange(S), p=tp[:, t, a_now])  # sampling next state
                if t == max(range(T)): # no action is needed after the planning horizon
                    a_next = np.nan
                else: # selecting next action according to greedy policy
                    a_next = np.argmax(Q_hat[t+1, :]) # selecting next action
            else:  # next state has to be the absorig state (patient is not healthy)
                a_now = 0 # no treatment is possible in absorving state
                s_next = absorving # patient remains in absorving state
                a_next = 0 # no treatment is possible in absorving state

            # Updating estimate of action-value function
            N_sa[t, a_now] += 1; alpha = 1/N_sa[t, a_now] # establishing step-size parameter (using the Harmonic series as the step-size)
            ## Receive reward only if the patient transitions to a healthy state
            if s_next == healthy:
                if t == max(range(T)):
                    Q_hat[t, a_now] += alpha*(((1 - disutility[a_now])+discount*Lterm)-Q_hat[t, a_now])  # updating action-value function with terminal reward
                else:
                    Q_hat[t, a_now] += alpha*(((1 - disutility[a_now])+discount*Q_hat[t+1, a_next])-Q_hat[t, a_now])  # updating action-value function
            else:
                Q_hat[t, a_now] += alpha*(0-Q_hat[t, a_now])  # updating action-value function

            # Updating epsilon-greedy policy
            b[t, :] = np.ones(A)*epsilon/A # assigning epsilon/A probability of selection to all actions
            greedy = np.argmax(Q_hat[t, feasible[t]], axis=0) # identifying greedy action in current state
            b[t, greedy] += (1-epsilon) # increasing the probability of selection of the best action in current state

            # Updating current state
            s_now = s_next

    # Identifying approximately optimal policy
    pi = [np.argmax(Q_hat[t, feasible[t]], axis=0) for t in range(len(feasible))]  # identifying greedy action in each state from feasible actions

    return Q_hat, pi

# TD-SVP algorithm (assuming 1/n step-size and epsilon-greedy behavior policy)
def td_svp(V, zeta, tp, Lterm, disutility, healthy, absorving, feasible, discount, epsilon, time, limit):

    # Extracting parameters
    S = tp.shape[0]  # number of states
    T = tp.shape[1]  # number of decision epochs
    A = tp.shape[2]  # number of actions

    # Initializing parameters
    Q_hat = np.zeros((T, A)) # initializing action-value functions
    b = np.ones((T, A))*epsilon/A # assigning epsilon/A probability of selection to all actions
    greedy = [np.argmax(Q_hat[t, feasible[t]], axis=0) for t in range(len(feasible))]  # identifying greedy action in each state from feasible actions
    b[np.arange(b.shape[0]), greedy] += (1-epsilon) # increasing the probability of selection of the best action in each state
    pi = [[] for _ in range(T)] # initializing sets of near-optimal actions
    N_sa = np.zeros((T, A))  # matrix to store number of observations in each state and action pair (for step-size)
    seed = 100 # initial seed for pseudo-random number generator

    # Line for debugging purposes
    # time = tm.time(); limit = 60*2; V = V_hat_q_learn; zeta = 0.5

    while (tm.time()-time) < limit: # each episode

        # Generating initial state and initial set of near optimal actions
        s_now = healthy  # assuming patients are healthy at the beginning of the planning horizon
        pi[0] = np.where(Q_hat[0, :] >= (1 - zeta) * V[0])[0]  # identifying sets of near-optimal actions
        pi[0] = [a for a in pi[0] if a in feasible[0]] # including only feasible actions

        for t in range(T): # continue in episode until we reach end of episode (the length of the episodes are determined by the planning horizon)

            # Selecting current action according to epsilon-greedy policy, determining next state, and selecting next action to greedy policy
            if s_now == healthy:  # next state from transition probabilities (patient is healthy)
                np.random.seed(seed); seed += 1  # establishing seed
                a_now = np.random.choice(np.arange(b.shape[1]), p=b[t, :])  # selecting next action according to behavior policy
                s_next = np.random.choice(np.arange(S), p=tp[:, t, a_now])  # sampling next state

                if t == max(range(T)): # no action is needed after the planning horizon
                    a_next = np.nan
                else: # selecting next action according to TD-SVP method
                    pi[t + 1] = np.where(Q_hat[t + 1, :] >= (1 - zeta) * V[t + 1])[0] # identifying sets of near-optimal actions
                    pi[t + 1] = [a for a in pi[t + 1] if a in feasible[t + 1]]  # including only feasible actions
                    if len(pi[t+1]) > 0: # set of near-optimal actions is not empty
                        a_next = pi[t+1][Q_hat[t+1, pi[t+1]].argmin()] # selecting next action
                    else: # set is empty
                        a_next = np.argmax(Q_hat[t+1, :]) # selecting next action
            else:  # next state has to be the absorig state (patient is not healthy)
                a_now = 0 # no treatment is possible in absorving state
                s_next = absorving # patient remains in absorving state
                a_next = 0 # no treatment is possible in absorving state

            # Updating estimate of action-value function
            N_sa[t, a_now] += 1; alpha = 1/N_sa[t, a_now] # establishing step-size parameter (using the Harmonic series as the step-size)
            ## Receive reward only if the patient transitions to a healthy state
            if s_next == healthy:
                if t == max(range(T)):
                    Q_hat[t, a_now] += alpha*(((1 - disutility[a_now])+discount*Lterm)-Q_hat[t, a_now])  # updating action-value function with terminal reward
                else:
                    Q_hat[t, a_now] += alpha*(((1 - disutility[a_now])+discount*Q_hat[t+1, a_next])-Q_hat[t, a_now])  # updating action-value function
            else:
                Q_hat[t, a_now] += alpha*(0-Q_hat[t, a_now])  # updating action-value function

            # Updating epsilon-greedy policy
            b[t, :] = np.ones(A)*epsilon/A # assigning epsilon/A probability of selection to all actions
            greedy = np.argmax(Q_hat[t, feasible[t]]) # identifying greedy action in current state from feasible actions
            b[t, greedy] += (1-epsilon) # increasing the probability of selection of the best action in current state

            # Updating current state
            s_now = s_next

    # Converting set value policies to data frame
    pi = pd.DataFrame(pi).T

    return Q_hat, pi
