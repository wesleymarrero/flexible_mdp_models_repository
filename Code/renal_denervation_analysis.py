# ===================================================================================
# Evaluation of Adding Renal Sympathetic Denervation at the First Year of the Study
# ==================================================================================

# Loading modules
import os  # directory changes
import pandas as pd  # data frame operations
import numpy as np  # matrix operations
import time as tm  # timing code
import pickle as pk  # saving results
import gc # garbage collector
import multiprocessing as mp  # parallel computations
from transition_probabilities import TP_RSN  # transition probability calculations
from policy_evaluation import evaluate_pi # policy evaluation
from backwards_induction_mdp import backwards_induction # backward induction
import algorithms_mc_td as alg # SBBI, MC control, Sarsa, Q-learning, and TD-SVP

# Importing parameters from main module
from hypertension_treatment_sbbi_sbmcc import home_dir, results_dir, rev_arisk, ptdata_list, \
    lifedata, strokedeathdata, chddeathdata, alldeathdata, years, numhealth, events, order, healthy, \
    sbpmin, sbpmax, dbpmin, sbp_reduction, dbp_reduction, rel_risk_chd, rel_risk_stroke, meds, disutility, \
    discount, numtrt, sbp_drop_rsn, dbp_drop_rsn, rel_risk_chd_rsn, rel_risk_stroke_rsn, disutility_rsn, beta, \
    reps, ctrl_reps

# Selecting number of cores for parallel processing
cores = mp.cpu_count() - 1

# Running analysis on 50-54 population only
ptdata = ptdata_list[0] 
del ptdata_list; gc.collect()

# Objects to store results of patient simulation
Q_sbbi, var_sbbi, feasibility_list, obs_list = [[] for _ in range(4)]
pt_sim = pd.DataFrame()

# Adjusting parameters with the possibility of renal sympathetic denervation
meds = np.tile(meds, 2) # adjusting the vector of number of medications
disutility = np.tile(disutility, 2) # expanding disutility vector
disutility[numtrt:] = disutility[numtrt:]+disutility_rsn # adding RSN disutility to appropriate actions

# Running simulation
reps = reps - ctrl_reps # subtracting replications used to generate controls in main analysis
# (reps was used to get the same total number of observations as in the main analysis to get comparable variance estimates)
id_seq_sm = range(ptdata.id.unique().shape[0]) # all patients

if __name__ == '__main__':
    for i in id_seq_sm:

        # Keeping track of progress
        print(tm.asctime(tm.localtime())[:-5], "Evaluating patient", i)

        # Extracting patient's data from larger data matrix
        patientdata = ptdata[ptdata.id == i]

        # life expectancy and death likelihood data index
        if patientdata.sex.iloc[0] == 0:  # male
            sexcol = 1  # column in deathdata corresponding to male
        else:
            sexcol = 2  # column in deathdata corresponding to female

        # Death rates
        chddeath = chddeathdata.iloc[list(np.where([j in patientdata.age.values
                                                    for j in list(chddeathdata.iloc[:, 0])])[0]), sexcol]
        strokedeath = strokedeathdata.iloc[list(np.where([j in patientdata.age.values
                                                          for j in list(strokedeathdata.iloc[:, 0])])[0]), sexcol]
        alldeath = alldeathdata.iloc[list(np.where([j in patientdata.age.values
                                                    for j in list(alldeathdata.iloc[:, 0])])[0]), sexcol]

        # Estimating terminal conditions
        Lterm = lifedata.iloc[np.where(patientdata.age.iloc[max(range(years))] == lifedata.iloc[:, 0])[0][0], sexcol] # healthy life expectancy for males and females of the patient's current age

        ## Storing risk calculations
        ascvdrisk1 = np.full((years, events), np.nan)  # 1-year CHD and stroke risk (for transition probabilities)

        ## Calculating risk for healthy state only (before ordering of states)
        for t in range(years): # each age
            for k in range(events): # each event type

                # 1-year ASCVD risk calculation (for transition probabilities)
                ascvdrisk1[t, k] = rev_arisk(k, patientdata.sex.iloc[t], patientdata.black.iloc[t], patientdata.age.iloc[t],
                                             patientdata.sbp.iloc[t], patientdata.smk.iloc[t], patientdata.tc.iloc[t],
                                             patientdata.hdl.iloc[t], patientdata.diab.iloc[t], 0, 1)

        # Calculating transition probabilities
        feas, tp = TP_RSN(ascvdrisk1, chddeath, strokedeath, alldeath, patientdata.sbp.values, patientdata.dbp.values,
                          sbpmin, sbpmax, dbpmin, sbp_reduction, dbp_reduction, rel_risk_chd, rel_risk_stroke,
                          sbp_drop_rsn, dbp_drop_rsn, rel_risk_chd_rsn, rel_risk_stroke_rsn, numhealth)

        # Sorting transition probabilities to satisfy stochastic ordering with respect to states
        tp = tp[order, :, :]

        # Extracting list of feasible actions per state and decision epoch
        feasible = []  # stores index of feasible actions
        for t in range(feas.shape[0]):
            feasible.append(list(np.where(feas[t, :] == 1)[0]))
        del feas

        # Evaluating no treatment in true transition probabilities
        V_no_trt = evaluate_pi(np.zeros(years, dtype=int), tp, healthy, Lterm, disutility, discount)

        # Calculating policies using backwards induction
        Q, V, pi = backwards_induction(tp, healthy, Lterm, disutility, discount, feasible)

        # Calculating action-value functions and policies using SBBI and SBBI + SBMCC
        ## Calculating necessary observations for a beta confidence level
        obs = np.ceil(2*(((np.sum(discount**np.arange(years))+(discount**9)*Lterm)-0)**2)*np.log(len(meds)/beta)).astype(int)

        ## Initializing objects to store results
        ### SBBI
        Q_hat_sbbi = np.full((years, len(meds)), np.nan)  # initializing overall estimate of Q-values
        pi_hat_sbbi = np.full(years, -999, dtype='int')  # initializing approximate optimal actions (using invalid action index)
        sigma2_hat = np.full((years, len(meds)), np.nan)  # initializing estimate of the variance of the average Q-values per simulation replicate

        ## Executing algorithms
        for t in reversed(range(years)): # number of decisions remaining
            # Q-value at next period
            if t == max(range(years)):
                Q_hat_next = Lterm # expected lifetime as terminal reward
            else:
                Q_hat_next = Q_hat_sbbi[t+1, pi_hat_sbbi[t+1]] # Q-value associated with the approximately optimal action

            # Extracting transition probabilities for the current decision epoch and excluding RSN actions in years 2 through 10
            if t == 0: # initial year
                tp_t = tp[:, t, :]
            else: # remaining years
                tp_t = tp[:, t, :numtrt]

            # Running Simulation-based backwards induction algorithm (only for healthy state)
            ## Running health trajectories in parallel (only for healthy state)
            with mp.Pool(cores) as pool:  # creating pool of parallel workers
                Q_sim = pool.starmap_async(alg.sbbi, [(tp_t, disutility, healthy, Q_hat_next, discount, r)
                                                      for r in range(obs*reps)]).get()

            ## For a single replication
            ### Converting results into array (with appropriate dimensions)
            Q_sim = np.array(Q_sim)

            ### Calculating estimates of Q-values and their variances per replication
            if t == 0: # initial year
                Q_hat_sbbi[t, ...] = np.nanmean(Q_sim, axis=0)  # estimated Q-value
                sigma2_hat[t, ...] = np.nanvar(Q_sim, axis=0, ddof=1)  # estimated variance per replication
            else: # remaining years
                Q_hat_sbbi[t, :numtrt] = np.nanmean(Q_sim, axis=0)  # estimated Q-value
                sigma2_hat[t, :numtrt] = np.nanvar(Q_sim, axis=0, ddof=1)  # estimated variance per replication
            pi_hat_sbbi[t] = np.argmax(Q_hat_sbbi[t, feasible[t]], axis=0)  # approximately optimal policies

        # Evaluating policy from a single replication of SBBI in true transition probabilities
        V_pi_sbbi = evaluate_pi(pi_hat_sbbi, tp, healthy, Lterm, disutility, discount)

        ## Data frame of results for a single patient (single result per patient-year)
        ptresults = pd.concat([pd.Series(np.repeat(i, years), name='id'),
                               pd.Series(np.arange(years), name='year'),

                               pd.Series(V_no_trt, name='V_notrt'), # patient's true value functions for no treatment
                               pd.Series(V, name='V_opt'), # patient's true optimal value functions
                               pd.Series(V_pi_sbbi, name='V_pi_sbbi'), # patient's true value functions under SBBI

                               pd.Series(pi, name='pi_opt'), # patient's true optimal policy
                               pd.Series(pi_hat_sbbi, name='pi_hat_sbbi'), # patient's policy according to SBBI

                               ], axis=1)

        # Extracting sampling weights from original dataset
        wts = pd.DataFrame(ptdata.iloc[list(np.where([j in list(ptresults.id.unique()) for j in list(ptdata.id)])[0]),
                                       [ptdata.columns.get_loc(col) for col in ["id", "wt"]]], columns=["id", "wt"])
        wts.reset_index(drop=True, inplace=True)

        # Adding sampling weights to final results
        ptresults = pd.concat(
            [ptresults.iloc[:, [ptresults.columns.get_loc(col) for col in ["id", "year"]]], wts["wt"],
             ptresults.iloc[:, [ptresults.columns.get_loc(col)
                                    for col in ptresults.columns.difference(["id", "year"])]]],
            axis=1)

        # Merging single patient data in data frame with data from of all patients
        pt_sim = pd.concat([pt_sim, ptresults], ignore_index=True)
        ptresults = np.nan  # making sure values are not recycled

        # Saving patient-level results (for healthy states only)
        Q_sbbi.append(Q_hat_sbbi)  # patient's estimates of Q-values from SBBI
        var_sbbi.append(sigma2_hat)  # patient's estimates of Q-value variances
        feasibility_list.append(feasible) # indicators of feasible actions
        obs_list.append(obs) # number of observations considered

        # Saving all results (saving each time a patient is evaluated)
        os.chdir(results_dir) # chaging to results directory
        with open('Renal Sympathetic Denervation Results at Year 1 - Records '+str(min(id_seq_sm))+' to '+str(max(id_seq_sm))+'.pkl', 'wb') as f:
            pk.dump([feasibility_list, obs_list, Q_sbbi, var_sbbi, pt_sim], f, protocol=3)
        os.chdir(home_dir)  # returning to home directory
