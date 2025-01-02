# =========================================================
# Comparison of proposed algorithms to TD and MC methods
# =========================================================

# Note: this script does not run the proposed methods since the population is the same as in the main analysis
# executed in hypertension_treatment_sbbi_sbmcc.py. It executes on-policy Monte Carlo control, off-policy Monte Carlo control,
# Q-learning, Sarsa, and TD-SVP

# Loading modules
import os  # directory changes
import pandas as pd  # data frame operations
import numpy as np  # matrix operations
import time as tm  # timing code
import pickle as pk  # saving results
import gc # garbage collector
import multiprocessing as mp  # parallel computations
from transition_probabilities import TP  # transition probability calculations
from policy_evaluation import evaluate_pi # policy evaluation
from backwards_induction_mdp import backwards_induction # backward induction
import algorithms_mc_td as alg # SBBI, MC control, Sarsa, Q-learning, and TD-SVP

from ascvd_risk import arisk
from aha_2017_guideline import aha_guideline # 2017 AHA's guideline for hypertension treatment

# Importing parameters from main module
from hypertension_treatment_sbbi_sbmcc import home_dir, results_dir, fig_dir, rev_arisk, ptdata_list, \
    lifedata, strokedeathdata, chddeathdata, alldeathdata, years, numhealth, events, order, healthy, absorving, \
    sbpmin, sbpmax, dbpmin, sbp_reduction, dbp_reduction, rel_risk_chd, rel_risk_stroke, meds, disutility, \
    discount, numtrt, beta, alpha, targetrisk, targetsbp, targetdbp # reps, ctrl_reps,

# Selecting number of cores for parallel processing
cores = mp.cpu_count() - 1

# Running analysis on 50-54 population only
ptdata = ptdata_list[0] 
del ptdata_list; gc.collect()

# Time limits
## MC and TD
### Note: time limit was determined by running the SBBI algorithm using the number of observations to
# satisfy the conditions in Proposition 1 of the paper
limit = 30 # 30 seconds

## TD-SVP
### Note: time limit was determined using the same conditions as for the "limit" variable with 301 batches
limit_svp = 60*10 # 10 minutes (in simulations of 50 patients the maximum was 13.2 minutes with 20 cores)

# Parameters for TD-SVP
zeta = 0.02 # selected to match CI width of SBMCC using 300 batches
epsilon = 0.1 # exploration parameter (chosen arbritrarily, according to common practices)

# Objects to store results of patient simulation
# risk1, risk10, transitions = [[] for _ in range(3)]  # (save only for debugging purposes)
Q_bi, Q_sbbi, Q_on_mcc, Q_off_mcc, Q_sarsa, Q_q_learn, Q_td_svp, med_set_td_svp = [[] for _ in range(8)]
pt_sim = pd.DataFrame()

# Running simulation
time_list = []
id_seq_sm = range(np.ceil(len(ptdata.id.unique())).astype(int)) # sequence of records
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
        feas, tp = TP(ascvdrisk1, chddeath, strokedeath, alldeath, patientdata.sbp.values, patientdata.dbp.values,
                      sbpmin, sbpmax, dbpmin, sbp_reduction, dbp_reduction,
                      rel_risk_chd, rel_risk_stroke, numhealth)

        # Sorting transition probabilities to satisfy stochastic ordering with respect to states
        tp = tp[order, :, :]

        # Extracting list of feasible actions per state and decision epoch
        feasible = []  # stores index of feasible actions
        for t in range(feas.shape[0]):
            feasible.append(list(np.where(feas[t, :] == 1)[0]))
        del feas

        # Calculating action-value functions and policies using on-policy MC control
        time = tm.time()  # start time for algorithm
        Q_hat_on_mcc, pi_hat_on_mcc = alg.on_policy_mcc(tp, Lterm, disutility, healthy, absorving, feasible, discount, time, limit, epsilon=1) # executing algorithm
        pi_hat_on_mcc_gd = pi_hat_on_mcc.argmax(axis=1) # identifying greedy actions
        V_pi_on_mcc = evaluate_pi(pi_hat_on_mcc_gd, tp, healthy, Lterm, disutility, discount) # evaluating policy in true transition probabilities

        # Calculating action-value functions and policies using off-policy MC control
        time = tm.time()  # start time for algorithm
        Q_hat_off_mcc, pi_hat_off_mcc = alg.off_policy_mcc(tp, Lterm, disutility, healthy, absorving, feasible, discount, epsilon, time, limit)  # executing algorithm (assuming epsilon=0.1)
        V_pi_off_mcc = evaluate_pi(pi_hat_off_mcc, tp, healthy, Lterm, disutility, discount)  # evaluating policy in true transition probabilities

        # Calculating action-value functions and policies using Sarsa
        time = tm.time()  # start time for algorithm
        Q_hat_sarsa, pi_hat_sarsa = alg.sarsa(tp, Lterm, disutility, healthy, absorving, feasible, discount, time, limit, epsilon=1)  # executing algorithm
        pi_hat_sarsa_gd = pi_hat_sarsa.argmax(axis=1)  # identifying greedy actions
        V_pi_sarsa = evaluate_pi(pi_hat_sarsa_gd, tp, healthy, Lterm, disutility, discount)  # evaluating policy in true transition probabilities

        # Calculating action-value functions and policies using Q-learning
        time = tm.time()  # start time for algorithm
        Q_hat_q_learn, pi_hat_q_learn = alg.q_learning(tp, Lterm, disutility, healthy, absorving, feasible, discount, epsilon, time, limit)  # executing algorithm (assuming epsilon=0.1)
        V_hat_q_learn = np.max(Q_hat_q_learn, axis=1) # calculating estimates of value functions with Q-learning (for TD-SVP)
        V_pi_q_learn = evaluate_pi(pi_hat_q_learn, tp, healthy, Lterm, disutility, discount)  # evaluating policy in true transition probabilities

        # Calculating action-value functions and policies using TD-SVP
        time = tm.time()  # start time for algorithm
        Q_hat_td_svp, Pi_td_svp = alg.td_svp(V_hat_q_learn, zeta, tp, Lterm, disutility, healthy, absorving, feasible, discount, epsilon, time, limit_svp)
        Pi_td_svp_meds = pd.DataFrame(np.select([Pi_td_svp.isna()] + [Pi_td_svp == x for x in range(numtrt)], np.append(np.nan, meds))) # creating data frame of sets of medications
        
        ## Fewest number of medication in sets
        pi_fewest_td_svp = Pi_td_svp.min(axis=0) # identifying the fewest number of medications each year
        pi_fewest_td_svp = [pi_hat_q_learn[t] if np.isnan(pi_fewest_td_svp[t]) else pi_fewest_td_svp[t].astype(int) 
                            for t in range(pi_fewest_td_svp.shape[0])] # making sure that there are elements in each year's sets
        V_fewest_td_svp = evaluate_pi(pi_fewest_td_svp, tp, healthy, Lterm, disutility, discount)  # evaluating policy in true transition probabilities

        ## Median number of medications in sets
        pi_median_td_svp = np.ceil(Pi_td_svp.median(axis=0))  # identifying the fewest number of medications each year
        pi_median_td_svp = [pi_hat_q_learn[t] if np.isnan(pi_median_td_svp[t]) else pi_median_td_svp[t].astype(int)
                            for t in range(pi_median_td_svp.shape[0])]  # making sure that there are elements in each year's sets
        V_median_td_svp = evaluate_pi(pi_median_td_svp, tp, healthy, Lterm, disutility, discount)  # evaluating policy in true transition probabilities

        ## Best treatment in sets
        ### Identifying the best treatment in set each year or attaching action from Q-learning (if the set is empty)
        pi_best_td_svp = [] # list to store treatment strategy
        for t in range(years):
            elem = Pi_td_svp.iloc[:, t].dropna().astype(int)
            if len(elem) > 0:
                pi_best_td_svp.append(Pi_td_svp.iloc[np.argmax(Q_hat_td_svp[t, elem], axis=0), t].astype(int))
            else:
                pi_best_td_svp.append(pi_hat_q_learn[t])
        V_best_td_svp = evaluate_pi(pi_best_td_svp, tp, healthy, Lterm, disutility, discount)  # evaluating policy in true transition probabilities

        ## Data frame of results for a single patient (single result per patient-year)
        ptresults = pd.concat([pd.Series(np.repeat(i, years), name='id'),
                               pd.Series(np.arange(years), name='year'),

                               pd.Series(V_pi_on_mcc, name='V_pi_on_mcc'),  # patient's true value functions under on-policy MCC
                               pd.Series(V_pi_off_mcc, name='V_pi_off_mcc'),  # patient's true value functions under off-policy MCC
                               pd.Series(V_pi_sarsa, name='V_pi_sarsa'),  # patient's true value functions under Sarsa
                               pd.Series(V_pi_q_learn, name='V_pi_q_learn'),  # patient's true value functions under Q-learning
                               pd.Series(V_fewest_td_svp, name='V_fewest_td_svp'), # patient's true value functions using the fewest number of medications in TD-SVP
                               pd.Series(V_median_td_svp, name='V_median_td_svp'), # patient's true value functions using the median number of medications in TD-SVP
                               pd.Series(V_best_td_svp, name='V_best_td_svp'), # patient's true value functions using the best treatment in TD-SVP

                               pd.Series(pi_hat_on_mcc_gd, name='pi_hat_on_mcc'), # patient's policy according to on-policy MCC
                               pd.Series(pi_hat_off_mcc, name='pi_hat_off_mcc'), # patient's policy according to off-policy MCC
                               pd.Series(pi_hat_sarsa_gd, name='pi_hat_sarsa'), # patient's policy according to Sarsa
                               pd.Series(pi_hat_q_learn, name='pi_hat_q_learn'), # patient's policy according to Q-learning
                               pd.Series(pi_fewest_td_svp, name='pi_fewest_td_svp'), # patient's policy according using the fewest number of medications in TD-SVP
                               pd.Series(pi_median_td_svp, name='pi_median_td_svp'), # patient's policy according using the median number of medications in TD-SVP
                               pd.Series(pi_best_td_svp, name='pi_best_td_svp') # patient's policy according using the best treatment in TD-SVP

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

        # Saving patient-level results (for healthy state only)
        Q_on_mcc.append(Q_hat_on_mcc)  # patient's estimates of Q-values from on-policy MCC
        Q_off_mcc.append(Q_hat_off_mcc)  # patient's estimates of Q-values from off-policy MCC
        Q_sarsa.append(Q_hat_sarsa)  # patient's estimates of Q-values from Sarsa
        Q_q_learn.append(Q_hat_q_learn)  # patient's estimates of Q-values from Q-learning
        Q_td_svp.append(Q_hat_td_svp)  # patient's estimates of Q-values from Q-learning
        med_set_td_svp.append(Pi_td_svp_meds)  # patient's sets of near-optimal treatment choices using TD-SVP

        # Saving all results (saving each time a patient is evaluated)
        os.chdir(results_dir) # chaging to results directory
        with open('Algorithm Comparison Results - Records '+str(min(id_seq_sm))+' to '+str(max(id_seq_sm))+'.pkl', 'wb') as f:
            pk.dump([Q_on_mcc, Q_off_mcc, Q_sarsa, Q_q_learn, Q_td_svp, med_set_td_svp, pt_sim], f, protocol=3)
        os.chdir(home_dir)  # returning to home directory
