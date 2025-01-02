# ===========================================================================
# Comparison of proposed algorithms with TD and MC methods on MDP variations
# ===========================================================================

# Loading modules
import os  # directory changes
import pandas as pd  # data frame operations
import numpy as np  # matrix operations
import time as tm  # timing code
import pickle as pk  # saving results
import gc # garbage collector
import multiprocessing as mp  # parallel computations
from transition_probabilities import TP, TP_RED, TP_DIAB, TP_DRUG, TP_RED_DRUG, TP_DIAB_DRUG  # transition probability calculations
from diab_risk import drisk # 1-year type-2 disbetes risk
import algorithms_mdp_variations as alg # policy evaluation, backward induction, SBBI, MC control, Sarsa, Q-learning, and TD-SVP

# Importing parameters from main module
from hypertension_treatment_sbbi_sbmcc import home_dir, data_dir, results_dir, fig_dir, rev_arisk, \
    lifedata, strokedeathdata, chddeathdata, alldeathdata, riskslopedata, numhealth, events, absorving, \
    sbpmin, sbpmax, dbpmin, sbp_reduction, dbp_reduction, rel_risk_chd, rel_risk_stroke, std_index, disutility, \
    alldrugs, disutility_dt, discount, alpha, beta, reps, ctrl_reps

# Selecting number of cores for parallel processing
cores = mp.cpu_count() - 1

# Loading 2009-2016 Continuous NHANES data (imputed with random forests and forecasted with linear regression in R)
os.chdir(data_dir+'\\Continuous NHANES')
ptdata = pd.read_csv('Continuous NHANES 50-54 Dataset Until 73.csv') # ages 50-54 dataset forecasted until age 73

## Adding indicator of Black race in data frame (for revised risk calculations)
ptdata['black'] = np.where(ptdata.race==1, 0, 1)

# Parameters for TD-SVP
zeta = 0.02 # selected to match CI width of SBMCC using 300 batches
epsilon = 0.1 # exploration parameter (chosen arbritrarily, according to common practices)

# Simulation parameters
## Generating arrays of scenario indicators for 3 state space sizes, 3 action space sizes, and 3 planning horizon sizes
scenarios = np.array(np.meshgrid(np.arange(3), np.arange(3), np.arange(3))).T.reshape(-1, 3)[:, [2, 0, 1]] # scenario indicator (first index state, second action, third horizon)
state_action_scenarios = np.array(np.meshgrid(np.arange(3), np.arange(3))).T.reshape(-1, 2) # scenario indicator for states and actions (excluding horizon)
mdp_var = scenarios.shape[0] # 27 MDP variations, 3 state space sizes, 3 action space sizes, and 3 planning horizon sizes

## State parameters
healthy_ind = [0, 6] # indexes of healthy states (initial diabetes status and diabetic)

# Action parameters
disutility_base = disutility.copy() # base disutility (changing name for scenarios)
disutility_red = disutility[std_index] # disutility for reduced action space
del disutility

# SBMCC parameters
reps = reps-ctrl_reps # excluding control replications

# Objects to store results of patient simulation
# risk1, risk10, transitions = [[[] for _ in range(mdp_var)] for _ in range(3)]  # (save only for debugging purposes)
Q_bi, Q_sbbi, Q_on_mcc, Q_off_mcc, Q_sarsa, Q_q_learn, Q_td_svp, action_set_sbbi_sbmcc, action_set_td_svp = [[[] for _ in range(mdp_var)] for _ in range(9)]
action_set_sbbi_sbmcc, action_set_td_svp = [[[] for _ in range(mdp_var)] for _ in range(2)]
pt_sim_base, pt_sim_diab = [[pd.DataFrame() for _ in range(mdp_var)] for _ in range(2)]

# Running simulation on first 11 records - approximately 1% of total population
os.chdir(home_dir) # starting from home directory
id_seq_sm = range(6) # first half of records
# id_seq_sm = range(6, 11) # second half of records
if __name__ == '__main__':
    for i in id_seq_sm:

        # Keeping track of progress
        print(tm.asctime(tm.localtime())[:-5], "Evaluating patient", i)

        # Extracting patient's data from larger data matrix
        patientdata = ptdata[ptdata.id == i]

        # Extracting number of years considered in patient
        var_years = patientdata.shape[0]

        # Life expectancy and death likelihood data index
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

        # Risk slopes (for BP reductions using parameters in Law et al. 2003 and 2009)
        riskslope = riskslopedata.iloc[list(np.where([j in patientdata.age.values
                                                      for j in list(riskslopedata.iloc[:, 0])])[0]), 1:3]

        # Estimating terminal conditions for different planning horizons
        lterm_list = list() # generating list of terminal conditions
        lterm_list.append(lifedata.iloc[np.where(patientdata.age.iloc[4] ==
                                                 lifedata.iloc[:, 0])[0][0], sexcol])  # healthy life expectancy after 5 years
        lterm_list.append(lifedata.iloc[np.where(patientdata.age.iloc[9] ==
                                                 lifedata.iloc[:, 0])[0][0], sexcol])  # healthy life expectancy after 10 years
        lterm_list.append(lifedata.iloc[np.where(patientdata.age.max() ==
                                                 lifedata.iloc[:, 0])[0][0], sexcol]) # healthy life expectancy after age 73

        ## Storing risk calculations
        ascvdrisk1 = np.full((var_years, events, 2), np.nan)  # 1-year CHD and stroke risk with initial patient data and diabetes (for transition probabilities)
        diab_risk = np.full((var_years, 2), np.nan)  # 1-year risk for type-2 diabetes with and without BP treatment (for transition probabilities)

        ## Calculating risk for healthy state only (before ordering of states)
        for t in range(var_years): # each age
            for k in range(events): # each event type

                # 1-year ASCVD risk calculation (for transition probabilities)
                ascvdrisk1[t, k, 0] = rev_arisk(k, patientdata.sex.iloc[t], patientdata.black.iloc[t], patientdata.age.iloc[t],
                                                patientdata.sbp.iloc[t], patientdata.smk.iloc[t], patientdata.tc.iloc[t],
                                                patientdata.hdl.iloc[t], patientdata.diab.iloc[t], 0, 1) # original diabetes status
                ascvdrisk1[t, k, 1] = rev_arisk(k, patientdata.sex.iloc[t], patientdata.black.iloc[t],
                                                patientdata.age.iloc[t], patientdata.sbp.iloc[t], patientdata.smk.iloc[t],
                                                patientdata.tc.iloc[t], patientdata.hdl.iloc[t], 1, 0, 1)  # with diabetes

            # 1-year risk for type-2 diabetes
            diab_risk[t, 0] = drisk(patientdata.age.iloc[t], patientdata.bmi.iloc[t], patientdata.sex.iloc[t],
                                    patientdata.waist.iloc[t], 0, patientdata.diab.iloc[t]) # no BP treatment
            diab_risk[t, 1] = drisk(patientdata.age.iloc[t], patientdata.bmi.iloc[t], patientdata.sex.iloc[t],
                                    patientdata.waist.iloc[t], 1, patientdata.diab.iloc[t])  # BP treatment

        # Calculating transition probabilities
        # Note: results are stored in two separate lists for state and action space scenarios and ordered according to size of the scenario
        ## Base case (6 states and 21 treatment choices)
        feas_base, tp_base = TP(ascvdrisk1[..., 0], chddeath, strokedeath, alldeath, patientdata.sbp.values, patientdata.dbp.values,
                                sbpmin, sbpmax, dbpmin, sbp_reduction, dbp_reduction,
                                rel_risk_chd, rel_risk_stroke, numhealth)
        tp_base = tp_base[..., np.newaxis] # adding dimension to match diabetes status scenario

        ## State space variations
        ### Reduced state space (3 states and 21 treatment choices)
        feas_red, tp_red = TP_RED(ascvdrisk1[..., 0], chddeath, strokedeath, alldeath, patientdata.sbp.values, patientdata.dbp.values,
                                  sbpmin, sbpmax, dbpmin, sbp_reduction, dbp_reduction,
                                  rel_risk_chd, rel_risk_stroke, int(numhealth/2))
        tp_red = tp_red[..., np.newaxis]  # adding dimension to match diabetes status scenario

        ### Expanded state space with diabetes status (12 states and 21 treatment choices)
        feas_diab, tp_diab = TP_DIAB(ascvdrisk1, diab_risk, chddeath, strokedeath, alldeath, patientdata.sbp.values,
                                     patientdata.dbp.values, sbpmin, sbpmax, dbpmin, sbp_reduction, dbp_reduction,
                                     rel_risk_chd, rel_risk_stroke, int(numhealth*2))

        ## Action space variations
        ### Reduced action space to standard doses only (6 states and 6 treatment choices)
        feas_std, tp_std = [feas_base[:, std_index], tp_base[..., std_index, :]]

        ### Expanded action space including drug types (6 states and 196 treatment choices)
        feas_drug, tp_drug = TP_DRUG(ascvdrisk1[..., 0], chddeath, strokedeath, alldeath, riskslope, patientdata.sbp.values,
                                     patientdata.dbp.values, sbpmin, sbpmax, dbpmin, alldrugs, numhealth)
        tp_drug = tp_drug[..., np.newaxis]  # adding dimension to match diabetes status scenario

        ## Combined variation scenarios
        ### Reduced state and action spaces (3 states and 6 treatment choices)
        feas_red_std, tp_red_std = [feas_red[:, std_index], tp_red[..., std_index, :]]

        ### Expanded state space and reduced action space (12 states and 6 treatment choices)
        feas_diab_std, tp_diab_std = [feas_diab[:, std_index], tp_diab[..., std_index, :]]

        ### Reduced state space and expanded action space (3 state and 196 treatment choices)
        feas_red_drug, tp_red_drug = TP_RED_DRUG(ascvdrisk1[..., 0], chddeath, strokedeath, alldeath, riskslope,
                                                 patientdata.sbp.values, patientdata.dbp.values, sbpmin, sbpmax, dbpmin, alldrugs, int(numhealth/2))
        tp_red_drug = tp_red_drug[..., np.newaxis]  # adding dimension to match diabetes status scenario

        ### Expanded state and action spaces (12 states and 196 treatment choices)
        feas_diab_drug, tp_diab_drug = TP_DIAB_DRUG(ascvdrisk1, diab_risk, chddeath, strokedeath, alldeath, riskslope,
                                                    patientdata.sbp.values, patientdata.dbp.values, sbpmin, sbpmax, dbpmin,
                                                    alldrugs, int(numhealth*2))

        # Storing transition probabilites in list using order of MDP variations in scenarios
        tp_list = [tp_base, tp_std, tp_drug, tp_red, tp_red_std, tp_red_drug, tp_diab, tp_diab_std, tp_diab_drug]
        del tp_base, tp_std, tp_drug, tp_red, tp_red_std, tp_red_drug, tp_diab, tp_diab_std, tp_diab_drug; gc.collect()

        # Extracting list of feasible actions per state and decision epoch
        feasible_list = [[] for _ in range(9)]  # stores index of feasible actions in MDP variations using order in scenarios
        for t in range(feas_base.shape[0]):
            feasible_list[0].append(list(np.where(feas_base[t, :] == 1)[0])) # base case
            feasible_list[1].append(list(np.where(feas_std[t, :] == 1)[0])) # base states and reduced actions
            feasible_list[2].append(list(np.where(feas_drug[t, :] == 1)[0])) # base states and expanded actions
            feasible_list[3].append(list(np.where(feas_red[t, :] == 1)[0])) # reduced states and base actions
            feasible_list[4].append(list(np.where(feas_red_std[t, :] == 1)[0])) # reduced states and reduced actions
            feasible_list[5].append(list(np.where(feas_red_drug[t, :] == 1)[0])) # reduced states and expanded actions
            feasible_list[6].append(list(np.where(feas_diab[t, :] == 1)[0])) # expanded states and base actions
            feasible_list[7].append(list(np.where(feas_diab_std[t, :] == 1)[0]))  # expanded states and reduced actions
            feasible_list[8].append(list(np.where(feas_diab_drug[t, :] == 1)[0])) # expanded states and expanded actions
            
        del feas_base, feas_std, feas_drug, feas_red, feas_red_std, feas_red_drug, feas_diab, feas_diab_std, feas_diab_drug; gc.collect()

        ptresults_base = [pd.DataFrame() for _ in range(mdp_var)]  # initializing list of dataframes to store patient-level results with initial diabetes status
        ptresults_diab = [pd.DataFrame() for _ in range(mdp_var)]  # initializing list of dataframes to store patient-level results with diabetes
        for sc in range(mdp_var):

            # Keeping track of scenarios
            print(tm.asctime(tm.localtime())[:-5], "Evaluating scenario", sc, "in patient", i)

            # State and action space size selection
            ind = np.where((scenarios[sc][0] == state_action_scenarios[:, 0]) & (scenarios[sc][1] == state_action_scenarios[:, 1]))[0][0] # index of state-action space scenario
            tp = tp_list[ind]; feasible = feasible_list[ind]

            # Disutility selection
            if scenarios[sc][1] == 2: # expanded action space
                disutility = disutility_dt.copy()
            elif scenarios[sc][1] == 1: # reduced action space
                disutility = disutility_red.copy()
            else: # base action space
                disutility = disutility_base

            # Horizon size selection
            if scenarios[sc][2] == 1: # reduced horizon
                Lterm = lterm_list[0] # terminal condition
                years = 5 # 5-year planning horizon
                tp = tp[:, :years, ...]; feasible = feasible[:years]  # subsetting transition probabilities and feasibility indicators
            elif scenarios[sc][2] == 2: # expanded horizon
                Lterm = lterm_list[0] # terminal condition
                years = var_years # planning horizon until age 73
                tp = tp[:, :years, ...]; feasible = feasible[:years]  # subsetting transition probabilities and feasibility indicators
            else: # base horizon
                Lterm = lterm_list[0]  # terminal condition
                years = 10  # 10-year planning horizon
                tp = tp[:, :years, ...]; feasible = feasible[:years]  # subsetting transition probabilities and feasibility indicators

            # Extracting number of treatment choices from transition probabilities
            numtrt = tp.shape[2]

            # Diabetes status indicators
            diab_stat = tp.shape[3]
            healthy_list = healthy_ind[:diab_stat]
            
            # Evaluating no treatment in true transition probabilities
            V_no_trt = alg.evaluate_pi(np.zeros((years, diab_stat), dtype=int), tp, healthy_list, Lterm, disutility, discount)

            # Calculating policies using backwards induction
            Q, V, pi = alg.backwards_induction(tp, healthy_list, Lterm, disutility, discount, feasible)

            # Calculating action-value functions and policies using SBBI and SBBI + SBMCC
            st_time = tm.time()  # start time for combined algorithms
            ## Calculating necessary observations for a beta confidence level
            obs = np.ceil(2*(((np.sum(discount**np.arange(years))+(discount**9)*Lterm)-0)**2)*np.log(numtrt/beta)).astype(int)

            ## Initializing objects to store results
            ### SBBI
            Q_hat_sbbi = np.full((years, numtrt, diab_stat), np.nan)  # initializing overall estimate of Q-values
            pi_hat_sbbi = np.full((years, diab_stat), -999, dtype='int')  # initializing approximate optimal actions (using invalid action index)

            ### SBBI + SBMCC
            Q_hat_ctrl = np.full((years, numtrt, diab_stat), np.nan)  # initializing overall estimate of Q-values for a single replication
            Q_bar = np.full((years, numtrt, diab_stat, reps), np.nan)  # initializing estimate of Q-values per simulation replicate
            sigma2_bar = np.full((years, numtrt, diab_stat, reps), np.nan)  # initializing estimate of the variance of Q-values per simulation replicate
            sigma2_hat = np.full((years, numtrt, diab_stat), np.nan)  # initializing estimate of the variance of the average Q-values per simulation replicate

            ### SBMCC
            a_ctrl = np.full((years, diab_stat), -999, dtype='int')  # initializing control actions
            d_alpha = np.full((years, diab_stat), np.nan)  # array to store empirical 1-epsilon quantiles at all decision epochs accross sensitivity analysis scenarios
            Pi = [pd.DataFrame() for _ in range(diab_stat)] # list to store ranges of actions at all decision epochs across diabetes statuses
            Pi_meds = Pi.copy() # list to store ranges of medications at all decision epochs across diabetes statuses

            ## Executing a single replication of SBBI algorithm (to identify and controls and establish runninng time)
            for t in reversed(range(years)):  # number of decisions remaining
                # Q-value at next period
                if t == max(range(years)):
                    Q_hat_next = [Lterm, Lterm]  # expected lifetime as terminal reward
                else:
                    Q_hat_next = [Q_hat_ctrl[t+1, a_ctrl[t+1, d], d] for d in
                                  range(diab_stat)]  # Q-value associated with the approximately optimal action

                # Extracting transition probabilities for the current decision epoch
                tp_t = tp[:, t, ...]

                # Running Simulation-based backwards induction algorithm (only for healthy states)
                ## Running health trajectories in parallel (only for healthy states)
                with mp.Pool(cores) as pool:  # creating pool of parallel workers
                    Q_sim1 = pool.starmap_async(alg.sbbi, [(tp_t, disutility, healthy_list, Q_hat_next, discount, r)
                                                            for r in range(obs)]).get() # reps = 1

                ## Converting results into array (with appropriate dimensions)
                Q_sim1 = np.array(Q_sim1)

                ## Calculating estimates of Q-values and their variances per replication
                Q_hat_ctrl[t, ...] = np.nanmean(Q_sim1, axis=0)  # estimated Q-value in a single batch
                a_ctrl[t, :] = np.argmax(Q_hat_ctrl[t, feasible[t], :], axis=0)  # approximately optimal policies

            limit = tm.time()-st_time  # time elapsed for a single replication - time limit for MC and TD methods

            ## Executing combined SBBI+SBMCC algorithms
            for t in reversed(range(years)): # number of decisions remaining
                # Q-value at next period
                if t == max(range(years)):
                    Q_hat_next = [Lterm, Lterm] # expected lifetime as terminal reward
                else:
                    Q_hat_next = [Q_hat_sbbi[t+1, pi_hat_sbbi[t+1, d], d] for d in range(diab_stat)] # Q-value associated with the approximately optimal action

                # Extracting transition probabilities for the current decision epoch
                tp_t = tp[:, t, ...]

                # Running Simulation-based backwards induction algorithm (only for healthy states)
                ## Running health trajectories in parallel (only for healthy states)
                with mp.Pool(cores) as pool:  # creating pool of parallel workers
                    Q_sim = pool.starmap_async(alg.sbbi, [(tp_t, disutility, healthy_list, Q_hat_next, discount, r)
                                                           for r in range(obs*reps)]).get() # reps > 1

                ## Converting results into array (with appropriate dimensions)
                Q_sim = np.moveaxis(np.array(np.split(np.stack(Q_sim), reps, axis=0)), [0, 1], [-1, -2]) # for multiple replications

                ## Calculating estimates of Q-values and approximately optimal policy
                Q_bar[t, ...] = np.nanmean(Q_sim, axis=2)  # estimated Q-value at each replication (for a single replication)
                sigma2_bar[t, ...] = np.nanvar(Q_sim, axis=2, ddof=1)  # estimated variance per replication
                Q_hat_sbbi[t, :] = np.nanmean(Q_bar[t, ...], axis=2)  # overall estimated Q-value (excluding initial batch)
                pi_hat_sbbi[t, :] = np.argmax(Q_hat_sbbi[t, feasible[t], :], axis=0)  # approximately optimal policies
                sigma2_hat[t, ...] = np.nanvar(Q_bar[t, ...], axis=2, ddof=1)  # estimated variance of the replication average (excluding initial batch)

                # Running simulation-based multiple comparison with a control algorithm
                ## Calculating root statistic in parallel (only in healthy states)
                with mp.Pool(cores) as pool:  # creating pool of parallel workers
                    max_psi = pool.starmap_async(alg.sbmcc, [(Q_bar[t, ...], Q_hat_sbbi[t, :],
                                                                 sigma2_bar[t, ...], a_ctrl[t, :], obs, rep)
                                                                for rep in range(reps)]).get()

                ## Converting results to an array
                max_psi = np.array(max_psi)

                ## Calculating quantile values
                d_alpha[t, :] = np.apply_along_axis(np.quantile, axis=0, arr=max_psi, q=(1 - alpha), method="closest_observation")

                ## Identifying set of actions that are not significantly different from the approximately optimal action (in the set of feasible actions)
                for d in range(diab_stat):
                    Pi_epoch = np.where(Q_hat_sbbi[t, a_ctrl[t, d], d]-Q_hat_sbbi[t, :, d] <=
                                        d_alpha[t, d]*np.sqrt((sigma2_hat[t, :, d]+
                                                                   sigma2_hat[t, a_ctrl[t, d], d])/reps))[0]
                    Pi_epoch = [a for a in Pi_epoch if a in feasible[t]]  # including only feasible actions

                    ### Making sure that we at least get one element in the set (if there is no variation in Q-values Pi_epoch = [])
                    if len(Pi_epoch) == 0:
                        Pi_epoch = [a_ctrl[t, d]]
    
                    ### Saving range of near-optimal policies
                    Pi[d] = pd.concat([pd.DataFrame(Pi_epoch, columns=[str(t)]), Pi[d]], axis=1)
            limit_svp = tm.time() - st_time  # time elapsed in combined algorithms - time limit for TD-SVP

            # Evaluating policy from a single replication of SBBI in true transition probabilities
            V_pi_sbbi = alg.evaluate_pi(a_ctrl, tp, healthy_list, Lterm, disutility, discount)

            ## Fewest number of medication in sets
            pi_fewest_sbbi_sbmcc = [Pi[d].min(axis=0) for d in range(diab_stat)]  # identifying the fewest number of medications each year
            pi_fewest_sbbi_sbmcc = [[pi_hat_sbbi[t, d] if np.isnan(pi_fewest_sbbi_sbmcc[d].iloc[t]) else pi_fewest_sbbi_sbmcc[d].iloc[t].astype(int)
                                    for t in range(pi_fewest_sbbi_sbmcc[d].shape[0])] for d in range(diab_stat)]  # making sure that there are elements in each year's sets
            pi_fewest_sbbi_sbmcc = np.array(pi_fewest_sbbi_sbmcc).reshape(diab_stat, years).T # arranging in apropriate dimensions
            V_fewest_sbbi_sbmcc = alg.evaluate_pi(pi_fewest_sbbi_sbmcc, tp, healthy_list, Lterm, disutility, discount)  # evaluating policy in true transition probabilities

            ## Median number of medications in ranges
            pi_median_sbbi_sbmcc = [np.ceil(Pi[d].median(axis=0)) for d in range(diab_stat)]  # identifying the fewest number of medications each year
            pi_median_sbbi_sbmcc = [[pi_hat_sbbi[t, d] if np.isnan(pi_median_sbbi_sbmcc[d].iloc[t]) else pi_median_sbbi_sbmcc[d].iloc[t].astype(int)
                                    for t in range(pi_median_sbbi_sbmcc[d].shape[0])] for d in range(diab_stat)]  # making sure that there are elements in each year's sets
            pi_median_sbbi_sbmcc = np.array(pi_median_sbbi_sbmcc).reshape(diab_stat, years).T  # arranging in apropriate dimensions
            V_median_sbbi_sbmcc = alg.evaluate_pi(pi_median_sbbi_sbmcc, tp, healthy_list, Lterm, disutility, discount)  # evaluating policy in true transition probabilities

            ## Best treatment in ranges
            V_best_sbbi_sbmcc = alg.evaluate_pi(pi_hat_sbbi, tp, healthy_list, Lterm, disutility, discount)  # evaluating policy in true transition probabilities

            # Calculating action-value functions and policies using on-policy MC control
            time = tm.time()  # start time for algorithm
            Q_hat_on_mcc, pi_hat_on_mcc = alg.on_policy_mcc(tp, Lterm, disutility, healthy_list, absorving, feasible, discount, time, limit, epsilon=1) # executing algorithm
            pi_hat_on_mcc_gd = pi_hat_on_mcc.argmax(axis=1) # identifying greedy actions
            V_pi_on_mcc = alg.evaluate_pi(pi_hat_on_mcc_gd, tp, healthy_list, Lterm, disutility, discount) # evaluating policy in true transition probabilities

            # Calculating action-value functions and policies using off-policy MC control
            time = tm.time()  # start time for algorithm
            Q_hat_off_mcc, pi_hat_off_mcc = alg.off_policy_mcc(tp, Lterm, disutility, healthy_list, absorving, feasible, discount, epsilon, time, limit)  # executing algorithm (assuming epsilon=0.1)
            V_pi_off_mcc = alg.evaluate_pi(pi_hat_off_mcc, tp, healthy_list, Lterm, disutility, discount)  # evaluating policy in true transition probabilities

            # Calculating action-value functions and policies using Sarsa
            time = tm.time()  # start time for algorithm
            Q_hat_sarsa, pi_hat_sarsa = alg.sarsa(tp, Lterm, disutility, healthy_list, absorving, feasible, discount, time, limit, epsilon=1)  # executing algorithm
            pi_hat_sarsa_gd = pi_hat_sarsa.argmax(axis=1)  # identifying greedy actions
            V_pi_sarsa = alg.evaluate_pi(pi_hat_sarsa_gd, tp, healthy_list, Lterm, disutility, discount)  # evaluating policy in true transition probabilities

            # Calculating action-value functions and policies using Q-learning
            time = tm.time()  # start time for algorithm
            Q_hat_q_learn, pi_hat_q_learn = alg.q_learning(tp, Lterm, disutility, healthy_list, absorving, feasible, discount, epsilon, time, limit)  # executing algorithm (assuming epsilon=0.1)
            V_hat_q_learn = np.max(Q_hat_q_learn, axis=1) # calculating estimates of value functions with Q-learning (for TD-SVP)
            V_pi_q_learn = alg.evaluate_pi(pi_hat_q_learn, tp, healthy_list, Lterm, disutility, discount)  # evaluating policy in true transition probabilities

            # Calculating action-value functions and policies using TD-SVP
            time = tm.time()  # start time for algorithm
            Q_hat_td_svp, Pi_td_svp = alg.td_svp(V_hat_q_learn, zeta, tp, Lterm, disutility, healthy_list, absorving, feasible, discount, epsilon, time, limit_svp)
            
            ## Fewest number of medication in sets
            pi_fewest_td_svp = [Pi_td_svp[d].min(axis=0) for d in range(diab_stat)]  # identifying the fewest number of medications each year
            pi_fewest_td_svp = [[pi_hat_q_learn[t, d] if np.isnan(pi_fewest_td_svp[d][t]) else pi_fewest_td_svp[d][t].astype(int)
                                    for t in range(pi_fewest_td_svp[d].shape[0])] for d in range(diab_stat)]  # making sure that there are elements in each year's sets
            pi_fewest_td_svp = np.array(pi_fewest_td_svp).reshape(diab_stat, years).T # arranging in apropriate dimensions
            V_fewest_td_svp = alg.evaluate_pi(pi_fewest_td_svp, tp, healthy_list, Lterm, disutility, discount)  # evaluating policy in true transition probabilities

            ## Median number of medications in sets
            pi_median_td_svp = [np.ceil(Pi_td_svp[d].median(axis=0)) for d in range(diab_stat)]  # identifying the median number of medications each year
            pi_median_td_svp = [[pi_hat_q_learn[t, d] if np.isnan(pi_median_td_svp[d][t]) else pi_median_td_svp[d][t].astype(int)
                                for t in range(pi_median_td_svp[d].shape[0])] for d in range(diab_stat)]  # making sure that there are elements in each year's sets
            pi_median_td_svp = np.array(pi_median_td_svp).reshape(diab_stat, years).T  # arranging in apropriate dimensions
            V_median_td_svp = alg.evaluate_pi(pi_median_td_svp, tp, healthy_list, Lterm, disutility, discount)  # evaluating policy in true transition probabilities
            
            ## Best treatment in sets
            ### Identifying the best treatment in set each year or attaching action from Q-learning (if the set is empty)
            pi_best_td_svp = [[] for _ in range(diab_stat)] # list to store treatment strategy at each diabetes status
            for d in range(diab_stat):
                for t in range(years):
                    elem = Pi_td_svp[d].iloc[:, t].dropna().astype(int)
                    if len(elem) > 0:
                        pi_best_td_svp[d].append(Pi_td_svp[d].iloc[np.argmax(Q_hat_td_svp[t, elem, d], axis=0), t].astype(int))
                    else:
                        pi_best_td_svp[d].append(pi_hat_q_learn[t, d])
            pi_best_td_svp = np.array(pi_best_td_svp).reshape(diab_stat, years).T  # arranging in apropriate dimensions
            V_best_td_svp = alg.evaluate_pi(pi_best_td_svp, tp, healthy_list, Lterm, disutility, discount)  # evaluating policy in true transition probabilities

            ## Data frame of results for a single patient (single result per patient-year)
            ### Initial diabetes status
            ptresults_base[sc] = pd.concat([pd.Series(np.repeat(i, years), name='id'),
                                            pd.Series(np.arange(years), name='year'),

                                            pd.Series(V_no_trt[:, 0], name='V_notrt'), # patient's true value functions for no treatment
                                            pd.Series(V[:, 0], name='V_opt'), # patient's true optimal value functions
                                            pd.Series(V_pi_sbbi[:, 0], name='V_pi_sbbi'), # patient's true value functions under SBBI
                                            pd.Series(V_pi_on_mcc[:, 0], name='V_pi_on_mcc'),  # patient's true value functions under on-policy MCC
                                            pd.Series(V_pi_off_mcc[:, 0], name='V_pi_off_mcc'),  # patient's true value functions under off-policy MCC
                                            pd.Series(V_pi_sarsa[:, 0], name='V_pi_sarsa'),  # patient's true value functions under Sarsa
                                            pd.Series(V_pi_q_learn[:, 0], name='V_pi_q_learn'),  # patient's true value functions under Q-learning
                                            pd.Series(V_fewest_sbbi_sbmcc[:, 0], name='V_fewest_sbbi_sbmcc'), # patient's true value functions using the fewest number of medications in SBBI-SBMCC
                                            pd.Series(V_median_sbbi_sbmcc[:, 0], name='V_median_sbbi_sbmcc'), # patient's true value functions using the median number of medications in SBBI-SBMCC
                                            pd.Series(V_best_sbbi_sbmcc[:, 0], name='V_best_sbbi_sbmcc'), # patient's true value functions using the best treatment in SBBI-SBMCC
                                            pd.Series(V_fewest_td_svp[:, 0], name='V_fewest_td_svp'), # patient's true value functions using the fewest number of medications in TD-SVP
                                            pd.Series(V_median_td_svp[:, 0], name='V_median_td_svp'), # patient's true value functions using the median number of medications in TD-SVP
                                            pd.Series(V_best_td_svp[:, 0], name='V_best_td_svp'), # patient's true value functions using the best treatment in TD-SVP

                                            pd.Series(pi[:, 0], name='pi_opt'), # patient's true optimal policy
                                            pd.Series(a_ctrl[:, 0], name='pi_hat_sbbi'), # patient's policy according to SBBI
                                            pd.Series(pi_hat_on_mcc_gd[:, 0], name='pi_hat_on_mcc'), # patient's policy according to on-policy MCC
                                            pd.Series(pi_hat_off_mcc[:, 0], name='pi_hat_off_mcc'), # patient's policy according to off-policy MCC
                                            pd.Series(pi_hat_sarsa_gd[:, 0], name='pi_hat_sarsa'), # patient's policy according to Sarsa
                                            pd.Series(pi_hat_q_learn[:, 0], name='pi_hat_q_learn'), # patient's policy according to Q-learning
                                            pd.Series(pi_fewest_sbbi_sbmcc[:, 0], name='pi_fewest_sbbi_sbmcc'), # patient's policy using the fewest number of medications in SBBI-SBMCC
                                            pd.Series(pi_median_sbbi_sbmcc[:, 0], name='pi_median_sbbi_sbmcc'), # patient's policy using the median number of medications in SBBI-SBMCC
                                            pd.Series(pi_hat_sbbi[:, 0], name='pi_best_sbbi_sbmcc'), # patient's policy using the best treatment in SBBI-SBMCC
                                            pd.Series(pi_fewest_td_svp[:, 0], name='pi_fewest_td_svp'), # patient's policy using the fewest number of medications in TD-SVP
                                            pd.Series(pi_median_td_svp[:, 0], name='pi_median_td_svp'), # patient's policy using the median number of medications in TD-SVP
                                            pd.Series(pi_best_td_svp[:, 0], name='pi_best_td_svp') # patient's policy using the best treatment in TD-SVP

                                            ], axis=1)
            
            ### With diabetes
            if diab_stat == 2:
                ptresults_diab[sc] = pd.concat([pd.Series(np.repeat(i, years), name='id'),
                                                pd.Series(np.arange(years), name='year'),

                                                pd.Series(V_no_trt[:, 1], name='V_notrt'), # patient's true value functions for no treatment
                                                pd.Series(V[:, 1], name='V_opt'), # patient's true optimal value functions
                                                pd.Series(V_pi_sbbi[:, 1], name='V_pi_sbbi'), # patient's true value functions under SBBI
                                                pd.Series(V_pi_on_mcc[:, 1], name='V_pi_on_mcc'),  # patient's true value functions under on-policy MCC
                                                pd.Series(V_pi_off_mcc[:, 1], name='V_pi_off_mcc'),  # patient's true value functions under off-policy MCC
                                                pd.Series(V_pi_sarsa[:, 1], name='V_pi_sarsa'),  # patient's true value functions under Sarsa
                                                pd.Series(V_pi_q_learn[:, 1], name='V_pi_q_learn'),  # patient's true value functions under Q-learning
                                                pd.Series(V_fewest_sbbi_sbmcc[:, 1], name='V_fewest_sbbi_sbmcc'), # patient's true value functions using the fewest number of medications in SBBI-SBMCC
                                                pd.Series(V_median_sbbi_sbmcc[:, 1], name='V_median_sbbi_sbmcc'), # patient's true value functions using the median number of medications in SBBI-SBMCC
                                                pd.Series(V_best_sbbi_sbmcc[:, 1], name='V_best_sbbi_sbmcc'), # patient's true value functions using the best treatment in SBBI-SBMCC
                                                pd.Series(V_fewest_td_svp[:, 1], name='V_fewest_td_svp'), # patient's true value functions using the fewest number of medications in TD-SVP
                                                pd.Series(V_median_td_svp[:, 1], name='V_median_td_svp'), # patient's true value functions using the median number of medications in TD-SVP
                                                pd.Series(V_best_td_svp[:, 1], name='V_best_td_svp'), # patient's true value functions using the best treatment in TD-SVP

                                                pd.Series(pi[:, 1], name='pi_opt'), # patient's true optimal policy
                                                pd.Series(a_ctrl[:, 1], name='pi_hat_sbbi'), # patient's policy according to SBBI
                                                pd.Series(pi_hat_on_mcc_gd[:, 1], name='pi_hat_on_mcc'), # patient's policy according to on-policy MCC
                                                pd.Series(pi_hat_off_mcc[:, 1], name='pi_hat_off_mcc'), # patient's policy according to off-policy MCC
                                                pd.Series(pi_hat_sarsa_gd[:, 1], name='pi_hat_sarsa'), # patient's policy according to Sarsa
                                                pd.Series(pi_hat_q_learn[:, 1], name='pi_hat_q_learn'), # patient's policy according to Q-learning
                                                pd.Series(pi_fewest_sbbi_sbmcc[:, 1], name='pi_fewest_sbbi_sbmcc'), # patient's policy using the fewest number of medications in SBBI-SBMCC
                                                pd.Series(pi_median_sbbi_sbmcc[:, 1], name='pi_median_sbbi_sbmcc'), # patient's policy using the median number of medications in SBBI-SBMCC
                                                pd.Series(pi_hat_sbbi[:, 1], name='pi_best_sbbi_sbmcc'), # patient's policy using the best treatment in SBBI-SBMCC
                                                pd.Series(pi_fewest_td_svp[:, 1], name='pi_fewest_td_svp'), # patient's policy using the fewest number of medications in TD-SVP
                                                pd.Series(pi_median_td_svp[:, 1], name='pi_median_td_svp'), # patient's policy using the median number of medications in TD-SVP
                                                pd.Series(pi_best_td_svp[:, 1], name='pi_best_td_svp') # patient's policy using the best treatment in TD-SVP

                                                ], axis=1)
            
            # Extracting sampling weights from original dataset
            wts = pd.DataFrame(ptdata.iloc[list(np.where([j in list(ptresults_base[sc].id.unique()) for j in list(ptdata.id)])[0]),
                                           [ptdata.columns.get_loc(col) for col in ["id", "wt"]]], columns=["id", "wt"])
            wts.reset_index(drop=True, inplace=True)

            # Adding sampling weights to final results
            ## Initial diabetes status
            ptresults_base[sc] = pd.concat([ptresults_base[sc].iloc[:, [ptresults_base[sc].columns.get_loc(col) for col in ["id", "year"]]], wts["wt"],
                                            ptresults_base[sc].iloc[:, [ptresults_base[sc].columns.get_loc(col) 
                                                                        for col in ptresults_base[sc].columns.difference(["id", "year"])]]],
                                           axis=1)

            ## With diabetes
            if diab_stat == 2:
                ptresults_diab[sc] = pd.concat([ptresults_diab[sc].iloc[:, [ptresults_diab[sc].columns.get_loc(col) for col in ["id", "year"]]],
                                                wts["wt"], ptresults_diab[sc].iloc[:, [ptresults_diab[sc].columns.get_loc(col)
                                                                                       for col in ptresults_diab[sc].columns.difference(["id", "year"])]]],
                                               axis=1)

            # Merging single patient data in data frame with data from of all patients
            ## Initial diabetes status
            pt_sim_base[sc] = pd.concat([pt_sim_base[sc], ptresults_base[sc]], ignore_index=True)
            ptresults_base[sc] = np.nan  # making sure values are not recycled
            
            ## With diabetes
            pt_sim_diab[sc] = pd.concat([pt_sim_diab[sc], ptresults_diab[sc]], ignore_index=True)
            ptresults_diab[sc] = np.nan  # making sure values are not recycled
            
            # Saving patient-level results (for healthy states only)
            Q_bi[sc].append(Q) # patient's true Q-values
            Q_sbbi[sc].append(Q_hat_sbbi)  # patient's estimates of Q-values from SBBI
            Q_on_mcc[sc].append(Q_hat_on_mcc)  # patient's estimates of Q-values from on-policy MCC
            Q_off_mcc[sc].append(Q_hat_off_mcc)  # patient's estimates of Q-values from off-policy MCC
            Q_sarsa[sc].append(Q_hat_sarsa)  # patient's estimates of Q-values from Sarsa
            Q_q_learn[sc].append(Q_hat_q_learn)  # patient's estimates of Q-values from Q-learning
            Q_td_svp[sc].append(Q_hat_td_svp)  # patient's estimates of Q-values from Q-learning
            action_set_sbbi_sbmcc[sc].append(Pi)  # patient's sets of near-optimal treatment choices using SBBI+SBMCC
            action_set_td_svp[sc].append(Pi_td_svp)  # patient's sets of near-optimal treatment choices using TD-SVP

            # Saving all results (saving each time a patient and scenario are evaluated)
            os.chdir(results_dir) # chaging to results directory
            with open('Algorithm Comparison Results in MDP Variations - Records '+str(min(id_seq_sm))+' to '+str(max(id_seq_sm))+'.pkl', 'wb') as f:
                pk.dump([
                    Q_bi, Q_sbbi, Q_on_mcc, Q_off_mcc, Q_sarsa, Q_q_learn, Q_td_svp,
                    action_set_sbbi_sbmcc, action_set_td_svp, pt_sim_base, pt_sim_diab], f, protocol=3)
            os.chdir(home_dir)  # returning to home directory
