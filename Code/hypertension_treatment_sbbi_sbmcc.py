# *******************************************************************************************************
# Flexible Action Selection in Finite MDP Models Case Study - Hypertension Treatment Numerical Analysis
# *******************************************************************************************************

# --------
# Setup
# --------

# Loading modules
import os  # directory changes
import pandas as pd  # data frame operations
import numpy as np  # matrix operations
import time as tm  # timing code
import pickle as pk  # saving results
import multiprocessing as mp  # parallel computations
from bp_med_effects import med_effects, drug_type_effects # medication parameters (using generic and drug types)
from ascvd_risk import rev_arisk  # risk calculations
from transition_probabilities import TP  # transition probability calculations
import sb_bi  # simulation-based backwards induction
import sb_mcc # simulation-based multiple comparison with a control algorithm
from policy_evaluation import evaluate_pi, evaluate_events # policy evaluation
from aha_2017_guideline import aha_guideline # 2017 AHA's guideline for hypertension treatment
from backwards_induction_mdp import backwards_induction  # solving MDP using backwards induction

# Establishing directories (Please make sure the data and code are under the same main directory)
home_dir = os.getcwd() # Please change to apropriate path - using current directory as an example
os.chdir(home_dir) # changing directory to home directory
if not os.path.isdir("Data"): # verifying if there is a sub-directory where data is stored and if not creating it
    os.mkdir("Data")
data_dir = home_dir+"\\Data"

if not os.path.isdir("Results"): # verifying if there is a sub-directory to store results and if not creating it
    os.mkdir("Results")
results_dir = home_dir+"\\Results"

if not os.path.isdir("Figures"): # verifying if there is a sub-directory to store figures and if not creating it
    os.mkdir("Figures")
fig_dir = home_dir+"\\Figures"

# Selecting number of cores for parallel processing
cores = mp.cpu_count() - 1

# -----------------------
# Initializing parameters
# -----------------------

# Simulation parameters
reps = int(301) # number of simulation replications  # see number of batches analysis results
ctrl_reps = int(1) # initial 100 batches are used to identify a control only
beta = 0.001 # significance level for adaptive sample size

# MDP parameters
discount = 0.97  # discount factor for QoL and costs

# MCC parameters
alpha = 0.05 # significance level of simultaneous confidence intervals

# Transition probability parameters
numhealth = 6  # Number of health states
years = 10  # Number of years (non-stationary stages)
events = 2  # Number of events considered in model

# Treatment parameters
## BP clinical constraints
sbpmin = 120  # minimum allowable SBP
sbpmax = 150  # maximum allowable SBP
dbpmin = 55  # minimum allowable DBP

## AHA's guideline parameters
targetrisk = 0.1
targetsbp = 130
targetdbp = 80

## Half dosages compared to standard dosages
hf_red_frac = 2/3 # fraction of BP and risk reduction
hf_disut_frac = 1/2 # fraction of disutility

## Estimated change in BP by dosage (assuming absolute BP reductions and linear reductions with respect to dose)
sbp_drop_std = 5.5 # average SBP reduction per medication at standard dose in BPLTCC trials
sbp_drop_hf = sbp_drop_std*hf_red_frac # average SBP reduction per medication at half dose
dbp_drop_std = 3.3 # average DBP reduction per medication at standard dose in BPLTCC trials
dbp_drop_hf = dbp_drop_std*hf_red_frac # average DBP reduction per medication at half dose

# Estimated change in risk by dosage (assuming absolute risk reductions)
rel_risk_chd_std = 0.87 # estimated change in risk for CHD events per medication at standard dose in BPLTCC trials
rel_risk_stroke_std = 0.79 # estimated change in risk for stroke events per medication at standard dose in BPLTCC trials
rel_risk_chd_hf = 1-((1-rel_risk_chd_std)*hf_red_frac) # estimated change in risk for CHD events per medication at half dose
rel_risk_stroke_hf = 1-((1-rel_risk_stroke_std)*hf_red_frac) # estimated change in risk for stroke events per medication at half dose

## Estimated treatment disutility by dosage
disut_std = 0.002 # treatment disutility per medication at standard dose
disut_hf = disut_std*hf_disut_frac # treatment disutility per medication at half dose

## Treatment choices (21 trts: no treatment plus 1 to 5 drugs at standard and half dosages)
allmeds = list(range(21))  # index for possible treatment options
numtrt = len(allmeds)  # number of treatment choices

## Drug types for MDP variation analysis
drugs = ["ACE", "ARB", "BB", "CCB", "TH"] # ["ACE inhibitors", "Angiotensin II receptor blockers", "Beta-blockers", "Calcium channel blockers", "Thiazides"]

# Action order accroding to their SBP reduction (equivalent to ordering by RRR)
# (obtained from the TP_DRUG function in the transition_probabilities.py file retroactively by running a patient with the average BP [SBP = 154, DBP = 97] in Law et al. (2009))
action_order = [0, 1, 4, 5, 3, 2, 6, 9, 8, 19, 18, 17, 7, 15, 16, 14, 12, 13, 11, 10, 20, 23, 22, 29, 28, 21, 27, 46,
                49, 48, 47, 25, 26, 43, 44, 24, 45, 41, 42, 40, 37, 38, 39, 35, 36, 34, 32, 33, 31, 30, 50, 53, 52,
                59, 51, 58, 57, 69, 68, 56, 67, 55, 66, 100, 104, 103, 102, 101, 54, 63, 64, 65, 96, 97, 61, 98, 62,
                99, 93, 60, 94, 95, 91, 92, 86, 87, 88, 90, 89, 83, 84, 85, 81, 82, 77, 80, 78, 79, 75, 76, 74, 72,
                73, 71, 105, 108, 107, 114, 106, 113, 112, 70, 124, 111, 123, 110, 122, 121, 139, 138, 109, 120, 137,
                119, 136, 118, 135, 191, 193, 190, 192, 195, 194, 116, 131, 117, 132, 133, 134, 185, 186, 115, 128,
                187, 129, 188, 130, 189, 181, 126, 182, 127, 183, 184, 178, 125, 179, 170, 180, 171, 172, 176, 173,
                166, 177, 174, 167, 175, 168, 163, 169, 164, 156, 165, 161, 157, 162, 158, 153, 160, 159, 154, 151,
                155, 147, 152, 148, 150, 145, 149, 146, 144, 142, 143, 141, 140]

## Treatment effects (SBP reductions, DBP reductions, post-treatment relative risk for CHD events,
# post-treatment relative risk for CHD events, and treatment related disutilities)
### Generic medications
sbp_reduction, dbp_reduction, rel_risk_chd, rel_risk_stroke, disutility, meds = med_effects(hf_red_frac, sbp_drop_std,
                                                                                            sbp_drop_hf, dbp_drop_std,
                                                                                            dbp_drop_hf, rel_risk_chd_std,
                                                                                            rel_risk_chd_hf, rel_risk_stroke_std,
                                                                                            rel_risk_stroke_hf, disut_std,
                                                                                            disut_hf, numtrt,
                                                                                            "nondecreasing") # "nonincreasing"

std_index = [0, 2, 6, 11, 17, 20] # index of standard doses

### By drug type
alldrugs, disutility_dt = drug_type_effects(drugs, disut_std, action_order)

## Renal sympathetic denervation parameters
sbp_drop_rsn = 5.10 # office systolic blood pressure reduction in Ahmad et al. (2022)
dbp_drop_rsn = 3.11 # office diastolic blood pressure reduction in Ahmad et al. (2022)
rel_risk_chd_rsn = rel_risk_chd_std/sbp_drop_std*sbp_drop_rsn # assuming the same risk reduction per 5.5 mm Hg reduction in SBP as with one medication at standard dose for CHD events
rel_risk_stroke_rsn = rel_risk_stroke_std/sbp_drop_std*sbp_drop_rsn # assuming the same risk reduction per 5.5 mm Hg reduction in SBP as with one medication at standard dose for stroke events
disutility_rsn = 12500/100e03 # disutility from the cost of RSN in Geisler et al. (2012) assuming a WTP threshold of 100K/QALY

# State ordering
## Initial order: (healthy, surviving a CHD, surviving a stroke, dying from a CHD, dying from a stroke,
# dying from non-ASCVD related cause, and death or history of ASCVD)
order = [5, 4, 3, 2, 1, 0]  # ordering states in nondecreasing order of rewards
# oder = [0, 1, 2, 3, 4, 5] # ordering states in nonincreasing order of rewards
healthy = int(np.squeeze(np.where(np.array(order) == 0))) # identification of healthy state
event_id = [x for _, x in sorted(zip(order, [0, 1, 1, 1, 1, 0]))] # identification of states in where ASCVD events happen
absorving = len(order) # additional state to indicate death or history of ASCVD (absorving state)

# --------------
# Loading data
# --------------

# Loading life expectancy and death likelihood data (first column age, second column male, third column female)
os.chdir(data_dir)
lifedata = pd.read_csv('lifedata.csv', header=None)
strokedeathdata = pd.read_csv('strokedeathdata.csv', header=None)
chddeathdata = pd.read_csv('chddeathdata.csv', header=None)
alldeathdata = pd.read_csv('alldeathdata.csv', header=None)

# Loading risk slopes (first column age, second column CHD, third column stroke) - BP reductions using parameters in Law et al. 2003 and 2009
riskslopedata = pd.read_csv('riskslopes.csv', header=None)

# Loading 2009-2016 Continuous NHANES data (imputed with random forests and forecasted with linear regression in R)
os.chdir(data_dir+'\\Continuous NHANES')
ptdata50 = pd.read_csv('Continuous NHANES 50-54 Dataset.csv') # ages 50-54 dataset

# Adding indicator of Black race in data frame (for revised risk calculations)
ptdata50['black'] = np.where(ptdata50.race==1, 0, 1)

# Re-ordering columns
cols = ptdata50.columns.tolist()
cols = cols[:(int(np.squeeze(np.where(ptdata50.columns.str.match(pat='race'))))+1)] + \
       [cols[int(np.squeeze(np.where(ptdata50.columns.str.match(pat='black'))))]] + \
       cols[(int(np.squeeze(np.where(ptdata50.columns.str.match(pat='race'))))+1):-1]
ptdata50 = ptdata50[cols]; del cols

# --------------------
# Sensitivity analyses
# --------------------

# Future action scenarios
future_action = ['best', 'random', 'worst']#[:1] # best, random, or worst action in next decision epoch (best is the base case scenario)
future_action_len = len(future_action)

# Disutility scenarios
eq_trt_aha_disut = disutility/disut_std*0.01836 # approximately equal number of medications as the AHA's guidelines (multiplier obtained iterating through potential disutility values [see equal_trt_disutility_mpd_aha.py])
disut_list = [disutility, disutility/2, disutility*2, eq_trt_aha_disut]#[:1] # base case, half, double, approximately equal number of medications as the AHA's guidelines
disut_len = len(disut_list)

# Risk scenarios
risk_mult_list = [1, 1/2, 2]#[:1]
risk_mult_len = len(risk_mult_list)

# Tratment benefit scenarios
trt_ben_mult_list = [1, 1/2, 2]#[:1]
trt_ben_mult_len = len(trt_ben_mult_list)

# Known distributions of rewards
known_dist = ['Constant', 'Normal']#[:1] # Gaussian
known_dist_len = len(known_dist)

# Ages scenarios
age_list = ['50-54', '70-74']#[:1] # base case and ages 70-74
age_sens_len = len(age_list)

## Loading data
os.chdir(data_dir+'\\Continuous NHANES')
ptdata70 = pd.read_csv('Continuous NHANES 70-74 Dataset.csv')

## Modiying dataset to match 50-54 NHANES data
init_num_rows = ptdata70.shape[0]
ptdata70 = ptdata70.reindex(ptdata70.index.tolist() + list(range(ptdata50.shape[0]-init_num_rows))) # adding rows to match shape of base case data
ptdata70.iloc[init_num_rows:, :] = np.nan # indentifying addional rows with NaN (it will look like missing data)
ptdata70.reset_index(drop=True, inplace=True) # resetting indexes
ptdata70['id'] = ptdata50['id'].copy() # making sure that both data frames have the same ids

## Adding indicator of Black race in data frame (for revised risk calculations)
ptdata70['black'] = np.where(ptdata70.race==1, 0, 1)
ptdata70['black'] = np.where(np.isnan(ptdata70.race), np.nan, ptdata70.black)

## Re-ordering columns
cols = ptdata70.columns.tolist()
cols = cols[:(int(np.squeeze(np.where(ptdata70.columns.str.match(pat='race'))))+1)] + \
       [cols[int(np.squeeze(np.where(ptdata70.columns.str.match(pat='black'))))]] + \
       cols[(int(np.squeeze(np.where(ptdata70.columns.str.match(pat='race'))))+1):-1]
ptdata70 = ptdata70[cols]; del cols

## Combining 50-54 (base case) and 70-74 data into a list
ptdata_list = [ptdata50, ptdata70]; del ptdata50, ptdata70

# Scenario summary
sens_sc = age_sens_len + (risk_mult_len-1) + (trt_ben_mult_len-1) + (disut_len-1) + (known_dist_len-1) + (future_action_len-1) # total number of scenarios (base case and sensitivity analysis) - excluding missestimation scenarios
sens_sc_best = sens_sc - future_action_len # number of sensitivity analysis scenarios assuming best action in next decision epoch
sens_sc_70 = 1 # position of sensitivity analysis using a population with ages 70-74
base_disut_ind = np.concatenate((np.arange(age_sens_len+risk_mult_len+trt_ben_mult_len-2),
                                 np.arange(age_sens_len+risk_mult_len+trt_ben_mult_len+disut_len-3,
                                           sens_sc)))  # index of scenarios with base disutility
disut_ind = np.setdiff1d(np.arange(sens_sc), base_disut_ind)  # index of disutility sensitivity analysis scenarios

# ------------------
# Patient simulation
# ------------------

# Objects to store results of patient simulation
risk1 = [[] for _ in range(sens_sc)] # (save only for debugging purposes)
risk10 = [[] for _ in range(sens_sc)] # (save only for debugging purposes)
transitions = [[] for _ in range(sens_sc)] # (save only for debugging purposes)
bi_Q = [[] for _ in range(sens_sc)]
sb_Q = [[] for _ in range(sens_sc)]
sb_sigma2 = [[] for _ in range(sens_sc)]
action_range = [[] for _ in range(sens_sc)]
medication_range = [[] for _ in range(sens_sc)]
pt_sim = [pd.DataFrame() for _ in range(sens_sc)]
# ci_width_list = [] # store only for number of replications analysis

# Evaluating different patient profiles
## Sampling patient
sample_pt = ptdata_list[0][ptdata_list[0].id==783].copy() # random patient in data set
sample_pt.sex = 1 # starting with male patient
sample_pt.smk = 0 # starting with non-smoker patient
sample_pt.diab = 0 # starting with non-diabetic patient

## Adding modified versions of the patient
### Black race
tmp = sample_pt.copy(); tmp.race = 0; tmp.black = 1
sample_df = pd.concat([sample_pt, tmp], axis=0).reset_index(drop=True, inplace=False)

### Female
tmp = sample_pt.copy(); tmp.sex = 0
sample_df = pd.concat([sample_df, tmp], axis=0).reset_index(drop=True, inplace=False)

### Smoker
tmp = sample_pt.copy(); tmp.smk = 1
sample_df = pd.concat([sample_df, tmp], axis=0).reset_index(drop=True, inplace=False)

### Diabetic
tmp = sample_pt.copy(); tmp.diab = 1
sample_df = pd.concat([sample_df, tmp], axis=0).reset_index(drop=True, inplace=False)

### Diabetic and smoker
tmp = sample_pt.copy(); tmp.diab = 1; tmp.smk = 1
sample_df = pd.concat([sample_df, tmp], axis=0).reset_index(drop=True, inplace=False)

### Elevated BP
tmp = sample_pt.copy(); tmp.sbp -= 5; tmp.dbp -= 5
sample_df = pd.concat([sample_df, tmp], axis=0).reset_index(drop=True, inplace=False)

### Stage 2 hypertension
tmp = sample_pt.copy(); tmp.sbp += 15; tmp.dbp += 15
sample_df = pd.concat([sample_df, tmp], axis=0).reset_index(drop=True, inplace=False)

### 60 year old
tmp = sample_pt.copy(); tmp.age += 6
sample_df = pd.concat([sample_df, tmp], axis=0).reset_index(drop=True, inplace=False)

### 70 year old
tmp = sample_pt.copy(); tmp.age += 16
sample_df = pd.concat([sample_df, tmp], axis=0).reset_index(drop=True, inplace=False)

### Modifying ids
sample_df.id = pd.Series(np.repeat(np.arange(10), years))
del sample_pt # deleting sample patient

# Running simulation
os.chdir(home_dir) # starting from home directory
id_seq = range(len(ptdata_list[0].id.unique())) # Sequence of patient id's to evaluate in population-level analysis
# id_seq = range(len(sample_df.id.unique())) # Sequence of patient id's to evaluate in patient-level analysis
if __name__ == '__main__':
    for i in id_seq:

        # Keeping track of progress
        print(tm.asctime(tm.localtime(tm.time()))[:-5], "Evaluating patient", i)

        # Calculating risks and transition probabilities in each sensitivity analysis scenario
        patient_list = [] # list to store patient's data in each age scenario
        lterm_list = [] # list to store life expectancies in each age scenario
        lterm_std_list = [] # list to store standard deviation of life expectancy across sex in each age scenario
        age_sens_list = [np.nan for _ in range(age_sens_len)]  # list to store risks and transition probabilities in each age scenario
        risk_sens_list = [np.nan for _ in range(risk_mult_len)] # list to store risks and transition probabilities in each risk scenario
        trt_ben_sens_list = [np.nan for _ in range(trt_ben_mult_len)] # list to store transition probabilities in each treatment benefit scenario
        ascvdrisk1, ascvdrisk10, tp, feasible = [np.nan for _ in range(4)] # initializing risk, transition probsbilities, and feasibility checks
        for a in range(age_sens_len):
            # Extracting patient's data from larger data matrix run for population-level analysis (main and sensitivity)
            patient_list.append(ptdata_list[a][ptdata_list[a].id == i])
            patientdata = patient_list[a]

            # # Extracting patient's data from dataframe of patient profiles run for patient-level analysis
            # patient_list.append(sample_df[sample_df.id == i])
            # patientdata = patient_list[a]

            if not patientdata.isnull().values.any(): # only run if there is a patient with the current id (not for the added rows to match the shape of the base case data)
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
                hle = lifedata.iloc[np.where(patientdata.age.iloc[max(range(years))] == lifedata.iloc[:, 0])[0][0], 1:].to_numpy() # healthy life expectancy for males and females of the patient's current age
                lterm_list.append(hle[sexcol-1]) # healthy life expectancy
                lterm_std_list.append(np.sqrt((1/12)*(((hle[0]-2*np.median(hle)+hle[1])**2)/4+(hle[1]-hle[0])**2))) # estimate of the standard deviation (as per Hozo et al. 2005)

                if a == 0: # run risk scenarios on base case population only
                    risk_loop = range(risk_mult_len)
                else: # run only base case risk
                    risk_loop = range(risk_mult_len)[:1]

                for r in risk_loop: # each risk sensitivity analysis scenario
                    ## Storing risk calculations
                    ascvdrisk1 = np.full((years, events), np.nan)  # 1-year CHD and stroke risk (for transition probabilities)
                    ascvdrisk10 = np.full((years, events), np.nan)  # 10-year CHD and stroke risk (for AHA's guidelines)

                    ## Calculating risk for healthy state only (before ordering of states)
                    for t in range(years): # each age
                        for k in range(events): # each event type

                            # 1-year ASCVD risk calculation (for transition probabilities)
                            ascvdrisk1[t, k] = rev_arisk(k, patientdata.sex.iloc[t], patientdata.black.iloc[t], patientdata.age.iloc[t],
                                                         patientdata.sbp.iloc[t], patientdata.smk.iloc[t], patientdata.tc.iloc[t],
                                                         patientdata.hdl.iloc[t], patientdata.diab.iloc[t], 0, 1)*risk_mult_list[r]

                            # 10-year ASCVD risk calculation (for AHA's guidelines)
                            ascvdrisk10[t, k] = rev_arisk(k, patientdata.sex.iloc[t], patientdata.black.iloc[t], patientdata.age.iloc[t],
                                                          patientdata.sbp.iloc[t], patientdata.smk.iloc[t], patientdata.tc.iloc[t],
                                                          patientdata.hdl.iloc[t], patientdata.diab.iloc[t], 0, 10)*risk_mult_list[r]

                    if r == 0 and a==0: # only run treatment benefit sensitivity in base case risk
                        trt_ben_mult_loop = range(trt_ben_mult_len)
                    else: # run only base case treatment benefit
                        trt_ben_mult_loop = range(trt_ben_mult_len)[:1]

                    for b in trt_ben_mult_loop: # each treatment benefit scenario
                        # Calculating transition probabilities
                        feas, tp = TP(ascvdrisk1, chddeath, strokedeath, alldeath, patientdata.sbp.values, patientdata.dbp.values,
                                      sbpmin, sbpmax, dbpmin, sbp_reduction*trt_ben_mult_list[b], dbp_reduction*trt_ben_mult_list[b],
                                      rel_risk_chd**trt_ben_mult_list[b], rel_risk_stroke**trt_ben_mult_list[b], numhealth)

                        # Sorting transition probabilities to satisfy stochastic ordering with respect to states
                        tp = tp[order, :, :]

                        # Extracting list of feasible actions per state and decision epoch
                        feasible = []  # stores index of feasible actions
                        for t in range(feas.shape[0]):
                            feasible.append(list(np.where(feas[t, :] == 1)[0]))
                        del feas

                        # Storing treatment benefit scenarios
                        if a==0 and r==0: # store only on age and risk base case
                            trt_ben_sens_list[b] = (tp, feasible)

                    # Storing risk scenarios
                    if a==0: # store only on age base case
                        tmp_tp, tmp_feasible = trt_ben_sens_list[0]
                        risk_sens_list[r] = (ascvdrisk1, ascvdrisk10, tmp_tp, tmp_feasible)
                        del tmp_tp, tmp_feasible

                # Storing age scenarios
                if a==0: # 50-54 age group
                    age_sens_list[a] = (risk_sens_list[0])
                elif a==1: #70-74 age group
                    age_sens_list[a] = (ascvdrisk1, ascvdrisk10, tp, feasible)

            else: # do not run if there is no patient (only the additional rows in the dataframe have missing data)
                age_sens_list[a] = age_sens_list[0]

        # Calculating necessary observations for a beta confidence level
        obs = np.ceil(2*(((np.sum(discount**np.arange(years))+(discount**9)*lterm_list[0]) - 0)**2)*np.log(numtrt/beta)*np.sqrt(1)).astype(int) # using life expectancy of the 50-54 patient (always larger than the life expectancy of the 70-74 patient)

        # Estimating Q-values using BI, SBBI, AHA's guidelines (only for healthy state)
        ## Initializing objects to store BI results
        Q = np.full((sens_sc, years, numtrt), np.nan)  # initializing true Q-values
        V = np.full((sens_sc, years), np.nan)  # initializing true value functions
        pi = np.full((sens_sc, years), np.nan)  # initializing optimal actions

        ## Initializing objects to store SBBI results
        Q_bar = np.full((sens_sc, years, numtrt, reps), np.nan)  # initializing estimate of Q-values per simulation replicate
        Q_hat = np.full((sens_sc, years, numtrt), np.nan)  # initializing overall estimate of Q-values
        pi_hat = np.full((sens_sc, years), -999, dtype='int')  # initializing approximate optimal actions (using invalid action index)
        sigma2_bar = np.full((sens_sc, years, numtrt, reps), np.nan)  # initializing estimate of the variance of Q-values per simulation replicate
        sigma2_hat = np.full((sens_sc, years, numtrt), np.nan) # initializing estimate of the variance of the average Q-values per simulation replicate

        ## Initializing objects to store SBMCC results
        a_ctrl = np.full((sens_sc, years), -999, dtype='int')  # initializing control actions (using invalid action index)
        d_alpha = np.full((sens_sc, years), np.nan)  # array to store empirical 1-epsilon quantiles at all decision epochs accross sensitivity analysis scenarios
        Pi = [pd.DataFrame() for _ in range(sens_sc)] # list to store ranges of actions at all decision epochs accross sensitivity analysis scenarios
        Pi_meds = Pi.copy() # list to store ranges of medications at all decision epochs accross sensitivity analysis scenarios

        for t in reversed(range(years)): # number of decisions remaining
            # Extracting transition probabilities and feasibility checks from appropriate scenarios at current year
            tp = [age_sens_list[x][2][:, t, :] for x in range(age_sens_len)] +\
                 [risk_sens_list[x][2][:, t, :] for x in range(1, risk_mult_len)] +\
                 [trt_ben_sens_list[x][0][:, t, :] for x in range(1, trt_ben_mult_len)] # excluding base case scenario (already included in age_sens_list)
            feasible = [age_sens_list[x][3][t] for x in range(age_sens_len)] +\
                       [risk_sens_list[x][3][t] for x in range(1, risk_mult_len)] +\
                       [trt_ben_sens_list[x][1][t] for x in range(1, trt_ben_mult_len)] +\
                       [age_sens_list[0][3][t]]*(disut_len-1) + \
                       [age_sens_list[0][3][t]]*(known_dist_len-1)+ \
                       [age_sens_list[0][3][t]]*(future_action_len-1) # excluding base case scenario (already included in age_sens_list)

            # Q-value at next period
            if t == max(range(years)): # terminal rewards
                Q_hat_next = [lterm_list[0].copy() for _ in range(future_action_len)] # expected lifetime as terminal reward

                # Rewards from known distributions
                np.random.seed(i)
                Q_hat_rv = [[np.nan, np.nan], [np.random.normal(1, 0.001, size=obs*reps), np.random.normal(lterm_list[0], lterm_std_list[0], size=obs*reps)]][:known_dist_len]

                # Terminal reward for patient with ages 70-74
                if len(lterm_list) > 1: # only run if there is a patient with the current id
                    Q_hat70_next = lterm_list[1]
                else: # added record to match shape of 50-54 dataframe
                    Q_hat70_next = np.nan
            else: # immediate rewards
                # Assuming that the patient will be treated next year with the best possible treatment choice in range
                sens_id = 0 # base case number
                next_trt = Pi[sens_id][str(t+1)][np.argmax(Q_hat[sens_id, t+1, Pi[sens_id][str(t+1)].dropna().astype(int)])].astype(int) # best action in next year's range
                Q_hat_next = [Q_hat[sens_id, t+1, next_trt]] # Q-value associated with the best action in next year's range

                if future_action_len > 1:
                    # Assuming that the patient will be treated next year with the median number of medications in range (rounding up, if necessary)
                    sens_id = sens_sc_best+1 # sensitivity analysis scenario number (this scenario is evaluated after all the scenarios assuming the best action in next year's range have been completed)
                    next_trt = np.ceil(Pi[sens_id][str(t+1)].median()).astype(int) # median action in next year's range
                    Q_hat_next_rnd = Q_hat[sens_id, t+1, next_trt] # Q-value associated with the random action in next year's range
                    Q_hat_next.append(Q_hat_next_rnd); del Q_hat_next_rnd

                    # Assuming that the patient will be treated next year with the fewest possible number of medications in range
                    sens_id = sens_sc_best+2 # sensitivity analysis scenario number (this scenario is evaluated after all the scenarios assuming the best action in next year's range have been completed)
                    next_trt = Pi[sens_id][str(t+1)].min().astype(int) # smallest action in next year's range
                    Q_hat_next_worst = Q_hat[sens_id, t+1, next_trt] # Q-value associated with the worst action in next year's range
                    Q_hat_next.append(Q_hat_next_worst); del Q_hat_next_worst

                # Rewards (immediate and value to go) from known distributions (constant, normal)
                Q_hat_rv = [[np.nan, np.nan], [np.random.normal(1, 0.001, size=obs*reps), np.repeat(Q_hat[sens_sc_best, t+1, pi_hat[sens_sc_best, t+1]], repeats=obs*reps)]][:known_dist_len]

                # Value to go for patient with ages 70-74
                if len(lterm_list) > 1: # only run if there is a patient with the current id
                    Q_hat70_next = Q_hat[sens_sc_70, t+1, pi_hat[sens_sc_70, t+1]]
                else: # added record to match shape of base case dataframe
                    Q_hat70_next = np.nan

            # Running simulation-based backwards induction algorithm (only for healthy state)
            ## Running health trajectories in parallel (only for healthy state)
            with mp.Pool(cores) as pool:  # creating pool of parallel workers
                Q_sim = pool.starmap_async(sb_bi.sbbi, [(tp, healthy, Q_hat_next, Q_hat_rv, Q_hat70_next, discount, r)
                                                        for r in range(obs*reps)]).get()

            ## Converting results into array (with appropriate dimensions)
            Q_sim = np.array(np.split(np.stack(Q_sim), reps, axis=0)).T  # Splitting results into reps number of batches of obs number of observations (and transposing array to match destination)

            # Calculating estimates of Q-values and their variances per replication
            Q_bar[base_disut_ind, t, :, :] = np.nanmean(Q_sim, axis=2)  # estimated Q-value at each replication (for scenarios with base disutility only)
            Q_bar[disut_ind, t, :, :] = Q_bar[0, t, :, :] # disutility scenarios are equal to the base case except for the disutility
            sigma2_bar[base_disut_ind, t, :, :] = np.nanvar(Q_sim, axis=2, ddof=1)  # estimated variance per replication (for scenarios with base disutility only)
            sigma2_bar[disut_ind, t, :, :] = sigma2_bar[0, t, :, :] # disutility scenarios are equal to the base case except for the disutility

            # Disutility sensitivity analysis (if the future best action is being chosen)
            for d in range(disut_len):
                # Generating disutility analysis scenarios
                if d == 0: # subtracting base case sensitivity
                    Q_bar[base_disut_ind, t, ...] = np.stack([Q_bar[base_disut_ind, t, a, ...]-disut_list[d][a] for a in range(numtrt)], axis=1)  # substracting base disutility per action
                else: # adding disutility scenario
                    Q_bar[disut_ind[d-1], t, ...] = np.stack([Q_bar[disut_ind[d-1], t, a, ...]-disut_list[d][a] for a in range(numtrt)], axis=0)  # substracting sensitivity disutility per action

            # Calculating estimates of Q-values and their variances across replications (for all scenarios)
            Q_bar[:, t, ...][Q_bar[:, t, ...] < 0] = 0  # making sure that rewards are non-negative (assuming that there is nothing worse than death)
            Q_hat[:, t, :] = np.nanmean(Q_bar[:, t, :, ctrl_reps:], axis=2)  # overall estimated Q-value (excluding initial batch)
            sigma2_hat[:, t, :] = np.nanvar(Q_bar[:, t, :, ctrl_reps:], axis=2, ddof=1)  # estimated variance of the replication (batch) average (excluding initial batch)
            pi_hat[:, t] = [np.argmax(Q_hat[f, t, feasible[f]], axis=0) for f in range(len(feasible))]  # approximately optimal policies

            # Running simulation-based multiple comparison with a control algorithm
            ## Identifying control
            Q_hat_ctrl = np.nanmean(Q_bar[:, t, :, :ctrl_reps], axis=2)
            a_ctrl[:, t] = [np.argmax(Q_hat_ctrl[f, feasible[f]], axis=0) for f in range(len(feasible))] # only considering initial batch
            ## Calculating root statistic in parallel (only in healthy state)
            with mp.Pool(cores) as pool:  # creating pool of parallel workers
                max_psi = pool.starmap_async(sb_mcc.sbmcc, [(Q_bar[:, t, :, ctrl_reps:], Q_hat[:, t, :],
                                                             sigma2_bar[:, t, :, ctrl_reps:], a_ctrl[:, t],
                                                             obs, rep)
                                                            for rep in range(reps-ctrl_reps)]).get() # excluding initial batch # add: byrep=True as the last argument for number of batches analysis

            # ## Estimating width of confidence intervals (only for base case - if byrep=True) # run only for number of batches analysis
            # d_alpha_reps = [np.quantile(max_psi[x], q=(1-alpha), method="closest_observation") for x in range(1, reps-1)] # calculating 1-alpha quantile for each replication (excluding initial batch)
            # sigma2_hat_reps = [np.nanvar(Q_bar[0, t, :, :(y+1)], axis=1, ddof=1) for y in range(1, reps-1)] # calculating estiamte of variance of the relication average (excluding initial batch)
            # ci_width = [np.amax(d_alpha_reps[x]*np.sqrt((sigma2_hat_reps[x][feasible[0]] + sigma2_hat_reps[x][a_ctrl[0, t]])/(x+1)))
            #             for x in range(reps-2)] # calculating width (excluding initial batch and results with a single replication)
            # ci_width_list.append(ci_width)

            ## Converting results to an array (if byrep=False)
            max_psi = np.array(max_psi)

            ## Calculating quantile values
            d_alpha[:, t] = np.apply_along_axis(np.quantile, axis=0, arr=max_psi,
                                                q=(1-alpha), method="closest_observation")
            d_alpha[sens_sc_best, :] = 2.658061 # replacing empirical quantile with normal quantile
            ### Note: normal quantile was obtained using qNCDun(p=0.95,nu=Inf,rho = rep(0.5,21),delta = rep(0,21),two.sided = F) = 2.658061 from the nCDunnett package in R

            ## Identifying set of actions that are not significantly different from the approximately optimal action (in the set of feasible actions)
            for sn_sc in range(sens_sc):
                Pi_epoch = np.where(Q_hat[sn_sc, t, a_ctrl[sn_sc, t]]-Q_hat[sn_sc, t, feasible[sn_sc]] <=
                                    d_alpha[sn_sc, t]*np.sqrt((sigma2_hat[sn_sc, t, feasible[sn_sc]] +
                                                               sigma2_hat[sn_sc, t, a_ctrl[sn_sc, t]])/(reps-ctrl_reps)))[0]

                ### Making sure that we at least get one element in the set (if there is no variation in Q-values Pi_epoch = [])
                if Pi_epoch.shape[0] == 0:
                    Pi_epoch = [a_ctrl[sn_sc, t]]

                ### Saving range of near-optimal policies
                Pi[sn_sc] = pd.concat([pd.DataFrame(Pi_epoch, columns=[str(t)]), Pi[sn_sc]], axis=1)

            del Q_sim, Q_hat70_next, Q_hat_rv, Q_hat_next # deleting unnecessary variables (making sure that they are not recycled from next year)

        # Evaluating near-optimal policies in all scenarios
        ptresults = [pd.DataFrame() for _ in range(sens_sc)] # initializing list of dataframes to store patient-level results by scenario (including missestimation scenarios)
        for sc in range(sens_sc): # base case and all sensitivity scenarios

            # Extracting data, risks, transition probabilities, and disutilities from appropriate scenario (index 0 in sesntivity lists is the base case)
            if sc < age_sens_len: # base case and ages 70-74
                ptdata = ptdata_list[sc]
                patientdata = patient_list[sc]
                ascvdrisk1, ascvdrisk10, tp, feasible = age_sens_list[sc]
                disutility = disut_list[0]
                if not patientdata.isnull().values.any(): # current id belongs to a patient
                    Lterm = lterm_list[sc]
                else: # added record to match shape of base case dataframe
                    Lterm = np.nan
            elif age_sens_len <= sc < (age_sens_len + risk_mult_len - 1): # risk sensitivity scenarios
                ptdata = ptdata_list[0]
                patientdata = patient_list[0]
                ascvdrisk1, ascvdrisk10, tp, feasible = risk_sens_list[sc-age_sens_len+1]
                disutility = disut_list[0]
                Lterm = lterm_list[0]
            elif (age_sens_len + risk_mult_len - 1) <= sc < (age_sens_len + risk_mult_len + trt_ben_mult_len - 2): # treatment benefit scenarios
                ptdata = ptdata_list[0]
                patientdata = patient_list[0]
                ascvdrisk1, ascvdrisk10, _, _ = age_sens_list[0]
                tp, feasible = trt_ben_sens_list[sc-age_sens_len-risk_mult_len+2]
                disutility = disut_list[0]
                Lterm = lterm_list[0]
            elif (age_sens_len + risk_mult_len + trt_ben_mult_len - 2) <= sc < (age_sens_len + risk_mult_len + trt_ben_mult_len + disut_len - 3): # disutility scenarios
                ptdata = ptdata_list[0]
                patientdata = patient_list[0]
                ascvdrisk1, ascvdrisk10, tp, feasible = age_sens_list[0]
                disutility = disut_list[sc-age_sens_len-risk_mult_len-trt_ben_mult_len+3]
                Lterm = lterm_list[0]
            elif (age_sens_len + risk_mult_len + trt_ben_mult_len + disut_len - 3) <= sc < (age_sens_len + risk_mult_len + trt_ben_mult_len + disut_len + known_dist_len + future_action_len - 5): # known distribution and future action scenarios
                ptdata = ptdata_list[0]
                patientdata = patient_list[0]
                ascvdrisk1, ascvdrisk10, tp, feasible = age_sens_list[0]
                disutility = disut_list[0]
                Lterm = lterm_list[0]
            else: # making sure all scenarios were considered
                print("Scenario", sc, "was not taken into account")
                ptdata = ptdata_list[0]
                patientdata = patient_list[0]
                ascvdrisk1, ascvdrisk10, tp, feasible = age_sens_list[0]
                disutility = disut_list[0]
                Lterm = lterm_list[0]

            if not np.isnan(Lterm): # run only if there is patient data
                # Evaluating policy from SBBI in true transition probabilities
                V_pi_sb = evaluate_pi(pi_hat[sc, ...], tp, healthy, Lterm, disutility, discount)

                # Evaluating policy from SBBI in true transition probabilities in terms of ASCVD events
                sb_evt, sb_time_evt = evaluate_events(pi_hat[sc, ...], tp, healthy, event_id) # for main analysis

                # Evaluating random near-optimal policies from range in true transition probabilities
                ## Sampling median number of medications (rounding up, if necessary) at each year
                pi_med_range = np.full(years, np.nan)  # stores median action in range
                for t in range(Pi[sc].shape[1]):
                    if Pi[sc].iloc[:, t].shape[0] > 0:  # Making sure that there are elements in the range of actions
                        pi_med_range[t] = np.ceil(Pi[sc][str(t)].median()).astype(int) # median action in range
                    else: # Only the control action is part of the range
                        pi_med_range[t] = a_ctrl[sc, t].astype(int)

                ## Evaluating policy from range with the median number of medications in true transition probabilities
                V_pi_med_range = evaluate_pi(pi_med_range.astype(int), tp, healthy, Lterm, disutility, discount)

                ## Evaluating policy from range with the median number of medications in true transition probabilities in terms of ASCVD events
                med_range_evt, med_range_time_evt = evaluate_events(pi_med_range.astype(int), tp, healthy, event_id)

                # Evaluating worst near-optimal policies from range in true transition probabilities
                ## Extracting estimates of Q-values in ranges (on healthy state only)
                Q_hat_pi = pd.DataFrame()
                for t in range(years):
                    Q_hat_pi = pd.concat([Q_hat_pi, pd.DataFrame(Q_hat[sc, t, np.array(Pi[sc].iloc[:, t].dropna()).astype(int)])],
                                         axis=1)

                ## Sampling fewest number of medications at each year
                pi_fewest_range = np.full(years, np.nan)  # stores smallest action in range
                for t in range(Pi[sc].shape[1]):
                    if Pi[sc].iloc[:, t].shape[0] > 0:  # Making sure that there are elements in the range of actions
                        pi_fewest_range[t] = Pi[sc][str(t)].min().astype(int) # sampling smallest action
                    else: # Only the control action is part of the range
                        pi_fewest_range[t] = a_ctrl[sc, t]

                ## Evaluating policy from range with the fewest number of medications in true transition probabilities
                V_pi_fewest_range = evaluate_pi(pi_fewest_range.astype(int), tp, healthy, Lterm, disutility, discount)

                ## Evaluating policy from range with the fewest number of medications in true transition probabilities in terms of ASCVD events
                fewest_range_evt, fewest_range_time_evt = evaluate_events(pi_fewest_range.astype(int), tp, healthy, event_id)

                # Creating data frame of range of medications
                Pi_meds[sc] = pd.DataFrame(np.select([Pi[sc].isna()] + [Pi[sc] == x for x in range(numtrt)], np.append(np.nan, meds)))

                # Calculating policies according to 2017 AHA's guidelines
                aha = aha_guideline(ascvdrisk10, patientdata.sbp.values, patientdata.dbp.values,
                                    targetrisk, targetsbp, targetdbp, sbpmin, dbpmin,
                                    sbp_reduction, dbp_reduction, rel_risk_chd, rel_risk_stroke)

                # Calculating policies using backwards induction
                Q, V, pi = backwards_induction(tp, healthy, Lterm, disutility, discount, feasible)

                # Evaluating no treatment in true transition probabilities
                V_no_trt = evaluate_pi(np.zeros(years, dtype=int), tp, healthy, Lterm, disutility, discount)

                # Evaluating policy from AHA's guidelines in true transition probabilities
                V_pi_aha = evaluate_pi(aha.astype(int), tp, healthy, Lterm, disutility, discount)

                # Evaluating no treatmentin true transition probabilities in terms of ASCVD events
                notrt_evt, notrt_time_evt = evaluate_events(np.zeros(years, dtype=int), tp, healthy, event_id)

                # Evaluating policy from AHA's guidelines in true transition probabilities in terms of ASCVD events
                aha_evt, aha_time_evt = evaluate_events(aha.astype(int), tp, healthy, event_id)

                # Evaluating optimal policy in true transition probabilities in terms of ASCVD events
                bi_evt, bi_time_evt = evaluate_events(pi.astype(int), tp, healthy, event_id)

                ## Data frame of results for a single patient (single result per patient-year)
                ptresults[sc] = pd.concat([pd.Series(np.repeat(i, years), name='id'),
                                           pd.Series(np.arange(years), name='year'),

                                           pd.Series(V_no_trt, name='V_notrt'), # patient's true value functions for no treatment
                                           pd.Series(V_pi_aha, name='V_aha'), # patient's true value functions under AHA's guidelines
                                           pd.Series(V, name='V_opt'), # patient's true optimal value functions
                                           pd.Series(V_pi_sb, name='V_apr'), # patient's true value functions under approximately optimal policies
                                           pd.Series(V_pi_med_range, name='V_med_range'), # patient's true value functions using a random treatment option in range
                                           pd.Series(V_pi_fewest_range, name='V_fewest_range'), # patient's true value functions using worst treatment option in range

                                           pd.Series(aha, name='pi_aha'), # patient's policy according to AHA's guidelines
                                           pd.Series(pi, name='pi_opt'), # patient's true optimal policy
                                           pd.Series(pi_hat[sc, ...], name='pi_apr'), # patient's approximately optimal policy
                                           pd.Series(pi_med_range, name='pi_med_range'), # patient's policy from a random treatment option in range
                                           pd.Series(pi_fewest_range, name='pi_fewest_range'), # patient's policy from the worst treatment option in range
                                           pd.Series(a_ctrl[sc, ...], name='pi_ctrl'), # patient's control actions

                                           pd.Series(notrt_evt, name='evt_notrt'), # expected number of events under no treatment
                                           pd.Series(aha_evt, name='evt_aha'), # expected number of events under AHA's guideline
                                           pd.Series(bi_evt, name='evt_opt'), # expected number of events under optimal policy
                                           pd.Series(sb_evt, name='evt_apr'), # expected number of events under approximately optimal policy
                                           pd.Series(med_range_evt, name='evt_med_range'), # expected number of events using a random treatment option in range
                                           pd.Series(fewest_range_evt, name='evt_fewest_range'), # expected number of events using worst treatment option in range

                                           pd.Series(notrt_time_evt, name='time_evt_notrt'), # expected years until an adverse event (including non-ASCVD related death) under no treatment
                                           pd.Series(aha_time_evt, name='time_evt_aha'), # expected years until an adverse event (including non-ASCVD related death) under AHA's guideline
                                           pd.Series(bi_time_evt, name='time_evt_opt'), # expected years until an adverse event (including non-ASCVD related death) under optimal policy
                                           pd.Series(sb_time_evt, name='time_evt_apr'), # expected years until an adverse event (including non-ASCVD related death) under approximately optimal policy
                                           pd.Series(med_range_time_evt, name='time_evt_med_range'), # expected years until an adverse event (including non-ASCVD related death) using a random treatment option in range
                                           pd.Series(fewest_range_time_evt, name='time_evt_fewest_range') # expected years until an adverse event (including non-ASCVD related death) using worst treatment option in range

                                           ], axis=1)

            else: # otherwise, add NaN sub-dataframe with appropriate id
                colnames = ['id', 'year',
                            'V_notrt', 'V_aha', 'V_opt', 'V_apr', 'V_med_range', 'V_fewest_range',
                            'pi_aha', 'pi_opt', 'pi_apr', 'pi_med_range', 'pi_fewest_range', 'pi_ctrl',
                            'evt_notrt', 'evt_aha', 'evt_opt', 'evt_apr', 'evt_med_range', 'evt_fewest_range',
                            'time_evt_notrt', 'time_evt_aha', 'time_evt_opt', 'time_evt_apr', 'time_evt_med_range', 'time_evt_fewest_range'
                            ]
                ptresults[sc] = pd.DataFrame(np.full((years, len(colnames)), np.nan), columns=colnames)
                ptresults[sc]['id'] = pd.Series(np.repeat(i, years), name='id')

            # Extracting sampling weights from original dataset
            wts = pd.DataFrame(ptdata.iloc[list(np.where([j in list(ptresults[sc].id.unique()) for j in list(ptdata.id)])[0]),
                                           [ptdata.columns.get_loc(col) for col in ["id", "wt"]]], columns=["id", "wt"])
            wts.reset_index(drop=True, inplace=True)

            # Adding sampling weights to final results
            ptresults[sc] = pd.concat([ptresults[sc].iloc[:, [ptresults[sc].columns.get_loc(col) for col in ["id", "year"]]], wts["wt"],
                                       ptresults[sc].iloc[:, [ptresults[sc].columns.get_loc(col)
                                                              for col in ptresults[sc].columns.difference(["id", "year"])]]],
                                      axis=1)

            # Merging single patient data in data frame with data from of all patients
            pt_sim[sc] = pd.concat([pt_sim[sc], ptresults[sc]], ignore_index=True)
            ptresults[sc] = np.nan # making sure values are not recycled

            # Saving patient-level results (for healthy state only)
            ## List of results (multiple results per patient-year)
            # risk1[sc].append(ascvdrisk1)  # patient's 1-year risk calculations (save only for debugging purposes)
            # risk10[sc].append(ascvdrisk10)  # patient's 10-year risk calculations (save only for debugging purposes)
            # transitions[sc].append(tp)  # patient's transition probabilities (save only for debugging purposes)
            bi_Q[sc].append(Q) # patient's true Q-values
            sb_Q[sc].append(Q_hat[sc, ...])  # patient's estimates of Q-values
            sb_sigma2[sc].append(sigma2_hat[sc, ...]) # varianve of average Q-values per replication
            action_range[sc].append(Pi[sc])  # patient's ranges of near-optimal actions (for healthy state)
            medication_range[sc].append(Pi_meds[sc])  # patient's ranges of near-optimal treatment choices (for healthy state)

        # Saving all results (saving each time a patient is evaluated)
        os.chdir(results_dir) # chaging to results directory
        with open('Results for patients ' + str(min(id_seq)) + ' to ' + str(max(id_seq)) + ' until patient ' + str(i) +
                  ' using adaptive observations and ' + str(reps) + ' batches.pkl', 'wb') as f: # full simulation
            pk.dump([#risk1, risk10, transitions,
                     bi_Q, sb_Q, sb_sigma2, action_range, medication_range,
                     pt_sim], f, protocol=3)
        # # with open('Number of batches analysis for '+str(i+1)+' patients.pkl', 'wb') as f:  # number of batches analysis
        # #     pk.dump(ci_width_list, f, protocol=3)
        os.chdir(home_dir) # returning to home directory
