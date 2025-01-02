# =======================
# Summarizing Results
# =======================

# Loading modules
import os  # directory changes
import pandas as pd  # data frame operations
import numpy as np  # matrix operations
import pickle as pk  # saving results
import gc # garbage collector
from algorithmic_comparison_plots import plot_qalys_saved, plot_trt_dist

# Importing parameters from main module
from hypertension_treatment_sbbi_sbmcc import results_dir, fig_dir, ptdata_list, meds, numtrt

# --------------------
# Loading data
# --------------------

# Loading results (if previously combined)
os.chdir(results_dir)
with open('Renal Sympathetic Denervation Results at Year 1 - Records 0 to 1112.pkl', 'rb') as f:
    [feasibility1, obs_list, Q_sbbi1, var_sbbi1, sb_Q1, sb_sigma21, pt_sim1] = pk.load(f)

# # Loading results (if not previously combined)
# os.chdir(results_dir)
# with open('Renal Sympathetic Denervation Results - Records 0 to 1112.pkl',
#           'rb') as f:
#     [feasibility_list, obs_list, Q_sbbi, var_sbbi, pt_sim] = pk.load(f)
#
# ## Loading results of main analysis (base case only)
# os.chdir(results_dir)
# with open('Results for patients 0 to 1112 until patient 1112 using adaptive observations and 301 batches.pkl',
#           'rb') as f:
#     [_, sb_Q, sb_sigma2, _, _, pt_sim_main] = pk.load(f)
# sb_Q = sb_Q[0]; sb_sigma2 = sb_sigma2[0]; pt_sim_main = pt_sim_main[0]; del _
#
# ### Incorporating results SBBI-SBMCC from main analysis
# cols = ['id', 'year', 'V_apr', 'pi_apr']
# pt_sim = pd.merge(pt_sim, pt_sim_main[cols], on=['id', 'year']); del pt_sim_main
#
# ## Subsetting first year of data (RSD was only considered in year 0)
# pt_sim1 = pt_sim.groupby(['id']).first().reset_index(drop=False, inplace=False) # first year of result data (only base case scenario)
# Q_sbbi1, var_sbbi1, sb_Q1, sb_sigma21 = [[ar[0, :] for ar in lst] for lst in [Q_sbbi, var_sbbi, sb_Q, sb_sigma2]]
# feasibility1 = [feas[0] for feas in feasibility_list]
#
# ## Saving combined results
# os.chdir(results_dir)
# with open('Renal Sympathetic Denervation Results at Year 1 - Records 0 to 1112.pkl', 'wb') as f:
#     pk.dump([feasibility1, obs_list, Q_sbbi1, var_sbbi1, sb_Q1, sb_sigma21, pt_sim1], f, protocol=3)

# --------------------
# Data Preparation
# --------------------

# Incorporating demographic and grouping information
## Extracting first year data
ptdata1 = ptdata_list[0].groupby('id').first().reset_index(drop=False, inplace=False) # extracting first year data
del ptdata_list; gc.collect()

## Identifying BP categories
bp_cats = [(ptdata1.sbp < 120) & (ptdata1.dbp < 80),
            (ptdata1.sbp >= 120) & (ptdata1.dbp < 80) & (ptdata1.sbp < 130),
            ((ptdata1.sbp >= 130) | (ptdata1.dbp >= 80)) & ((ptdata1.sbp < 140) | (ptdata1.dbp < 90)),
            (ptdata1.sbp >= 140) | (ptdata1.dbp >= 90)]
bp_cat_labels = ['Normal', 'Elevated', 'Stage 1', 'Stage 2']
ptdata1['bp_cat'] = np.select(bp_cats, bp_cat_labels)

## Incorporating information to results
pt_sim1 = pd.merge(pt_sim1, ptdata1[['id', 'bp_cat']], on='id')

# -------------------------------------------------------------------------------------------------
# Identifying Patients for whom RSD should not be considered at the first year of the planning horizon
# -------------------------------------------------------------------------------------------------

# Evaluating inequality in Remark 1 for each patient
ind_ineq = np.full(pt_sim1.shape[0], np.nan) # list to store whether inequality is satisfied for RSN
for i in range(pt_sim1.shape[0]):

    # Identifying feasible actions including RSN
    feas_tmp = np.array(feasibility1[i])
    feas_tmp = feas_tmp[np.where(feas_tmp >= numtrt)]
    obs = obs_list[i]

    # Identifying patient for whom RSN does not need to be considered
    ind_ineq[i] = int(np.where(np.any(Q_sbbi1[i][pt_sim1.pi_hat_sbbi[i]]-Q_sbbi1[i][feas_tmp] >
                                      np.sqrt((obs**(-1/2))*(var_sbbi1[i][pt_sim1.pi_hat_sbbi[i]] + var_sbbi1[i][feas_tmp]))), 0, 1))

# Creating dataframe of proportion of patients for whom RSN should be considered
ind_ineq = ind_ineq*pt_sim1.wt
ind_ineq = pd.Series(ind_ineq*pt_sim1.wt, name="RSN")

prop_df = pd.concat([pt_sim1.loc[:, ['bp_cat', 'wt']], ind_ineq], axis=1).\
    groupby(['bp_cat']).sum().reset_index(drop=False, inplace=False)

## Calculating proportions
prop_df['prop'] = prop_df['RSN']/prop_df['wt']
