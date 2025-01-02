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
from hypertension_treatment_sbbi_sbmcc import results_dir, fig_dir, ptdata_list, meds

# *************************************
# Algorithmic Comparison in Base MDP
# *************************************

# --------------------
# Loading data
# --------------------

# Loading results
os.chdir(results_dir)
with open('Algorithm Comparison Results - Records 0 to 1112.pkl', 'rb') as f:
    [Q_bi, Q_sbbi, Q_on_mcc, Q_off_mcc, Q_sarsa, Q_q_learn, Q_td_svp, _, _, pt_sim] = pk.load(f) # medication_range, med_set_td_svp
del _; gc.collect()

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
pt_sim = pd.merge(pt_sim, ptdata1[['id', 'age', 'sex', 'race', 'smk', 'diab', 'bp_cat']], on='id')

# Adjusting values by sampling weights
cols = ['V_notrt', 'V_opt', 'V_pi_sbbi', 'V_pi_on_mcc', 'V_pi_off_mcc', 'V_pi_sarsa', 'V_pi_q_learn',
        'V_apr', 'V_med_range', 'V_fewest_range', 'V_best_td_svp', 'V_median_td_svp', 'V_fewest_td_svp',]
pt_sim[cols] = pt_sim[cols].multiply(pt_sim.wt, axis="index")/1e06

# --------------------------------------------------------------------
# Plot of expected total QALYs saved in approximately optimal policies
# --------------------------------------------------------------------

# Data frame of expected total QALYs saved (compared to no treatment) per year and BP group
sub_cols = ['year', 'bp_cat', 'V_notrt', 'V_opt', 'V_pi_sbbi', 'V_pi_on_mcc', 'V_pi_off_mcc', 'V_pi_sarsa', 'V_pi_q_learn']
qalys_df = pt_sim.loc[:, sub_cols].groupby(['year', 'bp_cat']).sum().reset_index(drop=False, inplace=False)
qalys_df.year += 1
qalys_df[sub_cols[2:]] = qalys_df[sub_cols[2:]].subtract(qalys_df.V_notrt, axis="index")
qalys_df = qalys_df.drop(['V_notrt'], axis=1); sub_cols.remove('V_notrt')

# Preparing data for plot
meaning = ['year', 'bp_cat', 'Optimal Policy', 'SBBI', 'On-Policy MC', 'Off-Policy MC', 'Sarsa', 'Q-learning']
qalys_df = qalys_df.rename(columns=dict(zip(sub_cols, meaning)))
qalys_df = qalys_df.melt(id_vars=['year', 'bp_cat'], var_name='policy', value_name='qalys')
qalys_df['policy'] = pd.Categorical(qalys_df['policy'], categories=meaning[2:], ordered=True) # converting scenario to ordered category
qalys_df = qalys_df.sort_values(['policy', 'year']) # sorting dataframe based on selected columns
qalys_df = qalys_df[qalys_df.bp_cat!='Normal'] # removing normal BP group

# Making plot
os.chdir(fig_dir)
plot_qalys_saved(qalys_df)

# -------------------------------------------------------------
# Plot of expected total QALYs saved in set-valued policies
# -------------------------------------------------------------

# Data frame of expected total QALYs saved (compared to no treatment) per year and BP group
sub_cols = ['year', 'bp_cat', 'V_notrt', 'V_apr', 'V_best_td_svp', 'V_med_range', 'V_median_td_svp', 'V_fewest_range', 'V_fewest_td_svp']
qalys_svp_df = pt_sim.loc[:, sub_cols].groupby(['year', 'bp_cat']).sum().reset_index(drop=False, inplace=False)
qalys_svp_df.year += 1
qalys_svp_df[sub_cols[2:]] = qalys_svp_df[sub_cols[2:]].subtract(qalys_svp_df.V_notrt, axis="index")
qalys_svp_df = qalys_svp_df.drop(['V_notrt'], axis=1); sub_cols.remove('V_notrt')
qalys_svp_df.loc[qalys_svp_df.year==1, sub_cols[2:]].sum(axis=0).round(2) # result for paper

# Preparing data for plot
meaning = ['year', 'bp_cat', 'Best in SBBI-SBMCC', 'Best in TD-SVP', 'Median in SBBI-SBMCC', 'Median in TD-SVP', 'Fewest in SBBI-SBMCC', 'Fewest in TD-SVP']
qalys_svp_df = qalys_svp_df.rename(columns=dict(zip(sub_cols, meaning)))
qalys_svp_df = qalys_svp_df.melt(id_vars=['year', 'bp_cat'], var_name='policy', value_name='qalys')
qalys_svp_df['policy'] = pd.Categorical(qalys_svp_df['policy'], categories=meaning[2:], ordered=True) # converting scenario to ordered category
qalys_svp_df = qalys_svp_df.sort_values(['policy', 'year']) # sorting dataframe based on selected columns
qalys_svp_df = qalys_svp_df[qalys_svp_df.bp_cat!='Normal'] # removing normal BP group

# Making plot
os.chdir(fig_dir)
plot_qalys_saved(qalys_svp_df)

# ------------------------------------------------------------------------
# Plot of treatment distribution in set-valued policies at years 1 and 10
# ------------------------------------------------------------------------

## Converting actions to number of medications
cols = ['pi_opt', 'pi_hat_sbbi', 'pi_hat_on_mcc', 'pi_hat_off_mcc', 'pi_hat_sarsa', 'pi_hat_q_learn',
        'pi_apr', 'pi_med_range', 'pi_fewest_range', 'pi_best_td_svp', 'pi_median_td_svp', 'pi_fewest_td_svp']
new_cols = ['meds_opt', 'meds_hat_sbbi', 'meds_hat_on_mcc', 'meds_hat_off_mcc', 'meds_hat_sarsa', 'meds_hat_q_learn',
            'meds_apr', 'meds_med_range', 'meds_fewest_range', 'meds_best_td_svp', 'meds_median_td_svp', 'meds_fewest_td_svp']
tmp = pd.DataFrame.from_dict(dict(zip(new_cols, [np.select([pt_sim[c]==x for x in range(len(meds))], meds) for c in cols])))
pt_sim_meds = pd.concat([pt_sim, tmp], axis=1); del tmp

## Extracting first and last year results
pt_sim1 = pt_sim_meds.groupby(['id']).first().reset_index(drop=False, inplace=False) # first year of result data (only base case scenario)
pt_sim10 = pt_sim_meds.groupby(['id']).last().reset_index(drop=False, inplace=False) # last year of result data (only base case scenario)

## Data frame of number of medications
sub_cols = ['year', 'bp_cat', 'meds_apr', 'meds_best_td_svp', 'meds_med_range', 'meds_median_td_svp', 'meds_fewest_range', 'meds_fewest_td_svp']
trt_df = pd.concat([pt_sim1[sub_cols], pt_sim10[sub_cols]], axis=0)
del pt_sim1, pt_sim10; gc.collect()

### Preparing data for plot
meaning = ['year', 'bp_cat', 'Best in SBBI-SBMCC', 'Best in TD-SVP', 'Median in SBBI-SBMCC', 'Median in TD-SVP', 'Fewest in SBBI-SBMCC', 'Fewest in TD-SVP']
trt_df = trt_df.rename(columns=dict(zip(sub_cols, meaning)))
trt_df = trt_df.melt(id_vars=['year', 'bp_cat'], var_name='policy', value_name='meds')
trt_df.year += 1

# Sorting dataframe according to BP categories
trt_df['bp_cat'] = pd.Categorical(trt_df['bp_cat'], categories=bp_cat_labels, ordered=True) # converting scenario to ordered category
trt_df = trt_df.sort_values(['bp_cat', 'year']) # sorting dataframe based on selected columns

## Making plot
os.chdir(fig_dir)
plot_trt_dist(trt_df)

# ********************************************
# Algorithmic Comparison in MDP Variations
# ********************************************

# --------------------
# Loading data
# --------------------

# Loading results
os.chdir(results_dir)
with open('Algorithm Comparison Results in MDP Variations - Records 0 to 10.pkl', 'rb') as f:
    [pt_sim_base] = pk.load(f)

## Removing unnecesary rows in each data frame
pt_sim_base = [df.dropna() for df in pt_sim_base]

# ----------------------------------------------------------------
# Table of life years lost by each approximately optimal policy
# ----------------------------------------------------------------

# Extracting first year results
pt_sim_base1 = [df.groupby(['id']).first().reset_index(drop=False, inplace=False) for df in pt_sim_base]

## Calculating difference between approximately optimal polcies and the optimal policy and taking the average
sub_cols = ['V_pi_sbbi', 'V_pi_on_mcc', 'V_pi_off_mcc', 'V_pi_sarsa', 'V_pi_q_learn'] # column names of policies of interest
avg = pd.DataFrame() # empty data frame to store results
for df in pt_sim_base1:
    df[sub_cols] = df[sub_cols].subtract(df.V_opt, axis="index").multiply(-1, axis="index")
    avg = pd.concat([avg, df[sub_cols].mean(axis=0)], axis=1, ignore_index=True)

# Summary table
avg = avg.T
# avg.to_clipboard()

# ----------------------------------------------------------------------
# Table of life years saved by each strategy in the set-based policies
# ----------------------------------------------------------------------

# Extracting first year results
pt_sim_base1 = [df.groupby(['id']).first().reset_index(drop=False, inplace=False) for df in pt_sim_base]

## Calculating difference between approximately optimal polcies and the optimal policy and taking the average
sub_cols = ['V_best_sbbi_sbmcc', 'V_median_sbbi_sbmcc', 'V_fewest_sbbi_sbmcc',
            'V_best_td_svp', 'V_median_td_svp', 'V_fewest_td_svp'] # column names of policies of interest
avg_svp = pd.DataFrame() # empty data frame to store results
for df in pt_sim_base1:
    avg_svp = pd.concat([avg_svp, df[sub_cols].mean(axis=0)], axis=1, ignore_index=True)

# Summary table
avg_svp = avg_svp.T
# avg_svp.to_clipboard()
