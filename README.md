# flexible_mdp_models_repository

This repository includes the data sets and code used in ***Ranges of Near-Optimal Choices for Personalized Hypertension Treatment Plans***. The repository contains a directory where the data is stored and 22 Python scripts used to conduct the analyses in the paper.

## I. Data
This directory includes the input files and pre-processing in an R script for the analyses conducted in the paper. It contains the following files:
1. **lifedata.csv** - this file contains the life expectancy for adults in the USA with ages 40 to 100
2. **strokedeathdata.csv** - this file contains the likelihood of death due to a stroke for adults in the USA with ages 40 to 100
3. **chddeathdata.csv** - this file contains the likelihood of death due to a coronary heart disease (CHD) event (i.e., heart attack) for adults in the USA with ages 40 to 100
4. **alldeathdata.csv** - this file contains the likelihood of death not related to atherosclerotic cardiovascular disease (ASCVD) for adults in the USA with ages 40 to 100
5. **riskslopes.csv** - this file contains blood pressure (BP) reductions using parameters in Law et al. 2003 and 2009
6. **Continuous NHANES** - this sub-directory contains the following files:
    - **Continuous_NHANES_Data_Extraction_and_Filtering.R** - this R script contains the code to obtain the final patient datasets from the 2009-2016 files publicly available at: https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx
    - **Continuous NHANES 50-54 Dataset.csv** - this file is the end result of the Continuous_NHANES_Data_Extraction_and_Filtering.R script, which was used as the patient dataset in the main analyses for ages 50-54.
	- **Continuous NHANES 70-74 Dataset.csv** - this file is the end result of the Continuous_NHANES_Data_Extraction_and_Filtering.R script, which was used as the patient dataset in the sensitivity analysis for ages 70-74.
	- **Continuous NHANES 50-54 Dataset Until 73.csv** - this file is the end result of the Continuous_NHANES_Data_Extraction_and_Filtering.R script, which was used as the patient dataset in the algorthmic comparison scenario that considers the possibility of patients' developing diabetes.

## II. Code
### A. Analysis Scripts
1. **hypertension_treatment_sbbi_sbmcc.py** - this Python script serves as the **master file** for the main anlyses. Interested readers are encouraged to start with this file. The script exeuctes the main and sensitivity analyses, except for the algorithmic comparisons. 
2. **bp_med_effects.py** - this script estimates reductions in BP and risk due to treatment. 
3. **ascvd_risk.py** - this script estimates the risk for ASCVD event for each patient at every state and decision epoch.
4. **transition_probabilities.py** - this script estimates the transition probabilities for each patient included in the analysis.
5. **sb_bi.py** - this script executes the simulation-based backward induction (SBBI) algorithm.
6. **sb_mcc.py** - this script executes the simulation-based multiple comparison with a control (SBMCC) algorithm.
7. **policy_evaluation.py** - this script evaluates a policy in terms of expected quality-adjusted life years (QALYs) and expected ASCVD events
8. **aha_2017_guideline.py** - this script mimics the treatment recommendations from the 2017 Hypertension Clinical Practice Guidelines
9. **backwards_induction_mdp.py** - this script obtains the (true) optimal policy using the standard backward indusction algorithm. 

### B.Algorithmic Comprarison Scripts
10. **algorithmic_comparison.py** - this script compares the performance of SBBI to common Monte Carlo and temporal difference methods (i.e., on-policy Monte Carlo control, off-policy Monte Carlo control, Q-learning, Sarsa) as well as the combination of SBBI and SBMCC to TD-SVP by Tang and coauthors in the main case study population and MDP.  
11. **algorithmic_comparison_mdp_variations.py** - this script compares the performance of SBBI to common Monte Carlo and temporal difference methods (i.e., on-policy Monte Carlo control, off-policy Monte Carlo control, Q-learning, Sarsa) as well as the combination of SBBI and SBMCC to TD-SVP by Tang and coauthors in a subset of the main case study population under MDP variations.
12. **diab_risk.py** - this script estimates the risk for type II diabetes for each patient at every decision epoch.
13. **sbp_reductions_drugtype.py** - this script estimates the systolic blood pressure (SBP) reductions for each drug type.
14. **dbp_reductions_drugtype.py** - this script estimates the diastolic blood pressure (DBP) reductions for each drug type.
15. **algorithms_mc_td.py** - this script executes our algorithms as well as on-policy Monte Carlo control, off-policy Monte Carlo control, Q-learning, Sarsa, and TD-SVP in the main algorithmic comparison. 
16. **algorithms_mdp_variations.py** - this script executes an adjusted version of our algorithms as well as on-policy Monte Carlo control, off-policy Monte Carlo control, Q-learning, Sarsa, and TD-SVP for the MDP variations.

### C. Renal Sympathetic Denervation Scripts
17. **renal_denervation_analysis.py** - this script executes the evaluation of adding renal sympathetic denervation at the first year of the study to illustrate Remark 1 in the paper.
	
*Please note that the scripts were listed in the order they are used in the analyses, starting from the master file **hypertension_treatment_sbbi_sbmcc.py**.*

### D. Result Scripts
18. **case_study_results.py** - this script produces most of the outputs (except for algorithmic comparison and renal denervation results) included in the paper
19. **case_study_plots.py** - this script includes the functions of the plots generated in the case_study_results.py file.
20. **algorithmic_comparison_results.py** - this script produces all the outputs for the algorithmic comparison.
21. **algorithmic_comparison_plots.py** - this script includes the functions of the plots generated in the algorithmic_comparison_results.py file.
22. **renal_denervation_results.py** - this script produces all the outputs for the renal sympathetic denervation analysis.

*Please note that the scripts were listed in the order they are used in the examination of results, starting from **case_study_results.py**.* 

The following package versions were used to execute the Pyton scripts:
- NumPy 1.26.2
- pandas 1.5.3
- matplotlib 3.8.2
- seaborn 0.13.0