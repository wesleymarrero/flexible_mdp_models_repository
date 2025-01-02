# ======================================================
# Calculating reductions in BP and risk due to treatment
# ======================================================

# Loading modules
import numpy as np
import itertools as it  # recursive operations

# ---------------------------------------------------------------
# Using parameters in Sundstrom et al. 2014 and 2015 (base case)
# ---------------------------------------------------------------

# Drug combination function (sorted in non-decreasing order of actions -
# sorted by absolute risk reduction with ties broken with their estimated disutility)
def med_effects(hf_red_frac, sbp_drop_std, sbp_drop_hf, dbp_drop_std, dbp_drop_hf, rel_risk_chd_std, rel_risk_chd_hf,
                rel_risk_stroke_std, rel_risk_stroke_hf, disut_std, disut_hf, numtrt, order):

    # Storing reductions and disutilities due to medications
    sbp_reduction = np.empty(numtrt); dbp_reduction = np.empty(numtrt); rel_risk_chd = np.empty(numtrt)
    rel_risk_stroke = np.empty(numtrt); trtharm = np.empty(numtrt); meds = np.empty(numtrt)

    for trt in range(numtrt):
        if trt == 0:  # no treatment
            sbp_reduction[trt] = 0
            dbp_reduction[trt] = 0
            rel_risk_chd[trt] = 1
            rel_risk_stroke[trt] = 1
            trtharm[trt] = 0
            meds[trt] = 0
        elif trt == 1:  # 1 half
            sbp_reduction[trt] = sbp_drop_hf
            dbp_reduction[trt] = dbp_drop_hf
            rel_risk_chd[trt] = rel_risk_chd_hf
            rel_risk_stroke[trt] = rel_risk_stroke_hf
            trtharm[trt] = disut_hf
            meds[trt] = hf_red_frac
        elif trt == 2:  # 1 standard
            sbp_reduction[trt] = sbp_drop_std
            dbp_reduction[trt] = dbp_drop_std
            rel_risk_chd[trt] = rel_risk_chd_std
            rel_risk_stroke[trt] = rel_risk_stroke_std
            trtharm[trt] = disut_std
            meds[trt] = 1
        elif trt == 3:  # 2 half
            sbp_reduction[trt] = sbp_drop_hf*2
            dbp_reduction[trt] = dbp_drop_hf*2
            rel_risk_chd[trt] = rel_risk_chd_hf**2
            rel_risk_stroke[trt] = rel_risk_stroke_hf**2
            trtharm[trt] = disut_hf*2
            meds[trt] = hf_red_frac*2
        elif trt == 4:  # 1 standard, 1 half
            sbp_reduction[trt] = sbp_drop_std + sbp_drop_hf
            dbp_reduction[trt] = dbp_drop_std + dbp_drop_hf
            rel_risk_chd[trt] = rel_risk_chd_std*rel_risk_chd_hf
            rel_risk_stroke[trt] = rel_risk_stroke_std*rel_risk_stroke_hf
            trtharm[trt] = disut_std + disut_hf
            meds[trt] = 1 + hf_red_frac
        elif trt == 5:  # 3 half
            sbp_reduction[trt] = sbp_drop_hf*3
            dbp_reduction[trt] = dbp_drop_hf*3
            rel_risk_chd[trt] = rel_risk_chd_hf**3
            rel_risk_stroke[trt] = rel_risk_stroke_hf**3
            trtharm[trt] = disut_hf*3
            meds[trt] = hf_red_frac*3
        elif trt == 6:  # 2 standard
            sbp_reduction[trt] = sbp_drop_std*2
            dbp_reduction[trt] = dbp_drop_std*2
            rel_risk_chd[trt] = rel_risk_chd_std**2
            rel_risk_stroke[trt] = rel_risk_stroke_std**2
            trtharm[trt] = disut_std*2
            meds[trt] = 2
        elif trt == 7:  # 1 standard, 2 half
            sbp_reduction[trt] = sbp_drop_std + sbp_drop_hf*2
            dbp_reduction[trt] = dbp_drop_std + dbp_drop_hf*2
            rel_risk_chd[trt] = rel_risk_chd_std*(rel_risk_chd_hf**2)
            rel_risk_stroke[trt] = rel_risk_stroke_std*(rel_risk_stroke_hf**2)
            trtharm[trt] = disut_std + disut_hf*2
            meds[trt] = 1 + hf_red_frac*2
        elif trt == 8:  # 4 half
            sbp_reduction[trt] = sbp_drop_hf*4
            dbp_reduction[trt] = dbp_drop_hf*4
            rel_risk_chd[trt] = rel_risk_chd_hf**4
            rel_risk_stroke[trt] = rel_risk_stroke_hf**4
            trtharm[trt] = disut_hf*4
            meds[trt] = hf_red_frac*4
        elif trt == 9:  # 2 standard, 1 half
            sbp_reduction[trt] = sbp_drop_std*2 + sbp_drop_hf
            dbp_reduction[trt] = dbp_drop_std*2 + dbp_drop_hf
            rel_risk_chd[trt] = (rel_risk_chd_std**2)*rel_risk_chd_hf
            rel_risk_stroke[trt] = (rel_risk_stroke_std**2)*rel_risk_stroke_hf
            trtharm[trt] = disut_std*2 + disut_hf
            meds[trt] = 2 + hf_red_frac
        elif trt == 10:  # 1 standard, 3 half
            sbp_reduction[trt] = sbp_drop_std + sbp_drop_hf*3
            dbp_reduction[trt] = dbp_drop_std + dbp_drop_hf*3
            rel_risk_chd[trt] = rel_risk_chd_std*(rel_risk_chd_hf**3)
            rel_risk_stroke[trt] = rel_risk_stroke_std*(rel_risk_stroke_hf**3)
            trtharm[trt] = disut_std + disut_hf*3
            meds[trt] = 1 + hf_red_frac*3
        elif trt == 11:  # 3 standard
            sbp_reduction[trt] = sbp_drop_std*3
            dbp_reduction[trt] = dbp_drop_std*3
            rel_risk_chd[trt] = rel_risk_chd_std**3
            rel_risk_stroke[trt] = rel_risk_stroke_std**3
            trtharm[trt] = disut_std*3
            meds[trt] = 3
        elif trt == 12:  # 5 half
            sbp_reduction[trt] = sbp_drop_hf*5
            dbp_reduction[trt] = dbp_drop_hf*5
            rel_risk_chd[trt] = rel_risk_chd_hf**5
            rel_risk_stroke[trt] = rel_risk_stroke_hf**5
            trtharm[trt] = disut_hf*5
            meds[trt] = hf_red_frac*5
        elif trt == 13:  # 2 standard, 2 half
            sbp_reduction[trt] = sbp_drop_std*2 + sbp_drop_hf*2
            dbp_reduction[trt] = dbp_drop_std*2 + dbp_drop_hf*2
            rel_risk_chd[trt] = (rel_risk_chd_std**2)*(rel_risk_chd_hf**2)
            rel_risk_stroke[trt] = (rel_risk_stroke_std**2)*(rel_risk_stroke_hf**2)
            trtharm[trt] = disut_std*2 + disut_hf*2
            meds[trt] = 2 + hf_red_frac*2
        elif trt == 14:  # 1 standard, 4 half
            sbp_reduction[trt] = sbp_drop_std + sbp_drop_hf*4
            dbp_reduction[trt] = dbp_drop_std + dbp_drop_hf*4
            rel_risk_chd[trt] = rel_risk_chd_std*(rel_risk_chd_hf**4)
            rel_risk_stroke[trt] = rel_risk_stroke_std*(rel_risk_stroke_hf**4)
            trtharm[trt] = disut_std + disut_hf*4
            meds[trt] = 1 + hf_red_frac*4
        elif trt == 15:  # 3 standard, 1 half
            sbp_reduction[trt] = sbp_drop_std*3 + sbp_drop_hf
            dbp_reduction[trt] = dbp_drop_std*3 + dbp_drop_hf
            rel_risk_chd[trt] = (rel_risk_chd_std**3)*rel_risk_chd_hf
            rel_risk_stroke[trt] = (rel_risk_stroke_std**3)*rel_risk_stroke_hf
            trtharm[trt] = disut_std*3 + disut_hf
            meds[trt] = 3 + hf_red_frac
        elif trt == 16:  # 2 standard, 3 half
            sbp_reduction[trt] = sbp_drop_std*2 + sbp_drop_hf*3
            dbp_reduction[trt] = dbp_drop_std*2 + dbp_drop_hf*3
            rel_risk_chd[trt] = (rel_risk_chd_std**2)*(rel_risk_chd_hf**3)
            rel_risk_stroke[trt] = (rel_risk_stroke_std**2)*(rel_risk_stroke_hf**3)
            trtharm[trt] = disut_std*2 + disut_hf*3
            meds[trt] = 2 + hf_red_frac*3
        elif trt == 17:  # 4 standard
            sbp_reduction[trt] = sbp_drop_std*4
            dbp_reduction[trt] = dbp_drop_std*4
            rel_risk_chd[trt] = rel_risk_chd_std**4
            rel_risk_stroke[trt] = rel_risk_stroke_std**4
            trtharm[trt] = disut_std*4
            meds[trt] = 4
        elif trt == 18:  # 3 standard, 2 half
            sbp_reduction[trt] = sbp_drop_std*3 + sbp_drop_hf*2
            dbp_reduction[trt] = dbp_drop_std*3 + dbp_drop_hf*2
            rel_risk_chd[trt] = (rel_risk_chd_std**3)*(rel_risk_chd_hf**2)
            rel_risk_stroke[trt] = (rel_risk_stroke_std**3)*(rel_risk_stroke_hf**2)
            trtharm[trt] = disut_std*3 + disut_hf*2
            meds[trt] = 3 + hf_red_frac*2
        elif trt == 19:  # 4 standard, 1 half
            sbp_reduction[trt] = sbp_drop_std*4 + sbp_drop_hf
            dbp_reduction[trt] = dbp_drop_std*4 + dbp_drop_hf
            rel_risk_chd[trt] = (rel_risk_chd_std**4)*rel_risk_chd_hf
            rel_risk_stroke[trt] = (rel_risk_stroke_std**4)*rel_risk_stroke_hf
            trtharm[trt] = disut_std*4 + disut_hf
            meds[trt] = 4 + hf_red_frac
        elif trt == 20:  # 5 standard
            sbp_reduction[trt] = sbp_drop_std*5
            dbp_reduction[trt] = dbp_drop_std*5
            rel_risk_chd[trt] = rel_risk_chd_std**5
            rel_risk_stroke[trt] = rel_risk_stroke_std**5
            trtharm[trt] = disut_std*5
            meds[trt] = 5

        # print([trt, np.round(sbp_reduction[trt], 2), np.round(dbp_reduction[trt], 2),
        #        np.round(1-rel_risk_chd[trt], 2), np.round(1-rel_risk_stroke[trt], 2),
        #        np.round(trtharm[trt], 3), frac.Fraction(meds[trt]).limit_denominator(3)])

    if order == "nonincreasing": # sorting actions in non-increasing order
        sbp_reduction, dbp_reduction = np.array(list(reversed(sbp_reduction))), np.array(list(reversed(dbp_reduction)))
        rel_risk_chd, rel_risk_stroke = np.array(list(reversed(rel_risk_chd))), np.array(list(reversed(rel_risk_stroke)))
        trtharm, meds = np.array(list(reversed(trtharm))), np.array(list(reversed(meds)))

    return sbp_reduction, dbp_reduction, rel_risk_chd, rel_risk_stroke, trtharm, meds

# ---------------------------------------------------------------
# Using parameters in Law et al. 2009 (MDP variations analysis)
# ---------------------------------------------------------------

# Function to identify treatment combinations and estimate disutility by drug type
# Note: the effect of each drug type combination  is estimated in function TP_DRUG in the transition_probabilities.py file
def drug_type_effects(drugs, disut_std, action_order):

    # Generating list of treatment options from list of drug types
    ##Two drugs simultaneously
    drugcomb2 = it.combinations_with_replacement(drugs, 2)
    drugcomb2 = np.array(list(drugcomb2))
    drugcomb2 = np.sort(drugcomb2, axis=1)
    drugcomb2 = drugcomb2[drugcomb2[:, 1].argsort(kind='mergesort')]
    drugcomb2 = drugcomb2[drugcomb2[:, 0].argsort(kind='mergesort')]

    ##Three drugs simultaneously
    drugcomb3 = np.array(np.meshgrid(drugs, drugs, drugs)).reshape(3, len(drugs) ** 3).T
    drugcomb3 = np.sort(drugcomb3, axis=1)
    drugcomb3 = np.unique(drugcomb3, axis=0)

    ##Four drugs simultaneously
    drugcomb4 = np.array(np.meshgrid(drugs, drugs, drugs, drugs)).reshape(4, len(drugs) ** 4).T
    drugcomb4 = np.sort(drugcomb4, axis=1)
    drugcomb4 = np.unique(drugcomb4, axis=0)

    ##Five drugs simultaneously (could extend to n number of drugs - limited by having to repeat the "drugs" list inside np.meshgrid [drugs for i in range(n)] doesn't work)
    drugcomb5 = np.array(np.meshgrid(drugs, drugs, drugs, drugs, drugs)).reshape(5, len(drugs) ** 5).T
    drugcomb5 = np.sort(drugcomb5, axis=1)
    drugcomb5 = np.unique(drugcomb5, axis=0)

    ###Removing potentially dangerous drug combinations (could extend to any simulataneous number of drugs)
    drugcomb2 = np.delete(drugcomb2, np.intersect1d(np.unique(np.where(drugcomb2 == "ACE")[0]), np.unique(np.where(drugcomb2 == "ARB")[0])), axis=0)
    drugcomb3 = np.delete(drugcomb3, np.intersect1d(np.unique(np.where(drugcomb3 == "ACE")[0]), np.unique(np.where(drugcomb3 == "ARB")[0])), axis=0)
    drugcomb4 = np.delete(drugcomb4, np.intersect1d(np.unique(np.where(drugcomb4 == "ACE")[0]), np.unique(np.where(drugcomb4 == "ARB")[0])), axis=0)
    drugcomb5 = np.delete(drugcomb5, np.intersect1d(np.unique(np.where(drugcomb5 == "ACE")[0]), np.unique(np.where(drugcomb5 == "ARB")[0])), axis=0)

    ##Combining all treatment choices in a list
    drugs.insert(0, "NT") # incorporating no treatment
    alldrugs = drugs + list(drugcomb2) + list(drugcomb3) + list(drugcomb4) + list(drugcomb5) # number of treatments to consider (196 trts: no treatment plus 1 to 5 drugs from 5 different types at standard dosage - excluding combinations of ACE and ARB)

    # Disutility parameters
    drugs_perdisut = [i / 100 for i in [9.9, 7.5, 3.9, 0, 8.3]]  # percentage of people showing side effects per drug in Law etl a. 2003 (same order as drugs list)
    drugs_disut = [disut_std * (1 + i) for i in drugs_perdisut]  # treatment disutility per drug type at standard drug

    ## Generating lists of disutility per drug combination
    trtharm = []
    for d in range(len(alldrugs)):

        # Making sure evaluated treatment is in a list or string format
        if type(alldrugs[d]) == str or type(alldrugs[d]) == list:
            drugcomb = alldrugs[d]
        else:
            drugcomb = list(alldrugs[d])

        # Counting number of times a drug is being given
        th = drugcomb.count('TH')
        bb = drugcomb.count('BB')
        ace = drugcomb.count('ACE')
        a2ra = drugcomb.count('ARB')
        ccb = drugcomb.count('CCB')

        # Calculating treatment harm per drug combination
        trtharm.append(th*drugs_disut[0] + bb*drugs_disut[1] + ace*drugs_disut[2] + a2ra*drugs_disut[3] + ccb*drugs_disut[4])

    # Sorting list of drugs and treatment harm according to action order
    alldrugs = np.array(alldrugs, dtype=object)[action_order]  # sorting drug combinations according to action order
    trtharm = np.array(trtharm)[action_order]  # sorting disutilities according to action order

    return alldrugs, trtharm

# Function to calculate relative risk reductions based on risk slopes from Law et al. 2009 (not parameters in Sundstrom et al. 2014 and 2015)
# Note: SBP and DBP reductions are calculated in separate scripts (sbp_reductions_drugtype.py and dbp_reductions_drugtype.py)
def new_risk(sbpreduc, riskslope, pretrtrisk, event):

    # post trt risk for each event type (0=CHD, 1=stroke)

    RR = (list(riskslope)[event])**(sbpreduc/20)
    risk = RR*pretrtrisk

    return risk

