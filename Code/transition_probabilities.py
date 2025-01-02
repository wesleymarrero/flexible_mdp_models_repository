# ====================================
# Estimating transition probabilities
# ====================================

# Loading modules
import numpy as np
from sbp_reductions_drugtype import sbp_reductions
from dbp_reductions_drugtype import dbp_reductions
from bp_med_effects import new_risk

# Transition probabilities function
def TP(periodrisk, chddeath, strokedeath, alldeath, pretrtsbp, pretrtdbp, sbpmin, sbpmax, dbpmin,
       sbp_reduction, dbp_reduction, rel_risk_chd, rel_risk_stroke, numhealth):
    # Calculating probability of health states transitions given ASCVD risk, fatality likelihoods,
    # and probability of mortality due to non-ASCVD events

    # Inputs
    ## periodrisk: 1-year risk of CHD and stroke
    ## chddeath: likelihood of death given a CHD events
    ## strokedeath: likelihood of death given a stroke events
    ## alldeath: likelihood of death due to non-ASCVD events
    ## trtvector: vector of treatments to consider
    ## pretrtsbp: pre-treatment SBP
    ## pretrtdbp: pre-treatment DBP
    ## sbpmin (sbpmax): Minimum (maximum) SBP allowed (clinical constraint)
    ## dbpmin: minimum DBP allowed (clinical constraint)

    # Extracting parameters
    years = periodrisk.shape[0]  # number of non-stationary stages
    events = periodrisk.shape[1]  # number of events
    numtrt = len(sbp_reduction) # number of treatment choices

    # Storing feasibility indicators
    feasible = np.full((years, numtrt), np.nan) # indicators of whether the treatment is feasible at each time

    # Storing risk and TP calculations
    risk = np.full((years, events, numtrt), np.nan)  # stores post-treatment risks
    ptrans = np.zeros((numhealth, years, numtrt))  # state transition probabilities (default of 0, to reduce computations)

    for t in range(years):  # each stage (year)
        for j in range(numtrt):  # each treatment
            if j == 0:  # the do nothing treatment
                if pretrtsbp[t] > sbpmax:  # cannot do nothing when pre-treatment SBP is too high
                    feasible[t, j] = 0
                else:
                    feasible[t, j] = 1  # otherwise, do nothing is always feasible
            else:  # prescibe >0 drugs
                newsbp = pretrtsbp[t] - sbp_reduction[j]
                newdbp = pretrtdbp[t] - dbp_reduction[j]

                # Violates minimum allowable SBP constraint or min allowable DBP constrain and
                # the do nothing option is feasible
                if (newsbp < sbpmin or newdbp < dbpmin) and feasible[t, 0] == 1:
                    feasible[t, j] = 0
                else:  # does not violate min BP
                    feasible[t, j] = 1

            for k in range(events):  # each event type
                # Calculating post-treatment risks
                if k == 0: #CHD events
                    risk[t, k, j] = rel_risk_chd[j]*periodrisk[t, k]
                elif k == 1: #stroke events
                    risk[t, k, j] = rel_risk_stroke[j]*periodrisk[t, k]

            # Health condition transition probabilities
            quits = 0
            while quits == 0:  # compute transition probabilities, using a "break" command if you've exceeded 1 (this never happens!)
                ptrans[3, t, j] = min(1, chddeath.iloc[t] * risk[t, 0, j])  # likelihood of death from CHD event
                cumulprob = ptrans[3, t, j]

                ptrans[4, t, j] = min(1, strokedeath.iloc[t] * risk[t, 1, j])  # likelihood of death from stroke
                if cumulprob + ptrans[4, t, j] >= 1:  # check for invalid probabilities
                    ptrans[4, t, j] = 1 - cumulprob
                    break  # all other probabilities should be left as 0 [as initialized before loop]
                cumulprob = cumulprob + ptrans[4, t, j]

                ptrans[5, t, j] = min(1, alldeath.iloc[t])  # likelihood of death from non CVD cause
                if cumulprob + ptrans[5, t, j] >= 1:  # check for invalid probabilities
                    ptrans[5, t, j] = 1 - cumulprob
                    break  # all other probabilities should be left as 0 [as initialized before loop]
                cumulprob = cumulprob + ptrans[5, t, j]

                ptrans[1, t, j] = min(1, (1 - chddeath.iloc[t]) * risk[t, 0, j])  # likelihood of having CHD and surviving
                if cumulprob + ptrans[1, t, j] >= 1:  # check for invalid probabilities
                    ptrans[1, t, j] = 1 - cumulprob
                    break  # all other probabilities should be left as 0 [as initialized before loop]
                cumulprob = cumulprob + ptrans[1, t, j]

                ptrans[2, t, j] = min(1, (1 - strokedeath.iloc[t]) * risk[t, 1, j])  # likelihood of having stroke and surviving
                if cumulprob + ptrans[2, t, j] >= 1:  # check for invalid probabilities
                    ptrans[2, t, j] = 1 - cumulprob
                    break  # all other probabilities should be left as 0 [as initialized before loop]
                cumulprob = cumulprob + ptrans[2, t, j]

                ptrans[0, t, j] = 1 - cumulprob  # otherwise, patient is still healthy
                break  # computed all probabilities, now quit

    return feasible, ptrans

# Transition probabilities function in reduced state space of healthy, adverse event, death
def TP_RED(periodrisk, chddeath, strokedeath, alldeath, pretrtsbp, pretrtdbp, sbpmin, sbpmax, dbpmin,
           sbp_reduction, dbp_reduction, rel_risk_chd, rel_risk_stroke, numhealth):

    # Calculating probability of health states transitions given ASCVD risk, fatality likelihoods,
    # and probability of mortality due to non-ASCVD events in smaller state space of healthy, adverse event, and death

    # Inputs
    ## periodrisk: 1-year risk of CHD and stroke
    ## chddeath: likelihood of death given a CHD events
    ## strokedeath: likelihood of death given a stroke events
    ## alldeath: likelihood of death due to non-ASCVD events
    ## trtvector: vector of treatments to consider
    ## pretrtsbp: pre-treatment SBP
    ## pretrtdbp: pre-treatment DBP
    ## sbpmin (sbpmax): Minimum (maximum) SBP allowed (clinical constraint)
    ## dbpmin: minimum DBP allowed (clinical constraint)

    # Extracting parameters
    years = periodrisk.shape[0]  # number of non-stationary stages
    events = periodrisk.shape[1]  # number of events
    numtrt = len(sbp_reduction) # number of treatment choices

    # Storing feasibility indicators
    feasible = np.full((years, numtrt), np.nan) # indicators of whether the treatment is feasible at each time

    # Storing risk and TP calculations
    risk = np.full((years, events, numtrt), np.nan)  # stores post-treatment risks
    ptrans = np.zeros((numhealth, years, numtrt))  # state transition probabilities (default of 0, to reduce computations)

    for t in range(years):  # each stage (year)
        for j in range(numtrt):  # each treatment
            if j == 0:  # the do nothing treatment
                if pretrtsbp[t] > sbpmax:  # cannot do nothing when pre-treatment SBP is too high
                    feasible[t, j] = 0
                else:
                    feasible[t, j] = 1  # otherwise, do nothing is always feasible
            else:  # prescibe >0 drugs
                newsbp = pretrtsbp[t] - sbp_reduction[j]
                newdbp = pretrtdbp[t] - dbp_reduction[j]

                # Violates minimum allowable SBP constraint or min allowable DBP constrain and
                # the do nothing option is feasible
                if (newsbp < sbpmin or newdbp < dbpmin) and feasible[t, 0] == 1:
                    feasible[t, j] = 0
                else:  # does not violate min BP
                    feasible[t, j] = 1

            for k in range(events):  # each event type
                # Calculating post-treatment risks
                if k == 0: # CHD events
                    risk[t, k, j] = rel_risk_chd[j]*periodrisk[t, k]
                elif k == 1: # stroke events
                    risk[t, k, j] = rel_risk_stroke[j]*periodrisk[t, k]

            # Health condition transition probabilities
            quits = 0
            while quits == 0:  # compute transition probabilities, using a "break" command if you've exceeded 1 (this never happens!)
                ptrans[2, t, j] = min(1, chddeath.iloc[t] * risk[t, 0, j] +
                                      strokedeath.iloc[t] * risk[t, 1, j] +
                                      alldeath.iloc[t])  # likelihood of death
                cumulprob = ptrans[2, t, j]

                ptrans[1, t, j] = min(1, (1 - chddeath.iloc[t]) * risk[t, 0, j] +
                                      (1 - strokedeath.iloc[t]) * risk[t, 1, j])  # likelihood of having an ASCVD event and surviving
                if cumulprob + ptrans[1, t, j] >= 1:  # check for invalid probabilities
                    ptrans[1, t, j] = 1 - cumulprob
                    break  # all other probabilities should be left as 0 [as initialized before loop]
                cumulprob = cumulprob + ptrans[1, t, j]

                ptrans[0, t, j] = 1 - cumulprob  # otherwise, patient is still healthy
                break  # computed all probabilities, now quit

    return feasible, ptrans

# Transition probabilities function in expanded state space including dynamic diabetes status
def TP_DIAB(periodrisk, diab_risk, chddeath, strokedeath, alldeath, pretrtsbp, pretrtdbp, sbpmin, sbpmax, dbpmin,
            sbp_reduction, dbp_reduction, rel_risk_chd, rel_risk_stroke, numhealth):

    # Calculating probability of health states transitions given ASCVD risk, fatality likelihoods,
    # probability of mortality due to non-ASCVD events, and risk for type 2 diabetes

    # Inputs
    ## periodrisk: 1-year risk of CHD and stroke
    ## chddeath: likelihood of death given a CHD events
    ## strokedeath: likelihood of death given a stroke events
    ## alldeath: likelihood of death due to non-ASCVD events
    ## trtvector: vector of treatments to consider
    ## pretrtsbp: pre-treatment SBP
    ## pretrtdbp: pre-treatment DBP
    ## sbpmin (sbpmax): Minimum (maximum) SBP allowed (clinical constraint)
    ## dbpmin: minimum DBP allowed (clinical constraint)

    # Extracting parameters
    years = periodrisk.shape[0]  # number of non-stationary stages
    events = periodrisk.shape[1]  # number of events
    numtrt = len(sbp_reduction) # number of treatment choices
    sep = int(numhealth/2)  # index of separation between nondiabetic and diabetic states

    # Checking if the patient already has diabetes and determining transition ptobabilities should include without and with diabetes
    diab_stat = np.where(np.all(periodrisk[..., 1] == periodrisk[..., 0]), 1, 2) # considering patient with diabetes only or withiout and with diabetes

    # Storing feasibility indicators
    feasible = np.full((years, numtrt), np.nan) # indicators of whether the treatment is feasible at each time

    # Storing risk and TP calculations
    risk = np.full((years, events, numtrt, diab_stat), np.nan)  # stores post-treatment risks (original and with diabetes)
    ptrans = np.zeros((numhealth, years, numtrt, diab_stat))  # state transition probabilities (original and with diabetes) - default of 0, to reduce computations

    for t in range(years):  # each stage (year)
        for j in range(numtrt):  # each treatment
            if j == 0:  # the do nothing treatment
                if pretrtsbp[t] > sbpmax:  # cannot do nothing when pre-treatment SBP is too high
                    feasible[t, j] = 0
                else:
                    feasible[t, j] = 1  # otherwise, do nothing is always feasible
            else:  # prescibe >0 drugs
                newsbp = pretrtsbp[t] - sbp_reduction[j]
                newdbp = pretrtdbp[t] - dbp_reduction[j]

                # Violates minimum allowable SBP constraint or min allowable DBP constrain and
                # the do nothing option is feasible
                if (newsbp < sbpmin or newdbp < dbpmin) and feasible[t, 0] == 1:
                    feasible[t, j] = 0
                else:  # does not violate min BP
                    feasible[t, j] = 1

            for k in range(events):  # each event type
                # Calculating post-treatment risks
                for d in range(diab_stat): # each diabetes status
                    if k == 0: #CHD events
                        risk[t, k, j, d] = rel_risk_chd[j]*periodrisk[t, k, d]
                    elif k == 1: #stroke events
                        risk[t, k, j, d] = rel_risk_stroke[j]*periodrisk[t, k, d]

            # Health condition transition probabilities for each diabetes status
            for d in range(diab_stat): # for each diabetes status
                quits = 0
                while quits == 0:  # compute transition probabilities, using a "break" command if you've exceeded 1 (this never happens!)
                    ptrans[3, t, j, d] = min(1, chddeath.iloc[t] * risk[t, 0, j, d])  # likelihood of death from CHD event
                    cumulprob = ptrans[3, t, j, d]
    
                    ptrans[4, t, j, d] = min(1, strokedeath.iloc[t] * risk[t, 1, j, d])  # likelihood of death from stroke
                    if cumulprob + ptrans[4, t, j, d] >= 1:  # check for invalid probabilities
                        ptrans[4, t, j, d] = 1 - cumulprob
                        break  # all other probabilities should be left as 0 [as initialized before loop]
                    cumulprob = cumulprob + ptrans[4, t, j, d]
    
                    ptrans[5, t, j, d] = min(1, alldeath.iloc[t])  # likelihood of death from non CVD cause
                    if cumulprob + ptrans[5, t, j, d] >= 1:  # check for invalid probabilities
                        ptrans[5, t, j, d] = 1 - cumulprob
                        break  # all other probabilities should be left as 0 [as initialized before loop]
                    cumulprob = cumulprob + ptrans[5, t, j, d]
    
                    ptrans[1, t, j, d] = min(1, (1 - chddeath.iloc[t]) * risk[t, 0, j, d])  # likelihood of having CHD and surviving
                    if cumulprob + ptrans[1, t, j, d] >= 1:  # check for invalid probabilities
                        ptrans[1, t, j, d] = 1 - cumulprob
                        break  # all other probabilities should be left as 0 [as initialized before loop]
                    cumulprob = cumulprob + ptrans[1, t, j, d]
    
                    ptrans[2, t, j, d] = min(1, (1 - strokedeath.iloc[t]) * risk[t, 1, j, d])  # likelihood of having stroke and surviving
                    if cumulprob + ptrans[2, t, j, d] >= 1:  # check for invalid probabilities
                        ptrans[2, t, j, d] = 1 - cumulprob
                        break  # all other probabilities should be left as 0 [as initialized before loop]
                    cumulprob = cumulprob + ptrans[2, t, j, d]
    
                    ptrans[0, t, j, d] = 1 - cumulprob  # otherwise, patient is still healthy
                    break  # computed all probabilities, now quit

            # Generating nondiabetic and diabetic states
            if diab_stat == 1: # patient already has diabetes
                ptrans[sep:, t, j, 0] = ptrans[:sep, t, j, 0].copy() # copying calculation to diabetic state
                ptrans[:sep, t, j, 0] = 0 # patient cannot transition to nondiabetic states
            elif diab_stat == 2: # patient does not have diabetes originally
                ## Nondiabetic status
                ptrans[sep:, t, j, 0] = ptrans[:sep, t, j, 0].copy()  # starting from same baseline probability in nondiabetic and diabetic states
                if j==0: # no BP treatment
                    ptrans[:sep, t, j, 0] = ptrans[:sep, t, j, 0]*(1-diab_risk[t, 0]) # nondiabetic states
                    ptrans[sep:, t, j, 0] = ptrans[sep:, t, j, 0]*diab_risk[t, 0] # diabetic states
                else: # BP treatment
                    ptrans[:sep, t, j, 0] = ptrans[:sep, t, j, 0]*(1-diab_risk[t, 1])  # nondiabetic states
                    ptrans[sep:, t, j, 0] = ptrans[sep:, t, j, 0]*diab_risk[t, 1]  # diabetic states

                ## Diabetic status
                ptrans[sep:, t, j, 1] = ptrans[:sep, t, j, 1].copy()  # copying calculation to diabetic state
                ptrans[:sep, t, j, 1] = 0  # patient cannot transition to nondiabetic states

    return feasible, ptrans

# Transition probabilities function in expanded action space including drug type
# Note: this function uses the parameters in Law et al. 2003 and 2009 not Sundstrom et al. 2014 and 2015
def TP_DRUG(periodrisk, chddeath, strokedeath, alldeath, riskslope, pretrtsbp, pretrtdbp, sbpmin, sbpmax, dbpmin,
            alldrugs, numhealth):
    # Calculating probability of health states transitions given ASCVD risk, fatality likelihoods, drug type combinations,
    # and probability of mortality due to non-ASCVD events

    # Inputs
    ##periodrisk: 1-year risk of CHD and stroke
    ##chddeath: likelihood of death given a CHD events
    ##strokedeath: likelihood of death given a stroke events
    ##alldeath: likelihood of death due to non-ASCVD events
    ##riskslope: relative risk estimates of CHD and stroke events
    ##pretrtsbp: pre-treatment SBP
    ##pretrtdbp: pre-treatment DBP
    ##sbpmin (sbpmax): Minimum (maximum) SBP allowed (clinical constraint)
    ##dbpmin: minimum DBP allowed (clinical constraint)
    ##alldrugs: treatment options being considered (196 trts: no treatment plus 1 to 5 drugs from 5 different types at standard dosage)

    # Extracting parameters
    years = periodrisk.shape[0]  # number of non-stationary stages
    events = periodrisk.shape[1]  # number of events
    numtrt = len(alldrugs)  # number of treatment choices

    # # Line to obtain action ordering based on an average patient with SBP = 154 and DBP = 97 in Law et al. (2009) - comment if not needed
    # pretrtsbp = 154*np.ones([numhealth, years])
    # pretrtdbp = 97*np.ones([numhealth, years])

    # Storing feasibility indicators
    feasible = np.empty((years, numtrt)); feasible[:] = np.nan  # indicators of whether the treatment is clinically feasible at each time

    # Storing risk and TP calculations
    risk = np.empty((years, events, numtrt)); risk[:] = np.nan  # stores post-treatment risks
    ptrans = np.zeros((numhealth, years, numtrt))  # state transition probabilities--default of 0, to reduce coding/computations

    # Storing BP reductions
    sbpreduc = np.empty((years, numtrt))  # stores SBP reductions
    dbpreduc = np.empty((years, numtrt))  # stores DBP reductions

    for t in range(years):  # each stage (year)
        for j in range(numtrt):  # each treatment
            if j == 0:  # the do nothing treatment
                sbpreduc[t, j] = 0; dbpreduc[t, j] = 0  # no reduction when taking 0 drugs
                if pretrtsbp[t] > sbpmax:
                    feasible[t, j] = 0  # must give treatment
                else:
                    feasible[t, j] = 1  # do nothing is always feasible
            else: # prescibe >0 drugs
                sbpreduc[t, j] = sbp_reductions(j, pretrtsbp[t], alldrugs)
                dbpreduc[t, j] = dbp_reductions(j, pretrtdbp[t], alldrugs)
                newsbp = pretrtsbp[t] - sbpreduc[t, j]
                newdbp = pretrtdbp[t] - dbpreduc[t, j]

                # Violates minimum allowable SBP constraint or min allowable DBP constrain and
                # the do nothing option is feasible
                if (newsbp < sbpmin or newdbp < dbpmin) and feasible[t, 0] == 1:
                    feasible[t, j] = 0
                else:  # does not violate min BP
                    feasible[t, j] = 1

            for k in range(events):  # each event type
                # Calculating post-treatment risks
                risk[t, k, j] = new_risk(sbpreduc[t, j], riskslope.iloc[t, :], periodrisk[t, k], k)

            # Health condition transition probabilities
            quits = 0
            while quits == 0:  # compute transition probabilities, using a "break" command if you've exceeded 1 (this never happens!)
                ptrans[3, t, j] = min(1, chddeath.iloc[t]*risk[t, 0, j])  # likelihood of death from CHD event
                cumulprob = ptrans[3, t, j]

                ptrans[4, t, j] = min(1, strokedeath.iloc[t]*risk[t, 1, j])  # likelihood of death from stroke
                if cumulprob+ptrans[4, t, j] >= 1:  # check for invalid probabilities
                    ptrans[4, t, j] = 1-cumulprob
                    break  # all other probabilities should be left as 0 [as initialized before loop]
                cumulprob = cumulprob+ptrans[4, t, j]

                ptrans[5, t, j] = min(1, alldeath.iloc[t])  # likelihood of death from non CVD cause
                if cumulprob+ptrans[5, t, j] >= 1:  # check for invalid probabilities
                    ptrans[5, t, j] = 1-cumulprob
                    break  # all other probabilities should be left as 0 [as initialized before loop]
                cumulprob = cumulprob+ptrans[5, t, j]

                ptrans[1, t, j] = min(1, (1-chddeath.iloc[t])*risk[t, 0, j])  # likelihood of having CHD and surviving
                if cumulprob+ptrans[1, t, j] >= 1:  # check for invalid probabilities
                    ptrans[1, t, j] = 1-cumulprob
                    break  # all other probabilities should be left as 0 [as initialized before loop]
                cumulprob = cumulprob+ptrans[1, t, j]

                ptrans[2, t, j] = min(1, (1-strokedeath.iloc[t])*risk[t, 1, j])  # likelihood of having stroke and surviving
                if cumulprob+ptrans[2, t, j] >= 1:  # check for invalid probabilities
                    ptrans[2, t, j] = 1-cumulprob
                    break  # all other probabilities should be left as 0 [as initialized before loop]
                cumulprob = cumulprob+ptrans[2, t, j]

                ptrans[0, t, j] = 1-cumulprob  # otherwise, patient is still healthy
                break  # computed all probabilities, now quit

    return feasible, ptrans

# Transition probabilities function in reduced state space and expanded action space including drug type
# Note: this function uses the parameters in Law et al. 2003 and 2009 not Sundstrom et al. 2014 and 2015
def TP_RED_DRUG(periodrisk, chddeath, strokedeath, alldeath, riskslope, pretrtsbp, pretrtdbp, sbpmin, sbpmax, dbpmin,
                alldrugs, numhealth):
    # Calculating probability of health states transitions given ASCVD risk, fatality likelihoods, drug type combinations,
    # and probability of mortality due to non-ASCVD events

    # Inputs
    ##periodrisk: 1-year risk of CHD and stroke
    ##chddeath: likelihood of death given a CHD events
    ##strokedeath: likelihood of death given a stroke events
    ##alldeath: likelihood of death due to non-ASCVD events
    ##riskslope: relative risk estimates of CHD and stroke events
    ##pretrtsbp: pre-treatment SBP
    ##pretrtdbp: pre-treatment DBP
    ##sbpmin (sbpmax): Minimum (maximum) SBP allowed (clinical constraint)
    ##dbpmin: minimum DBP allowed (clinical constraint)
    ##alldrugs: treatment options being considered (196 trts: no treatment plus 1 to 5 drugs from 5 different types at standard dosage)

    # Extracting parameters
    years = periodrisk.shape[0]  # number of non-stationary stages
    events = periodrisk.shape[1]  # number of events
    numtrt = len(alldrugs)  # number of treatment choices

    # # Line to obtain action ordering based on an average patient with SBP = 154 and DBP = 97 in Law et al. (2009) - comment if not needed
    # pretrtsbp = 154*np.ones([numhealth, years])
    # pretrtdbp = 97*np.ones([numhealth, years])

    # Storing feasibility indicators
    feasible = np.empty((years, numtrt)); feasible[:] = np.nan  # indicators of whether the treatment is clinically feasible at each time

    # Storing risk and TP calculations
    risk = np.empty((years, events, numtrt)); risk[:] = np.nan  # stores post-treatment risks
    ptrans = np.zeros((numhealth, years, numtrt))  # state transition probabilities--default of 0, to reduce coding/computations

    # Storing BP reductions
    sbpreduc = np.empty((years, numtrt))  # stores SBP reductions
    dbpreduc = np.empty((years, numtrt))  # stores DBP reductions

    for t in range(years):  # each stage (year)
        for j in range(numtrt):  # each treatment
            if j == 0:  # the do nothing treatment
                sbpreduc[t, j] = 0; dbpreduc[t, j] = 0  # no reduction when taking 0 drugs
                if pretrtsbp[t] > sbpmax:
                    feasible[t, j] = 0  # must give treatment
                else:
                    feasible[t, j] = 1  # do nothing is always feasible
            else: # prescibe >0 drugs
                sbpreduc[t, j] = sbp_reductions(j, pretrtsbp[t], alldrugs)
                dbpreduc[t, j] = dbp_reductions(j, pretrtdbp[t], alldrugs)
                newsbp = pretrtsbp[t] - sbpreduc[t, j]
                newdbp = pretrtdbp[t] - dbpreduc[t, j]

                # Violates minimum allowable SBP constraint or min allowable DBP constrain and
                # the do nothing option is feasible
                if (newsbp < sbpmin or newdbp < dbpmin) and feasible[t, 0] == 1:
                    feasible[t, j] = 0
                else:  # does not violate min BP
                    feasible[t, j] = 1

            for k in range(events):  # each event type
                # Calculating post-treatment risks
                risk[t, k, j] = new_risk(sbpreduc[t, j], riskslope.iloc[t, :], periodrisk[t, k], k)

            # Health condition transition probabilities
            quits = 0
            while quits == 0:  # compute transition probabilities, using a "break" command if you've exceeded 1 (this never happens!)
                ptrans[2, t, j] = min(1, chddeath.iloc[t]*risk[t, 0, j]+
                                      strokedeath.iloc[t]*risk[t, 1, j]+
                                      alldeath.iloc[t])  # likelihood of death
                cumulprob = ptrans[2, t, j]

                ptrans[1, t, j] = min(1, (1-chddeath.iloc[t])*risk[t, 0, j]+
                                      (1-strokedeath.iloc[t])*risk[
                                          t, 1, j])  # likelihood of having an ASCVD event and surviving
                if cumulprob+ptrans[1, t, j] >= 1:  # check for invalid probabilities
                    ptrans[1, t, j] = 1-cumulprob
                    break  # all other probabilities should be left as 0 [as initialized before loop]
                cumulprob = cumulprob+ptrans[1, t, j]

                ptrans[0, t, j] = 1-cumulprob  # otherwise, patient is still healthy
                break  # computed all probabilities, now quit

    return feasible, ptrans

# Transition probabilities function in expanded state space including dynamic diabetes status and expanded action space including drug type
# Note: this function uses the parameters in Law et al. 2003 and 2009 not Sundstrom et al. 2014 and 2015
def TP_DIAB_DRUG(periodrisk, diab_risk, chddeath, strokedeath, alldeath, riskslope, pretrtsbp, pretrtdbp, sbpmin, sbpmax, dbpmin,
                 alldrugs, numhealth):
    # Calculating probability of health states transitions given ASCVD risk, fatality likelihoods, drug type combinations,
    # and probability of mortality due to non-ASCVD events

    # Inputs
    ##periodrisk: 1-year risk of CHD and stroke
    ##chddeath: likelihood of death given a CHD events
    ##strokedeath: likelihood of death given a stroke events
    ##alldeath: likelihood of death due to non-ASCVD events
    ##riskslope: relative risk estimates of CHD and stroke events
    ##pretrtsbp: pre-treatment SBP
    ##pretrtdbp: pre-treatment DBP
    ##sbpmin (sbpmax): Minimum (maximum) SBP allowed (clinical constraint)
    ##dbpmin: minimum DBP allowed (clinical constraint)
    ##alldrugs: treatment options being considered (196 trts: no treatment plus 1 to 5 drugs from 5 different types at standard dosage)

    # Extracting parameters
    years = periodrisk.shape[0]  # number of non-stationary stages
    events = periodrisk.shape[1]  # number of events
    numtrt = len(alldrugs)  # number of treatment choices
    sep = int(numhealth/2)  # index of separation between nondiabetic and diabetic states

    # # Line to obtain action ordering based on an average patient with SBP = 154 and DBP = 97 in Law et al. (2009) - comment if not needed
    # pretrtsbp = 154*np.ones([numhealth, years])
    # pretrtdbp = 97*np.ones([numhealth, years])

    # Checking if the patient already has diabetes and determining transition ptobabilities should include without and with diabetes
    diab_stat = np.where(np.all(periodrisk[..., 1] == periodrisk[..., 0]), 1, 2)  # considering patient with diabetes only or withiout and with diabetes

    # Storing feasibility indicators
    feasible = np.empty((years, numtrt)); feasible[:] = np.nan  # indicators of whether the treatment is clinically feasible at each time

    # Storing risk and TP calculations
    risk = np.empty((years, events, numtrt, diab_stat)); risk[:] = np.nan  # stores post-treatment risks
    ptrans = np.zeros((numhealth, years, numtrt, diab_stat))  # state transition probabilities--default of 0, to reduce coding/computations

    # Storing BP reductions
    sbpreduc = np.empty((years, numtrt))  # stores SBP reductions
    dbpreduc = np.empty((years, numtrt))  # stores DBP reductions

    for t in range(years):  # each stage (year)
        for j in range(numtrt):  # each treatment
            if j == 0:  # the do nothing treatment
                sbpreduc[t, j] = 0; dbpreduc[t, j] = 0  # no reduction when taking 0 drugs
                if pretrtsbp[t] > sbpmax:
                    feasible[t, j] = 0  # must give treatment
                else:
                    feasible[t, j] = 1  # do nothing is always feasible
            else: # prescibe >0 drugs
                sbpreduc[t, j] = sbp_reductions(j, pretrtsbp[t], alldrugs)
                dbpreduc[t, j] = dbp_reductions(j, pretrtdbp[t], alldrugs)
                newsbp = pretrtsbp[t] - sbpreduc[t, j]
                newdbp = pretrtdbp[t] - dbpreduc[t, j]

                # Violates minimum allowable SBP constraint or min allowable DBP constrain and
                # the do nothing option is feasible
                if (newsbp < sbpmin or newdbp < dbpmin) and feasible[t, 0] == 1:
                    feasible[t, j] = 0
                else:  # does not violate min BP
                    feasible[t, j] = 1

            # Calculating post-treatment risks
            for k in range(events):  # each event type
                for d in range(diab_stat):  # each diabetes status
                    risk[t, k, j, d] = new_risk(sbpreduc[t, j], riskslope.iloc[t, :], periodrisk[t, k, d], k)

            # Health condition transition probabilities for each diabetes status
            for d in range(diab_stat):  # for each diabetes status
                quits = 0
                while quits == 0:  # compute transition probabilities, using a "break" command if you've exceeded 1 (this never happens!)
                    ptrans[3, t, j, d] = min(1, chddeath.iloc[t]*risk[t, 0, j, d])  # likelihood of death from CHD event
                    cumulprob = ptrans[3, t, j, d]
    
                    ptrans[4, t, j, d] = min(1, strokedeath.iloc[t]*risk[t, 1, j, d])  # likelihood of death from stroke
                    if cumulprob+ptrans[4, t, j, d] >= 1:  # check for invalid probabilities
                        ptrans[4, t, j, d] = 1-cumulprob
                        break  # all other probabilities should be left as 0 [as initialized before loop]
                    cumulprob = cumulprob+ptrans[4, t, j, d]
    
                    ptrans[5, t, j, d] = min(1, alldeath.iloc[t])  # likelihood of death from non CVD cause
                    if cumulprob+ptrans[5, t, j, d] >= 1:  # check for invalid probabilities
                        ptrans[5, t, j, d] = 1-cumulprob
                        break  # all other probabilities should be left as 0 [as initialized before loop]
                    cumulprob = cumulprob+ptrans[5, t, j, d]
    
                    ptrans[1, t, j, d] = min(1, (1-chddeath.iloc[t])*risk[t, 0, j, d])  # likelihood of having CHD and surviving
                    if cumulprob+ptrans[1, t, j, d] >= 1:  # check for invalid probabilities
                        ptrans[1, t, j, d] = 1-cumulprob
                        break  # all other probabilities should be left as 0 [as initialized before loop]
                    cumulprob = cumulprob+ptrans[1, t, j, d]
    
                    ptrans[2, t, j, d] = min(1, (1-strokedeath.iloc[t])*risk[t, 1, j, d])  # likelihood of having stroke and surviving
                    if cumulprob+ptrans[2, t, j, d] >= 1:  # check for invalid probabilities
                        ptrans[2, t, j, d] = 1-cumulprob
                        break  # all other probabilities should be left as 0 [as initialized before loop]
                    cumulprob = cumulprob+ptrans[2, t, j, d]
    
                    ptrans[0, t, j, d] = 1-cumulprob  # otherwise, patient is still healthy
                    break  # computed all probabilities, now quit

            # Generating nondiabetic and diabetic states
            if diab_stat == 1:  # patient already has diabetes
                ptrans[sep:, t, j, 0] = ptrans[:sep, t, j, 0].copy()  # copying calculation to diabetic state
                ptrans[:sep, t, j, 0] = 0  # patient cannot transition to nondiabetic states
            elif diab_stat == 2:  # patient does not have diabetes originally
                ## Nondiabetic status
                ptrans[sep:, t, j, 0] = ptrans[:sep, t, j, 0].copy()  # starting from same baseline probability in nondiabetic and diabetic states
                if j == 0:  # no BP treatment
                    ptrans[:sep, t, j, 0] = ptrans[:sep, t, j, 0]*(1-diab_risk[t, 0])  # nondiabetic states
                    ptrans[sep:, t, j, 0] = ptrans[sep:, t, j, 0]*diab_risk[t, 0]  # diabetic states
                else:  # BP treatment
                    ptrans[:sep, t, j, 0] = ptrans[:sep, t, j, 0]*(1-diab_risk[t, 1])  # nondiabetic states
                    ptrans[sep:, t, j, 0] = ptrans[sep:, t, j, 0]*diab_risk[t, 1]  # diabetic states

                ## Diabetic status
                ptrans[sep:, t, j, 1] = ptrans[:sep, t, j, 1].copy()  # copying calculation to diabetic state
                ptrans[:sep, t, j, 1] = 0  # patient cannot transition to nondiabetic states

    return feasible, ptrans

# Transition probabilities function in expanded action space with renal sympathetic denervation
def TP_RSN(periodrisk, chddeath, strokedeath, alldeath, pretrtsbp, pretrtdbp, sbpmin, sbpmax, dbpmin,
           sbp_reduction, dbp_reduction, rel_risk_chd, rel_risk_stroke,
           sbp_drop_rsn, dbp_drop_rsn, rel_risk_chd_rsn, rel_risk_stroke_rsn, numhealth):

    # Calculating probability of health states transitions given ASCVD risk, fatality likelihoods,
    # and probability of mortality due to non-ASCVD events

    # Inputs
    ## periodrisk: 1-year risk of CHD and stroke
    ## chddeath: likelihood of death given a CHD events
    ## strokedeath: likelihood of death given a stroke events
    ## alldeath: likelihood of death due to non-ASCVD events
    ## trtvector: vector of treatments to consider
    ## pretrtsbp: pre-treatment SBP
    ## pretrtdbp: pre-treatment DBP
    ## sbpmin (sbpmax): Minimum (maximum) SBP allowed (clinical constraint)
    ## dbpmin: minimum DBP allowed (clinical constraint)

    # Extracting parameters
    years = periodrisk.shape[0]  # number of non-stationary stages
    events = periodrisk.shape[1]  # number of events
    numtrt_meds = len(sbp_reduction)  # number of treatment choices (before renal sympathetic denervation)

    # Expanding BP and risk reductions with renal sympathetic denervation
    sbp_reduction = np.tile(sbp_reduction, 2)
    dbp_reduction = np.tile(dbp_reduction, 2)
    rel_risk_chd = np.tile(rel_risk_chd, 2)
    rel_risk_stroke = np.tile(rel_risk_stroke, 2)
    numtrt_rsn = len(sbp_reduction)  # number of treatment choices (expanded with renal sympathetic denervation)

    # Adjusting BP and risk reductions with renal sympathetic denervation
    sbp_reduction[numtrt_meds:] = sbp_reduction[numtrt_meds:] + sbp_drop_rsn
    dbp_reduction[numtrt_meds:] = dbp_reduction[numtrt_meds:] + dbp_drop_rsn
    rel_risk_chd[numtrt_meds:] = rel_risk_chd[numtrt_meds:] + rel_risk_chd_rsn
    rel_risk_stroke[numtrt_meds:] = rel_risk_stroke[numtrt_meds:] + rel_risk_stroke_rsn

    # Storing feasibility indicators
    feasible = np.full((years, numtrt_rsn), np.nan) # indicators of whether the treatment is feasible at each time

    # Storing risk and TP calculations
    risk = np.full((years, events, numtrt_rsn), np.nan)  # stores post-treatment risks
    ptrans = np.zeros((numhealth, years, numtrt_rsn))  # state transition probabilities (default of 0, to reduce computations)

    for t in range(years):  # each stage (year)
        # Allowing RSN only in the first year of the analysis
        if t == min(range(years)):
            numtrt = numtrt_rsn
        else:
            numtrt = numtrt_meds
        for j in range(numtrt):  # each treatment
            if j == 0:  # the do nothing treatment
                if pretrtsbp[t] > sbpmax:  # cannot do nothing when pre-treatment SBP is too high
                    feasible[t, j] = 0
                else:
                    feasible[t, j] = 1  # otherwise, do nothing is always feasible
            else:  # prescibe >0 drugs
                newsbp = pretrtsbp[t] - sbp_reduction[j]
                newdbp = pretrtdbp[t] - dbp_reduction[j]

                # Violates minimum allowable SBP constraint or min allowable DBP constrain and
                # the do nothing option is feasible
                if (newsbp < sbpmin or newdbp < dbpmin) and feasible[t, 0] == 1:
                    feasible[t, j] = 0
                else:  # does not violate min BP
                    feasible[t, j] = 1

            for k in range(events):  # each event type
                # Calculating post-treatment risks
                if k == 0: #CHD events
                    risk[t, k, j] = rel_risk_chd[j]*periodrisk[t, k]
                elif k == 1: #stroke events
                    risk[t, k, j] = rel_risk_stroke[j]*periodrisk[t, k]

            # Health condition transition probabilities
            quits = 0
            while quits == 0:  # compute transition probabilities, using a "break" command if you've exceeded 1 (this never happens!)
                ptrans[3, t, j] = min(1, chddeath.iloc[t] * risk[t, 0, j])  # likelihood of death from CHD event
                cumulprob = ptrans[3, t, j]

                ptrans[4, t, j] = min(1, strokedeath.iloc[t] * risk[t, 1, j])  # likelihood of death from stroke
                if cumulprob + ptrans[4, t, j] >= 1:  # check for invalid probabilities
                    ptrans[4, t, j] = 1 - cumulprob
                    break  # all other probabilities should be left as 0 [as initialized before loop]
                cumulprob = cumulprob + ptrans[4, t, j]

                ptrans[5, t, j] = min(1, alldeath.iloc[t])  # likelihood of death from non CVD cause
                if cumulprob + ptrans[5, t, j] >= 1:  # check for invalid probabilities
                    ptrans[5, t, j] = 1 - cumulprob
                    break  # all other probabilities should be left as 0 [as initialized before loop]
                cumulprob = cumulprob + ptrans[5, t, j]

                ptrans[1, t, j] = min(1, (1 - chddeath.iloc[t]) * risk[t, 0, j])  # likelihood of having CHD and surviving
                if cumulprob + ptrans[1, t, j] >= 1:  # check for invalid probabilities
                    ptrans[1, t, j] = 1 - cumulprob
                    break  # all other probabilities should be left as 0 [as initialized before loop]
                cumulprob = cumulprob + ptrans[1, t, j]

                ptrans[2, t, j] = min(1, (1 - strokedeath.iloc[t]) * risk[t, 1, j])  # likelihood of having stroke and surviving
                if cumulprob + ptrans[2, t, j] >= 1:  # check for invalid probabilities
                    ptrans[2, t, j] = 1 - cumulprob
                    break  # all other probabilities should be left as 0 [as initialized before loop]
                cumulprob = cumulprob + ptrans[2, t, j]

                ptrans[0, t, j] = 1 - cumulprob  # otherwise, patient is still healthy
                break  # computed all probabilities, now quit

    return feasible, ptrans
