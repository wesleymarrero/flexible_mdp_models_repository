# ==============================================================
# Estimating change in SBP from standard dose for each drug type
# ==============================================================

# Note: these functions use the parameters in Law et al. 2003 and 2009 not Sundstrom et al. 2014 and 2015

# Loading modules
import numpy as np

# Functions to estimate the effect of each drug on SBP
def aceinhibitors(pretreatment):
    BPdrop = 8.5+0.1*(pretreatment-154)
    return BPdrop

def calciumcb(pretreatment):
    BPdrop = 8.8+0.1*(pretreatment-154)
    return BPdrop

def thiazides(pretreatment):
    BPdrop = 8.8+0.1*(pretreatment-154)
    return BPdrop

def betablock(pretreatment):
    BPdrop = 9.2+0.1*(pretreatment-154)
    return BPdrop

def arb(pretreatment):
    BPdrop = 10.3+0.1*(pretreatment-154)
    return BPdrop

# Calculating SBP reductions for each combination
def sbp_reductions(trt, pretreatment, alldrugs):

    #    #Initializing post-treatment SBP
    #    posttreatment = pretreatment

    # Initializing SBP reduction
    sbp_reduc = 0

    # Making sure evaluated treatment is in a list or string format
    if type(alldrugs[trt]) == str or type(alldrugs[trt]) == list:
        drugcomb = alldrugs[trt]
    else:
        drugcomb = list(alldrugs[trt])

    # Counting number of times a drug is given
    th = drugcomb.count('TH')
    bb = drugcomb.count('BB')
    ace = drugcomb.count('ACE')
    a2ra = drugcomb.count('ARB')
    ccb = drugcomb.count('CCB')

    if th > 0:  # Reductions due to Thiazides
        for r in range(th):
            #            posttreatment = posttreatment-thiazides(posttreatment)
            sbp_reduc = sbp_reduc+thiazides(pretreatment-sbp_reduc)
    if bb > 0:  # Reductions due to Beta-blockers
        for r in range(bb):
            #            posttreatment = posttreatment-betablock(posttreatment)
            sbp_reduc = sbp_reduc+betablock(pretreatment-sbp_reduc)
    if ace > 0:  # Reductions due to ACE inhibitors
        for r in range(ace):
            #            posttreatment = posttreatment-aceinhibitors(posttreatment)
            sbp_reduc = sbp_reduc+aceinhibitors(pretreatment-sbp_reduc)
    if a2ra > 0:  # Reductions due to Angiotensin II receptor antagonists
        for r in range(a2ra):
            #            posttreatment = posttreatment-arb(posttreatment)
            sbp_reduc = sbp_reduc+arb(pretreatment-sbp_reduc)
    if ccb > 0:  # Reductions due to Calcium channel blockers
        for r in range(ccb):
            #            posttreatment = posttreatment-calciumcb(posttreatment)
            sbp_reduc = sbp_reduc+calciumcb(pretreatment-sbp_reduc)

    return sbp_reduc  # ,posttreatment

