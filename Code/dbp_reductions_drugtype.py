# ==============================================================
# Estimating change in DBP from standard dose for each drug type
# ==============================================================

# Note: these functions use the parameters in Law et al. 2003 and 2009 not Sundstrom et al. 2014 and 2015

# Loading modules
import numpy as np

# Functions to estimate the effect of each drug on DBP
def thiazides(pretreatment):
    BPdrop = 4.4+0.11*(pretreatment-97)
    return BPdrop

def aceinhibitors(pretreatment):
    BPdrop = 4.7+0.11*(pretreatment-97)
    return BPdrop

def arb(pretreatment):
    BPdrop = 5.7+0.11*(pretreatment-97)
    return BPdrop

def calciumcb(pretreatment):
    BPdrop = 5.9+0.11*(pretreatment-97)
    return BPdrop

def betablock(pretreatment):
    BPdrop = 6.7+0.11*(pretreatment-97)
    return BPdrop

# Calculating DBP reductions for each combination
def dbp_reductions(trt, pretreatment, alldrugs):

    #    #Initializing post-treatment DBP
    #    posttreatment = pretreatment

    # Initializing DBP reduction
    dbp_reduc = 0

    # Making sure evaluated treatmnet is in a list or string format
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
            dbp_reduc = dbp_reduc+thiazides(pretreatment-dbp_reduc)
    if bb > 0:  # Reductions due to Beta-blockers
        for r in range(bb):
            #            posttreatment = posttreatment-betablock(posttreatment)
            dbp_reduc = dbp_reduc+betablock(pretreatment-dbp_reduc)
    if ace > 0:  # Reductions due to ACE inhibitors
        for r in range(ace):
            #            posttreatment = posttreatment-aceinhibitors(posttreatment)
            dbp_reduc = dbp_reduc+aceinhibitors(pretreatment-dbp_reduc)
    if a2ra > 0:  # Reductions due to Angiotensin II receptor antagonists
        for r in range(a2ra):
            #            posttreatment = posttreatment-arb(posttreatment)
            dbp_reduc = dbp_reduc+arb(pretreatment-dbp_reduc)
    if ccb > 0:  # Reductions due to Calcium channel blockers
        for r in range(ccb):
            #            posttreatment = posttreatment-calciumcb(posttreatment)
            dbp_reduc = dbp_reduc+calciumcb(pretreatment-dbp_reduc)

    return dbp_reduc  # ,posttreatment