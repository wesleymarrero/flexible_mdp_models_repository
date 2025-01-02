# ====================================================================================
# Estimating Set of Near-Optimal Choices using the Residual-based Bootstrap Algorithm
# ====================================================================================

# Loading modules
import numpy as np

# Simulation-based multiple comparison with a control algorithm
def sbmcc(Q_bar, Q_hat, sigma2_bar, a_ctrl, obs, rep, byrep=False):

    # Extracting parameters
    sens_sc = Q_hat.shape[0]
    numtrt = Q_hat.shape[1]

    if byrep is True:
        # Arrays to store results
        psi = np.full((sens_sc, numtrt, rep+1), np.nan)

        # Calculating root statistic for each repication until rep
        # Estimating Q-values and approximately optimal policies with rep number of replications
        Q_hat_rep = np.nanmean(Q_bar[:, :, :rep], axis=2) # estimated Q-values

        for sc in range(sens_sc): # each sensitivity analysis scenario
            for j in range(numtrt): # each treatment
                for r in range(rep): # each replication
                    psi[sc, j, r] = (Q_bar[sc, a_ctrl[sc], r]-Q_bar[sc, j, r]-(Q_hat_rep[sc, a_ctrl[sc]]-Q_hat_rep[sc, j]))/ \
                                    np.sqrt((sigma2_bar[sc, a_ctrl[sc], r]+sigma2_bar[sc, j, r])/obs)

        # Obtaining maximum psi
        psi_max = np.amax(psi, axis=1)

    else: # calculate max_psi only once for reps number of replications
        # Arrays to store results
        psi = np.full((sens_sc, numtrt), np.nan)

        # Calculating root statistic
        for sc in range(sens_sc): # each sensitivity analysis scenario
            for j in range(numtrt): # each treatment
                psi[sc, j] = (Q_bar[sc, a_ctrl[sc], rep]-Q_bar[sc, j, rep]-(Q_hat[sc, a_ctrl[sc]]-Q_hat[sc, j]))/ \
                          np.sqrt((sigma2_bar[sc, j, rep]+sigma2_bar[sc, a_ctrl[sc], rep])/obs)

        # Obtaining maximum psi
        psi_max = np.amax(psi, axis=1)

    return psi_max
