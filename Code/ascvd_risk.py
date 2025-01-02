# =======================
# Calculating ASCVD Risk
# =======================

# Loading modules
import math
import sys

# Risk calculator function
def arisk(event, sex, race, age, sbp, smk, tc, hdl, diab, trt, time):
    # ASCVD risk calculator (2013 ACC/AHA Guideline)
    # inputs: type of risk to calculate (1=CHD, 2=stroke), sex (1=male, 0=female),
    # race (1=white,0=black), age, SBP, smoking status (1=smoker, 0=nonsmoker), total
    # cholesterol, HDL, diabetes status (1=diabetic, 0=nondiabetic), trt
    # (1=BP reported is on treatment, 0=BP reported is untreated), time (1-year, 5-year, 10-year risk).
    # outputs: likelihood of CHD or stroke in the next "time" years

    if sex == 1:  # male
        if race == 1:  # white
            b_age = 12.344
            b_age2 = 0
            b_tc = 11.853
            b_age_tc = -2.664
            b_hdl = -7.990
            b_age_hdl = 1.769

            if trt == 1:  # SBP is treated SBP
                b_sbp = 1.797
                b_age_sbp = 0
            else:  # SBP is untreated SBP
                b_sbp = 1.764
                b_age_sbp = 0

            b_smk = 7.837
            b_age_smk = -1.795
            b_diab = 0.658
            meanz = 61.18

            if time == 1:
                basesurv = 0.99358
            elif time == 5:
                basesurv = 0.96254
            elif time == 10:
                basesurv = 0.9144
            else:
                sys.exit(str(time) + " is an improper time length for risk calculation")

        else:  # black
            b_age = 2.469
            b_age2 = 0
            b_tc = 0.302
            b_age_tc = 0
            b_hdl = -0.307
            b_age_hdl = 0

            if trt == 1:  # SBP is treated SBP
                b_sbp = 1.916
                b_age_sbp = 0
            else:  # SBP is untreated SBP
                b_sbp = 1.809
                b_age_sbp = 0

            b_smk = 0.549
            b_age_smk = 0
            b_diab = 0.645
            meanz = 19.54

            if time == 1:
                basesurv = 0.99066
            elif time == 5:
                basesurv = 0.95726
            elif time == 10:
                basesurv = 0.8954
            else:
                sys.exit(str(time) + " is an improper time length for risk calculation")

    else:  # female
        if race == 1:  # white
            b_age = -29.799
            b_age2 = 4.884
            b_tc = 13.540
            b_age_tc = -3.114
            b_hdl = -13.578
            b_age_hdl = 3.149

            if trt == 1:  # SBP is treated SBP
                b_sbp = 2.019
                b_age_sbp = 0
            else:  # SBP is untreated SBP
                b_sbp = 1.957
                b_age_sbp = 0

            b_smk = 7.574
            b_age_smk = -1.665
            b_diab = 0.661
            meanz = -29.18

            if time == 1:
                basesurv = 0.99828
            elif time == 5:
                basesurv = 0.98898
            elif time == 10:
                basesurv = 0.9665
            else:
                sys.exit(str(time) + " is an improper time length for risk calculation")

        else:  # black
            b_age = 17.114
            b_age2 = 0
            b_tc = 0.940
            b_age_tc = 0
            b_hdl = -18.920
            b_age_hdl = 4.475

            if trt == 1:  # SBP is treated SBP
                b_sbp = 29.291
                b_age_sbp = -6.432
            else:  # SBP is untreated SBP
                b_sbp = 27.820
                b_age_sbp = -6.087

            b_smk = 0.691
            b_age_smk = 0
            b_diab = 0.874
            meanz = 86.61

            if time == 1:
                basesurv = 0.99834
            elif time == 5:
                basesurv = 0.98194
            elif time == 10:
                basesurv = 0.9533
            else:
                sys.exit(str(time) + " is an improper time length for risk calculation")

    # proportion of ascvd assumed to be CHD or stroke, respectively
    eventprop = [0.7, 0.3] # updated 1/23/2020 after email with Sussman

    indivz = b_age * math.log(age) + b_age2 * (math.log(age)) ** 2 + b_tc * math.log(tc) + b_age_tc * math.log(
        age) * math.log(tc) + b_hdl * math.log(hdl) + b_age_hdl * math.log(age) * math.log(hdl) + b_sbp * math.log(
        sbp) + b_age_sbp * math.log(age) * math.log(sbp) + b_smk * smk + b_age_smk * math.log(age) * smk + b_diab * diab

    risk = eventprop[event] * (1-basesurv**(math.exp(indivz-meanz)))

    return risk

# Revised risk calculator function
def rev_arisk(event, sex, black, age, sbp, smk, tc, hdl, diab, trt, time):
    # ASCVD risk calculator (Yadlowsky et al. 2018)
    # inputs: type of risk to calculate (1=CHD, 2=stroke), sex (1=male, 0=female),
    # black (1=Black race,0=otherwise), age, SBP, smoking status (1=smoker, 0=nonsmoker), total
    # cholesterol, HDL, diabetes status (1=diabetic, 0=nondiabetic), trt
    # (1=BP reported is on treatment, 0=BP reported is untreated), time (1-year, 10-year risk).
    # outputs: likelihood of CHD or stroke in the next "time" years

    # Coefficients for male patients
    if sex == 1:
        intercept = -11.679980
        b_age = 0.064200
        b_black = 0.482835
        b_sbp2 = -0.000061
        b_sbp = 0.038950
        b_trt = 2.055533
        b_diab = 0.842209
        b_smk = 0.895589
        b_tc_hdl = 0.193307
        b_black_age = 0
        b_trt_sbp = -0.014207
        b_black_sbp = 0.011609
        b_black_trt = -0.119460
        b_age_sbp = 0.000025
        b_black_diab = -0.077214
        b_black_smk = -0.226771
        b_black_tc_hdl = -0.117749
        b_black_trt_sbp = 0.004190
        b_black_age_sbp = -0.000199

    # Coefficients for female patients
    else:
        intercept = -12.823110
        b_age = 0.106501
        b_black = 0.432440
        b_sbp2 = 0.000056
        b_sbp = 0.017666
        b_trt = 0.731678
        b_diab = 0.943970
        b_smk = 1.009790
        b_tc_hdl = 0.151318
        b_black_age = -0.008580
        b_trt_sbp = -0.003647
        b_black_sbp = 0.006208
        b_black_trt = 0.152968
        b_age_sbp = -0.000153
        b_black_diab = 0.115232
        b_black_smk = -0.092231
        b_black_tc_hdl = 0.070498
        b_black_trt_sbp = -0.000173
        b_black_age_sbp = -0.000094

    # Proportion of ascvd assumed to be CHD or stroke, respectively
    eventprop = [0.7, 0.3] # updated 1/23/2020 after email with Sussman

    # Calculating sum of terms
    betaX = intercept+b_age*age+b_black*black+b_sbp2*(sbp**2)+b_sbp*sbp+b_trt*trt+b_diab*diab+ \
            b_smk*smk+b_tc_hdl*(tc/hdl)+b_black_age*(black*age)+b_trt_sbp*(trt*sbp)+b_black_sbp*(black*sbp)+ \
            b_black_trt*(black*trt)+b_age_sbp*(age*sbp)+b_black_diab*(black*diab)+b_black_smk*(black*smk)+ \
            b_black_tc_hdl*(black*tc/hdl)+b_black_trt_sbp*(black*trt*sbp)+b_black_age_sbp*(black*age*sbp)

    # Estimating risk
    risk = eventprop[event]*(1/(1+(math.exp(-betaX))))

    # Estimating 1-year or 10-year risk for ASCVD events
    if time == 1:
        mult = 0.082 # this multiplier was obtained by solving the LP in the '1_year_ascvd_calibration.py' script
        a_risk = risk*mult
    elif time == 10:
        mult = 1
        a_risk = risk*mult
    else:
        sys.exit(str(time)+" is an improper time length for risk calculation")

    return a_risk
