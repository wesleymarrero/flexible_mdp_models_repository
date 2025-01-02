# =====================================
# Calculating Risk for Type 2 Diabetes
# =====================================

# Loading modules
import math

# Risk calculator function
def drisk(age, bmi, sex, waist, bp_med, h_gluc):
    # Type 2 diabetes risk calculator (Lindstrom and Toumilehto, 2003)
    # inputs: age, sex (1=male, 0=female), BMI in kg/m^2, waist circumference in cm, use of BP medication (1=yes, 0=no),
    # use of BP medications (1=yes, 0=no), history of high glucose (i.e., ever been told by a professional that
    # they have diabetes or latent diabetes) (1=yes, 0=no)
    # outputs: likelihood of type 2 diabetes in the next year (10 year estimate divided by 10)

    # Coefficients
    intercept = -5.514

    ## Age coefficient
    if 45 <= age < 55:
        b_age = 0.628
    elif age >= 55:
        b_age = 0.892
    else:
        b_age = 0

    ## BMI coefficient
    if 25 < bmi <= 30:
        b_bmi = 0.165
    elif bmi > 30:
        b_bmi = 1.096
    else:
        b_bmi = 0

    ## Waist circumference coefficient
    if sex == 1: # males
        if 94 <= waist < 102:
            b_waist = 0.857
        elif waist >= 102:
            b_waist = 1.350
        else:
            b_waist = 0
    else: # females
        if 80 <= waist < 88:
            b_waist = 0.857
        elif waist >= 88:
            b_waist = 1.350
        else:
            b_waist = 0

    b_bp_meds = 0.711
    b_h_gluc = 2.139

    # Calculating sum of terms
    betaX = intercept + b_age + b_bmi + b_waist + bp_med*b_bp_meds + h_gluc*b_h_gluc

    # Estimating risk 10-year risk
    risk = 1/(1+(math.exp(-betaX)))

    # Estimating 1-year risk from 10-year risk of diabetes
    d_risk = risk/10

    return d_risk
