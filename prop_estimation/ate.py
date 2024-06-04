# Get a point estimate 
def get_estimate(ppm):
    if ppm <= 0:
        return -1
    elif ppm <= 100:
        return 10
    elif ppm <= 500:
        return 100
    elif ppm <= 2500:
        return 700
    elif ppm <= 20000:
        return 4500
    else:
        return 5500     # Ceiling value for LC50 --> Non-toxic gases


# Calculate ATE of a mixture
def calculate_toxicity(frac, ate):
    res = 0
    unknown_percent = 0
    for i in range(len(frac)):
        if ate[i] > 0:
            res += frac[i] / ate[i]
        if ate[i] < 0:
            unknown_percent += frac[i]
    try:
        if unknown_percent > 10:
            return (100-unknown_percent) / res
        else:
            return 100 / res
    except ZeroDivisionError:
        return -1


# Run a batch calculation for ATE
def run_ate(num, compounds, compound_fractions):
    compound_estimate = []
    for i in range(num):
        # Enter LC50 values for 4H exposure in ppm
        compound_estimate.append(get_estimate(compounds[i].toxic))

    results = []

    for compound_fraction in compound_fractions:
        # Adding data to the file
        value = calculate_toxicity(compound_fraction, compound_estimate)
        results.append(value)

    return results

