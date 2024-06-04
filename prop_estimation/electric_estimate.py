
# Calculate E/N of a multiple mixtures 
def en_estimate_batch(compounds, compound_fractions):
    result = []
    for fractions in compound_fractions:
        estimate = 0
        for i in range(len(fractions)):
            estimate += fractions[i]/100 * compounds[i].en_crit
        result.append(estimate)
    return result

# Calculate E/N of a single mixture 
def en_estimate_single(mixture, fractions):
    estimate = 0
    for i in range(len(fractions)):
        estimate += fractions[i]/100 * mixture[i].en_crit
    
    return estimate