
def gwp_calc(fraction, weight_all, gwp_all):
    denominator = 0
    res = 0
    for i in range(len(fraction)):
        denominator += fraction[i]*weight_all[i]

    for i in range(len(fraction)):
        numerator = fraction[i]*weight_all[i]
        if denominator != 0:
            res += (numerator / denominator) * gwp_all[i] 
            
    return res

# Batch calculation for GWP
def run_gwp(compounds, compound_fractions):

    results = []
    weight = []
    gwp_all = []

    for compound in compounds:
        weight.append(compound.weight)
        gwp_all.append(compound.gwp)

    for compound_fraction in compound_fractions:
        # Adding data to the file
        results.append(gwp_calc(compound_fraction, weight, gwp_all))

    return results