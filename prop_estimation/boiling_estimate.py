# Bezier taken from < BezierCurveFunction-v1.ipynb > on 2019-05-02
# Bisection from https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array

from prop_estimation.bezier import Bezier
import numpy as np


def bisection(array,value):
    '''Given an ``array`` , and given a ``value`` , returns an index j such that ``value`` is between array[j]
    and array[j+1]. ``array`` must be monotonic increasing. j=-1 or j=len(array) is returned
    to indicate that ``value`` is out of range below and above respectively.'''
    n = len(array)
    if (value < array[0]):
        return -1
    elif (value > array[n-1]):
        return n
    jl = 0      # Initialize lower
    ju = n-1    # and upper limits.
    while (ju-jl > 1):      # If we are not yet done,
        jm=(ju+jl) >> 1     # compute a midpoint with a bitshift
        if (value >= array[jm]):
            jl=jm   # and replace either the lower limit
        else:
            ju=jm   # or the upper limit, as appropriate.
        # Repeat until the test condition is satisfied.
    if (value == array[0]):# edge cases at bottom
        return 0
    elif (value == array[n-1]):# and top
        return n-1
    else:
        return jl


# Linear implementation of boiling point estimation
def boiling_estimate_linear(compounds, compound_fractions):
    results = []
    for fractions in compound_fractions:
        estimate = 0
        for i in range(len(fractions)):
            estimate += fractions[i]/100 * compounds[i].boiling_point
        results.append(estimate)
    return results


# Return max boiling point of compounds
def boiling_max(compounds, compound_fractions):
    results = []
    for fractions in compound_fractions:
        max_bp = 0
        for comp in compounds:
            if max_bp < comp.boiling_point:
                max_bp = comp.boiling_point
        results.append(max_bp)
    return results


# Bezier curve implementation of boiling point estimation for multiple mixture fractions
def boiling_bezier_batch(compounds, compound_fractions):
    results = []
    max_bp_val = -1
    min_bp_val = 9999
    min_bp_index = -1

    # Find max and min bp compound positions
    for i in range(len(compounds)):
        if max_bp_val < compounds[i].boiling_point:
            max_bp_val = compounds[i].boiling_point
        if min_bp_val > compounds[i].boiling_point:
            min_bp_val = compounds[i].boiling_point
            min_bp_index = i

    # Calculate Bezier 
    t_points = np.arange(0, 1, 0.01) #................................. Creates an iterable list from 0 to 1.
    points = np.array([[0, max_bp_val], [1, max_bp_val], [1, min_bp_val]]) #.... Creates an array of coordinates.
    curve = Bezier.Curve(t_points, points) #......................... Returns an array of coordinates.
    
    for fractions in compound_fractions:        
        index = bisection(curve[:, 0], fractions[min_bp_index] / 100)
        if index == len(curve[:, 0]):
            index -= 1   
        results.append(curve[index][1])
    return results
