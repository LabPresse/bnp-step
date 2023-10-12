"""
Implementation of BIC-based step-finding algorithm from Kalafut and Visscher's 2008 paper

Alex Rojewski, 2023
"""
import numpy as np


def CalcStepMean(segment):
    """
    Helper function to calculate the MLE of a step's mean.
    
    Arguments:
    segment -- Numpy array of observations

    Returns:
    MLE of step mean (single value)
    """
    n = len(segment)
    sum_x_i = np.sum(segment)
    return sum_x_i / n


def CalcStepVariance(slices, sliced_array, n):
    """
    Helper function to calculate the MLE of the common variance.

    Arguments:
    slices -- numpy array of where steps occur
    sliced_array -- numpy array of arrays containing the observations grouped by step
    n -- number of observations
    
    Returns: 
    MLE of the variance (single value)
    """
    # If we have at least one change point, do each slice individually
    if (slices[0] != 0):
        outer_sum = 0
        for i in range(len(slices)+1):
            inner_sum = 0
            mu = CalcStepMean(sliced_array[i])
            for l in range(len(sliced_array[i])):
                inner_sum += (sliced_array[i][l] - mu)**2
            outer_sum += inner_sum
        return outer_sum / n
    # Otherwise, calculate the variance based on the whole trace as one "step"
    else:
        outer_sum = 0
        mu = CalcStepMean(sliced_array)
        for l in range(len(sliced_array)):
            outer_sum += (sliced_array[l] - mu)**2
        return outer_sum / n


def CalcSIC(k, sigma, n):
    """
    Calculates the Schwarz Information Criterion (SIC)

    Arguments:
    k -- number of steps
    sigma -- variance of the steps
    n -- number of observations

    Returns:
    SIC (single value)
    """
    return(k + 2)*np.log(n) + n*np.log(sigma)


def SliceSteps(slices, test_point, input_data):
    """
    Helper function to slice an array of observations into distinct steps given a new candidate step

    Arguments:
    slices -- numpy array of existing steps
    test_point -- location of candidate step (single value)
    input_data -- numpy array of observations

    Returns:
    sliced_array -- numpy array of arrays containing the observations grouped by step
    tmp_slices -- numpy array of where steps occur
    """
    tmp_slices = slices.copy()
    if (len(slices) == 1 and slices[0] == 0):
        tmp_slices[0] = test_point
    else:
        tmp_slices = np.append(slices,[test_point])
    tmp_slices = np.sort(tmp_slices)
    tmp_slices = tmp_slices.astype(int)
    sliced_array = np.split(input_data, tmp_slices)
    return sliced_array, tmp_slices


def GreedyBIC(data, N, t_n):
    """
    Locates steps using the BIC-based step-finding algorithm 
    from Kalafut and Visscher's 2008 paper.

    Arguments:
    data -- numpy array of observations
    N -- number of observations
    t_n -- numpy array of time points associated with each observation

    Returns:
    jump_times -- numpy array of time points where steps occur
    mu_calc -- numpy array of mean values for each holding period
    sig_calc -- value of the variance
    BIC_st -- numerical value of the final BIC
    """
    # Initial conditions: assume no change points, so there are no jump points.
    # K_i = 0 since we assume zero steps; this gives p = k + 2 as there is always at
    # least one mean and one common variance for a set of data.
    # Assume for the BIC that the constant is zero and the O(N) terms can be dropped.
    K_i = 0
    J = np.zeros(1)
    # Calculate sig_k^2 for K = 0
    sig_ksq_i = CalcStepVariance(J, data, N)
    # Calculate initial BIC
    BIC_i = CalcSIC(K_i, sig_ksq_i, N)

    # Main loop for greedy BIC algorithm
    # Loop control variable
    notDone = True
    # Initial value for new K - should always be the initial K value
    K_test = K_i
    # Sets a "standard" BIC to compare to - updates after each increment of K. For first run, equals initial BIC
    BIC_st = BIC_i
    # Copy of data array to use - ensures fidelity of original synthetic data
    Y = data.copy()

    # Keep adding jump points until you aren't reducing the BIC appreciably
    while(notDone):
        # Test with another jump point
        K_test += 1
        # Reset sig_k^2 and BIC test values
        BIC_test = 0
        sig_ksq_test = 0
        # Re-initialize lowest BIC value - this is how we find exactly where new jump should go.
        BIC_lowest = BIC_st
        # Initialize the new jump point as -1
        j_new = -1
        # If this stays true, we stop adding steps
        no_jumps = True
        # Current jump point we are testing - resets each outer loop
        j_test = 1
        # For the current jump point, test to see if BIC is reduced
        while(j_test < N):
            # Reset sig_k^2
            sig_ksq_test = 0
            # See if the new jump point is already a jump point; if so, skip it
            if (J[0] != 0):
                skip = False
                for q in J:
                    if (int(j_test) == int(q)):
                        skip = True
                if (skip):
                    j_test += 1
                    continue
            Y_sliced, tmp_J = SliceSteps(J, j_test, Y)
            sig_ksq_test = CalcStepVariance(tmp_J, Y_sliced, N)
            BIC_test = CalcSIC(K_test, sig_ksq_test, N)
            if (BIC_test < BIC_st):
                no_jumps = False
                if (BIC_test < BIC_lowest):
                    BIC_lowest = BIC_test
                    j_new = j_test
            j_test += 1
        # If we failed to lower the BIC, stop the loop
        if no_jumps:
            notDone = False
            # We have found our best number of parameters - set K (accounting for incrementing K at beginning of loop)
            K = K_test - 1
            continue
        # Otherwise, prepare for next loop
        # New BIC standard
        BIC_st = BIC_lowest
        # Add the first jump point
        if (J.size == 1 and J[0] == 0):
            J[0] = j_new
        # Otherwise, add new jump point to J array
        else:
            J = np.append(J,[j_new])
        # Sort jump points in J array (ascending order)
        J = np.sort(J)

    # Report found jump points
    # Convert J to ints in preparation for using it as an index
    J = J.astype(int)
    # Initialize jump time vector
    jump_times = np.zeros(J.size)
    # Find the time indices for our jumps
    ctl = 0
    while (ctl < J.size):
        jump_times[ctl] = t_n[J[ctl]]
        ctl += 1
    # Now, maximize the likelihood for each model to find the means and the variance
    # Split Y array into sub-arrays based on jump points
    Y2 = np.split(Y, J)
    # Initialize mu array
    mu_calc = np.zeros((J.size)+1)
    # Calculate MLE for sigma
    sig_calc = CalcStepVariance(J, Y2, N)
    # Calculate MLEs for means
    for c in range((J.size)+1):
        mu_calc[c] = CalcStepMean(Y2[c])
    # Return results
    return jump_times, mu_calc, sig_calc, BIC_st
