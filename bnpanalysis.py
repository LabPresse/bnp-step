"""
Analysis tools for BNP-Step and the two alternative methods mentioned in the paper.

Alex Rojewski, 2023
"""

import os
import numpy as np
import scipy as sp
from scipy import special
import matplotlib.pyplot as plt
import csv


def load_ihmm_mode_means(filename : str, 
                         path = None
                         ):
    """
    Loads the mean emission levels from all samples with the mode number of states
    generated from the iHMM method of the Sgouralis 2016 paper.

    TODO: mention what these are used for

    Note: to obtain results in the correct format, run the method with the custom MATLAB
    script included in the repository, then run the cleanup scripts as described in the
    repository readme.

    Arguments:
    filename (str) -- Name of the file to be loaded. The format is found in the repository readme.
    path -- Path where the file is located.

    Returns:
    heights (numpy array) -- The mode means from the iHMM.
    """
    # Input validation
    if not isinstance(filename, str):
            raise TypeError(f"filename should be of type str instead of {type(filename)}")
    # TODO: validate path
    full_name = filename + '.csv'
    if path is not None:
        full_path = os.path.join(path, full_name)
    else:
        full_path = full_name
    
    # Load data
    # TODO: make this use pandas for consistency with other functions
    heights = []
    with open(full_path) as f:
        csv_reader = csv.reader(f,delimiter=',')
        for row in csv_reader:
            tmp_data = row
            for n in range(len(tmp_data)):
                heights.append(float(tmp_data[n]))
    heights = np.asarray(heights)

    return heights


def load_ihmm_mode_mean_trajectory(filename : str,
                                   path = None
                                   ):
    """
    Loads mode mean trajectory from iHMM method from the Sgouralis 2016 paper. This is what
    is used to generate the step plot for the iHMM when comparing to BNP-Step.

    Note: to obtain results in the correct format, run the method with the custom MATLAB
    script included in the repository, then run the cleanup scripts as described in the
    repository readme.

    Arguments:
    filename (str) -- Name of the file to be loaded. The format is found in the repository readme.
    path -- Path where the file is located.

    Returns:
    sampled_heights (numpy array) -- The mode mean trajectory (as defined in Sgouralis 2016)
                                     from the iHMM.
    """
    # Input validation
    if not isinstance(filename, str):
            raise TypeError(f"filename should be of type str instead of {type(filename)}")
    # TODO: validate path
    full_name = filename + '.csv'
    if path is not None:
        full_path = os.path.join(path, full_name)
    else:
        full_path = full_name
    
    # Read in data
    sampled_heights  = []
    with open(full_path) as f:
        csv_reader = csv.reader(f,delimiter=',')
        for row in csv_reader:
            tmp_data = row
            for n in range(len(tmp_data)):
                sampled_heights.append(float(tmp_data[n]))
    
    sampled_heights = np.asarray(sampled_heights)

    return sampled_heights    


def load_ihmm_samples(filename : str, 
                      skip_indices : str, 
                      path = None
                      ):
    """
    Loads generated samples from iHMM method from the Sgouralis 2016 paper. Only those samples
    with the mode number of states are selected.

    Note: to obtain results in the correct format, run the method with the custom MATLAB
    script included in the repository, then run the cleanup scripts as described in the
    repository readme.

    Arguments:
    filename (str) -- Name of the file to be loaded. The format is found in the repository readme.
    skip_indices (str) -- File with the list of samples to skip (those that do not have the mode
                          number of states).
    path -- Path where the file is located.

    Returns:
    sampled_heights (numpy array) -- The mode mean trajectory (as defined in Sgouralis 2016)
                                     from the iHMM.
    """
    # Input validation
    if not isinstance(filename, str):
            raise TypeError(f"filename should be of type str instead of {type(filename)}")
    if not isinstance(skip_indices, str):
            raise TypeError(f"skip_indices should be of type str instead of {type(skip_indices)}")
    # TODO: validate path

    full_name = filename + '.csv'
    if path is not None:
        full_path = os.path.join(path, full_name)
    else:
        full_path = full_name

    ind_name = skip_indices + '.csv'
    if path is not None:
        ind_path = os.path.join(path, ind_name)
    else:
        ind_path = ind_name
    
    # Read in skip indices
    skip_ind  = []
    with open(ind_path) as f:
        csv_reader = csv.reader(f,delimiter=',')
        for row in csv_reader:
            tmp_data = row
            for n in range(len(tmp_data)):
                skip_ind.append(int(tmp_data[n]))
    
    # Read in samples
    samples = []
    cur_ind = 0
    with open(full_path) as f:
        csv_reader = csv.reader(f,delimiter=',')
        ind_count = 1
        for row in csv_reader:
            if cur_ind < len(skip_ind) and ind_count == skip_ind[cur_ind]:
                cur_ind += 1
                ind_count += 1
                continue
            tmp_data = row
            samples.append(tmp_data)
            ind_count += 1

    samples = np.asarray(samples)

    return samples


"""# Functions for cleaning analyzed data
def graph_log_posterior(post_vec):
    # Chop off the first value (from the initialization) as this is usually very large in magnitude
    posteriors = post_vec[1:-1:1]

    # Plot a graph of the posterior over time
    x_vals = np.arange(len(posteriors))
    fig, ax = plt.subplots(1)
    ax.plot(x_vals, posteriors, color='g')
    plt.xlabel('Iteration')
    plt.ylabel('Log posterior')
    plt.show()

    burn_in = input('Enter the number of burn-in samples to discard: ')
    burn_in = int(burn_in)

    return burn_in"""


def remove_burn_in(b_vec, h_vec, t_vec, f_vec, eta_vec, post_vec, n):
    """
    Removes the first n samples (burn-in).

    Arguments:
    b_vec (numpy array) -- Array of samples of the loads b_(1:M). Each sample is itself an array of M values.
    h_vec (numpy array) -- Array of samples of the step heights h_(1:M). Each sample is itself an array of M values.
    t_vec (numpy array) -- Array of samples of the step times t_(1:M). Each sample is itself an array of M values.
    f_vec (numpy array) -- Array of samples of the background F_bg.
    eta_vec (numpy array) -- Array of samples of the noise variance eta.
    post_vec (numpy array) -- Array of the calculated log posterior for each sample.
    n (int) -- Number of burn-in samples to discard.

    Returns:
    b_vec_clean (numpy array) -- Array of samples of the loads b_(1:M) with burn-in removed.
    h_vec_clean (numpy array) -- Array of samples of the step heights h_(1:M) with burn-in removed.
    t_vec_clean (numpy array) -- Array of samples of the step times t_(1:M) with burn-in removed.
    f_vec_clean (numpy array) -- Array of samples of the background F_bg with burn-in removed.
    eta_vec_clean (numpy array) -- Array of samples of the noise variance eta with burn-in removed.
    post_vec_clean (numpy array) -- Array of log posteriors for each sample with burn-in removed.
    """
    b_vec_clean = np.delete(b_vec, np.s_[0:n], 0)
    h_vec_clean = np.delete(h_vec, np.s_[0:n], 0)
    t_vec_clean = np.delete(t_vec, np.s_[0:n], 0)
    f_vec_clean = np.delete(f_vec, np.s_[0:n], 0)
    eta_vec_clean = np.delete(eta_vec, np.s_[0:n], 0)
    post_vec_clean = np.delete(post_vec, np.s_[0:n], 0)
    
    return b_vec_clean, h_vec_clean, t_vec_clean, f_vec_clean, eta_vec_clean, post_vec_clean


def parallel_bubble_sort(times, data):
    """
    Function for sorting data sets with time points in chronological order.

    Arguments:
    times (array, numpy array) -- Array of time points
    data (array, numpy array) -- Array of observations

    Returns:
    sorted_times (numpy array) -- Array of sorted time points
    sorted_data (numpy array) -- Array of observations sorted according to their time points
    """
    num_times = len(times)
    for i in range(num_times):
        done_sorting = True
        for j in range(num_times - i - 1):
            if times[j] > times[j + 1]:
                times[j], times[j + 1] = times[j + 1], times[j]
                data[j], data[j + 1] = data[j + 1], data[j]
                done_sorting = False
        if done_sorting:
            break

    sorted_times = np.asarray(times)
    sorted_data = np.asarray(data)

    return sorted_times, sorted_data


def find_map(b_vec, h_vec, t_vec, f_vec, eta_vec, posteriors):
    """
    Locates the maximum a posteriori (MAP) estimate sample from the results returned by BNP-Step.

    Arguments:
    b_vec (numpy array) -- Array of samples of the loads b_(1:M). Each sample is itself an array of M values.
    h_vec (numpy array) -- Array of samples of the step heights h_(1:M). Each sample is itself an array of M values.
    t_vec (numpy array) -- Array of samples of the step times t_(1:M). Each sample is itself an array of M values.
    f_vec (numpy array) -- Array of samples of the background F_bg.
    eta_vec (numpy array) -- Array of samples of the noise variance eta.
    posteriors (numpy array) -- Array of the calculated log posterior for each sample.

    Returns:
    b_clean (numpy array) -- Array containing the b_m for the MAP estimate sample.
    h_clean (numpy array) -- Array containing the h_m for the MAP estimate sample.
    t_clean (numpy array) -- Array containing the t_m for the MAP estimate sample.
    f_clean (numpy array) -- MAP estimate value for F_bg
    eta_clean (numpy array) -- MAP estimate value for eta
    """
    map_index = np.argmax(posteriors)
    f_clean = np.asarray(f_vec[int(map_index)])
    b_clean = np.asarray(b_vec[int(map_index)])
    h_clean = np.asarray(h_vec[int(map_index)])
    t_clean = np.asarray(t_vec[int(map_index)])
    eta_clean = np.asarray(eta_vec[int(map_index)])

    return b_clean, h_clean, t_clean, f_clean, eta_clean


def find_top_n_samples(b_vec, h_vec, t_vec, f_vec, eta_vec, posteriors, weak_limit, num_samples=10):
    """
    Picks out the top n samples (how many is specified by the user) from those generated by BNP-Step,
    regardless of how many steps are in the trajectory.

    Note: this function is not used in the original paper, as it is difficult to compare models with
    differing amounts of steps in a rigorous manner. We include this function in case the user wishes
    to directly compare the trajectories of the "best" samples.

    Arguments:
    b_vec (numpy array) -- Array of samples of the loads b_(1:M). Each sample is itself an array of M values.
    h_vec (numpy array) -- Array of samples of the step heights h_(1:M). Each sample is itself an array of M values.
    t_vec (numpy array) -- Array of samples of the step times t_(1:M). Each sample is itself an array of M values.
    f_vec (numpy array) -- Array of samples of the background F_bg.
    eta_vec (numpy array) -- Array of samples of the noise variance eta.
    posteriors (numpy array) -- Array of the calculated log posterior for each sample.
    num_samples (int) -- Amount of samples to return.
    weak_limit (int) -- Maximum number of possible steps

    Returns:
    b_m_top (numpy array) -- Array containing the b_m for the top samples.
    h_m_top (numpy array) -- Array containing the h_m for top samples.
    t_m_top (numpy array) -- Array containing the t_m for top samples.
    f_top (numpy array) -- Top sample values for F_bg
    eta_top (numpy array) -- Top sample values for eta
    """
    f_top = np.zeros(num_samples)
    eta_top = np.zeros(num_samples)
    b_m_top = np.zeros((num_samples, weak_limit))
    h_m_top = np.zeros((num_samples, weak_limit))
    t_m_top = np.zeros((num_samples, weak_limit))

    # Sort the arrays in order of decreasing MAP
    sorting_indices = np.argsort(posteriors)
    sorting_indices = np.flip(sorting_indices)

    for i in range(num_samples):
        f_top[i] = f_vec[int(sorting_indices[i])]
        eta_top[i] = eta_vec[int(sorting_indices[i])]
        for j in range(weak_limit):
            b_m_top[i, j] = b_vec[int(sorting_indices[i]), j]
            h_m_top[i, j] = h_vec[int(sorting_indices[i]), j]
            t_m_top[i, j] = t_vec[int(sorting_indices[i]), j]

    return b_m_top, h_m_top, t_m_top, f_top, eta_top


def find_top_samples_by_jumps(b_vec, h_vec, t_vec, f_vec, eta_vec, posteriors):
    """
    Picks out all samples with the MAP number of steps.

    Arguments:
    b_vec (numpy array) -- Array of samples of the loads b_(1:M). Each sample is itself an array of M values.
    h_vec (numpy array) -- Array of samples of the step heights h_(1:M). Each sample is itself an array of M values.
    t_vec (numpy array) -- Array of samples of the step times t_(1:M). Each sample is itself an array of M values.
    f_vec (numpy array) -- Array of samples of the background F_bg.
    eta_vec (numpy array) -- Array of samples of the noise variance eta.
    posteriors (numpy array) -- Array of the calculated log posterior for each sample.

    Returns: 
    good_b_m (numpy array) -- Array containing the b_m for the samples.
    good_h_m (numpy array) -- Array containing the h_m for the samples.
    good_t_m (numpy array) -- Array containing the t_m for the samples.
    good_f_s (numpy array) -- Sample values for F_bg
    good_eta (numpy array) -- Sample values for eta
    """
    # First determine MAP number of jumps
    map_index = np.argmax(posteriors)
    map_jump_number = np.sum(b_vec[int(map_index)])
    # Count how many jumps are in each sample
    num_jumps = np.sum(b_vec, axis=1)

    # Pick out the samples with the MAP number of steps.
    good_b_m = []
    good_h_m = []
    good_t_m = []
    good_f_s = []
    good_eta = []

    for i in range(len(num_jumps)):
        if num_jumps[i] == map_jump_number:
            good_b_m.append(b_vec[i])
            good_h_m.append(h_vec[i])
            good_t_m.append(t_vec[i])
            good_f_s.append(f_vec[i])
            good_eta.append(eta_vec[i])

    good_b_m = np.asarray(good_b_m)
    good_h_m = np.asarray(good_h_m)
    good_t_m = np.asarray(good_t_m)
    good_f_s = np.asarray(good_f_s)
    good_eta = np.asarray(good_eta)

    return good_b_m, good_h_m, good_t_m, good_f_s, good_eta


# Functions for generating graph-able data
def generate_step_plot_data(b_vec, h_vec, t_vec, f_vec, weak_limit, t_n):
    """
    Generates a plottable trajectory from a MAP estimate sample from BNP-Step.

    Arguments:
    b_vec (numpy array) -- Array of MAP estimate b_m from a BNP-Step sample
    h_vec (numpy array) -- Array of MAP estimate h_m from a BNP-Step sample
    t_vec (numpy array) -- Array of MAP estimate t_m from a BNP-Step sample
    f_vec (numpy array) -- MAP estimate F_bg from a BNP-Step sample
    weak_limit (int) -- Maximum possible number of steps in the data set
    t_n (array, numpy array) -- Array of time points for the trajectory.

    Returns:
    sorted_times (numpy array) -- Array of time points for the trajectory
    sorted_data (numpy array) -- Array of pseudo-observations (calculated using
                                 the samples and the forward model) that define 
                                 the sampled trajectory
    """
    # Count total number of transitions
    jmp_count = 0
    for i in range(weak_limit):
        if b_vec[i] == 1:
            jmp_count += 1
    # Initialize clean arrays to store only 'on' loads
    sampled_loads = np.ones(jmp_count)
    sampled_times = np.zeros(jmp_count)
    sampled_heights = np.zeros(jmp_count)
    # Strip out all the 'off' loads
    ind = 0
    for i in range(weak_limit):
        if b_vec[i] == 1:
            sampled_heights[ind] = h_vec[i]
            sampled_times[ind] = t_vec[i]
            ind += 1
    # Pre-calculate matrices required for vectorized sum calculations
    times_matrix = np.broadcast_to(sampled_times, (jmp_count, jmp_count))
    obs_time_matrix = np.broadcast_to(sampled_times, (jmp_count, jmp_count)).T
    height_matrix = np.broadcast_to(sampled_heights, (jmp_count, jmp_count))
    load_matrix = np.broadcast_to(sampled_loads, (jmp_count, jmp_count))
    # Calculate product of b_m and h_m term-wise
    bh_matrix = np.multiply(load_matrix, height_matrix)
    # Reconstruct "data" based on our sampled values
    for i in range(jmp_count):
        # Calculate Heaviside terms times loads and heights
        bht_matrix = np.multiply(bh_matrix,
                                 np.heaviside((-1 * (obs_time_matrix - times_matrix)),
                                              np.ones((jmp_count, jmp_count))))
    # Calculate sum term - these are the pseudo-observations
    sampled_data = f_vec + np.sum(bht_matrix, axis=1)

    # Make arrays for graphing step plots
    sorted_times, sorted_data = parallel_bubble_sort(sampled_times, sampled_data)
    sorted_times = np.insert(sorted_times, 0, 0)

    # mpl's step plotting functions needs a zero point and an end point to display
    # all the steps correctly
    sorted_times[0] = t_n[0]
    sorted_times = np.append(sorted_times, t_n[int(len(t_n)) - 1])
    # TODO: this "end point" is based on synthetic data sets which end with
    # zero signal. This behavior does not generalize to other sets; fix this!
    sorted_data = np.append(sorted_data, f_vec)

    return sorted_times, sorted_data


def generate_gt_step_plot_data(ground_b_m, ground_h_m, ground_t_m, ground_f, data_times, weak_limit):
    """
    Generates a ground truth trajectory for data sets where the ground truth is known.
    Currently, this only supports the synthetic data sets described in the paper.

    Arguments:
    ground_b_m (numpy array) -- Ground truth loads
    ground_h_m (numpy array) -- Ground truth step heights
    ground_t_m (numpy array) -- Ground truth step times
    ground_f (numpy array) -- Ground truth value for F_bg
    data_times (numpy array) -- Array of time points for the trajectory
    weak_limit (int) -- Maximum possible number of steps in the data set

    Returns:
    sorted_times (numpy array) -- Array of time points for the trajectory
    sorted_data (numpy array) -- Array of pseudo-observations (calculated using
                                 the ground truths and the forward model) that define 
                                 the ground truth trajectory
    """
    # Count total number of jump points
    jmp_count_gnd = 0
    for i in range(weak_limit):
        if (ground_b_m[i] == 1):
            jmp_count_gnd += 1
    # Strip out all non-jump points
    ground_jumps = np.zeros(jmp_count_gnd)
    ground_times = np.zeros(jmp_count_gnd)
    ground_heights = np.zeros(jmp_count_gnd)
    ind = 0
    for i in range(weak_limit):
        if (ground_b_m[i] == 1):
            ground_heights[ind] = ground_h_m[i]
            ground_jumps[ind] = ground_b_m[i]
            ground_times[ind] = ground_t_m[i]
            ind += 1
    # Make array of ground truth data
    # Pre-calculate matrices required for vectorized sum calculations
    times_matrix = np.broadcast_to(ground_times, (jmp_count_gnd, jmp_count_gnd))
    obs_time_matrix = np.broadcast_to(ground_times, (jmp_count_gnd, jmp_count_gnd)).T
    height_matrix = np.broadcast_to(ground_heights, (jmp_count_gnd, jmp_count_gnd))
    load_matrix = np.broadcast_to(ground_jumps, (jmp_count_gnd, jmp_count_gnd))
    # Calculate product of b_m and h_m term-wise
    bh_matrix = np.multiply(load_matrix, height_matrix)
    # Reconstruct "data" based on our sampled values
    for i in range(jmp_count_gnd):
        # Calculate Heaviside terms times loads and heights
        bht_matrix = np.multiply(bh_matrix,
                                 np.heaviside((-1 * (obs_time_matrix - times_matrix)),
                                              np.ones((jmp_count_gnd, jmp_count_gnd))))
    # Calculate sum term - this is your sampled data
    ground_data = ground_f + np.sum(bht_matrix, axis=1)
    # Make arrays for graphing step plots
    sorted_times, sorted_data = parallel_bubble_sort(ground_times, ground_data)
    sorted_times = np.insert(sorted_times, 0, 0)

    sorted_times[0] = data_times[0]
    sorted_times = np.append(sorted_times, data_times[int(len(data_times)) - 1])
    sorted_data = np.append(sorted_data, ground_f)

    return sorted_times, sorted_data


def generate_kv_step_plot_data(jump_times, heights, background, data_times):
    """
    Generates step plot data from BIC-based method results.

    Arguments:
    jump_times (numpy array) -- Array of jump times returned by BIC method
    heights (numpy array) -- Array of inter-step means returned by BIC method. Does not include the final mean.
    background (float) -- Value of final inter-step mean returned by BIC method.
    data_times (numpy array) -- Array of time points

    Returns:
    plot_times (numpy array) -- Array of time points for the trajectory
    plot_heights (numpy array) -- Array of pseudo-observations (calculated using
                                  the BIC results and the forward model) that define 
                                  the learned trajectory
    """
    # Add first observation time point to start of array, and duplicate end point (otherwise
    # mpl's stairs function will fail)
    plot_times = np.append(jump_times, data_times[int(len(data_times))-1])
    plot_times = np.insert(plot_times, 0, data_times[0])

    # Re append the background value
    # TODO: look into streamlining this by never separating the final mean in the first place
    plot_heights = np.append(heights, background)

    return plot_times, plot_heights


def generate_histogram_data(b_vec, h_vec, t_vec, num_samples, weak_limit, times):
    """
    Processes raw BNP-Step results into a format that can be histogrammed.
    Note: it is strongly recommended to use only the samples with the MAP number
    of jumps.
    # TODO: clarify why this is in simple language

    Arguments:
    b_vec (numpy array) -- Array of b_m from BNP-Step samples
    h_vec (numpy array) -- Array of h_m from BNP-Step samples
    t_vec (numpy array) -- Array of t_m from BNP-Step samples
    num_samples (int) -- Number of samples kept for histogramming
    weak_limit (int) -- Maximum possible number of steps in the data set

    Returns:
    histogram_heights (numpy array) -- Array of absolute values of the step heights
    histogram_lengths (numpy array) -- Array of holding times between the steps
    """
    # Prepare data for histogramming
    histogram_heights = []
    histogram_lengths = []

    for i in range(num_samples):
        temp_times = []
        for j in range(weak_limit):
            if b_vec[i, j] == 1:
                histogram_heights.append(h_vec[i, j])
                temp_times.append(t_vec[i, j])
        temp_times = np.sort(temp_times)
        temp_times = np.insert(temp_times, 0, times[0])
        temp_times = np.append(temp_times, times[-1])
        for j in range(len(temp_times)):
            if j != 0:
                histogram_lengths.append(temp_times[j] - temp_times[j - 1])

    histogram_heights = np.absolute(np.asarray(histogram_heights))
    histogram_lengths = np.asarray(histogram_lengths)

    return histogram_heights, histogram_lengths


def generate_histogram_data_ihmm(samples, times):
    """
    Converts iHMM results to step height form for histogramming. Also
    returns the holding times.

    Arguments:
    samples (numpy array) -- Array of iHMM samples with the mode number of states.
    times (numpy array) -- Time points for each observation

    Returns:
    histogram_heights (numpy array) -- Array of absolute values of the step heights
    histogram_lengths (numpy array) -- Array of holding times between the steps
    """
    histogram_heights = []
    histogram_times = []

    num_samples = samples.shape[0]
    traj_len = samples.shape[1]

    for i in range(num_samples):
        time_prev = times[0]
        for j in range(traj_len):
            if j == 0:
                continue
            else:
                if samples[i, j] != samples[i, j - 1]:
                    histogram_times.append(times[j] - time_prev)
                    time_prev = times[j]
                    histogram_heights.append(float(samples[i, j]) - float(samples[i, j - 1]))

    histogram_heights = np.absolute(np.asarray(histogram_heights))
    histogram_times = np.asarray(histogram_times)

    return histogram_heights, histogram_times


def generate_histogram_data_emission(b_vec, h_vec, t_vec, f_vec, weak_limit):
    """
    Generates histogrammable data sets for comparison to iHMM method.
    In this case, emission levels are calculated for histogramming,
    rather than just using the step heights themselves.

    Arguments:
    b_vec (numpy array) -- Array of b_m from BNP-Step samples
    h_vec (numpy array) -- Array of h_m from BNP-Step samples
    t_vec (numpy array) -- Array of t_m from BNP-Step samples
    f_vec (numpy array) -- F_bg from BNP-Step samples
    weak_limit (int) -- Maximum possible number of steps in the data set

    Returns:
    all_sorted_times (numpy array) -- Numpy array of all step times
    all_sorted_data (numpy array) -- Numpy array of all emission levels
    """
    # Count total number of samples and total number of jump points
    num_samps = b_vec.shape[0]
    jmp_count = 0
    for i in range(weak_limit):
        if b_vec[0,i] == 1:
            jmp_count += 1
    # Initialize sample arrays
    all_sorted_times = []
    all_sorted_data = []
    # Process samples
    for x in range(num_samps):
        # Initialize clean arrays to store only 'on' loads
        sampled_loads = np.ones(jmp_count)
        sampled_times = np.zeros(jmp_count)
        sampled_heights = np.zeros(jmp_count)
        # Strip out all non-jump points
        ind = 0
        for i in range(weak_limit):
            if b_vec[x,i] == 1:
                sampled_heights[ind] = h_vec[x,i]
                sampled_times[ind] = t_vec[x,i]
                ind += 1
        # Pre-calculate matrices required for vectorized sum calculations
        times_matrix = np.broadcast_to(sampled_times, (jmp_count, jmp_count))
        obs_time_matrix = np.broadcast_to(sampled_times, (jmp_count, jmp_count)).T
        height_matrix = np.broadcast_to(sampled_heights, (jmp_count, jmp_count))
        load_matrix = np.broadcast_to(sampled_loads, (jmp_count, jmp_count))
        # Calculate product of b_m and h_m term-wise
        bh_matrix = np.multiply(load_matrix, height_matrix)
        # Reconstruct "data" based on our sampled values
        for i in range(jmp_count):
            # Calculate Heaviside terms times loads and heights
            bht_matrix = np.multiply(bh_matrix,
                                    np.heaviside((-1 * (obs_time_matrix - times_matrix)),
                                                np.ones((jmp_count, jmp_count))))
        # Calculate sum term - this is your sampled data
        sampled_data = f_vec[x] + np.sum(bht_matrix, axis=1)

        # Sort then append to sample array
        sorted_times, sorted_data = parallel_bubble_sort(sampled_times, sampled_data)
        sorted_data = np.append(sorted_data, f_vec[x])
        if x == 0:
            all_sorted_times = sorted_times.copy()
            all_sorted_data = sorted_data.copy()
        else:
            all_sorted_times = np.vstack((all_sorted_times, sorted_times))
            all_sorted_data = np.vstack((all_sorted_data, sorted_data))
            
    return all_sorted_times, all_sorted_data


def generate_histogram_data_kv(heights, jumptimes):
    """
    Processes BIC-based method results into a format that can be histogrammed.

    Arguments:
    heights (numpy array) -- Array of inter-step means from BIC-based method
    jumptimes (numpy array) -- Array of jump times from BIC-based method

    Note: this returns a frequentist-style histogram. For a single data set,
    histograms of the results may not be useful unless step heights and holding
    times are repeated frequently (as is the case with the synthetic data sets
    used in the paper.)

    Returns:
    histogram_heights (numpy array) -- Array of absolute values of the step heights
    histogram_lengths (numpy array) -- Array of holding times between the steps
    """
    # Prepare data for histogramming
    histogram_heights = []
    histogram_lengths = []

    for i in range(len(heights)):
        if i == 0:
            continue
        else:
            histogram_heights.append(heights[i]-heights[i-1])
    jumptimes = np.sort(jumptimes)
    for j in range(len(jumptimes)):
        if j != 0:
            histogram_lengths.append(jumptimes[j] - jumptimes[j - 1])

    histogram_heights = np.absolute(np.asarray(histogram_heights))
    histogram_lengths = np.asarray(histogram_lengths)

    return histogram_heights, histogram_lengths


"""# Functions for calculating log-posterior and log-likelihood - used to calculate the ground truth values
# for synthetic data.
def calculate_gt_loglikelihood(weak_limit, num_data, data_points, data_times, b_m_vec, h_m_vec,
                               t_m_vec, f_vec, eta_vec):
    # Calculate matrices required for vectorized sum calculations
    times_matrix = np.broadcast_to(t_m_vec, (num_data, weak_limit))
    obs_time_matrix = np.broadcast_to(data_times, (weak_limit, num_data)).T
    height_matrix = np.broadcast_to(h_m_vec, (num_data, weak_limit))
    load_matrix = np.broadcast_to(b_m_vec, (num_data, weak_limit))

    # Calculate product of b_m and h_m term-wise
    bh_matrix = np.multiply(load_matrix, height_matrix)
    # Calculate Heaviside terms times loads and heights
    bht_matrix = np.multiply(bh_matrix,
                             np.heaviside((-1 * (obs_time_matrix - times_matrix)),
                                          np.ones((num_data, weak_limit))))
    # Calculate sum term for the exponent
    bht_sum = f_vec + np.sum(bht_matrix, axis=1)
    exponent_term = (eta_vec / 2) * np.sum(np.square(data_points - bht_sum))
    # Calculate log likelihood
    new_log_likelihood = ((num_data / 2) * np.log(eta_vec / (2 * np.pi))) - exponent_term

    return new_log_likelihood


def calculate_gt_logposterior(weak_limit, num_data, data_points, data_times, b_m_vec, h_m_vec,
                              t_m_vec, f_vec, eta_vec, chi, h_ref, gamma, phi, eta_ref, psi, f_ref):
    # Calculate the log likelihood
    log_likelihood = calculate_gt_loglikelihood(weak_limit, num_data, data_points, data_times, b_m_vec, h_m_vec,
                                                t_m_vec, f_vec, eta_vec)

    # Calculate priors on b_m
    b_m = np.asarray(b_m_vec)
    on_prior = (gamma / weak_limit) * b_m
    off_prior = (np.ones(weak_limit) - b_m) - ((gamma / weak_limit) * (np.ones(weak_limit) - b_m))
    prior_b_m = np.sum(np.log(on_prior + off_prior))

    # Calculate priors on h_m
    h_m = np.asarray(h_m_vec)
    prior_h_m = ((weak_limit / 2) * np.log(chi / (2 * np.pi))) - ((chi / 2) * np.sum(np.square(h_m - h_ref)))

    # Calculate priors on t_m
    prior_t_m = -weak_limit * np.log(num_data)

    # Calculate prior on eta
    prior_eta = ((phi - 1) * np.log(eta_vec)) - ((phi * eta_vec) / eta_ref) - np.log(sp.special.gamma(phi)) - \
                (phi * np.log(eta_ref / phi))

    # Calculate prior on F
    prior_f = ((1 / 2) * np.log(psi / (2 * np.pi))) - ((psi / 2) * ((f_vec - f_ref) ** 2))

    log_posterior = log_likelihood + prior_b_m + prior_h_m + prior_t_m + prior_eta + prior_f

    return log_posterior, log_likelihood, prior_b_m, prior_h_m, prior_t_m, prior_f, prior_eta


# Calculate auto-correlation among a set of samples
def autocorrelation(sample_vector, threshold):
    # Set up variables
    total_samples = len(sample_vector)
    mean = (1 / total_samples) * np.sum(sample_vector)
    denominator = np.sum(np.square(sample_vector - mean))
    distance = 1
    not_done = True
    while not_done:
        first_term = sample_vector[:(total_samples - distance - 1)] - mean
        second_term = sample_vector[(1 + distance):] - mean
        numerator = np.sum(np.multiply(first_term, second_term))
        rho = numerator / denominator
        # Check threshold
        if np.sqrt(rho ** 2) < threshold:
            not_done = False
        else:
            distance += 1
    return distance

# Calculates the inverse cumulative distribution for a data set
def make_inverse_dist(t_m, h_m):
    step_lengths = []
    for i in range(len(t_m)):
        if i == 0:
            continue
        else:
            step_lengths.append(t_m[i] - t_m[i-1])
    unfolding = []
    folding = []
    for i in range(len(step_lengths)):
        if i == 0:
            continue
        else:
            if h_m[i] > h_m[i-1]:
                unfolding.append(step_lengths[i-1])
            elif h_m[i] < h_m[i-1]:
                folding.append(step_lengths[i-1])
            else:
                continue
    folding = np.sort(np.asarray(folding))
    survivors_folding = np.flip(np.arange(0,len(folding)+1))
    survivors_folding = (1/len(folding))*survivors_folding
    folding = np.insert(folding, 0, 0)
    folding = np.append(folding, folding[-1])
    
    unfolding = np.sort(np.asarray(unfolding))
    survivors_unfolding = np.flip(np.arange(0,len(unfolding)+1))
    survivors_unfolding = (1/len(unfolding))*survivors_unfolding
    unfolding = np.insert(unfolding, 0, 0)
    unfolding = np.append(unfolding, unfolding[-1])
    
    return folding, unfolding, survivors_folding, survivors_unfolding

def make_inverse_dist_unfolding(t_m, h_m, lower_limit, upper_limit):
    step_lengths = []
    for i in range(len(t_m)):
        if i == 0:
            continue
        else:
            step_lengths.append(t_m[i] - t_m[i-1])
    unfolding_short = []
    unfolding_long = []
    folding = []
    for i in range(len(step_lengths)):
        if i == 0:
            continue
        else:
            if (h_m[i] > h_m[i-1] and np.abs(h_m[i] - h_m[i-1]) > upper_limit):
                unfolding_long.append(step_lengths[i-1])
            elif (h_m[i] > h_m[i-1] and np.abs(h_m[i] - h_m[i-1]) > lower_limit):
                unfolding_short.append(step_lengths[i-1])
            elif (h_m[i] < h_m[i-1] and np.abs(h_m[i] - h_m[i-1]) > lower_limit):
                folding.append(step_lengths[i-1])
            else:
                continue
    folding = np.sort(np.asarray(folding))
    
    survivors_folding = np.flip(np.arange(0,len(folding)+1))
    survivors_folding = (1/len(folding))*survivors_folding
    folding = np.insert(folding, 0, 0)
    folding = np.append(folding, folding[-1])

    unfolding_short = np.sort(np.asarray(unfolding_short))
    survivors_unfolding_short = np.flip(np.arange(0,len(unfolding_short)+1))
    survivors_unfolding_short = (1/len(unfolding_short))*survivors_unfolding_short
    unfolding_short = np.insert(unfolding_short, 0, 0)
    unfolding_short = np.append(unfolding_short, unfolding_short[-1])

    unfolding_long = np.sort(np.asarray(unfolding_long))
    survivors_unfolding_long = np.flip(np.arange(0,len(unfolding_long)+1))
    survivors_unfolding_long = (1/len(unfolding_long))*survivors_unfolding_long
    unfolding_long = np.insert(unfolding_long, 0, 0)
    unfolding_long = np.append(unfolding_long, unfolding_long[-1])
    
    return folding, unfolding_short, unfolding_long, survivors_folding, survivors_unfolding_short, survivors_unfolding_long"""

def get_credible_intervals(states):
    """
    Calculates the credible intervals associated with an array.

    Arguments:
    states (numpy array) -- Array of values for which CI's will be calculated.

    Returns:
    mean -- Mean of the values in the array
    under95 -- Left boundary for 95% CI
    under50 -- Left boundary for 50% CI
    median -- Median of the values in the array
    upper50 -- Right boundary for 50% CI
    upper95 -- Right boundary for 95% CI
    """
    mean = np.mean(states)
    under95 = np.percentile(states,2.5)
    under50 = np.percentile(states,25)
    median = np.percentile(states,50)
    upper50 = np.percentile(states,75)
    upper95 = np.percentile(states,97.5)
    return mean, under95, under50, median, upper50, upper95

"""def get_confidence_wide(states):
    mean = np.mean(states)
    under99 = np.percentile(states,2.5)
    under50 = np.percentile(states,25)
    median = np.percentile(states,50)
    upper50 = np.percentile(states,75)
    upper99 = np.percentile(states,97.5)
    low = np.min(states)
    high = np.max(states)
    return mean,under99,under50,median,upper50,upper99,low,high

def calculate_SIC(input_data, b_vec, eta_vec):
    n = len(input_data)
    k = np.sum(b_vec)
    return (k + 2)*np.log(n) + n*np.log(eta_vec)

def find_unique_nosort(array):
    result = []
    for i in range(len(array)):
        if i == 0:
            result.append(array[i])
        else:
            if array[i-1] != array[i]:
                result.append(array[i])
    return np.asarray(result)

# Functions for outputting cleaned-up data to a file (useful for documentation of
# what was used in figures)
def output_clean_data(base_path, filenum, file_label, b_vec, h_vec, t_vec, f_vec, eta_vec):
    with open(base_path + 'B_' + file_label + filenum + '.txt', 'w') as outfile4:
        b_vec.tofile(outfile4, sep="\n")

    with open(base_path + 'F_' + file_label + filenum + '.txt', 'w') as outfile5:
        f_vec.tofile(outfile5, sep="\n")

    with open(base_path + 'H_' + file_label + filenum + '.txt', 'w') as outfile6:
        h_vec.tofile(outfile6, sep="\n")

    with open(base_path + 'T_' + file_label + filenum + '.txt', 'w') as outfile7:
        t_vec.tofile(outfile7, sep="\n")

    with open(base_path + 'ETA_' + file_label + filenum + '.txt', 'w') as outfile8:
        eta_vec.tofile(outfile8, sep="\n")"""
