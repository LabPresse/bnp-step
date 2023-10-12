import numpy as np
import scipy as sp
from scipy import special
import matplotlib.pyplot as plt
import csv

"""
def load_clean_data(filetype, filenum):
    # Get parameters of data sets
    with open('params' + filenum + '.txt', 'r') as f:
        # Read out all values to strings
        gt_num_jumps_str = f.readline().strip()
        num_data_str = f.readline().strip()
        t_aqr_str = f.readline().strip()
        t_exp_str = f.readline().strip()
        f_gt_str = f.readline().strip()
        h_stp_str = f.readline().strip()
        t_stp_str = f.readline().strip()
        eta_str = f.readline().strip()
        t_min_str = f.readline().strip()
        weak_limit_str = f.readline().strip()
        num_samples_str = f.readline().strip()

    # Convert strings to floats
    gt_num_jumps = int(float(gt_num_jumps_str))
    num_data = int(float(num_data_str))
    t_aqr = float(t_aqr_str)
    t_exp = float(t_exp_str)
    f_gt = float(f_gt_str)
    h_stp = float(h_stp_str)
    t_stp = float(t_stp_str)
    eta = float(eta_str)
    t_min = float(t_min_str)
    weak_limit = int(float(weak_limit_str))

    # Determine the actual number of samples we have.
    num_samples = 0
    with open('ETA_' + filetype + filenum + '.txt', 'r') as f:
        item = f.readline().strip()
        while item != '':
            num_samples += 1
            item = f.readline().strip()

    # Arrange the parameters into a numpy array
    parameters = [gt_num_jumps, num_data, t_aqr, t_exp, f_gt, h_stp, t_stp, eta, t_min, weak_limit, num_samples]
    parameters = np.asarray(parameters)

    # Initialize numpy vectors to store our samples
    eta_vec = np.zeros(num_samples)
    f_vec = np.zeros(num_samples)
    b_vec = np.zeros((num_samples, weak_limit))
    h_vec = np.zeros((num_samples, weak_limit))
    t_vec = np.zeros((num_samples, weak_limit))
    posteriors = np.zeros(num_samples)
    data = np.zeros(num_data)
    t_n = np.zeros(num_data)

    # Open files and read data
    # TODO: see if there is a faster way to do this
    with open('B_' + filetype + filenum + '.txt', 'r') as f:
        for i in range(num_samples):
            for j in range(weak_limit):
                item = f.readline().strip()
                value = float(item)
                b_vec[i, j] = value

    with open('H_' + filetype + filenum + '.txt', 'r') as f:
        for i in range(num_samples):
            for j in range(weak_limit):
                item = f.readline().strip()
                value = float(item)
                h_vec[i, j] = value

    with open('T_' + filetype + filenum + '.txt', 'r') as f:
        for i in range(num_samples):
            for j in range(weak_limit):
                item = f.readline().strip()
                value = float(item)
                t_vec[i, j] = value

    with open('ETA_' + filetype + filenum + '.txt', 'r') as f:
        for i in range(num_samples):
            item = f.readline().strip()
            value = float(item)
            eta_vec[i] = value

    with open('F_' + filetype + filenum + '.txt', 'r') as f:
        for i in range(num_samples):
            item = f.readline().strip()
            value = float(item)
            f_vec[i] = value

    with open('posteriors' + filenum + '.txt', 'r') as f:
        for i in range(num_samples):
            item = f.readline().strip()
            value = float(item)
            posteriors[i] = value

    # Correct posteriors for nan's - change them to -inf <--------------------------------------------------
    for x in range(len(posteriors)):
        if np.isnan(posteriors[x]):
            posteriors[x] = -np.inf

    # Load synthetic data
    with open('synthdata' + filenum + '.txt', 'r') as f:
        for i in range(num_data):
            item = f.readline().strip()
            value = float(item)
            data[i] = value
        for i in range(num_data):
            item = f.readline().strip()
            value = float(item)
            t_n[i] = value

    return parameters, b_vec, h_vec, t_vec, f_vec, eta_vec, posteriors, data, t_n

def load_ground_truth_ihmm(num_data, filenum):
    ground_traj = np.zeros(num_data)
    ground_t = np.zeros(num_data)
    with open('groundtruths' + filenum + '.txt', 'r') as f:
        for i in range(num_data):
            item = f.readline().strip()
            value = float(item)
            ground_traj[i] = value
        for i in range(num_data):
            item = f.readline().strip()
            value = float(item)
            ground_t[i] = value
    return ground_traj, ground_t

def load_kv_data(filenum):
    # Get parameters of datasets
    with open('params_KV' + filenum + '.txt', 'r') as f:
        # Read out all values to strings
        B_str = f.readline().strip()
        N_str = f.readline().strip()
        t_aqr_str = f.readline().strip()
        t_exp_str = f.readline().strip()
        F_str = f.readline().strip()
        h_stp_str = f.readline().strip()
        t_stp_str = f.readline().strip()
        eta_str = f.readline().strip()
        t_min_str = f.readline().strip()
        B_max_str = f.readline().strip()
    
    # Convert strings to floats
    B_file = float(B_str)
    N_file = float(N_str)
    t_aqr = float(t_aqr_str)
    t_exp = float(t_exp_str)
    F = float(F_str)
    h_stp = float(h_stp_str)
    t_stp = float(t_stp_str)
    eta = float(eta_str)
    t_min = float(t_min_str)
    B_max_file = float(B_max_str)

    # Change eta to standard deviation
    eta = 1/(np.sqrt(eta))

    # Convert certain values to ints
    B = int(B_file)
    N = int(N_file)
    B_max = int(B_max_file)

    # Arrange the parameters into a numpy array
    parameters = [B, N, t_aqr, t_exp, F, h_stp, t_stp, eta, t_min, B_max]
    parameters = np.asarray(parameters)

    # Extract synthetic data
    data = np.zeros(N)
    t_n = np.zeros(N)
    with open('synthdata_KV' + filenum + '.txt', 'r') as f:
        for i in range(N):
            item = f.readline().strip()
            value = float(item)
            data[i] = value
        for i in range(N):
            item = f.readline().strip()
            value = float(item)
            t_n[i] = value
    
    # Extract learned values
    ETA = np.zeros(1)
    H_M = np.zeros(1)
    T_M = np.zeros(1)

    with open('heights_KV' + filenum + '.txt', 'r') as f:
        item = f.readline().strip()
        if (item != ''):
            value = float(item)
            H_M[0] = value
            item = f.readline().strip()
            while (item != ''):
                value = float(item)
                H_M = np.append(H_M, [value])
                item = f.readline().strip()
    
    # The last KV height is the "background" - strip it from height array
    F_S = H_M[-1]
    heights = np.zeros(H_M.size-1)
    for i in range(H_M.size-1):
        heights[i] = H_M[i]
    
    with open('jumptimes_KV' + filenum + '.txt', 'r') as f:
        item = f.readline().strip()
        while (item != ''):
            value = float(item)
            T_M = np.append(T_M, [value])
            item = f.readline().strip()
    
    with open('variance_KV' + filenum + '.txt', 'r') as f:
        item = f.readline().strip()
        if (item != ''):
            value = float(item)
            ETA[0] = value
            item = f.readline().strip()
            while (item != ''):
                value = float(item)
                ETA = np.append(ETA, [value])
                item = f.readline().strip()
    
    return parameters, H_M, T_M, F_S, ETA, data, t_n
    
def load_ihmm_heights(filenum):
    heights = []
    with open('mode_means_' + filenum + '.csv') as f:
        csv_reader = csv.reader(f,delimiter=',')
        for row in csv_reader:
            tmp_data = row
            for n in range(len(tmp_data)):
                heights.append(float(tmp_data[n]))
    heights = np.asarray(heights)
    return heights"""

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


"""def remove_burn_in(b_vec, h_vec, t_vec, f_vec, eta_vec, post_vec, burn_in):
    b_vec_clean = np.delete(b_vec, np.s_[0:burn_in], 0)
    h_vec_clean = np.delete(h_vec, np.s_[0:burn_in], 0)
    t_vec_clean = np.delete(t_vec, np.s_[0:burn_in], 0)
    f_vec_clean = np.delete(f_vec, np.s_[0:burn_in], 0)
    eta_vec_clean = np.delete(eta_vec, np.s_[0:burn_in], 0)
    post_vec_clean = np.delete(post_vec, np.s_[0:burn_in], 0)
    return b_vec_clean, h_vec_clean, t_vec_clean, f_vec_clean, eta_vec_clean, post_vec_clean"""


# Functions for sorting analyzed data - helper function for creating
# graph-able results from BNP samples.
# TODO: this is slow, make it faster
def parallel_bubble_sort(times, data):
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
    map_index = np.argmax(posteriors)
    f_clean = np.asarray(f_vec[int(map_index)])
    b_clean = np.asarray(b_vec[int(map_index)])
    h_clean = np.asarray(h_vec[int(map_index)])
    t_clean = np.asarray(t_vec[int(map_index)])
    eta_clean = np.asarray(eta_vec[int(map_index)])

    return b_clean, h_clean, t_clean, f_clean, eta_clean


"""def find_top_samples(b_vec, h_vec, t_vec, f_vec, eta_vec, posteriors, num_samples, weak_limit):
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

    return good_b_m, good_h_m, good_t_m, good_f_s, good_eta"""


# Functions for generating graph-able data
def generate_step_plot_data(b_vec, h_vec, t_vec, f_vec, weak_limit, t_n):
    # Count total number of transitions
    jmp_count = 0
    for i in range(weak_limit):
        if b_vec[i] == 1:
            jmp_count += 1
    # Initialize clean arrays to store only 'on' loads
    sampled_loads = np.ones(jmp_count)
    sampled_times = np.zeros(jmp_count)
    sampled_heights = np.zeros(jmp_count)
    # Strip out all non-jump points
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
    # Calculate sum term - this is your sampled data
    sampled_data = f_vec + np.sum(bht_matrix, axis=1)

    # Make arrays for graphing step plots
    sorted_times, sorted_data = parallel_bubble_sort(sampled_times, sampled_data)
    sorted_times = np.insert(sorted_times, 0, 0)

    sorted_times[0] = t_n[0]
    sorted_times = np.append(sorted_times, t_n[int(len(t_n)) - 1])
    sorted_data = np.append(sorted_data, f_vec)

    return sorted_times, sorted_data

def generate_gt_step_plot_data(ground_b_m, ground_h_m, ground_t_m, ground_f, data_times, weak_limit):
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

"""def generate_kv_step_plot_data(jump_times, heights, background, data_times):
    plot_times = np.append(jump_times, data_times[int(len(data_times))-1])
    plot_times = np.append(plot_times, data_times[int(len(data_times))-1])
    plot_times[0] = data_times[0]
    plot_heights = np.append(heights, background)

    return plot_times, plot_heights"""

"""def generate_histogram_data(b_vec, h_vec, t_vec, num_samples, weak_limit):
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
        for j in range(len(temp_times)):
            if j != 0:
                histogram_lengths.append(temp_times[j] - temp_times[j - 1])

    histogram_heights = np.absolute(np.asarray(histogram_heights))
    histogram_lengths = np.asarray(histogram_lengths)

    return histogram_heights, histogram_lengths

def generate_histogram_data_ihmm(b_vec, h_vec, t_vec, f_vec, weak_limit):
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


# Functions for calculating log-posterior and log-likelihood - used to calculate the ground truth values
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
    
    return folding, unfolding_short, unfolding_long, survivors_folding, survivors_unfolding_short, survivors_unfolding_long

def get_confidence(states):
    mean = np.mean(states)
    under95 = np.percentile(states,2.5)
    under50 = np.percentile(states,25)
    median = np.percentile(states,50)
    upper50 = np.percentile(states,75)
    upper95 = np.percentile(states,97.5)
    return mean,under95,under50,median,upper50,upper95

def get_confidence_wide(states):
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
