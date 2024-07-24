"""
bnpinputs: Contains all functions that load data sets for BNP-Step.
"""
import os
import numpy as np
import pandas as pd


def load_data_txt(filename: str, 
                  has_timepoints: bool, 
                  path = None
                  ):
    """
    Data loader for generic data sets in .txt format.

    Files with no time points must be newline (\n) delimited.

    For files with time points, each entry must consist of time, observation pairs 
    delimited by a comma without spaces, and individual time-observation pairs 
    must be newline (\n) delimited.

    Arguments:
    filename (str) -- Name of the file to be loaded. Must not contain the file extension.
    has_timepoints (bool) -- Whether or not the file has time points associated with observations.
    path -- Path to the file to be loaded. If path is None, the file must be in the same 
            directory as bnpstep.py. Default: None

    Returns:
    Dictionary with data (numpy array), time points (numpy array or None), ground_truths (None), and parameters (None).
    """
    # Validate input and construct paths
    if not isinstance(filename, str):
        raise TypeError(f"filename should be of type str, got {type(filename)}")
    # TODO: validate path
    full_name = filename + '.txt'
    if path is not None:
        full_path = os.path.join(path, full_name)
    else:
        full_path = full_name
    
    # Read in data from file
    dataset = {}
    if has_timepoints:
        data = []
        times = []
        # TODO: validate observation formatting
        with open(full_path, 'r') as f:
            line = f.readline().strip()
            while (line != ''):
                split_data = line.split(',')
                times.append(split_data[0])
                data.append(split_data[1])
                line = f.readline().strip()
        
        # Convert data and times arrays to numpy arrays
        data = np.asarray(data).astype(float)
        times = np.asarray(times).astype(float)

        # Build dictionary for output
        dataset["data"] = data
        dataset["times"] = times

    else:
        data = []
        # TODO: validate file is newline delimited
        with open(full_path, 'r') as f:
            line = f.readline().strip()
            while (line != ''):
                data.append(line)
                line = f.readline().strip()
        
        # Convert data array to numpy array
        data = np.asarray(data)

        # Build dictionary for output
        dataset["data"] = data.astype(float)
        dataset["times"] = None
    
    dataset["ground_truths"] = None
    dataset["parameters"] = None

    return dataset


def load_data_csv(filename: str, 
                  has_timepoints: bool, 
                  path = None
                  ):
    """
    Data loader for generic data sets in .csv format.

    For files without time points, observations may come in single row or single column format.

    For files with time points, observations must be formatted with one time, observation pair per row, 
    with the times in the first column and the observations in the second column.

    Arguments:
    filename (str) -- Name of the file to be loaded. Must not contain the file extension.
    has_timepoints (bool) -- Whether or not the file has time points associated with observations.
    path -- Path to the file to be loaded. If path is None, the file must be in the same 
            directory as bnpstep.py. Default: None

    Returns:
    Dictionary with data (numpy array), time points (numpy array or None), ground_truths (None), and parameters (None).
    """
    # Validate input and build path
    if not isinstance(filename, str):
            raise TypeError(f"filename should be of type str, got {type(filename)}")
    # TODO: validate path
    full_name = filename + '.csv'
    if path is not None:
        full_path = os.path.join(path, full_name)
    else:
        full_path = full_name
    
    # Read in data from files
    dataset = {}
    if has_timepoints:
        data_fr = pd.read_csv(full_path, header=None)
        data_np = pd.DataFrame.to_numpy(data_fr)
        times = data_np[:, 0]
        data = data_np[:, 1]
        
        # Build dictionary for output
        dataset["data"] = data
        dataset["times"] = times

    else:
        data_fr = pd.read_csv(full_path, header=None)
        data_np = pd.DataFrame.to_numpy(data_fr)
        data = np.ravel(data_np)

        # Build dictionary for output
        dataset["data"] = data
        dataset["times"] = None
    
    dataset["ground_truths"] = None
    dataset["parameters"] = None

    return dataset


def load_data_HMM(filename: str, 
                  path = None
                  ):
    """
    Data loader for HMM-style data sets, as given in "An accurate probabilistic step finder for time-series analysis",
    doi: 10.1101/2023.09.19.558535

    This function ONLY supports .csv format. 

    Arguments:
    filename (str) -- Name of the file to be loaded. Must not contain the file extension.
    path -- Path to the file to be loaded. If path is None, the file must be in the same 
            directory as bnpstep.py. Default: None

    Returns:
    Dictionary with data (numpy array), time points (numpy array), ground_truths (dict), and parameters (dict).
    """
    # Validate input and build path
    if not isinstance(filename, str):
            raise TypeError(f"filename should be of type str, got {type(filename)}")
    # TODO: validate path
    full_name = filename + '.csv'
    if path is not None:
        full_path = os.path.join(path, full_name)
    else:
        full_path = full_name

    ### Load all data from the csv file
    data_fr = pd.read_csv(full_path)
    data_mat = pd.DataFrame.to_numpy(data_fr)
    times = data_mat[:, 0]
    data = data_mat[:, 1]
    # Extract ground truth trajectory data
    ground = {"x": data_mat[:, 2], "u": data_mat[:, 3]}
    # Extract synthetic data generation parameters
    params = {}
    params["type"] = 'hmm'
    # Count the ground truth number of steps (transitions) in the data
    ctl = 1
    num_steps = 0
    while (ctl < ground["x"].size):
        if (ground["x"][ctl] != ground["x"][ctl - 1]):
            num_steps += 1
        ctl += 1
    params["gt_steps"] = num_steps
    params["num_observations"] = ground["x"].size
    params["f_back"] = float(data_fr.columns[2])
    if len(np.unique(data_mat[:, 2])) > 2:
        params["h_step"] = float(data_fr.columns[5])
        params["h_step2"] = float(data_fr.columns[6])
        params["h_step3"] = float(data_fr.columns[7])
        params["h_step4"] = float(data_fr.columns[8])
        params["h_step5"] = float(data_fr.columns[9])
    else:
        params["h_step"] = 0
        params["h_step2"] = float(data_fr.columns[5])
        params["h_step3"] = None
        params["h_step4"] = None
        params["h_step5"] = None
    params["eta"] = float(data_fr.columns[3])

    # Pack everything into a dict
    dataset = {}
    dataset["data"] = data
    dataset["times"] = times
    
    dataset["ground_truths"] = ground
    dataset["parameters"] = params

    return dataset


def load_data_expt(filename: str,
                   path = None
                   ):
    """
    Data loader for experimental data sets, as given in "An accurate probabilistic step finder for time-series analysis",
    doi: 10.1101/2023.09.19.558535

    This function ONLY supports .txt format. 

    Arguments:
    filename (str) -- Name of the file to be loaded. Must not contain the file extension.
    path -- Path to the file to be loaded. If path is None, the file must be in the same 
            directory as bnpstep.py. Default: None

    Returns:
    Dictionary with data (numpy array), time points (numpy array), ground_truths (None), and parameters (None).
    """
    # Validate input and build path
    if not isinstance(filename, str):
            raise TypeError(f"filename should be of type str, got {type(filename)}")
    # TODO: validate path
    full_name = filename + '.txt'
    if path is not None:
        full_path = os.path.join(path, full_name)
    else:
        full_path = full_name

    times = []
    data = []
    with open(full_path, 'r') as f:
        line = f.readline().strip()
        while (line != ''):
            split_data = line.split(',')
            times.append(split_data[0])
            data.append(split_data[1])
            line = f.readline().strip()

    # Convert data and times arrays to numpy arrays
    data = np.asarray(data).astype(float)
    times = np.asarray(times).astype(float)

    # Build dictionary for output
    dataset = {}
    dataset["data"] = data
    dataset["times"] = times

    dataset["ground_truths"] = None
    dataset["parameters"] = None

    return dataset


def load_data_kv(filename: str,
                 path = None
                 ):
    """
    Data loader for KV-type data sets, as given in "An accurate probabilistic step finder for time-series analysis",
    doi: 10.1101/2023.09.19.558535

    This function ONLY supports .txt format. 

    Arguments:
    filename (str) -- Name of the file to be loaded. Must not contain the file extension.
    path -- Path to the file to be loaded. If path is None, the file must be in the same 
            directory as bnpstep.py. Default: None

    Returns:
    Dictionary with data (numpy array), time points (numpy array), ground_truths (None), and parameters (None).
    """
    # Validate input and build path
    if not isinstance(filename, str):
            raise TypeError(f"filename should be of type str, got {type(filename)}")
    # TODO: validate path
    full_name = filename + '.txt'
    if path is not None:
        full_path = os.path.join(path, full_name)
    else:
        full_path = full_name

    # Read in file
    ground_b = []
    ground_h = []
    ground_t = []
    data = []
    times = []
    with open(full_path, 'r') as f:
        # Read in ground truth parameters
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

        N_file = float(N_str)
        N_file = int(N_file)

        # Skip padding zeros
        for i in range(N_file - 10):
            dump = f.readline()
        
        for i in range(5):
            for j in range(N_file):
                item = f.readline().strip()
                value = float(item)
                if i == 0:
                    ground_b.append(value)
                elif i == 1:
                    ground_h.append(value)
                elif i == 2:
                    ground_t.append(value)
                elif i == 3:
                    data.append(value)
                else:
                    times.append(value)
    
    # Extract synthetic data generation parameters
    B_file = float(B_str)
    B_file = int(B_file)
    F_file = float(F_str)
    h_stp_file = float(h_stp_str)
    t_stp_file = float(t_stp_str)
    eta_file = float(eta_str)
    B_max_file = float(B_max_str)
    B_max_file = int(B_max_file)

    params = {}
    params["type"] = 'kv'
    params["gt_steps"] = B_file
    params["num_observations"] = len(data)
    params["f_back"] = F_file
    params["h_step"] = h_stp_file
    params["t_step"] = t_stp_file
    params["eta"] = eta_file
    params["B_max"] = B_max_file

    # Sanitize and pack ground truth trajectory data
    ground_b = np.asarray(ground_b).astype(np.int)
    ground_h = np.asarray(ground_h).astype(float)
    ground_t = np.asarray(ground_t).astype(float)
    data = np.asarray(data).astype(float)
    times = np.asarray(times).astype(float)

    ground_b = ground_b[:B_max_file+1]
    ground_h = ground_h[:B_max_file+1]
    ground_t = ground_t[:B_max_file+1]

    ground = {"b_m": ground_b, "h_m": ground_h, "t_m": ground_t}

    # Pack everything into a dict
    dataset = {}
    dataset["data"] = data
    dataset["times"] = times
    
    dataset["ground_truths"] = ground
    dataset["parameters"] = params

    return dataset
