"""
bnpinputs: Contains all functions that load data sets for BNP-Step.
"""
import os
import numpy as np
import pandas as pd
import ast
import re

def clean_and_parse(column):
    """
    Cleans and converts a column containing mixed newline and space-delimited lists into numpy arrays.
    """
    def parse_value(x):
        x_in = x
        if isinstance(x, str) and x.startswith("["):
            try:
                x = x_in
                # Standardize delimiters: replace newlines with spaces, remove extra spaces
                x = re.sub(r"[\s\n]+", " ", x.strip())  
                try:
                    return np.array(ast.literal_eval(x), dtype=float)  # Convert string to list, then to np.array
                except (ValueError, SyntaxError):
                    raise ValueError(f"Malformed list format: {x}")
            except:
                x = x_in
                # Standardize delimiters: replace newlines with spaces, remove extra spaces
                x = re.sub(r',','',re.sub(r"[\s\n]+", ",", x.strip())  ,1)
                try:
                    return np.array(ast.literal_eval(x), dtype=float)  # Convert string to list, then to np.array
                except (ValueError, SyntaxError):
                    raise ValueError(f"Malformed list format: {x}")
        else:
            try:
                return float(x)
            except ValueError:
                raise ValueError(f"Expected a number but got: {x}")

    return column.apply(parse_value)

def load_data_garcia(filename: str, data_format: str, path=None):
    """
    Loads a specific trace from a CSV file based on the given model type and iteration.

    Arguments:
    filename (str) -- Name of the CSV file (without extension).
    data_format (str) -- A two-character string specifying the model type and iteration (e.g., 'B1', '0R').
    path -- Path to the file. If None, assumes the file is in the current directory. Default: None.

    Returns:
    Dictionary containing:
    - 'data' (numpy array): The extracted trace values.
    - 'times' (numpy array): The corresponding time points.
    - 'nu_vec' (numpy array): The associated nu values (if present).
    - 'ground_truths' (None)
    - 'parameters' (None)
    """
    if not isinstance(filename, str):
        raise TypeError(f"filename should be of type str, got {type(filename)}")
    if not isinstance(data_format, str) or len(data_format) != 2:
        raise ValueError("data_format must be a string of length 2 (e.g., 'B1', '0R').")

    # Construct full file path
    full_name = filename + ".csv"
    full_path = os.path.join(path, full_name) if path else full_name

    # Load CSV into Pandas DataFrame
    df = pd.read_csv(full_path, dtype=str)  # Read everything as strings to avoid misinterpretation

    # Detect correct column names (handling different capitalizations)
    model_col = next((col for col in ["model", "Model"] if col in df.columns), None)
    iteration_col = next((col for col in ["iteration", "Iteration"] if col in df.columns), None)

    if model_col is None or iteration_col is None:
        raise ValueError("CSV must contain either 'model' or 'Model' and 'iteration' or 'Iteration' columns.")

    # Extract the specific trace using detected column names
    subset = df[(df[model_col] == data_format[0]) & (df[iteration_col] == data_format[1])]

    if subset.empty:
        raise ValueError(f"No data found for model {data_format[0]} and iteration {data_format[1]}.")

    # Convert relevant columns using `clean_and_parse`
    dataset = {
        "data": clean_and_parse(subset["trace_au"]).to_numpy()[0] ,
        "times": clean_and_parse(subset["time"]).to_numpy()[0],
        "nu_vec": clean_and_parse(subset["error"]).to_numpy()[0] if "error" in df.columns else None,
        "ground_truths": None,
        "parameters": None,
    }
    dataset["data"] = dataset["data"] + np.random.poisson(75,dataset["data"].shape)
    dataset["nu_vec"] = 1/ (dataset["nu_vec"] + 75)
    return dataset

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
        # MAX: here I added the nu_vector to the data loading
    dataset = {}
    if has_timepoints:
        data_fr = pd.read_csv(full_path, header=None)
        data_np = pd.DataFrame.to_numpy(data_fr)
        if isinstance(data_np[0,0],str):
            data_fr = pd.read_csv(full_path)
            data_np = pd.DataFrame.to_numpy(data_fr)
        times = data_np[:, 0]
        data = data_np[:, 1]
        nu_vec = data_np[:,2]
        # Build dictionary for output
        dataset["data"] = data.astype(float)
        dataset["times"] = times.astype(float)
        dataset["nu_vec"] = nu_vec.astype(float)
    else:
        data_fr = pd.read_csv(full_path, header=None)
        data_np = pd.DataFrame.to_numpy(data_fr)
        data = data_np[:, 0]
        nu_vec = data_np[:,1]

        # Build dictionary for output
        dataset["data"] = data
        dataset["times"] = None
        dataset["nu_vec"] = nu_vec
    
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
