"""
Bayesian NonParametric Step (BNP-Step)

Class for running BNP-Step and visualizing results.

Alex Rojewski, 2023

"""
import os
from typing import Optional, Dict, Union, List
from pathlib import Path
import warnings
import numpy as np
import bnpinputs as bnpi
import bnpsampler as sampler
import bnpanalysis as bnpa
import bic_tools as bic
import pickle
import csv
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerPatch
from matplotlib.gridspec import GridSpec



class BNPStep:
    def __init__(self, 
                 chi: float = 0.00028, 
                 h_ref: float = 0.0, 
                 psi: float = 0.00028, 
                 F_ref: float = 0.0, 
                 phi: float = 0.001, 
                 eta_ref: float = 10.0, 
                 gamma: float = 1.0, 
                 B_max: int = 70, 
                 load_initialization: str = 'prior',
                 use_annealing: bool = False,
                 init_temperature: int = 2250, 
                 scale_factor: float = 1.25, 
                 seed: Optional[int] = None,
                 ):
        """
        Initializes a fresh BNP-Step object with the specified options. All chains analyzed with this object
        will use the same seed, hyperparameter values, initialization options, and simulated annealing parameters.

        Arguments:
        chi (float) -- Precision for Normal priors on h_m. Default: 0.00028
        h_ref (float) -- Mean for Normal priors on h_m. Default: 0.0
        psi (float) -- Precision for Normal prior on F_bg. Default: 0.00028
        F_ref (float) -- Mean for Normal prior on F_bg. Default: 0.0
        phi (float) -- Shape parameter for Gamma prior on eta. Default: 1.0 (as eta_ref/phi)
        eta_ref (float) -- Sets scale parameter for Gamma prior on eta. Default: 10.0
        gamma (float) -- Hyperparameter for Bernoulli priors on b_m. Default: 1.0
        B_max (int) -- Maximum possible number of steps. Default: 50
        load_initialization (str) -- Initialization strategy for b_m. Allowed values are: 'prior' - from prior
                                                                                          'half' - 50% loads on
                                                                                          'all' - 100% loads on
                                     If init_from_prior is False, this parameter has no effect, and the initial
                                     value for the loads must be supplied along with all other initial values
                                     when calling analyze().
                                     Default: 'prior'
        use_annealing (bool) -- Whether or not to use simulated annealing. Default: False
        init_temperature (int) -- Initial temperature for simulated annealing. With a default scale_factor of 1.25, 
                                  the default value will yield a temperature of 1 at ~20,000 samples. Default: 2250
        scale_factor (float) -- Controls how fast the temperature drops off during simulated annealing. With a
                                default initial temperature of 2250, the default value will yield a temperature
                                of 1 at ~20,000 samples. Default: 1.25
        seed (int, array_like int, None) -- Seed for random number generator. Default: None
        """
        # Initialize hyperparameters
        if not isinstance(chi, float):
            raise TypeError(f"chi should be of type float instead of {type(chi)}")
        self.chi = chi

        if not isinstance(h_ref, float):
            raise TypeError(f"h_ref should be of type float instead of {type(h_ref)}")
        self.h_ref = h_ref

        if not isinstance(psi, float):
            raise TypeError(f"psi should be of type float instead of {type(psi)}")
        self.psi = psi

        if not isinstance(F_ref, float):
            raise TypeError(f"F_ref should be of type float instead of {type(F_ref)}")
        self.F_ref = F_ref

        if not isinstance(phi, float):
            raise TypeError(f"phi should be of type float instead of {type(phi)}")
        self.phi = phi

        if not isinstance(eta_ref, float):
            raise TypeError(f"eta_ref should be of type float instead of {type(eta_ref)}")
        self.eta_ref = eta_ref

        if not isinstance(gamma, float):
            raise TypeError(f"gamma should be of type float instead of {type(gamma)}")
        self.gamma = gamma

        if not isinstance(B_max, int):
            raise TypeError(f"B_max should be of type int instead of {type(B_max)}")
        self.B_max = B_max

        # Load initialization strategy
        if not isinstance(load_initialization, str):
            raise TypeError(f"load_initialization should be of type str instead of {type(load_initialization)}")
        if load_initialization != 'prior' and load_initialization != 'half' and load_initialization != 'all':
            warnings.warn(f"load_initialization must be 'prior', 'half', or 'all', got unknown option {load_initialization}. Defaulting to 'prior'.", UserWarning)
            load_initialization = 'prior'
        self.load_initialization = load_initialization

        # Simulated annealing parameters
        if not isinstance(use_annealing, bool):
            raise TypeError(f"use_annealing should be of type bool instead of {type(use_annealing)}")
        self.use_annealing = use_annealing

        # Initial temperature
        if not isinstance(init_temperature, int):
            raise TypeError(f"init_temperature should be of type int instead of {type(init_temperature)}")
        self.init_temperature = init_temperature

        # Factor for controlling how fast the temperature drops off
        if not isinstance(scale_factor, float):
            raise TypeError(f"scale_factor should be of type float instead of {type(scale_factor)}")
        self.scale_factor = scale_factor

        # Initialize RNG
        self.rng = np.random.default_rng(seed)

        # Declare variables for storing data sets and results
        self.dataset = None
        self.ETA = None
        self.F_S = None
        self.B_M = None
        self.H_M = None
        self.T_M = None
        self.post = None

        # Attributes to store results from alternative methods - these should not be used 
        # unless comparing to one of the other methods mentioned in the paper (iHMM or KV)
        self.alt_method_results_kv = None
        self.alt_method_results_ihmm = None


    def load_data(self, 
                  filename : str,
                  data_type : str = 'experimental', 
                  data_format : Optional[str] = None,
                  has_timepoints : bool = False
                  ):
        """
        Loads data sets to be analyzed with BNP-Step or graphed with results.

        Arguments:
        filename (str) -- Name of the data file to be loaded. Must include the file extension so
                          that the appropriate function from bnpinputs can be called. Valid file
                          extensions are .csv and .txt.
        data_type (str) -- Data set type. Valid types are 'synthetic' or 'experimental'. This determines
                           whether ground truth data files are expected. Default: 'experimental'
                           Most end-users should leave this at the default value unless using one of the
                           specific file formats used in the paper.
        data_format (str) -- Format of the data set. Along with data_type and the file extension,
                             this determines which function from bnpinputs will be called to load the data.
                             Valid options are 'hmm', 'kv', 'expt', 'user', and None. 'hmm', 'kv', and 'expt'
                             correspond to the data set types described in "An accurate probabilistic step 
                             finder for time-series analysis", doi: 10.1101/2023.09.19.558535
                             If None, this defaults to 'user'. Default: None
                             Most end-users should leave this at the default value unless using one of the
                             specific file formats used in the paper.
        has_timepoints (bool) -- Whether or not the data set contains time points. Default: False

        Returns:
        Dictionary with the following key, value pairs:
            "data" : numpy array of observations.
            "times" : if has_timepoints is True, this is a numpy array of time points; if has_timepoints is False, 
                      this is None.
            "ground_truths" : if data_format is 'hmm' or 'kv', this contains data for the ground truth trajectories;
                              otherwise it is None.
            "parameters" : if data_format is 'hmm' or 'kv', this contains data regarding ground truth parameter values;
                           otherwise it is None.
        """
        # Filename input validation
        if not isinstance(filename, str):
            raise TypeError(f"filename should be of type str instead of {type(filename)}")
        if filename[-4:] != '.txt' and filename[-4:] != '.csv' and filename[-4] == '.':
            raise ValueError(f"File extension should be .csv or .txt, got {filename[-4:]}")
        elif filename[-4:] != '.txt' and filename[-4:] != '.csv' and filename[-4] != '.':
            raise ValueError(f"No file extension identified, filename must include file extension.")
        file_extension = filename[-3:]
        filename = filename[:-4]

        # Data type validation
        if not isinstance(data_type, str):
            raise TypeError(f"data_type should be of type str instead of {type(data_type)}")
        if data_type != 'experimental' and data_type != 'synthetic':
            warnings.warn(f"data_type must be either 'experimental', or 'synthetic', got unknown option {data_type}. Defaulting to 'experimental'.", UserWarning)
            data_type = 'experimental'
        
        # Data format validation
        if data_format is None:
            data_format = 'user'
        if not isinstance(data_format, str):
            raise TypeError(f"data_format should be of type str instead of {type(data_format)}")
        if data_format != 'hmm' and data_format != 'kv' and data_format != 'expt' and data_format != 'user':
            warnings.warn(f"Unknown data type {data_format} detected. Defaulting to 'user'.", UserWarning)
            data_format = 'user'
        
        # Validate has_timepoints
        if not isinstance(has_timepoints, bool):
            raise TypeError(f"has_timepoints should be of type bool instead of {type(has_timepoints)}")
        
        # Load data files
        if file_extension == 'txt':
            if data_type == 'experimental':
                if data_format == 'expt':
                    self.dataset = bnpi.load_data_expt(filename)
                else:
                    self.dataset = bnpi.load_data_txt(filename, has_timepoints)
            else:
                if data_format == 'kv':
                    self.dataset = bnpi.load_data_kv(filename)
                    # KV data supplies a weak limit from the datafile
                    self.B_max = self.dataset["parameters"]["B_max"]
                else:
                    self.dataset = bnpi.load_data_txt(filename, has_timepoints)
        else:
            if data_type == 'experimental':
                self.dataset = bnpi.load_data_csv(filename, has_timepoints)
            else:
                if data_format == 'hmm':
                    self.dataset = bnpi.load_data_HMM(filename)
                else:
                    self.dataset = bnpi.load_data_csv(filename, has_timepoints)


    def analyze(self,
                data : Optional[Dict] = None,
                num_samples: int = 50000
                ):
        """
        Analyzes a dataset using BNP-Step and stores the results in object attributes.

        Arguments:
        data (Dict) -- Dictionary with minimum of two key, value pairs:
                            "data" : numpy array of observations
                            "times" : numpy array of time points or None.
                        If None is passed, the dataset assigned to self.dataset will be used.
                        If None is passed and no dataset has been assigned to the object, an exception will be thrown.
                        Default: None
        num_samples (int) -- Number of samples to generate. Default: 50000
        """
        # If we did not pass the data parameter, analyze the dataset currently assigned to the object.
        # Otherwise, assign the dataset to the object and analyze it.
        if data is None:
            if self.dataset is None:
                raise ValueError("Cannot pass None as first argument when dataset has not been assigned via load_data()")
            else:
                data = self.dataset
        else:
            self.dataset = data
        
        # Input validation
        if not isinstance(data['data'], np.ndarray):
            raise TypeError(f"'data' value in dataset dictionary should be of type ndarray instead of {type(data['data'])}")
        if data["times"] is not None:
            if not isinstance(data['times'], np.ndarray):
                raise TypeError(f"'times' value in dataset dictionary should be of type ndarray instead of {type(data['times'])}")
        if not isinstance(num_samples, int):
            raise TypeError(f"num_samples should be of type int instead of {type(num_samples)}")
        
        # Get timepoints if we have them, otherwise pass a generic arange numpy array based on how many observations
        # we have
        if data["times"] is not None:
            t_n = data["times"]
        else:
            t_n = np.arange(len(data["data"]))

        # Initialize sample arrays
        self.ETA = []
        self.F_S = []
        self.B_M = []
        self.H_M = []
        self.T_M = []
        self.post = []

        # Initialize F_bg
        self.F_S = np.asarray([self.rng.normal(self.F_ref, np.sqrt(1 / self.psi))])
        # Initialize h_ms
        H_init = []
        for _ in range(self.B_max):
            H_init.append(self.rng.normal(self.h_ref, np.sqrt(1 / self.chi)))
        self.H_M = np.asarray([H_init])
        # Initialize t_ms
        T_init = []
        for _ in range(self.B_max):
            T_init.append(self.rng.choice(t_n))
        self.T_M = np.asarray([T_init])
        # Initialize b_ms
        # TODO: vectorize these inefficient for loops where possible
        B_init = []
        if (self.load_initialization == 'prior'):
            # Initialize randomly based on gamma
            for _ in range(self.B_max):
                unif = self.rng.random()
                if (unif < (self.gamma / self.B_max)):
                    B_init.append(1)
                else:
                    B_init.append(0)
        elif (self.load_initialization == 'half'):
            # Initialize with half the loads on
            for _ in range(self.B_max):
                unif = self.rng.random()
                if (unif < 0.5):
                    B_init.append(1)
                else:
                    B_init.append(0)
        else:
            # Initialize with all loads on
            for _ in range(self.B_max):
                B_init.append(1)
        self.B_M = np.asarray([B_init])
        # Initialize eta
        self.ETA = np.asarray([self.rng.gamma(self.phi, self.eta_ref / self.phi)])

        # Generate baseline posterior
        poster = sampler.calculate_logposterior(self.B_max, len(data["data"]), data["data"], t_n, self.B_M, self.H_M, 
                                                self.T_M, self.F_S, self.ETA, self.chi, self.h_ref, self.gamma, 
                                                self.phi, self.eta_ref, self.psi, self.F_ref)
        self.post = np.asarray([poster])

        # # Set temperature for simulated annealing if we are using it
        # temperature = 1
        # # Gibbs sampler, random sweep
        # for i in np.random.permutation(range(4)):
        #     if (i == 0):
        #         # Sample F and h_m
        #         Fh_tmp = sampler.sample_fh(self.B_max, len(data["data"]), data["data"], t_n, self.B_M, self.T_M, 
        #                                     self.ETA, self.psi, self.chi, self.F_ref, self.h_ref, self.rng, 
        #                                     temperature)
        #         h_new = Fh_tmp[1:]
        #         self.F_S = np.concatenate((self.F_S, np.asarray([Fh_tmp[0]])), axis=0)
        #         self.H_M = np.vstack((self.H_M, h_new))
        #     elif (i == 1):
        #         # Sample b_m
        #         b_new = sampler.sample_b(self.B_max, len(data["data"]), data["data"], t_n, self.B_M, self.H_M, 
        #                                     self.T_M, self.F_S, self.ETA, self.gamma, self.rng, temperature)
        #         self.B_M = np.vstack((self.B_M, b_new))
        #     elif (i == 2):
        #         # Sample t_m
        #         t_new = sampler.sample_t(self.B_max, len(data["data"]), data["data"], t_n, self.B_M, self.H_M, 
        #                                     self.T_M, self.F_S, self.ETA, self.rng, temperature)
        #         self.T_M = np.vstack((self.T_M, t_new))
        #     elif (i == 3):
        #         # Sample eta
        #         new_eta = sampler.sample_eta(self.B_max, len(data["data"]), data["data"], t_n, self.B_M, self.H_M, 
        #                                         self.T_M, self.F_S, self.phi, self.eta_ref, self.rng, temperature)
        #         self.ETA = np.concatenate((self.ETA, np.asarray([new_eta])), axis=0)






        # Main sampler
        for samp in range(num_samples):
            # Set temperature for simulated annealing if we are using it
            if self.use_annealing:
                temperature = self.init_temperature * np.exp(-samp / (self.init_temperature / self.scale_factor)) + 1
            else:
                temperature = 1
            # Gibbs sampler, random sweep
            for i in np.random.permutation(range(4)):
                if (i == 0):
                    # Sample F and h_m
                    Fh_tmp = sampler.sample_fh(self.B_max, len(data["data"]), data["data"], t_n, self.B_M, self.T_M, 
                                                self.ETA, self.psi, self.chi, self.F_ref, self.h_ref, self.rng, 
                                                temperature)
                    h_new = Fh_tmp[1:]
                    self.F_S = np.concatenate((self.F_S, np.asarray([Fh_tmp[0]])), axis=0)
                    self.H_M = np.vstack((self.H_M, h_new))
                elif (i == 1):
                    # Sample b_m
                    b_new = sampler.sample_b(self.B_max, len(data["data"]), data["data"], t_n, self.B_M, self.H_M, 
                                                self.T_M, self.F_S, self.ETA, self.gamma, self.rng, temperature)
                    self.B_M = np.vstack((self.B_M, b_new))
                elif (i == 2):
                    # Sample t_m
                    t_new = sampler.sample_t(self.B_max, len(data["data"]), data["data"], t_n, self.B_M, self.H_M, 
                                                self.T_M, self.F_S, self.ETA, self.rng, temperature)
                    self.T_M = np.vstack((self.T_M, t_new))
                elif (i == 3):
                    # Sample eta
                    new_eta = sampler.sample_eta(self.B_max, len(data["data"]), data["data"], t_n, self.B_M, self.H_M, 
                                                 self.T_M, self.F_S, self.phi, self.eta_ref, self.rng, temperature)
                    self.ETA = np.concatenate((self.ETA, np.asarray([new_eta])), axis=0)
            # Calculate log posterior for this round
            poster = sampler.calculate_logposterior(self.B_max, len(data["data"]), data["data"], t_n, self.B_M, self.H_M,
                                                    self.T_M, self.F_S, self.ETA, self.chi, self.h_ref, self.gamma,
                                                    self.phi, self.eta_ref, self.psi, self.F_ref)
            self.post = np.concatenate((self.post, np.asarray([poster])), axis=0)
        
        # Clean up posteriors - make all NaN's negative infinity, which will allow sorting by posterior
        for x in range(len(self.post)):
            if np.isnan(self.post[x]):
                self.post[x] = -np.inf


    def results_to_file(self,
                        outfile: str = 'output',
                        path = None,
                        simple = False):
        """
        Outputs the results currently stored in the sample attributes (B_M, H_M, T_M, F_S, ETA) to a .pkl file.

        Arguments:
        outfile (str) -- Name of the output file to be saved. Default: 'output'
        path -- Path to the file to be loaded. If path is None, the file must be in the same 
                directory as bnpstep.py. Default: None

        """

        if not isinstance(outfile, str):
            raise TypeError(f"outfile should be of type str instead of {type(outfile)}")
        # TODO: validate path
        if simple:
            full_name = outfile + '.csv'
        else:
            full_name = outfile + '.pkl'

        if path is not None:
            full_path = os.path.join(path, full_name)
        else:
            full_path = full_name
        
        if simple:
            if self.dataset["times"] is not None:
                t_n = self.dataset["times"]
            else:
                t_n = np.arange(len(self.dataset["data"]))
            
            burn_in_samples = int(0.25 * len(self.post))
            b_clean, h_clean, t_clean, f_clean, eta_clean, post_clean = bnpa.remove_burn_in(self.B_M, self.H_M, self.T_M, self.F_S, self.ETA, self.post, burn_in_samples)

            map_index = np.argmax(post_clean)
            step_data                 = bnpa.get_step_plot_data(b_clean , h_clean , t_clean , f_clean, t_n , self.B_max , t_n.size , map_index)
            results = np.concatenate((t_n,step_data)).reshape((-1, 2), order='F')
            with open(full_path, 'w') as fp:
                writerr = csv.writer(fp)
                writerr.writerow(['observation_times','step_heights'])
                writerr.writerows(results)
        else:

            if self.B_M is None or self.H_M is None or self.T_M is None or self.F_S is None or self.ETA is None:
                raise ValueError("One or more sample attributes is None, cannot save output file.")
            if len(self.B_M) == 0 or len(self.H_M) == 0 or len(self.T_M) == 0 or len(self.F_S) == 0 or len(self.ETA) == 0:
                raise ValueError("One or more sample attribute arrays is empty, cannot save output file.")
            
            # Output data to file
            results = {"b_m": self.B_M, "h_m": self.H_M, "t_m": self.T_M, "F_bg": self.F_S, "eta": self.ETA, "posterior": self.post}
            
            with open(full_path, 'wb') as fp:
                pickle.dump(results, fp)
    

    def results_from_file(self,
                          filename: str,
                          path = None
                          ):
        """
        Reads in previously saved results from .pkl file on disk and stores them in the appropriate sample attributes
        (B_M, H_M, T_M, F_S, ETA).

        Arguments:
        filename (str) -- Name of the output file to be loaded.
        path -- Path to the file to be loaded. If path is None, the file must be in the same 
                directory as bnpstep.py. Default: None

        """
        if not isinstance(filename, str):
            raise TypeError(f"filename should be of type str, got {type(filename)}")
        # TODO: validate path
        full_name = filename + '.pkl'
        if path is not None:
            full_path = os.path.join(path, full_name)
        else:
            full_path = full_name
        with open(full_path, 'rb') as fp:
            results = pickle.load(fp)

        self.B_M = results["b_m"]
        self.H_M = results["h_m"]
        self.T_M = results["t_m"]
        self.F_S = results["F_bg"]
        self.ETA = results["eta"]
        self.post = results["posterior"]


    def run_BIC(self,
                data : Optional[Dict] = None
                ):
        """
        Analyzes a dataset using an implementation of a BIC-based method and stores the
        results in an object attribute for later analysis.

        Arguments:
        data (Dict) -- Dictionary with minimum of two key, value pairs:
                            "data" : numpy array of observations
                            "times" : numpy array of time points or None.
                        If None is passed, the dataset assigned to self.dataset will be used.
                        If None is passed and no dataset has been assigned to the object, an exception will be thrown.
                        Default: None
        """
        if data is None:
            if self.dataset is None:
                raise ValueError("Cannot pass None as first argument when dataset has not been assigned via load_data()")
            else:
                data = self.dataset
        else:
            self.dataset = data
        
        # Input validation
        if not isinstance(data['data'], np.ndarray):
            raise TypeError(f"'data' value in dataset dictionary should be of type ndarray instead of {type(data['data'])}")
        if data["times"] is not None:
            if not isinstance(data['times'], np.ndarray):
                raise TypeError(f"'times' value in dataset dictionary should be of type ndarray instead of {type(data['times'])}")
        
        # Get timepoints if we have them, otherwise pass a generic arange numpy array based on how many observations
        # we have
        if data["times"] is not None:
            t_n = data["times"]
        else:
            t_n = np.arange(len(data["data"]))
        
        # Run method and store the results into a dict
        jump_times, means, st_dev, bic_value = bic.GreedyBIC(data['data'], len(data['data']), t_n)

        # Treat the last mean learned by KV as "F_bg" for compatibility with the graphing functions
        background = means[-1]
        means = means[:-1]

        self.alt_method_results_kv = {"jump_times": jump_times, "means": means, "background": background, "st_dev": st_dev, "bic": bic_value}


    def load_ihmm_results(self, 
                          filename_mm : str, 
                          filename_mmt : str,
                          filename_samples : str,
                          filename_inds : str,
                          path = None
                          ):
        """
        Loads results from iHMM method into self.alt_method_results_ihmm for comparison plotting.

        Arguments:
        filename_mm (str) -- Name of file where all state emission levels (conditioned on the mode number of states)
                             are stored.
        filename_mmt (str) -- Name of file where mode emission mean trajectory is stored.
        filename_samples (str) -- Name of the file where all samples are stored. Samples with the mode number of states
                                  will have their state label replaced with the appropriate emission level. Samples that
                                  do not have the mode number of states are represented by their state label only and
                                  are removed prior to storage in the object attribute.
        filename_inds (str) -- Name of file storing indices of samples that do not have the mode number of states. This
                               guides which samples are skipped over during loading.
        """
        # Input validation
        if not isinstance(filename_mm, str):
                raise TypeError(f"filename_mm should be of type str instead of {type(filename_mm)}")
        if not isinstance(filename_mmt, str):
                raise TypeError(f"filename_mmt should be of type str instead of {type(filename_mmt)}")
        if not isinstance(filename_samples, str):
                raise TypeError(f"filename_samples should be of type str instead of {type(filename_samples)}")
        if not isinstance(filename_inds, str):
                raise TypeError(f"filename_inds should be of type str instead of {type(filename_inds)}")
        # TODO: validate path

        ihmm_mode_means = bnpa.load_ihmm_mode_means(filename_mm, path)

        ihmm_mode_mean_trajectory =  bnpa.load_ihmm_mode_mean_trajectory(filename_mmt, path)

        ihmm_samples = bnpa.load_ihmm_samples(filename_samples, filename_inds, path)

        self.alt_method_results_ihmm = {"mode_means": ihmm_mode_means, "mode_mean_traj": ihmm_mode_mean_trajectory, "samples": ihmm_samples}

    
    def plot_data(self,
                  font_size : int = 16,
                  datacolor : str = '#929591',
                  x_label : str =  'x-values', 
                  y_label : str =  'y-values',
                  ):
        """
        Plots the currently loaded dataset.

        Arguments:
        font_size (int) -- Font size of plot labels. Default: 16
        datacolor (str) -- Color of plotted data points. Must be a valid color for matplotlib. Default: '#929591'
        x_label (str) -- Label for the x-axis. Default: 'x-values'
        y_label (str) -- Label for the y-axis. Default: 'y-values'

        Returns:
        matplotlib plot of the currently loaded data set
        """
        if self.dataset is None:
            raise ValueError("Dataset is empty, cannot generate graph.")
        if not isinstance(font_size, int):
            raise TypeError(f"font_size should be of type int instead of {type(font_size)}")
        if not isinstance(datacolor, str):
            raise TypeError(f"datacolor should be of type str instead of {type(datacolor)}")
        if not isinstance(x_label, str):
            raise TypeError(f"x_label should be of type str instead of {type(x_label)}")
        if not isinstance(y_label, str):
            raise TypeError(f"y_label should be of type str instead of {type(y_label)}")
        
        # Get timepoints if we have them, otherwise pass a generic arange numpy array based on number of observations
        if self.dataset["times"] is not None:
            t_n = self.dataset["times"]
        else:
            t_n = np.arange(len(self.dataset["data"]))
        
        # General figure setup
        fig = plt.figure()
        fnt_mgr = mpl.font_manager.FontProperties(size=font_size)
        gs1 = GridSpec(1, 1, bottom=0.8)  # For legend
        ax0 = fig.add_subplot(gs1[0])
        ax0.axis('off')
        gs2 = GridSpec(1, 1, top=0.85, wspace=0.07, hspace=0.05)  # For actual plot
        ax1 = fig.add_subplot(gs2[0])

        # Axis labels
        fig.supxlabel(x_label, fontsize=font_size)
        fig.supylabel(y_label, x=0.05, fontsize=font_size)

        # Data point legend icon setup
        class HandlerCircle(HandlerPatch):
            def create_artists(self, legend, orig_handle,
                            xdescent, ydescent, width, height, fontsize, trans):
                center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
                p = mpatches.Circle(xy=center, radius=1.5)
                self.update_prop(p, orig_handle, legend)
                p.set_transform(trans)
                return [p]

        c = mpatches.Circle((0.5, 0.5), 0.25, facecolor="None",
                            edgecolor=datacolor, linewidth=1)
        
        # Setup x-axis vectors for plotting
        T = np.linspace(t_n[0], t_n[-1], len(self.dataset["data"]))

        # Plot synthetic data
        ax1.scatter(T, self.dataset["data"], alpha=0.7, color=datacolor, facecolors='none', marker='.')

        # Add subplot titles
        # ax1.set_title('SNR = 2.0', font=fpath, fontsize=fntsize)

        # Generate legend
        ax0.legend([c],
                ['Data'], handler_map={mpatches.Circle: HandlerCircle()},
                bbox_to_anchor=(0., 1.15, 1., .102), loc='lower center',
                        ncol=1, mode="none", borderaxespad=0., edgecolor=(1.0, 1.0, 1.0, 0.0), prop=fnt_mgr, borderpad=0.8)
        
        # Configure axes
        ax1.set_xticks([t_n[0], t_n[int(len(self.dataset["data"])/2)], t_n[-1]])
        ax1.set_xticklabels([str(int(t_n[0])), str(int((t_n[int(len(self.dataset["data"])/2)]))), str(int(t_n[-1]))], fontsize=font_size)

        ax1.set_yticks([int(np.amin(self.dataset["data"])), int((np.amax(self.dataset["data"])+np.amin(self.dataset["data"]))/2), int(np.amax(self.dataset["data"]))])
        ax1.set_yticklabels([str(int(np.amin(self.dataset["data"]))), str(int((np.amax(self.dataset["data"])+np.amin(self.dataset["data"]))/2)), str(int(np.amax(self.dataset["data"])))], fontsize=font_size)

        # Show plot
        plt.show() 


    def visualize_results(self,
                          plot_type : Union[str, List[str]] = 'step',
                          font_size : int = 16,
                          datacolor : str = '#929591',
                          learncolor : str = '#f97306',
                          gtcolor : str = '#00ffff',
                          alt_color1 : str = '#5d3a9b',
                          alt_color2 : str = '#c20078',
                          alt_color3 : str = '#d31a0c',
                          alt_color4 : str = '#fac205',
                          x_label : str =  'x-values',
                          y_label : str =  'y-values',
                          show_prior : bool = True,
                          show_ci : bool = True,
                          plot_alt_results : bool = False,
                          alt_results : Union[str, List[str]] = '',
                          fig_savename : str = 'figure'
                          ):
        """
        Draws plots of BNP-Step results with the dataset (and ground truths if present).

        Arguments:
        plot_type (str or list of str) -- Type of plot to be generated. If more than one is desired, pass the options
                                          as a list of strings. Valid options are:
                                                'step': Learned trajectory plotted over the dataset. For BNP-Step the
                                                        MAP estimate sample is used. For BIC results, the point estimate
                                                        is used. For iHMM results, the mode emission mean trajectory is
                                                        used.
                                                'hist_step_height': Histograms of the step heights (the difference between
                                                                    two adjacent emission levels). For BNP-Step, this amounts
                                                                    to the joint posterior distribution over all h_m, conditioned
                                                                    on the MAP number of steps. For BIC results, this is a frequentist
                                                                    histogram over all learned step heights in a trace; since this
                                                                    is a point estimate, it is recommended to only use this option
                                                                    when comparing aggregated datasets (as was done in the paper).
                                                                    For iHMM results, this is a distribution over all step heights
                                                                    from all samples conditioned on the mode number of states.
                                                'hist_dwell_time': Histograms of the holding times. For BNP-Step, this amounts
                                                                   to the joint posterior distribution over all holding times,
                                                                   conditioned on the MAP number of steps. For BIC results, this is
                                                                   a frequentist histogram of all learned holding times in a trace
                                                                   (see description under 'hist_step_height' for caveats). For
                                                                   iHMM results, this is a distribution over all holding times
                                                                   from all samples conditioned on the mode number of states.
                                                'hist_emission': Histograms of the emission levels. For BNP-Step, this is the 
                                                                 joint posterior distribution over all emission levels, conditioned
                                                                 on the MAP number of steps. For BIC results, this is a frequentist
                                                                 histogram of all learned emission levels in a trace (see above for
                                                                 caveats). For iHMM results, this is a distribution over all emission
                                                                 levels from all samples conditioned on the mode number of states.
                                          Planned options include 'hist_height_separated', 'hist_dwell_separated', 
                                          'hist_emission_separated', 'survivorship', 'hist_f', and 'hist_eta'
                                          Default: 'step'
        font_size : int = 16,
        datacolor : str = '#929591',
        learncolor : str = '#f97306',
        gtcolor : str = '#00ffff',
        alt_color1 : str = '#5d3a9b',
        alt_color2 : str = '#c20078',
        alt_color3 : str = '#d31a0c',
        alt_color4 : str = '#fac205',
        x_label : str =  'x-values',
        y_label : str =  'y-values',
        show_prior : bool = True,
        show_ci : bool = True,
        plot_alt_results : bool = False,
        alt_results : Union[str, List[str]] = '',
        fig_savename : str = 'figure'

        Results:
        The selected plots, which are displayed on-screen and saved to disk in BNP-Step's directory.
        """
        # TODO: Tons of repeated code that needs to be refactored/moved to their own functions
        # TODO: Better input validation for dataset dict
        # Validate that we actually have non-empty dataset and results loaded
        if self.B_M is None or self.H_M is None or self.T_M is None or self.F_S is None or self.ETA is None:
            raise ValueError("One or more sample attributes is None, cannot generate graphs.")
        if len(self.B_M) == 0 or len(self.H_M) == 0 or len(self.T_M) == 0 or len(self.F_S) == 0 or len(self.ETA) == 0:
            raise ValueError("One or more sample attribute arrays is empty, cannot generate graphs.")
        if self.dataset is None:
            raise ValueError("Dataset is empty, cannot generate graphs.")

        # Input validation
        if not isinstance(plot_type, str) and not isinstance(plot_type, list):
            raise TypeError(f"plot_type should be of type str or list of str instead of {type(plot_type)}")
        if isinstance(plot_type, list):
            for x in plot_type:
                if not isinstance(x, str):
                    raise TypeError(f"At least one element of plot_type is {type(x)} instead of str")
        if isinstance(plot_type, str):
            # Listify if a single str object
            plot_type = [plot_type]
        if not isinstance(font_size, int):
            raise TypeError(f"font_size should be of type int instead of {type(font_size)}")
        if not isinstance(datacolor, str):
            raise TypeError(f"datacolor should be of type str instead of {type(datacolor)}")
        if not isinstance(learncolor, str):
            raise TypeError(f"learncolor should be of type str instead of {type(learncolor)}")
        if not isinstance(gtcolor, str):
            raise TypeError(f"gtcolor should be of type str instead of {type(gtcolor)}")
        if not isinstance(alt_color1, str):
            raise TypeError(f"alt_color1 should be of type str instead of {type(alt_color1)}")
        if not isinstance(alt_color2, str):
            raise TypeError(f"alt_color2 should be of type str instead of {type(alt_color2)}")
        if not isinstance(alt_color3, str):
            raise TypeError(f"alt_color3 should be of type str instead of {type(alt_color3)}")
        if not isinstance(x_label, str):
            raise TypeError(f"x_label should be of type str instead of {type(x_label)}")
        if not isinstance(y_label, str):
            raise TypeError(f"y_label should be of type str instead of {type(y_label)}")
        if not isinstance(show_prior, bool):
            raise TypeError(f"show_prior should be of type bool instead of {type(show_prior)}")
        if not isinstance(show_ci, bool):
            raise TypeError(f"show_ci should be of type bool instead of {type(show_ci)}")
        if not isinstance(plot_alt_results, bool):
            raise TypeError(f"plot_alt_results should be of type bool instead of {type(plot_alt_results)}")
        if not isinstance(alt_results, str) and not isinstance(alt_results, list):
            raise TypeError(f"alt_results should be of type str or list of str instead of {type(alt_results)}")
        if isinstance(alt_results, list):
            for x in alt_results:
                if not isinstance(x, str):
                    raise TypeError(f"At least one element of alt_results is {type(x)} instead of str")
        if isinstance(alt_results, str):
            # Listify if a single str object
            alt_results = [alt_results]
        if not isinstance(fig_savename, str):
            raise TypeError(f"fig_savename should be of type str instead of {type(fig_savename)}")
        
        # Warn the user if invalid alt_results type was chosen.
        # If the user chose an alt_results type and there are no alt_results stored in the object attribute, throw exception.
        if plot_alt_results:
            for x in alt_results:
                if x != 'kv' and x != 'ihmm':
                    warnings.warn(f"Valid values for alt_results elements are 'kv' and 'ihmm', got {x}. No alternative method results will be shown.", UserWarning)
                    plot_alt_results = False
                elif x == 'kv' and self.alt_method_results_kv is None:
                    raise ValueError("KV results attribute is empty, unable to plot results. Call run_BIC() to generate results from the dataset before plotting.")
                elif x == 'ihmm' and self.alt_method_results_ihmm is None:
                    raise ValueError("iHMM results attribute is empty, unable to plot results. Ensure you have all iHMM results files in the appropriate directory before plotting.")
                
        # Ensure we have at least one valid plot type in plot_type list; otherwise warn the user and default to step plot
        has_valid_plot_type = False
        for plot in plot_type:
            if plot == 'step':
                has_valid_plot_type = True
            # elif plot == 'step_residual':
            #     has_valid_plot_type = True
            elif plot == 'hist_step_height':
                has_valid_plot_type = True
            elif plot == 'hist_dwell_time':
                has_valid_plot_type = True
            elif plot == 'hist_emission':
                has_valid_plot_type = True
            elif plot == 'hist_height_separated':
                has_valid_plot_type = True
            elif plot == 'hist_dwell_separated':
                has_valid_plot_type = True
            elif plot == 'hist_emission_separated':
                has_valid_plot_type = True
            elif plot == 'survivorship':
                has_valid_plot_type = True
            elif plot == 'hist_f':
                has_valid_plot_type = True
            elif plot == 'hist_eta':
                has_valid_plot_type = True
        
        if not has_valid_plot_type:
            warnings.warn("No valid plot types were passed in plot_type parameter; reverting to default ('step')", UserWarning)
            plot_type = ['step']

        # Get timepoints if we have them, otherwise pass a generic arange numpy array based on number of observations
        if self.dataset["times"] is not None:
            t_n = self.dataset["times"]
        else:
            t_n = np.arange(len(self.dataset["data"]))

        # Data point legend icon setup
        class HandlerCircle(HandlerPatch):
            def create_artists(self, legend, orig_handle,
                            xdescent, ydescent, width, height, fontsize, trans):
                center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
                p = mpatches.Circle(xy=center, radius=1.5)
                self.update_prop(p, orig_handle, legend)
                p.set_transform(trans)
                return [p]

        c = mpatches.Circle((0.5, 0.5), 0.25, facecolor="None",
                            edgecolor=datacolor, linewidth=1)

        # Font manager
        fnt_mgr = mpl.font_manager.FontProperties(size=font_size)

        # Remove burn-in
        # TODO: allow an option for the user to display the log posterior and manually select burn-in point.
        # By default, remove the first 25% of samples as burn-in.
        burn_in_samples = int(0.25 * len(self.post))
        b_clean, h_clean, t_clean, f_clean, eta_clean, post_clean = bnpa.remove_burn_in(self.B_M, self.H_M, self.T_M, self.F_S, self.ETA, self.post, burn_in_samples)

        for plot in plot_type:
            # if (plot == 'step'):
            #     # General figure setup
            #     fig = plt.figure()

            #     if plot_alt_results and len(alt_results) == 1:
            #         gs1 = GridSpec(1, 1, bottom=0.8)  # For legend
            #         ax0 = fig.add_subplot(gs1[0])
            #         ax0.axis('off')
            #         gs2 = GridSpec(1, 2, top=0.85, wspace=0.07, hspace=0.05)  # For actual plot
            #         ax1 = fig.add_subplot(gs2[0])
            #         ax2 = fig.add_subplot(gs2[1])
            #     elif plot_alt_results and len(alt_results) == 2:
            #         gs1 = GridSpec(1, 1, bottom=0.8)  # For legend
            #         ax0 = fig.add_subplot(gs1[0])
            #         ax0.axis('off')
            #         gs2 = GridSpec(1, 3, top=0.85, wspace=0.07, hspace=0.05)  # For actual plot
            #         ax1 = fig.add_subplot(gs2[0])
            #         ax2 = fig.add_subplot(gs2[1])
            #         ax3 = fig.add_subplot(gs2[2])
            #     else:
            #         gs1 = GridSpec(1, 1, bottom=0.8)  # For legend
            #         ax0 = fig.add_subplot(gs1[0])
            #         ax0.axis('off')
            #         gs2 = GridSpec(1, 1, top=0.85, wspace=0.07, hspace=0.05)  # For actual plot
            #         ax1 = fig.add_subplot(gs2[0])

            #     fig.supxlabel(x_label, fontsize=font_size)
            #     fig.supylabel(y_label, x=0.05, fontsize=font_size)

            #     # Setup x-axis vectors for plotting
            #     T = np.linspace(t_n[0], t_n[-1], len(self.dataset["data"]))
                
            #     # Find the MAP estimate for our step plot
            #     map_index = np.argmax(post_clean)
            #     step_data                 = bnpa.get_step_plot_data(b_clean , h_clean , t_clean , f_clean, t_n , self.B_max , t_n.size , map_index)
            #     # Plot synthetic data
            #     ax1.scatter(T, self.dataset["data"], alpha=0.7, color=datacolor, facecolors='none', marker='.')

            #     # If we have ground truths, plot them
            #     if (self.dataset["ground_truths"] is not None) and (self.dataset["parameters"] is not None):
            #         if self.dataset["parameters"]["type"] == 'kv':
            #             ground_times, ground_data = bnpa.generate_gt_step_plot_data(self.dataset["ground_truths"]["b_m"], self.dataset["ground_truths"]["h_m"], self.dataset["ground_truths"]["t_m"], self.dataset["parameters"]["f_back"], t_n, self.B_max)
            #             ax1.stairs(ground_data, ground_times, baseline=None, color=gtcolor, linewidth=3.0, zorder=5.0)
            #         if self.dataset["parameters"]["type"] == 'hmm':
            #             ground_data = self.dataset["ground_truths"]["u"]
            #             gt_times = np.asarray([0])
            #             gt_times = np.append(gt_times, t_n)
            #             ax1.stairs(ground_data, gt_times, baseline=None, color=gtcolor, linewidth=3.0, zorder=5.0)

            #     # Plot discovered steps
            #     ax1.stairs(step_data, np.insert(T,0,T[0]), baseline=None, color=learncolor, linewidth=3.0, zorder=10)

            #     # Add subplot titles
            #     # ax1.set_title('SNR = 2.0', font=fpath, fontsize=fntsize)
                
            #     # Configure axes
            #     ax1.set_xticks([t_n[0], t_n[int(len(self.dataset["data"])/2)], t_n[-1]])
            #     ax1.set_xticklabels([str(int(t_n[0])), str(int((t_n[int(len(self.dataset["data"])/2)]))), str(int(t_n[-1]))], fontsize=font_size)

            #     # ax1.set_yticks([int(np.amin(sorted_data)), int((np.amax(sorted_data)+np.amin(sorted_data))/2), int(np.amax(sorted_data))])
            #     # ax1.set_yticklabels([str(int(np.amin(sorted_data))), str(int((np.amax(sorted_data)+np.amin(sorted_data))/2)), str(int(np.amax(sorted_data)))], fontsize=font_size)

            #     # If we also have alternative results, plot those and configure the axes.
            #     if plot_alt_results:
            #         if len(alt_results) == 1:
            #             ax2.scatter(T, self.dataset["data"], alpha=0.7, color=datacolor, facecolors='none', marker='.')
                        
            #             # Alternative method plotting
            #             if alt_results[0] == 'kv':
            #                 plot_times, plot_heights = bnpa.generate_kv_step_plot_data(self.alt_method_results_kv["jump_times"], 
            #                                                                         self.alt_method_results_kv["means"], 
            #                                                                         self.alt_method_results_kv["background"], 
            #                                                                         t_n)
            #                 ax2.stairs(plot_heights, plot_times, baseline=None, color=alt_color1, linewidth=3.0, zorder=10)
            #                 # Generate legend
            #                 if (self.dataset["ground_truths"] is not None) and (self.dataset["parameters"] is not None):
            #                     ax0.legend([Line2D([0], [0], color=gtcolor,lw=3.0),
            #                                     Line2D([0], [0], color=learncolor,lw=3.0), Line2D([0], [0], color=alt_color1,lw=3.0),c],
            #                                 ['Ground truth', 'BNP-Step', 'BIC', 'Data'], handler_map={mpatches.Circle: HandlerCircle()},
            #                                 bbox_to_anchor=(0., 1.15, 1., .102), loc='lower center',
            #                                         ncol=4, mode="none", borderaxespad=0., edgecolor=(1.0, 1.0, 1.0, 0.0), prop=fnt_mgr, borderpad=0.8)
            #                 else:
            #                     ax0.legend([Line2D([0], [0], color=learncolor,lw=3.0), Line2D([0], [0], color=alt_color1,lw=3.0), c],
            #                             ['BNP-Step', 'BIC', 'Data'], handler_map={mpatches.Circle: HandlerCircle()},
            #                             bbox_to_anchor=(0., 1.15, 1., .102), loc='lower center',
            #                                     ncol=3, mode="none", borderaxespad=0., edgecolor=(1.0, 1.0, 1.0, 0.0), prop=fnt_mgr, borderpad=0.8)
            #             elif alt_results[0] == 'ihmm':
            #                 sampled_heights = self.alt_method_results_ihmm["mode_mean_traj"]
            #                 sampled_times = t_n.copy()
            #                 sampled_times = np.append(sampled_times, t_n[-1])
            #                 ax2.stairs(sampled_heights, sampled_times, baseline=None, color=alt_color4, linewidth=3.0, zorder=10)
            #                 # Generate legend
            #                 if (self.dataset["ground_truths"] is not None) and (self.dataset["parameters"] is not None):
            #                     ax0.legend([Line2D([0], [0], color=gtcolor,lw=3.0),
            #                                     Line2D([0], [0], color=learncolor,lw=3.0), Line2D([0], [0], color=alt_color4,lw=3.0),c],
            #                                 ['Ground truth', 'BNP-Step', 'iHMM', 'Data'], handler_map={mpatches.Circle: HandlerCircle()},
            #                                 bbox_to_anchor=(0., 1.15, 1., .102), loc='lower center',
            #                                         ncol=4, mode="none", borderaxespad=0., edgecolor=(1.0, 1.0, 1.0, 0.0), prop=fnt_mgr, borderpad=0.8)
            #                 else:
            #                     ax0.legend([Line2D([0], [0], color=learncolor,lw=3.0), Line2D([0], [0], color=alt_color4,lw=3.0), c],
            #                             ['BNP-Step', 'iHMM', 'Data'], handler_map={mpatches.Circle: HandlerCircle()},
            #                             bbox_to_anchor=(0., 1.15, 1., .102), loc='lower center',
            #                                     ncol=3, mode="none", borderaxespad=0., edgecolor=(1.0, 1.0, 1.0, 0.0), prop=fnt_mgr, borderpad=0.8)
            #             # Plot ground truths too if we have them
            #             if (self.dataset["ground_truths"] is not None) and (self.dataset["parameters"] is not None):
            #                 if self.dataset["parameters"]["type"] == 'kv':
            #                     ground_times, ground_data = bnpa.generate_gt_step_plot_data(self.dataset["ground_truths"]["b_m"], self.dataset["ground_truths"]["h_m"], self.dataset["ground_truths"]["t_m"], self.dataset["parameters"]["f_back"], t_n, self.B_max)
            #                     ax2.stairs(ground_data, ground_times, baseline=None, color=gtcolor, linewidth=3.0, zorder=5)
            #                 if self.dataset["parameters"]["type"] == 'hmm':
            #                     ground_data = self.dataset["ground_truths"]["u"]
            #                     gt_times = np.asarray([0])
            #                     gt_times = np.append(gt_times, t_n)
            #                     ax2.stairs(ground_data, gt_times, baseline=None, color=gtcolor, linewidth=3.0, zorder=5.0)
            #             # Configure axes
            #             ax2.set_xticks([t_n[0], t_n[int(len(self.dataset["data"])/2)], t_n[-1]])
            #             ax2.set_xticklabels([str(int(t_n[0])), str(int((t_n[int(len(self.dataset["data"])/2)]))), str(int(t_n[-1]))], fontsize=font_size)
            #             ax2.set_yticks([int(np.amin(sorted_data)), int((np.amax(sorted_data)+np.amin(sorted_data))/2), int(np.amax(sorted_data))])
            #             ax2.set_yticklabels(['', '', ''], fontsize=font_size)
            #         elif len(alt_results) == 2:
            #             ax2.scatter(T, self.dataset["data"], alpha=0.7, color=datacolor, facecolors='none', marker='.')
            #             ax3.scatter(T, self.dataset["data"], alpha=0.7, color=datacolor, facecolors='none', marker='.')
                        
            #             # Plot KV first, then iHMM
            #             plot_times, plot_heights = bnpa.generate_kv_step_plot_data(self.alt_method_results_kv["jump_times"], 
            #                                                                         self.alt_method_results_kv["means"], 
            #                                                                         self.alt_method_results_kv["background"], 
            #                                                                         t_n)
            #             ax2.stairs(plot_heights, plot_times, baseline=None, color=alt_color1, linewidth=3.0, zorder=10)

            #             sampled_heights = self.alt_method_results_ihmm["mode_mean_traj"]
            #             sampled_times = t_n.copy()
            #             sampled_times = np.append(sampled_times, t_n[-1])
            #             ax3.stairs(sampled_heights, sampled_times, baseline=None, color=alt_color4, linewidth=3.0, zorder=10)

            #             # Generate legend
            #             if (self.dataset["ground_truths"] is not None) and (self.dataset["parameters"] is not None):
            #                     ax0.legend([Line2D([0], [0], color=gtcolor,lw=3.0),
            #                                     Line2D([0], [0], color=learncolor,lw=3.0), Line2D([0], [0], color=alt_color1,lw=3.0), Line2D([0], [0], color=alt_color4,lw=3.0),c],
            #                                 ['Ground truth', 'BNP-Step', 'BIC', 'iHMM', 'Data'], handler_map={mpatches.Circle: HandlerCircle()},
            #                                 bbox_to_anchor=(0., 1.15, 1., .102), loc='lower center',
            #                                         ncol=5, mode="none", borderaxespad=0., edgecolor=(1.0, 1.0, 1.0, 0.0), prop=fnt_mgr, borderpad=0.8)
            #             else:
            #                 ax0.legend([Line2D([0], [0], color=learncolor,lw=3.0), Line2D([0], [0], color=alt_color1,lw=3.0), Line2D([0], [0], color=alt_color4,lw=3.0), c],
            #                         ['BNP-Step', 'BIC', 'iHMM', 'Data'], handler_map={mpatches.Circle: HandlerCircle()},
            #                         bbox_to_anchor=(0., 1.15, 1., .102), loc='lower center',
            #                                 ncol=4, mode="none", borderaxespad=0., edgecolor=(1.0, 1.0, 1.0, 0.0), prop=fnt_mgr, borderpad=0.8)
                                
            #             # Plot ground truths too if we have them
            #             if (self.dataset["ground_truths"] is not None) and (self.dataset["parameters"] is not None):
            #                 if self.dataset["parameters"]["type"] == 'kv':
            #                     ground_times, ground_data = bnpa.generate_gt_step_plot_data(self.dataset["ground_truths"]["b_m"], self.dataset["ground_truths"]["h_m"], self.dataset["ground_truths"]["t_m"], self.dataset["parameters"]["f_back"], t_n, self.B_max)
            #                     ax2.stairs(ground_data, ground_times, baseline=None, color=gtcolor, linewidth=3.0, zorder=5)
            #                     ax3.stairs(ground_data, ground_times, baseline=None, color=gtcolor, linewidth=3.0, zorder=5)
            #                 if self.dataset["parameters"]["type"] == 'hmm':
            #                     ground_data = self.dataset["ground_truths"]["u"]
            #                     gt_times = np.asarray([0])
            #                     gt_times = np.append(gt_times, t_n)
            #                     ax2.stairs(ground_data, gt_times, baseline=None, color=gtcolor, linewidth=3.0, zorder=5.0)
            #                     ax3.stairs(ground_data, gt_times, baseline=None, color=gtcolor, linewidth=3.0, zorder=5.0)

            #             # Configure axes
            #             ax2.set_xticks([t_n[0], t_n[int(len(self.dataset["data"])/2)], t_n[-1]])
            #             ax2.set_xticklabels([str(int(t_n[0])), str(int((t_n[int(len(self.dataset["data"])/2)]))), str(int(t_n[-1]))], fontsize=font_size)
            #             ax2.set_yticks([int(np.amin(sorted_data)), int((np.amax(sorted_data)+np.amin(sorted_data))/2), int(np.amax(sorted_data))])
            #             ax2.set_yticklabels(['', '', ''], fontsize=font_size)

            #             ax3.set_xticks([t_n[0], t_n[int(len(self.dataset["data"])/2)], t_n[-1]])
            #             ax3.set_xticklabels([str(int(t_n[0])), str(int((t_n[int(len(self.dataset["data"])/2)]))), str(int(t_n[-1]))], fontsize=font_size)
            #             ax3.set_yticks([int(np.amin(sorted_data)), int((np.amax(sorted_data)+np.amin(sorted_data))/2), int(np.amax(sorted_data))])
            #             ax3.set_yticklabels(['', '', ''], fontsize=font_size)
            #     else:
            #         # Configure legend
            #         if (self.dataset["ground_truths"] is not None) and (self.dataset["parameters"] is not None):
            #             ax0.legend([Line2D([0], [0], color=gtcolor,lw=3.0),
            #                             Line2D([0], [0], color=learncolor,lw=3.0),c],
            #                         ['Ground truth', 'BNP-Step', 'Data'], handler_map={mpatches.Circle: HandlerCircle()},
            #                         bbox_to_anchor=(0., 1.15, 1., .102), loc='lower center',
            #                                 ncol=3, mode="none", borderaxespad=0., edgecolor=(1.0, 1.0, 1.0, 0.0), prop=fnt_mgr, borderpad=0.8)
            #         else:
            #             ax0.legend([Line2D([0], [0], color=learncolor,lw=3.0), c],
            #                     ['BNP-Step', 'Data'], handler_map={mpatches.Circle: HandlerCircle()},
            #                     bbox_to_anchor=(0., 1.15, 1., .102), loc='lower center',
            #                             ncol=2, mode="none", borderaxespad=0., edgecolor=(1.0, 1.0, 1.0, 0.0), prop=fnt_mgr, borderpad=0.8)

            #     # Show plot, then save figure
            #     plt.show()
            #     # TODO: add user option to generate figure type other than pdf
            #     output_filename = fig_savename + '_' + plot + '.pdf'
            #     fig.savefig(output_filename, format='pdf')
            if (plot == 'step'):
                # General figure setup
                fig = plt.figure()

                if plot_alt_results and len(alt_results) == 1:
                    gs1 = GridSpec(1, 1, bottom=0.8)  # For legend
                    ax0 = fig.add_subplot(gs1[0])
                    ax0.axis('off')
                    gs2 = GridSpec(1, 2, top=0.85, wspace=0.07, hspace=0.05)  # For actual plot
                    ax1 = fig.add_subplot(gs2[0])
                    ax2 = fig.add_subplot(gs2[1])
                elif plot_alt_results and len(alt_results) == 2:
                    gs1 = GridSpec(1, 1, bottom=0.8)  # For legend
                    ax0 = fig.add_subplot(gs1[0])
                    ax0.axis('off')
                    gs2 = GridSpec(1, 3, top=0.85, wspace=0.07, hspace=0.05)  # For actual plot
                    ax1 = fig.add_subplot(gs2[0])
                    ax2 = fig.add_subplot(gs2[1])
                    ax3 = fig.add_subplot(gs2[2])
                else:
                    gs1 = GridSpec(1, 1, bottom=0.8)  # For legend
                    ax0 = fig.add_subplot(gs1[0])
                    ax0.axis('off')
                    gs2 = GridSpec(1, 1, top=0.85, wspace=0.07, hspace=0.05)  # For actual plot
                    ax1 = fig.add_subplot(gs2[0])

                fig.supxlabel(x_label, fontsize=font_size)
                fig.supylabel(y_label, x=0.05, fontsize=font_size)

                # Setup x-axis vectors for plotting
                T = np.linspace(t_n[0], t_n[-1], len(self.dataset["data"]))
                
                samples  = np.array([[0.0, 0.0], [0.0, 0.0]])
                # Find the MAP estimate for our step plot
                map_index_list = np.argsort(post_clean)[-1000:]
                for map_index in map_index_list:
                    step_data                 = bnpa.get_step_plot_data(b_clean , h_clean , t_clean , f_clean, t_n , self.B_max , t_n.size , map_index)
                    samples = np.append(samples,np.squeeze([T,step_data]).T,axis=0)
                samples = samples[2:,:]
                # Plot discovered steps
                ax1.hist2d(samples[:,0],samples[:,1],bins = 50,cmap = 'Blues')
                # Plot synthetic data
                ax1.scatter(T, self.dataset["data"], alpha=0.7, color=datacolor, facecolors='none', marker='.')
                ax1.stairs(step_data, np.insert(T,0,T[0]), baseline=None, color=learncolor, linewidth=1.0, zorder=10)


                # If we have ground truths, plot them
                if (self.dataset["ground_truths"] is not None) and (self.dataset["parameters"] is not None):
                    if self.dataset["parameters"]["type"] == 'kv':
                        ground_times, ground_data = bnpa.generate_gt_step_plot_data(self.dataset["ground_truths"]["b_m"], self.dataset["ground_truths"]["h_m"], self.dataset["ground_truths"]["t_m"], self.dataset["parameters"]["f_back"], t_n, self.B_max)
                        ax1.stairs(ground_data, ground_times, baseline=None, color=gtcolor, linewidth=3.0, zorder=5.0)
                    if self.dataset["parameters"]["type"] == 'hmm':
                        ground_data = self.dataset["ground_truths"]["u"]
                        gt_times = np.asarray([0])
                        gt_times = np.append(gt_times, t_n)
                        ax1.stairs(ground_data, gt_times, baseline=None, color=gtcolor, linewidth=3.0, zorder=5.0)


                # Add subplot titles
                # ax1.set_title('SNR = 2.0', font=fpath, fontsize=fntsize)
                
                # Configure axes
                ax1.set_xticks([t_n[0], t_n[int(len(self.dataset["data"])/2)], t_n[-1]])
                ax1.set_xticklabels([str(int(t_n[0])), str(int((t_n[int(len(self.dataset["data"])/2)]))), str(int(t_n[-1]))], fontsize=font_size)

                # ax1.set_yticks([int(np.amin(sorted_data)), int((np.amax(sorted_data)+np.amin(sorted_data))/2), int(np.amax(sorted_data))])
                # ax1.set_yticklabels([str(int(np.amin(sorted_data))), str(int((np.amax(sorted_data)+np.amin(sorted_data))/2)), str(int(np.amax(sorted_data)))], fontsize=font_size)

                # If we also have alternative results, plot those and configure the axes.
                if plot_alt_results:
                    if len(alt_results) == 1:
                        ax2.scatter(T, self.dataset["data"], alpha=0.7, color=datacolor, facecolors='none', marker='.')
                        
                        # Alternative method plotting
                        if alt_results[0] == 'kv':
                            plot_times, plot_heights = bnpa.generate_kv_step_plot_data(self.alt_method_results_kv["jump_times"], 
                                                                                    self.alt_method_results_kv["means"], 
                                                                                    self.alt_method_results_kv["background"], 
                                                                                    t_n)
                            ax2.stairs(plot_heights, plot_times, baseline=None, color=alt_color1, linewidth=3.0, zorder=10)
                            # Generate legend
                            if (self.dataset["ground_truths"] is not None) and (self.dataset["parameters"] is not None):
                                ax0.legend([Line2D([0], [0], color=gtcolor,lw=3.0),
                                                Line2D([0], [0], color=learncolor,lw=3.0), Line2D([0], [0], color=alt_color1,lw=3.0),c],
                                            ['Ground truth', 'BNP-Step', 'BIC', 'Data'], handler_map={mpatches.Circle: HandlerCircle()},
                                            bbox_to_anchor=(0., 1.15, 1., .102), loc='lower center',
                                                    ncol=4, mode="none", borderaxespad=0., edgecolor=(1.0, 1.0, 1.0, 0.0), prop=fnt_mgr, borderpad=0.8)
                            else:
                                ax0.legend([Line2D([0], [0], color=learncolor,lw=3.0), Line2D([0], [0], color=alt_color1,lw=3.0), c],
                                        ['BNP-Step', 'BIC', 'Data'], handler_map={mpatches.Circle: HandlerCircle()},
                                        bbox_to_anchor=(0., 1.15, 1., .102), loc='lower center',
                                                ncol=3, mode="none", borderaxespad=0., edgecolor=(1.0, 1.0, 1.0, 0.0), prop=fnt_mgr, borderpad=0.8)
                        elif alt_results[0] == 'ihmm':
                            sampled_heights = self.alt_method_results_ihmm["mode_mean_traj"]
                            sampled_times = t_n.copy()
                            sampled_times = np.append(sampled_times, t_n[-1])
                            ax2.stairs(sampled_heights, sampled_times, baseline=None, color=alt_color4, linewidth=3.0, zorder=10)
                            # Generate legend
                            if (self.dataset["ground_truths"] is not None) and (self.dataset["parameters"] is not None):
                                ax0.legend([Line2D([0], [0], color=gtcolor,lw=3.0),
                                                Line2D([0], [0], color=learncolor,lw=3.0), Line2D([0], [0], color=alt_color4,lw=3.0),c],
                                            ['Ground truth', 'BNP-Step', 'iHMM', 'Data'], handler_map={mpatches.Circle: HandlerCircle()},
                                            bbox_to_anchor=(0., 1.15, 1., .102), loc='lower center',
                                                    ncol=4, mode="none", borderaxespad=0., edgecolor=(1.0, 1.0, 1.0, 0.0), prop=fnt_mgr, borderpad=0.8)
                            else:
                                ax0.legend([Line2D([0], [0], color=learncolor,lw=3.0), Line2D([0], [0], color=alt_color4,lw=3.0), c],
                                        ['BNP-Step', 'iHMM', 'Data'], handler_map={mpatches.Circle: HandlerCircle()},
                                        bbox_to_anchor=(0., 1.15, 1., .102), loc='lower center',
                                                ncol=3, mode="none", borderaxespad=0., edgecolor=(1.0, 1.0, 1.0, 0.0), prop=fnt_mgr, borderpad=0.8)
                        # Plot ground truths too if we have them
                        if (self.dataset["ground_truths"] is not None) and (self.dataset["parameters"] is not None):
                            if self.dataset["parameters"]["type"] == 'kv':
                                ground_times, ground_data = bnpa.generate_gt_step_plot_data(self.dataset["ground_truths"]["b_m"], self.dataset["ground_truths"]["h_m"], self.dataset["ground_truths"]["t_m"], self.dataset["parameters"]["f_back"], t_n, self.B_max)
                                ax2.stairs(ground_data, ground_times, baseline=None, color=gtcolor, linewidth=3.0, zorder=5)
                            if self.dataset["parameters"]["type"] == 'hmm':
                                ground_data = self.dataset["ground_truths"]["u"]
                                gt_times = np.asarray([0])
                                gt_times = np.append(gt_times, t_n)
                                ax2.stairs(ground_data, gt_times, baseline=None, color=gtcolor, linewidth=3.0, zorder=5.0)
                        # Configure axes
                        ax2.set_xticks([t_n[0], t_n[int(len(self.dataset["data"])/2)], t_n[-1]])
                        ax2.set_xticklabels([str(int(t_n[0])), str(int((t_n[int(len(self.dataset["data"])/2)]))), str(int(t_n[-1]))], fontsize=font_size)
                        ax2.set_yticks([int(np.amin(sorted_data)), int((np.amax(sorted_data)+np.amin(sorted_data))/2), int(np.amax(sorted_data))])
                        ax2.set_yticklabels(['', '', ''], fontsize=font_size)
                    elif len(alt_results) == 2:
                        ax2.scatter(T, self.dataset["data"], alpha=0.7, color=datacolor, facecolors='none', marker='.')
                        ax3.scatter(T, self.dataset["data"], alpha=0.7, color=datacolor, facecolors='none', marker='.')
                        
                        # Plot KV first, then iHMM
                        plot_times, plot_heights = bnpa.generate_kv_step_plot_data(self.alt_method_results_kv["jump_times"], 
                                                                                    self.alt_method_results_kv["means"], 
                                                                                    self.alt_method_results_kv["background"], 
                                                                                    t_n)
                        ax2.stairs(plot_heights, plot_times, baseline=None, color=alt_color1, linewidth=3.0, zorder=10)

                        sampled_heights = self.alt_method_results_ihmm["mode_mean_traj"]
                        sampled_times = t_n.copy()
                        sampled_times = np.append(sampled_times, t_n[-1])
                        ax3.stairs(sampled_heights, sampled_times, baseline=None, color=alt_color4, linewidth=3.0, zorder=10)

                        # Generate legend
                        if (self.dataset["ground_truths"] is not None) and (self.dataset["parameters"] is not None):
                                ax0.legend([Line2D([0], [0], color=gtcolor,lw=3.0),
                                                Line2D([0], [0], color=learncolor,lw=3.0), Line2D([0], [0], color=alt_color1,lw=3.0), Line2D([0], [0], color=alt_color4,lw=3.0),c],
                                            ['Ground truth', 'BNP-Step', 'BIC', 'iHMM', 'Data'], handler_map={mpatches.Circle: HandlerCircle()},
                                            bbox_to_anchor=(0., 1.15, 1., .102), loc='lower center',
                                                    ncol=5, mode="none", borderaxespad=0., edgecolor=(1.0, 1.0, 1.0, 0.0), prop=fnt_mgr, borderpad=0.8)
                        else:
                            ax0.legend([Line2D([0], [0], color=learncolor,lw=3.0), Line2D([0], [0], color=alt_color1,lw=3.0), Line2D([0], [0], color=alt_color4,lw=3.0), c],
                                    ['BNP-Step', 'BIC', 'iHMM', 'Data'], handler_map={mpatches.Circle: HandlerCircle()},
                                    bbox_to_anchor=(0., 1.15, 1., .102), loc='lower center',
                                            ncol=4, mode="none", borderaxespad=0., edgecolor=(1.0, 1.0, 1.0, 0.0), prop=fnt_mgr, borderpad=0.8)
                                
                        # Plot ground truths too if we have them
                        if (self.dataset["ground_truths"] is not None) and (self.dataset["parameters"] is not None):
                            if self.dataset["parameters"]["type"] == 'kv':
                                ground_times, ground_data = bnpa.generate_gt_step_plot_data(self.dataset["ground_truths"]["b_m"], self.dataset["ground_truths"]["h_m"], self.dataset["ground_truths"]["t_m"], self.dataset["parameters"]["f_back"], t_n, self.B_max)
                                ax2.stairs(ground_data, ground_times, baseline=None, color=gtcolor, linewidth=3.0, zorder=5)
                                ax3.stairs(ground_data, ground_times, baseline=None, color=gtcolor, linewidth=3.0, zorder=5)
                            if self.dataset["parameters"]["type"] == 'hmm':
                                ground_data = self.dataset["ground_truths"]["u"]
                                gt_times = np.asarray([0])
                                gt_times = np.append(gt_times, t_n)
                                ax2.stairs(ground_data, gt_times, baseline=None, color=gtcolor, linewidth=3.0, zorder=5.0)
                                ax3.stairs(ground_data, gt_times, baseline=None, color=gtcolor, linewidth=3.0, zorder=5.0)

                        # Configure axes
                        ax2.set_xticks([t_n[0], t_n[int(len(self.dataset["data"])/2)], t_n[-1]])
                        ax2.set_xticklabels([str(int(t_n[0])), str(int((t_n[int(len(self.dataset["data"])/2)]))), str(int(t_n[-1]))], fontsize=font_size)
                        ax2.set_yticks([int(np.amin(sorted_data)), int((np.amax(sorted_data)+np.amin(sorted_data))/2), int(np.amax(sorted_data))])
                        ax2.set_yticklabels(['', '', ''], fontsize=font_size)

                        ax3.set_xticks([t_n[0], t_n[int(len(self.dataset["data"])/2)], t_n[-1]])
                        ax3.set_xticklabels([str(int(t_n[0])), str(int((t_n[int(len(self.dataset["data"])/2)]))), str(int(t_n[-1]))], fontsize=font_size)
                        ax3.set_yticks([int(np.amin(sorted_data)), int((np.amax(sorted_data)+np.amin(sorted_data))/2), int(np.amax(sorted_data))])
                        ax3.set_yticklabels(['', '', ''], fontsize=font_size)
                else:
                    # Configure legend
                    if (self.dataset["ground_truths"] is not None) and (self.dataset["parameters"] is not None):
                        ax0.legend([Line2D([0], [0], color=gtcolor,lw=3.0),
                                        Line2D([0], [0], color=learncolor,lw=3.0),c],
                                    ['Ground truth', 'BNP-Step', 'Data'], handler_map={mpatches.Circle: HandlerCircle()},
                                    bbox_to_anchor=(0., 1.15, 1., .102), loc='lower center',
                                            ncol=3, mode="none", borderaxespad=0., edgecolor=(1.0, 1.0, 1.0, 0.0), prop=fnt_mgr, borderpad=0.8)
                    else:
                        ax0.legend([Line2D([0], [0], color=learncolor,lw=3.0), c],
                                ['BNP-Step', 'Data'], handler_map={mpatches.Circle: HandlerCircle()},
                                bbox_to_anchor=(0., 1.15, 1., .102), loc='lower center',
                                        ncol=2, mode="none", borderaxespad=0., edgecolor=(1.0, 1.0, 1.0, 0.0), prop=fnt_mgr, borderpad=0.8)

                # Show plot, then save figure
                plt.show()
                # TODO: add user option to generate figure type other than pdf
                output_filename = fig_savename + '_' + plot + '.pdf'
                fig.savefig(output_filename, format='pdf')
            elif (plot == 'hist_step_height'):
                # General figure setup
                fig = plt.figure()

                if plot_alt_results and len(alt_results) == 1:
                    gs1 = GridSpec(1, 1, bottom=0.8)  # For legend
                    ax0 = fig.add_subplot(gs1[0])
                    ax0.axis('off')
                    gs2 = GridSpec(1, 2, top=0.85, wspace=0.07, hspace=0.05)  # For actual plot
                    ax1 = fig.add_subplot(gs2[0])
                    ax2 = fig.add_subplot(gs2[1])
                elif plot_alt_results and len(alt_results) == 2:
                    gs1 = GridSpec(1, 1, bottom=0.8)  # For legend
                    ax0 = fig.add_subplot(gs1[0])
                    ax0.axis('off')
                    gs2 = GridSpec(1, 3, top=0.85, wspace=0.07, hspace=0.05)  # For actual plot
                    ax1 = fig.add_subplot(gs2[0])
                    ax2 = fig.add_subplot(gs2[1])
                    ax3 = fig.add_subplot(gs2[2])
                else:
                    gs1 = GridSpec(1, 1, bottom=0.8)  # For legend
                    ax0 = fig.add_subplot(gs1[0])
                    ax0.axis('off')
                    gs2 = GridSpec(1, 1, top=0.85, wspace=0.07, hspace=0.05)  # For actual plot
                    ax1 = fig.add_subplot(gs2[0])

                fig.supxlabel(x_label, fontsize=font_size)
                fig.supylabel(y_label, x=0.05, fontsize=font_size)

                # Legend setup
                orange_patch = mpatches.Patch(color=learncolor, alpha=0.5, label='BNP-Step')
                purple_patch = mpatches.Patch(color=alt_color1, alpha=0.5, label='BIC')
                yellow_patch = mpatches.Patch(color=alt_color4, alpha=0.5, label='iHMM')
                cyan_patch = mpatches.Patch(color=gtcolor, label='Ground truths')
                magenta_patch = mpatches.Patch(color=alt_color2, label='Prior')
                red_patch = mpatches.Patch(color=alt_color3, alpha=0.2, label='95% CR')
                
                # Prepare fixed bins - NYI, should be an option in input arguments
                #fixed_bins = np.arange(0,41,2)

                # Generate histogram sets
                good_b_m, good_h_m, good_t_m, good_f_s, good_eta = bnpa.find_top_samples_by_jumps(b_clean, h_clean, t_clean, f_clean, eta_clean, post_clean)
                # TODO: let the user choose how many samples go into the histograms.
                hist_heights, _ = bnpa.generate_histogram_data(good_b_m, good_h_m, good_t_m, len(good_eta), self.B_max, t_n)
                
                # Plotting - todo: set bins=fixed_bins
                n, bins, patches = ax1.hist(hist_heights, alpha=0.5, edgecolor=learncolor, color=learncolor, density=True, histtype='stepfilled')

                # Legend parameters
                handle_array = [orange_patch]
                num_cols = 1

                # Configure axes and plot alternative methods' results if we have them
                if plot_alt_results:
                    if len(alt_results) == 1:
                        if alt_results[0] == 'kv':
                            hist_heights_kv, _ = bnpa.generate_histogram_data_kv(self.alt_method_results_kv["means"], self.alt_method_results_kv["jump_times"])
                            # TODO: add option for bins=fixed_bins
                            n_kv, bins_kv, patches_kv = ax2.hist(hist_heights_kv, alpha=0.5, edgecolor=alt_color1, color=alt_color1, density=True, histtype='stepfilled')
                            bins_tot_h = np.concatenate((bins, bins_kv))
                            n_tot_h = np.concatenate((n, n_kv))
                            # Configure axes
                            ax1.set_ylim(0,np.amax(n_tot_h)+np.amax(n_tot_h)*0.1)
                            ax2.set_ylim(0,np.amax(n_tot_h)+np.amax(n_tot_h)*0.1)
                            ax1.set_yticks([0, np.amax(n_tot_h)/2, np.amax(n_tot_h)])
                            ax1.set_yticklabels(['0', f'{np.amax(n_tot_h)/2:.1f}', f'{np.amax(n_tot_h):.1f}'], fontsize=font_size)
                            ax2.set_yticks([0, np.amax(n_tot_h)/2, np.amax(n_tot_h)])
                            ax2.set_yticklabels(['', '', ''], fontsize=font_size)
                            ax1.set_xlim(int(np.amin(bins_tot_h)-2),int(np.amax(bins_tot_h)+2))
                            ax2.set_xlim(int(np.amin(bins_tot_h)-2),int(np.amax(bins_tot_h)+2))
                            ax1.set_xticks([int(np.amin(bins_tot_h)), int((np.amin(bins_tot_h) + np.amax(bins_tot_h)) / 2), int(np.amax(bins_tot_h))])
                            ax1.set_xticklabels([str(int(np.amin(bins_tot_h))), str(int((np.amin(bins_tot_h) + np.amax(bins_tot_h)) / 2)), str(int(np.amax(bins_tot_h)))], fontsize=font_size)
                            ax2.set_xticks([int(np.amin(bins_tot_h)), int((np.amin(bins_tot_h) + np.amax(bins_tot_h)) / 2), int(np.amax(bins_tot_h))])
                            ax2.set_xticklabels([str(int(np.amin(bins_tot_h))), str(int((np.amin(bins_tot_h) + np.amax(bins_tot_h)) / 2)), str(int(np.amax(bins_tot_h)))], fontsize=font_size)
                            if show_prior:
                                ax_1 = ax1.twinx()
                                ax_1.set_ylim(0,np.amax(n_tot_h)+np.amax(n_tot_h)*0.1)
                                ax_1.set_yticks([0, np.amax(n_tot_h)/2, np.amax(n_tot_h)])
                                ax_1.set_yticklabels(['','',''], fontsize=font_size)
                            # Plot ground truths if we have them
                            if (self.dataset["ground_truths"] is not None) and (self.dataset["parameters"] is not None):
                                if self.dataset["parameters"]["type"] == 'kv':
                                    ground_h_trim = np.trim_zeros(self.dataset["ground_truths"]["h_m"])
                                    ax1.vlines(ground_h_trim, 0 ,np.amax(n_tot_h)+np.amax(n_tot_h)*0.1 ,color=gtcolor, zorder=10)
                                    ax2.vlines(ground_h_trim, 0 ,np.amax(n_tot_h)+np.amax(n_tot_h)*0.1 ,color=gtcolor, zorder=10)
                                    handle_array.append(cyan_patch)
                                    num_cols += 1
                                elif self.dataset["parameters"]["type"] == 'hmm':
                                    ground_h_ihmm = []
                                    for itm in range(1, len(self.dataset["ground_truths"]["u"])):
                                        if self.dataset["ground_truths"]["u"][itm] != self.dataset["ground_truths"]["u"][itm-1]:
                                            ground_h_ihmm.append(self.dataset["ground_truths"]["u"][itm] - self.dataset["ground_truths"]["u"][itm-1])
                                    ground_h_ihmm = np.unique(np.abs(np.asarray(ground_h_ihmm)))
                                    ax1.vlines(ground_h_ihmm, 0 ,np.amax(n)+np.amax(n)*0.1 ,color=gtcolor, zorder=10)
                                    ax2.vlines(ground_h_ihmm, 0 ,np.amax(n)+np.amax(n)*0.1 ,color=gtcolor, zorder=10)
                                    handle_array.append(cyan_patch)
                                    num_cols += 1
                            handle_array.append(purple_patch)
                            num_cols += 1
                        elif alt_results[0] == 'ihmm':
                            hist_heights_ihmm, _ = bnpa.generate_histogram_data_ihmm(self.alt_method_results_ihmm["samples"], t_n)
                            # TODO: add option for bins=fixed_bins
                            n_kv, bins_kv, patches_kv = ax2.hist(hist_heights_ihmm, alpha=0.5, edgecolor=alt_color4, color=alt_color4, density=True, histtype='stepfilled')
                            bins_tot_h = np.concatenate((bins, bins_kv))
                            n_tot_h = np.concatenate((n, n_kv))
                            # Configure axes
                            ax1.set_ylim(0,np.amax(n_tot_h)+np.amax(n_tot_h)*0.1)
                            ax2.set_ylim(0,np.amax(n_tot_h)+np.amax(n_tot_h)*0.1)
                            ax1.set_yticks([0, np.amax(n_tot_h)/2, np.amax(n_tot_h)])
                            ax1.set_yticklabels(['0', f'{np.amax(n_tot_h)/2:.1f}', f'{np.amax(n_tot_h):.1f}'], fontsize=font_size)
                            ax2.set_yticks([0, np.amax(n_tot_h)/2, np.amax(n_tot_h)])
                            ax2.set_yticklabels(['', '', ''], fontsize=font_size)
                            ax1.set_xlim(int(np.amin(bins_tot_h)-2),int(np.amax(bins_tot_h)+2))
                            ax2.set_xlim(int(np.amin(bins_tot_h)-2),int(np.amax(bins_tot_h)+2))
                            ax1.set_xticks([int(np.amin(bins_tot_h)), int((np.amin(bins_tot_h) + np.amax(bins_tot_h)) / 2), int(np.amax(bins_tot_h))])
                            ax1.set_xticklabels([str(int(np.amin(bins_tot_h))), str(int((np.amin(bins_tot_h) + np.amax(bins_tot_h)) / 2)), str(int(np.amax(bins_tot_h)))], fontsize=font_size)
                            ax2.set_xticks([int(np.amin(bins_tot_h)), int((np.amin(bins_tot_h) + np.amax(bins_tot_h)) / 2), int(np.amax(bins_tot_h))])
                            ax2.set_xticklabels([str(int(np.amin(bins_tot_h))), str(int((np.amin(bins_tot_h) + np.amax(bins_tot_h)) / 2)), str(int(np.amax(bins_tot_h)))], fontsize=font_size)
                            if show_prior:
                                ax_1 = ax1.twinx()
                                ax_1.set_ylim(0,np.amax(n_tot_h)+np.amax(n_tot_h)*0.1)
                                ax_1.set_yticks([0, np.amax(n_tot_h)/2, np.amax(n_tot_h)])
                                ax_1.set_yticklabels(['','',''], fontsize=font_size)
                            if (self.dataset["ground_truths"] is not None) and (self.dataset["parameters"] is not None):
                                if self.dataset["parameters"]["type"] == 'kv':
                                    ground_h_trim = np.trim_zeros(self.dataset["ground_truths"]["h_m"])
                                    ax1.vlines(ground_h_trim, 0 ,np.amax(n_tot_h)+np.amax(n_tot_h)*0.1 ,color=gtcolor, zorder=10)
                                    ax2.vlines(ground_h_trim, 0 ,np.amax(n_tot_h)+np.amax(n_tot_h)*0.1 ,color=gtcolor, zorder=10)
                                    handle_array.append(cyan_patch)
                                    num_cols += 1
                                elif self.dataset["parameters"]["type"] == 'hmm':
                                    ground_h_ihmm = []
                                    for itm in range(1, len(self.dataset["ground_truths"]["u"])):
                                        if self.dataset["ground_truths"]["u"][itm] != self.dataset["ground_truths"]["u"][itm-1]:
                                            ground_h_ihmm.append(self.dataset["ground_truths"]["u"][itm] - self.dataset["ground_truths"]["u"][itm-1])
                                    ground_h_ihmm = np.unique(np.abs(np.asarray(ground_h_ihmm)))
                                    ax1.vlines(ground_h_ihmm, 0 ,np.amax(n)+np.amax(n)*0.1 ,color=gtcolor, zorder=10)
                                    ax2.vlines(ground_h_ihmm, 0 ,np.amax(n)+np.amax(n)*0.1 ,color=gtcolor, zorder=10)
                                    handle_array.append(cyan_patch)
                                    num_cols += 1
                            handle_array.append(yellow_patch)
                            num_cols += 1
                    elif len(alt_results) == 2:
                        hist_heights_kv, _ = bnpa.generate_histogram_data_kv(self.alt_method_results_kv["means"], self.alt_method_results_kv["jump_times"])
                        hist_heights_ihmm, _ = bnpa.generate_histogram_data_ihmm(self.alt_method_results_ihmm["samples"], t_n)
                        # TODO: add option for bins=fixed_bins
                        n_kv, bins_kv, patches_kv = ax2.hist(hist_heights_kv, alpha=0.5, edgecolor=alt_color1, color=alt_color1, density=True, histtype='stepfilled')
                        n_kv2, bins_kv2, patches_kv2 = ax3.hist(hist_heights_ihmm, alpha=0.5, edgecolor=alt_color4, color=alt_color4, density=True, histtype='stepfilled')
                        bins_tot_h = np.concatenate((bins, bins_kv, bins_kv2))
                        n_tot_h = np.concatenate((n, n_kv, n_kv2))
                        # Configure axes
                        ax1.set_ylim(0,np.amax(n_tot_h)+np.amax(n_tot_h)*0.1)
                        ax2.set_ylim(0,np.amax(n_tot_h)+np.amax(n_tot_h)*0.1)
                        ax3.set_ylim(0,np.amax(n_tot_h)+np.amax(n_tot_h)*0.1)
                        ax1.set_yticks([0, np.amax(n_tot_h)/2, np.amax(n_tot_h)])
                        ax1.set_yticklabels(['0', f'{np.amax(n_tot_h)/2:.1f}', f'{np.amax(n_tot_h):.1f}'], fontsize=font_size)
                        ax2.set_yticks([0, np.amax(n_tot_h)/2, np.amax(n_tot_h)])
                        ax2.set_yticklabels(['', '', ''], fontsize=font_size)
                        ax3.set_yticks([0, np.amax(n_tot_h)/2, np.amax(n_tot_h)])
                        ax3.set_yticklabels(['', '', ''], fontsize=font_size)
                        ax1.set_xlim(int(np.amin(bins_tot_h)-2),int(np.amax(bins_tot_h)+2))
                        ax2.set_xlim(int(np.amin(bins_tot_h)-2),int(np.amax(bins_tot_h)+2))
                        ax3.set_xlim(int(np.amin(bins_tot_h)-2),int(np.amax(bins_tot_h)+2))
                        ax1.set_xticks([int(np.amin(bins_tot_h)), int((np.amin(bins_tot_h) + np.amax(bins_tot_h)) / 2), int(np.amax(bins_tot_h))])
                        ax1.set_xticklabels([str(int(np.amin(bins_tot_h))), str(int((np.amin(bins_tot_h) + np.amax(bins_tot_h)) / 2)), str(int(np.amax(bins_tot_h)))], fontsize=font_size)
                        ax2.set_xticks([int(np.amin(bins_tot_h)), int((np.amin(bins_tot_h) + np.amax(bins_tot_h)) / 2), int(np.amax(bins_tot_h))])
                        ax2.set_xticklabels([str(int(np.amin(bins_tot_h))), str(int((np.amin(bins_tot_h) + np.amax(bins_tot_h)) / 2)), str(int(np.amax(bins_tot_h)))], fontsize=font_size)
                        ax3.set_xticks([int(np.amin(bins_tot_h)), int((np.amin(bins_tot_h) + np.amax(bins_tot_h)) / 2), int(np.amax(bins_tot_h))])
                        ax3.set_xticklabels([str(int(np.amin(bins_tot_h))), str(int((np.amin(bins_tot_h) + np.amax(bins_tot_h)) / 2)), str(int(np.amax(bins_tot_h)))], fontsize=font_size)
                        if show_prior:
                            ax_1 = ax1.twinx()
                            ax_1.set_ylim(0,np.amax(n_tot_h)+np.amax(n_tot_h)*0.1)
                            ax_1.set_yticks([0, np.amax(n_tot_h)/2, np.amax(n_tot_h)])
                            ax_1.set_yticklabels(['','',''], fontsize=font_size)
                        # Plot ground truths if we have them
                        if (self.dataset["ground_truths"] is not None) and (self.dataset["parameters"] is not None):
                            if self.dataset["parameters"]["type"] == 'kv':
                                ground_h_trim = np.trim_zeros(self.dataset["ground_truths"]["h_m"])
                                ax1.vlines(ground_h_trim, 0 ,np.amax(n_tot_h)+np.amax(n_tot_h)*0.1 ,color=gtcolor, zorder=10)
                                ax2.vlines(ground_h_trim, 0 ,np.amax(n_tot_h)+np.amax(n_tot_h)*0.1 ,color=gtcolor, zorder=10)
                                ax3.vlines(ground_h_trim, 0 ,np.amax(n_tot_h)+np.amax(n_tot_h)*0.1 ,color=gtcolor, zorder=10)
                                handle_array.append(cyan_patch)
                                num_cols += 1
                            elif self.dataset["parameters"]["type"] == 'hmm':
                                ground_h_ihmm = []
                                for itm in range(1, len(self.dataset["ground_truths"]["u"])):
                                    if self.dataset["ground_truths"]["u"][itm] != self.dataset["ground_truths"]["u"][itm-1]:
                                        ground_h_ihmm.append(self.dataset["ground_truths"]["u"][itm] - self.dataset["ground_truths"]["u"][itm-1])
                                ground_h_ihmm = np.unique(np.abs(np.asarray(ground_h_ihmm)))
                                ax1.vlines(ground_h_ihmm, 0 ,np.amax(n)+np.amax(n)*0.1 ,color=gtcolor, zorder=10)
                                ax2.vlines(ground_h_ihmm, 0 ,np.amax(n)+np.amax(n)*0.1 ,color=gtcolor, zorder=10)
                                ax3.vlines(ground_h_ihmm, 0 ,np.amax(n)+np.amax(n)*0.1 ,color=gtcolor, zorder=10)
                                handle_array.append(cyan_patch)
                                num_cols += 1
                        handle_array.append(purple_patch)
                        handle_array.append(yellow_patch)
                        num_cols += 2
                else:
                    ax1.set_ylim(0,np.amax(n)+np.amax(n)*0.1)
                    ax1.set_yticks([0, np.amax(n)/2, np.amax(n)])
                    ax1.set_yticklabels(['0', f'{np.amax(n)/2:.1f}', f'{np.amax(n):.1f}'], fontsize=font_size)
                    ax1.set_xlim(int(np.amin(bins)-2),int(np.amax(bins)+2))
                    ax1.set_xticks([int(np.amin(bins)), int((np.amin(bins) + np.amax(bins)) / 2), int(np.amax(bins))])
                    ax1.set_xticklabels([str(int(np.amin(bins))), str(int((np.amin(bins) + np.amax(bins)) / 2)), str(int(np.amax(bins)))], fontsize=font_size)
                    if show_prior:
                        ax_1 = ax1.twinx()
                        ax_1.set_ylim(0,np.amax(n)+np.amax(n)*0.1)
                        ax_1.set_yticks([0, np.amax(n)/2, np.amax(n)])
                        ax_1.set_yticklabels(['','',''], fontsize=font_size)
                    # Plot ground truths if we have them
                    if (self.dataset["ground_truths"] is not None) and (self.dataset["parameters"] is not None):
                        if self.dataset["parameters"]["type"] == 'kv':
                            ground_h_trim = np.trim_zeros(self.dataset["ground_truths"]["h_m"])
                            ax1.vlines(ground_h_trim, 0 ,np.amax(n)+np.amax(n)*0.1 ,color=gtcolor, zorder=10)
                        elif self.dataset["parameters"]["type"] == 'hmm':
                            ground_h_ihmm = []
                            for itm in range(1, len(self.dataset["ground_truths"]["u"])):
                                if self.dataset["ground_truths"]["u"][itm] != self.dataset["ground_truths"]["u"][itm-1]:
                                    ground_h_ihmm.append(self.dataset["ground_truths"]["u"][itm] - self.dataset["ground_truths"]["u"][itm-1])
                            ground_h_ihmm = np.unique(np.abs(np.asarray(ground_h_ihmm)))
                            ax1.vlines(ground_h_ihmm, 0 ,np.amax(n)+np.amax(n)*0.1 ,color=gtcolor, zorder=10)
                            handle_array.append(cyan_patch)
                            num_cols += 1

                # Plot prior if we chose this option
                if show_prior:
                    prior_x = np.linspace(self.h_ref-2*(1/np.sqrt(self.chi)), self.h_ref+2*(1/np.sqrt(self.chi)), 200)
                    prior_h = np.sqrt(self.chi/(2*np.pi))*np.exp(-(self.chi/2)*(prior_x-self.h_ref)**2)
                    ax_1.plot(prior_x, prior_h,label='Prior', color=alt_color2, zorder=5)
                    handle_array.append(magenta_patch)
                    num_cols += 1
                
                # Plot CI if we chose this option
                # TODO: add options to show 50% CI or have user-defined CI
                if show_ci:
                    y_limits_ci = [0, 100]
                    mean, under95, under50, median, upper50, upper95 = bnpa.get_credible_intervals(hist_heights)
                    ax1.fill_betweenx(y=y_limits_ci, x1=upper95, x2=under95, color=alt_color3, alpha=0.2, zorder=-10)
                    handle_array.append(red_patch)
                    num_cols += 1

                # Generate legend
                ax0.legend(bbox_to_anchor=(0., 1.08, 1., .102), handles=handle_array,
                                loc='lower center', ncol=num_cols, mode="none", borderaxespad=0., borderpad=0.8, edgecolor=(1.0, 1.0, 1.0, 0.0), prop=fnt_mgr)
                            
                # Show plot, then save figure
                plt.show()
                # TODO: add user option to generate figure type other than pdf
                output_filename = fig_savename + '_' + plot + '.pdf'
                fig.savefig(output_filename, format='pdf')

            elif (plot == 'hist_dwell_time'):
                # General figure setup
                fig = plt.figure()

                if plot_alt_results and len(alt_results) == 1:
                    gs1 = GridSpec(1, 1, bottom=0.8)  # For legend
                    ax0 = fig.add_subplot(gs1[0])
                    ax0.axis('off')
                    gs2 = GridSpec(1, 2, top=0.85, wspace=0.07, hspace=0.05)  # For actual plot
                    ax1 = fig.add_subplot(gs2[0])
                    ax2 = fig.add_subplot(gs2[1])
                elif plot_alt_results and len(alt_results) == 2:
                    gs1 = GridSpec(1, 1, bottom=0.8)  # For legend
                    ax0 = fig.add_subplot(gs1[0])
                    ax0.axis('off')
                    gs2 = GridSpec(1, 3, top=0.85, wspace=0.07, hspace=0.05)  # For actual plot
                    ax1 = fig.add_subplot(gs2[0])
                    ax2 = fig.add_subplot(gs2[1])
                    ax3 = fig.add_subplot(gs2[2])
                else:
                    gs1 = GridSpec(1, 1, bottom=0.8)  # For legend
                    ax0 = fig.add_subplot(gs1[0])
                    ax0.axis('off')
                    gs2 = GridSpec(1, 1, top=0.85, wspace=0.07, hspace=0.05)  # For actual plot
                    ax1 = fig.add_subplot(gs2[0])

                fig.supxlabel(x_label, fontsize=font_size)
                fig.supylabel(y_label, x=0.05, fontsize=font_size)

                # Legend setup
                orange_patch = mpatches.Patch(color=learncolor, alpha=0.5, label='BNP-Step')
                purple_patch = mpatches.Patch(color=alt_color1, alpha=0.5, label='BIC')
                yellow_patch = mpatches.Patch(color=alt_color4, alpha=0.5, label='iHMM')
                cyan_patch = mpatches.Patch(color=gtcolor, label='Ground truths')
                magenta_patch = mpatches.Patch(color=alt_color2, label='Prior')
                red_patch = mpatches.Patch(color=alt_color3, alpha=0.2, label='95% CR')
                
                # Prepare fixed bins - NYI, should be an option in input arguments
                #fixed_bins = np.arange(0,41,2)

                # Generate histogram sets
                good_b_m, good_h_m, good_t_m, good_f_s, good_eta = bnpa.find_top_samples_by_jumps(b_clean, h_clean, t_clean, f_clean, eta_clean, post_clean)
                # TODO: let the user choose how many samples go into the histograms.
                _, hist_dwells = bnpa.generate_histogram_data(good_b_m, good_h_m, good_t_m, len(good_eta), self.B_max, t_n)
                
                # Plotting - todo: set bins=fixed_bins
                n, bins, patches = ax1.hist(hist_dwells, alpha=0.5, edgecolor=learncolor, color=learncolor, density=True, histtype='stepfilled')

                # Legend parameters
                handle_array = [orange_patch]
                num_cols = 1

                # Configure axes and plot alternative methods' results if we have them
                if plot_alt_results:
                    if len(alt_results) == 1:
                        if alt_results[0] == 'kv':
                            _, hist_dwells_kv = bnpa.generate_histogram_data_kv(self.alt_method_results_kv["means"], self.alt_method_results_kv["jump_times"])
                            # TODO: add option for bins=fixed_bins
                            n_kv, bins_kv, patches_kv = ax2.hist(hist_dwells_kv, alpha=0.5, edgecolor=alt_color1, color=alt_color1, density=True, histtype='stepfilled')
                            bins_tot_h = np.concatenate((bins, bins_kv))
                            n_tot_h = np.concatenate((n, n_kv))
                            # Configure axes
                            ax1.set_ylim(0,np.amax(n_tot_h)+np.amax(n_tot_h)*0.1)
                            ax2.set_ylim(0,np.amax(n_tot_h)+np.amax(n_tot_h)*0.1)
                            ax1.set_yticks([0, np.amax(n_tot_h)/2, np.amax(n_tot_h)])
                            ax1.set_yticklabels(['0', f'{np.amax(n_tot_h)/2:.1f}', f'{np.amax(n_tot_h):.1f}'], fontsize=font_size)
                            ax2.set_yticks([0, np.amax(n_tot_h)/2, np.amax(n_tot_h)])
                            ax2.set_yticklabels(['', '', ''], fontsize=font_size)
                            ax1.set_xlim(int(np.amin(bins_tot_h)-2),int(np.amax(bins_tot_h)+2))
                            ax2.set_xlim(int(np.amin(bins_tot_h)-2),int(np.amax(bins_tot_h)+2))
                            ax1.set_xticks([int(np.amin(bins_tot_h)), int((np.amin(bins_tot_h) + np.amax(bins_tot_h)) / 2), int(np.amax(bins_tot_h))])
                            ax1.set_xticklabels([str(int(np.amin(bins_tot_h))), str(int((np.amin(bins_tot_h) + np.amax(bins_tot_h)) / 2)), str(int(np.amax(bins_tot_h)))], fontsize=font_size)
                            ax2.set_xticks([int(np.amin(bins_tot_h)), int((np.amin(bins_tot_h) + np.amax(bins_tot_h)) / 2), int(np.amax(bins_tot_h))])
                            ax2.set_xticklabels([str(int(np.amin(bins_tot_h))), str(int((np.amin(bins_tot_h) + np.amax(bins_tot_h)) / 2)), str(int(np.amax(bins_tot_h)))], fontsize=font_size)
                            if show_prior:
                                ax_1 = ax1.twinx()
                                ax_1.set_ylim(0,np.amax(n_tot_h)+np.amax(n_tot_h)*0.1)
                                ax_1.set_yticks([0, np.amax(n_tot_h)/2, np.amax(n_tot_h)])
                                ax_1.set_yticklabels(['','',''], fontsize=font_size)
                            # Plot ground truths if we have them
                            if (self.dataset["ground_truths"] is not None) and (self.dataset["parameters"] is not None):
                                if self.dataset["parameters"]["type"] == 'kv':
                                    ground_h_trim = np.trim_zeros(self.dataset["ground_truths"]["t_m"])
                                    # Process these so we get holding times, not absolute step times
                                    ground_dwells =[]
                                    for jmps in range(len(ground_h_trim)):
                                        if jmps == 0:
                                            continue
                                        else:
                                            ground_dwells.append(ground_h_trim[jmps] - ground_h_trim[jmps - 1])
                                    ax1.vlines(ground_dwells, 0 ,np.amax(n_tot_h)+np.amax(n_tot_h)*0.1 ,color=gtcolor, zorder=10)
                                    ax2.vlines(ground_dwells, 0 ,np.amax(n_tot_h)+np.amax(n_tot_h)*0.1 ,color=gtcolor, zorder=10)
                                    handle_array.append(cyan_patch)
                                    num_cols += 1
                                elif self.dataset["parameters"]["type"] == 'hmm':
                                    ground_t_ihmm = []
                                    time_prev = t_n[0]
                                    for itm in range(1, len(self.dataset["ground_truths"]["u"])):
                                        if self.dataset["ground_truths"]["u"][itm] != self.dataset["ground_truths"]["u"][itm-1]:
                                            ground_t_ihmm.append(t_n[itm] - time_prev)
                                            time_prev = t_n[itm]
                                    ground_t_ihmm = np.unique(np.abs(np.asarray(ground_t_ihmm)))
                                    ax1.vlines(ground_t_ihmm, 0 ,np.amax(n)+np.amax(n)*0.1 ,color=gtcolor, zorder=10)
                                    ax2.vlines(ground_t_ihmm, 0 ,np.amax(n)+np.amax(n)*0.1 ,color=gtcolor, zorder=10)
                                    handle_array.append(cyan_patch)
                                    num_cols += 1
                            handle_array.append(purple_patch)
                            num_cols += 1
                        elif alt_results[0] == 'ihmm':
                            _, hist_dwells_ihmm = bnpa.generate_histogram_data_ihmm(self.alt_method_results_ihmm["samples"], t_n)
                            # TODO: add option for bins=fixed_bins
                            n_kv, bins_kv, patches_kv = ax2.hist(hist_dwells_ihmm, alpha=0.5, edgecolor=alt_color4, color=alt_color4, density=True, histtype='stepfilled')
                            bins_tot_h = np.concatenate((bins, bins_kv))
                            n_tot_h = np.concatenate((n, n_kv))
                            # Configure axes
                            ax1.set_ylim(0,np.amax(n_tot_h)+np.amax(n_tot_h)*0.1)
                            ax2.set_ylim(0,np.amax(n_tot_h)+np.amax(n_tot_h)*0.1)
                            ax1.set_yticks([0, np.amax(n_tot_h)/2, np.amax(n_tot_h)])
                            ax1.set_yticklabels(['0', f'{np.amax(n_tot_h)/2:.1f}', f'{np.amax(n_tot_h):.1f}'], fontsize=font_size)
                            ax2.set_yticks([0, np.amax(n_tot_h)/2, np.amax(n_tot_h)])
                            ax2.set_yticklabels(['', '', ''], fontsize=font_size)
                            ax1.set_xlim(int(np.amin(bins_tot_h)-2),int(np.amax(bins_tot_h)+2))
                            ax2.set_xlim(int(np.amin(bins_tot_h)-2),int(np.amax(bins_tot_h)+2))
                            ax1.set_xticks([int(np.amin(bins_tot_h)), int((np.amin(bins_tot_h) + np.amax(bins_tot_h)) / 2), int(np.amax(bins_tot_h))])
                            ax1.set_xticklabels([str(int(np.amin(bins_tot_h))), str(int((np.amin(bins_tot_h) + np.amax(bins_tot_h)) / 2)), str(int(np.amax(bins_tot_h)))], fontsize=font_size)
                            ax2.set_xticks([int(np.amin(bins_tot_h)), int((np.amin(bins_tot_h) + np.amax(bins_tot_h)) / 2), int(np.amax(bins_tot_h))])
                            ax2.set_xticklabels([str(int(np.amin(bins_tot_h))), str(int((np.amin(bins_tot_h) + np.amax(bins_tot_h)) / 2)), str(int(np.amax(bins_tot_h)))], fontsize=font_size)
                            if show_prior:
                                ax_1 = ax1.twinx()
                                ax_1.set_ylim(0,np.amax(n_tot_h)+np.amax(n_tot_h)*0.1)
                                ax_1.set_yticks([0, np.amax(n_tot_h)/2, np.amax(n_tot_h)])
                                ax_1.set_yticklabels(['','',''], fontsize=font_size)
                            if (self.dataset["ground_truths"] is not None) and (self.dataset["parameters"] is not None):
                                if self.dataset["parameters"]["type"] == 'kv':
                                    ground_h_trim = np.trim_zeros(self.dataset["ground_truths"]["h_m"])
                                    # Process these so we get holding times, not absolute step times
                                    ground_dwells =[]
                                    for jmps in range(len(ground_h_trim)):
                                        if jmps == 0:
                                            continue
                                        else:
                                            ground_dwells.append(ground_h_trim[jmps] - ground_h_trim[jmps - 1])
                                    ax1.vlines(ground_dwells, 0 ,np.amax(n_tot_h)+np.amax(n_tot_h)*0.1 ,color=gtcolor, zorder=10)
                                    ax2.vlines(ground_dwells, 0 ,np.amax(n_tot_h)+np.amax(n_tot_h)*0.1 ,color=gtcolor, zorder=10)
                                    handle_array.append(cyan_patch)
                                    num_cols += 1
                                elif self.dataset["parameters"]["type"] == 'hmm':
                                    ground_t_ihmm = []
                                    time_prev = t_n[0]
                                    for itm in range(1, len(self.dataset["ground_truths"]["u"])):
                                        if self.dataset["ground_truths"]["u"][itm] != self.dataset["ground_truths"]["u"][itm-1]:
                                            ground_t_ihmm.append(t_n[itm] - time_prev)
                                            time_prev = t_n[itm]
                                    ground_t_ihmm = np.unique(np.abs(np.asarray(ground_t_ihmm)))
                                    ax1.vlines(ground_t_ihmm, 0 ,np.amax(n)+np.amax(n)*0.1 ,color=gtcolor, zorder=10)
                                    ax2.vlines(ground_t_ihmm, 0 ,np.amax(n)+np.amax(n)*0.1 ,color=gtcolor, zorder=10)
                                    handle_array.append(cyan_patch)
                                    num_cols += 1
                            handle_array.append(yellow_patch)
                            num_cols += 1
                    elif len(alt_results) == 2:
                        _, hist_dwells_kv = bnpa.generate_histogram_data_kv(self.alt_method_results_kv["means"], self.alt_method_results_kv["jump_times"])
                        _, hist_dwells_ihmm = bnpa.generate_histogram_data_ihmm(self.alt_method_results_ihmm["samples"], t_n)
                        # TODO: add option for bins=fixed_bins
                        n_kv, bins_kv, patches_kv = ax2.hist(hist_dwells_kv, alpha=0.5, edgecolor=alt_color1, color=alt_color1, density=True, histtype='stepfilled')
                        n_kv2, bins_kv2, patches_kv2 = ax3.hist(hist_dwells_ihmm, alpha=0.5, edgecolor=alt_color4, color=alt_color4, density=True, histtype='stepfilled')
                        bins_tot_h = np.concatenate((bins, bins_kv, bins_kv2))
                        n_tot_h = np.concatenate((n, n_kv, n_kv2))
                        # Configure axes
                        ax1.set_ylim(0,np.amax(n_tot_h)+np.amax(n_tot_h)*0.1)
                        ax2.set_ylim(0,np.amax(n_tot_h)+np.amax(n_tot_h)*0.1)
                        ax3.set_ylim(0,np.amax(n_tot_h)+np.amax(n_tot_h)*0.1)
                        ax1.set_yticks([0, np.amax(n_tot_h)/2, np.amax(n_tot_h)])
                        ax1.set_yticklabels(['0', f'{np.amax(n_tot_h)/2:.1f}', f'{np.amax(n_tot_h):.1f}'], fontsize=font_size)
                        ax2.set_yticks([0, np.amax(n_tot_h)/2, np.amax(n_tot_h)])
                        ax2.set_yticklabels(['', '', ''], fontsize=font_size)
                        ax3.set_yticks([0, np.amax(n_tot_h)/2, np.amax(n_tot_h)])
                        ax3.set_yticklabels(['', '', ''], fontsize=font_size)
                        ax1.set_xlim(int(np.amin(bins_tot_h)-2),int(np.amax(bins_tot_h)+2))
                        ax2.set_xlim(int(np.amin(bins_tot_h)-2),int(np.amax(bins_tot_h)+2))
                        ax3.set_xlim(int(np.amin(bins_tot_h)-2),int(np.amax(bins_tot_h)+2))
                        ax1.set_xticks([int(np.amin(bins_tot_h)), int((np.amin(bins_tot_h) + np.amax(bins_tot_h)) / 2), int(np.amax(bins_tot_h))])
                        ax1.set_xticklabels([str(int(np.amin(bins_tot_h))), str(int((np.amin(bins_tot_h) + np.amax(bins_tot_h)) / 2)), str(int(np.amax(bins_tot_h)))], fontsize=font_size)
                        ax2.set_xticks([int(np.amin(bins_tot_h)), int((np.amin(bins_tot_h) + np.amax(bins_tot_h)) / 2), int(np.amax(bins_tot_h))])
                        ax2.set_xticklabels([str(int(np.amin(bins_tot_h))), str(int((np.amin(bins_tot_h) + np.amax(bins_tot_h)) / 2)), str(int(np.amax(bins_tot_h)))], fontsize=font_size)
                        ax3.set_xticks([int(np.amin(bins_tot_h)), int((np.amin(bins_tot_h) + np.amax(bins_tot_h)) / 2), int(np.amax(bins_tot_h))])
                        ax3.set_xticklabels([str(int(np.amin(bins_tot_h))), str(int((np.amin(bins_tot_h) + np.amax(bins_tot_h)) / 2)), str(int(np.amax(bins_tot_h)))], fontsize=font_size)
                        if show_prior:
                            ax_1 = ax1.twinx()
                            ax_1.set_ylim(0,np.amax(n_tot_h)+np.amax(n_tot_h)*0.1)
                            ax_1.set_yticks([0, np.amax(n_tot_h)/2, np.amax(n_tot_h)])
                            ax_1.set_yticklabels(['','',''], fontsize=font_size)
                        # Plot ground truths if we have them
                        if (self.dataset["ground_truths"] is not None) and (self.dataset["parameters"] is not None):
                            if self.dataset["parameters"]["type"] == 'kv':
                                ground_h_trim = np.trim_zeros(self.dataset["ground_truths"]["h_m"])
                                # Process these so we get holding times, not absolute step times
                                ground_dwells =[]
                                for jmps in range(len(ground_h_trim)):
                                    if jmps == 0:
                                        continue
                                    else:
                                        ground_dwells.append(ground_h_trim[jmps] - ground_h_trim[jmps - 1])
                                ax1.vlines(ground_dwells, 0 ,np.amax(n_tot_h)+np.amax(n_tot_h)*0.1 ,color=gtcolor, zorder=10)
                                ax2.vlines(ground_dwells, 0 ,np.amax(n_tot_h)+np.amax(n_tot_h)*0.1 ,color=gtcolor, zorder=10)
                                ax3.vlines(ground_dwells, 0 ,np.amax(n_tot_h)+np.amax(n_tot_h)*0.1 ,color=gtcolor, zorder=10)
                                handle_array.append(cyan_patch)
                                num_cols += 1
                            elif self.dataset["parameters"]["type"] == 'hmm':
                                ground_t_ihmm = []
                                time_prev = t_n[0]
                                for itm in range(1, len(self.dataset["ground_truths"]["u"])):
                                    if self.dataset["ground_truths"]["u"][itm] != self.dataset["ground_truths"]["u"][itm-1]:
                                        ground_t_ihmm.append(t_n[itm] - time_prev)
                                        time_prev = t_n[itm]
                                ground_t_ihmm = np.unique(np.abs(np.asarray(ground_t_ihmm)))
                                ax1.vlines(ground_t_ihmm, 0 ,np.amax(n)+np.amax(n)*0.1 ,color=gtcolor, zorder=10)
                                ax2.vlines(ground_t_ihmm, 0 ,np.amax(n)+np.amax(n)*0.1 ,color=gtcolor, zorder=10)
                                ax3.vlines(ground_t_ihmm, 0 ,np.amax(n)+np.amax(n)*0.1 ,color=gtcolor, zorder=10)
                                handle_array.append(cyan_patch)
                                num_cols += 1
                        handle_array.append(purple_patch)
                        handle_array.append(yellow_patch)
                        num_cols += 2
                else:
                    ax1.set_ylim(0,np.amax(n)+np.amax(n)*0.1)
                    ax1.set_yticks([0, np.amax(n)/2, np.amax(n)])
                    ax1.set_yticklabels(['0', f'{np.amax(n)/2:.1f}', f'{np.amax(n):.1f}'], fontsize=font_size)
                    ax1.set_xlim(int(np.amin(bins)-2),int(np.amax(bins)+2))
                    ax1.set_xticks([int(np.amin(bins)), int((np.amin(bins) + np.amax(bins)) / 2), int(np.amax(bins))])
                    ax1.set_xticklabels([str(int(np.amin(bins))), str(int((np.amin(bins) + np.amax(bins)) / 2)), str(int(np.amax(bins)))], fontsize=font_size)
                    if show_prior:
                        ax_1 = ax1.twinx()
                        ax_1.set_ylim(0,np.amax(n)+np.amax(n)*0.1)
                        ax_1.set_yticks([0, np.amax(n)/2, np.amax(n)])
                        ax_1.set_yticklabels(['','',''], fontsize=font_size)
                    # Plot ground truths if we have them
                    if (self.dataset["ground_truths"] is not None) and (self.dataset["parameters"] is not None):
                        if self.dataset["parameters"]["type"] == 'kv':
                            ground_h_trim = np.trim_zeros(self.dataset["ground_truths"]["h_m"])
                            # Process these so we get holding times, not absolute step times
                            ground_dwells =[]
                            for jmps in range(len(ground_h_trim)):
                                if jmps == 0:
                                    continue
                                else:
                                    ground_dwells.append(ground_h_trim[jmps] - ground_h_trim[jmps - 1])
                            ax1.vlines(ground_dwells, 0 ,np.amax(n_tot_h)+np.amax(n_tot_h)*0.1 ,color=gtcolor, zorder=10)
                        elif self.dataset["parameters"]["type"] == 'hmm':
                            ground_t_ihmm = []
                            time_prev = t_n[0]
                            for itm in range(1, len(self.dataset["ground_truths"]["u"])):
                                if self.dataset["ground_truths"]["u"][itm] != self.dataset["ground_truths"]["u"][itm-1]:
                                    ground_t_ihmm.append(t_n[itm] - time_prev)
                                    time_prev = t_n[itm]
                            ground_t_ihmm = np.unique(np.abs(np.asarray(ground_t_ihmm)))
                            ax1.vlines(ground_t_ihmm, 0 ,np.amax(n)+np.amax(n)*0.1 ,color=gtcolor, zorder=10)
                            handle_array.append(cyan_patch)
                            num_cols += 1

                # Plot prior if we chose this option
                if show_prior:
                    prior_x = np.linspace(30, t_n[-1], 200)
                    prior_t = (1/(len(t_n)))*np.ones(200)
                    ax_1.plot(prior_x, prior_t, label='Prior', color=alt_color2, zorder=5)
                    handle_array.append(magenta_patch)
                    num_cols += 1
                
                # Plot CI if we chose this option
                # TODO: add options to show 50% CI or have user-defined CI
                if show_ci:
                    y_limits_ci = [0, 100]
                    mean, under95, under50, median, upper50, upper95 = bnpa.get_credible_intervals(hist_dwells)
                    ax1.fill_betweenx(y=y_limits_ci, x1=upper95, x2=under95, color=alt_color3, alpha=0.2, zorder=-10)
                    handle_array.append(red_patch)
                    num_cols += 1

                # Generate legend
                ax0.legend(bbox_to_anchor=(0., 1.08, 1., .102), handles=handle_array,
                                loc='lower center', ncol=num_cols, mode="none", borderaxespad=0., borderpad=0.8, edgecolor=(1.0, 1.0, 1.0, 0.0), prop=fnt_mgr)
                            
                # Show plot, then save figure
                plt.show()
                # TODO: add user option to generate figure type other than pdf
                output_filename = fig_savename + '_' + plot + '.pdf'
                fig.savefig(output_filename, format='pdf')

            elif (plot == 'hist_emission'):
                # General figure setup
                fig = plt.figure()

                if plot_alt_results and len(alt_results) == 1:
                    gs1 = GridSpec(1, 1, bottom=0.8)  # For legend
                    ax0 = fig.add_subplot(gs1[0])
                    ax0.axis('off')
                    gs2 = GridSpec(1, 2, top=0.85, wspace=0.07, hspace=0.05)  # For actual plot
                    ax1 = fig.add_subplot(gs2[0])
                    ax2 = fig.add_subplot(gs2[1])
                elif plot_alt_results and len(alt_results) == 2:
                    gs1 = GridSpec(1, 1, bottom=0.8)  # For legend
                    ax0 = fig.add_subplot(gs1[0])
                    ax0.axis('off')
                    gs2 = GridSpec(1, 3, top=0.85, wspace=0.07, hspace=0.05)  # For actual plot
                    ax1 = fig.add_subplot(gs2[0])
                    ax2 = fig.add_subplot(gs2[1])
                    ax3 = fig.add_subplot(gs2[2])
                else:
                    gs1 = GridSpec(1, 1, bottom=0.8)  # For legend
                    ax0 = fig.add_subplot(gs1[0])
                    ax0.axis('off')
                    gs2 = GridSpec(1, 1, top=0.85, wspace=0.07, hspace=0.05)  # For actual plot
                    ax1 = fig.add_subplot(gs2[0])

                fig.supxlabel(x_label, fontsize=font_size)
                fig.supylabel(y_label, x=0.05, fontsize=font_size)

                # Legend setup
                orange_patch = mpatches.Patch(color=learncolor, alpha=0.5, label='BNP-Step')
                purple_patch = mpatches.Patch(color=alt_color1, alpha=0.5, label='BIC')
                yellow_patch = mpatches.Patch(color=alt_color4, alpha=0.5, label='iHMM')
                cyan_patch = mpatches.Patch(color=gtcolor, label='Ground truths')
                magenta_patch = mpatches.Patch(color=alt_color2, label='Prior')

                # Prepare fixed bins - NYI, should be an option users choose
                #fixed_bins = np.arange(-10,20,0.05)

                # Strip out samples with the MAP number of steps, then generate the histogrammable data set
                b_m_top, h_m_top, t_m_top, f_back_top, eta_top = bnpa.find_top_samples_by_jumps(b_clean, h_clean, t_clean, f_clean, eta_clean, post_clean)
                _, sample_data = bnpa.generate_histogram_data_emission(b_m_top, h_m_top, t_m_top, f_back_top, self.B_max)
                
                # Ravel the dataset for histogramming
                hist_data = np.ravel(sample_data)

                # Plot histogram
                # TODO: add bins=fixed_bins option
                n, bins, patches = ax1.hist(hist_data, alpha=0.5, edgecolor=learncolor, density=True, color=learncolor, histtype='stepfilled')
                
                # Legend parameters
                handle_array = [orange_patch]
                num_cols = 1

                # Configure axes and plot alternative methods' results if we have them
                if plot_alt_results:
                    if len(alt_results) == 1:
                        if alt_results[0] == 'ihmm':
                            # TODO: add option for bins=fixed_bins
                            n_kv, bins_kv, patches_kv = ax2.hist(self.alt_method_results_ihmm["mode_means"], alpha=0.5, edgecolor=alt_color4, color=alt_color4, density=True, histtype='stepfilled')
                            bins_tot_h = np.concatenate((bins, bins_kv))
                            n_tot_h = np.concatenate((n, n_kv))
                            # Configure axes
                            ax1.set_ylim(0,np.amax(n_tot_h)+np.amax(n_tot_h)*0.1)
                            ax2.set_ylim(0,np.amax(n_tot_h)+np.amax(n_tot_h)*0.1)
                            ax1.set_yticks([0, np.amax(n_tot_h)/2, np.amax(n_tot_h)])
                            ax1.set_yticklabels(['0', f'{np.amax(n_tot_h)/2:.1f}', f'{np.amax(n_tot_h):.1f}'], fontsize=font_size)
                            ax2.set_yticks([0, np.amax(n_tot_h)/2, np.amax(n_tot_h)])
                            ax2.set_yticklabels(['', '', ''], fontsize=font_size)
                            ax1.set_xlim(int(np.amin(bins_tot_h)-2),int(np.amax(bins_tot_h)+2))
                            ax2.set_xlim(int(np.amin(bins_tot_h)-2),int(np.amax(bins_tot_h)+2))
                            ax1.set_xticks([int(np.amin(bins_tot_h)), int((np.amin(bins_tot_h) + np.amax(bins_tot_h)) / 2), int(np.amax(bins_tot_h))])
                            ax1.set_xticklabels([str(int(np.amin(bins_tot_h))), str(int((np.amin(bins_tot_h) + np.amax(bins_tot_h)) / 2)), str(int(np.amax(bins_tot_h)))], fontsize=font_size)
                            ax2.set_xticks([int(np.amin(bins_tot_h)), int((np.amin(bins_tot_h) + np.amax(bins_tot_h)) / 2), int(np.amax(bins_tot_h))])
                            ax2.set_xticklabels([str(int(np.amin(bins_tot_h))), str(int((np.amin(bins_tot_h) + np.amax(bins_tot_h)) / 2)), str(int(np.amax(bins_tot_h)))], fontsize=font_size)
                            if show_prior:
                                ax_1 = ax1.twinx()
                                ax_1.set_ylim(0,np.amax(n_tot_h)+np.amax(n_tot_h)*0.1)
                                ax_1.set_yticks([0, np.amax(n_tot_h)/2, np.amax(n_tot_h)])
                                ax_1.set_yticklabels(['','',''], fontsize=font_size)
                            # Plot ground truths if we have them
                            if (self.dataset["ground_truths"] is not None) and (self.dataset["parameters"] is not None):
                                if self.dataset["parameters"]["type"] == 'hmm':
                                    gt_emissions_ihmm = np.unique(self.dataset["ground_truths"]["u"])
                                    ax1.vlines(gt_emissions_ihmm, 0, 100, color=gtcolor, zorder=10)
                                    ax2.vlines(gt_emissions_ihmm, 0, 100, color=gtcolor, zorder=10)
                                    handle_array.append(cyan_patch)
                                    num_cols += 1
                                elif self.dataset["parameters"]["type"] == 'kv':
                                    _, ground_data = bnpa.generate_gt_step_plot_data(self.dataset["ground_truths"]["b_m"], self.dataset["ground_truths"]["h_m"], self.dataset["ground_truths"]["t_m"], self.dataset["parameters"]["f_back"], t_n, self.B_max)
                                    ground_data = ground_data - self.dataset["parameters"]["f_back"]
                                    ax1.vlines(ground_data, 0, 100, color=gtcolor, zorder=10)
                                    ax2.vlines(ground_data, 0, 100, color=gtcolor, zorder=10)
                                    handle_array.append(cyan_patch)
                                    num_cols += 1
                            handle_array.append(yellow_patch)
                            num_cols += 1
                        elif alt_results[0] == 'kv':
                            # TODO: add option for bins=fixed_bins
                            _, kv_emissions = bnpa.generate_kv_step_plot_data(self.alt_method_results_kv["jump_times"], 
                                                                                    self.alt_method_results_kv["means"], 
                                                                                    self.alt_method_results_kv["background"], 
                                                                                    t_n)
                            n_kv, bins_kv, patches_kv = ax2.hist(kv_emissions, alpha=0.5, edgecolor=alt_color1, color=alt_color1, density=True, histtype='stepfilled')
                            bins_tot_h = np.concatenate((bins, bins_kv))
                            n_tot_h = np.concatenate((n, n_kv))
                            # Configure axes
                            ax1.set_ylim(0,np.amax(n_tot_h)+np.amax(n_tot_h)*0.1)
                            ax2.set_ylim(0,np.amax(n_tot_h)+np.amax(n_tot_h)*0.1)
                            ax1.set_yticks([0, np.amax(n_tot_h)/2, np.amax(n_tot_h)])
                            ax1.set_yticklabels(['0', f'{np.amax(n_tot_h)/2:.1f}', f'{np.amax(n_tot_h):.1f}'], fontsize=font_size)
                            ax2.set_yticks([0, np.amax(n_tot_h)/2, np.amax(n_tot_h)])
                            ax2.set_yticklabels(['', '', ''], fontsize=font_size)
                            ax1.set_xlim(int(np.amin(bins_tot_h)-2),int(np.amax(bins_tot_h)+2))
                            ax2.set_xlim(int(np.amin(bins_tot_h)-2),int(np.amax(bins_tot_h)+2))
                            ax1.set_xticks([int(np.amin(bins_tot_h)), int((np.amin(bins_tot_h) + np.amax(bins_tot_h)) / 2), int(np.amax(bins_tot_h))])
                            ax1.set_xticklabels([str(int(np.amin(bins_tot_h))), str(int((np.amin(bins_tot_h) + np.amax(bins_tot_h)) / 2)), str(int(np.amax(bins_tot_h)))], fontsize=font_size)
                            ax2.set_xticks([int(np.amin(bins_tot_h)), int((np.amin(bins_tot_h) + np.amax(bins_tot_h)) / 2), int(np.amax(bins_tot_h))])
                            ax2.set_xticklabels([str(int(np.amin(bins_tot_h))), str(int((np.amin(bins_tot_h) + np.amax(bins_tot_h)) / 2)), str(int(np.amax(bins_tot_h)))], fontsize=font_size)
                            if show_prior:
                                ax_1 = ax1.twinx()
                                ax_1.set_ylim(0,np.amax(n_tot_h)+np.amax(n_tot_h)*0.1)
                                ax_1.set_yticks([0, np.amax(n_tot_h)/2, np.amax(n_tot_h)])
                                ax_1.set_yticklabels(['','',''], fontsize=font_size)
                            # Plot ground truths if we have them
                            if (self.dataset["ground_truths"] is not None) and (self.dataset["parameters"] is not None):
                                if self.dataset["parameters"]["type"] == 'hmm':
                                    gt_emissions_ihmm = np.unique(self.dataset["ground_truths"]["u"])
                                    ax1.vlines(gt_emissions_ihmm, 0, 100, color=gtcolor, zorder=10)
                                    ax2.vlines(gt_emissions_ihmm, 0, 100, color=gtcolor, zorder=10)
                                    handle_array.append(cyan_patch)
                                    num_cols += 1
                                elif self.dataset["parameters"]["type"] == 'kv':
                                    _, ground_data = bnpa.generate_gt_step_plot_data(self.dataset["ground_truths"]["b_m"], self.dataset["ground_truths"]["h_m"], self.dataset["ground_truths"]["t_m"], self.dataset["parameters"]["f_back"], t_n, self.B_max)
                                    ground_data = ground_data - self.dataset["parameters"]["f_back"]
                                    ax1.vlines(ground_data, 0, 100, color=gtcolor, zorder=10)
                                    ax2.vlines(ground_data, 0, 100, color=gtcolor, zorder=10)
                                    handle_array.append(cyan_patch)
                                    num_cols += 1
                            handle_array.append(purple_patch)
                            num_cols += 1
                    elif len(alt_results) == 2:
                        # TODO: add option for bins=fixed_bins
                        _, kv_emissions = bnpa.generate_kv_step_plot_data(self.alt_method_results_kv["jump_times"], 
                                                                                    self.alt_method_results_kv["means"], 
                                                                                    self.alt_method_results_kv["background"], 
                                                                                    t_n)
                        n_kv, bins_kv, patches_kv = ax3.hist(self.alt_method_results_ihmm["mode_means"], alpha=0.5, edgecolor=alt_color4, color=alt_color4, density=True, histtype='stepfilled')
                        n_kv2, bins_kv2, patches_kv2 = ax2.hist(kv_emissions, alpha=0.5, edgecolor=alt_color1, color=alt_color1, density=True, histtype='stepfilled')
                        bins_tot_h = np.concatenate((bins, bins_kv, bins_kv2))
                        n_tot_h = np.concatenate((n, n_kv, n_kv2))
                        # Configure axes
                        ax1.set_ylim(0,np.amax(n_tot_h)+np.amax(n_tot_h)*0.1)
                        ax2.set_ylim(0,np.amax(n_tot_h)+np.amax(n_tot_h)*0.1)
                        ax3.set_ylim(0,np.amax(n_tot_h)+np.amax(n_tot_h)*0.1)
                        ax1.set_yticks([0, np.amax(n_tot_h)/2, np.amax(n_tot_h)])
                        ax1.set_yticklabels(['0', f'{np.amax(n_tot_h)/2:.1f}', f'{np.amax(n_tot_h):.1f}'], fontsize=font_size)
                        ax2.set_yticks([0, np.amax(n_tot_h)/2, np.amax(n_tot_h)])
                        ax2.set_yticklabels(['', '', ''], fontsize=font_size)
                        ax3.set_yticks([0, np.amax(n_tot_h)/2, np.amax(n_tot_h)])
                        ax3.set_yticklabels(['', '', ''], fontsize=font_size)
                        ax1.set_xlim(int(np.amin(bins_tot_h)-2),int(np.amax(bins_tot_h)+2))
                        ax2.set_xlim(int(np.amin(bins_tot_h)-2),int(np.amax(bins_tot_h)+2))
                        ax3.set_xlim(int(np.amin(bins_tot_h)-2),int(np.amax(bins_tot_h)+2))
                        ax1.set_xticks([int(np.amin(bins_tot_h)), int((np.amin(bins_tot_h) + np.amax(bins_tot_h)) / 2), int(np.amax(bins_tot_h))])
                        ax1.set_xticklabels([str(int(np.amin(bins_tot_h))), str(int((np.amin(bins_tot_h) + np.amax(bins_tot_h)) / 2)), str(int(np.amax(bins_tot_h)))], fontsize=font_size)
                        ax2.set_xticks([int(np.amin(bins_tot_h)), int((np.amin(bins_tot_h) + np.amax(bins_tot_h)) / 2), int(np.amax(bins_tot_h))])
                        ax2.set_xticklabels([str(int(np.amin(bins_tot_h))), str(int((np.amin(bins_tot_h) + np.amax(bins_tot_h)) / 2)), str(int(np.amax(bins_tot_h)))], fontsize=font_size)
                        ax3.set_xticks([int(np.amin(bins_tot_h)), int((np.amin(bins_tot_h) + np.amax(bins_tot_h)) / 2), int(np.amax(bins_tot_h))])
                        ax3.set_xticklabels([str(int(np.amin(bins_tot_h))), str(int((np.amin(bins_tot_h) + np.amax(bins_tot_h)) / 2)), str(int(np.amax(bins_tot_h)))], fontsize=font_size)
                        if show_prior:
                            ax_1 = ax1.twinx()
                            ax_1.set_ylim(0,np.amax(n_tot_h)+np.amax(n_tot_h)*0.1)
                            ax_1.set_yticks([0, np.amax(n_tot_h)/2, np.amax(n_tot_h)])
                            ax_1.set_yticklabels(['','',''], fontsize=font_size)
                        # Plot ground truths if we have them
                        if (self.dataset["ground_truths"] is not None) and (self.dataset["parameters"] is not None):
                            if self.dataset["parameters"]["type"] == 'hmm':
                                gt_emissions_ihmm = np.unique(self.dataset["ground_truths"]["u"])
                                ax1.vlines(gt_emissions_ihmm, 0, 100, color=gtcolor, zorder=10)
                                ax2.vlines(gt_emissions_ihmm, 0, 100, color=gtcolor, zorder=10)
                                ax3.vlines(gt_emissions_ihmm, 0, 100, color=gtcolor, zorder=10)
                                handle_array.append(cyan_patch)
                                num_cols += 1
                            elif self.dataset["parameters"]["type"] == 'kv':
                                _, ground_data = bnpa.generate_gt_step_plot_data(self.dataset["ground_truths"]["b_m"], self.dataset["ground_truths"]["h_m"], self.dataset["ground_truths"]["t_m"], self.dataset["parameters"]["f_back"], t_n, self.B_max)
                                ground_data = ground_data - self.dataset["parameters"]["f_back"]
                                ax1.vlines(ground_data, 0, 100, color=gtcolor, zorder=10)
                                ax2.vlines(ground_data, 0, 100, color=gtcolor, zorder=10)
                                ax3.vlines(ground_data, 0, 100, color=gtcolor, zorder=10)
                                handle_array.append(cyan_patch)
                                num_cols += 1
                        handle_array.append(purple_patch)
                        handle_array.append(yellow_patch)
                        num_cols += 2
                else:
                    ax1.set_ylim(0,np.amax(n)+np.amax(n)*0.1)
                    ax1.set_yticks([0, np.amax(n)/2, np.amax(n)])
                    ax1.set_yticklabels(['0', f'{np.amax(n)/2:.1f}', f'{np.amax(n):.1f}'], fontsize=font_size)
                    ax1.set_xlim(int(np.amin(bins)-2),int(np.amax(bins)+2))
                    ax1.set_xticks([int(np.amin(bins)), int((np.amin(bins) + np.amax(bins)) / 2), int(np.amax(bins))])
                    ax1.set_xticklabels([str(int(np.amin(bins))), str(int((np.amin(bins) + np.amax(bins)) / 2)), str(int(np.amax(bins)))], fontsize=font_size)
                    if show_prior:
                        ax_1 = ax1.twinx()
                        ax_1.set_ylim(0,np.amax(n)+np.amax(n)*0.1)
                        ax_1.set_yticks([0, np.amax(n)/2, np.amax(n)])
                        ax_1.set_yticklabels(['','',''], fontsize=font_size)
                    # Plot ground truths if we have them
                    if (self.dataset["ground_truths"] is not None) and (self.dataset["parameters"] is not None):
                        if self.dataset["parameters"]["type"] == 'hmm':
                            gt_emissions_ihmm = np.unique(self.dataset["ground_truths"]["u"])
                            ax1.vlines(gt_emissions_ihmm, 0, 100, color=gtcolor, zorder=10)
                            handle_array.append(cyan_patch)
                            num_cols += 1
                        elif self.dataset["parameters"]["type"] == 'kv':
                            _, ground_data = bnpa.generate_gt_step_plot_data(self.dataset["ground_truths"]["b_m"], self.dataset["ground_truths"]["h_m"], self.dataset["ground_truths"]["t_m"], self.dataset["parameters"]["f_back"], t_n, self.B_max)
                            ground_data = ground_data - self.dataset["parameters"]["f_back"]
                            ax1.vlines(ground_data, 0, 100, color=gtcolor, zorder=10)
                            handle_array.append(cyan_patch)
                            num_cols += 1
                
                # Plot prior if we chose this option
                if show_prior:
                    prior_x = np.linspace(self.h_ref-2*(1/np.sqrt(self.chi)), self.h_ref+2*(1/np.sqrt(self.chi)), 200)
                    prior_h = np.sqrt(self.chi/(2*np.pi))*np.exp(-(self.chi/2)*(prior_x-self.h_ref)**2)
                    ax_1.plot(prior_x, prior_h, label='Prior', color=alt_color2, zorder=5, linewidth=3.0)
                    handle_array.append(magenta_patch)
                    num_cols += 1

                # Add legend
                ax0.legend(bbox_to_anchor=(0., 1.08, 1., .102), handles=handle_array,
                            loc='lower center', ncol=num_cols, mode="none", borderaxespad=0., borderpad=0.8, edgecolor=(1.0, 1.0, 1.0, 0.0), prop=fnt_mgr)
            
                # Show plot, then save figure
                plt.show()
                # TODO: add user option to generate figure type other than pdf
                output_filename = fig_savename + '_' + plot + '.pdf'
                fig.savefig(output_filename, format='pdf')

            elif (plot == 'hist_height_separated'):
                warnings.warn(f"Separated emission level histograms NYI", UserWarning)
            
            elif (plot == 'hist_dwell_separated'):
                warnings.warn(f"Separated emission level histograms NYI", UserWarning)
            
            elif (plot == 'hist_emission_separated'):
                warnings.warn(f"Separated emission level histograms NYI", UserWarning)

            elif (plot == 'survivorship'):
                warnings.warn(f"Survivorship plots NYI", UserWarning)

            elif (plot == 'hist_f'):
                warnings.warn(f"F_bg histogram NYI", UserWarning)

            elif (plot == 'hist_eta'):
                warnings.warn(f"eta histogram NYI", UserWarning)
