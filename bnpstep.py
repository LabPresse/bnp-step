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
import pickle
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
                 phi: float = 1.0, 
                 eta_ref: float = 10.0, 
                 gamma: float = 1.0, 
                 B_max: int = 50, 
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

        # Dictionary to store results from alternative methods - this should not be used 
        # unless comparing to one of the other methods mentioned in the paper (iHMM or KV)
        self.alt_method_results = {}


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
        Analyzes a dataset using BNP-Step and prints the resulting samples to a .pkl file. Samples are also
        stored in object attributes.

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

        
    def results_to_file(self,
                        outfile: str = 'output',
                        path = None
                        ):
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
        full_name = outfile + '.pkl'
        if path is not None:
            full_path = os.path.join(path, full_name)
        else:
            full_path = full_name
        
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


    def load_alt_method_results():
        """
        Loads results from iHMM or KV method into self.alt_method_results for comparison plotting
        """
        pass

    
    def plot_data(self):
        # Plot data set using same format as visualization - NYI
        pass


    def visualize_results(self,
                          plot_type : Union[str, List[str]] = 'step',
                          font_size : int = 16,
                          datacolor : str = '#929591',
                          learncolor : str = '#f97306',
                          gtcolor : str = '#00ffff',
                          x_label : str =  'x-values',
                          y_label : str =  'y-values',
                          plot_alt_results : bool = False,
                          alt_results : str = ''
                          ):
        # TODO: IN PROGRESS
        # TODO: Better input validation for dataset dict
        """
        Draws plots of BNP-Step results with the dataset (and ground truths if present).
        """
        # Plot data set
        # Plot ground truth, if available
        # Plot inference results from BNP Step
        # Do this for trajectory and/or histograms, depending on options selected

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
        if not isinstance(x_label, str):
            raise TypeError(f"x_label should be of type str instead of {type(x_label)}")
        if not isinstance(y_label, str):
            raise TypeError(f"y_label should be of type str instead of {type(y_label)}")
        if not isinstance(plot_alt_results, bool):
            raise TypeError(f"plot_alt_results should be of type bool instead of {type(plot_alt_results)}")
        if not isinstance(alt_results, str):
            raise TypeError(f"alt_results should be of type str instead of {type(alt_results)}")
        
        # Warn the user if invalid alt_results type was chosen
        if plot_alt_results and alt_results != 'kv' and alt_results != 'ihmm':
            warnings.warn(f"Valid values for alt_results are 'kv' and 'ihmm', got {alt_results}. No alternative method results will be shown.", UserWarning)
            plot_alt_results = False
        
        # Ensure we have at least one valid plot type in plot_type list; otherwise warn the user and default to step plot
        has_valid_plot_type = False
        for plot in plot_type:
            if plot == 'step':
                has_valid_plot_type = True
        
        if not has_valid_plot_type:
            warnings.warn("No valid plot types were passed in plot_type parameter; reverting to default ('step')", UserWarning)
            plot_type = ['step']
        
        # Get timepoints if we have them, otherwise pass a generic arange numpy array based on number of observations
        if self.dataset["times"] is not None:
            t_n = self.dataset["times"]
        else:
            t_n = np.arange(len(self.dataset["data"]))
        
        # General figure setup
        fig = plt.figure()
        fnt_mgr = mpl.font_manager.FontProperties(size=font_size)

        if plot_alt_results:
            gs1 = GridSpec(1, 1, bottom=0.8)  # For legend
            ax0 = fig.add_subplot(gs1[0])
            ax0.axis('off')
            gs2 = GridSpec(1, 2, top=0.85, wspace=0.07, hspace=0.05)  # For actual plot
            ax1 = fig.add_subplot(gs2[0])
            ax2 = fig.add_subplot(gs2[1])
        else:
            gs1 = GridSpec(1, 1, bottom=0.8)  # For legend
            ax0 = fig.add_subplot(gs1[0])
            ax0.axis('off')
            gs2 = GridSpec(1, 1, top=0.85, wspace=0.07, hspace=0.05)  # For actual plot
            ax1 = fig.add_subplot(gs2[0])

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

        for plot in plot_type:
            if (plot == 'step'):
                # Setup x-axis vectors for plotting
                T = np.linspace(t_n[0], t_n[-1], len(self.dataset["data"]))
                
                # Find the MAP estimate for our step plot
                # TODO: right now this assumes we already removed burn-in, change this later
                b_map, h_map, t_map, f_map, eta_map = bnpa.find_map(self.B_M, self.H_M, self.T_M, self.F_S, self.ETA, self.post)

                # Generate step plot data from our results
                sorted_times, sorted_data = bnpa.generate_step_plot_data(b_map, h_map, t_map, f_map, self.B_max, t_n)

                # Plot synthetic data
                ax1.scatter(T, self.dataset["data"], alpha=0.7, color=datacolor, facecolors='none', marker='.')

                # If we have ground truths, plot them
                if (self.dataset["ground_truths"] is not None) and (self.dataset["parameters"] is not None):
                    if self.dataset["parameters"]["type"] == 'kv':
                        ground_times, ground_data = bnpa.generate_gt_step_plot_data(self.dataset["ground_truths"]["b_m"], self.dataset["ground_truths"]["h_m"], self.dataset["ground_truths"]["t_m"], self.dataset["parameters"]["f_back"], t_n, self.B_max)
                        ax1.stairs(ground_data, ground_times, baseline=None, color=gtcolor, linewidth=3.0)

                # Plot discovered steps
                ax1.stairs(sorted_data, sorted_times, baseline=None, color=learncolor, linewidth=3.0)

                # Add subplot titles
                # ax1.set_title('SNR = 2.0', font=fpath, fontsize=fntsize)

                # Generate legend
                if (self.dataset["ground_truths"] is not None) and (self.dataset["parameters"] is not None):
                    ax0.legend([Line2D([0], [0], color=gtcolor,lw=3.0),
                                    Line2D([0], [0], color=learncolor,lw=3.0),c],
                                ['Ground truth', 'Learned trajectory', 'Data'], handler_map={mpatches.Circle: HandlerCircle()},
                                bbox_to_anchor=(0., 1.15, 1., .102), loc='lower center',
                                        ncol=3, mode="none", borderaxespad=0., edgecolor=(1.0, 1.0, 1.0, 0.0), prop=fnt_mgr, borderpad=0.8)
                else:
                    ax0.legend([Line2D([0], [0], color=learncolor,lw=3.0),c],
                            ['Learned trajectory', 'Data'], handler_map={mpatches.Circle: HandlerCircle()},
                            bbox_to_anchor=(0., 1.15, 1., .102), loc='lower center',
                                    ncol=2, mode="none", borderaxespad=0., edgecolor=(1.0, 1.0, 1.0, 0.0), prop=fnt_mgr, borderpad=0.8)
                
                # Configure axes
                ax1.set_xticks([t_n[0], t_n[int(len(self.dataset["data"])/2)], t_n[-1]])
                ax1.set_xticklabels([str((t_n[0])), str(int((t_n[int(len(self.dataset["data"])/2)]))), str((t_n[-1]))], fontsize=font_size)

                ax1.set_yticks([int(np.amin(sorted_data)), int((np.amax(sorted_data)+np.amin(sorted_data))/2), int(np.amax(sorted_data))])
                ax1.set_yticklabels([str(int(np.amin(sorted_data))), str(int((np.amax(sorted_data)+np.amin(sorted_data))/2)), str(int(np.amax(sorted_data)))], fontsize=font_size)

                # If we also have alternative results, plot those and configure the axes.
                if plot_alt_results:
                    # TODO: add actual plotting of alternative method's results
                    ax2.scatter(T, self.dataset["data"], alpha=0.7, color=datacolor, facecolors='none', marker='.')
                    
                    # Put alternative method plotting here
                    warnings.warn("Plotting of alternative method results is NYI", UserWarning)

                    # Configure axes
                    ax2.set_xticks([t_n[0], t_n[int(len(self.dataset["data"])/2)], t_n[-1]])
                    ax2.set_xticklabels([str(int(t_n[0])), str(int((t_n[int(len(self.dataset["data"])/2)]))), str(int(t_n[-1]))], fontsize=font_size)
                    ax2.set_yticks([int(np.amin(sorted_data)), int((np.amax(sorted_data)+np.amin(sorted_data))/2), int(np.amax(sorted_data))])
                    ax2.set_yticklabels(['', '', ''], fontsize=font_size)

                # Show plot, then save figure
                warnings.warn("Automatic saving of figures NYI", UserWarning)
                plt.show()
                #fig.savefig('stepPlot.pdf', format='pdf')
