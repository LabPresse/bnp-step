# bnp-step

:construction: This page is actively under construction! Check back often for updates. :construction:

This repository contains the Python scripts and helper functions for BNP-Step, a computational method described in [An accurate probabilistic step finder for time-series analysis, bioRxiv 2023](https://www.biorxiv.org/content/10.1101/2023.09.19.558535v1).

## Installation

BNP-Step was developed in Python 3.11, and uses numpy 1.24.3, scipy 1.10.1, matplotlib 3.7.1, and pandas 1.5.3. For Anaconda users, we've included stepfind.yml which can be used to recreate our development environment:

```
conda env create -f stepfind.yml
```

Alternatively, you can install the required packages on your own using pip. However, we caution that BNP-Step has not currently been tested using versions of Python other than 3.11 or versions of numpy, scipy, matplotlib, and pandas other than those specified.

Once your environment has been set up, clone the bnp-step repository by running the following from the command line in the directory of your choice:

```
git clone https://github.com/arojewski/bnp-step.git
```

This will create a folder in the current directory containing the BNP-Step code and two Jupyter notebooks. That's it! You're now ready to use BNP-Step.

## Usage

Once your environment is set up, take a look at the tutorial file, BNPStepTutorial.ipynb, to help you get started. Alternatively, if you're ready to jump right in, you can use RunBNPStep.ipynb. 

In the near future, we plan to add an option for running BNP-Step using a simple GUI.

## Comparisons to other methods

If you would like to compare results from BNP-Step to a BIC-based method, we have included an implementation of the BIC-based method we compare to in our paper. In the near future, we plan to also include utilities to import results from the iHMM we compare to.

## Questions? Contact us!

BNP-Step is a work in progress. Further documentation will be provided as it is created. If you require assistance or would like more details, please do not hesitate to contact us at arojewsk@asu.edu or spresse@asu.edu
