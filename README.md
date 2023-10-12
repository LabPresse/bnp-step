# bnp-step

:construction: This page is actively under construction! Check back often for updates. :construction:

This repository contains the Python scripts and helper functions for BNP-Step, a computational method described in [An accurate probabilistic step finder for time-series analysis, bioRxiv 2023](https://www.biorxiv.org/content/10.1101/2023.09.19.558535v1).

## Usage

BNP-Step was developed in Python 3.11 using numpy 1.24.3, scipy 1.10.1, matplotlib 3.7.1, and pandas 1.5.3. Conda users can replicate the development environment by creating a new environment using the included stepfind.yml file:

```python
conda env create -f stepfind.yml
```

Once your environment is set up, you can run BNP-Step through the command line, or through the included Jupyter notebook, RunBNPStep.ipynb.

In the near future, we will add options for running BNP-Step using a GUI interface.

## Comparisons to other methods

If you would like to compare results from BNP-Step to a BIC-based method or to an iHMM, we plan to include several functions and utilities to facilitate doing so.

## Questions? Contact us!

BNP-Step is a work in progress. Further documentation will be provided as it is created. If you require assistance or would like more details, please do not hesitate to contact us at arojewsk@asu.edu or spresse@asu.edu
