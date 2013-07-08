#! /usr/bin/env python

"""Survival data for ovarian cancer"""

__docformat__ = 'restructuredtext'

COPYRIGHT   = """This is public domain."""
TITLE       = ""
SOURCE      = """
This data is taken from Konstantinopoulos PA, Cannistra SA, Fountzilas H,
Culhane A, Pillay K, et al. (2011) [1] and downloaded from [2]

[1] Konstantinopoulos PA, Cannistra SA, Fountzilas H, Culhane A, Pillay K,
    et al. 2011 Integrated Analysis of Multiple Microarray Datasets Identifies
    a Reproducible Survival Predictor in Ovarian Cancer. PLoS ONE 6(3): e18202.
    doi:10.1371/journal.pone.0018202

[2] ncbi.nlm.nih.gov/geo/query/acc.cgi?token=bdgzfwmysouamxe&acc=GSE19161
"""

DESCRSHORT  = """Survival data for ovarian cancer"""

DESCRLONG   = """Survival data for ovarian cancer with gene expression
covariates, compiled from four seperate studies."""

NOTE        = """
Number of Observations: 239
Number of Variables: 660
Variable name definitions:
    time      - survival time (in months)
    event - 0 if observation is censored, 1 if observation is
                an event

    The remaining variables are gene expression values identified by
    their names. For details see [1] above.
"""

import numpy as np
from statsmodels.datasets import utils as du
from statsmodels.sandbox.survival2 import Survival
from os.path import dirname, abspath


def load():
    """
    Load the survival and gene expression data

    Returns
    -------
    Dataset instance:
        See DATASET_PROPOSAL.txt for more information.

    """
    data = _get_data()
    new_data = du.process_recarray(data, exog_idx=[2,3,4,5], dtype=float)
    endog = Survival(time1=data['time'],
                     event=data['event'])
    new_data.endog = endog
    return new_data

def load_pandas():
    data = _get_data()
    new_data = du.process_recarray_pandas(data, exog_idx=[2,3,4,5], dtype=float)
    endog = Survival(time1=data['time'],
                     event=data['event'])
    new_data.endog = endog
    return new_data


def _get_data():
    filepath = dirname(abspath(__file__))
    data = data = np.recfromtxt(open(filepath+"/ovarian_cancer_data.csv",
                                     "rb"), names=True, dtype=float)
    return data
