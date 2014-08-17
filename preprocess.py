import pandas as pd
import numpy as np


def readData(filename, idx_col = None):
    """returns a pandas data frame"""
    if idx_col == None:
        data = pd.read_csv(filename)
    else:
        data = pd.read_csv(filename, index_col = idx_col)
    return data

def priors(data, colname):
    """returns a pandas series containing the prior for each output class"""
    priors = data.groupby(colname).count()[colname]
    total = sum(priors)
    priors = priors.to_dict()
    for key in priors.keys():
        priors[key] = np.float64(priors[key])/total
    return priors
