import numpy as np


def hyper_prior(dataset, mu, sigma):
    return np.exp(- (dataset['zeta'] - mu)**2 / (2 * sigma**2)) /\
        (2 * np.pi * sigma**2)**0.5
