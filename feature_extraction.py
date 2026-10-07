"""Step 3: Feature extraction from the (trial, frequency, time) arrays."""
import numpy as np


def maxim(x):
    return np.max(x, axis=2)[:, 0]


def minim(x):
    return np.min(x, axis=2)[:, 0]


def std(x):
    return np.std(x, axis=2)[:, 0]


def extract_positive_area_voltage(data):
    """Sum of positive values over time, per trial.
    NOTE: negative values are set to 0 IN PLACE (modifies `data`)."""
    data[data < 0] = 0
    return np.sum(data, axis=2)[:, 0]


def calculate_mean_psd_amplitude(data):
    """Mean over the frequency axis, first time point, per trial."""
    return np.mean(data, axis=1)[:, 0]


def con_features(x):
    """Stack the three features: positive area, mean PSD amplitude, max."""
    return np.vstack((extract_positive_area_voltage(x),
                      calculate_mean_psd_amplitude(x),
                      maxim(x)))
