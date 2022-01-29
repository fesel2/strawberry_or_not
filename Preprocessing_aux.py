# auxiliary functions for preprocessing of spectral data
from scipy.spatial import ConvexHull
import numpy as np

def msc(X, reference=None):
    """
    Performs multiplicative scatter correction.

    Adapted from:
    https://nirpyresearch.com/two-scatter-correction-techniques-nir-spectroscopy-python/

    Parameters
    ----------
    X : numpy.ndarray of shape (n_samples, n_features)
        Feature matrix, one row per spectrum.
    
    reference : numpy array, optional
        Reference spectrum.
        If not given estimate from mean,
        by default None

    Returns
    -------
    numpy.ndarray of shape (n_samples, n_features)
    """
    # mean centre correction
    for i in range(X.shape[0]):
        X[i, :] -= X[i, :].mean()
    # Get the reference spectrum. If not given, estimate it from the mean
    if reference is None:
        # Calculate mean
        ref = np.mean(X, axis=0)
    else:
        ref = reference
    # Define a new array and populate it with the corrected data
    X_msc = np.zeros_like(X)
    for i in range(X.shape[0]):
        # Run regression
        fit = np.polyfit(ref, X[i, :], 1, full=True)
        # Apply correction
        X_msc[i, :] = (X[i, :] - fit[0][1]) / fit[0][0]

    return X_msc
    
def rubberband(X, dim):
    """
    Performs rubberband baseline correction.

    Adapted from:
    https://dsp.stackexchange.com/questions/2725/how-to-perform-a-rubberband-correction-on-spectroscopic-data

    Parameters
    ----------
    X : numpy.ndarray of shape (n_samples, n_features)
        Feature matrix, one row per spectrum.
    
    dim : numpy.ndarray of shape (n_features,)
        Wavelength, -number etc.


    Returns
    -------
    numpy.ndarray of shape (n_samples, n_features)
    """
    value_list = []
    dim = dim

    for i in range(len(X)):
        # Find the convex hull
        v = ConvexHull(np.array(list(zip(dim, X[i, :])))).vertices
        # Rotate convex hull vertices until they start from the lowest one
        v = np.roll(v, -v.argmin())
        # Leave only the ascending part
        v = v[:(v.argmax()+1)]  # different from stackexchange!*
        # *E.g., v[:159] does not include vertex #159, so we are missing the last part of the spectrum in the baseline correction!!
        
        # Create baseline using linear interpolation between vertices
        baseline = np.interp(dim, dim[v], X[i, :][v])
        new_values = X[i, :] - baseline
        value_list.append(new_values)

    return np.stack(value_list)
