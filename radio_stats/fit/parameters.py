import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import curve_fit

from radio_stats.reconstruct import reconstruct_gauss
from radio_stats.stats import calculate_distances, calculate_offset


def fit_lin(
    _parameters: ArrayLike, shape: tuple = (1024, 1024), **kwargs
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate parameters and errors of linear regression for intensity and brightness of fitted
    gaussians.

    Parameters
    ----------
    _parameters : array_like
        Parameters to be fitted

    shape : tuple
        Shape of the internal calculation image

    kwargs
        Keyword arguments for reconstruct_gauss

    Returns
    -------
    params : np.ndarray
        Fitted parameters for the linear function

    errors : np.ndarray
        Errors of the fitted parameters
    """
    try:
        _parameters[np.argwhere(np.abs(_parameters[:, 2:4]) > shape[0])[:, 0], 1] = 0
        parameters = _parameters[: np.min(np.argwhere(_parameters[:, 1] == 0))]
    except ValueError:
        parameters = _parameters

    # if only one point is fitted, we can't fit a x1 function
    if parameters.shape[0] <= 1:
        return np.array([-np.inf, -np.inf]), np.array([-np.inf, -np.inf])

    image = reconstruct_gauss(parameters, shape, **kwargs)

    distances = calculate_distances(parameters)

    x, y = calculate_offset(parameters, image)

    try:
        flux = image[y, x]
    except IndexError as e:
        if shape[0] >= 4096:
            print("IndexError:", e)
            return np.array([-np.inf, -np.inf]), np.array([-np.inf, -np.inf])
        else:
            return fit_lin(parameters, tuple([shape[0] * 2] * 2), **kwargs)

    def x1(x, a, b) -> float:
        return a * x + b

    params, covmat = curve_fit(x1, distances, np.log(flux), maxfev=10000)
    errors = np.sqrt(np.diag(covmat))

    if parameters.shape[0] == 2 and np.all(errors == np.inf):
        errors = np.array([0, 0])

    return params, errors
