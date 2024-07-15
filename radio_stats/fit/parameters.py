from radio_stats.reconstruct.gaussians import reconstruct_gauss
from radio_stats.stats.parameters import calculate_distances, calculate_offset
import numpy as np
from scipy.optimize import curve_fit


def fit_lin(_parameters, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate parameters and errors of linear regression for intensity and brightness of fitted
    gaussians.

    Parameters
    ----------
    _parameters : array_like
        Parameters to be fitted

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
        _parameters[np.argwhere(np.abs(_parameters[:, 2:4]) > 1024)[:, 0], 1] = 0
        parameters = _parameters[: np.min(np.argwhere(_parameters[:, 1] == 0))]
    except ValueError:
        parameters = _parameters
    # if only one point is fittet, we can't fit a x1 function
    if parameters.shape[0] <= 1:
        return np.array([-np.inf, -np.inf]), np.array([-np.inf, -np.inf])

    image = reconstruct_gauss(parameters, (1024, 1024), **kwargs)

    distances = calculate_distances(parameters)

    x, y = calculate_offset(parameters, image)

    try:
        flux = image[y, x]
    except IndexError as e:
        print("IndexError:", e)
        return np.array([-np.inf, -np.inf]), np.array([-np.inf, -np.inf])

    def x1(x, a, b) -> float:
        return a * x + b

    params, covmat = curve_fit(x1, distances, np.log(flux), maxfev=10000)
    errors = np.sqrt(np.diag(covmat))

    if parameters.shape[0] == 2 and np.all(errors == np.inf):
        errors = np.array([0, 0])

    return params, errors
