import numpy as np
from gaussfitter import twodgaussian


def reconstruct_gauss(
    parameters: list, shape: tuple, verbose=False, **kwargs
) -> np.ndarray:
    """
    Generate a centered image out of parameters for gaussians.

    Parameters
    ----------
    parameters : array_like
        Parameters for the gaussians

    shape : tuple
        Shape of the output image

    **kawrgs
        Keyword arguments for gaussfitter twodgaussian

    Returns
    -------
    summed_gaussian : np.ndarray
        Image consisting of the sum of individual gaussians
    """
    parameters = np.array(parameters)
    if len(parameters.shape) == 1:
        parameters = [parameters]

    peak = np.floor_divide(shape, 2)

    summed_gaussian = np.zeros(shape)
    if verbose:
        from tqdm import tqdm

        iterator = tqdm(parameters)
    else:
        iterator = parameters

    for params in iterator:
        params[2:4] += int(shape[0] / 2)
        if np.all(params == 0):
            continue
        summed_gaussian += twodgaussian(params, **kwargs)(*np.indices(shape))

    difference = peak - np.argwhere(summed_gaussian == np.max(summed_gaussian))[0]
    summed_gaussian = np.roll(summed_gaussian, difference, axis=(0, 1))

    return summed_gaussian
