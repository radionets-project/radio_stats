import numpy as np
from gaussfitter import twodgaussian

def reconstruct_gauss(parameters : list, shape : tuple, **kwargs)-> np.NDArray:
    '''
    Generate an image out of parameters for gaussians.

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
    summed_gaussian : np.NDArray
        Image consisting of the sum of individual gaussians
    '''
    parameters = np.array(parameters)
    if len(parameters.shape) == 1:
        parameters = [parameters]
    peak = np.floor_divide(shape)
    gaussians = []
    for params in parameters:
        if np.all(params == 0):
            continue
        gaussians.append(twodgaussian(params, **kwargs)(*np.indices(shape)))
    gaussians = np.array(gaussians)
    if len(gaussians.shape) == 2:
         summed_gaussian = gaussians
    else:
        summed_gaussian = gaussians.sum(axis = 0)
            
    difference = peak - np.argwhere(summed_gaussian == np.max(summed_gaussian))[0]
    
    summed_gaussian = np.roll(summed_gaussian, difference, axis=(0,1))
    
    return summed_gaussian