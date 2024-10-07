from radio_stats.reconstruct.gaussians import reconstruct_gauss
from radio_stats.stats.parameters import calculate_distances, calculate_offset
from radio_stats.cuts.dyn_range import rms_cut
import numpy as np
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA


def fit_lin(_parameters, shape=(1024,1024), **kwargs) -> tuple[np.ndarray, np.ndarray]:
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
    # if only one point is fittet, we can't fit a x1 function
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
            return fit_lin(parameters, tuple([shape[0]*2]*2), **kwargs)

    def x1(x, a, b) -> float:
        return a * x + b

    params, covmat = curve_fit(x1, distances, np.log(flux), maxfev=10000)
    errors = np.sqrt(np.diag(covmat))

    if parameters.shape[0] == 2 and np.all(errors == np.inf):
        errors = np.array([0, 0])

    return params, errors

def check_components(parameters, lim_xyrel=(4/5, 4/3), delta_angle=30):
    if len(parameters.shape) == 1:
        return parameters
    elif len(parameters) <= 2:
        return parameters

    delta_angle = np.deg2rad(delta_angle)

    img_pca = PCA().fit(parameters[:,2:4])

    def angle_difference(parameter, img_pca):
        img = rms_cut(reconstruct_gauss(parameter, (1024, 1024)), 2)
        points = np.argwhere(img)[:,::-1]
        points -= np.mean(points, axis=0, dtype=int)

        pca = PCA().fit(points)

        return np.arccos(np.dot(pca.components_[0], img_pca.components_[0])/(np.linalg.norm(pca.components_[0]) * np.linalg.norm(img_pca.components_[0])))

    angle_diff = np.array([angle_difference(params, img_pca) for params in parameters])

    xyrel = parameters[:,4] / parameters[:,5]

    dismiss = np.zeros([len(parameters), 2], dtype=bool)
    for idx in range(len(parameters)):
        if not np.all([xyrel[idx] > lim_xyrel[0], xyrel[idx] < lim_xyrel[1]]):
            dismiss[idx, 0] = True
        if not np.any([angle_diff[idx] < delta_angle, angle_diff[idx] > np.pi - delta_angle]):
            dismiss[idx, 1] = True
        # override if to narrow:
        if np.any([xyrel[idx] > 6, xyrel[idx] < 1/6]):
            dismiss[idx, 0] = True
            dismiss[idx, 1] = True
        # keep if enough flux
        if parameters[idx, 1] / parameters[0, 1] > 0.05:
            dismiss[idx, 0] = False            
            dismiss[idx, 1] = False


    cut_parameters = np.delete(parameters, np.argwhere(dismiss[:, 0] * dismiss[:, 1]).T[0], axis=0)
    try:
        if not np.all(cut_parameters[0] == parameters[0]):
            cut_parameters = np.insert(cut_parameters, 0, parameters[0], axis=0)
    except IndexError:
        cut_parameters = parameters[0]

    return cut_parameters
