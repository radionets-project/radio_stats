import numpy as np
import gaussfitter as gf
import warnings
from radio_stats.cuts.resize import truncate
from tqdm import tqdm
import time


def check_termination(
    params,
    image,
    work_img,
    iteration,
    max_amount,
    verbose,
    start_time,
    amplitude_cut=1e-4,
    res_flux=0.01,
    max_time=300,
) -> tuple[bool, str]:
    if params[0] + params[1] <= 0:
        if verbose:
            print(params[0] + params[1])
        end_msg = "height + amplitude < 0"
        return False, end_msg
    elif params[1] < amplitude_cut:
        if verbose:
            print(params[1])
        end_msg = "amplitude too small"
        return False, end_msg
    elif params[1] > 10 * np.max(work_img):
        if verbose:
            print(params[1])
        end_msg = "amplitude too large"
        return False, end_msg
    elif params[4] == 0:
        if verbose:
            print("x_width=0")
        end_msg = "x_width=0"
        return False, end_msg
    elif params[5] == 0:
        if verbose:
            print("y_width=0")
        end_msg = "y_width=0"
        return False, end_msg
    elif np.amax(work_img) < np.amax(image) * res_flux:
        if verbose:
            print("res_flux")
        end_msg = "res. flux"
        return False, end_msg
    elif iteration == max_amount - 1:
        if verbose:
            print("max_amount")
        end_msg = "max_amount"
        return False, end_msg
    elif time.time() - start_time > max_time:
        if verbose:
            print("Ran out of time")
        end_msg = "Ran out of time"
        return False, end_msg
    return True, "go on"


def fit_gaussians(
    _image,
    verbose=False,
    use_truncate=True,
    max_amount=10,
    cut_percentage=1e-2,
    **kwargs
):
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    image = _image
    if use_truncate:
        image = truncate(image)

    parameters = []
    work_img = np.copy(image)
    start_time = time.time()

    if verbose:
        iterator = tqdm(range(max_amount))
    else:
        iterator = range(max_amount)

    for i in iterator:
        tofit = np.copy(work_img)
        tofit[tofit < cut_percentage * np.amax(tofit)] = 0
        try:
            params, fitted_gauss = gf.gaussfit(tofit, returnfitimage=True)
        except ValueError:
            if verbose:
                print("Value Error")
            end_msg = "Value Error"
            break

        work_img = work_img - fitted_gauss
        work_img[work_img < 0] = 0
        parameters.append(params)
        pursue, end_msg = check_termination(
            params=params,
            image=image,
            work_img=work_img,
            iteration=i,
            max_amount=max_amount,
            verbose=verbose,
            start_time=start_time,
            **kwargs
        )
        if not pursue:
            break

    return np.array(parameters), np.array(end_msg)
