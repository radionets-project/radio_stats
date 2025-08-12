import time
import warnings

import gaussfitter as gf
import numpy as np
from tqdm import tqdm

from radio_stats.cuts.resize import truncate


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
            print("amplitude too small", params[1])
        end_msg = "amplitude too small"

        return False, end_msg

    elif params[1] > 10 * np.max(work_img):
        if verbose:
            print("amplitude too large", params[1])
        end_msg = "amplitude too large"

        return False, end_msg

    elif params[4] == 0:
        end_msg = "x_width=0"
        if verbose:
            print(end_msg)

        return False, end_msg

    elif params[5] == 0:
        end_msg = "y_width=0"
        if verbose:
            print(end_msg)

        return False, end_msg

    elif np.amax(work_img) < np.amax(image) * res_flux:
        end_msg = "res. flux"
        if verbose:
            print(end_msg)

        return False, end_msg

    elif iteration == max_amount - 1:
        end_msg = "max_amount"
        if verbose:
            print(end_msg)

        return False, end_msg

    elif time.time() - start_time > max_time:
        end_msg = "Ran out of time"
        if verbose:
            print(end_msg)

        return False, end_msg

    return True, "go on"


def fit_gaussians(
    _image,
    verbose=False,
    use_truncate=True,
    max_amount=10,
    cut_percentage=1e-2,
    fraction=1,
    hot_start=False,
    **kwargs,
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
        if hot_start:
            from gaussfitter import twodgaussian

            for params in hot_start[0]:
                work_img -= hot_start[1] * twodgaussian(params)(
                    *np.indices(work_img.shape)
                )
                work_img[work_img < 0] = 0
                parameters.append(params)
            hot_start = False
        # return work_img
        tofit = np.copy(work_img)
        tofit[tofit < cut_percentage * np.amax(tofit)] = 0

        try:
            params, fitted_gauss = gf.gaussfit(tofit, returnfitimage=True)
        except ValueError:
            end_msg = "Value Error"
            if verbose:
                print(end_msg)

            break

        work_img = work_img - fraction * fitted_gauss
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
            **kwargs,
        )
        if not pursue:
            break

    return np.array(parameters), np.array(end_msg)
