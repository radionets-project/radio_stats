import numpy as np


def calculate_distances(parameters: list, order=None) -> np.ndarray:
    """
    Calculates the distances of all source centers to the first parameter

    Parameters
    ----------
    parameters : array_like
        Parameters for the gaussians

    order : {non-zero int, inf, -inf, ‘fro’, ‘nuc’}, optional
        Order of the norm, numpy.linalg.norm()

    Returns
    -------
    distances : np.ndarray
        Distance of all centers to the first center
    """
    distances = []
    for pair in parameters[1:, 2:4]:
        if np.all(pair == 0):
            distances.append(-1)
            continue

        distances.append(np.linalg.norm(parameters[0, 2:4] - pair, order))

    distances.append(0)
    distances = np.array(distances)[np.array(distances) >= 0]
    distances = np.roll(distances, 1)
    distances[0] = 0

    return distances


def calculate_offset(parameters, image) -> tuple[int, int]:
    """
    Calculates the offset of the center of fitted gaussians to the brightest pixel in image.

    Parameters
    ----------
    parameters : array_like
        Parameters for the gaussians

    image : array_like
        Image for the offset

    Returns
    -------
    x,y : tuple[int, int]
        Offset for (x,y) axis
    """
    positions = parameters[:, 2:4]

    offset = np.argwhere(image == image.max())[0] - positions[0]
    pos_offset = positions + offset

    x = np.round(pos_offset).astype(int)[:, 0]
    y = np.round(pos_offset).astype(int)[:, 1]

    return x, y
