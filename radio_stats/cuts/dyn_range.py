import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def box_size(img, amount_boxes):
    """Calculates the edge length of the calculation boxes.

    Parameters
    ----------
    img : 2d np.array
        Image to be analyzed.
    amount_boxes : int
        Amount of the boxes to be calculated.
        Needs to be quadratic number.

    Returns
    -------
    length : int
        Length of the boxes for the calculation.
    """
    if np.sqrt(amount_boxes).round(0) != np.sqrt(amount_boxes):
        raise ValueError("Amount boxes has to be a quadratic number!")
    return int(img.shape[0] / np.sqrt(amount_boxes))


def check_validity(img, amount_boxes=36, threshold=1e-3):
    """Checks if the source is inside of the drawn boxes.
        If the source is inside the box it will be returned as False,
        otherwise as True.
        If every box cannot be used, the function recalls itself and
        doubles the threshold.

    Parameters
    ----------
    img : 2d np.array
        Image to be analyzed
    amount_boxes : int, default: 36
        Amount of the boxes to be calculated.
        Needs to be quadratic number.
    threshold : float
        Cutoff value to determine if the source is inside the box.

    Returns
    -------
    use_box : 2d np.array
        2d array of the usability of the boxes.
        If the box can be used, eg. no source in box, it will return True,
        otherwise False.
    """
    size = box_size(img, amount_boxes)
    row = int(np.sqrt(amount_boxes))
    use_box = np.zeros([row, row], dtype=bool)
    for j in range(row):
        for k in range(row):
            if np.any(
                img[size * j : size * (j + 1), size * k : size * (k + 1)] > threshold
            ):
                use_box[j, k] = False
            else:
                use_box[j, k] = True
    if np.sum(use_box) < row:
        use_box = check_validity(img, amount_boxes, threshold * 2)
    return use_box


def plot_boxes(img, boxes, axis=None, **kwargs):
    """Plots the boxes ontop of the original image.

    Parameters
    ----------
    img : 2d np.array
        Image to be analyzed
    boxes : 2d np.array
        2d array boolean of the usability of the boxes.
    axis : `~matplotlib.axes.Axes`, optinal
        Axes on whick the plot should be drawn.
        If None a new figure will be created.

    Returns
    -------
    plot : `~matplotlib.image.AxesImage`
        Plot of the original image colourd with the boxes.
        Red -> box not used, Green -> box used
    """
    row = boxes.shape[0]
    size = box_size(img, row**2)
    if axis == None:
        fig, axis = plt.subplots(1, 1)
    plot = axis.imshow(img, interpolation="none", **kwargs)
    color = np.zeros(boxes.shape, dtype=str)
    color[boxes] = "g"
    color[np.invert(boxes)] = "r"
    for j in range(row):
        for k in range(row):
            axis.fill_between(
                [size * j, size * (j + 1)],
                size * k,
                size * (k + 1),
                color=color[k, j],
                alpha=0.3,
            )
    return plot


def calc_rms_boxes(img, boxes):
    """Calculates the root mean square of the boxes.

    Parameters
    ----------
    img : 2d np.array
        Image to be analyzed
    boxes : 2d np.array
        2d array boolean of the usability of the boxes.

    Returns
    -------
    rms_boxes: 2d np.array
        Root mean square for each box.
        If the source is inside one box, the rms is set to -1.
    """
    row = boxes.shape[0]
    size = box_size(img, row**2)
    rms_boxes = -np.ones(boxes.shape)
    for j in range(boxes.shape[0]):
        for k in range(boxes.shape[1]):
            if not boxes[k, j]:
                continue
            else:
                rms_boxes[k, j] = np.sqrt(
                    np.mean(
                        img[k * size : (k + 1) * size, j * size : (j + 1) * size] ** 2
                    )
                )
    return rms_boxes


def rms_cut(_img, sigma=3, **kwargs):
    """Cuts an image using the rms.
        All values below the mean of the rms times sigma are set to zero.

    Parameters
    ----------
    img : 2d np.array or array of 2d arrays
        Images  to be analyzed
    sigma : float, default: 3.0
        Multiplier for the cut value.

    Reurns
    ------
    cut_img : 2d np.array or array of 2d arrays
        Cutted images.
    """
    img = np.copy(_img)
    if len(img.shape) == 2:
        ranges = calc_rms_boxes(img, check_validity(img, **kwargs))
        range_img = np.mean(ranges[ranges >= 0])
        img[img < sigma * range_img] = 0
        return img
    else:
        cut_img = []
        for pic in tqdm(img):
            ranges = calc_rms_boxes(pic, check_validity(pic, **kwargs))
            range_img = np.mean(ranges[ranges >= 0])
            pic[pic < sigma * range_img] = 0
            cut_img.append(pic)
        return np.array(cut_img)
