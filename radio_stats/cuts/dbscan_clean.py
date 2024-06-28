import numpy as np
from sklearn.cluster import DBSCAN
from tqdm import tqdm

def dbscan_clean(_images, max_groups=5, min_brightness=0.05, eps=5, **kwargs):
    """
    Clean a pre-cleaned radio image.
    
    Parameters
    ----------
    _images : array_like
        Image or array of images to be cleaned.
    
    max_groups : int, default 5
        Maximal amount of left groups per image.
        
    min_brightness : float, default 0.01
        Minimal cumulative brightness per group to be considered relevant.
        
    eps : float, default 5
        Parameter for DBSCAN algorithm
        
    **kwargs
        Parameters passed through to DBSCAN.
        
    Returns
    -------
    cleaned_images : array like
        Cleaned image or array of cleaned images.   
    """
    images = np.copy(_images)
    is_list = len(images.shape) != 2
    cleaned_images = []

    if not is_list:
        images = [images]
        
    for img in tqdm(images):
        points = np.argwhere(img)
        
        clusters = DBSCAN(eps, **kwargs).fit(points)
        labels = clusters.labels_
        
        index, amount = np.unique(labels[labels>=0], return_counts=True)
        
        ascending_size = np.flip(np.argsort(amount))
        label_mask = np.isin(labels, index[ascending_size[:max_groups]])
        
        intensities = img[points[:,0], points[:,1]]
        
        brightnesses = []
        for labl in index:
            brightnesses.append(np.sum(intensities[labels==labl]))
        brightnesses = np.array(brightnesses)
        
        brightness_mask = np.isin(labels, index[brightnesses>=min_brightness*np.max(brightnesses)])
        combined_mask = label_mask & brightness_mask
        
        cleaned = np.zeros(img.shape)
        cleaned[points[:,0][combined_mask], points[:,1][combined_mask]] = img[points[:,0][combined_mask], points[:,1][combined_mask]]
        
        cleaned_images.append(cleaned)
    
    if not is_list:
        return cleaned_images[0]
    
    return np.array(cleaned_images)
