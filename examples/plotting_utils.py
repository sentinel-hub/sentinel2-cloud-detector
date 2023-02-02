"""
Plotting utilities for example notebooks
"""
from typing import Any, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import Axes


def plot_image(
    image: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    ax: Optional[Axes] = None,
    factor: float = 3.5 / 255,
    clip_range: Tuple[float, float] = (0, 1),
    **kwargs: Any
) -> None:
    """Utility function for plotting RGB images and masks."""
    if ax is None:
        _, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))

    mask_color = [255, 255, 255, 255] if image is None else [255, 255, 0, 100]

    if image is None:
        if mask is None:
            raise ValueError("image or mask should be given")
        image = np.zeros(mask.shape + (3,), dtype=np.uint8)

    ax.imshow(np.clip(image * factor, *clip_range), **kwargs)

    if mask is not None:
        cloud_image = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)

        cloud_image[mask == 1] = np.asarray(mask_color, dtype=np.uint8)

        ax.imshow(cloud_image)


def plot_probabilities(image: np.ndarray, proba: np.ndarray, factor: float = 3.5 / 255) -> None:
    """Utility function for plotting a RGB image and its cloud probability map next to each other."""
    plt.figure(figsize=(15, 15))
    ax = plt.subplot(1, 2, 1)
    ax.imshow(np.clip(image * factor, 0, 1))
    ax = plt.subplot(1, 2, 2)
    ax.imshow(proba, cmap=plt.cm.inferno)
