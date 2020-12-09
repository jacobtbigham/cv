import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from IPython import get_ipython

def show(image, x=5, y=5):
    """
    Displays a BGR or grayscale/single-channel image inline.
    """
    get_ipython().run_line_magic('matplotlib', 'inline')
    rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    plt.figure(figsize=(x, y))
    plt.imshow(rgb)
    plt.axis('off')
    plt.show()


def show_sbs(images, spacer_width = 15, show_width=10, show_height=10):
    """
    Displays the images in the 'images' iterable side-by-side, separated by a white spacer of spacer_width width.
    show_height and show_width optionally adjust the size of the displayed combined image.
    Images can be either BGR or grayscale.
    This is not useful for more than 2-5 images, depending on size.
    """
    num_images = len(images)
    height = max([image.shape[0] for image in images])
    width = sum([image.shape[1] for image in images]) + spacer_width * (num_images - 1)
    vertical_spacer = np.full((height, spacer_width, 3), 255, dtype=np.uint8)
    image_list = []
    for i, image in enumerate(images):
        if len(image.shape) == 2:
            image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
        if image.shape[0] != height:
            padding = np.full((height - image.shape[0], image.shape[1], 3), 255, dtype=np.uint8)
            padded = np.concatenate((image, padding), axis=0)
            image_list.append(padded)
        else:
            image_list.append(image)
        if i < (num_images - 1):
            image_list.append(vertical_spacer)
    sbs = np.concatenate(image_list, axis=1)
    show(sbs, show_width, show_height)


def image_as_function(image, alpha=70, beta=75):
    """
    Plots an image as a three-dimensional grayscale function.
    Supports flat and 3-channel grayscale images, but does not convert to grayscale from color.
    Alpha and beta parameters control default rotation of the generated figure.
    """
    get_ipython().run_line_magic('matplotlib', 'notebook')
    if len(image.shape) == 3:
        height, width, _ = image.shape
        Z = image[:, :, 0]
    else:
        height, width = image.shape
        Z = image
    x = np.arange(width-1, -1, -1)
    y = np.arange(0, height, 1)
    X, Y = np.meshgrid(x, y)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='gray')
    ax.view_init(alpha, beta)
    plt.show()


def better_edge_detector(image, threshold=15):
    """
    Plots edges along the horizontal and vertical axes of an image.
    An edge is any location where the gradient exceeds the threshold.
    """
    if len(image.shape) > 2:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    delta_x = np.gradient(image, axis = 1)
    delta_y = np.gradient(image, axis = 0)
    x_mask = np.abs(delta_x) >= threshold
    y_mask = np.abs(delta_y) >= threshold
    mask = np.logical_or(x_mask, y_mask)
    delta = (mask*255).astype(np.uint8)
    show(delta)
