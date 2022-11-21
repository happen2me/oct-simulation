import matplotlib.pyplot as plt
import numpy as np

def plot_horizontal(images, x_labels, figsize=(15,3)):
    """Plot images horizontally
    """
    n = len(images)
    _, axes = plt.subplots(1,n, figsize=figsize)
    for i in range(n):
        axes[i].imshow(images[i])
        axes[i].set_xlabel(f"{x_labels[i]}")    


def visualize_segmentation(bscan, label, show_original=False, alpha=0.8):
    """Visualize segmentation mask on top of bscan
    """
    color_map = {
        1: 'white',
        2: 'red',
        3: 'gray',
        4: 'orange',
        5: 'yellow',
        6: 'gainsboro'
    }
    if show_original:
        _, axs = plt.subplots(1, 2, figsize=(5, 10))
        axs[0].imshow(bscan)
        axs[0].axis('off')
        axs[0].set_title('original')
        axs[1].set_title('segmentation')
        draw_handle = axs[1]
    else:
        draw_handle = plt
    draw_handle.axis('off')
    draw_handle.imshow(bscan)
    for k, color in color_map.items():
        x, y = np.where(label==k)
        draw_handle.scatter(y, x, color=color, alpha=alpha, linewidths=0, s=0.5)
