import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

def draw_history(history):
    plt.rcParams['figure.figsize'] = [8, 4]
    figure, ax = plt.subplots(1, 2)
    ax[0].plot(history.history["loss"], label=f"Loss trainig data ")
    ax[0].plot(history.history["val_loss"], label=f"Loss validation data ")
    ax[0].grid()
    ax[0].legend()

    ax[1].plot(history.history["mse"], label=f"MSE trainig data ")
    ax[1].plot(history.history["val_mse"], label=f"MSE validation data ")
    ax[1].grid()
    ax[1].legend()
    plt.show()


def draw_distribution(y, y_hat, label="Train", names=["A1", "A2", "T1", "T2"]):
    plt.rcParams['figure.figsize'] = [8, 8]
    fig, ax = plt.subplots(2, 2)
    name = names[0]
    ax[0, 0].hist(y_hat[:, 0], bins=100, label=f"Computed distribution of {name}")
    ax[0, 0].hist(y[:, 0], bins=100, label=f"Real distribution of {name}", alpha=0.5, color="red")
    ax[0, 0].legend()
    ax[0, 0].grid()

    name = names[1]
    ax[0, 1].hist(y_hat[:, 1], bins=100, label=f"Computed distribution of {name}")
    ax[0, 1].hist(y[:, 1], bins=100, label=f"Real distribution of {name}", alpha=0.5, color="red")
    ax[0, 1].legend()
    ax[0, 1].grid()

    name = names[2]
    ax[1, 0].hist(y_hat[:, 2], bins=100, label=f"Computed distribution of {name}")
    ax[1, 0].hist(y[:, 2], bins=100, label=f"Real distribution of {name}", alpha=0.5, color="red")
    ax[1, 0].legend()
    ax[1, 0].grid()

    name = names[3]
    ax[1, 1].hist(y_hat[:, 3], bins=100, label=f"Computed distribution of {name}")
    ax[1, 1].hist(y[:, 3], bins=100, label=f"Real distribution of {name}", alpha=0.5, color="red")
    ax[1, 1].legend()
    ax[1, 1].grid()

    fig.suptitle(f"Distribution {label}:{names}")
    plt.show()


def draw_corr(y, y_hat, label="Train", names=["A1", "A2", "T1", "T2"], out_dim=4):
    import mpl_scatter_density
    plt.rcParams['figure.figsize'] = [8, 8]
    white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
        (0, '#ffffff'),
        (1e-20, '#440053'),
        (0.2, '#404388'),
        (0.4, '#2a788e'),
        (0.6, '#21a784'),
        (0.8, '#78d151'),
        (1, '#fde624'),
    ], N=256)
    fig = plt.figure()
    for i in range(out_dim):
        ax = fig.add_subplot(2, 2, i+1, projection='scatter_density')
        name = names[i]
        corr = np.corrcoef(y[:, i], y_hat[:, i])[0, 1]
        density = ax.scatter_density(y[:, i], y_hat[:, i], cmap=white_viridis, dpi=None)
        ax.set_title(name+f" corr: {corr}")
        fig.colorbar(density, label='Density')
    fig.suptitle(f"Correlation {label}:{names}")
    plt.show()

