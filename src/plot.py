import cv2
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kendalltau, spearmanr, pearsonr


def pixel_error(img1, img2, cmap_name="plasma", vmin=0.0, vmax=None, to_uint8=True):
    """
    img1, img2: HxWx3 RGB numpy arrays (float in [0,1] or uint8).
    Returns HxWx3 RGB array with the colormap applied.
    """
    # Ensure float in [0,1]
    a = img1.astype(np.float32)
    b = img2.astype(np.float32)
    if a.max() > 1.0 or b.max() > 1.0:
        a /= 255.0
        b /= 255.0

    # Per-pixel absolute error -> grayscale
    gray = cv2.cvtColor(
        np.abs(a - b), cv2.COLOR_RGB2GRAY
    )  # shape (H, W), float in [0,1]

    # Normalize to vmin..vmax for coloring
    if vmax is None or vmax == 0:
        vmax = float(gray.max() if gray.max() > 0 else 1.0)

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap(cmap_name)

    # RGBA in [0,1], drop alpha -> RGB
    rgb = cmap(norm(gray))[..., :3]

    if to_uint8:
        rgb = (rgb * 255).astype(np.uint8)

    return rgb


def plot_bxdf_polar(
    ax, vals, only_upper_hemisphere: bool = True, colors: str = "plasma"
):
    if only_upper_hemisphere:
        vals = vals[: vals.shape[0] // 2, :]
    theta_o = np.linspace(0, np.pi / 2, vals.shape[0])  # Only up to hemisphere
    phi_o = np.linspace(0, 2 * np.pi, vals.shape[1])  # Full azimuth

    # Create meshgrid for mapping
    Theta_o, Phi_o = np.meshgrid(theta_o, phi_o, indexing="ij")

    # Convert spherical to Cartesian (mapped to a disk)
    R = np.sin(Theta_o)  # Radial coordinate
    X = R * np.cos(Phi_o)
    Y = R * np.sin(Phi_o)

    # Plot the hemisphere projection
    c = ax.pcolormesh(X, Y, vals, shading="auto", cmap=colors)

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect("equal")

    # Hide frame but keep labels
    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=True)
    return ax


def plot_rdm(RDM: np.ndarray, title="RDM"):
    fig = plt.figure(figsize=(6, 5))
    plt.imshow(RDM, cmap="plasma", interpolation="nearest")
    plt.title(title)
    plt.xlabel("Stimuli")
    plt.ylabel("Stimuli")
    plt.colorbar(label="Dissimilarity")
    plt.tight_layout()
    return fig


def plot_rsa_scatter(RDM_A, RDM_B, method="kendall"):
    i, j = np.triu_indices(RDM_A.shape[0], k=1)
    a = RDM_A[i, j]
    b = RDM_B[i, j]

    fig = plt.figure(figsize=(5, 5))
    plt.scatter(a, b, alpha=0.6)
    plt.xlabel("Dissimilarities in RDM A")
    plt.ylabel("Dissimilarities in RDM B")

    # Compute correlation
    if method == "spearman":
        r, _ = spearmanr(a, b)
    elif method == "kendall":
        r, _ = kendalltau(a, b)
    else:
        r, _ = pearsonr(a, b)
    plt.title(f"RSA ({method}) r = {r:.3f}")
    plt.tight_layout()
    return fig

def plot_agg_similarity(ax: plt.axis, data: dict[str, np.ndarray]):
    for k, v in data.items():
        mean = np.mean(v, axis=0)
        sd = np.std(v, axis=0)
        X = np.arange(mean.shape[0])
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.plot(X, mean, label=k)
        y0, y1 = mean - sd, mean + sd
        ax.fill_between(X, y0, y1, alpha=0.25)
        ax.legend()

    return ax
