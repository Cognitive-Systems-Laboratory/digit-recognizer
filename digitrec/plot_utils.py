import torch
import matplotlib.pyplot as plt


def _plot_tensor(X: torch.Tensor, y: torch.Tensor = None, ax=None):

    assert X.ndim == 2, f"Please provide n*n array. Given: {X.ndim}, {X.shape}"
    if ax is None:
        fig, ax = plt.subplots()
    X = X.numpy()
    ax.imshow(X, cmap="gray");
    if y is not None:
        ax.set_title(f"Label: {y}")
        ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

def plot_tensors(X: torch.Tensor, y: torch.Tensor):
    fig, _ax = plt.subplots(figsize=(6, 3), nrows=2, ncols=4)
    for i in range(8):
        r, c = i // 4, i % 4
        ax = _ax[r, c]
        _plot_tensor(X=X[i], y=y[i], ax=ax)
    fig.tight_layout()
