"""Per-tile vector quantisation: k-means codebooks and reconstruction.

``quantize_tile`` returns a ``(k, 128)`` codebook and an ``(H, W)`` index map for an
``(H, W, 128)`` tile. ``reconstruct_tile`` looks up the codebook by the index map.

For ``distance="cosine"`` we L2-normalise the inputs before clustering — euclidean
k-means on the unit sphere is equivalent to cosine k-means.
"""

from __future__ import annotations

from typing import Literal, cast

import numpy as np
import numpy.typing as npt
from sklearn.cluster import KMeans, MiniBatchKMeans

Distance = Literal["euclidean", "cosine"]

# Above this codebook size we use MiniBatchKMeans (KMeans n_init becomes expensive).
_MINIBATCH_K_THRESHOLD = 64


def quantize_tile(
    tile: npt.NDArray[np.float32],
    k: int,
    distance: Distance = "euclidean",
    seed: int = 42,
    *,
    n_init: int = 3,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int32]]:
    """K-means quantise an ``(H, W, 128)`` tile; returns ``(codebook, indices)``."""
    h, w, c = tile.shape
    x = tile.reshape(-1, c).astype(np.float32, copy=False)
    if distance == "cosine":
        norm = np.linalg.norm(x, axis=1, keepdims=True)
        x = (x / np.where(norm > 0, norm, 1.0)).astype(np.float32, copy=False)
    k_eff = min(k, x.shape[0])
    if k_eff > _MINIBATCH_K_THRESHOLD:
        model = MiniBatchKMeans(n_clusters=k_eff, batch_size=1024, n_init=n_init, random_state=seed)
    else:
        model = KMeans(n_clusters=k_eff, n_init=n_init, random_state=seed)
    labels = model.fit_predict(x)
    centers = cast("npt.NDArray[np.float32]", np.asarray(model.cluster_centers_, dtype=np.float32))
    return centers, labels.reshape(h, w).astype(np.int32)


def reconstruct_tile(
    codebook: npt.NDArray[np.float32], indices: npt.NDArray[np.int32]
) -> npt.NDArray[np.float32]:
    """Reconstruct ``(H, W, 128)`` from a codebook and an index map."""
    return np.asarray(codebook[indices], dtype=np.float32)
