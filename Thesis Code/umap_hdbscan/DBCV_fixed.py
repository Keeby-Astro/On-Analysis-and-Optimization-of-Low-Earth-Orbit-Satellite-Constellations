import numpy as np
from scipy.spatial.distance import euclidean, cdist
from scipy.sparse.csgraph import minimum_spanning_tree
from numba import njit, prange


# -----------------------------------------------------------------------------
# Optional CUDA pairwise distance support
# -----------------------------------------------------------------------------

def _has_numba_cuda():
    try:
        from numba import cuda  # noqa: F401
        return True
    except Exception:
        return False


def _distance_matrix_size_bytes(n_samples: int) -> int:
    return int(n_samples) * int(n_samples) * np.dtype(np.float64).itemsize


def _memory_guard_for_distance_matrix(n_samples: int, *, use_gpu: bool) -> None:
    if n_samples <= 0:
        return

    bytes_needed = _distance_matrix_size_bytes(n_samples)
    if bytes_needed > 8 * 1024**3:
        raise MemoryError(
            f"DBCV needs an {n_samples}x{n_samples} distance matrix "
            f"(~{bytes_needed / 1024**3:.1f} GiB). This is not practical. "
            "Consider sampling/downsizing, or compute DBCV on a subset."
        )

    if not use_gpu:
        return

    try:
        from numba import cuda
        free_bytes, _total_bytes = cuda.current_context().get_memory_info()
        if bytes_needed > 0.8 * free_bytes:
            raise MemoryError(
                f"Not enough free GPU memory for {n_samples}x{n_samples} distances "
                f"(~{bytes_needed / 1024**3:.1f} GiB). "
                f"Free GPU memory ~{free_bytes / 1024**3:.1f} GiB. "
                "Use use_gpu=False or reduce n."
            )
    except MemoryError:
        raise
    except Exception:
        # Best-effort only. If memory info is unavailable, allow allocation to fail naturally.
        pass


def _cdist_optional_numba_cuda(X, metric, use_gpu):
    """Compute pairwise distances, optionally using Numba CUDA on GPU.

    Notes
    -----
    Only the dense square pairwise distance stage is GPU-accelerated. All later
    DBCV steps remain on CPU. This keeps compatibility with the existing script
    while avoiding changes to its public API.
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    n_samples = int(X.shape[0])
    if n_samples <= 0:
        return np.empty((0, 0), dtype=np.float64)

    _memory_guard_for_distance_matrix(n_samples, use_gpu=use_gpu)

    if not use_gpu:
        return cdist(X, X, metric=metric)

    if not _has_numba_cuda():
        raise ImportError(
            "use_gpu=True requires Numba CUDA (numba.cuda) and an NVIDIA GPU driver. "
            "Install a CUDA-capable Numba build or set use_gpu=False."
        )

    from numba import cuda

    if metric == "euclidean":
        kernel = _cdist_euclidean_cuda
    elif metric == "cityblock":
        kernel = _cdist_cityblock_cuda
    elif metric == "chebyshev":
        kernel = _cdist_chebyshev_cuda
    elif metric == "canberra":
        kernel = _cdist_canberra_cuda
    elif metric == "braycurtis":
        kernel = _cdist_braycurtis_cuda
    else:
        raise ValueError(
            f"Unsupported GPU metric '{metric}'. Supported: euclidean, cityblock, "
            "chebyshev, canberra, braycurtis"
        )

    X_dev = cuda.to_device(X)
    D_dev = cuda.device_array((n_samples, n_samples), dtype=np.float64)

    if n_samples < 128:
        threadsperblock = (8, 8)
    elif n_samples < 1024:
        threadsperblock = (16, 16)
    else:
        threadsperblock = (32, 8)

    blockspergrid_x = (n_samples + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (n_samples + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    kernel[blockspergrid, threadsperblock](X_dev, D_dev)
    return D_dev.copy_to_host()


try:
    from numba import cuda
    import math

    @cuda.jit(fastmath=True)
    def _cdist_euclidean_cuda(X, D):
        i, j = cuda.grid(2)
        n = X.shape[0]
        if i >= n or j >= n:
            return
        if j < i:
            return
        if i == j:
            D[i, j] = 0.0
            return
        acc = 0.0
        d = X.shape[1]
        for k in range(d):
            diff = X[i, k] - X[j, k]
            acc += diff * diff
        dist = math.sqrt(acc)
        D[i, j] = dist
        D[j, i] = dist

    @cuda.jit(fastmath=True)
    def _cdist_cityblock_cuda(X, D):
        i, j = cuda.grid(2)
        n = X.shape[0]
        if i >= n or j >= n:
            return
        if j < i:
            return
        if i == j:
            D[i, j] = 0.0
            return
        acc = 0.0
        d = X.shape[1]
        for k in range(d):
            acc += abs(X[i, k] - X[j, k])
        D[i, j] = acc
        D[j, i] = acc

    @cuda.jit(fastmath=True)
    def _cdist_chebyshev_cuda(X, D):
        i, j = cuda.grid(2)
        n = X.shape[0]
        if i >= n or j >= n:
            return
        if j < i:
            return
        if i == j:
            D[i, j] = 0.0
            return
        m = 0.0
        d = X.shape[1]
        for k in range(d):
            v = abs(X[i, k] - X[j, k])
            if v > m:
                m = v
        D[i, j] = m
        D[j, i] = m

    @cuda.jit(fastmath=True)
    def _cdist_canberra_cuda(X, D):
        i, j = cuda.grid(2)
        n = X.shape[0]
        if i >= n or j >= n:
            return
        if j < i:
            return
        if i == j:
            D[i, j] = 0.0
            return
        acc = 0.0
        d = X.shape[1]
        for k in range(d):
            denom = abs(X[i, k]) + abs(X[j, k])
            if denom != 0.0:
                acc += abs(X[i, k] - X[j, k]) / denom
        D[i, j] = acc
        D[j, i] = acc

    @cuda.jit(fastmath=True)
    def _cdist_braycurtis_cuda(X, D):
        i, j = cuda.grid(2)
        n = X.shape[0]
        if i >= n or j >= n:
            return
        if j < i:
            return
        if i == j:
            D[i, j] = 0.0
            return
        num = 0.0
        den = 0.0
        d = X.shape[1]
        for k in range(d):
            num += abs(X[i, k] - X[j, k])
            den += abs(X[i, k] + X[j, k])
        if den == 0.0:
            D[i, j] = 0.0
            D[j, i] = 0.0
        else:
            dist = num / den
            D[i, j] = dist
            D[j, i] = dist

except Exception:
    pass


# -----------------------------------------------------------------------------
# Metric helpers and compatibility aliases
# -----------------------------------------------------------------------------

def _distance_manhattan(x, y):
    return np.sum(np.abs(x - y))


def _distance_chebyshev(x, y):
    return np.max(np.abs(x - y))


def _distance_canberra(x, y):
    s = 0.0
    for i in range(x.shape[0]):
        denom = abs(x[i]) + abs(y[i])
        if denom != 0.0:
            s += abs(x[i] - y[i]) / denom
    return s


def _distance_braycurtis(x, y):
    num = np.sum(np.abs(x - y))
    den = np.sum(np.abs(x + y))
    if den == 0.0:
        return 0.0
    return num / den


DISTANCE_FUNCTIONS = {
    "euclidean": euclidean,
    "manhattan": _distance_manhattan,
    "chebyshev": _distance_chebyshev,
    "canberra": _distance_canberra,
    "braycurtis": _distance_braycurtis,
}

PAIRWISE_METRIC_MAP = {
    "euclidean": "euclidean",
    "manhattan": "cityblock",
    "cityblock": "cityblock",
    "chebyshev": "chebyshev",
    "canberra": "canberra",
    "braycurtis": "braycurtis",
}


@njit(cache=True)
def _all_points_core_distance_from_distance_matrix(distance_matrix, d):
    n = distance_matrix.shape[0]
    out = np.zeros(n, dtype=np.float64)
    if n <= 1:
        return out

    denom = n - 1
    for i in range(n):
        s = 0.0
        for j in range(n):
            dij = distance_matrix[i, j]
            if dij != 0.0:
                s += (1.0 / dij) ** d
        s /= denom
        if s > 0.0:
            out[i] = s ** (-1.0 / d)
        else:
            out[i] = 0.0
    return out


@njit(cache=True, parallel=True)
def _mutual_reachability_from_distance_matrix(distance_matrix, core_distances):
    n = distance_matrix.shape[0]
    mr = np.empty((n, n), dtype=np.float64)
    for i in prange(n):
        ci = core_distances[i]
        for j in range(n):
            val = distance_matrix[i, j]
            cj = core_distances[j]
            if ci > val:
                val = ci
            if cj > val:
                val = cj
            mr[i, j] = val
    return mr


# Compatibility aliases for existing custom-script structure
APTSCORE_FUNCTIONS = {}
CORE_CLUSTER_FUNCTIONS = {}


def _normalize_metric(dist_function):
    """Resolve user metric input to a stable internal representation.

    Returns
    -------
    metric_name : str or None
        Internal normalized metric name when recognized.
    dist_callable : callable
        Callable usable for fallback distance calculations.
    cdist_metric : str or callable
        Object suitable for scipy.spatial.distance.cdist.
    """
    if isinstance(dist_function, str):
        key = dist_function.strip().lower()
        if key in DISTANCE_FUNCTIONS:
            return key, DISTANCE_FUNCTIONS[key], PAIRWISE_METRIC_MAP[key]
        if key in PAIRWISE_METRIC_MAP:
            # Accept scipy-style cityblock in addition to custom "manhattan".
            metric_name = "manhattan" if key == "cityblock" else key
            callable_metric = DISTANCE_FUNCTIONS[metric_name]
            return metric_name, callable_metric, PAIRWISE_METRIC_MAP[key]
        raise ValueError("Unsupported distance metric: " + dist_function)

    if dist_function == euclidean:
        return "euclidean", dist_function, "euclidean"

    name = getattr(dist_function, "__name__", "")
    if name in {"_distance_manhattan", "manhattan", "cityblock"}:
        return "manhattan", _distance_manhattan, "cityblock"
    if name in {"_distance_chebyshev", "chebyshev"}:
        return "chebyshev", _distance_chebyshev, "chebyshev"
    if name in {"_distance_canberra", "canberra"}:
        return "canberra", _distance_canberra, "canberra"
    if name in {"_distance_braycurtis", "braycurtis"}:
        return "braycurtis", _distance_braycurtis, "braycurtis"

    # Custom callables remain supported through SciPy's callable cdist path.
    return None, dist_function, dist_function


# -----------------------------------------------------------------------------
# DBCV core routines, updated to follow the hdbscan.validity formulation
# -----------------------------------------------------------------------------

def _pairwise_distance_matrix(X, cdist_metric, *, use_gpu=False):
    X = np.ascontiguousarray(X, dtype=np.float64)
    n = int(X.shape[0])
    if n == 0:
        return np.empty((0, 0), dtype=np.float64)

    if isinstance(cdist_metric, str) and cdist_metric in {"euclidean", "cityblock", "chebyshev", "canberra", "braycurtis"}:
        return _cdist_optional_numba_cuda(X, cdist_metric, use_gpu=use_gpu)

    _memory_guard_for_distance_matrix(n, use_gpu=False)
    return cdist(X, X, metric=cdist_metric)


@njit(cache=True)
def _dense_label_inverse(indices):
    out = np.empty(indices.shape[0], dtype=np.int64)
    for i in range(indices.shape[0]):
        out[i] = i
    return out


def _remap_nonnoise_labels_to_dense(labels):
    unique = np.unique(labels)
    mapping = {label: i for i, label in enumerate(unique.tolist())}
    dense = np.empty(labels.shape[0], dtype=np.int64)
    for idx, lbl in enumerate(labels.tolist()):
        dense[idx] = mapping[lbl]
    return dense, unique


def _cluster_distance_and_core(X_cluster, cdist_metric, ambient_dim, *, use_gpu=False):
    distance_matrix = _pairwise_distance_matrix(X_cluster, cdist_metric, use_gpu=use_gpu)
    core_distances = _all_points_core_distance_from_distance_matrix(
        np.ascontiguousarray(distance_matrix, dtype=np.float64), float(ambient_dim)
    )
    mr_distances = _mutual_reachability_from_distance_matrix(
        np.ascontiguousarray(distance_matrix, dtype=np.float64),
        np.ascontiguousarray(core_distances, dtype=np.float64),
    )
    return distance_matrix, core_distances, mr_distances


def _internal_minimum_spanning_tree(mr_distances):
    """Return internal MST nodes and weighted edges for one cluster.

    This mirrors the documented hdbscan.validity behavior closely enough for a
    drop-in custom script: the MST is computed per cluster, internal vertices are
    those of degree > 1, and when no internal edges exist the full cluster MST is
    used for density sparseness.
    """
    n = mr_distances.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.int64), np.empty((0, 3), dtype=np.float64)
    if n == 1:
        return np.array([0], dtype=np.int64), np.empty((0, 3), dtype=np.float64)

    mst = minimum_spanning_tree(mr_distances).tocoo()
    rows = mst.row.astype(np.int64)
    cols = mst.col.astype(np.int64)
    data = mst.data.astype(np.float64)
    edges = np.column_stack((rows, cols, data))

    degrees = np.zeros(n, dtype=np.int64)
    for u in rows:
        degrees[u] += 1
    for v in cols:
        degrees[v] += 1

    internal_nodes = np.where(degrees > 1)[0].astype(np.int64)
    if internal_nodes.size == 0:
        internal_nodes = np.array([0], dtype=np.int64)

    if edges.size == 0:
        return internal_nodes, edges.reshape(0, 3)

    mask_u = np.isin(edges[:, 0].astype(np.int64), internal_nodes)
    mask_v = np.isin(edges[:, 1].astype(np.int64), internal_nodes)
    internal_edges = edges[mask_u & mask_v]

    if internal_edges.size == 0:
        internal_edges = edges.copy()

    return internal_nodes, internal_edges


def _density_separation(
    X,
    dense_labels,
    cluster_id1,
    cluster_id2,
    internal_nodes1,
    internal_nodes2,
    core_distances1,
    core_distances2,
    cdist_metric,
):
    cluster1 = X[dense_labels == cluster_id1][internal_nodes1]
    cluster2 = X[dense_labels == cluster_id2][internal_nodes2]

    if cluster1.size == 0 or cluster2.size == 0:
        return np.inf

    distance_matrix = cdist(cluster1, cluster2, metric=cdist_metric)
    core_dist_matrix1 = np.tile(core_distances1[internal_nodes1], (distance_matrix.shape[1], 1)).T
    core_dist_matrix2 = np.tile(core_distances2[internal_nodes2], (distance_matrix.shape[0], 1))
    mr_dist_matrix = np.dstack([distance_matrix, core_dist_matrix1, core_dist_matrix2]).max(axis=-1)
    return float(np.min(mr_dist_matrix))


def _cluster_density_sparseness(internal_edges):
    if internal_edges.size == 0:
        return 0.0
    return float(np.max(internal_edges[:, 2]))


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def DBCV(X, labels, dist_function, *, use_gpu=False):
    """Compute the Density-Based Clustering Validation (DBCV) index.

    This function preserves the existing custom-script public API:

        DBCV(X, labels, dist_function, *, use_gpu=False)

    while fixing the following issues in the previous custom implementation:
    1. HDBSCAN label 0 is no longer treated as noise.
    2. DBCV is now computed using per-cluster mutual-reachability MSTs instead of
       a single global MST.
    3. Density separation is computed from direct mutual-reachability distances
       between internal-node sets of cluster pairs, rather than shortest paths on
       a global tree.
    4. Noise is handled by excluding negative labels from cluster construction but
       retaining their contribution in the final weighted average through the
       original sample count.
    5. The CPU path now applies the same dense-matrix memory guard that already
       existed on the GPU path.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Input data.
    labels : ndarray, shape (n_samples,)
        Cluster labels. Negative labels are treated as noise. Non-negative labels
        are treated as valid clusters, including label 0.
    dist_function : str or callable
        Distance metric. Supported strings are: "euclidean", "manhattan",
        "chebyshev", "canberra", and "braycurtis". Custom callables remain
        supported through scipy.spatial.distance.cdist.
    use_gpu : bool, keyword-only, default=False
        Whether to use CUDA acceleration for supported dense square pairwise
        distance matrices.

    Returns
    -------
    float
        DBCV score in [-1, 1] in the ideal case, with noise penalization induced
        by weighting clusters by the original total sample count.
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    labels = np.ascontiguousarray(labels, dtype=np.int64)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array of shape (n_samples, n_features)")
    if labels.ndim != 1:
        raise ValueError("labels must be a 1D array")
    if X.shape[0] != labels.shape[0]:
        raise ValueError("X and labels must contain the same number of samples")
    if X.shape[0] == 0:
        return 0.0

    metric_name, _dist_callable, cdist_metric = _normalize_metric(dist_function)

    # Preserve HDBSCAN semantics: negative labels are noise, label 0 is a valid cluster.
    nonnoise_mask = labels >= 0
    n_total = float(labels.shape[0])
    if not np.any(nonnoise_mask):
        return 0.0

    X_valid = X[nonnoise_mask]
    labels_valid = labels[nonnoise_mask]

    dense_labels, original_cluster_ids = _remap_nonnoise_labels_to_dense(labels_valid)
    n_clusters = original_cluster_ids.size
    if n_clusters < 2:
        return 0.0

    ambient_dim = X.shape[1]
    core_distances = {}
    mst_internal_nodes = {}
    mst_internal_edges = {}
    density_sparseness = {}

    for cluster_id in range(n_clusters):
        X_cluster = np.ascontiguousarray(X_valid[dense_labels == cluster_id], dtype=np.float64)
        _distance_matrix, cluster_core_distances, mr_distances = _cluster_distance_and_core(
            X_cluster, cdist_metric, ambient_dim, use_gpu=use_gpu
        )
        internal_nodes, internal_edges = _internal_minimum_spanning_tree(mr_distances)
        core_distances[cluster_id] = cluster_core_distances
        mst_internal_nodes[cluster_id] = internal_nodes
        mst_internal_edges[cluster_id] = internal_edges
        density_sparseness[cluster_id] = _cluster_density_sparseness(internal_edges)

    density_sep = np.full((n_clusters, n_clusters), np.inf, dtype=np.float64)
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            sep = _density_separation(
                X_valid,
                dense_labels,
                i,
                j,
                mst_internal_nodes[i],
                mst_internal_nodes[j],
                core_distances[i],
                core_distances[j],
                cdist_metric,
            )
            density_sep[i, j] = sep
            density_sep[j, i] = sep

    result = 0.0
    for cluster_id in range(n_clusters):
        min_density_sep = float(np.min(density_sep[cluster_id]))
        sparseness = float(density_sparseness[cluster_id])
        denom = max(min_density_sep, sparseness)
        if not np.isfinite(min_density_sep) or denom == 0.0:
            cluster_validity = 0.0
        else:
            cluster_validity = (min_density_sep - sparseness) / denom

        cluster_size = float(np.sum(dense_labels == cluster_id))
        result += (cluster_size / n_total) * cluster_validity

    return float(result)
