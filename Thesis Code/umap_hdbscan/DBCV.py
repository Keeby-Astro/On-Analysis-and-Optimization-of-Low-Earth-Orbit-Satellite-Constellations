import numpy as np
from scipy.spatial.distance import euclidean, cdist
from scipy.sparse.csgraph import minimum_spanning_tree, dijkstra
from numba import njit, prange


def _has_numba_cuda():
    try:
        from numba import cuda  # noqa: F401
        return True
    except Exception:
        return False


def _cdist_optional_numba_cuda(X, metric, use_gpu):
    """Compute pairwise distances, optionally using Numba CUDA on GPU.

    Notes:
        - This only accelerates the pairwise distance computation.
        - Downstream MST + Dijkstra are still computed on CPU (SciPy).
        - Supported GPU metrics: euclidean, cityblock, chebyshev, canberra, braycurtis.
    """
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
            f"Unsupported GPU metric '{metric}'. Supported: euclidean, cityblock, chebyshev, canberra, braycurtis"
        )

    n_samples = int(X.shape[0])
    if n_samples <= 0:
        return np.empty((0, 0), dtype=np.float64)

    # DBCV requires a full n x n distance matrix (O(n^2) memory/time).
    # Guard against pathological sizes early with a clear error.
    bytes_needed = n_samples * n_samples * np.dtype(np.float64).itemsize
    # Quick CPU-side sanity bound (prevents accidental 10+ TB allocations).
    if bytes_needed > 8 * 1024**3:
        raise MemoryError(
            f"DBCV needs an {n_samples}x{n_samples} distance matrix (~{bytes_needed/1024**3:.1f} GiB). "
            "This is not practical. Consider sampling/downsizing, or compute DBCV on a subset."
        )

    # GPU memory sanity check (best-effort)
    try:
        free_bytes, _ = cuda.current_context().get_memory_info()
        if bytes_needed > 0.8 * free_bytes:
            raise MemoryError(
                f"Not enough free GPU memory for {n_samples}x{n_samples} distances (~{bytes_needed/1024**3:.1f} GiB). "
                f"Free GPU memory ~{free_bytes/1024**3:.1f} GiB. Use use_gpu=False or reduce n."
            )
    except Exception:
        # If memory info isn't available, proceed and let allocation fail naturally.
        pass

    X_dev = cuda.to_device(X)
    D_dev = cuda.device_array((n_samples, n_samples), dtype=np.float64)

    # Adaptive block sizes:
    # - For small n, use smaller blocks to increase total blocks (reduces low-occupancy warnings).
    # - For large n, use more threads per block for throughput.
    if n_samples < 128:
        threadsperblock = (8, 8)
    elif n_samples < 1024:
        threadsperblock = (16, 16)
    else:
        threadsperblock = (32, 8)  # 256 threads/block

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
        # Only compute upper triangle and mirror; halves the math and global reads.
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
    # If numba.cuda isn't available at import time, we keep CPU-only functionality.
    pass

# APT SCORE FUNCTIONS
"""
These functions compute the all-points-core-distance for a given point
based on its neighbors. Each function implements a different distance metric.
The all-points-core-distance is used later in computing the mutual reachability
distances between points for cluster validity assessment.
"""
@njit(fastmath=True, parallel=True, cache=True)
def _aptscoredist_euclidean(point, neighbors, d):
    """
    Compute the all-points-core-distance for a given point using the Euclidean metric.
    
    Parameters:
        point (ndarray): The target point.
        neighbors (ndarray): Array of neighbor points.
        d (int): Dimensionality of the data.
    
    Returns:
        float: The computed core distance.
    """
    n_neighbors = neighbors.shape[0]
    s_sum = 0.0
    count = 0

    # Iterate over each neighbor to accumulate contributions
    for i in prange(n_neighbors):
        dist_sq = 0.0
        for j in range(d):
            diff = point[j] - neighbors[i, j]
            dist_sq += diff * diff
        dist = np.sqrt(dist_sq)
        
        # Avoid self-distance or near-zero distances
        if dist > 1e-12:
            s_sum += (1.0 / dist) ** d
            count += 1
    if count == 0:
        return 0.0
    return (s_sum / count) ** (-1.0 / d)

@njit(fastmath=True, parallel=True, cache=True)
def _aptscoredist_manhattan(point, neighbors, d):
    """
    Compute the all-points-core-distance using the Manhattan (cityblock) metric.
    
    Parameters:
        point (ndarray): The target point.
        neighbors (ndarray): Array of neighbor points.
        d (int): Dimensionality of the data.
    
    Returns:
        float: The computed core distance.
    """
    n_neighbors = neighbors.shape[0]
    s_sum = 0.0
    count = 0
    for i in prange(n_neighbors):
        dist = 0.0
        for j in range(d):
            dist += abs(point[j] - neighbors[i, j])
        if dist > 1e-12:
            s_sum += (1.0 / dist) ** d
            count += 1
    if count == 0:
        return 0.0
    return (s_sum / count) ** (-1.0 / d)

@njit(fastmath=True, parallel=True, cache=True)
def _aptscoredist_chebyshev(point, neighbors, d):
    """
    Compute the all-points-core-distance using the Chebyshev metric.
    
    Parameters:
        point (ndarray): The target point.
        neighbors (ndarray): Array of neighbor points.
        d (int): Dimensionality of the data.
    
    Returns:
        float: The computed core distance.
    """
    n_neighbors = neighbors.shape[0]
    s_sum = 0.0
    count = 0
    for i in prange(n_neighbors):
        dist = 0.0

        # Chebyshev distance is the maximum absolute difference over dimensions
        for j in range(d):
            diff = abs(point[j] - neighbors[i, j])
            if diff > dist:
                dist = diff
        if dist > 1e-12:
            s_sum += (1.0 / dist) ** d
            count += 1
    if count == 0:
        return 0.0
    return (s_sum / count) ** (-1.0 / d)

@njit(fastmath=True, parallel=True, cache=True)
def _aptscoredist_canberra(point, neighbors, d):
    """
    Compute the all-points-core-distance using the Canberra metric.
    
    Parameters:
        point (ndarray): The target point.
        neighbors (ndarray): Array of neighbor points.
        d (int): Dimensionality of the data.
    
    Returns:
        float: The computed core distance.
    """
    n_neighbors = neighbors.shape[0]
    s_sum = 0.0
    count = 0
    for i in prange(n_neighbors):
        dist = 0.0
        for j in range(d):
            abs_diff = abs(point[j] - neighbors[i, j])
            abs_sum = abs(point[j]) + abs(neighbors[i, j])
            if abs_sum != 0:
                dist += abs_diff / abs_sum
        if dist > 1e-12:
            s_sum += (1.0 / dist) ** d
            count += 1
    if count == 0:
        return 0.0
    return (s_sum / count) ** (-1.0 / d)

@njit(fastmath=True, parallel=True, cache=True)
def _aptscoredist_braycurtis(point, neighbors, d):
    """
    Compute the all-points-core-distance using the Bray-Curtis metric.
    
    Parameters:
        point (ndarray): The target point.
        neighbors (ndarray): Array of neighbor points.
        d (int): Dimensionality of the data.
    
    Returns:
        float: The computed core distance.
    """
    n_neighbors = neighbors.shape[0]
    s_sum = 0.0
    count = 0
    for i in prange(n_neighbors):
        num = 0.0
        den = 0.0
        for j in range(d):
            diff = abs(point[j] - neighbors[i, j])
            num += diff
            den += abs(point[j] + neighbors[i, j])
        if den == 0:
            dist = 0.0
        else:
            dist = num / den
        if dist > 1e-12:
            s_sum += (1.0 / dist) ** d
            count += 1
    if count == 0:
        return 0.0
    return (s_sum / count) ** (-1.0 / d)

# Mapping of metric names to the corresponding all-points-core-distance function.
APTSCORE_FUNCTIONS = {"euclidean": _aptscoredist_euclidean,
                      "manhattan": _aptscoredist_manhattan,
                      "chebyshev": _aptscoredist_chebyshev,
                      "canberra": _aptscoredist_canberra,
                      "braycurtis": _aptscoredist_braycurtis}


@njit(fastmath=True, parallel=True, cache=True)
def _core_distances_cluster_euclidean(X_cluster):
    m, d = X_cluster.shape
    out = np.empty(m, dtype=np.float64)
    for i in prange(m):
        out[i] = _aptscoredist_euclidean(X_cluster[i], X_cluster, d)
    return out


@njit(fastmath=True, parallel=True, cache=True)
def _core_distances_cluster_manhattan(X_cluster):
    m, d = X_cluster.shape
    out = np.empty(m, dtype=np.float64)
    for i in prange(m):
        out[i] = _aptscoredist_manhattan(X_cluster[i], X_cluster, d)
    return out


@njit(fastmath=True, parallel=True, cache=True)
def _core_distances_cluster_chebyshev(X_cluster):
    m, d = X_cluster.shape
    out = np.empty(m, dtype=np.float64)
    for i in prange(m):
        out[i] = _aptscoredist_chebyshev(X_cluster[i], X_cluster, d)
    return out


@njit(fastmath=True, parallel=True, cache=True)
def _core_distances_cluster_canberra(X_cluster):
    m, d = X_cluster.shape
    out = np.empty(m, dtype=np.float64)
    for i in prange(m):
        out[i] = _aptscoredist_canberra(X_cluster[i], X_cluster, d)
    return out


@njit(fastmath=True, parallel=True, cache=True)
def _core_distances_cluster_braycurtis(X_cluster):
    m, d = X_cluster.shape
    out = np.empty(m, dtype=np.float64)
    for i in prange(m):
        out[i] = _aptscoredist_braycurtis(X_cluster[i], X_cluster, d)
    return out


CORE_CLUSTER_FUNCTIONS = {
    "euclidean": _core_distances_cluster_euclidean,
    "manhattan": _core_distances_cluster_manhattan,
    "chebyshev": _core_distances_cluster_chebyshev,
    "canberra": _core_distances_cluster_canberra,
    "braycurtis": _core_distances_cluster_braycurtis,
}

# Mapping of our metric names to the metric names expected by scipy's cdist.
PAIRWISE_METRIC_MAP = {"euclidean": "euclidean",
                       "manhattan": "cityblock",
                       "chebyshev": "chebyshev",
                       "canberra": "canberra",
                       "braycurtis": "braycurtis"}

# DISTANCE FUNCTIONS
"""
These helper functions provide alternative implementations for non-Euclidean
distance calculations between two points.
"""
def _distance_manhattan(x, y):
    """
    Compute the Manhattan (cityblock) distance between two points.
    
    Parameters:
        x (ndarray): First point.
        y (ndarray): Second point.
    
    Returns:
        float: The Manhattan distance.
    """
    return np.sum(np.abs(x - y))

def _distance_chebyshev(x, y):
    """
    Compute the Chebyshev distance between two points.
    
    Parameters:
        x (ndarray): First point.
        y (ndarray): Second point.
    
    Returns:
        float: The Chebyshev distance.
    """
    return np.max(np.abs(x - y))

def _distance_canberra(x, y):
    """
    Compute the Canberra distance between two points.
    
    Parameters:
        x (ndarray): First point.
        y (ndarray): Second point.
    
    Returns:
        float: The Canberra distance.
    """
    s = 0.0
    for i in range(x.shape[0]):
        denom = abs(x[i]) + abs(y[i])
        if denom != 0:
            s += abs(x[i] - y[i]) / denom
    return s

def _distance_braycurtis(x, y):
    """
    Compute the Bray-Curtis distance between two points.
    
    Parameters:
        x (ndarray): First point.
        y (ndarray): Second point.
    
    Returns:
        float: The Bray-Curtis distance.
    """
    num = np.sum(np.abs(x - y))
    den = np.sum(np.abs(x + y))
    if den == 0:
        return 0.0
    return num / den

# Mapping of metric names to their corresponding distance functions.
DISTANCE_FUNCTIONS = {"euclidean": euclidean,
                      "manhattan": _distance_manhattan,
                      "chebyshev": _distance_chebyshev,
                      "canberra": _distance_canberra,
                      "braycurtis": _distance_braycurtis}

# CORE DISTANCE CALCULATION
def _aptscoredist(point, neighbors, dist_function):
    """
    Calculate the all-points-core-distance for a point given its neighbors
    and a specified distance function. This is used as an intermediary step
    in computing mutual reachability distances.

    Parameters:
        point (ndarray): The target point.
        neighbors (ndarray): Array of neighboring points (typically within the same cluster).
        dist_function (function): The distance function to use.

    Returns:
        float: The computed core distance.
    """
    d = point.shape[0]
    metric_name = None

    # Determine which metric to use based on the distance function
    if dist_function == euclidean:
        metric_name = "euclidean"
    elif dist_function.__name__ in ["_distance_manhattan", "manhattan"]:
        metric_name = "manhattan"
    elif dist_function.__name__ in ["_distance_chebyshev", "chebyshev"]:
        metric_name = "chebyshev"
    elif dist_function.__name__ in ["_distance_canberra", "canberra"]:
        metric_name = "canberra"
    elif dist_function.__name__ in ["_distance_braycurtis", "braycurtis"]:
        metric_name = "braycurtis"
    
    if metric_name is not None:
        func = APTSCORE_FUNCTIONS[metric_name]
        return func(point, neighbors, d)
    else:
        # Fallback: use standard Euclidean distances if metric is unknown
        neighbors = np.ascontiguousarray(neighbors)
        dists = np.linalg.norm(neighbors - point, axis=1)
        dists = dists[dists > 1e-12]  # filter out self-distance
        if dists.size == 0:
            return 0.0
        numerator = np.sum((1.0 / dists) ** d)
        return (numerator / dists.size) ** (-1.0 / d)

# MUTUAL REACHABILITY DISTANCE
def _mutual_reachability_dist(point_i, point_j, members_i, members_j, dist_function):
    """
    Compute the mutual reachability distance between two points.
    This distance takes into account the core distances of the two points.

    Parameters:
        point_i (ndarray): First point.
        point_j (ndarray): Second point.
        members_i (ndarray): Neighbors (or cluster members) for point_i.
        members_j (ndarray): Neighbors (or cluster members) for point_j.
        dist_function (function): The distance function to compute direct distances.

    Returns:
        float: The mutual reachability distance.
    """
    core_i = _aptscoredist(point_i, members_i, dist_function)
    core_j = _aptscoredist(point_j, members_j, dist_function)
    d = dist_function(point_i, point_j)
    return np.max([core_i, core_j, d])

def _mutual_reachability_dist_graph(X, labels, dist_function):
    """
    Build a full graph of mutual reachability distances for all points.
    For each pair of points, the mutual reachability distance is computed.
    
    Parameters:
        X (ndarray): Data points array.
        labels (ndarray): Cluster labels for each data point.
        dist_function (function): Distance function to use.
    
    Returns:
        ndarray: A symmetric matrix of mutual reachability distances.
    """
    n_samples = X.shape[0]
    d = X.shape[1]
    metric_name = None

    # Identify the metric based on the provided distance function
    if dist_function == euclidean:
        metric_name = "euclidean"
    elif dist_function.__name__ in ["_distance_manhattan", "manhattan"]:
        metric_name = "manhattan"
    elif dist_function.__name__ in ["_distance_chebyshev", "chebyshev"]:
        metric_name = "chebyshev"
    elif dist_function.__name__ in ["_distance_canberra", "canberra"]:
        metric_name = "canberra"
    elif dist_function.__name__ in ["_distance_braycurtis", "braycurtis"]:
        metric_name = "braycurtis"
    
    if metric_name is not None:
        # Precompute cluster indices
        unique_clusters = np.unique(labels)
        clusters = {cl: np.where(labels == cl)[0] for cl in unique_clusters}
        core = np.empty(n_samples, dtype=np.float64)

        # Compute core distances in larger Numba batches (per cluster) to reduce Python overhead
        core_cluster_func = CORE_CLUSTER_FUNCTIONS[metric_name]
        for cl in unique_clusters:
            idx = clusters[cl]
            Xc = np.ascontiguousarray(X[idx], dtype=np.float64)
            core[idx] = core_cluster_func(Xc)

        X_cont = np.ascontiguousarray(X, dtype=np.float64)

        pairwise_metric = PAIRWISE_METRIC_MAP[metric_name]
        D = _cdist_optional_numba_cuda(X_cont, pairwise_metric, use_gpu=False)
        core_i = core.reshape(-1, 1)
        core_j = core.reshape(1, -1)
        graph = np.maximum(np.maximum(core_i, core_j), D)
        return graph
    else:
        # Fallback for an unknown metric: compute pairwise distances with full loops.
        graph = np.empty((n_samples, n_samples), dtype=np.float64)
        cluster_cache = {}
        for i in range(n_samples):
            for j in range(n_samples):
                members_i = _get_label_members(X, labels, labels[i], cluster_cache)
                members_j = _get_label_members(X, labels, labels[j], cluster_cache)
                graph[i, j] = _mutual_reachability_dist(X[i], X[j], members_i, members_j, dist_function)
        return graph

# MINIMUM SPANNING TREE (MST)
def _compute_MST(dist_graph):
    """
    Compute the symmetric minimum spanning tree (MST) for a given distance graph.
    
    Parameters:
        dist_graph (ndarray): A symmetric matrix of pairwise distances.
    
    Returns:
        ndarray: The computed MST as a symmetric matrix.
    """
    mst = minimum_spanning_tree(dist_graph).toarray()
    return mst + mst.T

# CLUSTER MEMBER RETRIEVAL
def _get_label_members(X, labels, cluster, cache=None):
    """
    Retrieve all data points that belong to a given cluster.
    
    Parameters:
        X (ndarray): Data points.
        labels (ndarray): Array of cluster labels.
        cluster (int): The cluster label to retrieve.
        cache (dict, optional): Cache to store/retrieve computed cluster members.
    
    Returns:
        ndarray: Subset of X corresponding to the specified cluster.
    """
    if cache is not None and cluster in cache:
        return cache[cluster]
    indices = np.where(labels == cluster)[0]
    members = X[indices]
    if cache is not None:
        cache[cluster] = members
    return members

# INTERNAL INDICES DETERMINATION
def _get_internal_indices(cluster_MST):
    """
    Identify the internal indices of a cluster's MST.
    Internal indices are those points with more than one connection.
    
    Parameters:
        cluster_MST (ndarray): The MST restricted to a cluster.
    
    Returns:
        ndarray: Array of indices considered internal.
    """
    degrees = np.count_nonzero(cluster_MST, axis=1)
    internal = np.where(degrees > 1)[0]
    if internal.size == 0:
        # Fallback: if no internal nodes, consider all nodes as internal.
        return np.arange(cluster_MST.shape[0])
    return internal

# CLUSTER DENSITY EVALUATION
def _cluster_density_sparseness(MST, labels, cluster):
    """
    Compute the density sparseness of a cluster. This is defined as the maximum
    edge weight within the cluster's MST (using only internal points).
    
    Parameters:
        MST (ndarray): The full MST of the data.
        labels (ndarray): Cluster labels for each data point.
        cluster (int): The cluster label to evaluate.
    
    Returns:
        float: The density sparseness value.
    """
    indices = np.where(labels == cluster)[0]
    cluster_MST = MST[np.ix_(indices, indices)]
    internal = _get_internal_indices(cluster_MST)
    if internal.size < 2:
        return np.max(cluster_MST)
    internal_MST = cluster_MST[np.ix_(internal, internal)]
    return np.max(internal_MST)

def _cluster_density_separation(MST, labels, cluster_i, cluster_j):
    """
    Compute the density separation between two clusters. This is the minimum
    distance between internal points of the two clusters based on the MST.
    
    Parameters:
        MST (ndarray): The full MST of the data.
        labels (ndarray): Cluster labels for each data point.
        cluster_i (int): First cluster label.
        cluster_j (int): Second cluster label.
    
    Returns:
        float: The density separation between the two clusters.
    """
    indices_i = np.where(labels == cluster_i)[0]
    indices_j = np.where(labels == cluster_j)[0]
    
    MST_i = MST[np.ix_(indices_i, indices_i)]
    MST_j = MST[np.ix_(indices_j, indices_j)]
    internal_i = _get_internal_indices(MST_i)
    internal_j = _get_internal_indices(MST_j)
    if internal_i.size == 0:
        internal_i = np.arange(len(indices_i))
    if internal_j.size == 0:
        internal_j = np.arange(len(indices_j))
    global_internal_i = indices_i[internal_i]
    global_internal_j = indices_j[internal_j]
    
    # Compute shortest paths between internal nodes of the two clusters
    shortest_paths = dijkstra(MST, indices=global_internal_i)
    relevant_paths = shortest_paths[:, global_internal_j]
    return np.min(relevant_paths)

def _cluster_validity_index(MST, labels, cluster):
    """
    Calculate the validity index for a single cluster. This index is based on
    the difference between the minimum separation and the sparseness of the cluster.
    
    Parameters:
        MST (ndarray): The full MST of the data.
        labels (ndarray): Cluster labels for each data point.
        cluster (int): The cluster label to evaluate.
    
    Returns:
        float: The validity index for the cluster.
    """
    min_density_separation = np.inf
    
    # Iterate over all other clusters to find the minimum separation
    for cl in np.unique(labels):
        if cl != cluster:
            sep = _cluster_density_separation(MST, labels, cluster, cl)
            if sep < min_density_separation:
                min_density_separation = sep
    sparseness = _cluster_density_sparseness(MST, labels, cluster)
    numerator = min_density_separation - sparseness
    denominator = np.max([min_density_separation, sparseness])
    return numerator / denominator

def _clustering_validity_index(MST, labels):
    """
    Calculate the overall clustering validity index by weighting each cluster's
    validity by its relative size.
    
    Parameters:
        MST (ndarray): The full MST of the data.
        labels (ndarray): Cluster labels for each data point.
    
    Returns:
        float: The overall clustering validity index.
    """
    n_samples = len(labels)
    validity_index = 0.0

    # Weight validity index for each cluster by its fraction of points
    for cl in np.unique(labels):
        fraction = np.sum(labels == cl) / float(n_samples)
        validity_index += fraction * _cluster_validity_index(MST, labels, cl)
    return validity_index


def _clustering_validity_index_fast(MST, labels):
    """Compute the clustering validity index with fewer Dijkstra calls.

    This is algebraically equivalent to `_clustering_validity_index`, but computes
    each cluster's minimum density separation by running `dijkstra` once per cluster
    (instead of once per cluster-pair).
    """
    unique_clusters = np.unique(labels)
    n_samples = len(labels)

    # Precompute global internal-node indices per cluster
    internal_globals = {}
    cluster_indices = {}
    for cl in unique_clusters:
        idx = np.where(labels == cl)[0]
        cluster_indices[cl] = idx
        cluster_MST = MST[np.ix_(idx, idx)]
        internal_local = _get_internal_indices(cluster_MST)
        internal_globals[cl] = idx[internal_local]

    validity_index = 0.0
    for cl in unique_clusters:
        idx = cluster_indices[cl]
        fraction = idx.size / float(n_samples)

        # sparseness
        cluster_MST = MST[np.ix_(idx, idx)]
        internal_local = _get_internal_indices(cluster_MST)
        if internal_local.size < 2:
            sparseness = np.max(cluster_MST)
        else:
            internal_MST = cluster_MST[np.ix_(internal_local, internal_local)]
            sparseness = np.max(internal_MST)

        # min separation (one Dijkstra per cluster)
        sources = internal_globals[cl]
        shortest_paths = dijkstra(MST, indices=sources)
        min_sep = np.inf
        for other in unique_clusters:
            if other == cl:
                continue
            targets = internal_globals[other]
            sep = np.min(shortest_paths[:, targets])
            if sep < min_sep:
                min_sep = sep

        numerator = min_sep - sparseness
        denominator = np.max([min_sep, sparseness])
        validity_index += fraction * (numerator / denominator)

    return validity_index

# MAIN DBCV FUNCTION
def DBCV(X, labels, dist_function, *, use_gpu=False):
    """
    Compute the Density-Based Clustering Validation (DBCV) index for a given
    clustering of the data. The DBCV index measures the quality of the clustering
    based on the density properties of the clusters.
    
    Parameters:
        X (ndarray): Array of data points.
        labels (ndarray): Cluster labels for each data point. Negative labels
                          are reassigned to 0.
        dist_function (str or function): Distance metric to be used. If a string,
                          it must be one of the keys in DISTANCE_FUNCTIONS.
    
    Returns:
        float: The computed DBCV index (adjusted by the coverage of valid clusters).
    """
    # Ensure input arrays are contiguous and have correct dtypes
    X = np.ascontiguousarray(X, dtype=np.float64)
    labels = np.ascontiguousarray(labels, dtype=np.int64)
    
    # Convert negative labels to 0 (considered noise or unassigned)
    labels = np.where(labels < 0, 0, labels)
    original_labels = labels.copy()
    
    # Handle clusters with only one member by reassigning them to noise (label 0)
    counts = np.bincount(labels)
    for cl, cnt in enumerate(counts):
        if cnt == 1:
            labels[labels == cl] = 0
    
    valid_indices = labels != 0
    if np.sum(valid_indices) == 0:
        return 0.0
    X_valid = X[valid_indices]
    labels_valid = labels[valid_indices]
    
    # At least two clusters are required to compute a meaningful index
    if np.unique(labels_valid).size < 2:
        return 0.0
    
    # Allow the user to specify the distance function as a string
    if isinstance(dist_function, str):
        if dist_function in DISTANCE_FUNCTIONS:
            dist_function = DISTANCE_FUNCTIONS[dist_function]
        else:
            raise ValueError("Unsupported distance metric: " + dist_function)
    
    # Build the mutual reachability distance graph
    # (optional GPU acceleration applies only to pairwise distances; MST+Dijkstra stay on CPU)
    metric_name = None
    if dist_function == euclidean:
        metric_name = "euclidean"
    elif getattr(dist_function, "__name__", "") in ["_distance_manhattan", "manhattan"]:
        metric_name = "manhattan"
    elif getattr(dist_function, "__name__", "") in ["_distance_chebyshev", "chebyshev"]:
        metric_name = "chebyshev"
    elif getattr(dist_function, "__name__", "") in ["_distance_canberra", "canberra"]:
        metric_name = "canberra"
    elif getattr(dist_function, "__name__", "") in ["_distance_braycurtis", "braycurtis"]:
        metric_name = "braycurtis"

    if metric_name is None:
        graph = _mutual_reachability_dist_graph(X_valid, labels_valid, dist_function)
    else:
        # Inline the fast path so we can optionally use GPU for the cdist step
        n_samples = X_valid.shape[0]
        unique_clusters = np.unique(labels_valid)
        clusters = {cl: np.where(labels_valid == cl)[0] for cl in unique_clusters}

        core = np.empty(n_samples, dtype=np.float64)
        core_cluster_func = CORE_CLUSTER_FUNCTIONS[metric_name]
        for cl in unique_clusters:
            idx = clusters[cl]
            Xc = np.ascontiguousarray(X_valid[idx], dtype=np.float64)
            core[idx] = core_cluster_func(Xc)

        X_cont = np.ascontiguousarray(X_valid, dtype=np.float64)
        pairwise_metric = PAIRWISE_METRIC_MAP[metric_name]
        D = _cdist_optional_numba_cuda(X_cont, pairwise_metric, use_gpu=use_gpu)
        core_i = core.reshape(-1, 1)
        core_j = core.reshape(1, -1)
        graph = np.maximum(np.maximum(core_i, core_j), D)

    # Compute the Minimum Spanning Tree (MST) of the graph
    mst = _compute_MST(graph)
    # Evaluate the clustering validity using the MST
    validity = _clustering_validity_index_fast(mst, labels_valid)
    
    # Adjust the index by the fraction of points that are in valid clusters
    coverage = np.sum(valid_indices) / float(len(original_labels))
    return validity * coverage