import numpy as np
from scipy.spatial import cKDTree

def group_sources(xcen, ycen, pix_dim, beam_dim, aper_sup):
    '''
    Groups sources based on proximity within the beam scale, ensuring no duplicate groups and transitive merging.
    
    Optimized using cKDTree for O(n log n) spatial queries and Union-Find with path compression
    and union by rank for efficient group merging.

    Parameters:
        xcen, ycen: arrays of source positions in pixels
        pix_dim: pixel scale (arcsec)
        beam_dim: beam size (arcsec)
        aper_sup: aperture scaling factor

    Returns:
        start_group: 1 for blended sources, 0 otherwise
        common_group: 2D array of group membership (list of sources per group)
        deblend: number of neighbors (for Gaussian deblending)
    '''
    n = len(xcen)
    if n == 0:
        return np.array([], dtype=int), np.array([]).reshape(0, 0), np.array([], dtype=int)
    
    xcen = np.asarray(xcen)
    ycen = np.asarray(ycen)

    max_dist = beam_dim * aper_sup * 2.0
    max_dist_pix = max_dist / pix_dim
        
    start_group = np.zeros(n, dtype=int)
    common_group = -1 * np.ones((n, n), dtype=int)
    deblend = np.zeros(n, dtype=int)
    
    # Union-Find with path compression and union by rank
    parent = np.arange(n, dtype=int)
    rank = np.zeros(n, dtype=int)
    
    def find(i):
        # Path compression: make every node point directly to root
        root = i
        while parent[root] != root:
            root = parent[root]
        # Compress path
        while parent[i] != root:
            next_i = parent[i]
            parent[i] = root
            i = next_i
        return root
    
    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            # Union by rank
            if rank[root_i] < rank[root_j]:
                parent[root_i] = root_j
            elif rank[root_i] > rank[root_j]:
                parent[root_j] = root_i
            else:
                parent[root_j] = root_i
                rank[root_i] += 1

    # Build KD-tree for efficient spatial queries
    coords = np.column_stack([xcen, ycen])
    tree = cKDTree(coords)
    
    # Find all pairs within max_dist_pix using KD-tree query_pairs
    # This is O(n log n + k) where k is the number of pairs, much faster than O(n²)
    pairs = tree.query_pairs(r=max_dist_pix, output_type='ndarray')
    
    # Union all close pairs
    for i, j in pairs:
        union(i, j)

    # Flatten all group pointers (ensure all point to root)
    roots = np.array([find(i) for i in range(n)])
    
    # Group sources by their root - vectorized approach
    unique_roots = np.unique(roots)
    root_to_members = {root: np.where(roots == root)[0] for root in unique_roots}
    
    # Assign group info for each source
    for i in range(n):
        group_members = root_to_members[roots[i]]
        common_group[i, :len(group_members)] = group_members
        deblend[i] = len(group_members) - 1
        if len(group_members) > 1:
            start_group[i] = 1

    return start_group, common_group, deblend