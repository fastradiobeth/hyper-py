import numpy as np
import time
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from scipy.ndimage import convolve
from scipy.spatial import cKDTree
from astropy.table import Table
from hyper_py_playground.performance_timer import get_timer


def select_channel_map(map_struct):
    beam_dim_ref = map_struct["beam_dim"]
    pix_dim_ref = map_struct["pix_dim"]
    FWHM_pix = beam_dim_ref / pix_dim_ref
        
    return map_struct, FWHM_pix


# Cache for high-pass filter kernels to avoid repeated allocation
_kernel_cache = {}

def high_pass_filter(image, kernel_dim=9):
    """
    Apply high-pass filter using cached kernels for efficiency.
    """
    ny, nx = image.shape
    kdim = min(kernel_dim, ny, nx)
    if kdim % 2 == 0:
        kdim -= 1

    # Use cached kernel if available
    if kdim not in _kernel_cache:
        kernel = np.full((kdim, kdim), -1.0)
        kernel[kdim // 2, kdim // 2] = kdim**2 - 1.0
        _kernel_cache[kdim] = kernel
    
    kernel = _kernel_cache[kdim]
    filtered = convolve(image.astype(np.float64), kernel, mode='nearest')
    
    # Use np.maximum for in-place-like efficiency
    return np.maximum(filtered, 0.0)


def normalize_filtered_image(filtered):
    """
    Normalize filtered image to peak = 100, zeroing negative values.
    Optimized to minimize array copies.
    """
    # Use np.maximum to zero negatives without copy
    normalized = np.maximum(filtered, 0.0)
    
    # Normalize to peak = 100
    peak = np.nanmax(normalized)
    if peak > 0:
        normalized = normalized * (100.0 / peak)
    
    return normalized


# --- low values to get as many sources as possible in this first filter stage --- #
def estimate_rms(image, sigma_clip=3.0):
    """
    Estimate RMS using sigma-clipped statistics.
    Optimized with early return for empty arrays.
    """
    # Use boolean indexing directly - more efficient than creating intermediate
    mask = image > 0
    if not np.any(mask):
        return 0.0
    
    values = image[mask]
    _, _, sigma = sigma_clipped_stats(values, sigma=sigma_clip, maxiters=10, mask_value=0.0)
    
    return sigma


def detect_peaks(filtered_image, threshold, fwhm_pix, roundlim=(-1.0, 1.0), sharplim=(-1.0, 2.0)):
    finder = DAOStarFinder(
        threshold=threshold,
        fwhm=fwhm_pix,
        roundlo=roundlim[0], roundhi=roundlim[1],
        sharplo=sharplim[0], sharphi=sharplim[1]
    )
    return finder(filtered_image)


def filter_peaks(peaks_table, fwhm_pix, image_shape, min_dist_pix, aper_sup):
    """
    Filter peaks by removing those too close to image border and close neighbors.
    
    Uses cKDTree for O(n log n) spatial queries instead of O(n²) nested loops.
    Maintains exact compatibility with original algorithm by sorting pairs.
    """
    if min_dist_pix is None:
        min_dist_pix = fwhm_pix

    ny, nx = image_shape
    margin = int(fwhm_pix) * aper_sup
    
    # Step 1: remove peaks too close to image border
    valid = (
        (peaks_table['xcentroid'] > margin) &
        (peaks_table['xcentroid'] < nx - margin) &
        (peaks_table['ycentroid'] > margin) &
        (peaks_table['ycentroid'] < ny - margin)
    )
    peaks = peaks_table[valid]
    
    if len(peaks) == 0:
        return peaks
    
    # Step 2: remove close neighbors (keep brightest) using KD-tree
    coords = np.vstack([peaks['xcentroid'], peaks['ycentroid']]).T
    peak_values = np.array(peaks['peak'])
    
    # Build KD-tree for efficient spatial queries
    tree = cKDTree(coords)
    
    # Find all pairs within min_dist_pix
    pairs = tree.query_pairs(r=min_dist_pix, output_type='ndarray')
    
    # Sort pairs to match original algorithm's processing order
    # This ensures exact compatibility with the original O(n²) version
    if len(pairs) > 0:
        pairs = np.sort(pairs, axis=1)  # Ensure i < j for each pair
        order = np.lexsort((pairs[:, 1], pairs[:, 0]))  # Sort by i, then j
        pairs = pairs[order]
    
    # Mark peaks to keep
    keep = np.ones(len(peaks), dtype=bool)
    
    # Process pairs: for each pair, remove the fainter peak
    for i, j in pairs:
        if not keep[i] or not keep[j]:
            continue
        # Keep the brighter one (>= means keep i on tie, matching original)
        if peak_values[i] >= peak_values[j]:
            keep[j] = False
        else:
            keep[i] = False
                    
    return peaks[keep]


# --- save only sources above a sigma-clipped rms estimation in the maps, or use a manual value ---
def filter_by_snr(peaks_table, real_map, rms_real, snr_threshold):
    """
    Filter peaks by signal-to-noise ratio using vectorized operations.
    """
    if len(peaks_table) == 0:
        return peaks_table
    
    # Vectorized coordinate extraction and rounding
    x_coords = np.round(peaks_table['xcentroid']).astype(int)
    y_coords = np.round(peaks_table['ycentroid']).astype(int)
    
    ny, nx = real_map.shape
    
    # Vectorized bounds checking
    in_bounds = (
        (y_coords >= 0) & (y_coords < ny) &
        (x_coords >= 0) & (x_coords < nx)
    )
    
    # Initialize keep array with False for out-of-bounds
    keep = np.zeros(len(peaks_table), dtype=bool)
    
    # Get peak values for valid coordinates using advanced indexing
    valid_indices = np.where(in_bounds)[0]
    if len(valid_indices) > 0:
        peak_vals = real_map[y_coords[valid_indices], x_coords[valid_indices]]
        snr = peak_vals / rms_real if rms_real > 0 else np.zeros_like(peak_vals)
        keep[valid_indices] = snr >= snr_threshold
            
    return peaks_table[keep]


def filter_local_maximum(peaks_table, real_map, fwhm_pix, local_max_radius=0.0, local_max_steepness=0.0):
    """
    Filter sources that are not true local maxima within a larger radius,
    and/or don't show sufficient flux drop at the edge (sitting on extended emission).
    
    Parameters
    ----------
    peaks_table : astropy Table
        Table of detected peaks with 'xcentroid', 'ycentroid' columns
    real_map : 2D numpy array
        The actual (non-filtered) image for flux measurements
    fwhm_pix : float
        FWHM of the beam in pixels
    local_max_radius : float
        Search radius in units of FWHM. Source must be the brightest within this radius.
        Set to 0 to disable this check. Default: 0.0
    local_max_steepness : float
        Required fractional flux drop at the edge of local_max_radius.
        E.g., 0.5 means flux at edge must be <= 50% of peak flux.
        This rejects sources sitting on "pedestals" of extended emission.
        Set to 0 to disable this check. Default: 0.0
    
    Returns
    -------
    filtered_table : astropy Table
        Peaks table with non-local-maximum sources removed
    """
    if len(peaks_table) == 0:
        return peaks_table
    
    if local_max_radius <= 0 and local_max_steepness <= 0:
        return peaks_table
    
    search_radius_pix = local_max_radius * fwhm_pix if local_max_radius > 0 else fwhm_pix
    
    x_coords = np.round(peaks_table['xcentroid']).astype(int)
    y_coords = np.round(peaks_table['ycentroid']).astype(int)
    
    ny, nx = real_map.shape
    keep = np.ones(len(peaks_table), dtype=bool)
    
    # Get peak flux values
    peak_fluxes = np.zeros(len(peaks_table))
    for i, (x, y) in enumerate(zip(x_coords, y_coords)):
        if 0 <= y < ny and 0 <= x < nx:
            peak_fluxes[i] = real_map[y, x]
    
    # Build KD-tree if checking local maximum
    if local_max_radius > 0:
        coords = np.vstack([peaks_table['xcentroid'], peaks_table['ycentroid']]).T
        tree = cKDTree(coords)
    
    for i in range(len(peaks_table)):
        if peak_fluxes[i] <= 0:
            keep[i] = False
            continue
        
        cx, cy = x_coords[i], y_coords[i]
        
        # Check 1: Is this the brightest source within local_max_radius?
        if local_max_radius > 0:
            neighbors = tree.query_ball_point([peaks_table['xcentroid'][i], 
                                                peaks_table['ycentroid'][i]], 
                                               r=search_radius_pix)
            for j in neighbors:
                if i != j and peak_fluxes[j] > peak_fluxes[i]:
                    keep[i] = False
                    break
        
        if not keep[i]:
            continue
        
        # Check 2: Does flux drop sufficiently at the edge? (steepness check)
        if local_max_steepness > 0:
            # Sample flux at the edge of the search radius (8 points around the circle)
            n_samples = 8
            angles = np.linspace(0, 2*np.pi, n_samples, endpoint=False)
            edge_fluxes = []
            
            for angle in angles:
                ex = int(round(cx + search_radius_pix * np.cos(angle)))
                ey = int(round(cy + search_radius_pix * np.sin(angle)))
                
                if 0 <= ey < ny and 0 <= ex < nx:
                    edge_fluxes.append(real_map[ey, ex])
            
            if len(edge_fluxes) > 0:
                # Use median edge flux to be robust against noise
                median_edge_flux = np.median(edge_fluxes)
                
                # Required: edge_flux <= (1 - steepness) * peak_flux
                # E.g., steepness=0.5 means edge must be <= 50% of peak
                max_allowed_edge = (1.0 - local_max_steepness) * peak_fluxes[i]
                
                if median_edge_flux > max_allowed_edge:
                    keep[i] = False
    
    return peaks_table[keep]


def detect_sources(map_struct_list, dist_limit_arcsec, real_map, rms_real, snr_threshold, roundlim, sharplim, config):
    t_detect_start = time.time()
    timer = get_timer()
    
    map_struct, FWHM_pix = select_channel_map(map_struct_list)
    image = map_struct["map"]
    pix_dim_ref = map_struct["pix_dim"]
    beam_dim_ref = map_struct["beam_dim"]
    aper_sup=config.get("photometry", "aper_sup")

    my_dist_limit_arcsec = beam_dim_ref if dist_limit_arcsec == 0 else dist_limit_arcsec
    dist_limit_pix = my_dist_limit_arcsec / pix_dim_ref

    # Get local maximum filter parameters
    local_max_radius = config.get("detection", "local_max_radius", 0.0)
    local_max_steepness = config.get("detection", "local_max_steepness", 0.0)

    # --- identify multiple peaks in filtered image and save good peaks with real snr threshold --- #
    filtered = high_pass_filter(image)
    norm_filtered = normalize_filtered_image(filtered)
        
    filtered_rms_detect = estimate_rms(norm_filtered)
    filtered_threshold = 2. * filtered_rms_detect
        
    peaks = detect_peaks(norm_filtered, filtered_threshold, FWHM_pix, roundlim=roundlim, sharplim=sharplim)
    good_peaks = filter_peaks(peaks, FWHM_pix, image.shape, dist_limit_pix, aper_sup)
    snr_filtered = filter_by_snr(good_peaks, real_map, rms_real, snr_threshold)
    
    # Apply local maximum filter (removes sources on halos/extended emission)
    final_sources = filter_local_maximum(snr_filtered, real_map, FWHM_pix, 
                                          local_max_radius, local_max_steepness)

    if timer:
        timer.log_timing("detection.py", 123, 145, f"detect_sources (found {len(final_sources)} sources)", 
                       time.time() - t_detect_start)
    
    return final_sources
