import os
import sys
from pathlib import Path
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import warnings

# Thread limits are set by run_hyper_playground.py BEFORE this module is imported
# Do NOT set them here - they must be set before numpy is imported to take effect

import numpy as np
from astropy.io import ascii, fits
from astropy.table import vstack
from astropy.wcs import WCS

from hyper_py_playground.single_map import main as single_map
from hyper_py_playground.config import HyperConfig
from hyper_py_playground.logger import setup_logger
from hyper_py_playground.create_background_slices import create_background_cubes
from hyper_py_playground.performance_timer import init_timer, get_timer
from .extract_cubes import extract_maps_from_cube
from contextlib import contextmanager

# IMPORTANT: Use 'spawn' to avoid fork deadlocks with sklearn/lmfit
# 'fork' causes deadlocks when worker processes have complex library state
# 'spawn' is safer but slower (each worker re-imports everything)
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass  # Already set


@contextmanager
def _dummy_context():
    """Dummy context manager when timer is not available."""
    yield

def start_hyper(cfg_path):
    # Start overall timing
    start_time_total = time.time()
    
    # === Load config ===
    os.chdir(os.path.dirname(__file__))

    config_path = cfg_path if not None else "hyper_config.yaml"
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    t0 = time.time()
    cfg = HyperConfig(config_path)
    
    # --- Initialize paths --- #
    # - common - #
    paths = cfg.get("paths")
    dir_root = paths["output"]["dir_root"]
    
    # # - input - #
    dir_maps = paths["input"]["dir_maps"]
    dir_slices_out = Path(dir_root, cfg.get("control")["dir_datacube_slices"])
    base_table_name = cfg.get("files", "file_table_base")
    map_names = cfg.get("files", "file_map_name")
    datacube = cfg.get("control", "datacube", False)
    fix_min_box = cfg.get("background", "fix_min_box", 3)     # minimum padding value (multiple of FWHM)
    convert_mjy=cfg.get("units", "convert_mJy")
    
    # If it's a path to a .txt file, read it #
    if isinstance(map_names, str) and map_names.endswith('.txt'):
        map_list_path = os.path.join(dir_maps, map_names)
        with open(map_list_path, 'r') as f:
            map_names = [line.strip() for line in f if line.strip()]
    # If it's a single string but not a .txt, wrap it in a list
    elif isinstance(map_names, str):
        map_names = [map_names]
        
    if datacube:
        map_names, cube_header = extract_maps_from_cube(map_names, dir_slices_out, dir_maps)
        background_slices = []
        slice_cutout_header = []
    
    # - output - #
    output_dir = paths["output"]["dir_table_out"]
    
    # --- Set up logging for warnings --- #
    dir_log = paths["output"]["dir_log_out"]
    file_log = cfg.get("files", "file_log_name")
    log_path = os.path.join(dir_root, dir_log, file_log)
    
    # Ensure the log directory exists
    log_path_dir = os.path.join(dir_root, dir_log)
    os.makedirs(log_path_dir, exist_ok=True)

    logger, logger_file_only = setup_logger(log_path, logger_name="HyperLogger", overwrite=True)
    
    # Initialize performance timer
    timing_dir = os.path.join(dir_root, "performance_logs")
    timer = init_timer(timing_dir, "hyper_main")
    timer.log_timing("hyper.py", 24, 39, "Configuration loading and path setup", time.time() - t0)
    
    logger.info("******************* 🔥 Hyper starts !!! *******************")
    
    # --- Parallel control ---
    control_cfg = cfg.get("control", {})
    use_parallel = control_cfg.get("parallel_maps", False)
    n_cores = control_cfg.get("n_cores", os.cpu_count())
    max_total_threads = control_cfg.get("max_total_threads", 50)  # Default: limit to 50 threads total
    parallel_mode = control_cfg.get("parallel_mode", "maps")  # "maps" or "sources"

    # Thread limits are set by run_hyper_playground.py BEFORE numpy imports
    # Log the actual configured values (workers inherit from parent environment)
    threads_per_worker = int(os.environ.get('OMP_NUM_THREADS', '1'))
    if use_parallel:
        if parallel_mode == "sources":
            logger.info(f"🔧 Source-level parallelism: {n_cores} workers × {threads_per_worker} threads each = {threads_per_worker * n_cores} max total")
        else:
            logger.info(f"🔧 Map-level parallelism: {n_cores} workers × {threads_per_worker} threads each = {threads_per_worker * n_cores} max total")

    # --- Main parallel or serial execution ---
    mode_desc = 'source-parallel' if (use_parallel and parallel_mode == "sources") else ('multiprocessing' if use_parallel else 'serial')
    logger.info(f"🔄 Starting map analysis using {mode_desc} mode")
    
    t_maps_start = time.time()
    results = []
    map_timeout = control_cfg.get("map_timeout_minutes", 0) * 60  # 0 = no timeout

    if use_parallel and parallel_mode == "sources":
        # INTRA-MAP PARALLELISM: process maps sequentially, parallelize source fitting within each map
        logger.info(f"📡 Running HYPER on {len(map_names)} maps with {n_cores} parallel source workers...")
        for map_name in map_names:
            logger.info(f"📡 Processing: {map_name}")
            try:
                suffix, bg_model, cutout_header, initial_header = single_map(map_name, cfg, dir_root, logger, logger_file_only)
                results.append(suffix)
                if datacube:
                    background_slices.append(bg_model)
                    slice_cutout_header.append(cutout_header)
                logger.info(f"✅ Finished processing {map_name}")
            except Exception as e:
                logger.error(f"❌ Error processing {map_name}: {e}")
    elif use_parallel:
        logger.info(f"📡 Running HYPER on {len(map_names)} maps using {n_cores} cores...")
        if map_timeout > 0:
            logger.info(f"⏱️  Per-map timeout: {map_timeout/60:.0f} minutes")
        
        with ProcessPoolExecutor(max_workers=n_cores) as executor:
            futures = {
                executor.submit(single_map, name, cfg, dir_root): name
                for name in map_names
            }
            for future in as_completed(futures):
                map_name = futures[future]
                try:
                    # Apply timeout if configured
                    timeout = map_timeout if map_timeout > 0 else None
                    suffix, bg_model, cutout_header, initial_header = future.result(timeout=timeout)
                    results.append(suffix)
                    if datacube:
                        background_slices.append(bg_model)
                        slice_cutout_header.append(cutout_header)
                    
                    logger.info(f"✅ Finished processing {map_name}")
                except TimeoutError:
                    logger.error(f"⏰ TIMEOUT processing {map_name} (>{map_timeout/60:.0f} min) - skipping")
                except Exception as e:
                    logger.error(f"❌ Error processing {map_name}: {e}")
    else:
        for map_name in map_names:
            logger.info(f"📡 Running HYPER on: {map_name}")
            suffix, bg_model, cutout_header, initial_header = single_map(map_name, cfg, dir_root, logger, logger_file_only)
            results.append(suffix)
            if datacube:
                background_slices.append(bg_model)
                slice_cutout_header.append(cutout_header)
    
    timer.log_timing("hyper.py", 90, 123, f"Processing {len(map_names)} maps ({'parallel' if use_parallel else 'serial'})", 
                     time.time() - t_maps_start)
                            
    # --- Collect all output tables --- #
    t_collect = time.time()
    all_tables = []
    for suffix in results:
        try:
            suffix_clean = Path(suffix).stem  # remove ".fits"
            output_table_path = os.path.join(dir_root, output_dir, f"{base_table_name}_{suffix_clean}.txt")
            table = ascii.read(output_table_path, format="ipac")
            all_tables.append(table)
        except Exception as e:
            logger_file_only.error(f"[ERROR] Failed to load table for {suffix}: {e}")
    

    timer.log_timing("hyper.py", 126, 137, "Collecting output tables from all maps", time.time() - t_collect)
    
    # === Merge and write combined tables ===
    t_merge = time.time()
    final_table = vstack(all_tables)
    
    # Keep only the comments (headers) from the first table
    if hasattr(all_tables[0], 'meta') and 'comments' in all_tables[0].meta:
        final_table.meta['comments'] = all_tables[0].meta['comments']
    else:
        final_table.meta['comments'] = []
    
    # Output file paths
    ipac_path = os.path.join(dir_root, output_dir, f"{base_table_name}_ALL.txt")
    csv_path = os.path.join(dir_root, output_dir, f"{base_table_name}_ALL.csv")
    
    # Write outputs
    final_table.write(ipac_path, format='ipac', overwrite=True)
    final_table.write(csv_path, format='csv', overwrite=True)
    logger_file_only.info(f"\n✅ Final merged table saved to:\n- {ipac_path}\n- {csv_path}")
    
    timer.log_timing("hyper.py", 140, 157, "Merging and writing final output tables", time.time() - t_merge)
    
    # === Combine all bg_models into a datacube ===
    if datacube:
        t_datacube = time.time()
        create_background_cubes(background_slices, slice_cutout_header, cube_header, dir_slices_out, fix_min_box, convert_mjy, logger)
        timer.log_timing("hyper.py", 161, 162, "Creating background datacubes", time.time() - t_datacube)

    # Log total time and write summary
    timer.log_timing("hyper.py", 24, 164, "TOTAL EXECUTION TIME", time.time() - start_time_total)
    timer.write_summary()
    logger.info(f"\n⏱️  Performance timing saved to: {timer.timing_file}")
    
    logger.info("****************** ✅ Hyper finished !!! ******************")