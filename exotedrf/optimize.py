#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 15 August 2025

@author: PSD

Custom JWST optimizer for exoTEDRF pipeline
"""
print("RUN OPTIMIZER")
# ======== STANDARD LIBRARY IMPORTS ========
import os       # For file paths and environment variable handling
import sys      # For exiting with error messages
import glob     # For matching file patterns
import time     # For timing operations
import argparse # For parsing command-line arguments
import yaml     # For reading YAML configuration files

# ======== PROJECT IMPORTS ========
from exotedrf import utils  # Utility functions for exoTEDRF 

# --------------------------------------------------------
# 1) EARLY CONFIG PARSING
# --------------------------------------------------------
# Create a lightweight ArgumentParser that only looks for --config/-c
# This is done *before* the main parser so we can load config and set env vars
early = argparse.ArgumentParser(add_help=False)
early.add_argument(
    "--config", "-c",
    default="run_optimize.yaml",       # Default config file if none given
    help="Path to your DMS config YAML"
)
# parse_known_args() -> returns parsed args plus the remaining unparsed args
args, remaining = early.parse_known_args()

# --------------------------------------------------------
# 2) LOAD CONFIG & SET JWST CRDS ENVIRONMENT VARIABLES
# --------------------------------------------------------
# Read the YAML config file to get CRDS path/context before importing JWST modules
try:
    cfg_early = yaml.safe_load(open(args.config))
except FileNotFoundError:
    sys.exit(f"ERROR: config file '{args.config}' not found.")

# Set CRDS cache path (local storage for JWST calibration files)
os.environ.setdefault(
    "CRDS_PATH",
    cfg_early.get("crds_cache_path", "./crds_cache")
)
# Set CRDS server URL (location of JWST calibration data online)
os.environ.setdefault(
    "CRDS_SERVER_URL",
    "https://jwst-crds.stsci.edu"
)
# Set CRDS context (specific calibration reference mapping to use)
os.environ.setdefault(
    "CRDS_CONTEXT",
    cfg_early.get("crds_context", "jwst_1322.pmap")
)

# --------------------------------------------------------
# 3) NUMERICAL / PLOTTING IMPORTS
# --------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from astropy.io import fits

# --------------------------------------------------------
# 4) ADDITIONAL PROJECT IMPORTS
# --------------------------------------------------------
from exotedrf.utils import parse_config, unpack_input_dir, fancyprint
from exotedrf.stage1 import run_stage1
from exotedrf.stage2 import run_stage2
from exotedrf.stage3 import run_stage3


# ======== OUTPUT DIRECTORY DEFINITIONS ========
# Define where to store outputs for each pipeline stage
outdir    = 'pipeline_outputs_directory'         # Main output root
outdir_f  = 'pipeline_outputs_directory/Files'   # Generic files (tables, logs, etc.)
outdir_s1 = 'pipeline_outputs_directory/Stage1/' # Stage 1 calibrated data
outdir_s2 = 'pipeline_outputs_directory/Stage2/' # Stage 2 calibrated data
outdir_s3 = 'pipeline_outputs_directory/Stage3/' # Stage 3 processed data
outdir_s4 = 'pipeline_outputs_directory/Stage4/' # Stage 4 final results

# Ensure that all required output directories exist (create if missing)
utils.verify_path('pipeline_outputs_directory')
utils.verify_path('pipeline_outputs_directory/Files')
utils.verify_path('pipeline_outputs_directory/Stage1')
utils.verify_path('pipeline_outputs_directory/Stage2')
utils.verify_path('pipeline_outputs_directory/Stage3')
utils.verify_path('pipeline_outputs_directory/Stage4')

# ======== OBSERVING CONFIG PARAMETERS ========
# Observation mode in lowercase (e.g., 'niriss', 'nirspec', 'miri')
obs_early = (cfg_early.get('observing_mode') or '').lower()
# Detector filter in lowercase (e.g., 'clear', 'nrs1', 'nrs2')
filter_early = (cfg_early.get('filter_detector') or '').lower()
# Wavelength range limits for analysis and plotting (if provided in config)
wave_range_early      = cfg_early.get('wave_range', None)
wave_range_plot_early = cfg_early.get('wave_range_plot', None)
# Weighting factors for cost function or metrics
w1 = cfg_early.get('w1', 0.0)
w2 = cfg_early.get('w2', 1.0)

# ======== INSTRUMENT WAVELENGTH LIMITS ========
# Allowed wavelength coverage for each instrument (microns)
bands = {
    'miri':    (5.0, 13.0),
    'nirspec': (1.0, 5.0),
    'niriss':  (1.0, 2.8)
}

# ======== VALIDATION: CHECK WAVELENGTH RANGE AGAINST INSTRUMENT LIMITS ========
# Loop through instruments to find the matching one for this observation
for key, (lo, hi) in bands.items():
    if key in obs_early:
        # Validate that provided ranges (if any) fall within allowed limits
        for name, rng in (('wave_range', wave_range_early),
                          ('wave_range_plot', wave_range_plot_early)):
            if rng is not None and not (lo <= min(rng) and max(rng) <= hi):
                raise ValueError(f"{name}={rng!r} out of allowed band [{lo}, {hi}]")
        break
# If no instrument key matched the observation mode, throw an error
else:
    raise ValueError(f"Unrecognized observing_mode: {cfg_early.get('observing_mode')}")


# ----------------------------------------
# Plot the cost values from a parameter sweep
# ----------------------------------------
def plot_cost(name_str, table_height=0.4):
    """
    Reads a tab-delimited cost file, detects parameter sweeps, highlights 
    the best parameter set(s), and produces a figure showing cost trends.

    Parameters
    ----------
    name_str : str
        Identifier used to find the cost file (Cost_<name_str>.txt).
    table_height : float
        Fraction of the figure height to allocate to the table display.
    """

    # ======== LOAD AND CLEAN DATA ========
    # Read cost file for the given run name
    df = pd.read_csv(f"pipeline_outputs_directory/Files/Cost_{name_str}.txt",
                     delimiter="\t")

    # Remove rows where 'cost' is not numeric
    df = df[pd.to_numeric(df["cost"], errors="coerce").notna()].reset_index(drop=True)

    # Get all parameter columns (exclude 'duration_s' and 'cost' at the end)
    param_cols = df.columns[:-2]

    # ======== DETECT WHICH PARAMETER CHANGED PER ROW (sweep-aware) ========
    changed_param_per_row = [None] * len(df)
    
    # current sweep = first differing column between row 0 and 1 (fallback to first varying col)
    if len(df) > 1:
        diffs01 = [c for c in param_cols if df.at[1, c] != df.at[0, c]]
        if diffs01:
            current_param = diffs01[0]
        else:
            # fallback: first column that varies anywhere
            vary = [c for c in param_cols if df[c].nunique(dropna=False) > 1]
            current_param = vary[0] if vary else param_cols[0]
    else:
        current_param = param_cols[0]
    
    changed_param_per_row[0] = current_param
    changed_param_per_row[1 if len(df) > 1 else 0] = current_param
    
    # find sweep boundaries: as soon as ANY other parameter changes, the next sweep starts
    sweep_lines = []  # indices where a new sweep begins
    for i in range(1, len(df)):
        diffs = [c for c in param_cols if df.at[i, c] != df.at[i-1, c]]
        if not diffs:  # nothing changed -> stay in current sweep
            changed_param_per_row[i] = current_param
            continue
    
        if current_param in diffs and len(diffs) == 1:
            # only the active param changed -> still same sweep
            changed_param_per_row[i] = current_param
        else:
            # another param appeared (possibly with the current one reverting)
            # new sweep starts at this row
            new_param = next((c for c in diffs if c != current_param), diffs[0])
            sweep_lines.append(i)
            current_param = new_param
            changed_param_per_row[i] = current_param
    
    # first row label belongs to the first detected sweep
    if len(df) >= 2 and changed_param_per_row[0] is None:
        changed_param_per_row[0] = changed_param_per_row[1] or param_cols[0]
    
    sweep_boundaries = [0] + sweep_lines + [len(df)]

    # ======== BUILD LABELS AND FIND SWEEP BOUNDARIES ========
    labels = []
    sweep_lines = []  # indices where a new parameter sweep starts
    last_changed_param = None
    for idx, row in df.iterrows():
        changed_param = changed_param_per_row[idx]
        # Start a new sweep if parameter changes
        if changed_param != last_changed_param and last_changed_param is not None:
            sweep_lines.append(idx)

        # Format value (use integer if no fractional part)
        value = row[changed_param]
        try:
            fv = float(value)
            value = int(fv) if fv.is_integer() else fv
        except Exception:
            pass

        labels.append(f"{changed_param}={value}")
        last_changed_param = changed_param

    df["changed_label"] = labels

    # ======== HIGHLIGHT BEST COST PER SWEEP ========
    sweep_boundaries = [0] + sweep_lines + [len(df)]
    colors = ['gray'] * len(df)  # default color
    for i in range(len(sweep_boundaries) - 1):
        start = sweep_boundaries[i]
        end = sweep_boundaries[i+1]
        # Get index of min cost in this sweep
        min_idx = df.iloc[start:end]["cost"].idxmin()
        colors[min_idx] = 'green'

    # ======== BEST OVERALL PARAMETERS ========
    best_row = df.loc[df["cost"].idxmin(), param_cols.tolist() + ["cost"]].copy()
    # Pretty-print numeric values
    for col in best_row.index:
        val = best_row[col]
        try:
            fv = float(val)
            best_row[col] = int(fv) if fv.is_integer() else fv
        except Exception:
            best_row[col] = val
    best_df = pd.DataFrame([best_row]).reset_index(drop=True)

    # ======== FIGURE LAYOUT ========
    fig = plt.figure(figsize=(max(14, len(df) * 0.25), 10))
    gs = GridSpec(nrows=2, ncols=1, height_ratios=[1 - table_height, table_height])
    ax_plot = fig.add_subplot(gs[0])
    ax_table = fig.add_subplot(gs[1])

    # Scatter plot of cost
    ax_plot.scatter(range(len(df)), df["cost"].values, color=colors)
    # Vertical dashed lines for sweep boundaries
    for x in sweep_lines:
        ax_plot.axvline(x=x - 0.5, color='gray', linestyle='--', linewidth=1)

    # X-axis tick labels -> just the value part of "param=value"
    values = [lbl.split('=', 1)[1] for lbl in df["changed_label"]]
    ax_plot.set_xticks(range(len(df)))
    ax_plot.set_xticklabels(values, rotation=0, fontsize=8)

    # Drop parameter names under x-axis, alternating heights to avoid overlap
    ymin, ymax = ax_plot.get_ylim()
    base_y = ymin - 0.08 * (ymax - ymin)
    alt_y  = ymin - 0.15 * (ymax - ymin)
    for i, (start, end) in enumerate(zip(sweep_boundaries[:-1], sweep_boundaries[1:])):
        param_name = df.loc[start, "changed_label"].split("=", 1)[0]
        center = (start + end - 1) / 2
        y_pos = base_y if i % 2 == 0 else alt_y
        ax_plot.text(center, y_pos, param_name, ha="center", va="top", fontsize=10)

    fig.subplots_adjust(bottom=0.30)
    ax_plot.set_ylabel("Cost (ppm)")
    ax_plot.set_title(f"Cost by Single Parameter Sweep: {name_str}")

    # ======== TABLE OF BEST PARAMETERS ========
    ax_table.axis("off")
    ax_table.text(0.5, 0.65, "Best Parameters", ha="center", va="bottom", fontsize=12)
    table = ax_table.table(
        cellText=best_df.values,
        colLabels=best_df.columns,
        cellLoc='center',
        loc='center'
    )
    table.scale(1.0, 1.8)
    table.auto_set_font_size(False)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_fontsize(7)   # header
        else:
            cell.set_fontsize(10)  # data

    # Save final plot to PNG
    fig.savefig(f"pipeline_outputs_directory/Files/Cost_{name_str}.png",
                dpi=300, bbox_inches='tight')

# ----------------------------------------
# create filenames
# ----------------------------------------
def make_step_filenames(input_files, output_dir, possible_steps, 
                        output_dir_2nd=None, possible_steps_2nd=None):
    """
    Search for files in output_dir matching any of the given step suffixes.
    If found, return regenerated filenames aligned to input_files.
    If not found and a second dir/list are given, search there.
    If still nothing, raise FileNotFoundError.

    Parameters
    ----------
    input_files : list[str]
        List of original input file paths.
    output_dir : str
        Primary directory to search for processed files.
    possible_steps : list[str]
        Ordered list of step suffixes to try (e.g., ['darkcurrentstep', 'refpixstep']).
    output_dir_2nd : str, optional
        Secondary directory to search if nothing found in primary.
    possible_steps_2nd : list[str], optional
        Steps to try in secondary directory.

    Returns
    -------
    list[str]
        Paths to regenerated filenames for the found step.
    """

    # Helper function to regenerate full output filenames
    def _regen(dirpath, step):
        """
        Given a directory and a step suffix, build output filenames
        by replacing the suffix of each input file with the given step.
        """
        out = []
        for f in input_files:
            base = os.path.basename(f)                # just filename, no path
            root = base[: base.rfind("_")]            # remove everything after last underscore
            out.append(os.path.join(dirpath, f"{root}_{step}.fits"))
        return out

    # 1) Primary search: loop over possible steps and check for matches in output_dir
    for step in possible_steps:
        if glob.glob(os.path.join(output_dir, f"*_{step}.fits")):
            print(f"Found step '{step}' in {output_dir}")
            return _regen(output_dir, step)

    # 2) Secondary search: same logic, but in output_dir_2nd if provided
    if output_dir_2nd and possible_steps_2nd:
        for step in possible_steps_2nd:
            if glob.glob(os.path.join(output_dir_2nd, f"*_{step}.fits")):
                print(f"Found step '{step}' in {output_dir_2nd}")
                return _regen(output_dir_2nd, step)

    # 3) No match found in either directory -> raise error
    raise FileNotFoundError(
        f"No matching step files found in '{output_dir}'"
        + (f" or '{output_dir_2nd}'" if output_dir_2nd else "")
    )


# ----------------------------------------
# cost function (P2P-based)
# ----------------------------------------
def cost_function(st3, baseline_ints=None, wave_range=None, w1=0.0, w2=1.0, tol=0.05):
    """
    Compute a combined white-light + spectral P2P (point-to-point) metric.

    Parameters
    ----------
    st3 : dict-like
        Must contain:
          - 'Flux' (or 'Flux O1'/'Flux O2' for NIRISS) -> 2D array (n_int, n_wave)
          - 'Wave' (or 'Wave O1'/'Wave O2') -> 1D array (n_wave,)
    baseline_ints : list of 1 or 2 ints
        Integration indices defining baseline(s) for the spectral term.
    wave_range : None or [min, max]
        If given, restrict spectral term to this wavelength range (within ±tol).
    w1, w2 : float
        Weights for white-light and spectral terms in final cost.
    tol : float
        Allowed deviation when matching wave_range endpoints.

    Returns
    -------
    cost : float
        Combined cost = w1*ptp2_white + w2*ptp2_spec
    ptp2_spec_wave : np.ndarray
        Per-wavelength ptp2 metric values.
    """

    # ======== NIRISS-SPECIFIC WAVE + FLUX MERGE ========
    if 'niriss' in obs_early:
        flux_O1 = np.asarray(st3['Flux O1'], float)  # Order 1 flux
        flux_O2 = np.asarray(st3['Flux O2'], float)  # Order 2 flux
        wave_O1 = np.asarray(st3['Wave O1'], float)  # Order 1 wavelengths
        wave_O2 = np.asarray(st3['Wave O2'], float)  # Order 2 wavelengths

        cutoff = 0.85  # μm — wavelength boundary between O2 and O1 segments

        # Find O2 indices up to cutoff
        i2 = np.where(wave_O2 <= cutoff)[0]
        # Find O1 indices above cutoff
        i1 = np.where(wave_O1 > cutoff)[0]

        if i2.size == 0 or i1.size == 0:
            raise ValueError("Cutoff produces empty segment: "
                             f"O2<= {cutoff}: {i2.size}, O1> {cutoff}: {i1.size}")

        idx2 = i2[-1]  # last valid O2 index
        idx1 = i1[0]   # first valid O1 index

        # Concatenate O2 segment + O1 segment along wavelength axis
        wave = np.concatenate([wave_O2[:idx2+1],        wave_O1[idx1:]])
        flux = np.concatenate([flux_O2[:, :idx2+1],     flux_O1[:, idx1:]], axis=1)

        # Sort by wavelength just in case
        s = np.argsort(wave)
        wave = wave[s]
        flux = flux[:, s]

    else:
        # For non-NIRISS: take flux/wave arrays directly
        flux = np.asarray(st3['Flux'], float)
        wave = np.asarray(st3['Wave'], float)

    # ======== WHITE-LIGHT TERM ========
    # Collapse all wavelengths into single white-light curve
    white      = np.nansum(flux, axis=1)
    white      = white[~np.isnan(white)]
    norm_white = white / np.median(white)
    # 2nd finite difference (neighbor avg - center)
    d2_white   = 0.5*(norm_white[:-2] + norm_white[2:]) - norm_white[1:-1]
    ptp2_white = np.nanmedian(np.abs(d2_white))

    # ======== SPECTRAL TERM (PER-WAVELENGTH P2P) ========
    wave_meds = np.nanmedian(flux, axis=0, keepdims=True)
    norm_spec = flux / wave_meds
    d2_spec   = 0.5*(norm_spec[:-2] + norm_spec[2:]) - norm_spec[1:-1]

    # Select baseline integrations for spectral metric
    if baseline_ints is None:
        ptp2_spec_wave = np.nanmedian(np.abs(d2_spec), axis=0)
    elif len(baseline_ints) == 1:
        N = int(baseline_ints[0])
        ptp2_spec_wave = np.nanmedian(np.abs(d2_spec[:N]), axis=0)
    elif len(baseline_ints) == 2:
        Nlow, Nhigh = map(int, baseline_ints)
        low_term  = np.nanmedian(np.abs(d2_spec[:Nlow]), axis=0)
        high_term = np.nanmedian(np.abs(d2_spec[Nhigh:]), axis=0)
        ptp2_spec_wave = 0.5 * (low_term + high_term)
    else:
        raise ValueError(f"baseline_ints must be length 1 or 2, got {len(baseline_ints)}")

    # ======== WAVELENGTH RANGE FILTER (OPTIONAL) ========
    if wave_range is None:
        ptp2_spec = np.nanmedian(ptp2_spec_wave)

    elif isinstance(wave_range, (list, tuple)) and len(wave_range) == 2:
        lo, hi = wave_range
        finite = np.isfinite(wave)
        if not finite.any():
            raise ValueError("All entries in wave are NaN!")

        # Distances from requested range edges
        dist_lo = np.abs(wave - lo); dist_lo[~finite] = np.inf
        dist_hi = np.abs(wave - hi); dist_hi[~finite] = np.inf

        idx_lo = int(np.argmin(dist_lo))
        idx_hi = int(np.argmin(dist_hi))

        # Check tolerance
        if dist_lo[idx_lo] > tol or dist_hi[idx_hi] > tol:
            raise ValueError(f"wave_range {wave_range} not found within ±{tol}")

        # Slice range in correct order
        i0, i1 = sorted((idx_lo, idx_hi))
        sub = ptp2_spec_wave[i0:i1+1]
        if np.all(np.isnan(sub)):
            raise ValueError(f"No valid ptp2_spec values in wave range {wave_range}")
        ptp2_spec = np.nanmedian(sub)

    else:
        raise ValueError("wave_range must be None or a length-2 list/tuple")

    # ======== FINAL COST COMBINATION ========
    cost = w1 * ptp2_white + w2 * ptp2_spec

    return cost, ptp2_spec_wave



# ----------------------------------------
# diagnostic plot
# ----------------------------------------
def diagnostic_plot(st3, name_str, baseline_ints, outdir=outdir_f):
    """
    Create two diagnostic plots from Stage-3 data:
      1) Normalized white-light curve
      2) Normalized flux image with true wavelength mapping

    Parameters
    ----------
    st3 : dict-like
        Stage-3 outputs containing flux and wavelength arrays.
        For NIRISS/SOSS: requires 'Flux_O1', 'Flux_O2', 'Wave_O1', 'Wave_O2'.
        For others: requires 'Flux', 'Wave'.
    name_str : str
        Identifier used in output filenames.
    baseline_ints : list[int]
        One or two integers for baseline integrations:
            [N] -> normalize by median of first N integrations
            [Nlow, Nhigh] -> normalize by mean of medians of start and end segments
    outdir : str
        Output directory for saved figures.
    """

    os.makedirs(outdir, exist_ok=True)

    # ======== WAVELENGTH RANGE SELECTION BASED ON MODE/FILTER ========
    # obs_early and filter_early must be defined globally before calling
    if 'miri' in obs_early:
        wave_min, wave_max = 5.0, 12.0
    elif 'niriss' in obs_early:
        wave_min, wave_max = 0.6, 2.8
    elif 'nirspec' in obs_early:
        if filter_early == 'nrs1':
            wave_min, wave_max = 2.9, None
        elif filter_early == 'nrs2':
            wave_min, wave_max = None, 2.9
        else:
            raise ValueError(f"Unknown nirspec filter_detector: {filter_early}")
    else:
        raise ValueError(f"Unknown observing_mode: {obs_early}")

    # --- Build stitched spectrum ---
    if 'niriss' in obs_early:
        # Load flux and wavelength for both spectral orders
        flux_O1 = np.asarray(st3['Flux_O1'], float)
        flux_O2 = np.asarray(st3['Flux_O2'], float)
        wave_O1 = np.asarray(st3['Wave_O1'], float)
        wave_O2 = np.asarray(st3['Wave_O2'], float)

        # Cutoff wavelength separating orders
        cutoff = 0.85  # µm

        # Indices: O2 wavelengths ≤ cutoff, O1 wavelengths > cutoff
        i2 = np.where(wave_O2 <= cutoff)[0]
        i1 = np.where(wave_O1 > cutoff)[0]
        if i2.size == 0 or i1.size == 0:
            raise ValueError(
                f"Cutoff {cutoff} yields empty segment: "
                f"O2<= {i2.size}, O1> {i1.size}"
            )

        # Concatenate both orders along wavelength axis
        wave = np.concatenate([wave_O2[:i2[-1]+1], wave_O1[i1[0]:]])
        flux = np.concatenate([flux_O2[:, :i2[-1]+1], flux_O1[:, i1[0]:]], axis=1)
    else:
        # Non-NIRISS: directly load single flux/wavelength arrays
        flux = np.asarray(st3['Flux'], float)
        wave = np.asarray(st3['Wave'], float)

    # --- Apply wavelength range filter ---
    mask = np.isfinite(wave)
    if wave_min is not None:
        mask &= wave >= wave_min
    if wave_max is not None:
        mask &= wave <= wave_max
    wave = wave[mask]
    flux = flux[:, mask]

    # --- Sort by wavelength ---
    # mergesort preserves order for equal wavelengths (stable sort)
    s = np.argsort(wave, kind='mergesort')
    wave = wave[s]
    flux = flux[:, s]

    # --- Drop bad columns and enforce strictly increasing wavelengths ---
    # Column median across time for each spectral channel
    col_med = np.nanmedian(flux, axis=0)
    # Keep only finite wavelengths, finite medians, and non-zero medians
    good = np.isfinite(wave) & np.isfinite(col_med) & (col_med != 0)
    wave = wave[good]
    flux = flux[:, good]

    # --- Collapse duplicate wavelengths ---
    # Round wavelengths to tolerance to handle floating-point noise
    w_round = np.round(wave, 12)
    _, keep_idx = np.unique(w_round, return_index=True)
    keep_idx.sort()  # keep in ascending order
    wave = wave[keep_idx]
    flux = flux[:, keep_idx]

    # --- White-light curve ---
    # Sum flux over all spectral channels for each integration
    white = np.nansum(flux, axis=1)
    if len(baseline_ints) == 1:
        # Normalize by median of first N integrations
        N = int(baseline_ints[0])
        norm_white = white / np.median(white[:N])
    else:
        # Normalize by mean of medians from start and end segments
        Nlow, Nhigh = map(int, baseline_ints)
        base = 0.5 * (
            np.median(white[:Nlow]) +
            np.median(white[Nhigh:])
        )
        norm_white = white / base

    # --- Plot normalized white-light curve ---
    plt.figure()
    plt.plot(norm_white, marker='.')
    plt.xlabel("Integration Number")
    plt.ylabel("Normalized White Flux")
    plt.title("Normalized White-light Curve")
    plt.grid(True)
    plt.savefig(f"{outdir}/norm_white_{name_str}.png", dpi=300)
    plt.close()

    # --- Normalized flux image with true wavelength mapping ---
    # Normalize each column by its time median (post-cleaning)
    img = np.full_like(flux, np.nan, dtype=float)
    img[:, :] = flux / col_med[good][keep_idx]  # safe: filtered for finite non-zero values

    n_int, n_pix = img.shape

    # Require strictly increasing wavelength for pcolormesh bin edges
    if not np.all(np.diff(wave) > 0):
        raise ValueError("wave must be strictly increasing for pcolormesh")

    # Compute wavelength bin edges for pcolormesh
    dw = np.diff(wave)
    edges = np.empty(n_pix + 1, float)
    edges[1:-1] = 0.5 * (wave[:-1] + wave[1:])  # midpoints
    edges[0] = wave[0] - dw[0] / 2              # lower bound
    edges[-1] = wave[-1] + dw[-1] / 2           # upper bound

    # Integration edges for x-axis
    x = np.arange(n_int + 1)

    # Plot normalized flux image
    plt.figure()
    plt.pcolormesh(x, edges, img.T, shading="auto", vmin=0.98, vmax=1.02)
    plt.xlabel("Integration Number")
    plt.ylabel("Wavelength (µm)")
    plt.title("Normalized Flux Image")
    plt.colorbar(label="Relative Flux")
    plt.savefig(f"{outdir}/flux_img_{name_str}.png", dpi=300)
    plt.close()



# ----------------------------------------
# Plot Scatter
# ----------------------------------------
def plot_scatter(  
    txtfile, rows,
    wave_range=None, smooth=None,
    spectrum_files=None,
    style='line', ylim=None, save_path=None,
    tol=0.05
):
    """
    Plot point-to-point (P2P) scatter vs wavelength for selected rows from a scatter table.

    Overlays for each selected row:
      1) Smoothed series using a moving-average window (`smooth`) if provided
      2) Raw (unsmoothed) series

    Photon-noise curves are intentionally excluded from this plot.

    Parameters
    ----------
    txtfile : str
        Path to the whitespace-delimited scatter table.
    rows : list[int]
        Indices of the table rows to plot. Negative indices count from the end.
    wave_range : tuple(float, float), optional
        Wavelength range to plot (μm), with tolerance `tol`.
    smooth : int, optional
        Window size (in pixels) for moving-average smoothing.
    spectrum_files : list[str]
        List of spectrum FITS files to retrieve wavelength axis from.
    style : {'line', 'scatter'}
        Plotting style.
    ylim : tuple(float, float), optional
        y-axis limits.
    save_path : str, optional
        If given, save the plot to this file.
    tol : float
        Allowed margin when applying wave_range filtering.
    """

    # --- Load scatter table ---
    # Read whitespace-delimited table, replace NaNs with 0.0
    df = pd.read_csv(txtfile, sep=r'\s+', header=None).fillna(0.0)
    n_rows, n_cols = df.shape

    # --- Validate requested rows ---
    valid = []
    for r in rows:
        # Convert negative indices to positive equivalents
        i = r if r >= 0 else n_rows + r
        if 0 <= i < n_rows:
            valid.append(i)
        else:
            print(f"Warning: row {r} out of range, skipping.")
    if not valid:
        raise ValueError("No valid rows to plot.")

    # --- Load wavelength grid to match scatter columns ---
    if not spectrum_files:
        raise ValueError("`spectrum_files` is required to read the wavelength axis.")
    with fits.open(spectrum_files[0]) as hdus:
        # Create dict mapping sanitized HDU names to HDU objects
        name_map = {h.name.replace(" ", "_"): h
                    for h in hdus if h.data is not None and h.name != "PRIMARY"}

        # Special handling for NIRISS with two orders
        if ("Wave_O1" in name_map) and ("Wave_O2" in name_map):
            wave_O1 = np.asarray(name_map["Wave_O1"].data, float)
            wave_O2 = np.asarray(name_map["Wave_O2"].data, float)
            cutoff = 0.85  # μm: boundary between orders
            # Select O2 wavelengths <= cutoff
            i2 = np.where(np.isfinite(wave_O2) & (wave_O2 <= cutoff))[0]
            # Select O1 wavelengths > cutoff
            i1 = np.where(np.isfinite(wave_O1) & (wave_O1 > cutoff))[0]
            if i2.size == 0 or i1.size == 0:
                raise ValueError(f"Cutoff {cutoff} yields empty segment: "
                                 f"O2<={i2.size}, O1>{i1.size}")
            # Concatenate the valid segments
            wave_full = np.concatenate([wave_O2[:i2[-1]+1], wave_O1[i1[0]:]])
        else:
            # Fallback: read first extension array as wavelength grid
            wave_full = np.asarray(hdus[1].data, float)

    # --- Ensure monotonic wavelength and align scatter columns ---
    # Sort indices by wavelength, keeping equal values in original order (stable)
    s = np.argsort(wave_full, kind="mergesort")
    wave_sorted = wave_full[s]

    # Wavelength length must match scatter table column count
    if wave_sorted.size != n_cols:
        raise ValueError(f"Wavelength length {wave_sorted.size} != scatter columns {n_cols}")

    # Build boolean mask for desired wavelength range
    if wave_range is not None:
        wmin, wmax = wave_range
        mask = np.isfinite(wave_sorted) & (wave_sorted >= wmin - tol) & (wave_sorted <= wmax + tol)
    else:
        mask = np.isfinite(wave_sorted)
    if not mask.any():
        raise ValueError(f"No finite wavelengths within selected range {wave_range}.")

    # Final x-axis values
    x = wave_sorted[mask]

    # --- Plot ---
    plt.figure(figsize=(8, 4))

    for i in valid:
        # Extract row data and reorder columns to match wavelength order
        y_full = df.iloc[i, :].to_numpy(float)
        y_ord = y_full[s]

        # Raw series (convert to ppm)
        y_raw = (y_ord[mask]) * 1e6
        if style == 'line':
            plt.plot(x, y_raw, linewidth=0.6, linestyle='-', alpha=0.5,
                     color='grey', label="Best Parameter configuration (raw)")
        else:
            plt.scatter(x, y_raw, s=3, alpha=0.8,
                        label="Best Parameter configuration (raw)")

        # Smoothed series (moving average)
        if smooth and int(smooth) > 1:
            w = int(smooth)
            kern = np.ones(w, dtype=float) / w
            y_sm_all = np.convolve(y_ord, kern, mode='same')
            y_sm = (y_sm_all[mask]) * 1e6
            if style == 'line':
                plt.plot(x, y_sm, linewidth=1.0,
                         label=f"Best Parameter configuration (smoothed:{w})")
            else:
                plt.scatter(x, y_sm, s=6,
                            label=f"Best Parameter configuration (smoothed:{w})")

    # --- Finalize plot ---
    plt.xlim(x.min(), x.max())
    if ylim is not None:
        plt.ylim(ylim)
    plt.xlabel("Wavelength (μm)")
    plt.ylabel("Scatter (ppm)")
    plt.legend(ncol=2, fontsize='small')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Figure saved to {save_path}")
    plt.show()



# ----------------------------------------
# skip step list
# ----------------------------------------
def get_stage_skips(cfg, steps, always_skip=None, special_one_over_f=False):
    """
    Build a list of pipeline steps to skip based on a configuration dictionary.

    Parameters
    ----------
    cfg : dict
        Configuration mapping step names to actions (e.g., {'DarkCurrentStep': 'run'}).
    steps : list[str]
        Candidate step names to check.
    always_skip : list[str], optional
        Steps to skip unconditionally, regardless of cfg settings.
    special_one_over_f : bool
        If True, treat any step whose name starts with 'OneOverFStep' as 'OneOverFStep'
        when adding to skip list. Useful if different variants exist.

    Returns
    -------
    list[str]
        Steps to skip for this run.
    """

    # Initialize skip set from always_skip (if given)
    skips = set(always_skip or [])

    # Check each candidate step in config
    for step in steps:
        # If the config marks this step to 'skip'
        if cfg.get(step, 'run') == 'skip':
            # Special handling for OneOverFStep variants
            if special_one_over_f and step.startswith('OneOverFStep'):
                skips.add('OneOverFStep')
            else:
                skips.add(step)

    # Return as a list (order not guaranteed since set used)
    return list(skips)




# ----------------------------------------
# main
# ----------------------------------------

def main():
    # Set up argument parser for command-line usage
    parser = argparse.ArgumentParser(
        description="Coordinate‐descent optimizer for exoTEDRF Stages 1–3"
    )
    parser.add_argument(
        "--config", default="run_optimize.yaml",
        help="Path to your DMS config YAML"
    )
    
    args = parser.parse_args()

    # Load YAML config file into dictionary
    cfg = parse_config(args.config)

    # Observation mode (Instrument mode used; e.g., 'NIRISS/SOSS', 'NIRSpec/G395H', 'MIRI/LRS'.)
    obs = (cfg.get('observing_mode') or '').lower()
    # Detector filter (Type of filter or detector used. For SOSS: 'CLEAR'/'F277W'. For NIRSpec: 'NRS1'/'NRS2'.)
    filter = (cfg.get('filter_detector') or '').lower()

    # Read key parameters from config (or use defaults)
    baseline_ints     = cfg.get('baseline_ints', [100, -100])
    wave_range        = cfg.get('wave_range', None)
    name_str          = cfg.get('name_tag', 'default_run')
    wave_range_plot   = cfg.get('wave_range_plot', None)
    ylim_plot         = cfg.get('ylim_plot', None)

    if 'nrs1' in filter:
        if wave_range is None:
            wave_range = (2.9,5.0)
        if wave_range_plot is None:
            wave_range_plot = (2.9,5.0)

    elif 'niriss' in obs:
        if wave_range is None:
            wave_range = (0.6, 2.8)
        if wave_range_plot is None:
            wave_range_plot = (0.6, 2.8)

    elif 'miri' in obs:
        if wave_range is None:
            wave_range = (5, 12)
        if wave_range_plot is None:
            wave_range_plot = (5, 12)


    # Start total runtime timer
    t0_total = time.perf_counter()

    # Load input FITS files from directory
    input_files = unpack_input_dir(
        cfg["input_dir"],
        mode=cfg["observing_mode"],
        filetag=cfg["input_filetag"],
        filter_detector=cfg["filter_detector"],
    ) 
    if isinstance(input_files, np.ndarray):
        input_files = input_files.tolist()

    # If no files found, try globbing directly for *.fits
    if not input_files:
        fancyprint(f"[WARN] No files in {cfg['input_dir']}, globbing *.fits")
        input_files = sorted(glob.glob(os.path.join(cfg["input_dir"], "*.fits")))
    if not input_files:
        raise RuntimeError(f"No FITS found in {cfg['input_dir']}")
    fancyprint(f"Using {len(input_files)} segment(s) from {cfg['input_dir']}")

    # ----------------------------------------------------------------
    # Separate YAML parameters into sweep ranges vs fixed parameters
    # ----------------------------------------------------------------
    param_ranges = {}
    fixed_params = {} 
    for k, v in cfg.items():
        if k.startswith("optimize_"):
            param_name = k[len("optimize_"):]  # e.g., "soss_inner_mask_width"
            if v:  # True means: sweep over provided list
                vals = cfg[param_name]
                if not isinstance(vals, list):
                    raise ValueError(f"optimize_{param_name} is True but '{param_name}' is not a list in YAML: {vals}")
                param_ranges[param_name] = vals
            else:  # False means: fix to single provided value
                val = cfg[param_name]
                if isinstance(val, list):
                    raise ValueError(f"optimize_{param_name} is False but '{param_name}' is a list in YAML: {val}")
                fixed_params[param_name] = val

    # Order of parameters for coordinate descent
    param_order = list(param_ranges.keys())
    total_steps = sum(len(v) for v in param_ranges.values())

    # Initialize current parameter set to median values of ranges + fixed params
    current = {k: int(np.median(v)) for k,v in param_ranges.items()}
    current.update(fixed_params)

    # Open logs for cost function values & scatter curves
    logf = open(f"pipeline_outputs_directory/Files/Cost_{name_str}.txt","w")
    logs = open(f"pipeline_outputs_directory/Files/Scatter_{name_str}.txt", "w")
    logf.write("\t".join(param_order)+"\tduration_s\tcost\n")

    count = 1  # Step counter for progress tracking
    
    # ------------------------------------------------------
    # Coordinate descent optimization loop over parameters
    # ------------------------------------------------------
    for key in param_order:
        fancyprint(
            f"Optimizing {key} "
            f"(fixed-other={{{', '.join(f'{k}:{current[k]}' for k in current if k!=key)}}})"
        )
        best_val  = current[key]
        best_cost = None

        for trial in param_ranges[key]:
            # Report trial info
            fancyprint(f"Iteration {count}/{total_steps}: {key}={trial}")
            trial_params = {**current, key: trial}
            run_cfg = cfg.copy()
            run_cfg.update(trial_params)

            t0 = time.perf_counter()

            print(
                "\n############################################",
                f"\n Iteration: {count}/{total_steps} starting {key}={trial}",
                "\n############################################\n",
                flush=True
            )            

            # ----------------------------------------------------------------
            # Split args into per-stage overrides (e.g., JumpStep, BadPixStep)
            # ----------------------------------------------------------------
            s1_args, s2_args, s3_args = {}, {}, {}
            if "time_window" in trial_params:
                s1_args["JumpStep"] = {"time_window":trial_params["time_window"]}
            badpix = {}
            if "box_size" in trial_params:
                badpix["box_size"] = trial_params["box_size"]
            if "window_size" in trial_params:
                badpix["window_size"] = trial_params["window_size"]
            if badpix:
                s2_args["BadPixStep"] = badpix

            # Define full list of steps for each stage
            stage1_steps = [
                'DQInitStep','EmiCorrStep','SaturationStep','ResetStep','SuperBiasStep',
                'RefPixStep','DarkCurrentStep','OneOverFStep_grp','LinearityStep',
                'JumpStep','RampFitStep','GainScaleStep'
            ]
            stage2_steps = [
                'AssignWCSStep','Extract2DStep','SourceTypeStep','WaveCorrStep',
                'FlatFieldStep','BackgroundStep','OneOverFStep_int',
                'BadPixStep','PCAReconstructStep','TracingStep'
            ]
            stage3_steps = []

            # ----------------------------------------------------------------
            # The giant if/elif structure below chooses the fastest rerun path
            # depending on which parameter is being optimized
            # ----------------------------------------------------------------


            if best_cost is None:
                # ======================================================
                # First trial for this parameter -> run full Stage 1 -> 3
                # ======================================================

                # -------------------------
                # Stage 1: Detector-level processing
                # -------------------------

                # Steps to always skip in Stage 1 (none in this case)
                always_skip1 = []

                # Determine Stage-1 steps to skip based on config (cfg)
                # and 'always_skip1'. The `special_one_over_f=True` flag
                # triggers extra logic for OneOverFStep naming variations.
                stage1_skip = get_stage_skips(
                    cfg,
                    stage1_steps,
                    always_skip=always_skip1,
                    special_one_over_f=True
                )

                # Run Stage 1 on raw input files
                stage1_results = run_stage1(
                    input_files,
                    mode=run_cfg['observing_mode'],                # e.g. 'NIRSpec/PRISM'
                    soss_background_model=run_cfg['soss_background_file'],
                    baseline_ints=run_cfg['baseline_ints'],
                    oof_method=run_cfg['oof_method'],              # one-over-f noise correction method
                    superbias_method=run_cfg['superbias_method'],  # superbias subtraction method
                    soss_timeseries=run_cfg['soss_timeseries'],
                    soss_timeseries_o2=run_cfg['soss_timeseries_o2'],
                    save_results=True,
                    pixel_masks=run_cfg['outlier_maps'],           # optional pixel mask files
                    force_redo=True,                               # force rerun even if files exist
                    flag_up_ramp=run_cfg['flag_up_ramp'],          # flag groups during up-the-ramp fitting
                    rejection_threshold=run_cfg['jump_threshold'], # jump detection threshold
                    flag_in_time=run_cfg['flag_in_time'],          # time-based flagging
                    time_rejection_threshold=run_cfg['time_jump_threshold'],
                    output_tag=run_cfg['output_tag'],              # appended to filenames
                    skip_steps=stage1_skip,                        # steps to skip this run
                    do_plot=run_cfg['do_plots'],                   # plots
                    soss_inner_mask_width=run_cfg['soss_inner_mask_width'],
                    soss_outer_mask_width=run_cfg['soss_outer_mask_width'],
                    nirspec_mask_width=run_cfg['nirspec_mask_width'],
                    centroids=run_cfg['centroids'],                # optional centroids file
                    hot_pixel_map=run_cfg['hot_pixel_map'],        # hot-pixel map for masking
                    miri_drop_groups=run_cfg['miri_drop_groups'],  # MIRI group-dropping parameter
                    **run_cfg.get('stage1_kwargs', {}),             # extra Stage-1 kwargs from config
                    **s1_args                                       # overrides built earlier in loop
                )

                # -------------------------
                # Stage 2: Spectroscopic extraction & calibration
                # -------------------------

                always_skip2 = []
                # Determine Stage-2 steps to skip (no special handling here)
                stage2_skip = get_stage_skips(
                    cfg,
                    stage2_steps,
                    always_skip=always_skip2,
                    special_one_over_f=False
                )

                # Run Stage 2 using Stage-1 outputs
                stage2_results, centroids = run_stage2(
                    stage1_results,
                    mode=run_cfg['observing_mode'],
                    soss_background_model=run_cfg['soss_background_file'],
                    baseline_ints=run_cfg['baseline_ints'],
                    save_results=True,
                    force_redo=True,
                    space_thresh=run_cfg['space_outlier_threshold'],   # spatial outlier threshold
                    time_thresh=run_cfg['time_outlier_threshold'],     # temporal outlier threshold
                    remove_components=run_cfg['remove_components'],    # PCA components to remove
                    pca_components=run_cfg['pca_components'],          # number of PCA components
                    soss_timeseries=run_cfg['soss_timeseries'],
                    soss_timeseries_o2=run_cfg['soss_timeseries_o2'],
                    oof_method=run_cfg['oof_method'],
                    output_tag=run_cfg['output_tag'],
                    smoothing_scale=run_cfg['smoothing_scale'],        # smoothing scale for extraction
                    skip_steps=stage2_skip,
                    generate_lc=run_cfg['generate_lc'],                 # whether to generate light curves
                    soss_inner_mask_width=run_cfg['soss_inner_mask_width'],
                    soss_outer_mask_width=run_cfg['soss_outer_mask_width'],
                    nirspec_mask_width=run_cfg['nirspec_mask_width'],
                    pixel_masks=run_cfg['outlier_maps'],
                    generate_order0_mask=run_cfg['generate_order0_mask'],
                    f277w=run_cfg['f277w'],                              # filter-specific processing
                    do_plot=run_cfg['do_plots'],
                    centroids=run_cfg['centroids'],
                    miri_trace_width=run_cfg['miri_trace_width'],
                    miri_background_width=run_cfg['miri_background_width'],
                    miri_background_method=run_cfg['miri_background_method'],
                    **run_cfg.get('stage2_kwargs', {}),
                    **s2_args
                )

                # If Stage 2 returns centroids as numpy array, convert to DataFrame
                if isinstance(centroids, np.ndarray):
                    centroids = pd.DataFrame(centroids.T, columns=["xpos", "ypos"])

                # -------------------------
                # Stage 3: 1D spectrum extraction
                # -------------------------

                always_skip3 = []
                # Determine Stage-3 steps to skip (none special)
                stage3_skip = get_stage_skips(
                    cfg,
                    stage3_steps,
                    always_skip=always_skip3,
                    special_one_over_f=False
                )

                # Use user-provided centroids if present; otherwise take from Stage 2
                this_centroid = cfg['centroids'] if cfg['centroids'] is not None else centroids

                # Run Stage 3 to get final extracted spectrum
                stage3_results = run_stage3(
                    stage2_results,
                    save_results=True,
                    force_redo=True,
                    extract_method=run_cfg['extract_method'],       # e.g., 'box', 'optimal'
                    soss_specprofile=run_cfg['soss_specprofile'],   # spectral profile file for SOSS
                    centroids=this_centroid,
                    extract_width=run_cfg['extract_width'],         # aperture width in pixels
                    st_teff=run_cfg['st_teff'],                     # stellar Teff
                    st_logg=run_cfg['st_logg'],                     # stellar logg
                    st_met=run_cfg['st_met'],                       # stellar metallicity
                    planet_letter=run_cfg['planet_letter'],         # planet letter (b, c, etc.)
                    output_tag=run_cfg['output_tag'],
                    do_plot=run_cfg['do_plots'],
                    skip_steps=stage3_skip,
                    **run_cfg.get('stage3_kwargs', {}),
                    **s3_args
                )

            
            else:
                if key in ('nirspec_mask_width', 'soss_inner_mask_width', 'soss_outer_mask_width'):
                    # --- Stage 1 on darkcurrent‐stepped files ---
                    always_skip1 = ['DQInitStep', 'EmiCorrStep', 'SaturationStep','ResetStep','SuperBiasStep','RefPixStep', 'DarkCurrentStep']
                    stage1_skip = get_stage_skips(
                        cfg,
                        stage1_steps,
                        always_skip=always_skip1,
                        special_one_over_f=True
                    )
                    
                    possible_steps_int1 = ["darkcurrentstep", "refpixstep", "superbiasstep", 'resetstep','saturationstep','emicorrstep','dqinitstep']

                    filenames_int1 = make_step_filenames(
                        input_files,
                        output_dir=outdir_s1,
                        possible_steps=possible_steps_int1
                    )




                    stage1_results = run_stage1(
                        filenames_int1,
                        mode=run_cfg['observing_mode'],
                        soss_background_model=run_cfg['soss_background_file'],
                        baseline_ints=run_cfg['baseline_ints'],
                        oof_method=run_cfg['oof_method'],
                        superbias_method=run_cfg['superbias_method'],
                        soss_timeseries=run_cfg['soss_timeseries'],
                        soss_timeseries_o2=run_cfg['soss_timeseries_o2'],
                        save_results=True,
                        pixel_masks=run_cfg['outlier_maps'],
                        force_redo=True,
                        flag_up_ramp=run_cfg['flag_up_ramp'],
                        rejection_threshold=run_cfg['jump_threshold'],
                        flag_in_time=run_cfg['flag_in_time'],
                        time_rejection_threshold=run_cfg['time_jump_threshold'],
                        output_tag=run_cfg['output_tag'],
                        skip_steps=stage1_skip,
                        do_plot=False,
                        soss_inner_mask_width=run_cfg['soss_inner_mask_width'],
                        soss_outer_mask_width=run_cfg['soss_outer_mask_width'],
                        nirspec_mask_width=run_cfg['nirspec_mask_width'],
                        centroids=run_cfg['centroids'],
                        hot_pixel_map=run_cfg['hot_pixel_map'],
                        miri_drop_groups=run_cfg['miri_drop_groups'],
                        **run_cfg.get('stage1_kwargs', {}),
                        **s1_args
                    )
                    

                    # --- Stage 2 on those results ---
                    always_skip2 = []
                    stage2_skip = get_stage_skips(
                        cfg,
                        stage2_steps,
                        always_skip=always_skip2,
                        special_one_over_f=False
                    )
                    
                    stage2_results, centroids = run_stage2(
                        stage1_results,
                        mode=run_cfg['observing_mode'],
                        soss_background_model=run_cfg['soss_background_file'],
                        baseline_ints=run_cfg['baseline_ints'],
                        save_results=True,
                        force_redo=True,
                        space_thresh=run_cfg['space_outlier_threshold'],
                        time_thresh=run_cfg['time_outlier_threshold'],
                        remove_components=run_cfg['remove_components'],
                        pca_components=run_cfg['pca_components'],
                        soss_timeseries=run_cfg['soss_timeseries'],
                        soss_timeseries_o2=run_cfg['soss_timeseries_o2'],
                        oof_method=run_cfg['oof_method'],
                        output_tag=run_cfg['output_tag'],
                        smoothing_scale=run_cfg['smoothing_scale'],
                        skip_steps=stage2_skip,
                        generate_lc=run_cfg['generate_lc'],
                        soss_inner_mask_width=run_cfg['soss_inner_mask_width'],
                        soss_outer_mask_width=run_cfg['soss_outer_mask_width'],
                        nirspec_mask_width=run_cfg['nirspec_mask_width'],     
                        pixel_masks=run_cfg['outlier_maps'],
                        generate_order0_mask=run_cfg['generate_order0_mask'],
                        f277w=run_cfg['f277w'],
                        do_plot=False,
                        centroids=run_cfg['centroids'],
                        miri_trace_width=run_cfg['miri_trace_width'],
                        miri_background_width=run_cfg['miri_background_width'],
                        miri_background_method=run_cfg['miri_background_method'],
                        **run_cfg.get('stage2_kwargs', {}),
                        **s2_args
                    )
                    if isinstance(centroids, np.ndarray):
                        centroids = pd.DataFrame(centroids.T, columns=["xpos","ypos"])
                    
                    # --- Stage 3 on those results ---
                    always_skip3 = []
                    stage3_skip = get_stage_skips(
                        cfg,
                        stage3_steps,
                        always_skip=always_skip3,
                        special_one_over_f=False
                    )
               
                    this_centroid = cfg['centroids'] if cfg['centroids'] is not None else centroids
                    stage3_results = run_stage3(
                        stage2_results,
                        save_results=True,
                        force_redo=True,
                        extract_method=run_cfg['extract_method'],
                        soss_specprofile=run_cfg['soss_specprofile'],
                        centroids=this_centroid,
                        extract_width=run_cfg['extract_width'],
                        st_teff=run_cfg['st_teff'],
                        st_logg=run_cfg['st_logg'],
                        st_met=run_cfg['st_met'],
                        planet_letter=run_cfg['planet_letter'],
                        output_tag=run_cfg['output_tag'],
                        do_plot=False,
                        skip_steps=stage3_skip,
                        **run_cfg.get('stage3_kwargs', {}),
                        **s3_args
                    )                        


                elif key in ('time_jump_threshold', 'jump_threshold','time_rejection_threshold', 'time_window'):

                    # --- Stage 1 on the “linearized” intermediates ---
                    always_skip1 = ['DQInitStep', 'EmiCorrStep', 'SaturationStep','ResetStep','SuperBiasStep','RefPixStep', 'DarkCurrentStep',
                                    'OneOverFStep', 'LinearityStep']
                    stage1_skip = get_stage_skips(
                        cfg,
                        stage1_steps,
                        always_skip=always_skip1,
                        special_one_over_f=True
                    )

                    possible_steps_int2 = ['linearitystep','oneoverfstep',"darkcurrentstep", "refpixstep", "superbiasstep", 'resetstep','saturationstep','emicorrstep','dqinitstep']

                    filenames_int2 = make_step_filenames(
                        input_files,
                        output_dir=outdir_s1,
                        possible_steps=possible_steps_int2,
                    )                    

                    
                    stage1_results = run_stage1(
                        filenames_int2,
                        mode=run_cfg['observing_mode'],
                        soss_background_model=run_cfg['soss_background_file'],
                        baseline_ints=run_cfg['baseline_ints'],
                        oof_method=run_cfg['oof_method'],
                        superbias_method=run_cfg['superbias_method'],
                        soss_timeseries=run_cfg['soss_timeseries'],
                        soss_timeseries_o2=run_cfg['soss_timeseries_o2'],
                        save_results=True,
                        pixel_masks=run_cfg['outlier_maps'],
                        force_redo=True,
                        flag_up_ramp=run_cfg['flag_up_ramp'],
                        rejection_threshold=run_cfg['jump_threshold'],
                        flag_in_time=run_cfg['flag_in_time'],
                        time_rejection_threshold=run_cfg['time_jump_threshold'],
                        output_tag=run_cfg['output_tag'],
                        skip_steps=stage1_skip,
                        do_plot=False,
                        soss_inner_mask_width=run_cfg['soss_inner_mask_width'],
                        soss_outer_mask_width=run_cfg['soss_outer_mask_width'],
                        nirspec_mask_width=run_cfg['nirspec_mask_width'],
                        centroids=run_cfg['centroids'],
                        hot_pixel_map=run_cfg['hot_pixel_map'],
                        miri_drop_groups=run_cfg['miri_drop_groups'],
                        **run_cfg.get('stage1_kwargs', {}),
                        **s1_args
                    )
                   

                    # --- Stage 2 on those results ---
                    always_skip2 = []
                    stage2_skip = get_stage_skips(
                        cfg,
                        stage2_steps,
                        always_skip=always_skip2,
                        special_one_over_f=False
                    )

                    
                    stage2_results, centroids = run_stage2(
                        stage1_results,
                        mode=run_cfg['observing_mode'],
                        soss_background_model=run_cfg['soss_background_file'],
                        baseline_ints=run_cfg['baseline_ints'],
                        save_results=True,
                        force_redo=True,
                        space_thresh=run_cfg['space_outlier_threshold'],
                        time_thresh=run_cfg['time_outlier_threshold'],
                        remove_components=run_cfg['remove_components'],
                        pca_components=run_cfg['pca_components'],
                        soss_timeseries=run_cfg['soss_timeseries'],
                        soss_timeseries_o2=run_cfg['soss_timeseries_o2'],
                        oof_method=run_cfg['oof_method'],
                        output_tag=run_cfg['output_tag'],
                        smoothing_scale=run_cfg['smoothing_scale'],
                        skip_steps=stage2_skip,
                        generate_lc=run_cfg['generate_lc'],
                        soss_inner_mask_width=run_cfg['soss_inner_mask_width'],
                        soss_outer_mask_width=run_cfg['soss_outer_mask_width'],
                        nirspec_mask_width=run_cfg['nirspec_mask_width'],
                        pixel_masks=run_cfg['outlier_maps'],
                        generate_order0_mask=run_cfg['generate_order0_mask'],
                        f277w=run_cfg['f277w'],
                        do_plot=False,
                        centroids=run_cfg['centroids'],
                        miri_trace_width=run_cfg['miri_trace_width'],
                        miri_background_width=run_cfg['miri_background_width'],
                        miri_background_method=run_cfg['miri_background_method'],
                        **run_cfg.get('stage2_kwargs', {}),
                        **s2_args
                    )
                    if isinstance(centroids, np.ndarray):
                        centroids = pd.DataFrame(centroids.T, columns=["xpos","ypos"])                        
             

                    # --- Stage 3 on those results ---
                    always_skip3 = []
                    stage3_skip = get_stage_skips(
                        cfg,
                        stage3_steps,
                        always_skip=always_skip3,
                        special_one_over_f=False
                    )

                  
                    this_centroid = cfg['centroids'] if cfg['centroids'] is not None else centroids
                    stage3_results = run_stage3(
                        stage2_results,
                        save_results=True,
                        force_redo=True,
                        extract_method=run_cfg['extract_method'],
                        soss_specprofile=run_cfg['soss_specprofile'],
                        centroids=this_centroid,
                        extract_width=run_cfg['extract_width'],
                        st_teff=run_cfg['st_teff'],
                        st_logg=run_cfg['st_logg'],
                        st_met=run_cfg['st_met'],
                        planet_letter=run_cfg['planet_letter'],
                        output_tag=run_cfg['output_tag'],
                        do_plot=False,
                        skip_steps=stage3_skip,
                        **run_cfg.get('stage3_kwargs', {}),
                        **s3_args
                    )




                elif key in ('miri_trace_width', 'miri_background_width'):
                    # Stage 2 on precomputed Stage-1 intermediates (filenames_int3)
                    always_skip2 = ['AssignWCSStep', 'Extract2DStep', 'SourceTypeStep', 'WaveCorrStep', 'FlatFieldStep']
                    stage2_skip = get_stage_skips(
                        cfg,
                        stage2_steps,
                        always_skip=always_skip2,
                        special_one_over_f=False
                    )


                    possible_steps_int3 = ['flatfieldstep','wavecorrstep','sourcetypestep','extract2dstep','assignwcsstep']

                    filenames_int3 = make_step_filenames(
                        input_files,
                        output_dir=outdir_s2,   # Stage 2 outputs
                        possible_steps=possible_steps_int3,
                        output_dir_2nd=outdir_s1,   # Fall back to Stage 1 outputs
                        possible_steps_2nd=["gainscalestep", 'rampfitstep', 'jumpstep']
                    )


                   
                    stage2_results, centroids = run_stage2(
                        filenames_int3,
                        mode=run_cfg['observing_mode'],
                        baseline_ints=run_cfg['baseline_ints'],
                        save_results=True,
                        force_redo=True,
                        space_thresh=run_cfg['space_outlier_threshold'],
                        time_thresh=run_cfg['time_outlier_threshold'],
                        remove_components=run_cfg['remove_components'],
                        pca_components=run_cfg['pca_components'],
                        soss_timeseries=run_cfg['soss_timeseries'],
                        soss_timeseries_o2=run_cfg['soss_timeseries_o2'],
                        oof_method=run_cfg['oof_method'],
                        output_tag=run_cfg['output_tag'],
                        smoothing_scale=run_cfg['smoothing_scale'],
                        skip_steps=stage2_skip,
                        generate_lc=run_cfg['generate_lc'],
                        soss_inner_mask_width=run_cfg['soss_inner_mask_width'],
                        soss_outer_mask_width=run_cfg['soss_outer_mask_width'],
                        nirspec_mask_width=run_cfg['nirspec_mask_width'],
                        pixel_masks=run_cfg['outlier_maps'],
                        generate_order0_mask=run_cfg['generate_order0_mask'],
                        f277w=run_cfg['f277w'],
                        do_plot=False,
                        centroids=run_cfg['centroids'],
                        miri_trace_width=run_cfg['miri_trace_width'],
                        miri_background_width=run_cfg['miri_background_width'],
                        miri_background_method=run_cfg['miri_background_method'],
                        **run_cfg.get('stage2_kwargs', {}),
                        **s2_args
                    )
                    if isinstance(centroids, np.ndarray):
                        centroids = pd.DataFrame(centroids.T, columns=["xpos","ypos"])                        
    
                    # Stage 3
                    always_skip3 = []
                    stage3_skip = get_stage_skips(
                        cfg,
                        stage3_steps,
                        always_skip=always_skip3,
                        special_one_over_f=False
                    )

                    this_centroid = cfg['centroids'] if cfg['centroids'] is not None else centroids
                    stage3_results = run_stage3(
                        stage2_results,
                        save_results=True,
                        force_redo=True,
                        extract_method=run_cfg['extract_method'],
                        soss_specprofile=run_cfg['soss_specprofile'],
                        centroids=this_centroid,
                        extract_width=run_cfg['extract_width'],
                        st_teff=run_cfg['st_teff'],
                        st_logg=run_cfg['st_logg'],
                        st_met=run_cfg['st_met'],
                        planet_letter=run_cfg['planet_letter'],
                        output_tag=run_cfg['output_tag'],
                        do_plot=False,
                        skip_steps=stage3_skip,
                        **run_cfg.get('stage3_kwargs', {}),
                        **s3_args
                    )


                elif key in ('space_outlier_threshold', 'space_thresh', 'time_outlier_threshold', 'time_thresh','box_size', 'window_size'):
                    # Stage 2 on precomputed Stage-1 intermediates (filenames_int3)
                    always_skip2 =  ['AssignWCSStep', 'Extract2DStep', 'SourceTypeStep', 'WaveCorrStep', 'FlatFieldStep','BackgroundStep','OneOverFStep']
                    stage2_skip = get_stage_skips(
                        cfg,
                        stage2_steps,
                        always_skip=always_skip2,
                        special_one_over_f=False
                    )

                    possible_steps_int4 = ['oneoverfstep','backgroundstep','flatfieldstep','wavecorrstep','sourcetypestep','extract2dstep','assignwcsstep']

                    filenames_int4 = make_step_filenames(
                        input_files,
                        output_dir=outdir_s2,   # Stage 2 outputs
                        possible_steps=possible_steps_int4,
                        output_dir_2nd=outdir_s1,   # Fall back to Stage 1 outputs
                        possible_steps_2nd=["gainscalestep", 'rampfitstep', 'jumpstep']
                    )                    

                   
                    stage2_results, centroids = run_stage2(
                        filenames_int4,
                        mode=run_cfg['observing_mode'],
                        baseline_ints=run_cfg['baseline_ints'],
                        save_results=True,
                        force_redo=True,
                        space_thresh=run_cfg['space_outlier_threshold'],
                        time_thresh=run_cfg['time_outlier_threshold'],
                        remove_components=run_cfg['remove_components'],
                        pca_components=run_cfg['pca_components'],
                        soss_timeseries=run_cfg['soss_timeseries'],
                        soss_timeseries_o2=run_cfg['soss_timeseries_o2'],
                        oof_method=run_cfg['oof_method'],
                        output_tag=run_cfg['output_tag'],
                        smoothing_scale=run_cfg['smoothing_scale'],
                        skip_steps=stage2_skip,
                        generate_lc=run_cfg['generate_lc'],
                        soss_inner_mask_width=run_cfg['soss_inner_mask_width'],
                        soss_outer_mask_width=run_cfg['soss_outer_mask_width'],
                        nirspec_mask_width=run_cfg['nirspec_mask_width'],
                        pixel_masks=run_cfg['outlier_maps'],
                        generate_order0_mask=run_cfg['generate_order0_mask'],
                        f277w=run_cfg['f277w'],
                        do_plot=False,
                        centroids=run_cfg['centroids'],
                        miri_trace_width=run_cfg['miri_trace_width'],
                        miri_background_width=run_cfg['miri_background_width'],
                        miri_background_method=run_cfg['miri_background_method'],
                        **run_cfg.get('stage2_kwargs', {}),
                        **s2_args
                    )
                    if isinstance(centroids, np.ndarray):
                        centroids = pd.DataFrame(centroids.T, columns=["xpos","ypos"])                        
    
                    # Stage 3
                    always_skip3 = []
                    stage3_skip = get_stage_skips(
                        cfg,
                        stage3_steps,
                        always_skip=always_skip3,
                        special_one_over_f=False
                    )

                    this_centroid = cfg['centroids'] if cfg['centroids'] is not None else centroids
                    stage3_results = run_stage3(
                        stage2_results,
                        save_results=True,
                        force_redo=True,
                        extract_method=run_cfg['extract_method'],
                        soss_specprofile=run_cfg['soss_specprofile'],
                        centroids=this_centroid,
                        extract_width=run_cfg['extract_width'],
                        st_teff=run_cfg['st_teff'],
                        st_logg=run_cfg['st_logg'],
                        st_met=run_cfg['st_met'],
                        planet_letter=run_cfg['planet_letter'],
                        output_tag=run_cfg['output_tag'],
                        do_plot=False,
                        skip_steps=stage3_skip,
                        **run_cfg.get('stage3_kwargs', {}),
                        **s3_args
                    )
               


                elif key == 'extract_width':
                    # Stage 2 on precomputed Stage-1 intermediates (filenames_int4)
                    always_skip2 =  ['AssignWCSStep', 'Extract2DStep', 'SourceTypeStep', 'WaveCorrStep', 'FlatFieldStep','BackgroundStep','OneOverFStep']
                    stage2_skip = get_stage_skips(
                        cfg,
                        stage2_steps,
                        always_skip=always_skip2,
                        special_one_over_f=False
                    )


                    possible_steps_int5 = ['tracingstep','pcareconstructstep','badpixstep','oneoverfstep','backgroundstep','flatfieldstep','wavecorrstep','sourcetypestep','extract2dstep','assignwcsstep']

                    filenames_int5 = make_step_filenames(
                        input_files,
                        output_dir=outdir_s2,   # Stage 2 outputs
                        possible_steps=possible_steps_int5,
                        output_dir_2nd=outdir_s1,   # Fall back to Stage 1 outputs
                        possible_steps_2nd=["gainscalestep", 'rampfitstep', 'jumpstep']
                    )                            

           
                    stage2_results, centroids = run_stage2(
                        filenames_int5,
                        mode=run_cfg['observing_mode'],
                        baseline_ints=run_cfg['baseline_ints'],
                        save_results=True,
                        force_redo=True,
                        space_thresh=run_cfg['space_outlier_threshold'],
                        time_thresh=run_cfg['time_outlier_threshold'],
                        remove_components=run_cfg['remove_components'],
                        pca_components=run_cfg['pca_components'],
                        soss_timeseries=run_cfg['soss_timeseries'],
                        soss_timeseries_o2=run_cfg['soss_timeseries_o2'],
                        oof_method=run_cfg['oof_method'],
                        output_tag=run_cfg['output_tag'],
                        smoothing_scale=run_cfg['smoothing_scale'],
                        skip_steps=stage2_skip,
                        generate_lc=run_cfg['generate_lc'],
                        soss_inner_mask_width=run_cfg['soss_inner_mask_width'],
                        soss_outer_mask_width=run_cfg['soss_outer_mask_width'],
                        nirspec_mask_width=run_cfg['nirspec_mask_width'],
                        pixel_masks=run_cfg['outlier_maps'],
                        generate_order0_mask=run_cfg['generate_order0_mask'],
                        f277w=run_cfg['f277w'],
                        do_plot=False,
                        centroids=run_cfg['centroids'],
                        miri_trace_width=run_cfg['miri_trace_width'],
                        miri_background_width=run_cfg['miri_background_width'],
                        miri_background_method=run_cfg['miri_background_method'],
                        **run_cfg.get('stage2_kwargs', {}),
                        **s2_args
                    )
                    if isinstance(centroids, np.ndarray):
                        centroids = pd.DataFrame(centroids.T, columns=["xpos","ypos"])                        
                  

                    # Stage 3 with trial-specific extract_width
                    always_skip3 = []
                    stage3_skip = get_stage_skips(
                        cfg,
                        stage3_steps,
                        always_skip=always_skip3,
                        special_one_over_f=False
                    )

           
                    this_centroid = cfg['centroids'] if cfg['centroids'] is not None else centroids
                    stage3_results = run_stage3(
                        stage2_results,
                        save_results=True,
                        force_redo=True,
                        extract_method=run_cfg['extract_method'],
                        soss_specprofile=run_cfg['soss_specprofile'],
                        centroids=this_centroid,
                        extract_width=run_cfg['extract_width'],
                        st_teff=run_cfg['st_teff'],
                        st_logg=run_cfg['st_logg'],
                        st_met=run_cfg['st_met'],
                        planet_letter=run_cfg['planet_letter'],
                        output_tag=run_cfg['output_tag'],
                        do_plot=False,
                        skip_steps=stage3_skip,
                        **run_cfg.get('stage3_kwargs', {}),
                        **s3_args
                    )
    

            # ----------------------------------------------------------------
            # Run cost function & log results
            # ----------------------------------------------------------------
            st2, st3 = stage2_results, stage3_results
            cost, scatter = cost_function(st3, w1=w1, w2=w2,
                                          baseline_ints=baseline_ints,
                                          wave_range=wave_range)
            dt = time.perf_counter() - t0
            fancyprint(f"cost = {cost:.12f} in {dt:.1f}s")

            logf.write(
                "\t".join(str(trial_params[k]) for k in param_order)
                + f"\t{dt:.1f}\t{cost:.12f}\n"
            )
            line = " ".join(f"{x:.10g}" for x in scatter)
            logs.write(line + "\n")

            # Update best if cost improved
            if best_cost is None or cost < best_cost:
                best_cost, best_val = cost, trial
                diagnostic_plot(st3, name_str,
                                baseline_ints=baseline_ints,
                                outdir=outdir_f)

            print(
                "\n############################################",
                f"\n Iteration: {count}/{total_steps} completed (dt={dt:.1f}s)",
                "\n############################################\n",
                flush=True
            )     
 
            count += 1

        # Commit best value for this parameter before moving on
        current[key] = best_val
        fancyprint(f"Best {key} = {best_val} (cost={best_cost:.12f})")

    # ------------------------------------------------------
    # After loop: final reporting & fast validation run
    # ------------------------------------------------------

    # Compute total runtime and print it in h:mm:ss format
    t1 = time.perf_counter() - t0_total
    h, m = divmod(int(t1), 3600)
    m, s = divmod(m, 60)
    fancyprint(f"TOTAL runtime: {h}h {m:02d}min {s:04.1f}s")

    # Close log files
    logf.close()
    logs.close()

    # Print final set of optimized parameters from coordinate descent
    fancyprint("=== FINAL OPTIMUM ===")
    fancyprint(current)

    # Plot cost vs. parameter sweep history
    plot_cost(name_str)

    # Identify global best row from cost table
    cost_file = os.path.join(outdir_f, f"Cost_{name_str}.txt")
    df_cost   = pd.read_csv(cost_file, sep="\t")
    idx_min   = df_cost['cost'].idxmin()       # index of lowest cost
    best_row  = df_cost.loc[idx_min]           # the row with the best cost

    # Extract only the parameters that were swept (convert floats to int if they’re whole numbers)
    best_params = {
        col: int(best_row[col]) if float(best_row[col]).is_integer() else best_row[col]
        for col in param_order
    }
    fancyprint(f"Global best from cost table (row {idx_min}): {best_params}")

    # Merge best swept params with fixed params
    full_best = fixed_params.copy()
    full_best.update(best_params)

    # Make a new config containing the optimized parameter set
    final_cfg = cfg.copy()
    final_cfg.update(full_best)

    # --------------------------------------------------------------------
    # Fast final validation: only Stage 2 + Stage 3 on precomputed Stage 1
    # --------------------------------------------------------------------
    fancyprint("Running fast final validation: only Stage 2 + Stage 3…")

    # Stage 2 will use Stage-1 results already saved to disk
    stage2_skip = []

    # Define which Stage-2 steps to look for in filenames
    possible_steps_int6 = [
        'tracingstep','pcareconstructstep','badpixstep','oneoverfstep',
        'backgroundstep','flatfieldstep','wavecorrstep','sourcetypestep',
        'extract2dstep','assignwcsstep'
    ]

    # Find Stage-2 input files (fall back to Stage-1 outputs if needed)
    filenames_int6 = make_step_filenames(
        input_files,
        output_dir=outdir_s2,       # primary search location (Stage 2 outputs)
        possible_steps=possible_steps_int6,
        output_dir_2nd=outdir_s1,   # fallback location (Stage 1 outputs)
        possible_steps_2nd=["gainscalestep", 'rampfitstep', 'jumpstep']
    )

    # Run Stage 2 on precomputed Stage 1 outputs using final best parameters
    stage2_results, centroids = run_stage2(
        filenames_int6,
        mode=final_cfg['observing_mode'],
        baseline_ints=final_cfg['baseline_ints'],
        save_results=True,
        force_redo=True,  
        space_thresh=final_cfg['space_outlier_threshold'],
        time_thresh=final_cfg['time_outlier_threshold'],
        remove_components=final_cfg['remove_components'],
        pca_components=final_cfg['pca_components'],
        soss_timeseries=final_cfg['soss_timeseries'],
        soss_timeseries_o2=final_cfg['soss_timeseries_o2'],
        oof_method=final_cfg['oof_method'],
        output_tag=final_cfg['output_tag'],
        smoothing_scale=final_cfg['smoothing_scale'],
        skip_steps=stage2_skip,
        generate_lc=final_cfg['generate_lc'],
        soss_inner_mask_width=final_cfg['soss_inner_mask_width'],
        soss_outer_mask_width=final_cfg['soss_outer_mask_width'],
        nirspec_mask_width=final_cfg['nirspec_mask_width'],
        pixel_masks=final_cfg['outlier_maps'],
        generate_order0_mask=final_cfg['generate_order0_mask'],
        f277w=final_cfg['f277w'],
        do_plot=run_cfg['do_plots'],
        centroids=final_cfg['centroids'],
        miri_trace_width=final_cfg['miri_trace_width'],
        miri_background_width=final_cfg['miri_background_width'],
        miri_background_method=final_cfg['miri_background_method'],
        **final_cfg.get('stage2_kwargs', {})
    )

    # Convert centroids to DataFrame if returned as numpy array
    if isinstance(centroids, np.ndarray):
        centroids = pd.DataFrame(centroids.T, columns=["xpos","ypos"])

    # Use config-provided centroids if available; otherwise, use output from Stage 2
    final_centroids = final_cfg['centroids'] if final_cfg['centroids'] is not None else centroids
    print(final_centroids)

    # Run Stage 3 with best extraction width and other Stage 3 parameters
    stage3_results = run_stage3(
        stage2_results, 
        save_results=True,
        force_redo=True,
        extract_method=final_cfg['extract_method'],
        soss_specprofile=final_cfg['soss_specprofile'],
        centroids=final_centroids,
        extract_width=final_cfg['extract_width'],
        st_teff=final_cfg['st_teff'],
        st_logg=final_cfg['st_logg'],
        st_met=final_cfg['st_met'],
        planet_letter=final_cfg['planet_letter'],
        output_tag=final_cfg['output_tag'],
        do_plot=run_cfg['do_plots'],
        skip_steps=[], 
        **final_cfg.get('stage3_kwargs', {})
    )

    # Generate diagnostic plots for the final Stage 3 output
    diagnostic_plot(stage3_results, name_str, baseline_ints=baseline_ints, outdir=outdir_f)
    fancyprint("Final validation complete.")

    # ------------------------------------------------------
    # Visualize best scatter curve & photon noise floor
    # ------------------------------------------------------
    outfile  = os.path.join(outdir_f, f"Scatter_{name_str}.txt")
    specfile = glob.glob(os.path.join(outdir_s3, "*_box_spectra_fullres.fits"))[0]
    best_idx = pd.read_csv(os.path.join(outdir_f, f"Cost_{name_str}.txt"), sep="\t")['cost'].idxmin()

    # Plot best-scatter spectrum with smoothing applied
    plot_scatter(
        txtfile=outfile,
        rows=[best_idx],
        wave_range=wave_range_plot,
        smooth=10,
        spectrum_files=[specfile],
        ylim=ylim_plot,
        style="line",
        save_path=os.path.join(outdir_f, f"Scatter_Plot_{name_str}.png"),
    )

if __name__ == "__main__":
    main() 
