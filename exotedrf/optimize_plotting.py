#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 09 21:32 2026

@author: MCR, PSD, TRF

Optimizer plotting routines.
"""

from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

from exotedrf.optimize_utils import stitch_soss_orders
from exotedrf.plotting import make_2d_lightcurve_plot
from exotedrf.utils import fancyprint, format_out_frames


def make_lightcurve_plots(st3, baseline_ints, obs, outdir=None, name_str=None):
    """Create two diagnostic plots from Stage-3 data:
      1) Normalized white-light curve
      2) Normalized flux image with true wavelength mapping

    Parameters
    ----------
    st3 : dict-like
        Stage-3 outputs containing flux and wavelength arrays.
        For NIRISS/SOSS: requires 'Flux_O1', 'Flux_O2', 'Wave_O1', 'Wave_O2'.
        For others: requires 'Flux', 'Wave'.
    name_str : str, None
        Identifier used in output filenames.
    baseline_ints : list[int]
        One or two integers for baseline integrations:
            [nn] -> normalize by median of first nn integrations
            [nlow, nhigh] -> normalize by mean of medians of start and end segments
    obs : str
        Observing mode identifier.
    filter : str, None
        Filter/detector identifier.
    outdir : str, None
        Output directory for saved figures.
    """

    if 'niriss' in obs.lower():
        wave = st3['Wave O1']
        flux = st3['Flux O1']
        wave2 = st3['Wave O2']
        flux2 = st3['Flux O2']
        # If SUBSTRIP96, all flux2 values should be zero. None these to not appear in plots.
        if np.nansum(flux2) == 0:
            flux2 = None
            wave2 = None
        # If SUBSTRIP256, trim to only show relevant wavelengths.
        else:
            ii = np.where((wave2 >= 0.6) & (wave2 < 0.85))[0]
            wave2 = wave2[ii]
            flux2 = flux2[:, ii]
    else:
        wave = st3['Wave']
        flux = st3['Flux']
        wave2 = None
        flux2 = None

    # Make sure all wavelengths are valid.
    ii = np.where(np.isfinite(wave))[0]
    wave = wave[ii]
    flux = flux[:, ii]

    # Plot white light curve
    plot_whitelight(flux, baseline_ints, outdir, name_str)

    # Plot Spectroscopic light curves
    base = format_out_frames(baseline_ints)
    flux_norm = flux / np.nanmedian(flux[base], axis=0)
    if flux2 is not None:
        flux2_norm = flux2 / np.nanmedian(flux2[base], axis=0)
    else:
        flux2_norm = None
    inst = obs.split('/')[0]

    if name_str is not None:
        plot_path = f'{outdir}/2D_LightCurves{name_str}.png'
    else:
        plot_path = None

    make_2d_lightcurve_plot(wave, flux_norm, wave2, flux2_norm, instrument=inst, outpdf=plot_path)


def plot_cost(name_str, outdir, table_height=0.4):
    """Reads a tab-delimited cost file, detects parameter sweeps, highlights
    the best parameter set(s), and produces a figure showing cost trends.

    Parameters
    ----------
    name_str : str
        Identifier used to find the cost file (Cost_<name_str>.txt).
    table_height : float
        Fraction of the figure height to allocate to the table display.
    outdir : str
        Direcory to which to save outputs.
    """

    df = pd.read_csv(f'{outdir}/Cost_Summary{name_str}.txt',
                     delimiter='\t', keep_default_na=False)

    # Remove rows where 'cost' is not numeric, then keep the surviving values
    # numeric so per-sweep normalization does arithmetic instead of string math.
    df['cost'] = pd.to_numeric(df['cost'], errors='coerce')
    df = df[df['cost'].notna()].reset_index(drop=True)
    if df.empty:
        raise ValueError(f'No finite numeric costs found in {outdir}/Cost_Summar{name_str}.txt')

    # Get all parameter columns (exclude 'duration_s' and 'cost' at the end)
    param_cols = df.columns[:-2]

    # detect which parameter changed per row
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

    # Find sweep boundaries: as soon as any other parameter changes, the next sweep starts
    sweep_lines = []
    for i in range(1, len(df)):
        diffs = [c for c in param_cols if df.at[i, c] != df.at[i - 1, c]]
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

    # First row label belongs to the first detected sweep
    if len(df) >= 2 and changed_param_per_row[0] is None:
        changed_param_per_row[0] = changed_param_per_row[1] or param_cols[0]

    # Labels and sweep boundaries
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

        labels.append(f'{changed_param}={value}')
        last_changed_param = changed_param

    df['changed_label'] = labels

    # Normalize cost and highlight best
    sweep_boundaries = [0] + sweep_lines + [len(df)]
    colors = ['gray'] * len(df)  # default colour
    normalized_costs = np.zeros(len(df))

    for i in range(len(sweep_boundaries) - 1):
        start = sweep_boundaries[i]
        end = sweep_boundaries[i + 1]

        # Get costs for this sweep
        sweep_costs = df.iloc[start:end]['cost'].values

        # normalize to [0, 1] within this sweep
        min_cost = np.nanmin(sweep_costs)
        max_cost = np.nanmax(sweep_costs)
        if max_cost > min_cost:
            # scale to [0, 1]: 0 = best (lowest cost), 1 = worst (highest cost)
            normalized_sweep = (sweep_costs - min_cost) / (max_cost - min_cost)
        else:
            # all costs are the same in this sweep
            normalized_sweep = np.zeros(len(sweep_costs))

        normalized_costs[start:end] = normalized_sweep

        # highlight the best (minimum cost) in this sweep
        min_idx = df.iloc[start:end]['cost'].idxmin()
        colors[min_idx] = 'royalblue'

    best_row = df.loc[df['cost'].idxmin(), param_cols.tolist() + ['cost']].copy()
    for col in best_row.index:
        val = best_row[col]
        try:
            fv = float(val)
            best_row[col] = int(fv) if fv.is_integer() else fv
        except Exception:
            best_row[col] = val
    best_df = pd.DataFrame([best_row]).reset_index(drop=True)

    fig = plt.figure(figsize=(max(14, len(df) * 0.25), 10))
    gs = GridSpec(nrows=2, ncols=1, height_ratios=[1 - table_height, table_height])
    ax_plot = fig.add_subplot(gs[0])
    ax_table = fig.add_subplot(gs[1])

    ax_plot.scatter(range(len(df)), normalized_costs, color=colors)
    for x in sweep_lines:
        ax_plot.axvline(x=x - 0.5, color='gray', linestyle='--', linewidth=1)

    values = [lbl.split('=', 1)[1] for lbl in df['changed_label']]
    ax_plot.set_xticks(range(len(df)))
    ax_plot.set_xticklabels(values, rotation=0, fontsize=8)

    ymin, ymax = ax_plot.get_ylim()
    base_y = ymin - 0.08 * (ymax - ymin)
    alt_y = ymin - 0.15 * (ymax - ymin)
    for i, (start, end) in enumerate(zip(sweep_boundaries[:-1], sweep_boundaries[1:])):
        param_name = df.loc[start, 'changed_label'].split('=', 1)[0]
        center = (start + end - 1) / 2
        y_pos = base_y if i % 2 == 0 else alt_y
        ax_plot.text(center, y_pos, param_name, ha='center', va='top', fontsize=10)

    fig.subplots_adjust(bottom=0.30)
    ax_plot.set_ylabel('Relative Cost (normalized per sweep)')
    ax_plot.set_title(f'Cost by Single Parameter Sweep: {name_str}')
    ax_plot.set_ylim(-0.05, 1.05)

    # ======== TABLE OF BEST PARAMETERS ========
    ax_table.axis('off')
    ax_table.text(0.5, 0.65, 'Best Parameters', ha='center', va='bottom', fontsize=12)
    table = ax_table.table(cellText=best_df.values, colLabels=best_df.columns, cellLoc='center',
                           loc='center')
    table.scale(1.0, 1.8)
    table.auto_set_font_size(False)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_fontsize(7)  # header
        else:
            cell.set_fontsize(10)  # data

    fig.savefig(f'{outdir}/Cost_Summary{name_str}.png',
                dpi=300, bbox_inches='tight')


def plot_scatter(txtfile, rows, wave_range=None, smooth=1, spectrum_files=None, ylim=None,
                 save_path=None, tol=1e-5):
    """Plot point-to-point scatter vs wavelength for optimizer results.

    Parameters
    ----------
    txtfile : str
        Path to the whitespace-delimited scatter table.
    rows : list[int]
        Indices of the table rows to plot. Negative indices count from the end.
    wave_range : tuple(float, float), optional
        Wavelength range to plot (μm), with tolerance `tol`.
    smooth : int
        Window size (in pixels) for moving-average smoothing.
    spectrum_files : list[str]
        List of spectrum FITS files to retrieve wavelength axis from.
    ylim : tuple(float, float), optional
        y-axis limits.
    save_path : str, None
        Filepath to which to save plot.
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
            print(f'Warning: row {r} out of range, skipping.')
    if not valid:
        raise ValueError('No valid rows to plot.')

    # --- Load wavelength grid to match scatter columns ---
    if not spectrum_files:
        raise ValueError('`spectrum_files` is required to read the wavelength axis.')
    with fits.open(spectrum_files[0]) as hdus:
        # Create dict mapping sanitized HDU names to HDU objects
        name_map = {h.name.replace(' ', '_'): h
                    for h in hdus if h.data is not None and h.name != 'PRIMARY'}

        # Special handling for NIRISS with two orders - plot separately
        if ('Wave_O1' in name_map) and ('Wave_O2' in name_map):
            # Scatter rows are on the stitched, wavelength-sorted axis used by cost_function.
            wave_o1 = np.asarray(name_map['Wave_O1'].data, float)
            wave_o2 = np.asarray(name_map['Wave_O2'].data, float)
            if wave_o1.ndim == 2:
                wave_o1 = np.nanmedian(wave_o1, axis=0)
                wave_o2 = np.nanmedian(wave_o2, axis=0)
            wave_stitched = stitch_soss_orders(wave_o1, wave_o2)[0]
            is_niriss_two_orders = True
            # Plot each order's segment of the stitched axis separately
            orders = [
                {'sel': wave_stitched <= 0.85, 'name': 'Order 2'},
                {'sel': wave_stitched > 0.85, 'name': 'Order 1'}
            ]
        else:
            # Fallback: read first extension array as wavelength grid
            wave_full = np.asarray(hdus[1].data, float)
            is_niriss_two_orders = False

    # --- Plot for NIRISS with two orders (side-by-side subplots) ---
    if is_niriss_two_orders:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        if wave_stitched.size != n_cols:
            raise ValueError(f'Stitched SOSS wavelength axis ({wave_stitched.size}) does not match '
                             f'scatter columns ({n_cols}).')
        for idx, order_info in enumerate(orders):
            ax = axes[idx]
            order_name = order_info['name']

            # Build mask for this order's segment and the wavelength range
            mask = order_info['sel'] & np.isfinite(wave_stitched)
            if wave_range is not None:
                wmin, wmax = wave_range
                mask &= (wave_stitched >= wmin - tol) & (wave_stitched <= wmax + tol)

            if not mask.any():
                fancyprint(f'Warning: No finite wavelengths in {order_name} within range '
                           f'{wave_range}', msg_type='WARNING')
                continue

            x = wave_stitched[mask]

            # Plot each valid row
            for i in valid:
                # Extract data for this order from scatter table
                y_full = df.iloc[i, :].to_numpy(float)
                y_raw = (y_full[mask]) * 1e6

                ax.plot(x, y_raw, linewidth=0.6, alpha=0.25, color='grey')

                # Apply smoothing
                y_sm = median_filter(y_raw, size=smooth, mode='reflect')
                ax.plot(x, y_sm, c='royalblue')

            ax.set_xlabel('Wavelength [μm]', fontsize=12)
            if idx == 0:
                ax.set_ylabel('Light Curve Scatter [ppm]', fontsize=12)
            if ylim is not None:
                ax.set_ylim(ylim)
            ax.grid(True, alpha=0.3)

        # Sync y-axes
        low1, up1 = axes[1].get_ylim()
        low2, up2 = axes[0].get_ylim()
        ymax = np.max([up1, up2])
        ymin = np.min([low1, low2])
        axes[0].set_ylim(ymin, ymax)
        axes[1].set_ylim(ymin, ymax)
        axes[1].yaxis.set_major_formatter(plt.NullFormatter())

    # --- Plot for single-order instruments ---
    else:
        # Sort wavelengths
        s = np.argsort(wave_full, kind='mergesort')
        wave_sorted = wave_full[s]

        # Check size match
        if wave_sorted.size != n_cols:
            min_size = min(wave_sorted.size, n_cols)
            fancyprint(
                f'WARNING: Wavelength array size ({wave_sorted.size}) != scatter columns '
                f'({n_cols}).\n Truncating to {min_size} elements.', msg_type='WARNING'
            )
            wave_sorted = wave_sorted[:min_size]
            s = s[:min_size]

        # Build boolean mask for desired wavelength range
        if wave_range is not None:
            wmin, wmax = wave_range
            mask = np.isfinite(wave_sorted) & (wave_sorted >= wmin - tol) & (wave_sorted <= wmax + tol)
        else:
            mask = np.isfinite(wave_sorted)
        if not mask.any():
            raise ValueError(f'No finite wavelengths within selected range {wave_range}.')

        # Final x-axis values
        x = wave_sorted[mask]

        # --- Plot ---
        f, ax = plt.subplots(figsize=(8, 4))

        for i in valid:
            # Extract row data and reorder columns to match wavelength order
            y_full = df.iloc[i, :].to_numpy(float)
            y_ord = y_full[s]

            # Raw series (convert to ppm)
            y_raw = (y_ord[mask]) * 1e6
            plt.plot(x, y_raw, linewidth=0.6, alpha=0.5)

            # Smoothed series (moving average)
            if smooth and int(smooth) > 1:
                w = int(smooth)
                kern = np.ones(w, dtype=float) / w
                y_sm_all = np.convolve(y_ord, kern, mode='same')
                y_sm = (y_sm_all[mask]) * 1e6
                plt.plot(x, y_sm)

        # --- Finalize plot ---
        ax.set_xlim(np.min(x), np.max(x))
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.set_xlabel('Wavelength [μm]', fontsize=12)
        ax.set_ylabel('Light Curve Scatter [ppm]', fontsize=12)
        ax.grid(True, alpha=0.3)

    # Save or display figure.
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_whitelight(flux, baseline_ints, outdir, name_str):
    """Make a plot of the transit white light curve.
    """

    # Sum flux over all spectral channels for each integration
    white = np.nansum(flux, axis=1)
    base = format_out_frames(baseline_ints)
    norm_white = white / np.nanmedian(white[base])

    # --- Plot normalized white-light curve ---
    plt.figure()
    plt.errorbar(np.arange(len(norm_white)), norm_white, c='black', mfc='white', fmt='o', ms=3)
    plt.xlabel('Integration Number', fontsize=12)
    plt.ylabel('Normalized White Flux', fontsize=12)
    if name_str is not None:
        plt.savefig(f'{outdir}/WhiteLightCurve{name_str}.png')
        plt.close()
    else:
        plt.show()
