#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper functions for the optimize.py script with handling of 
DQ flags and extraction.
"""

import numpy as np
import pandas as pd
from astropy.io import fits
from tqdm import tqdm
import os

from exotedrf import utils
from exotedrf.utils import fancyprint
from exotedrf.stage3 import get_wave_soss, trace_spectrum, do_two_gaussian_extraction
import matplotlib.pyplot as plt


def _parse_width(width):
    """Normalize symmetric and asymmetric widths for optimizer-side extraction."""

    if isinstance(width, dict):
        if 'lower' not in width or 'upper' not in width:
            raise ValueError('Width dictionaries must contain "lower" and "upper" keys.')
        return float(width['lower']), float(width['upper'])
    if np.isscalar(width):
        half_width = float(width) / 2
        return half_width, half_width

    try:
        lower_width, upper_width = width
    except (TypeError, ValueError):
        raise ValueError('width must be a scalar, a two-element sequence, or a dictionary with '
                         '"lower"/"upper" keys.')
    return float(lower_width), float(upper_width)


def apply_dq_flags(datafiles):
    """
    Load data and apply DQ flags by NaN-ing out bad pixels.
    Errors are NOT loaded/returned since they're not needed for optimization.

    Parameters
    datafiles

    Returns
    cube  : array
        Flux with bad pixels as NaN
    is_4d : bool
        True if pre-RampFit (4D), False if post (3D)
    """
    datafiles = np.atleast_1d(datafiles)

    # get flux and DQ (errors not needed for optimization)
    for i, file in enumerate(datafiles):
        fancyprint(f'Loading segment {i}: {file if isinstance(file, str) else "datamodel"}')

        if isinstance(file, str):
            data = fits.getdata(file, 1)
            dq = fits.getdata(file, 3)
            fancyprint(f'  Loaded from FITS: data.shape={data.shape}, dq.shape={dq.shape}')
        else:
            with utils.open_filetype(file) as datamodel:
                data = datamodel.data
                dq = datamodel.dq
                fancyprint(f'  Loaded from datamodel: data.shape={data.shape}, dq.shape={dq.shape if dq is not None else None}')

        if dq is not None:
            # for 4D data (pre-rampfit), take last group
            is_4d = data.ndim == 4
            fancyprint(f'  is_4d={is_4d}, data.ndim={data.ndim}')

            if is_4d:
                # data shape: (nint, ngroup, y, x)
                # dq shape: (nint, ngroup, y, x) or (x, y, ngroup, nint) - need to check
                fancyprint(f'  4D processing: dq.shape={dq.shape}, data.shape={data.shape}')

                if dq.ndim == 4 and dq.shape[0] != data.shape[0]:
                    fancyprint(f'  Transposing DQ from {dq.shape} to match data')
                    dq = np.transpose(dq, (3, 2, 1, 0))
                    fancyprint(f'  After transpose: dq.shape={dq.shape}')

                # take last group for mask
                dq_for_mask = dq[:, -1, :, :]
                fancyprint(f'  Took last group: dq_for_mask.shape={dq_for_mask.shape}')

                # boolean mask - anything non-zero flag is bad
                bad_pixels = (dq_for_mask > 0).astype(bool)
                fancyprint(f'  bad_pixels (before broadcast).shape={bad_pixels.shape}')

                # expand to all groups
                bad_pixels = bad_pixels[:, np.newaxis, :, :]
                fancyprint(f'  After newaxis: bad_pixels.shape={bad_pixels.shape}')

                bad_pixels = np.broadcast_to(bad_pixels, data.shape)
                fancyprint(f'  After broadcast to data.shape: bad_pixels.shape={bad_pixels.shape}')
            else:
                # 3D data (post-RampFit) has shape (nint, y, x)
                fancyprint(f'  3D processing: dq.ndim={dq.ndim}, dq.shape={dq.shape}')

                if dq.ndim == 4:
                    fancyprint(f'  DQ is 4D, taking last group')
                    dq_for_mask = dq[:, -1, :, :]
                    fancyprint(f'  dq_for_mask.shape={dq_for_mask.shape}')
                elif dq.ndim == 3:
                    fancyprint(f'  DQ is 3D, using as-is')
                    dq_for_mask = dq
                elif dq.ndim == 2:
                    fancyprint(f'  DQ is 2D (PIXELDQ), broadcasting to data shape')
                    bad_pixels = (dq > 0).astype(bool)
                    bad_pixels = bad_pixels[np.newaxis, :, :]
                    bad_pixels = np.broadcast_to(bad_pixels, data.shape)
                    fancyprint(f'  bad_pixels.shape={bad_pixels.shape}')
                    dq_for_mask = None

                if dq_for_mask is not None:
                    bad_pixels = (dq_for_mask > 0).astype(bool)
                    fancyprint(f'  bad_pixels.shape={bad_pixels.shape}')

            # Apply mask
            fancyprint(f'  Applying mask: data.shape={data.shape}, bad_pixels.shape={bad_pixels.shape}')

            data[bad_pixels] = np.nan

            n_bad = np.sum(bad_pixels)
            fancyprint(f'Segment {i}: Flagged {n_bad}/{bad_pixels.size} pixels ({100*n_bad/bad_pixels.size:.2f}%)')
        else:
            fancyprint(f'Segment {i}: No DQ found', msg_type='WARNING')
            is_4d = data.ndim == 4

        # concatenate segments
        if i == 0:
            cube = data
        else:
            cube = np.concatenate([cube, data])

    return cube, is_4d


def do_box_extraction_nanaware(cube, ypos, width, extract_start=0, extract_end=None, progress=True):
    """
    Box extraction with nansum. Modified from stage3.do_box_extraction.
    Note: Errors are NOT calculated since they're not needed for optimization.

    Parameters
    cube :  (nint, y, x)
    ypos
        Y positions
    width :
        extraction  width
    extract_start : int
    extract_end : int or None

    Returns
    f :  (nint, nx) - Extracted flux
    """
    assert cube.ndim == 3, f"Expected 3D, got {cube.ndim}D shape {cube.shape}"

    nint, dimy, dimx = np.shape(cube)

    if extract_end is None:
        extract_end = dimx

    f = np.zeros((nint, dimx))

    lower_width, upper_width = _parse_width(width)
    edge_up = np.min([ypos + upper_width, np.ones_like(ypos) * dimy], axis=0)
    edge_low = np.max([ypos - lower_width, np.zeros_like(ypos)], axis=0)

    for i in tqdm(range(nint), disable=not progress, desc='Extracting'):
        for x in range(extract_start, extract_end):
            xx = x - extract_start
            if xx >= len(ypos):
                xx = len(ypos) - 1

            up_whole = np.floor(edge_up[xx]).astype(int)
            low_whole = np.ceil(edge_low[xx]).astype(int)

            #  total flux and total valid pixel area
            box = cube[i, low_whole:up_whole, x]

            total_flux = np.nansum(box)
            total_area = np.sum(np.isfinite(box))  #   valid whole pixels

            # add partial pixels
            if edge_up[xx] < (dimy-1) and edge_low[xx] > 0:
                up_part = edge_up[xx] % 1
                low_part = 1 - edge_low[xx] % 1

                up_val = cube[i, up_whole, x]
                low_val = cube[i, low_whole-1, x]

                # add partial pixel flux if valid
                if np.isfinite(up_val):
                    total_flux += up_part * up_val
                    total_area += up_part

                if np.isfinite(low_val):
                    total_flux += low_part * low_val
                    total_area += low_part

            # normalize by total valid pixel area
            if total_area > 0:
                f[i, x] = total_flux / total_area
            else:
                f[i, x] = np.nan

    return f


def extract_at_step(datafile, instrument, extract_width, centroids, baseline_ints, output_dir,
                    plot_diagnostic=False, extract_method='box', extract_width_soss2=None,
                    extract_step_kwargs=None):
    """
    Extract spectra from a datafile at any pipeline step.
    Note: Errors are NOT returned since they're not needed for optimization.

    Parameters
    datafile
         datafile to extract (should be first segment if all is well)
    instrument
        'NIRISS', 'NIRSPEC', or 'MIRI'
    extract_width
        Extraction width for the primary extraction aperture.
    centroids
        Centroids (will generate/cache if None)
    baseline_ints
        Baseline integrations
    output_dir
        For caching centroids
    plot_diagnostic : bool
        If True, save diagnostic plot showing extraction aperture
    extract_method : str
        Extraction method to emulate. Supported values here are 'box' and 'doublegauss'.
    extract_width_soss2
        Optional extraction width for SOSS order 2.
    extract_step_kwargs : dict, None
        Extra Stage 3 extraction settings, typically mirroring `stage3_kwargs['Extract1dStep']`.

    Returns
    spectral_dict
        Keys: 'Wave', 'Flux' (and O1/O2 versions for SOSS) - no errors
    centroids
        The centroids used (for caching)
    """
    fancyprint(f'=== Extracting {instrument} at current step ===')
    fancyprint(f'  datafile: {datafile if isinstance(datafile, str) else "datamodel"}')
    fancyprint(f'  extract_width: {extract_width}')
    if extract_step_kwargs is None:
        extract_step_kwargs = {}
    if extract_method == 'doublegauss' and instrument != 'NIRISS':
        raise ValueError('Optimizer-side double Gaussian extraction is currently only implemented '
                         'for NIRISS/SOSS.')

    # load with flags applied
    fancyprint(f'  Loading data with DQ flags...')
    cube, is_4d = apply_dq_flags([datafile])
    fancyprint(f'  Loaded: cube.shape={cube.shape}, is_4d={is_4d}')

    # convert 4D to 3D if needed
    if is_4d:
        fancyprint(f'  4D data detected: {cube.shape} -> taking last group')
        cube = cube[:, -1, :, :]
        fancyprint(f'  Now 3D: cube.shape={cube.shape}')

    assert cube.ndim == 3, f"Expected 3D after conversion, got {cube.ndim}D with shape {cube.shape}"

    # get centroids
    if centroids is None:
        fancyprint('Generating centroids from deep stack')
        centroids = {}
        deepstack = utils.make_baseline_stack_general(datafiles=[datafile], baseline_ints=baseline_ints)
        if np.ndim(deepstack) == 3:
            deepstack = deepstack[-1]

        if instrument == 'NIRISS':
            from jwst.pipeline import calwebb_spec2
            subarray = utils.get_soss_subarray(datafile)
            step = calwebb_spec2.extract_1d_step.Extract1dStep()
            tracetable = step.get_reference_file(datafile, 'spectrace')
            cens = utils.get_centroids_soss(deepstack, tracetable, subarray, save_results=False)
            centroids['xpos'] = cens[0][0]
            centroids['ypos o1'] = cens[0][1]
            centroids['ypos o2'] = cens[1][1]
            centroids['ypos o3'] = cens[2][1]
        elif instrument == 'NIRSPEC':
            det = utils.get_nrs_detector_name(datafile)
            subarray = utils.get_soss_subarray(datafile)
            grating = utils.get_nrs_grating(datafile)
            xstart = utils.get_nrs_trace_start(det, subarray, grating)
            cens = utils.get_centroids_nirspec(deepstack, xstart=xstart, save_results=False)
            centroids['xpos'], centroids['ypos'] = cens[0], cens[1]
        elif instrument == 'MIRI':
            cens = trace_spectrum([datafile], deepstack, output_dir=output_dir, save_results=False)
            if isinstance(cens, str):
                centroids = pd.read_csv(cens, comment='#')
            else:
                centroids['xpos'], centroids['ypos'] = cens[0], cens[1]

    # extract by instrument
    if instrument == 'NIRSPEC':
        x1, y1 = centroids['xpos'], centroids['ypos']
        det = utils.get_nrs_detector_name(datafile)
        subarray = utils.get_soss_subarray(datafile)
        grating = utils.get_nrs_grating(datafile)
        xstart = utils.get_nrs_trace_start(det, subarray, grating)

        flux = do_box_extraction_nanaware(cube, y1, width=extract_width, extract_start=xstart)

        # For Phase 1 optimization, wavelength not needed (just use pixel indices)
        # Wavelength calibration requires Stage 2 WCS, which we skip for efficiency
        wave = np.arange(flux.shape[1], dtype=float)  # Dummy wavelength array (pixel indices)

        fancyprint(f'  NIRSpec extraction: flux.shape={flux.shape}')
        fancyprint(f'  Flux stats: sum={np.nansum(flux):.6e}, mean={np.nanmean(flux):.6e}, median={np.nanmedian(flux):.6e}')
        fancyprint(f'  Using pixel indices as wavelength (Stage 1 optimization)')

        # Diagnostic plots
        if plot_diagnostic:
            median_frame = np.nanmedian(cube, axis=0)

            # Identify flagged (NaN) pixels in median frame
            nan_mask = np.isnan(median_frame)
            nan_y, nan_x = np.where(nan_mask)

            # Plot 1: 2D aperture overlay
            plt.figure(figsize=(12, 4))
            plt.imshow(median_frame, aspect='auto', origin='lower', vmin=np.nanpercentile(median_frame, 5), vmax=np.nanpercentile(median_frame, 95))

            # Overlay flagged pixels as red dots
            if len(nan_x) > 0:
                plt.plot(nan_x, nan_y, 'r.', markersize=0.5, alpha=0.5, label=f'Flagged pixels ({len(nan_x)})')

            lower_width, upper_width = _parse_width(extract_width)
            plt.plot(x1, y1, 'lime', linewidth=1.5, label='Trace center')
            plt.plot(x1, y1 + upper_width, 'y--', linewidth=1, label=f'Aperture (width={extract_width})')
            plt.plot(x1, y1 - lower_width, 'y--', linewidth=1)
            plt.xlabel('X pixel')
            plt.ylabel('Y pixel')
            plt.title(f'NIRSpec Extraction (width={extract_width})')
            plt.colorbar(label='Median Flux')
            plt.legend()
            plot_path = os.path.join(output_dir, f'extraction_diagnostic_nirspec_w{extract_width}.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            fancyprint(f'  Saved 2D diagnostic plot: {plot_path}')
            fancyprint(f'  Flagged pixels in median: {len(nan_x)}/{nan_mask.size} ({100*len(nan_x)/nan_mask.size:.2f}%)')

            # Plot 2: 1D extracted spectrum
            median_flux = np.nanmedian(flux, axis=0)
            fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

            # Top: median spectrum
            ax[0].plot(wave, median_flux, 'k-', linewidth=0.5, alpha=0.7)
            ax[0].set_ylabel('Median Flux')
            ax[0].set_title(f'NIRSpec Extracted Spectrum (width={extract_width})')
            ax[0].grid(alpha=0.3)

            # Bottom: first few integrations
            for i in range(min(5, flux.shape[0])):
                ax[1].plot(wave, flux[i], linewidth=0.5, alpha=0.5, label=f'Int {i}')
            ax[1].set_xlabel('Pixel')
            ax[1].set_ylabel('Flux')
            ax[1].legend(fontsize=8, ncol=5)
            ax[1].grid(alpha=0.3)

            plt.tight_layout()
            plot_path_1d = os.path.join(output_dir, f'extraction_spectrum_nirspec_w{extract_width}.png')
            plt.savefig(plot_path_1d, dpi=150, bbox_inches='tight')
            plt.close()
            fancyprint(f'  Saved 1D spectrum plot: {plot_path_1d}')

        return {'Wave': wave, 'Flux': flux}, centroids

    elif instrument == 'NIRISS':
        x1 = centroids['xpos']
        y1, y2 = centroids['ypos o1'], centroids['ypos o2']

        w1 = extract_width
        w2 = extract_width if extract_width_soss2 is None else extract_width_soss2

        fancyprint(f'  NIRISS extraction widths: O1={w1}, O2={w2}')

        ii = np.where(np.isfinite(y2))[0]
        y2_finite = y2[ii]

        if extract_method == 'doublegauss':
            separation_guess = extract_step_kwargs.get('double_gaussian_separation_guess', 4.0)
            separation_guess_o2 = extract_step_kwargs.get(
                'double_gaussian_separation_guess_soss2', separation_guess
            )
            fit_background = extract_step_kwargs.get('double_gaussian_fit_background', True)
            main_component = int(extract_step_kwargs.get('double_gaussian_main_component', 1))
            err = np.ones_like(cube, dtype=float)

            flux1_o1, _, flux2_o1, _, _ = do_two_gaussian_extraction(
                cube, err, y1, width=w1, progress=False, separation_guess=separation_guess,
                fit_background=fit_background
            )
            flux1_o2, _, flux2_o2, _, _ = do_two_gaussian_extraction(
                cube, err, y2_finite, width=w2, extract_end=len(y2_finite), progress=False,
                separation_guess=separation_guess_o2, fit_background=fit_background
            )

            if main_component == 1:
                flux_o1, flux_o2 = flux1_o1, flux1_o2
            else:
                flux_o1, flux_o2 = flux2_o1, flux2_o2
        else:
            flux_o1 = do_box_extraction_nanaware(cube, y1, width=w1)
            flux_o2 = do_box_extraction_nanaware(cube, y2_finite, width=w2,
                                                 extract_end=len(y2_finite))

        fancyprint(f'  O1 flux.shape={flux_o1.shape}, sum={np.nansum(flux_o1):.6e}, mean={np.nanmean(flux_o1):.6e}')
        fancyprint(f'  O2 flux.shape={flux_o2.shape}, sum={np.nansum(flux_o2):.6e}, mean={np.nanmean(flux_o2):.6e}')

        wave_o1, wave_o2 = get_wave_soss(datafile)

        # Diagnostic plots
        if plot_diagnostic:
            median_frame = np.nanmedian(cube, axis=0)

            # Identify flagged (NaN) pixels in median frame
            nan_mask = np.isnan(median_frame)
            nan_y, nan_x = np.where(nan_mask)

            # Plot 1: 2D aperture overlay
            plt.figure(figsize=(12, 8))
            plt.imshow(median_frame, aspect='auto', origin='lower', vmin=np.nanpercentile(median_frame, 5), vmax=np.nanpercentile(median_frame, 95))

            # Overlay flagged pixels as red dots
            if len(nan_x) > 0:
                plt.plot(nan_x, nan_y, 'r.', markersize=0.5, alpha=0.5, label=f'Flagged pixels ({len(nan_x)})')

            lower1, upper1 = _parse_width(w1)
            lower2, upper2 = _parse_width(w2)
            plt.plot(x1, y1, 'lime', linewidth=1.5, label='Order 1 center')
            plt.plot(x1, y1 + upper1, 'y--', linewidth=1, label=f'O1 aperture (width={w1})')
            plt.plot(x1, y1 - lower1, 'y--', linewidth=1)
            # Order 2 (only finite values)
            valid_o2 = np.isfinite(y2)
            plt.plot(x1[valid_o2], y2[valid_o2], 'c-', linewidth=1.5, label='Order 2 center')
            plt.plot(x1[valid_o2], y2[valid_o2] + upper2, 'm--', linewidth=1, label=f'O2 aperture (width={w2})')
            plt.plot(x1[valid_o2], y2[valid_o2] - lower2, 'm--', linewidth=1)
            plt.xlabel('X pixel')
            plt.ylabel('Y pixel')
            plt.title(f'NIRISS/SOSS Extraction (O1 width={w1}, O2 width={w2})')
            plt.colorbar(label='Median Flux')
            plt.legend()
            plot_path = os.path.join(output_dir, f'extraction_diagnostic_soss_w{w1}.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            fancyprint(f'  Saved 2D diagnostic plot: {plot_path}')
            fancyprint(f'  Flagged pixels in median: {len(nan_x)}/{nan_mask.size} ({100*len(nan_x)/nan_mask.size:.2f}%)')

            # Plot 2: 1D extracted spectra for both orders
            fig, axes = plt.subplots(2, 2, figsize=(14, 8))

            # Order 1 - top row
            median_flux_o1 = np.nanmedian(flux_o1, axis=0)
            axes[0, 0].plot(wave_o1, median_flux_o1, 'k-', linewidth=0.5, alpha=0.7)
            axes[0, 0].set_ylabel('Median Flux')
            axes[0, 0].set_title(f'Order 1 Spectrum (width={w1})')
            axes[0, 0].grid(alpha=0.3)

            for i in range(min(5, flux_o1.shape[0])):
                axes[0, 1].plot(wave_o1, flux_o1[i], linewidth=0.5, alpha=0.5, label=f'Int {i}')
            axes[0, 1].set_ylabel('Flux')
            axes[0, 1].set_title('Order 1 Sample Integrations')
            axes[0, 1].legend(fontsize=8, ncol=5)
            axes[0, 1].grid(alpha=0.3)

            # Order 2 - bottom row
            median_flux_o2 = np.nanmedian(flux_o2, axis=0)
            axes[1, 0].plot(wave_o2, median_flux_o2, 'k-', linewidth=0.5, alpha=0.7)
            axes[1, 0].set_xlabel('Wavelength (μm)')
            axes[1, 0].set_ylabel('Median Flux')
            axes[1, 0].set_title(f'Order 2 Spectrum (width={w2})')
            axes[1, 0].grid(alpha=0.3)

            for i in range(min(5, flux_o2.shape[0])):
                axes[1, 1].plot(wave_o2, flux_o2[i], linewidth=0.5, alpha=0.5, label=f'Int {i}')
            axes[1, 1].set_xlabel('Wavelength (μm)')
            axes[1, 1].set_ylabel('Flux')
            axes[1, 1].set_title('Order 2 Sample Integrations')
            axes[1, 1].legend(fontsize=8, ncol=5)
            axes[1, 1].grid(alpha=0.3)

            plt.tight_layout()
            plot_path_1d = os.path.join(output_dir, f'extraction_spectrum_soss_w{w1}.png')
            plt.savefig(plot_path_1d, dpi=150, bbox_inches='tight')
            plt.close()
            fancyprint(f'  Saved 1D spectrum plot: {plot_path_1d}')

        return {
            'Wave O1': wave_o1, 'Flux O1': flux_o1,
            'Wave O2': wave_o2, 'Flux O2': flux_o2
        }, centroids

    elif instrument == 'MIRI':
        x1, y1 = centroids['xpos'], centroids['ypos']

        flux = do_box_extraction_nanaware(
            cube.transpose(0, 2, 1), x1,
            width=extract_width, extract_start=int(np.min(y1)), extract_end=int(np.max(y1))
        )

        # For optimizer, use pixel indices as placeholder wavelengths (no calibration needed)
        # This is sufficient for cost function evaluation
        wave = np.arange(flux.shape[1]).astype(float)
        wave = np.repeat(wave[np.newaxis, :], flux.shape[0], axis=0)

        # Diagnostic plots (MIRI is rotated, so plot transpose)
        if plot_diagnostic:
            median_frame = np.nanmedian(cube.transpose(0, 2, 1), axis=0)

            # Identify flagged (NaN) pixels in median frame
            nan_mask = np.isnan(median_frame)
            nan_y, nan_x = np.where(nan_mask)

            # Plot 1: 2D aperture overlay
            plt.figure(figsize=(12, 4))
            plt.imshow(median_frame, aspect='auto', origin='lower', vmin=np.nanpercentile(median_frame, 5), vmax=np.nanpercentile(median_frame, 95))

            # Overlay flagged pixels as red dots
            if len(nan_x) > 0:
                plt.plot(nan_x, nan_y, 'r.', markersize=0.5, alpha=0.5, label=f'Flagged pixels ({len(nan_x)})')

            # Plot trace as function of Y position (for MIRI geometry)
            y_coords = np.arange(len(x1))
            lower_width, upper_width = _parse_width(extract_width)
            plt.plot(y_coords, x1, 'lime', linewidth=1.5, label='Trace center')
            plt.plot(y_coords, x1 + upper_width, 'y--', linewidth=1, label=f'Aperture (width={extract_width})')
            plt.plot(y_coords, x1 - lower_width, 'y--', linewidth=1)
            plt.xlabel('Y pixel')
            plt.ylabel('X pixel')
            plt.title(f'MIRI Extraction (width={extract_width})')
            plt.colorbar(label='Median Flux')
            plt.legend()
            plot_path = os.path.join(output_dir, f'extraction_diagnostic_miri_w{extract_width}.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            fancyprint(f'  Saved 2D diagnostic plot: {plot_path}')
            fancyprint(f'  Flagged pixels in median: {len(nan_x)}/{nan_mask.size} ({100*len(nan_x)/nan_mask.size:.2f}%)')

            # Plot 2: 1D extracted spectrum
            median_flux = np.nanmedian(flux, axis=0)
            fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

            # Top: median spectrum
            ax[0].plot(wave, median_flux, 'k-', linewidth=0.5, alpha=0.7)
            ax[0].set_ylabel('Median Flux')
            ax[0].set_title(f'MIRI Extracted Spectrum (width={extract_width})')
            ax[0].grid(alpha=0.3)

            # Bottom: first few integrations
            for i in range(min(5, flux.shape[0])):
                ax[1].plot(wave, flux[i], linewidth=0.5, alpha=0.5, label=f'Int {i}')
            ax[1].set_xlabel('Wavelength (μm)')
            ax[1].set_ylabel('Flux')
            ax[1].legend(fontsize=8, ncol=5)
            ax[1].grid(alpha=0.3)

            plt.tight_layout()
            plot_path_1d = os.path.join(output_dir, f'extraction_spectrum_miri_w{extract_width}.png')
            plt.savefig(plot_path_1d, dpi=150, bbox_inches='tight')
            plt.close()
            fancyprint(f'  Saved 1D spectrum plot: {plot_path_1d}')

        return {'Wave': wave, 'Flux': flux}, centroids

    else:
        raise ValueError(f"Unknown instrument: {instrument}")
