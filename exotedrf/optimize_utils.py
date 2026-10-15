#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: PSD, modified by TRF

Helper functions for the run_optimize.py script with handling of DQ flags and extraction.
"""

import ast
from astropy.io import fits
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
from tqdm import tqdm

from exotedrf.stage3 import get_wave_soss, trace_spectrum, _parse_extraction_width
from exotedrf import utils
from exotedrf.utils import fancyprint


def apply_dq_flags(datafiles):
    """Load data and apply DQ flags by NaN-ing out bad pixels.
    Errors are NOT loaded/returned since they're not needed for optimization.

    Parameters
    datafiles

    Returns
    cube  : array
        Flux with bad pixels as NaN
    is_4d : bool
        True if pre-RampFit (4D), False if post (3D)
    group : int
        For 4D input, the group whose DQ was used for the mask and which the
        caller must therefore score (see last_scoreable_group). -1 for 3D.
    """

    datafiles = np.atleast_1d(datafiles)
    group = None

    # get flux and DQ (errors not needed for optimization)
    for i, file in enumerate(datafiles):
        fancyprint(f'Loading segment {i}: {file if isinstance(file, str) else 'datamodel'}')

        if isinstance(file, str):
            data = fits.getdata(file, 1)
            dq = fits.getdata(file, 3)
            fancyprint(f'  Loaded from FITS: data.shape={data.shape}, dq.shape={dq.shape}')
        else:
            with utils.open_filetype(file) as datamodel:
                data = datamodel.data
                dq = datamodel.dq
                fancyprint(
                    f'  Loaded from datamodel: data.shape={data.shape}, dq.shape={dq.shape if dq is not None else None}')

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

                # Take the last group that still has usable pixels for the
                # mask. This is the final group except where DQ flags an
                # entire group (MIRI's first/last frame), which would leave
                # nothing to extract and make every trial cost NaN.
                segment_group = last_scoreable_group(dq)
                if group is None:
                    group = segment_group
                elif segment_group != group:
                    fancyprint(f'  Segment {i} would score group '
                               f'{segment_group} but segment 0 scored group '
                               f'{group}; keeping group {group} so all '
                               'segments are comparable.',
                               msg_type='WARNING')
                dq_for_mask = dq[:, group, :, :]
                fancyprint(f'  Took group {group}: dq_for_mask.shape={dq_for_mask.shape}')

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
                    bad_pixels = (np.asarray(dq).astype(np.uint64) & np.uint64(3)) != 0
                    bad_pixels = bad_pixels[np.newaxis, :, :]
                    bad_pixels = np.broadcast_to(bad_pixels, data.shape)
                    fancyprint(f'  bad_pixels.shape={bad_pixels.shape}')
                    dq_for_mask = None

                if dq_for_mask is not None:
                    # Mirror Stage 3 box extraction, which only masks DO_NOT_USE (1) and
                    # SATURATED (2). Masking every flag would also NaN pixels that BadPixStep
                    # already corrected or flagged as HIGH_VARIANCE, biasing the BadPixStep
                    # sweeps towards settings that flag more pixels.
                    bad_pixels = (np.asarray(dq_for_mask).astype(np.uint64) & np.uint64(3)) != 0
                    fancyprint(f'  bad_pixels.shape={bad_pixels.shape}')

            # Apply mask
            fancyprint(
                f'  Applying mask: data.shape={data.shape}, bad_pixels.shape={bad_pixels.shape}')

            data[bad_pixels] = np.nan

            n_bad = np.sum(bad_pixels)
            fancyprint(
                f'Segment {i}: Flagged {n_bad}/{bad_pixels.size} pixels ({100 * n_bad / bad_pixels.size:.2f}%)')
        else:
            fancyprint(f'Segment {i}: No DQ found', msg_type='WARNING')
            is_4d = data.ndim == 4

        # concatenate segments
        if i == 0:
            cube = data
        else:
            cube = np.concatenate([cube, data])

    # 3D input (or 4D input with no DQ at all) keeps the historical -1.
    return cube, is_4d, (-1 if group is None else group)


def delete_checkpoint_outputs(checkpoint_name, outdir_s1, outdir_s2):
    """Delete a checkpoint step's cached outputs, and those of every step downstream of it, so
    the next pipeline call recomputes them.
    """

    # Downstream Steps
    _STAGE1_AFTER_1OVERF = ['linearitystep', 'jump', 'rampfitstep', 'gainscalestep']
    _STAGE2_ALL = ['assignwcsstep', 'extract2dstep', 'sourcetypestep', 'wavecorrstep',
                   'flatfieldstep', 'photomstep', 'backgroundstep', 'oneoverfstep', 'badpixstep',
                   'pcareconstructstep']
    _STAGE2_AFTER_BKG = ['oneoverfstep', 'badpixstep', 'pcareconstructstep']

    patterns = []
    downstream = []
    if checkpoint_name == 'OneOverFStep_grp':
        patterns.append(f'{outdir_s1}*_oneoverfstep.fits')
        downstream += [f'{outdir_s1}*_{t}.fits' for t in _STAGE1_AFTER_1OVERF]
        downstream += [f'{outdir_s2}*_{t}.fits' for t in _STAGE2_ALL]
        downstream.append(f'{outdir_s2}*hot_pixels.npy')
    elif checkpoint_name == 'JumpStep':
        patterns.append(f'{outdir_s1}*_jump.fits')
        downstream += [f'{outdir_s1}*_{t}.fits' for t in ['rampfitstep', 'gainscalestep']]
        downstream += [f'{outdir_s2}*_{t}.fits' for t in _STAGE2_ALL]
        downstream.append(f'{outdir_s2}*hot_pixels.npy')
    elif checkpoint_name == 'BackgroundStep':
        patterns.append(f'{outdir_s2}*_backgroundstep.fits')
        downstream += [f'{outdir_s2}*_{t}.fits' for t in _STAGE2_AFTER_BKG]
        downstream.append(f'{outdir_s2}*hot_pixels.npy')
    elif checkpoint_name == 'BadPixStep':
        patterns.append(f'{outdir_s2}*_badpixstep.fits')
        # Also delete cached hot_pixels.npy to force spatial outlier
        # redetection with new parameters (space_thresh, box_size).
        patterns.append(f'{outdir_s2}*hot_pixels.npy')
        downstream.append(f'{outdir_s2}*_pcareconstructstep.fits')
    deleted = 0
    for pattern in patterns + downstream:
        files_to_delete = glob.glob(pattern)
        if files_to_delete:
            fancyprint(f'Deleting {len(files_to_delete)} cached file(s) for {checkpoint_name}:')
        for cached_file in files_to_delete:
            fancyprint(f'  Deleting: {cached_file}')
            os.remove(cached_file)
            if pattern in patterns:
                deleted += 1
    if patterns and deleted == 0:
        fancyprint(f'WARNING: No cached files found matching: {patterns}', msg_type='WARNING')


def do_box_extraction_nanaware(cube, ypos, width, extract_start=0, extract_end=None, progress=True):
    """Box extraction with nansum. Modified from stage3.do_box_extraction.
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

    assert cube.ndim == 3, f'Expected 3D, got {cube.ndim}D shape {cube.shape}'

    nint, dimy, dimx = np.shape(cube)

    if extract_end is None:
        extract_end = dimx

    f = np.zeros((nint, dimx))

    lower_width, upper_width = _parse_extraction_width(width)
    edge_up = np.min([ypos + upper_width, np.ones_like(ypos) * dimy], axis=0)
    edge_low = np.max([ypos - lower_width, np.zeros_like(ypos)], axis=0)

    for i in tqdm(range(nint), disable=not progress, desc='Extracting'):
        for x in range(extract_start, extract_end):
            xx = x - extract_start
            if xx >= len(ypos):
                xx = len(ypos) - 1

            # Fractional overlap of each detector row with [edge_low, edge_up], so partial
            # pixels are weighted correctly at each edge independently.
            rows = np.arange(max(int(np.floor(edge_low[xx])), 0),
                             min(int(np.ceil(edge_up[xx])), dimy))
            weights = np.clip(np.minimum(rows + 1, edge_up[xx]) - np.maximum(rows, edge_low[xx]),
                              0, 1)
            vals = cube[i, rows, x]
            good = np.isfinite(vals) & (weights > 0)

            #  total flux and total valid pixel area
            total_flux = np.sum(weights[good] * vals[good])
            total_area = np.sum(weights[good])

            # normalize by total valid pixel area
            if total_area > 0:
                f[i, x] = total_flux / total_area
            else:
                f[i, x] = np.nan

    return f


def extract_at_step(datafile, instrument, extract_width, centroids, baseline_ints, output_dir,
                    plot_diagnostic=False, extract_method='box', extract_width_soss2=None,
                    extract_step_kwargs=None):
    """Extract spectra from a datafile at any pipeline step.
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
    fancyprint(f'  datafile: {datafile if isinstance(datafile, str) else 'datamodel'}')
    fancyprint(f'  extract_width: {extract_width}')
    if extract_step_kwargs is None:
        extract_step_kwargs = {}

    # load with flags applied
    fancyprint(f'  Loading data with DQ flags...')
    cube, is_4d, group = apply_dq_flags([datafile])
    fancyprint(f'  Loaded: cube.shape={cube.shape}, is_4d={is_4d}')

    # convert 4D to 3D if needed. Score the same group apply_dq_flags built
    # the mask from, so a wholly-flagged final group (MIRI) does not leave an
    # all-NaN image behind.
    if is_4d:
        fancyprint(f'  4D data detected: {cube.shape} -> taking group {group}')
        cube = cube[:, group, :, :]
        fancyprint(f'  Now 3D: cube.shape={cube.shape}')

    assert cube.ndim == 3, f'Expected 3D after conversion, got {cube.ndim}D with shape {cube.shape}'

    # get centroids
    if centroids is None:
        fancyprint('Generating centroids from deep stack')
        centroids = {}
        deepstack = utils.make_baseline_stack_general(datafiles=[datafile], baseline_ints=baseline_ints)
        if np.ndim(deepstack) == 3:
            # Trace the same group that is being scored (identical to the old
            # behaviour whenever the final group is usable).
            deepstack = deepstack[group]

        if instrument.lower() == 'niriss':
            subarray = utils.get_soss_subarray(datafile)
            # Use the same tracetable source as the Stage 1-3 steps.
            outdir = os.environ['CRDS_PATH'] + '/references/jwst/niriss/'
            tracetable = utils.get_soss_tracetable(subarray, outdir)
            cens = utils.get_centroids_soss(deepstack, tracetable, subarray, save_results=False)
            centroids['xpos'] = cens[0][0]
            centroids['ypos o1'] = cens[0][1]
            centroids['ypos o2'] = cens[1][1]
            centroids['ypos o3'] = cens[2][1]
        elif instrument.lower() == 'nirspec':
            det = utils.get_nrs_detector_name(datafile)
            subarray = utils.get_soss_subarray(datafile)
            grating = utils.get_nrs_grating(datafile)
            xstart = utils.get_nrs_trace_start(det, subarray, grating)
            cens = utils.get_centroids_nirspec(deepstack, xstart=xstart, save_results=False)
            centroids['xpos'], centroids['ypos'] = cens[0], cens[1]
        elif instrument.lower() == 'miri':
            cens = trace_spectrum([datafile], deepstack, output_dir=output_dir, save_results=False)
            if isinstance(cens, str):
                centroids = pd.read_csv(cens, comment='#')
            else:
                centroids['xpos'], centroids['ypos'] = cens[0], cens[1]

    # extract by instrument
    if instrument.lower() == 'nirspec':
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

            lower_width, upper_width = _parse_extraction_width(extract_width)
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

    elif instrument.lower() == 'niriss':
        x1 = centroids['xpos']
        y1, y2 = centroids['ypos o1'], centroids['ypos o2']

        w1 = extract_width
        w2 = extract_width if extract_width_soss2 is None else extract_width_soss2

        fancyprint(f'  NIRISS extraction widths: O1={w1}, O2={w2}')

        ii = np.where(np.isfinite(y2))[0]
        y2_finite = y2[ii]

        flux_o1 = do_box_extraction_nanaware(cube, y1, width=w1)
        flux_o2 = do_box_extraction_nanaware(cube, y2_finite, width=w2, extract_end=len(y2_finite))

        fancyprint(f'  O1 flux.shape={flux_o1.shape}, sum={np.nansum(flux_o1):.6e}, mean={np.nanmean(flux_o1):.6e}')
        fancyprint(f'  O2 flux.shape={flux_o2.shape}, sum={np.nansum(flux_o2):.6e}, mean={np.nanmean(flux_o2):.6e}')

        outdir = os.environ['CRDS_PATH'] + '/references/jwst/niriss/'
        wave_o1, wave_o2 = get_wave_soss(datafile, outdir)

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

            lower1, upper1 = _parse_extraction_width(w1)
            lower2, upper2 = _parse_extraction_width(w2)
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

    elif instrument.lower() == 'miri':
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
            lower_width, upper_width = _parse_extraction_width(extract_width)
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
        raise ValueError(f'Unknown instrument: {instrument}')


def find_stage3_spectrum_file(extract_method, outdir, nrs_detector=None):
    """Return the first Stage 3 full-resolution spectrum file for the requested method.
    """

    if nrs_detector is not None:
        pattern = os.path.join(outdir, f'*_{nrs_detector}_{extract_method}_spectra_fullres.fits')
    else:
        pattern = os.path.join(outdir, f'*_{extract_method}_spectra_fullres.fits')
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f'No Stage 3 spectrum file found matching {pattern}')
    return matches[0]


def last_scoreable_group(dq):
    """Index of the last group that still has usable pixels.

    Optimizer trials taken before RampFit are scored on a single group of the
    ramp, with every DQ-flagged pixel NaN-ed out. Normally that is the final
    group, but a group whose DQ is set for EVERY pixel leaves nothing to
    extract: the aperture is entirely NaN and `cost_function` returns NaN for
    both its terms, so the whole sweep collapses.

    MIRI hits this on every exposure, because DQInitStep flags the first and
    last MIRI group DO_NOT_USE (stage1.py, flag_first/last_miri_frame). Only
    the stage-1 checkpoints score a 4D product, so for MIRI this silently
    disabled the `time_jump_threshold` and `time_window` sweeps.

    Walking back to the last group that is not fully flagged restores a
    meaningful comparison and is a no-op for SOSS/NIRSpec, whose final group
    is not flagged wholesale.

    Parameters
    dq : array
        4D group DQ cube, (nint, ngroup, y, x).

    Returns
    group : int
        Group index to mask and score. Falls back to -1 if every group is
        fully flagged.
    """

    ngroup = dq.shape[1]
    for group in range(ngroup - 1, -1, -1):
        if not np.all(dq[:, group] > 0):
            if group != ngroup - 1:
                fancyprint(f'  Final group is fully DQ-flagged; scoring group '
                           f'{group} of {ngroup} instead.')
            return group
    fancyprint('  Every group is fully DQ-flagged; scoring the final group, '
               'which will yield a non-finite cost.', msg_type='WARNING')
    return -1


def resolve_existing_centroids(cfg, outdir2, outdir3, fileroot_noseg=None):
    """Resolve the centroid table for a Stage 3 extraction or rerun. If `fileroot_noseg` is
    given, only centroid tables written for that dataset are considered.
    """

    centroids_path = cfg['centroids']
    if centroids_path is not None:
        fancyprint(f'  Using centroids from config: {centroids_path}')
        if isinstance(centroids_path, str):
            return pd.read_csv(centroids_path, comment='#')
        return centroids_path

    centroid_patterns = [
        (outdir3, 'Stage 3'),
        (outdir2, 'Stage 2'),
    ]
    for outdir, label in centroid_patterns:
        if fileroot_noseg is not None:
            centroid_files = sorted(glob.glob(f'{outdir}{glob.escape(fileroot_noseg)}centroids.csv'))
        else:
            centroid_files = sorted(glob.glob(f'{outdir}*centroids.csv'))
        if centroid_files:
            fancyprint(f'  Loading centroids from {label}: {centroid_files[0]}')
            return pd.read_csv(centroid_files[0], comment='#')

    raise FileNotFoundError(
        'No centroid table available for Stage 3. Set centroids in the config or provide '
        'a Stage 3/Stage 2 centroids.csv output.'
    )


def resolve_extract1d_kwargs(cfg):
    """Return the Stage 3 Extract1dStep kwargs block, if present.
    """

    return cfg.get('stage3_kwargs', {}).get('Extract1dStep', {})


def select_best_trial(costs, param_name='parameter'):
    """Return the index of the first finite minimum cost.

    np.argmin returns the index of a NaN if one is present, so a failed trial
    could otherwise be selected as the winner. Non-finite costs are skipped;
    ties keep the earliest candidate.
    """

    best_idx = None
    best_cost = None
    for idx, cost in enumerate(costs):
        cost = float(cost)
        if not np.isfinite(cost):
            continue
        if best_idx is None or cost < best_cost:
            best_idx, best_cost = idx, cost
    if best_idx is None:
        raise ValueError(f'All candidate values for {param_name} produced non-finite costs.')

    return best_idx


def stage1_kwargs_with_winners(cfg):
    """Stage 1 kwargs with the current scalar time_window forwarded to JumpStep.

    time_window was previously only forwarded while it was itself being swept,
    so later sweeps and the Phase 2 run silently fell back to the step default
    instead of the current best (or fixed) value.
    """

    kwargs = cfg['stage1_kwargs']
    time_window = cfg['time_window']
    if 'JumpStep' in kwargs.keys():
        step_kwargs = kwargs['JumpStep']
    else:
        step_kwargs = {}
    if isinstance(time_window, (int, float, np.integer, np.floating)):
        step_kwargs['time_window'] = time_window
        kwargs['JumpStep'] = step_kwargs

    return kwargs


def stage2_kwargs_with_winners(cfg):
    """Stage 2 kwargs with current scalar box_size/window_size forwarded to
    BadPixStep (same defect and fix as stage1_kwargs_with_winners).
    """

    kwargs = cfg['stage2_kwargs']
    if 'BadPixStep' in kwargs.keys():
        step_kwargs = kwargs['BadPixStep']
    else:
        step_kwargs = {}
    for key in ('box_size', 'window_size'):
        value = cfg[key]
        if isinstance(value, (int, float, np.integer, np.floating)):
            step_kwargs[key] = value
    if step_kwargs:
        kwargs['BadPixStep'] = step_kwargs

    return kwargs


def stitch_soss_orders(wave_o1, wave_o2, flux_o1=None, flux_o2=None, cutoff=0.85):
    """Stitch SOSS order 2 (<= cutoff) and order 1 (> cutoff) onto one wavelength axis.

    Selection is by wavelength, not by edge index, since SOSS wavelengths decrease with detector
    column before Stage 3 formatting. Returned arrays are sorted by wavelength.
    """

    wave_o1 = np.asarray(wave_o1, float)
    wave_o2 = np.asarray(wave_o2, float)
    i2 = np.where(wave_o2 <= cutoff)[0]
    i1 = np.where(wave_o1 > cutoff)[0]
    if i2.size == 0 or i1.size == 0:
        msg = 'Cutoff produces empty segment: O2<= {cutoff}: {i2.size}, O1> {cutoff}: {i1.size}'
        raise ValueError(msg)

    wave = np.concatenate([wave_o2[i2], wave_o1[i1]])
    s = np.argsort(wave, kind='mergesort')
    wave = wave[s]
    if flux_o1 is None or flux_o2 is None:
        return wave, None
    flux = np.concatenate([np.asarray(flux_o2, float)[:, i2],
                           np.asarray(flux_o1, float)[:, i1]], axis=1)
    return wave, flux[:, s]
