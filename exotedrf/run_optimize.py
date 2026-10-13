#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 15 00:00 2025

@author: PSD, modified by TRF

Script to run the exoTEDRF pipeline optimizer.
"""


from datetime import datetime
import glob
import numpy as np
import os
import pandas as pd
import shutil
import sys
import time

from exotedrf.utils import parse_config, unpack_input_dir, fancyprint, verify_path


# ===== Setup =====
# Read config file.
try:
    config_file = sys.argv[1]
except IndexError:
    raise FileNotFoundError('Config file must be provided')
config = parse_config(config_file)

# Set CRDS cache path.
os.environ['CRDS_PATH'] = config['crds_cache_path']
os.environ['CRDS_SERVER_URL'] = 'https://jwst-crds.stsci.edu'

# Import rest of pipeline stuff after initializing crds path.
from exotedrf.stage1 import run_stage1
from exotedrf.stage2 import run_stage2
from exotedrf.stage3 import run_stage3
from exotedrf import utils
import exotedrf.optimize_utils as opt_utils
from exotedrf.optimize_plotting import make_lightcurve_plots, plot_scatter, plot_cost


# ===== Define Global Variables =====
# All Pipeline Steps
stage1_steps = ['DQInitStep', 'INLCorrStep', 'EmiCorrStep', 'SuperBiasStep', 'RefPixStep',
                'DarkCurrentStep', 'OneOverFStep_grp', 'LinearityStep', 'JumpStep', 'RampFitStep',
                'GainScaleStep']
stage2_steps = ['AssignWCSStep', 'FlatFieldStep', 'OneOverFStep_int', 'BackgroundStep',
                'BadPixStep', 'PCAReconstructStep']

# Optimization Checkpoints
all_checkpoints = [
    # Stage 1 checkpoints
    {
        'name': 'OneOverFStep_grp',
        'stage': 1,
        'params': ['soss_inner_mask_width', 'soss_outer_mask_width', 'nirspec_mask_width'],
        'skip_before': ['DQInitStep', 'INLCorrStep', 'EmiCorrStep', 'SuperBiasStep', 'RefPixStep',
                        'DarkCurrentStep'],
        'skip_after': ['LinearityStep', 'JumpStep', 'RampFitStep', 'GainScaleStep'],
    },
    {
        'name': 'JumpStep',
        'stage': 1,
        'params': ['time_jump_threshold', 'time_window'],
        'skip_before': ['DQInitStep', 'INLCorrStep', 'EmiCorrStep', 'SuperBiasStep', 'RefPixStep',
                        'DarkCurrentStep', 'OneOverFStep_grp', 'LinearityStep'],
        'skip_after': ['RampFitStep', 'GainScaleStep'],
    },
    # Stage 2 checkpoints
    {
        'name': 'BackgroundStep',
        'stage': 2,
        'params': ['miri_trace_width', 'miri_background_width'],
        'skip_before': ['AssignWCSStep', 'FlatFieldStep'],
        'skip_after': ['OneOverFStep_int', 'BadPixStep', 'PCAReconstructStep'],
    },
    {
        'name': 'BadPixStep',
        'stage': 2,
        'params': ['space_outlier_threshold', 'time_outlier_threshold', 'box_size', 'window_size'],
        'skip_before': ['AssignWCSStep', 'FlatFieldStep', 'BackgroundStep', 'OneOverFStep_int'],
        'skip_after': ['PCAReconstructStep'],
    },
    # Stage 3 checkpoint - only for Phase 2 (full dataset)
    {
        'name': 'Extract',
        'stage': 3,
        'params': ['extract_width'],
        'skip_before': [],
        'skip_after': [],
        'phase_2_only': True,  # Only optimize in Phase 2
    },
]

# Default Wavebands
bands = {
    'miri':    (5.0, 12.0),
    'nirspec': (1.0, 5.0),
    'niriss':  (0.6, 2.8)
}

# Output Directories
root_dir = config['root_dir']
# The stages write to pipeline_outputs_directory + '_' + output_tag (expanding '~'), so mirror
# that here so cached outputs are found and invalidated in the right place.
_output_tag = config['output_tag']
_output_tag = '_' + _output_tag if _output_tag != '' else ''
full_outdir = os.path.join(root_dir, 'pipeline_outputs_directory' + _output_tag)

# Define where to store outputs for each pipeline stage
outdir_f = os.path.join(full_outdir, 'Optimizer_Files')
outdir_s1 = os.path.join(full_outdir, 'Stage1/')
outdir_s2 = os.path.join(full_outdir, 'Stage2/')
outdir_s3 = os.path.join(full_outdir, 'Stage3/')
utils.verify_path(full_outdir)
utils.verify_path(outdir_f)
utils.verify_path(outdir_s1)
utils.verify_path(outdir_s2)
utils.verify_path(outdir_s3)


# ===== Functions =====
def resolve_spectral_wave_range(wave_range, obs_mode, det, bands):
    """Use the configured wavelength range, or an instrument default when spectral cost is active.
    """

    # TODO: Fix these.
    # Default wavelength ranges.
    inst, mode = obs_mode.split('/')
    if wave_range is None:
        user = 'default'
        if inst.lower() == 'niriss':
            wave_range = [0.6, 2.8]
        elif inst.lower() == 'nirspec':
            if det.lower() == 'nrs1':
                wave_range = [2.9, 3.5]
            elif det.lower() == 'nrs2':
                wave_range = [4.0, 5.0]
            else:
                raise ValueError('NIRSpec optimization requires filter_detector=NRS1 or NRS2.')
        elif inst.lower() == 'miri':
            wave_range = [5.0, 12.0]
        else:
            raise ValueError('Unknown instrument {}'.format(inst))
    else:
        # Double check that user-defined bounds are okay.
        for key, (wave_low, wave_high) in bands.items():
            if key in obs_mode:
                if not (wave_low <= np.min(wave_range) and np.max(wave_range) <= wave_high):
                    raise ValueError(f'Wavelengths out of allowed band [{wave_low}, {wave_high}]')
                break
        # If nothing breaks, then all is good.
        user = 'user defined'

    # Format print string.
    if inst.lower() == 'nirspec':
        inst_str = obs_mode + ' ' + det
    else:
        inst_str = obs_mode
    fancyprint('Using {} wavelength range of {} for {} spectral optimization.'
               ''.format(user, wave_range, inst_str))

    return wave_range


def cost_function(st3, obs_mode, baseline_ints, wave_range=None, w1=0.0, w2=1.0):
    """Compute a combined white-light + spectral P2P (point-to-point) metric.

    Parameters
    ----------
    st3 : dict-like
        Must contain:
          - 'Flux' (or 'Flux O1'/'Flux O2' for NIRISS) -> 2D array (n_int, n_wave)
          - 'Wave' (or 'Wave O1'/'Wave O2') -> 1D array (n_wave,)
    obs_mode : str
        Observing mode identifier.
    baseline_ints : array-like(int)
        Integration indices defining baseline(s) for the spectral term.
    wave_range : None or [min, max]
        If given, restrict spectral term to this wavelength range.
    w1, w2 : float
        Weights for white-light and spectral terms in final cost.

    Returns
    -------
    cost : float
        Combined cost = w1*ptp2_white + w2*ptp2_spec
    ptp2_spec_wave : np.ndarray
        Per-wavelength ptp2 metric values.
    """

    # ======== NIRISS-SPECIFIC WAVE + FLUX MERGE ========
    if 'niriss' in obs_mode:
        wave, flux = opt_utils.stitch_soss_orders(st3['Wave O1'], st3['Wave O2'],
                                                  st3['Flux O1'], st3['Flux O2'])

    else:
        # For non-NIRISS: take flux/wave arrays directly
        flux = np.asarray(st3['Flux'], float)
        wave = np.asarray(st3['Wave'], float)
        if wave.ndim == 2:
            # MIRI/NIRSpec Stage 3 outputs store the same wavelength grid for each integration.
            if wave.shape == flux.shape:
                wave = np.nanmedian(wave, axis=0)
            elif 1 in wave.shape:
                wave = np.ravel(wave)
            else:
                raise ValueError(
                    f'Expected 1D wavelength axis or 2D array matching flux; got wave.shape={wave.shape} '
                    f'and flux.shape={flux.shape}'
                )
        elif wave.ndim != 1:
            raise ValueError(f'Expected 1D wavelength axis, got wave.ndim={wave.ndim}')

    # ======== WHITE LIGHT TERM ========
    # Collapse all wavelengths into single white light curve
    white = np.nansum(flux, axis=1)
    white = white[~np.isnan(white)]
    norm_white = white / np.median(white)
    # 2nd finite difference (neighbour avg - centre)
    d2_white = 0.5*(norm_white[:-2] + norm_white[2:]) - norm_white[1:-1]
    ptp2_white = np.nanmedian(np.abs(d2_white))

    # ======== SPECTRAL TERM (PER-WAVELENGTH P2P) ========
    wave_meds = np.nanmedian(flux, axis=0, keepdims=True)
    norm_spec = flux / wave_meds
    d2_spec = 0.5*(norm_spec[:-2] + norm_spec[2:]) - norm_spec[1:-1]

    # Select baseline integrations for spectral metric
    if baseline_ints is None:
        ptp2_spec_wave = np.nanmedian(np.abs(d2_spec), axis=0)
    elif len(baseline_ints) == 1:
        nn = int(baseline_ints[0])
        ptp2_spec_wave = np.nanmedian(np.abs(d2_spec[:nn]), axis=0)
    elif len(baseline_ints) == 2:
        nlow, nhigh = map(int, baseline_ints)
        low, high = d2_spec[:nlow], d2_spec[nhigh:]
        # Phase 1 only uses the first segment, so a positive nhigh can lie beyond its end. Fall
        # back on whichever baseline is available rather than returning an all-NaN metric.
        if len(low) > 0 and len(high) > 0:
            low_term = np.nanmedian(np.abs(low), axis=0)
            high_term = np.nanmedian(np.abs(high), axis=0)
            ptp2_spec_wave = 0.5 * (low_term + high_term)
        elif len(low) > 0 or len(high) > 0:
            ptp2_spec_wave = np.nanmedian(np.abs(low if len(low) > 0 else high), axis=0)
        else:
            raise ValueError(f'baseline_ints {baseline_ints} select no integrations '
                             f'(only {len(d2_spec) + 2} available).')
    else:
        raise ValueError(f'baseline_ints must be length 1 or 2, got {len(baseline_ints)}')

    # ======== WAVELENGTH RANGE FILTER (OPTIONAL) ========
    if wave_range is None:
        ptp2_spec = np.nanmedian(ptp2_spec_wave)

    elif isinstance(wave_range, (list, tuple)) and len(wave_range) == 2:
        lo, hi = wave_range
        finite = np.isfinite(wave)
        if not finite.any():
            raise ValueError('All entries in wave are NaN!')

        wave_min = np.nanmin(wave[finite])
        wave_max = np.nanmax(wave[finite])

        # Handle None values (means use data min/max)
        if lo is None:
            lo = wave_min
        if hi is None:
            hi = wave_max

        # Distances from requested range edges
        dist_lo = np.abs(wave - lo)
        dist_lo[~finite] = np.inf
        dist_hi = np.abs(wave - hi)
        dist_hi[~finite] = np.inf

        idx_lo = int(np.argmin(dist_lo))
        idx_hi = int(np.argmin(dist_hi))

        # Slice range in correct order
        i0, i1 = sorted((idx_lo, idx_hi))
        sub = ptp2_spec_wave[i0:i1+1]
        if np.all(np.isnan(sub)):
            raise ValueError(f'No valid ptp2_spec values in wave range {wave_range}')
        ptp2_spec = np.nanmedian(sub)

    else:
        raise ValueError('wave_range must be None or a length-2 list/tuple')

    # ======== FINAL COST COMBINATION ========
    # Avoid allowing a zero-weighted NaN term to poison the selected metric
    # (IEEE arithmetic makes 0.0 * NaN evaluate to NaN).
    cost = 0.0
    if w1 != 0:
        cost += w1 * ptp2_white
    if w2 != 0:
        cost += w2 * ptp2_spec

    return cost, ptp2_spec_wave


def run_stage3_for_width(stage2_inputs, cfg, centroids, deepframe, extract_width):
    """Run Stage 3 once for a specific extraction width.
    """

    return run_stage3(
        stage2_inputs,
        save_results=True,
        force_redo=True,
        extract_method=cfg['extract_method'],
        soss_specprofile=cfg['soss_specprofile'],
        centroids=centroids,
        extract_width=extract_width,
        extract_width_soss2=cfg['extract_width_soss2'],
        st_teff=cfg['st_teff'],
        st_logg=cfg['st_logg'],
        st_met=cfg['st_met'],
        planet_letter=cfg['planet_letter'],
        output_tag=cfg['output_tag'],
        do_plot=cfg['do_plots'],
        deepframe=deepframe,
        root_dir=root_dir,
        **cfg['stage3_kwargs']
    )


def run_ad_hoc_extract_width_search(stage2_inputs, cfg, centroids, deepframe, baseline_ints,
                                    wave_range, w1, w2, name_str, base_row_values):
    """Append an ad hoc Stage 3 extraction sweep to the optimizer logs.
    """

    obs_mode = cfg['observing_mode'].lower()

    if cfg['optimize_extract_width']:
        extract_widths = cfg['extract_width']
        if not isinstance(extract_widths, list):
            raise ValueError('extract_width must be a list when optimize_extract_width=True')
    else:
        extract_widths = cfg['extract_width']
        if isinstance(extract_widths, list):
            extract_widths = [extract_widths[0]]
        else:
            extract_widths = [extract_widths]

    required_cols = ['ad_hoc_mode', 'remove_components']
    if 'extract_width' not in required_cols:
        required_cols.append('extract_width')
    cost_path, param_cols, row_offset, best_logged = opt_utils.prepare_cost_log(name_str,
                                                                                required_cols,
                                                                                outdir_f)
    if best_logged:
        merged_row_values = best_logged.copy()
        merged_row_values.update(base_row_values)
    else:
        merged_row_values = base_row_values.copy()

    extract_costs = []
    appended_rows = []
    best_stage3_results = None

    for idx, width in enumerate(extract_widths):
        fancyprint(f'\n{'='*60}')
        fancyprint(f'Testing extract_width={width}')
        fancyprint(f'{'='*60}\n')
        t0 = time.perf_counter()

        stage3_results = run_stage3_for_width(stage2_inputs, cfg, centroids, deepframe, width)
        cost, scatter = cost_function(stage3_results, baseline_ints=baseline_ints,
                                      wave_range=wave_range, w1=w1, w2=w2, obs_mode=obs_mode)

        dt = time.perf_counter() - t0
        extract_costs.append(cost)
        appended_rows.append(row_offset + idx)
        this_row = merged_row_values.copy()
        this_row['extract_width'] = width
        opt_utils.append_cost_log_row(cost_path, param_cols, this_row, dt, cost)
        opt_utils.append_scatter_log_row(name_str, scatter, outdir_f)

        fancyprint(f'extract_width={width}: cost={cost:.12f} ({dt:.1f}s)')

        if best_stage3_results is None or cost <= np.nanmin(extract_costs):
            best_stage3_results = stage3_results

    best_idx = opt_utils.select_best_trial(extract_costs, 'extract_width')
    best_extract_width = extract_widths[best_idx]
    best_cost = extract_costs[best_idx]
    best_row_idx = appended_rows[best_idx]

    fancyprint(f'\n*** Best extract_width={best_extract_width} with cost={best_cost:.6f} ***\n')

    final_stage3_results = run_stage3_for_width(stage2_inputs, cfg, centroids, deepframe,
                                                best_extract_width)

    return final_stage3_results, best_extract_width, best_cost, best_row_idx


def save_config(cfg):
    """Save a copy of the DMS config file.
    """

    # Save a copy of the config file.
    if cfg['output_tag'] != '':
        output_tag = '_' + cfg['output_tag']
    else:
        output_tag = cfg['output_tag']
    root_dir = cfg['root_dir']
    verify_path(root_dir)
    root_dir += 'pipeline_outputs_directory' + output_tag
    verify_path(root_dir)
    root_dir += '/Optimizer_Files'
    verify_path(root_dir)
    i = 0
    copy_config = root_dir + '/' + config_file
    while os.path.exists(copy_config):
        i += 1
        copy_config = root_dir + '/' + config_file
        root = copy_config.split('.yaml')[0]
        copy_config = root + '_{}.yaml'.format(i)
    shutil.copy(config_file, copy_config)
    # Append time at which it was run.
    f = open(copy_config, 'a')
    runtime = datetime.utcnow().isoformat(sep=' ', timespec='minutes')
    f.write('\nRun at {}.'.format(runtime))
    f.close()

    return


def run_optimizer(cfg):
    """Run the optimizer.
    """

    # ===== Initial Setup =====
    obs_mode = cfg['observing_mode'].lower()
    filter = cfg['filter_detector'].lower()
    # Fail fast rather than after Phase 1: Stage 3 only supports these extraction methods.
    if cfg['extract_method'] not in ['box', 'atoca', 'optimal']:
        raise ValueError('extract_method must be one of "box", "atoca", or "optimal"; got '
                         '{!r}.'.format(cfg['extract_method']))
    instrument = obs_mode.split('/')[0].upper() if '/' in obs_mode else obs_mode.upper()

    # Key parameters
    baseline_ints = cfg['baseline_ints']
    name_str = cfg['run_name']
    if name_str != '':
        name_str = '_' + name_str

    # Read and double check wavelengths.
    wave_range = resolve_spectral_wave_range(cfg['wave_range'], obs_mode, filter, bands)

    # Cost function weights.
    w1 = cfg['w1']
    w2 = cfg['w2']

    t0_total = time.perf_counter()
    optimize_extract_width_only = cfg['optimize_extract_width_only']
    from_pca_only = cfg['from_pca_only']
    extract_method = cfg['extract_method']

    # Can only run one of the above at a atime.
    if optimize_extract_width_only and from_pca_only:
        raise ValueError('optimize_extract_width_only and from_pca_only cannot both be True.')

    # ==========================================================
    # ===== Special Case 1: Optimize Extraction Width Only =====
    # ==========================================================
    if optimize_extract_width_only:
        fancyprint(f'\n{'='*60}')
        fancyprint('EXTRACT WIDTH ONLY MODE ENABLED')
        fancyprint('Skipping directly to Stage 3 using existing Stage 2 outputs')
        fancyprint(f'{'='*60}\n')

        # Determine the source directory for Stage 2 inputs
        stage2_source_dir = outdir_s2
        input_dir_cfg = cfg.get('input_dir')
        if input_dir_cfg not in [None, 'None', 'null', '']:
            possible_dirs = [
                input_dir_cfg,
                os.path.join(input_dir_cfg, 'Stage2'),
                os.path.join(input_dir_cfg, 'Stage2/'),
            ]
            for p_dir in possible_dirs:
                if os.path.isdir(p_dir):
                    if glob.glob(os.path.join(p_dir, '*_badpixstep.fits')) or glob.glob(os.path.join(p_dir, '*_pcareconstructstep.fits')):
                        stage2_source_dir = p_dir
                        if not stage2_source_dir.endswith('/'):
                            stage2_source_dir += '/'
                        fancyprint(f'Detected Stage 2 outputs in input_dir: {stage2_source_dir}')
                        break

        fancyprint(f'Looking for existing Stage 2 outputs in {stage2_source_dir}...')
        pca_step = cfg.get('PCAReconstructStep', 'run')
        if pca_step == 'skip' or cfg.get('remove_components') in [None, 'None', 'null', '', []]:
            patterns = [
                f'{stage2_source_dir}*_badpixstep.fits',
                f'{stage2_source_dir}*_pcareconstructstep.fits',
            ]
        else:
            patterns = [
                f'{stage2_source_dir}*_pcareconstructstep.fits',
                f'{stage2_source_dir}*_badpixstep.fits',
            ]
        stage2_files = opt_utils.find_existing_stage2_outputs(
            patterns,
            f'No Stage 2 outputs found in {stage2_source_dir}. '
            'Please run the full pipeline first before using optimize_extract_width_only mode.'
        )
        fancyprint('Looking for centroids file...')
        centroids_df = opt_utils.load_ad_hoc_centroids(cfg, outdir_s2, outdir_s3,
                                                       stage2_source_dir=stage2_source_dir)
        deepframe = opt_utils.resolve_ad_hoc_deepframe(cfg, outdir_s2,
                                                       stage2_source_dir=stage2_source_dir)

        fancyprint(f'Using Stage 2 outputs: {stage2_files}')
        base_row_values = {
            'ad_hoc_mode': 'extract_width_only',
            'remove_components': cfg.get('remove_components'),
        }
        rerun_cfg = cfg.copy()
        rerun_cfg['extract_width'] = opt_utils.resolve_ad_hoc_extract_width(cfg, outdir_f,
                                                                            outdir_s3)
        stage3_results, best_extract_width, _, best_row_idx = run_ad_hoc_extract_width_search(
            stage2_files, rerun_cfg, centroids_df, deepframe, baseline_ints, wave_range, w1, w2,
            name_str, base_row_values
        )

        fancyprint('Generating optimization plots...')
        plot_cost(name_str, outdir=outdir_f)
        make_lightcurve_plots(stage3_results, name_str=name_str, baseline_ints=baseline_ints,
                              obs=obs_mode, outdir=outdir_f)

        outfile = os.path.join(outdir_f, f'LightCurve_Scatter{name_str}.txt')
        specfile = opt_utils.find_stage3_spectrum_file(extract_method, outdir_s3)
        plot_file_path = os.path.join(outdir_f, f'Scatter_Plot{name_str}.png')
        plot_scatter(txtfile=outfile, rows=[best_row_idx], wave_range=None,
                     spectrum_files=[specfile], save_path=plot_file_path)

        t1 = time.perf_counter() - t0_total
        h, m = divmod(int(t1), 3600)
        m, s = divmod(m, 60)
        width_label = 'OPTIMAL EXTRACT_WIDTH'
        if cfg.get('optimize_extract_width', False) is not True:
            width_label = 'USED EXTRACT_WIDTH'
        fancyprint(f'\n{'='*60}')
        fancyprint(f'TOTAL RUNTIME: {h}h {m:02d}min {s:02d}s')
        fancyprint(f'{width_label}: {best_extract_width}')
        fancyprint(f'{'='*60}\n')

        return

    # ===========================================================
    # ===== Special Case 2: PCA + Optimize Extraction Width =====
    # ===========================================================
    if from_pca_only:
        fancyprint(f'\n{'='*60}')
        fancyprint('FROM PCA ONLY MODE ENABLED')
        fancyprint('Restarting from existing BadPix outputs and rerunning PCA/Stage 3 only')
        fancyprint(f'{'='*60}\n')

        remove_components = cfg.get('remove_components')
        if remove_components in [None, 'None', 'null', '']:
            raise ValueError('remove_components must be set when from_pca_only=True')

        fancyprint('Looking for existing BadPix Step outputs...')
        badpix_files = opt_utils.find_existing_stage2_outputs(
            [f'{outdir_s2}*_badpixstep.fits'],
            f'No BadPix Step outputs found in {outdir_s2}. '
            'Please run the optimizer through BadPixStep before using from_pca_only mode.'
        )
        fancyprint('Looking for centroids file...')
        centroids_df = opt_utils.load_ad_hoc_centroids(cfg, outdir_s2, outdir_s3)

        pca_skip_steps = ['AssignWCSStep', 'FlatFieldStep', 'BackgroundStep', 'OneOverFStep',
                          'BadPixStep']
        fancyprint(f'Rerunning PCAReconstructStep with remove_components={remove_components}')
        stage2_results, deepframe = run_stage2(
            badpix_files,
            mode=cfg['observing_mode'],
            soss_background_model=cfg.get('soss_background_file'),
            baseline_ints=cfg['baseline_ints'],
            save_results=True,
            force_redo=True,
            space_thresh=cfg.get('space_outlier_threshold'),
            time_thresh=cfg.get('time_outlier_threshold'),
            remove_components=remove_components,
            pca_components=cfg.get('pca_components'),
            soss_timeseries=cfg.get('soss_timeseries'),
            soss_timeseries_o2=cfg.get('soss_timeseries_o2'),
            oof_method=cfg.get('oof_method'),
            output_tag=cfg['output_tag'],
            skip_steps=pca_skip_steps,
            generate_lc=cfg.get('generate_lc'),
            soss_inner_mask_width=cfg.get('soss_inner_mask_width'),
            soss_outer_mask_width=cfg.get('soss_outer_mask_width'),
            nirspec_mask_width=cfg.get('nirspec_mask_width'),
            pixel_masks=cfg.get('outlier_maps'),
            f277w=cfg.get('f277w'),
            do_plot=cfg.get('do_plots', False),
            centroids=cfg.get('centroids'),
            miri_trace_width=cfg.get('miri_trace_width'),
            miri_background_width=cfg.get('miri_background_width'),
            miri_background_method=cfg.get('miri_background_method'),
            root_dir=root_dir,
            **cfg.get('stage2_kwargs', {})
        )
        if deepframe is None:
            deepframe = resolve_ad_hoc_deepframe(cfg)

        base_row_values = {
            'ad_hoc_mode': 'from_pca_only',
            'remove_components': remove_components,
        }
        rerun_cfg = cfg.copy()
        rerun_cfg['extract_width'] = opt_utils.resolve_ad_hoc_extract_width(cfg, outdir_f,
                                                                            outdir_s3)
        stage3_results, best_extract_width, _, best_row_idx = run_ad_hoc_extract_width_search(
            stage2_results, rerun_cfg, centroids_df, deepframe, baseline_ints, wave_range, w1, w2,
            name_str, base_row_values
        )

        fancyprint('Generating optimization plots...')
        plot_cost(name_str, outdir=outdir_f)
        make_lightcurve_plots(stage3_results, name_str=name_str, baseline_ints=baseline_ints,
                              obs=obs_mode, outdir=outdir_f)

        outfile = os.path.join(outdir_f, f'LightCurve_Scatter{name_str}.txt')
        specfile = opt_utils.find_stage3_spectrum_file(extract_method, outdir_s3)
        plot_file_path = os.path.join(outdir_f, f'Scatter_Plot{name_str}.png')
        plot_scatter(txtfile=outfile, rows=[best_row_idx], wave_range=None,
                     spectrum_files=[specfile], save_path=plot_file_path)

        t1 = time.perf_counter() - t0_total
        h, m = divmod(int(t1), 3600)
        m, s = divmod(m, 60)
        width_label = 'OPTIMAL EXTRACT_WIDTH'
        if cfg.get('optimize_extract_width', False) is not True:
            width_label = 'USED EXTRACT_WIDTH'
        fancyprint(f'\n{'='*60}')
        fancyprint(f'TOTAL RUNTIME: {h}h {m:02d}min {s:02d}s')
        fancyprint(f'{width_label}: {best_extract_width}')
        fancyprint(f'REMOVE_COMPONENTS: {opt_utils.format_log_value(remove_components)}')
        fancyprint(f'{'='*60}\n')

        return

    # ==============================================
    # ===== Normal Mode: Run Full Optimization =====
    # ==============================================
    # Load input files
    input_files = unpack_input_dir(
        cfg['input_dir'],
        mode=cfg['observing_mode'],
        filetag=cfg['input_filetag'],
        filter_detector=cfg['filter_detector'],
    )
    if isinstance(input_files, np.ndarray):
        input_files = input_files.tolist()

    if not input_files:
        raise RuntimeError(f'No FITS found in {cfg['input_dir']}')

    fancyprint(f'Found {len(input_files)} segment(s) from {cfg['input_dir']}')
    fancyprint(f'=== PHASE 1: OPTIMIZATION ON FIRST SEGMENT ONLY ===')

    # use only first segment for optimization
    single_segment = [input_files[0]]

    param_ranges = {}  # parametrs to optimize
    fixed_params = {}  # fixed parameters

    optimizer_control_flags = {
        'optimize_extract_width_only',
        'optimize_from_pca_only',
    }
    for k, v in cfg.items():
        if k.startswith('optimize_'):
            if k in optimizer_control_flags:
                continue

            param_name = k[len('optimize_'):]

            # Special handling for extract_width - optimize in Phase 2 using custom cost function
            if param_name == 'extract_width':
                if v:
                    vals = cfg[param_name]
                    if not isinstance(vals, list):
                        raise ValueError(f'{param_name} must be list when optimize_{param_name}=True')
                    fancyprint(f'Will optimize: {param_name} in Phase 2 over {vals} (using spectral scatter cost)')
                    # Add to param_ranges so it shows up in logs and plots
                    param_ranges[param_name] = vals
                else:
                    fixed_params[param_name] = cfg[param_name]
                continue  # Skip the normal processing below

            if param_name not in cfg:
                fancyprint(
                    f'Skipping optimizer control flag "{k}" because "{param_name}" is not a config parameter.',
                    msg_type='WARNING'
                )
                continue

            if v:  # true = optimize (sweep)
                vals = cfg[param_name]
                if not isinstance(vals, list):
                    raise ValueError(f'{param_name} must be list when optimize_{param_name}=True')
                param_ranges[param_name] = vals
                fancyprint(f'Will optimize: {param_name} over {vals}')
            else:
                val = cfg[param_name]
                if isinstance(val, list):
                    raise ValueError(f'{param_name} must be single value when optimize_{param_name}=False')
                fixed_params[param_name] = val

    # Initialize swept parameters with the middle candidate until their own sweep runs (an
    # int-truncated mean can fall outside the candidate list, e.g. for float thresholds).
    current_best = {k: v[len(v) // 2] for k, v in param_ranges.items()}
    current_best.update(fixed_params)

    logf = open(f'{outdir_f}/Cost_Summary{name_str}.txt', 'w')
    logs = open(f'{outdir_f}/LightCurve_Scatter{name_str}.txt', 'w')
    logf.write('\t'.join(param_ranges.keys()) + '\tduration_s\tcost\n')

    # Filter checkpoints to only include those with parameters being optimized
    optimization_checkpoints = []
    for checkpoint in all_checkpoints:
        # Check if any params at this checkpoint are being optimized
        params_to_optimize = [p for p in checkpoint['params'] if p in param_ranges]
        if params_to_optimize:
            # Skip Phase 2-only checkpoints during Phase 1
            if checkpoint.get('phase_2_only', False):
                fancyprint(f'Skipping {checkpoint['name']} - will optimize in Phase 2 on full dataset')
                continue
            optimization_checkpoints.append(checkpoint)
            fancyprint(f'Including checkpoint: {checkpoint['name']} with params {params_to_optimize}')

    # Cache for centroids (generated once, reused)
    centroids = None

    # ~~~ OPTIMIZE EACH CHECKPOINT ~~~
    for checkpoint in optimization_checkpoints:
        # check if any params at this checkpoint need optimization
        params_to_optimize = [p for p in checkpoint['params'] if p in param_ranges]

        if not params_to_optimize:
            fancyprint(f'Skipping {checkpoint['name']}: no parameters to optimize')
            continue

        fancyprint(f'\n{'='*60}')
        fancyprint(f'OPTIMIZING AT: {checkpoint['name']} (Stage {checkpoint['stage']})')
        fancyprint(f'Parameters: {params_to_optimize}')
        fancyprint(f'{'='*60}\n')

        # for each parameter at this checkpoint
        for param_name in params_to_optimize:
            param_values = param_ranges[param_name]
            fancyprint(f'\n--- Sweeping {param_name}: {param_values} ---')

            costs = []
            scatters = []
            # sweep through parameter values
            for param_value in param_values:
                t0 = time.perf_counter()

                # updaete config with current parameter
                run_cfg = cfg.copy()
                run_cfg.update(current_best)  # use best values from previous optimizations
                run_cfg[param_name] = param_value  # Current  value

                fancyprint(f'\nTesting {param_name}={param_value}')

                # Delete cached output for the optimization step to force rerun from that step
                opt_utils.delete_checkpoint_outputs(checkpoint['name'], outdir_s1, outdir_s2)

                # run pipeline up to (including this step)
                if checkpoint['stage'] == 1:
                    # Build skip list: skip everything after this step
                    skip_list = checkpoint['skip_after'].copy()

                    # ALSO add user's skip preferences from YAML config
                    for step in stage1_steps:
                        if run_cfg.get(step) == 'skip' and step not in skip_list:
                            if step == 'OneOverFStep_grp':
                                skip_list.append('OneOverFStep')
                            else:
                                skip_list.append(step)

                    # Forward the current time_window (candidate while sweeping it,
                    # winner/fixed value otherwise) to JumpStep.
                    s1_kwargs = opt_utils.stage1_kwargs_with_winners(run_cfg)

                    # Run Stage1 with force_redo=False (deleted file triggers rerun from that step).
                    stage1_results = run_stage1(
                        single_segment,
                        mode=run_cfg['observing_mode'],
                        soss_background_model=run_cfg.get('soss_background_file'),
                        baseline_ints=run_cfg['baseline_ints'],
                        oof_method=run_cfg.get('oof_method'),
                        superbias_method=run_cfg.get('superbias_method'),
                        soss_timeseries=run_cfg.get('soss_timeseries'),
                        soss_timeseries_o2=run_cfg.get('soss_timeseries_o2'),
                        save_results=True,
                        pixel_masks=run_cfg.get('outlier_maps'),
                        force_redo=False,
                        flag_up_ramp=run_cfg.get('flag_up_ramp', False),
                        rejection_threshold=run_cfg.get('jump_threshold', 15),
                        flag_in_time=run_cfg.get('flag_in_time', True),
                        time_rejection_threshold=run_cfg.get('time_jump_threshold'),
                        output_tag=run_cfg['output_tag'],
                        skip_steps=skip_list,
                        do_plot=run_cfg.get('do_plots', False),
                        soss_inner_mask_width=run_cfg.get('soss_inner_mask_width'),
                        soss_outer_mask_width=run_cfg.get('soss_outer_mask_width'),
                        nirspec_mask_width=run_cfg.get('nirspec_mask_width'),
                        centroids=run_cfg.get('centroids'),
                        hot_pixel_map=run_cfg.get('hot_pixel_map'),
                        miri_drop_groups=run_cfg.get('miri_drop_groups'),
                        saturation_threshold=run_cfg.get('saturation_threshold', 80),
                        f277w=run_cfg.get('f277w'),
                        inl_amplitude_file=run_cfg.get('inl_amplitude_file'),
                        inl_periods=run_cfg.get('inl_periods'),
                        root_dir=root_dir,
                        **s1_kwargs
                    )

                    # Extract from Stage 1 output
                    datafile = stage1_results[0]

                elif checkpoint['stage'] == 2:
                    # First, need Stage 1 results (use cached)
                    # Build skip list for Stage 1 based on user config
                    stage1_skip_for_s2 = []
                    for step in stage1_steps:
                        if run_cfg.get(step) == 'skip':
                            if step == 'OneOverFStep_grp':
                                stage1_skip_for_s2.append('OneOverFStep')
                            else:
                                stage1_skip_for_s2.append(step)

                    stage1_results = run_stage1(
                        single_segment,
                        mode=run_cfg['observing_mode'],
                        soss_background_model=run_cfg.get('soss_background_file'),
                        baseline_ints=run_cfg['baseline_ints'],
                        oof_method=run_cfg.get('oof_method'),
                        superbias_method=run_cfg.get('superbias_method'),
                        soss_timeseries=run_cfg.get('soss_timeseries'),
                        soss_timeseries_o2=run_cfg.get('soss_timeseries_o2'),
                        save_results=True,
                        pixel_masks=run_cfg.get('outlier_maps'),
                        force_redo=False,  # Use cached Stage 1 results
                        flag_up_ramp=run_cfg.get('flag_up_ramp', False),
                        rejection_threshold=run_cfg.get('jump_threshold', 15),
                        flag_in_time=run_cfg.get('flag_in_time', True),
                        time_rejection_threshold=run_cfg.get('time_jump_threshold'),
                        output_tag=run_cfg['output_tag'],
                        skip_steps=stage1_skip_for_s2,
                        do_plot=run_cfg.get('do_plots', False),
                        soss_inner_mask_width=run_cfg.get('soss_inner_mask_width'),
                        soss_outer_mask_width=run_cfg.get('soss_outer_mask_width'),
                        nirspec_mask_width=run_cfg.get('nirspec_mask_width'),
                        centroids=run_cfg.get('centroids'),
                        hot_pixel_map=run_cfg.get('hot_pixel_map'),
                        miri_drop_groups=run_cfg.get('miri_drop_groups'),
                        root_dir=root_dir,
                        saturation_threshold=run_cfg.get('saturation_threshold', 80),
                        f277w=run_cfg.get('f277w'),
                        inl_amplitude_file=run_cfg.get('inl_amplitude_file'),
                        inl_periods=run_cfg.get('inl_periods'),
                        **opt_utils.stage1_kwargs_with_winners(run_cfg)
                    )

                    # Build skip list for Stage 2
                    skip_list = checkpoint['skip_after'].copy()

                    # ALSO add user's skip preferences from YAML config
                    for step in stage2_steps:
                        if run_cfg.get(step) == 'skip' and step not in skip_list:
                            if step == 'OneOverFStep_int':
                                skip_list.append('OneOverFStep')
                            else:
                                skip_list.append(step)

                    # Forward the current box_size/window_size (candidate while
                    # sweeping them, winner/fixed values otherwise) to BadPixStep.
                    s2_kwargs = opt_utils.stage2_kwargs_with_winners(run_cfg)

                    # Run Stage 2 with force_redo=False
                    # The deleted cached file will trigger rerun from that step onward
                    stage2_results, _ = run_stage2(
                        stage1_results,
                        mode=run_cfg['observing_mode'],
                        soss_background_model=run_cfg.get('soss_background_file'),
                        baseline_ints=run_cfg['baseline_ints'],
                        save_results=True,
                        force_redo=False,  # Use cached until missing file triggers rerun
                        space_thresh=run_cfg.get('space_outlier_threshold'),
                        time_thresh=run_cfg.get('time_outlier_threshold'),
                        remove_components=run_cfg.get('remove_components'),
                        pca_components=run_cfg.get('pca_components'),
                        soss_timeseries=run_cfg.get('soss_timeseries'),
                        soss_timeseries_o2=run_cfg.get('soss_timeseries_o2'),
                        oof_method=run_cfg.get('oof_method'),
                        output_tag=run_cfg['output_tag'],
                        skip_steps=skip_list,
                        generate_lc=run_cfg.get('generate_lc'),
                        soss_inner_mask_width=run_cfg.get('soss_inner_mask_width'),
                        soss_outer_mask_width=run_cfg.get('soss_outer_mask_width'),
                        nirspec_mask_width=run_cfg.get('nirspec_mask_width'),
                        pixel_masks=run_cfg.get('outlier_maps'),
                        f277w=run_cfg.get('f277w'),
                        do_plot=run_cfg.get('do_plots', False),
                        centroids=run_cfg.get('centroids'),
                        miri_trace_width=run_cfg.get('miri_trace_width'),
                        miri_background_width=run_cfg.get('miri_background_width'),
                        miri_background_method=run_cfg.get('miri_background_method'),
                        root_dir=root_dir,
                        **s2_kwargs
                    )

                    datafile = stage2_results[0]

                elif checkpoint['stage'] == 3:
                    # Need Stage 1 and 2 completed first (use cached)
                    # Build skip list for Stage 1 based on user config
                    stage1_skip_for_s3 = []
                    for step in stage1_steps:
                        if run_cfg.get(step) == 'skip':
                            if step == 'OneOverFStep_grp':
                                stage1_skip_for_s3.append('OneOverFStep')
                            else:
                                stage1_skip_for_s3.append(step)

                    stage1_results = run_stage1(
                        single_segment,
                        mode=run_cfg['observing_mode'],
                        soss_background_model=run_cfg.get('soss_background_file'),
                        baseline_ints=run_cfg['baseline_ints'],
                        oof_method=run_cfg.get('oof_method'),
                        superbias_method=run_cfg.get('superbias_method'),
                        soss_timeseries=run_cfg.get('soss_timeseries'),
                        soss_timeseries_o2=run_cfg.get('soss_timeseries_o2'),
                        save_results=True,
                        pixel_masks=run_cfg.get('outlier_maps'),
                        force_redo=False,
                        flag_up_ramp=run_cfg.get('flag_up_ramp', False),
                        rejection_threshold=run_cfg.get('jump_threshold', 15),
                        flag_in_time=run_cfg.get('flag_in_time', True),
                        time_rejection_threshold=run_cfg.get('time_jump_threshold'),
                        output_tag=run_cfg['output_tag'],
                        skip_steps=stage1_skip_for_s3,
                        do_plot=run_cfg.get('do_plots', False),
                        soss_inner_mask_width=run_cfg.get('soss_inner_mask_width'),
                        soss_outer_mask_width=run_cfg.get('soss_outer_mask_width'),
                        nirspec_mask_width=run_cfg.get('nirspec_mask_width'),
                        centroids=run_cfg.get('centroids'),
                        hot_pixel_map=run_cfg.get('hot_pixel_map'),
                        miri_drop_groups=run_cfg.get('miri_drop_groups'),
                        root_dir=root_dir,
                        saturation_threshold=run_cfg.get('saturation_threshold', 80),
                        f277w=run_cfg.get('f277w'),
                        inl_amplitude_file=run_cfg.get('inl_amplitude_file'),
                        inl_periods=run_cfg.get('inl_periods'),
                        **opt_utils.stage1_kwargs_with_winners(run_cfg)
                    )

                    # Build skip list for Stage 2 based on user config
                    stage2_skip_for_s3 = []
                    for step in stage2_steps:
                        if run_cfg.get(step) == 'skip':
                            if step == 'OneOverFStep_int':
                                stage2_skip_for_s3.append('OneOverFStep')
                            else:
                                stage2_skip_for_s3.append(step)

                    stage2_results, _ = run_stage2(
                        stage1_results,
                        mode=run_cfg['observing_mode'],
                        soss_background_model=run_cfg.get('soss_background_file'),
                        baseline_ints=run_cfg['baseline_ints'],
                        save_results=True,
                        force_redo=False,
                        space_thresh=run_cfg.get('space_outlier_threshold'),
                        time_thresh=run_cfg.get('time_outlier_threshold'),
                        remove_components=run_cfg.get('remove_components'),
                        pca_components=run_cfg.get('pca_components'),
                        soss_timeseries=run_cfg.get('soss_timeseries'),
                        soss_timeseries_o2=run_cfg.get('soss_timeseries_o2'),
                        oof_method=run_cfg.get('oof_method'),
                        output_tag=run_cfg['output_tag'],
                        skip_steps=stage2_skip_for_s3,
                        generate_lc=run_cfg.get('generate_lc'),
                        soss_inner_mask_width=run_cfg.get('soss_inner_mask_width'),
                        soss_outer_mask_width=run_cfg.get('soss_outer_mask_width'),
                        nirspec_mask_width=run_cfg.get('nirspec_mask_width'),
                        pixel_masks=run_cfg.get('outlier_maps'),
                        f277w=run_cfg.get('f277w'),
                        do_plot=run_cfg.get('do_plots', False),
                        centroids=run_cfg.get('centroids'),
                        miri_trace_width=run_cfg.get('miri_trace_width'),
                        miri_background_width=run_cfg.get('miri_background_width'),
                        miri_background_method=run_cfg.get('miri_background_method'),
                        root_dir=root_dir,
                        **opt_utils.stage2_kwargs_with_winners(run_cfg)
                    )

                    datafile = stage2_results[0]

                # Extract and compute cost <- new function
                # For Phase 1, use a fixed extract_width (will be optimized in Phase 2)
                phase1_extract_width = cfg.get('extract_width')
                if isinstance(phase1_extract_width, list):
                    # If it's a list (optimize_extract_width=True), use middle value for Phase 1
                    phase1_extract_width = phase1_extract_width[len(phase1_extract_width) // 2]
                    fancyprint(f'Using extract_width={phase1_extract_width} for Phase 1 (will optimize in Phase 2).')

                spectral_dict, centroids = opt_utils.extract_at_step(
                    datafile=datafile,
                    instrument=instrument,
                    extract_width=phase1_extract_width,
                    centroids=centroids,  # Reuse cached
                    baseline_ints=baseline_ints,
                    output_dir=outdir_s2,
                    extract_method=cfg.get('extract_method', 'box'),
                    extract_width_soss2=cfg.get('extract_width_soss2'),
                    extract_step_kwargs=opt_utils.resolve_extract1d_kwargs(cfg)
                )

                # The fast NIRSpec/MIRI optimizer-side extraction uses pixel indices as
                # placeholder wavelengths, so micron-space filtering is only valid here for SOSS.
                if instrument.lower() == 'niriss' and wave_range is not None:
                    phase1_wave_range = wave_range
                else:
                    phase1_wave_range = None

                cost, scatter = cost_function(spectral_dict, baseline_ints=baseline_ints,
                                              wave_range=phase1_wave_range, w1=w1, w2=w2,
                                              obs_mode=obs_mode)

                # Debug cost details
                fancyprint(f'  Cost function: w1={w1}, w2={w2}, wave_range={phase1_wave_range}')
                fancyprint(f'  Scatter: min={np.nanmin(scatter):.6e}, max={np.nanmax(scatter):.6e}, median={np.nanmedian(scatter):.6e}')
                fancyprint(f'  Valid scatter values: {np.sum(np.isfinite(scatter))}/{len(scatter)}')

                dt = time.perf_counter() - t0
                costs.append(cost)
                scatters.append(scatter)

                fancyprint(f'{param_name}={param_value}: cost={cost:.12f} ({dt:.1f}s)')

                # Log results
                log_line = '\t'.join(str(run_cfg.get(p, '')) for p in param_ranges.keys())
                logf.write(f'{log_line}\t{dt:.1f}\t{cost:.12f}\n')
                logf.flush()

                scatter_line = ' '.join(f'{x:.10g}' for x in scatter)
                logs.write(f'{scatter_line}\n')
                logs.flush()

            # Find best value for this parameter (non-finite costs cannot win)
            best_idx = opt_utils.select_best_trial(costs, param_name)
            best_value = param_values[best_idx]
            best_cost = costs[best_idx]

            current_best[param_name] = best_value
            fancyprint(f'\n*** Best {param_name}={best_value} with cost={best_cost:.6f} ***\n')

            # The cached step outputs on disk belong to the LAST value tested,
            # not necessarily the winner. Delete them so the next pipeline call
            # (the following sweep, or Phase 2) regenerates this checkpoint --
            # and, lazily, its downstream caches -- with the winning value.
            if best_idx != len(param_values) - 1:
                opt_utils.delete_checkpoint_outputs(checkpoint['name'], outdir_s1, outdir_s2)

    logf.close()
    logs.close()

    # Only plot if Phase 1 actually logged a sweep (e.g., not when only extract_width is
    # being optimized).
    phase1_costs = pd.read_csv(f'{outdir_f}/Cost_Summary{name_str}.txt', sep='\t')
    if len(phase1_costs) > 0:
        fancyprint('\n=== Plotting optimization results ===')
        plot_cost(name_str, outdir=outdir_f)

    # ===== PHASE 2: FULL PIPELINE WITH OPTIMAL PARAMETERS =====
    fancyprint(f'\n{'='*60}')
    fancyprint('PHASE 2: FULL PIPELINE WITH OPTIMAL PARAMETERS')
    fancyprint(f'Using ALL {len(input_files)} segments')
    fancyprint(f'Optimal parameters: {current_best}')
    fancyprint(f'{'='*60}\n')

    #  set up config of full pipeline with optimal parameters
    final_cfg = cfg.copy()
    final_cfg.update(current_best)

    # Build skip lists for Stage 1 and Stage 2 based on config settings
    stage1_skip = []
    for step in stage1_steps:
        if final_cfg.get(step) == 'skip':
            if step == 'OneOverFStep_grp':
                stage1_skip.append('OneOverFStep')
            else:
                stage1_skip.append(step)

    fancyprint(f'Stage 1 steps to skip: {stage1_skip}')

    # Stage 1
    stage1_results = run_stage1(
        input_files,
        mode=final_cfg['observing_mode'],
        soss_background_model=final_cfg.get('soss_background_file'),
        baseline_ints=final_cfg['baseline_ints'],
        oof_method=final_cfg.get('oof_method'),
        superbias_method=final_cfg.get('superbias_method'),
        soss_timeseries=final_cfg.get('soss_timeseries'),
        soss_timeseries_o2=final_cfg.get('soss_timeseries_o2'),
        save_results=True,
        pixel_masks=final_cfg.get('outlier_maps'),
        force_redo=True,
        flag_up_ramp=final_cfg.get('flag_up_ramp', False),
        rejection_threshold=final_cfg.get('jump_threshold', 15),
        flag_in_time=final_cfg.get('flag_in_time', True),
        time_rejection_threshold=final_cfg.get('time_jump_threshold'),
        output_tag=final_cfg['output_tag'],
        skip_steps=stage1_skip,
        do_plot=final_cfg.get('do_plots', False),
        soss_inner_mask_width=final_cfg.get('soss_inner_mask_width'),
        soss_outer_mask_width=final_cfg.get('soss_outer_mask_width'),
        nirspec_mask_width=final_cfg.get('nirspec_mask_width'),
        centroids=final_cfg.get('centroids'),
        hot_pixel_map=final_cfg.get('hot_pixel_map'),
        miri_drop_groups=final_cfg.get('miri_drop_groups'),
        root_dir=root_dir,
        saturation_threshold=final_cfg.get('saturation_threshold', 80),
        f277w=final_cfg.get('f277w'),
        inl_amplitude_file=final_cfg.get('inl_amplitude_file'),
        inl_periods=final_cfg.get('inl_periods'),
        **opt_utils.stage1_kwargs_with_winners(final_cfg)
    )

    # Build skip list for Stage 2
    stage2_skip = []
    for step in stage2_steps:
        if final_cfg.get(step) == 'skip':
            if step == 'OneOverFStep_int':
                stage2_skip.append('OneOverFStep')
            else:
                stage2_skip.append(step)

    fancyprint(f'Stage 2 steps to skip: {stage2_skip}')

    # Stage 2
    stage2_results, final_deepframe = run_stage2(
        stage1_results,
        mode=final_cfg['observing_mode'],
        soss_background_model=final_cfg.get('soss_background_file'),
        baseline_ints=final_cfg['baseline_ints'],
        save_results=True,
        force_redo=True,
        space_thresh=final_cfg.get('space_outlier_threshold'),
        time_thresh=final_cfg.get('time_outlier_threshold'),
        remove_components=final_cfg.get('remove_components'),
        pca_components=final_cfg.get('pca_components'),
        soss_timeseries=final_cfg.get('soss_timeseries'),
        soss_timeseries_o2=final_cfg.get('soss_timeseries_o2'),
        oof_method=final_cfg.get('oof_method'),
        output_tag=final_cfg['output_tag'],
        skip_steps=stage2_skip,
        generate_lc=final_cfg.get('generate_lc'),
        soss_inner_mask_width=final_cfg.get('soss_inner_mask_width'),
        soss_outer_mask_width=final_cfg.get('soss_outer_mask_width'),
        nirspec_mask_width=final_cfg.get('nirspec_mask_width'),
        pixel_masks=final_cfg.get('outlier_maps'),
        f277w=final_cfg.get('f277w'),
        do_plot=final_cfg.get('do_plots', False),
        centroids=final_cfg.get('centroids'),
        miri_trace_width=final_cfg.get('miri_trace_width'),
        miri_background_width=final_cfg.get('miri_background_width'),
        miri_background_method=final_cfg.get('miri_background_method'),
        root_dir=root_dir,
        **opt_utils.stage2_kwargs_with_winners(final_cfg)
    )

    # new_stage2.run_stage2 now returns (results, deepframe), not centroids.
    # If no centroids are explicitly provided (or already saved on disk), let new_stage3 trace
    # them directly from the deepframe during Stage 3 extraction.
    # Only consider centroid tables written for this dataset (e.g., not the other NIRSpec
    # detector's when NRS1 and NRS2 share an output directory).
    final_fileroot = utils.get_filename_root_noseg(utils.get_filename_root(stage2_results))
    try:
        this_centroid = opt_utils.resolve_existing_centroids(final_cfg, outdir_s2, outdir_s3,
                                                             fileroot_noseg=final_fileroot)
    except FileNotFoundError:
        fancyprint('No Stage 3 or Stage 2 centroid table found. Stage 3 will trace centroids '
                   'from the deepframe.')
        this_centroid = None

    # Use the deepframe Stage 2 just produced unless one is explicitly set in the config.
    this_deepframe = final_cfg.get('deepframe')
    if this_deepframe in [None, 'None', 'null', '']:
        this_deepframe = final_deepframe

    # ===== OPTIMIZE EXTRACT_WIDTH IF REQUESTED =====
    if cfg.get('optimize_extract_width', False):
        fancyprint(f'\n{'='*60}')
        fancyprint('OPTIMIZING EXTRACT_WIDTH ON FULL DATASET')
        fancyprint(f'Uses same cost function as Phase 1 (spectral scatter)')
        fancyprint(f'{'='*60}\n')

        extract_widths = cfg['extract_width']
        if not isinstance(extract_widths, list):
            extract_widths = [extract_widths]

        extract_costs = []

        # Reopen log files to append extract_width optimization results
        logf = open(f'{outdir_f}/Cost_Summary{name_str}.txt', 'a')
        logs = open(f'{outdir_f}/LightCurve_Scatter{name_str}.txt', 'a')

        for width in extract_widths:
            fancyprint(f'\nTesting extract_width={width}')
            t0 = time.perf_counter()

            # Run Stage 3 with this extract width
            stage3_results = run_stage3(
                stage2_results,
                save_results=True,
                force_redo=True,
                extract_method=final_cfg['extract_method'],
                soss_specprofile=final_cfg.get('soss_specprofile'),
                centroids=this_centroid,
                extract_width=width,
                extract_width_soss2=final_cfg.get('extract_width_soss2'),
                st_teff=final_cfg.get('st_teff'),
                st_logg=final_cfg.get('st_logg'),
                st_met=final_cfg.get('st_met'),
                planet_letter=final_cfg.get('planet_letter'),
                output_tag=final_cfg['output_tag'],
                do_plot=final_cfg.get('do_plots', False),
                deepframe=this_deepframe,
                saturation_rescue=final_cfg.get('saturation_rescue', False),
                mask_do_not_use_pixels=final_cfg.get('mask_do_not_use_pixels', True),
                root_dir=root_dir,
                **final_cfg.get('stage3_kwargs', {})
            )

            # Compute cost using same function as Phase 1
            cost, scatter = cost_function(stage3_results, baseline_ints=baseline_ints,
                                          wave_range=wave_range, w1=w1, w2=w2, obs_mode=obs_mode)

            dt = time.perf_counter() - t0
            extract_costs.append(cost)

            fancyprint(f'extract_width={width}: cost={cost:.12f} ({dt:.1f}s)')

            # Log results to files (same format as Phase 1)
            # Create a temporary config with this extract_width for logging
            log_cfg = current_best.copy()
            log_cfg['extract_width'] = width
            log_line = '\t'.join(str(log_cfg.get(p, '')) for p in param_ranges.keys())
            logf.write(f'{log_line}\t{dt:.1f}\t{cost:.12f}\n')
            logf.flush()

            scatter_line = ' '.join(f'{x:.10g}' for x in scatter)
            logs.write(f'{scatter_line}\n')
            logs.flush()

        # Select best extract_width (non-finite costs cannot win)
        best_width_idx = opt_utils.select_best_trial(extract_costs, 'extract_width')
        best_extract_width = extract_widths[best_width_idx]
        best_extract_cost = extract_costs[best_width_idx]

        # Close log files
        logf.close()
        logs.close()

        fancyprint(f'\n*** Best extract_width={best_extract_width} with cost={best_extract_cost:.6f} ***\n')

        # Update final config and run one more time with best width
        final_cfg['extract_width'] = best_extract_width
        current_best['extract_width'] = best_extract_width

        # Regenerate plot with extract_width optimization results
        fancyprint('\n=== Updating optimization plot with extract_width results ===')
        plot_cost(name_str, outdir=outdir_f)
        # Final Stage 3 with optimal width
        stage3_results = run_stage3(
            stage2_results,
            save_results=True,
            force_redo=True,
            extract_method=final_cfg['extract_method'],
            soss_specprofile=final_cfg.get('soss_specprofile'),
            centroids=this_centroid,
            extract_width=best_extract_width,
            extract_width_soss2=final_cfg.get('extract_width_soss2'),
            st_teff=final_cfg.get('st_teff'),
            st_logg=final_cfg.get('st_logg'),
            st_met=final_cfg.get('st_met'),
            planet_letter=final_cfg.get('planet_letter'),
            output_tag=final_cfg['output_tag'],
            do_plot=final_cfg.get('do_plots', False),
            deepframe=this_deepframe,
            saturation_rescue=final_cfg.get('saturation_rescue', False),
            mask_do_not_use_pixels=final_cfg.get('mask_do_not_use_pixels', True),
            root_dir=root_dir,
            **final_cfg.get('stage3_kwargs', {})
        )
    else:
        # No optimization, just run Stage 3 once with fixed width
        extract_width_to_use = final_cfg.get('extract_width')
        if isinstance(extract_width_to_use, list):
            extract_width_to_use = extract_width_to_use[0]
        fancyprint(f'\nUsing fixed extract_width={extract_width_to_use}')

        stage3_results = run_stage3(
            stage2_results,
            save_results=True,
            force_redo=True,
            extract_method=final_cfg['extract_method'],
            soss_specprofile=final_cfg.get('soss_specprofile'),
            centroids=this_centroid,
            extract_width=extract_width_to_use,
            extract_width_soss2=final_cfg.get('extract_width_soss2'),
            st_teff=final_cfg.get('st_teff'),
            st_logg=final_cfg.get('st_logg'),
            st_met=final_cfg.get('st_met'),
            planet_letter=final_cfg.get('planet_letter'),
            output_tag=final_cfg['output_tag'],
            do_plot=final_cfg.get('do_plots', False),
            deepframe=this_deepframe,
            saturation_rescue=final_cfg.get('saturation_rescue', False),
            mask_do_not_use_pixels=final_cfg.get('mask_do_not_use_pixels', True),
            root_dir=root_dir,
            **final_cfg.get('stage3_kwargs', {})
        )

    #  Diagnostics
    make_lightcurve_plots(stage3_results, name_str=name_str, baseline_ints=baseline_ints,
                          obs=obs_mode, outdir=outdir_f)

    #  scatter plot of the final Stage 3 spectrum. Phase 1 costs are computed on a single segment
    #  (and possibly a single group), so they are not comparable with the full-dataset Stage 3
    #  products and must not be plotted on the final wavelength axis.
    _, final_scatter = cost_function(stage3_results, baseline_ints=baseline_ints,
                                     wave_range=wave_range, w1=w1, w2=w2, obs_mode=obs_mode)
    outfile = os.path.join(outdir_f, f'LightCurve_Scatter_Final{name_str}.txt')
    with open(outfile, 'w') as f:
        f.write(' '.join(f'{x:.10g}' for x in final_scatter) + '\n')
    specfile = opt_utils.find_stage3_spectrum_file(final_cfg['extract_method'], outdir_s3)
    plot_file_path = os.path.join(outdir_f, f'Scatter_Plot{name_str}.png')
    plot_scatter(txtfile=outfile, rows=[0], wave_range=None, spectrum_files=[specfile],
                 save_path=plot_file_path)

    # ===== ARCHIVE TO LONG-TERM STORAGE =====
    archive_dest = cfg.get('archive_to_longterm_storage')
    if archive_dest and archive_dest not in [None, 'None', 'null', '']:
        fancyprint(f'\n{'='*60}')
        fancyprint('ARCHIVING TO LONG-TERM STORAGE')
        fancyprint(f'{'='*60}\n')

        # Full output directory path (pipeline_outputs_directory + output_tag)
        full_output_dir = full_outdir

        # Get input directory
        input_dir = cfg['input_dir']

        # Trust that archive_dest exists (don't try to create parent dirs like /cds2)
        # User must ensure the archive destination directory exists before running

        # Archive input directory
        if os.path.exists(input_dir):
            input_basename = os.path.basename(input_dir.rstrip('/'))
            archive_input = os.path.join(archive_dest, input_basename)
            try:
                fancyprint(f'Moving input data:')
                fancyprint(f'  From: {input_dir}')
                fancyprint(f'  To:   {archive_input}')
                shutil.move(input_dir, archive_input)
                fancyprint('  ✓ Input data archived successfully')
            except Exception as e:
                fancyprint(f'  ✗ Failed to archive input data: {e}', msg_type='WARNING')
        else:
            fancyprint(f'Input directory not found: {input_dir}', msg_type='WARNING')

        # Archive output directory
        if os.path.exists(full_output_dir):
            output_basename = os.path.basename(full_output_dir.rstrip('/'))
            archive_output = os.path.join(archive_dest, output_basename)
            try:
                fancyprint(f'\nMoving pipeline outputs:')
                fancyprint(f'  From: {full_output_dir}')
                fancyprint(f'  To:   {archive_output}')
                shutil.move(full_output_dir, archive_output)
                fancyprint('  ✓ Pipeline outputs archived successfully')
            except Exception as e:
                fancyprint(f'  ✗ Failed to archive outputs: {e}', msg_type='WARNING')
        else:
            fancyprint(f'Output directory not found: {full_output_dir}', msg_type='WARNING')

        fancyprint(f'\n{'='*60}')
        fancyprint('ARCHIVING COMPLETE')
        fancyprint(f'{'='*60}\n')

    #  timing
    t1 = time.perf_counter() - t0_total
    h, m = divmod(int(t1), 3600)
    m, s = divmod(m, 60)
    fancyprint(f'\n{'='*60}')
    fancyprint(f'TOTAL RUNTIME: {h}h {m:02d}min {s:02d}s')
    fancyprint(f'OPTIMAL PARAMETERS: {current_best}')
    fancyprint(f'{'='*60}\n')


# ===== Do Stuff =====
if __name__ == '__main__':
    save_config(config)
    run_optimizer(config)
    fancyprint('Done')
