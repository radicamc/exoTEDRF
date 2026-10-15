#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 11:12 2022

@author: MCR

Script to run JWST DMS with custom reduction steps.
"""

from datetime import datetime, UTC
import os
import shutil
import sys

from exotedrf.utils import fancyprint, parse_config, unpack_input_dir, verify_path

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


# ===== Functions =====
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
    root_dir += '/config_files'
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
    runtime = datetime.now(UTC).isoformat(sep=' ', timespec='minutes')
    f.write('\nRun at {}.'.format(runtime))
    f.close()

    return


def unpack_files(cfg):
    """Unpack input files.
    """

    # Unpack all files in the input directory.
    input_files = unpack_input_dir(cfg['input_dir'], mode=cfg['observing_mode'],
                                   filetag=cfg['input_filetag'],
                                   filter_detector=cfg['filter_detector'])
    if cfg['observing_mode'] == 'MIRI/LRS':
        thisfilter = ''
    else:
        thisfilter = cfg['filter_detector']
    fancyprint('Identified {0} {1} {2} observation segment(s)'
               .format(len(input_files), thisfilter, cfg['observing_mode']))
    for file in input_files:
        fancyprint(' ' + file)

    return input_files


def run_dms(cfg, input_files):
    """Run the DMS.
    """

    # ===== Run Stage 1 =====
    if 1 in cfg['run_stages']:
        # Determine which steps to run and which to skip.
        steps = ['DQInitStep', 'INLCorrStep', 'EmiCorrStep', 'SuperBiasStep', 'RefPixStep',
                 'DarkCurrentStep', 'OneOverFStep_grp', 'LinearityStep', 'JumpStep',
                 'RampFitStep', 'GainScaleStep']
        stage1_skip = []
        for step in steps:
            if cfg[step] == 'skip':
                if step == 'OneOverFStep_grp':
                    stage1_skip.append('OneOverFStep')
                else:
                    stage1_skip.append(step)
        # Run stage 1.
        stage1_results = run_stage1(
            input_files,
            mode=cfg['observing_mode'],
            soss_background_model=cfg['soss_background_file'],
            baseline_ints=cfg['baseline_ints'],
            oof_method=cfg['oof_method'],
            superbias_method=cfg['superbias_method'],
            soss_timeseries=cfg['soss_timeseries'],
            soss_timeseries_o2=cfg['soss_timeseries_o2'],
            save_results=cfg['save_results'],
            pixel_masks=cfg['outlier_maps'],
            force_redo=cfg['force_redo'],
            flag_up_ramp=cfg['flag_up_ramp'],
            rejection_threshold=cfg['jump_threshold'],
            flag_in_time=cfg['flag_in_time'],
            time_rejection_threshold=cfg['time_jump_threshold'],
            output_tag=cfg['output_tag'],
            skip_steps=stage1_skip,
            do_plot=cfg['do_plots'],
            soss_inner_mask_width=cfg['soss_inner_mask_width'],
            soss_outer_mask_width=cfg['soss_outer_mask_width'],
            nirspec_mask_width=cfg['nirspec_mask_width'],
            centroids=cfg['centroids'],
            hot_pixel_map=cfg['hot_pixel_map'],
            miri_drop_groups=cfg['miri_drop_groups'],
            saturation_threshold=cfg['saturation_threshold'],
            f277w=cfg['f277w'],
            inl_amplitude_file=cfg['inl_amplitudes_file'],
            inl_periods=cfg['inl_periods'],
            root_dir=cfg['root_dir'],
            **cfg['stage1_kwargs']
                                    )
    else:
        stage1_results = input_files

    # ===== Run Stage 2 =====
    if 2 in cfg['run_stages']:
        # Determine which steps to run and which to skip.
        steps = ['AssignWCSStep', 'FlatFieldStep', 'OneOverFStep_int', 'BackgroundStep',
                 'BadPixStep', 'PCAReconstructStep']
        stage2_skip = []
        for step in steps:
            if cfg[step] == 'skip':
                if step == 'OneOverFStep_int':
                    stage2_skip.append('OneOverFStep')
                else:
                    stage2_skip.append(step)
        # Run stage 2.
        stage2_results = run_stage2(
            stage1_results,
            mode=cfg['observing_mode'],
            soss_background_model=cfg['soss_background_file'],
            baseline_ints=cfg['baseline_ints'],
            save_results=cfg['save_results'],
            force_redo=cfg['force_redo'],
            space_thresh=cfg['space_outlier_threshold'],
            time_thresh=cfg['time_outlier_threshold'],
            remove_components=cfg['remove_components'],
            pca_components=cfg['pca_components'],
            soss_timeseries=cfg['soss_timeseries'],
            soss_timeseries_o2=cfg['soss_timeseries_o2'],
            oof_method=cfg['oof_method'],
            output_tag=cfg['output_tag'],
            skip_steps=stage2_skip,
            generate_lc=cfg['generate_lc'],
            soss_inner_mask_width=cfg['soss_inner_mask_width'],
            soss_outer_mask_width=cfg['soss_outer_mask_width'],
            nirspec_mask_width=cfg['nirspec_mask_width'],
            pixel_masks=cfg['outlier_maps'],
            f277w=cfg['f277w'],
            do_plot=cfg['do_plots'],
            centroids=cfg['centroids'],
            miri_trace_width=cfg['miri_trace_width'],
            miri_background_width=cfg['miri_background_width'],
            miri_background_method=cfg['miri_background_method'],
            root_dir=cfg['root_dir'],
            **cfg['stage2_kwargs']
                                    )
        stage2_results, deepframe = stage2_results
    else:
        stage2_results = input_files
        deepframe = cfg['deepframe']

    # ===== Run Stage 3 =====
    if 3 in cfg['run_stages']:
        # If a deepframe is passed in the config file, it takes precedence over anything
        # calculated in Stage 2.
        if cfg['deepframe'] is None:
            this_deepframe = deepframe
        else:
            this_deepframe = cfg['deepframe']
        _ = run_stage3(
            stage2_results,
            save_results=cfg['save_results'],
            force_redo=cfg['force_redo'],
            extract_method=cfg['extract_method'],
            soss_specprofile=cfg['soss_specprofile'],
            centroids=cfg['centroids'],
            extract_width=cfg['extract_width'],
            extract_width_soss2=cfg['extract_width_soss2'],
            st_teff=cfg['st_teff'],
            st_logg=cfg['st_logg'],
            st_met=cfg['st_met'],
            planet_letter=cfg['planet_letter'],
            output_tag=cfg['output_tag'],
            do_plot=cfg['do_plots'],
            deepframe=this_deepframe,
            root_dir=cfg['root_dir'],
            **cfg['stage3_kwargs']
                       )

    return


# ===== Do Stuff =====
if __name__ == '__main__':
    save_config(config)
    all_files = unpack_files(config)
    run_dms(config, all_files)
    fancyprint('Done')
