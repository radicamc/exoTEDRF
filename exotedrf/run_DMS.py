#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 11:12 2022

@author: MCR

Script to run JWST DMS with custom reduction steps.
"""

from datetime import datetime
import os
import shutil
import sys

from exotedrf.utils import fancyprint, parse_config, unpack_input_dir, \
    verify_path

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

# Save a copy of the config file.
if config['output_tag'] != '':
    output_tag = '_' + config['output_tag']
else:
    output_tag = config['output_tag']
root_dir = 'pipeline_outputs_directory' + output_tag
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
time = datetime.utcnow().isoformat(sep=' ', timespec='minutes')
f.write('\nRun at {}.'.format(time))
f.close()

# Unpack all files in the input directory.
input_files = unpack_input_dir(config['input_dir'],
                               mode=config['observing_mode'],
                               filetag=config['input_filetag'],
                               filter_detector=config['filter_detector'])
fancyprint('Identified {0} {1} {2} observation '
           'segment(s)'.format(len(input_files), config['filter_detector'],
                               config['observing_mode']))
for file in input_files:
    fancyprint(' ' + file)

# ===== Run Stage 1 =====
if 1 in config['run_stages']:
    # Determine which steps to run and which to skip.
    steps = ['DQInitStep', 'SaturationStep', 'SuperBiasStep', 'RefPixStep',
             'DarkCurrentStep', 'OneOverFStep_grp', 'LinearityStep',
             'JumpStep', 'RampFitStep', 'GainScaleStep']
    stage1_skip = []
    for step in steps:
        if config[step] == 'skip':
            if step == 'OneOverFStep_grp':
                stage1_skip.append('OneOverFStep')
            else:
                stage1_skip.append(step)
    # Run stage 1.
    stage1_results = run_stage1(input_files, mode=config['observing_mode'],
                                soss_background_model=config['soss_background_file'],
                                baseline_ints=config['baseline_ints'],
                                oof_method=config['oof_method'],
                                superbias_method=config['superbias_method'],
                                soss_timeseries=config['soss_timeseries'],
                                soss_timeseries_o2=config['soss_timeseries_o2'],
                                save_results=config['save_results'],
                                pixel_masks=config['outlier_maps'],
                                force_redo=config['force_redo'],
                                flag_up_ramp=config['flag_up_ramp'],
                                rejection_threshold=config['jump_threshold'],
                                flag_in_time=config['flag_in_time'],
                                time_rejection_threshold=config['time_jump_threshold'],
                                output_tag=config['output_tag'],
                                skip_steps=stage1_skip,
                                do_plot=config['do_plots'],
                                soss_inner_mask_width=config['soss_inner_mask_width'],
                                soss_outer_mask_width=config['soss_outer_mask_width'],
                                nirspec_mask_width=config['nirspec_mask_width'],
                                centroids=config['centroids'],
                                hot_pixel_map=config['hot_pixel_map'],
                                **config['stage1_kwargs'])
else:
    stage1_results = input_files

# ===== Run Stage 2 =====
if 2 in config['run_stages']:
    # Determine which steps to run and which to skip.
    steps = ['AssignWCSStep', 'Extract2DStep', 'SourceTypeStep',
             'WaveCorrStep', 'FlatFieldStep', 'OneOverFStep_int',
             'BackgroundStep', 'TracingStep', 'BadPixStep',
             'PCAReconstructStep']
    stage2_skip = []
    for step in steps:
        if config[step] == 'skip':
            if step == 'OneOverFStep_int':
                stage2_skip.append('OneOverFStep')
            else:
                stage2_skip.append(step)
    # Run stage 2.
    stage2_results = run_stage2(stage1_results, mode=config['observing_mode'],
                                soss_background_model=config['soss_background_file'],
                                baseline_ints=config['baseline_ints'],
                                save_results=config['save_results'],
                                force_redo=config['force_redo'],
                                space_thresh=config['space_outlier_threshold'],
                                time_thresh=config['time_outlier_threshold'],
                                remove_components=config['remove_components'],
                                pca_components=config['pca_components'],
                                soss_timeseries=config['soss_timeseries'],
                                soss_timeseries_o2=config['soss_timeseries_o2'],
                                oof_method=config['oof_method'],
                                output_tag=config['output_tag'],
                                smoothing_scale=config['smoothing_scale'],
                                skip_steps=stage2_skip,
                                generate_lc=config['generate_lc'],
                                soss_inner_mask_width=config['soss_inner_mask_width'],
                                soss_outer_mask_width=config['soss_outer_mask_width'],
                                nirspec_mask_width=config['nirspec_mask_width'],
                                pixel_masks=config['outlier_maps'],
                                generate_order0_mask=config['generate_order0_mask'],
                                f277w=config['f277w'],
                                do_plot=config['do_plots'],
                                centroids=config['centroids'],
                                **config['stage2_kwargs'])
else:
    stage2_results = input_files

# ===== Run Stage 3 =====
if 3 in config['run_stages']:
    stage3_results = run_stage3(stage2_results,
                                save_results=config['save_results'],
                                force_redo=config['force_redo'],
                                extract_method=config['extract_method'],
                                soss_specprofile=config['soss_specprofile'],
                                centroids=config['centroids'],
                                extract_width=config['extract_width'],
                                st_teff=config['st_teff'],
                                st_logg=config['st_logg'],
                                st_met=config['st_met'],
                                planet_letter=config['planet_letter'],
                                output_tag=config['output_tag'],
                                do_plot=config['do_plots'],
                                **config['stage3_kwargs'])

fancyprint('Done')
