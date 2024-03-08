#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 11:12 2022

@author: MCR

Script to run JWST DMS with custom reduction steps.
"""

from datetime import datetime
import numpy as np
import os
import shutil
import sys

from exotedrf.stage1 import run_stage1
from exotedrf.stage2 import run_stage2
from exotedrf.stage3 import run_stage3
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
                               filetag=config['input_filetag'],
                               exposure_type=config['exposure_type'])
fancyprint('Identified {0} {1} exposure '
           'segments'.format(len(input_files), config['exposure_type']))
for file in input_files:
    fancyprint(' ' + file)

# Open some of the input files.
background_model = np.load(config['background_file'])
if config['timeseries'] is not None:
    config['timeseries'] = np.load(config['timeseries'])
if config['timeseries_o2'] is not None:
    config['timeseries_o2'] = np.load(config['timeseries_o2'])
if config['f277w'] is not None:
    config['f277w'] = np.load(config['f277w'])

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
    stage1_results = run_stage1(input_files, background_model=background_model,
                                baseline_ints=config['baseline_ints'],
                                oof_method=config['oof_method'],
                                inner_mask_width=config['inner_mask_width'],
                                outer_mask_width=config['outer_mask_width'],
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
                                timeseries=config['timeseries'],
                                timeseries_o2=config['timeseries_o2'],
                                centroids=config['centroids'],
                                **config['stage1_kwargs'])
else:
    stage1_results = input_files

# ===== Run Stage 2 =====
if 2 in config['run_stages']:
    # Determine which steps to run and which to skip.
    steps = ['AssignWCSStep', 'SourceTypeStep', 'FlatFieldStep',
             'OneOverFStep_int', 'BackgroundStep', 'TracingStep',
             'BadPixStep']
    stage2_skip = []
    for step in steps:
        if config[step] == 'skip':
            if step == 'OneOverFStep_int':
                stage2_skip.append('OneOverFStep')
            else:
                stage2_skip.append(step)
    # Run stage 2.
    stage2_results = run_stage2(stage1_results,
                                background_model=background_model,
                                baseline_ints=config['baseline_ints'],
                                save_results=config['save_results'],
                                force_redo=config['force_redo'],
                                space_thresh=config['space_outlier_threshold'],
                                time_thresh=config['time_outlier_threshold'],
                                calculate_stability=config['calculate_stability'],
                                pca_components=config['pca_components'],
                                timeseries=config['timeseries'],
                                timeseries_o2=config['timeseries_o2'],
                                oof_method=config['oof_method'],
                                output_tag=config['output_tag'],
                                smoothing_scale=config['smoothing_scale'],
                                skip_steps=stage2_skip,
                                generate_lc=config['generate_lc'],
                                inner_mask_width=config['inner_mask_width'],
                                outer_mask_width=config['outer_mask_width'],
                                centroids=config['centroids'],
                                pixel_masks=config['outlier_maps'],
                                generate_order0_mask=config['generate_order0_mask'],
                                f277w=config['f277w'],
                                do_plot=config['do_plots'],
                                **config['stage2_kwargs'])
else:
    stage2_results = input_files

# ===== Run Stage 3 =====
if 3 in config['run_stages']:
    stage3_results = run_stage3(stage2_results,
                                save_results=config['save_results'],
                                force_redo=config['force_redo'],
                                extract_method=config['extract_method'],
                                soss_width=config['soss_width'],
                                specprofile=config['specprofile'],
                                centroids=config['centroids'],
                                st_teff=config['st_teff'],
                                st_logg=config['st_logg'],
                                st_met=config['st_met'],
                                planet_letter=config['planet_letter'],
                                output_tag=config['output_tag'],
                                do_plot=config['do_plots'],
                                **config['stage3_kwargs'])

fancyprint('Done')
