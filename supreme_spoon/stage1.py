#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thurs Jul 21 17:30 2022

@author: MCR

Custom JWST DMS pipeline steps for Stage 1 (detector level processing).
"""

from astropy.io import fits
import bottleneck as bn
import glob
import numpy as np
import os
from scipy.ndimage import median_filter
from scipy.signal import medfilt
from tqdm import tqdm
import warnings

from jwst import datamodels
from jwst.pipeline import calwebb_detector1

from supreme_spoon.stage2 import BackgroundStep
from supreme_spoon import utils, plotting
from supreme_spoon.utils import fancyprint


class GroupScaleStep:
    """Wrapper around default calwebb_detector1 Group Scale Correction step.
    """

    def __init__(self, input_data, output_dir):
        """Step initializer.
        """

        self.tag = 'groupscalestep.fits'
        self.output_dir = output_dir
        self.datafiles = utils.sort_datamodels(input_data)
        self.fileroots = utils.get_filename_root(self.datafiles)

    def run(self, save_results=True, force_redo=False, **kwargs):
        """Method to run the step.
        """

        results = []
        all_files = glob.glob(self.output_dir + '*')
        for i, segment in enumerate(self.datafiles):
            # If an output file for this segment already exists, skip the step.
            expected_file = self.output_dir + self.fileroots[i] + self.tag
            if expected_file in all_files and force_redo is False:
                fancyprint('File {} already exists.'.format(expected_file))
                fancyprint('Skipping Group Scale Step.')
                res = expected_file
            # If no output files are detected, run the step.
            else:
                step = calwebb_detector1.group_scale_step.GroupScaleStep()
                res = step.call(segment, output_dir=self.output_dir,
                                save_results=save_results, **kwargs)
                # Verify that filename is correct.
                if save_results is True:
                    current_name = self.output_dir + res.meta.filename
                    if expected_file != current_name:
                        res.close()
                        os.rename(current_name, expected_file)
                        res = datamodels.open(expected_file)
            results.append(res)

        return results


class DQInitStep:
    """Wrapper around default calwebb_detector1 Data Quality Initialization
    step.
    """

    def __init__(self, input_data, output_dir, deepframe=None):
        """Step initializer.
        """

        self.tag = 'dqinitstep.fits'
        self.output_dir = output_dir
        self.datafiles = utils.sort_datamodels(input_data)
        self.fileroots = utils.get_filename_root(self.datafiles)
        self.deepframe = deepframe

    def run(self, save_results=True, force_redo=False, **kwargs):
        """Method to run the step.
        """

        results = []
        all_files = glob.glob(self.output_dir + '*')
        hot_pix = None
        for i, segment in enumerate(self.datafiles):
            # If an output file for this segment already exists, skip the step.
            expected_file = self.output_dir + self.fileroots[i] + self.tag
            if expected_file in all_files and force_redo is False:
                fancyprint('File {} already exists.'.format(expected_file))
                fancyprint('Skipping Data Quality Initialization Step.')
                res = expected_file
            # If no output files are detected, run the step.
            else:
                step = calwebb_detector1.dq_init_step.DQInitStep()
                res = step.call(segment, output_dir=self.output_dir,
                                save_results=save_results, **kwargs)
                # If a deep frame is passed, use it to search for and flag
                # additional hot pixels not in the default map.
                if self.deepframe is not None:
                    res, hot_pix = flag_hot_pixels(res,
                                                   deepframe=self.deepframe,
                                                   hot_pix=hot_pix)
                    # Overwite the previous edition.
                    res.save(expected_file)
                # Verify that filename is correct.
                if save_results is True:
                    current_name = self.output_dir + res.meta.filename
                    if expected_file != current_name:
                        res.close()
                        os.rename(current_name, expected_file)
                        res = datamodels.open(expected_file)
            results.append(res)

        return results


class SaturationStep:
    """Wrapper around default calwebb_detector1 Saturation Detection step.
    """

    def __init__(self, input_data, output_dir):
        """Step initializer.
        """

        self.tag = 'saturationstep.fits'
        self.output_dir = output_dir
        self.datafiles = utils.sort_datamodels(input_data)
        self.fileroots = utils.get_filename_root(self.datafiles)

    def run(self, save_results=True, force_redo=False, **kwargs):
        """Method to run the step.
        """

        results = []
        all_files = glob.glob(self.output_dir + '*')
        for i, segment in enumerate(self.datafiles):
            # If an output file for this segment already exists, skip the step.
            expected_file = self.output_dir + self.fileroots[i] + self.tag
            if expected_file in all_files and force_redo is False:
                fancyprint('File {} already exists.'.format(expected_file))
                fancyprint('Skipping Saturation Detection Step.')
                res = expected_file
            # If no output files are detected, run the step.
            else:
                step = calwebb_detector1.saturation_step.SaturationStep()
                res = step.call(segment, output_dir=self.output_dir,
                                save_results=save_results, **kwargs)
                # Verify that filename is correct.
                if save_results is True:
                    current_name = self.output_dir + res.meta.filename
                    if expected_file != current_name:
                        res.close()
                        os.rename(current_name, expected_file)
                        res = datamodels.open(expected_file)
            results.append(res)

        return results


class SuperBiasStep:
    """Wrapper around default calwebb_detector1 Super Bias Subtraction step.
    """

    def __init__(self, input_data, output_dir):
        """Step initializer.
        """

        self.tag = 'superbiasstep.fits'
        self.output_dir = output_dir
        self.datafiles = utils.sort_datamodels(input_data)
        self.fileroots = utils.get_filename_root(self.datafiles)
        self.instrument = utils.get_instrument_name(self.datafiles[0])

    def run(self, save_results=True, force_redo=False, do_plot=False,
            show_plot=False, **kwargs):
        """Method to run the step.
        """

        results = []
        all_files = glob.glob(self.output_dir + '*')
        for i, segment in enumerate(self.datafiles):
            # If an output file for this segment already exists, skip the step.
            expected_file = self.output_dir + self.fileroots[i] + self.tag
            if expected_file in all_files and force_redo is False:
                fancyprint('File {} already exists.'.format(expected_file))
                fancyprint('Skipping Superbias Subtraction Step.')
                res = expected_file
                do_plot = False
            # If no output files are detected, run the step.
            else:
                step = calwebb_detector1.superbias_step.SuperBiasStep()
                res = step.call(segment, output_dir=self.output_dir,
                                save_results=save_results, **kwargs)
                # Verify that filename is correct.
                if save_results is True:
                    current_name = self.output_dir + res.meta.filename
                    if expected_file != current_name:
                        res.close()
                        os.rename(current_name, expected_file)
                        res = datamodels.open(expected_file)
            results.append(res)
        # Do step plot if requested.
        if do_plot is True:
            if save_results is True:
                plot_file = self.output_dir + self.tag.replace('fits', 'pdf')
                if self.instrument == 'NIRSPEC':
                    det = utils.get_detector_name(self.datafiles[0])
                    plot_file = plot_file.replace('.pdf',
                                                  '_{}.pdf'.format(det))
            else:
                plot_file = None
            plotting.make_superbias_plot(results, outfile=plot_file,
                                         show_plot=show_plot)

        return results


class OneOverFStep:
    """Wrapper around custom 1/f Correction Step.
    """

    def __init__(self, input_data, output_dir, baseline_ints=None,
                 smoothed_wlc=None, pixel_masks=None, background=None):
        """Step initializer.
        """

        self.tag = 'oneoverfstep.fits'
        self.output_dir = output_dir
        self.baseline_ints = baseline_ints
        self.smoothed_wlc = smoothed_wlc
        self.pixel_masks = pixel_masks
        self.background = background
        self.datafiles = utils.sort_datamodels(input_data)
        self.fileroots = utils.get_filename_root(self.datafiles)
        self.instrument = utils.get_instrument_name(self.datafiles[0])

    def run(self, save_results=True, force_redo=False, do_plot=False,
            show_plot=False, **kwargs):
        """Method to run the step.
        """

        all_files = glob.glob(self.output_dir + '*')
        do_step = 1
        results = []
        for i in range(len(self.datafiles)):
            # If an output file for this segment already exists, skip the step.
            expected_file = self.output_dir + self.fileroots[i] + self.tag
            if expected_file not in all_files:
                do_step = 0
                break
            else:
                results.append(expected_file)
        if do_step == 1 and force_redo is False:
            fancyprint('Output files already exist.')
            fancyprint('Skipping 1/f Correction Step.')
        # If no output files are detected, run the step.
        else:
            if self.instrument == 'NIRISS':
                results = oneoverfstep_soss(self.datafiles,
                                            baseline_ints=self.baseline_ints,
                                            background=self.background,
                                            smoothed_wlc=self.smoothed_wlc,
                                            output_dir=self.output_dir,
                                            save_results=save_results,
                                            pixel_masks=self.pixel_masks,
                                            fileroots=self.fileroots,
                                            **kwargs)
            else:
                results = oneoverfstep_nirspec(self.datafiles,
                                               output_dir=self.output_dir,
                                               save_results=save_results,
                                               pixel_masks=self.pixel_masks,
                                               fileroots=self.fileroots,
                                               **kwargs)
            # Do step plots if requested.
            if do_plot is True:
                if save_results is True:
                    plot_file1 = self.output_dir + self.tag.replace('.fits',
                                                                    '_1.pdf')
                    plot_file2 = self.output_dir + self.tag.replace('.fits',
                                                                    '_2.pdf')
                    if self.instrument == 'NIRSPEC':
                        det = utils.get_detector_name(self.datafiles[0])
                        plot_file1 = plot_file1.replace('_1.pdf', '_1_{}.pdf'.format(det))
                        plot_file2 = plot_file2.replace('_2.pdf', '_2_{}.pdf'.format(det))
                else:
                    plot_file1, plot_file2 = None, None
                plotting.make_oneoverf_plot(results,
                                            timeseries=self.smoothed_wlc,
                                            baseline_ints=self.baseline_ints,
                                            outfile=plot_file1,
                                            show_plot=show_plot)
                # Get the correct frame times for the psd.
                if self.instrument == 'NIRSPEC':
                    tframe = 0.902
                else:
                    tframe = 5.494
                plotting.make_oneoverf_psd(results, self.datafiles,
                                           timeseries=self.smoothed_wlc,
                                           baseline_ints=self.baseline_ints,
                                           pixel_masks=self.pixel_masks,
                                           outfile=plot_file2,
                                           show_plot=show_plot, tframe=tframe)

        return results


class LinearityStep:
    """Wrapper around default calwebb_detector1 Linearity Correction step.
    """

    def __init__(self, input_data, output_dir):
        """Step initializer.
        """

        self.tag = 'linearitystep.fits'
        self.output_dir = output_dir
        self.datafiles = utils.sort_datamodels(input_data)
        self.fileroots = utils.get_filename_root(self.datafiles)
        self.instrument = utils.get_instrument_name(self.datafiles[0])

    def run(self, save_results=True, force_redo=False, do_plot=False,
            show_plot=False, **kwargs):
        """Method to run the step.
        """

        results = []
        all_files = glob.glob(self.output_dir + '*')
        for i, segment in enumerate(self.datafiles):
            # If an output file for this segment already exists, skip the step.
            expected_file = self.output_dir + self.fileroots[i] + self.tag
            if expected_file in all_files and force_redo is False:
                fancyprint('File {} already exists.'.format(expected_file))
                fancyprint('Skipping Linearity Correction Step.')
                res = expected_file
                do_plot = False
            # If no output files are detected, run the step.
            else:
                step = calwebb_detector1.linearity_step.LinearityStep()
                res = step.call(segment, output_dir=self.output_dir,
                                save_results=save_results, **kwargs)
                # Verify that filename is correct.
                if save_results is True:
                    current_name = self.output_dir + res.meta.filename
                    if expected_file != current_name:
                        res.close()
                        os.rename(current_name, expected_file)
                        res = datamodels.open(expected_file)
            results.append(res)
        # Do step plot if requested.
        if do_plot is True:
            if save_results is True:
                plot_file = self.output_dir + self.tag.replace('fits',
                                                               'pdf')
                if self.instrument == 'NIRSPEC':
                    det = utils.get_detector_name(self.datafiles[0])
                    plot_file = plot_file.replace('.pdf',
                                                  '_{}.pdf'.format(det))
            else:
                plot_file = None
            plotting.make_linearity_plot(results, self.datafiles,
                                         outfile=plot_file,
                                         show_plot=show_plot)

        return results


class JumpStep:
    """Wrapper around default calwebb_detector1 Jump Detection step with some
    custom modifications.
    """

    def __init__(self, input_data, output_dir):
        """Step initializer.
        """

        self.tag = 'jump.fits'
        self.output_dir = output_dir
        self.datafiles = utils.sort_datamodels(input_data)
        self.fileroots = utils.get_filename_root(self.datafiles)
        self.instrument = utils.get_instrument_name(self.datafiles[0])

    def run(self, save_results=True, force_redo=False, rejection_threshold=15,
            flag_in_time=False, time_rejection_threshold=10, time_window=10,
            do_plot=False, show_plot=False, **kwargs):
        """Method to run the step.
        """

        results = []
        all_files = glob.glob(self.output_dir + '*')
        for i, segment in enumerate(self.datafiles):
            # If an output file for this segment already exists, skip the step.
            expected_file = self.output_dir + self.fileroots[i] + self.tag
            if expected_file in all_files and force_redo is False:
                fancyprint('File {} already exists.'.format(expected_file))
                fancyprint('Skipping Jump Detection Step.')
                results.append(datamodels.open(expected_file))
                do_plot = False
            # If no output files are detected, proceed.
            else:
                # Get number of groups in the observation - ngroup=2 must be
                # treated in a special way as the default pipeline JumpStep
                # will fail.
                testfile = datamodels.open(self.datafiles[0])
                ngroups = testfile.meta.exposure.ngroups
                testfile.close()
                # For ngroup > 2, default JumpStep can be used.
                if ngroups > 2:
                    step = calwebb_detector1.jump_step.JumpStep()
                    res = step.call(segment, output_dir=self.output_dir,
                                    save_results=save_results,
                                    rejection_threshold=rejection_threshold,
                                    maximum_cores='quarter', **kwargs)
                # Time domain jump step must be run for ngroup=2.
                else:
                    res = segment
                    flag_in_time = True
                # Do time-domain flagging.
                if flag_in_time is True:
                    res = jumpstep_in_time(res, window=time_window,
                                           thresh=time_rejection_threshold,
                                           fileroot=self.fileroots[i],
                                           save_results=save_results,
                                           output_dir=self.output_dir)
                # Verify that filename is correct.
                if save_results is True:
                    current_name = self.output_dir + res.meta.filename
                    if expected_file != current_name:
                        res.close()
                        os.rename(current_name, expected_file)
                        res = datamodels.open(expected_file)
                results.append(res)

        # Do step plot if requested.
        if do_plot is True:
            if save_results is True:
                plot_file = self.output_dir + self.tag.replace('fits',
                                                               'pdf')
                if self.instrument == 'NIRSPEC':
                    det = utils.get_detector_name(self.datafiles[0])
                    plot_file = plot_file.replace('.pdf',
                                                  '_{}.pdf'.format(det))
            else:
                plot_file = None
            plotting.make_jump_location_plot(results, outfile=plot_file,
                                             show_plot=show_plot)

        return results


class RampFitStep:
    """Wrapper around default calwebb_detector1 Ramp Fit step.
    """

    def __init__(self, input_data, output_dir):
        """Step initializer.
        """

        self.tag = 'rampfitstep.fits'
        self.output_dir = output_dir
        self.datafiles = utils.sort_datamodels(input_data)
        self.fileroots = utils.get_filename_root(self.datafiles)

    def run(self, save_results=True, force_redo=False, **kwargs):
        """Method to run the step.
        """

        results = []
        all_files = glob.glob(self.output_dir + '*')
        for i, segment in enumerate(self.datafiles):
            # If an output file for this segment already exists, skip the step.
            expected_file = self.output_dir + self.fileroots[i] + self.tag
            if expected_file in all_files and force_redo is False:
                fancyprint('File {} already exists.'.format(expected_file))
                fancyprint('Skipping Ramp Fit Step.')
                res = expected_file
            # If no output files are detected, run the step.
            else:
                step = calwebb_detector1.ramp_fit_step.RampFitStep()
                res = step.call(segment, output_dir=self.output_dir,
                                save_results=save_results,
                                maximum_cores='quarter', **kwargs)[1]
                if save_results is True:
                    # Store flags for use in 1/f correction.
                    flags = res.dq
                    flags[flags != 0] = 1  # Convert to binary mask.
                    hdu = fits.PrimaryHDU(flags)
                    outfile = self.output_dir + self.fileroots[i] + 'pixelflags.fits'
                    hdu.writeto(outfile, overwrite=True)
                    # Remove rate file because we don't need it and I don't
                    # like having extra files.
                    rate = res.meta.filename.replace('_1_ramp', '_0_ramp')
                    os.remove(self.output_dir + rate)
                    # Verify that filename is correct.
                    current_name = self.output_dir + res.meta.filename
                    if expected_file != current_name:
                        res.close()
                        os.rename(current_name, expected_file)
                        res = datamodels.open(expected_file)
            results.append(res)

        return results


class GainScaleStep:
    """Wrapper around default calwebb_detector1 Gain Scale Correction step.
    """

    def __init__(self, input_data, output_dir):
        """Step initializer.
        """

        self.tag = 'gainscalestep.fits'
        self.output_dir = output_dir
        self.datafiles = utils.sort_datamodels(input_data)
        self.fileroots = utils.get_filename_root(self.datafiles)

    def run(self, save_results=True, force_redo=False, **kwargs):
        """Method to run the step.
        """

        results = []
        all_files = glob.glob(self.output_dir + '*')
        for i, segment in enumerate(self.datafiles):
            # If an output file for this segment already exists, skip the step.
            expected_file = self.output_dir + self.fileroots[i] + self.tag
            if expected_file in all_files and force_redo is False:
                fancyprint('File {} already exists.'.format(expected_file))
                fancyprint('Skipping Gain Scale Correction Step.')
                res = expected_file
            # If no output files are detected, run the step.
            else:
                step = calwebb_detector1.gain_scale_step.GainScaleStep()
                res = step.call(segment, output_dir=self.output_dir,
                                save_results=save_results, **kwargs)
                # Verify that filename is correct.
                if save_results is True:
                    current_name = self.output_dir + res.meta.filename
                    if expected_file != current_name:
                        res.close()
                        os.rename(current_name, expected_file)
                        res = datamodels.open(expected_file)
            results.append(res)

        return results


def flag_hot_pixels(result, deepframe, box_size=10, thresh=15, hot_pix=None):
    """Identify and flag additional hot pixels in a SOSS TSO which are not
    already in the default pipeline flags.

    Parameters
    ----------
    result : jwst.datamodel, str
        Input datamodel, or path to.
    deepframe : array-like(float), str
        Deep stack of the time series, or path to.
    box_size : int
        Size of box around each pixel to consider.
    thresh : int
        Sigma threshhold above which a pixel will be flagged.
    hot_pix : array-like(bool), None
        Map of pixels to flag.

    Returns
    -------
    result : jwst.datamodel
        Input datamodel with newly flagged pixels added to pixeldq extension.
    hot_pix : np.array(bool)
        Map of new flagged pixels.
    """

    fancyprint('Identifying additional unflagged hot pixels...')

    result = utils.open_filetype(result)
    # Open the deep frame.
    if isinstance(deepframe, str):
        deepframe = fits.getdata(deepframe, 2)
    dimy, dimx = np.shape(deepframe)
    all_med = np.nanmedian(deepframe)
    # Get location of all pixels already flagged as warm or hot.
    hot = utils.get_dq_flag_metrics(result.pixeldq, ['HOT', 'WARM'])

    if hot_pix is not None:
        fancyprint('Using provided hot pixel map...')
        assert np.shape(hot_pix) == np.shape(deepframe)
        result.pixeldq[hot_pix] += 2048

    else:
        hot_pix = np.zeros_like(deepframe).astype(bool)
        for i in tqdm(range(4, dimx - 4)):
            for j in range(dimy):
                box_size_i = box_size
                box_prop = utils.get_interp_box(deepframe, box_size_i, i, j,
                                                start=4, end=2040)
                # Ensure that the median and std dev extracted are good.
                # If not, increase the box size until they are.
                while np.any(np.isnan(box_prop)):
                    box_size_i += 1
                    box_prop = utils.get_interp_box(deepframe, box_size_i, i,
                                                    j, start=4, end=2040)
                med, std = box_prop[0], box_prop[1]

                # If central pixel is too deviant...
                if np.abs(deepframe[j, i] - med) >= (thresh * std):
                    # And reasonably bright (don't want to flag noise)...
                    if deepframe[j, i] > all_med:
                        # And not already flagged...
                        if hot[j, i] == 0:
                            # Flag it.
                            result.pixeldq[j, i] += 2048
                            hot_pix[j, i] = True

        count = int(np.sum(hot_pix))
        fancyprint('{} additional hot pixels identified.'.format(count))

    return result, hot_pix


def jumpstep_in_time(datafile, window=5, thresh=10, fileroot=None,
                     save_results=True, output_dir=None):
    """Jump detection step in the temporal domain. This algorithm is based off
    of Nikolov+ (2014) and identifies cosmic ray hits in the temporal domain.
    All jumps for ngroup<=2 are replaced with the median of surrounding
    integrations, whereas jumps for ngroup>3 are flagged.

    Parameters
    ----------
    datafile : str, RampModel
        Path to data file, or RampModel itself for a segment of the TSO.
        Should be 4D ramp.
    window : int
        Number of integrations to use for cosmic ray flagging.
    thresh : int
        Sigma threshold for a pixel to be flagged.
    output_dir : str
        Directory to which to save results.
    save_results : bool
        If True, save results to disk.
    fileroot : str, None
        Root name for output file.

    Returns
    -------
    datafile : RampModel
        Data file corrected for cosmic ray hits.
    """

    fancyprint('Starting time-domain jump detection step.')

    # If saving results, ensure output directory and fileroots are provided.
    if save_results is True:
        assert output_dir is not None
        assert fileroot is not None
        # Output directory formatting.
        if output_dir[-1] != '/':
            output_dir += '/'

    # Load in the datafile.
    datafile = utils.open_filetype(datafile)
    cube = datafile.data
    dqcube = datafile.groupdq

    nints, ngroups, dimy, dimx = np.shape(cube)
    # Jump detection algorithm based on Nikolov+ (2014). For each integration,
    # create a difference image using the median of surrounding integrations.
    # Flag all pixels with deviations more than X-sigma as comsic rays hits.
    count, interp = 0, 0
    for g in tqdm(range(ngroups)):
        # Filter the data using the specified window
        cube_filt = medfilt(cube[:, g], (window, 1, 1))
        # Calculate the point-to-point scatter along the temporal axis.
        scatter = np.median(np.abs(0.5 * (cube[0:-2, g] + cube[2:, g]) - cube[1:-1, g]), axis=0)
        scatter = np.where(scatter == 0, np.inf, scatter)
        # Find pixels which deviate more than the specified threshold.
        scale = np.abs(cube[:, g] - cube_filt) / scatter
        ii = np.where((scale > thresh) & (cube[:, g] > np.nanpercentile(cube, 10)))

        # If ngroup<=2, replace the pixel with the stack median so that a
        # ramp can still be fit.
        if ngroups <= 2:
            # Do not want to interpolate pixels which are flagged for
            # another reason, so only select good pixels or those which
            # are flagged for jumps.
            jj = (dqcube[:, g] == 0) | (dqcube[:, g] == 4)
            # Replace these pixels with the stack median and remove the
            # dq flag.
            replace = ii & jj
            cube[:, g][replace] = cube_filt[replace]
            dqcube[:, g][replace] = 0
            interp += np.sum(replace)
        # If ngroup>2, flag the pixel as having a jump.
        else:
            # Want to ignore pixels which are already flagged for a jump.
            jj = np.where(utils.get_dq_flag_metrics(dqcube[:, g], ['JUMP_DET']) == 1)
            alrdy_flg = np.ones_like(dqcube[:, g]).astype(bool)
            alrdy_flg[jj] = False
            new_flg = np.zeros_like(dqcube[:, g]).astype(bool)
            new_flg[ii] = True
            to_flag = new_flg & alrdy_flg
            # Add the jump detection flag.
            dqcube[:, g][to_flag] += 4
            count += int(np.sum(to_flag))

    fancyprint('{} jumps flagged'.format(count))
    fancyprint('and {} interpolated'.format(interp))

    datafile.data = cube
    datafile.groupdq = dqcube
    if save_results is True:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            datafile.write(output_dir + fileroot + 'jump.fits')
    datafile.close()

    fancyprint('Done')

    return datafile


def oneoverfstep_soss(datafiles, baseline_ints, even_odd_rows=True,
                      background=None, smoothed_wlc=None, output_dir=None,
                      save_results=True, pixel_masks=None, fileroots=None):
    """Custom 1/f correction routine to be applied at the group level. A
    median stack is constructed using all out-of-transit integrations and
    subtracted from each individual integration. The column-wise median of
    this difference image is then subtracted from the original frame to
    correct 1/f noise. Outlier pixels, background contaminants, and the target
    trace itself can (should) be masked to improve the estimation.

    Parameters
    ----------
    datafiles : array-like[str], array-like[RampModel], array-like[CubeModel]
        List of paths to data files, or datamodels themselves for each segment
        of the TSO. Should be 4D ramps, but 3D rate files are also accepted.
    baseline_ints : array-like[int]
        Integration numbers of ingress and egress.
    even_odd_rows : bool
        If True, calculate 1/f noise seperately for even and odd numbered rows.
    background : str, array-like[float], None
        Model of background flux.
    smoothed_wlc : array-like[float], str, None
        Estimate of the normalized light curve, or path to file containing it.
    output_dir : str, None
        Directory to which to save results. Only necessary if saving results.
    save_results : bool
        If True, save results to disk.
    pixel_masks : array-like[str], None
        List of paths to maps of pixels to mask for each data segment. Can be
        3D (nints, dimy, dimx), or 2D (dimy, dimx).
    fileroots : array-like[str], None
        Root names for output files. Only necessary if saving results.

    Returns
    -------
    corrected_rampmodels : array-like[CubeModel]
        RampModels for each segment, corrected for 1/f noise.
    """

    fancyprint('Starting 1/f correction step.')

    # If saving results, ensure output directory and fileroots are provided.
    if save_results is True:
        assert output_dir is not None
        assert fileroots is not None
        # Output directory formatting.
        if output_dir[-1] != '/':
            output_dir += '/'

    # Format the baseline frames.
    baseline_ints = utils.format_out_frames(baseline_ints)

    datafiles = np.atleast_1d(datafiles)
    # If outlier maps are passed, ensure that there is one for each segment.
    if pixel_masks is not None:
        pixel_masks = np.atleast_1d(pixel_masks)
        if len(pixel_masks) == 1:
            pixel_masks = [pixel_masks[0] for d in datafiles]

    # Load in datamodels from all segments.
    for i, file in enumerate(datafiles):
        with utils.open_filetype(file) as currentfile:
            # FULL frame uses multiple amplifiers and probably has to be
            # treated differently. Break if we encounter a FULL frame exposure.
            if currentfile.meta.subarray.name == 'FULL':
                raise NotImplementedError
            # To create the deepstack, join all segments together.
            if i == 0:
                cube = currentfile.data
            else:
                cube = np.concatenate([cube, currentfile.data], axis=0)

    # Generate the 3D deep stack (ngroup, dimy, dimx) using only
    # baseline integrations.
    fancyprint('Generating a deep stack for each group using baseline '
               'integrations...')
    deepstack = utils.make_deepstack(cube[baseline_ints])

    # In order to subtract off the trace as completely as possible, the median
    # stack must be scaled, via the transit curve, to the flux level of each
    # integration.
    # Try to open light curve file. If not, estimate it from data.
    if isinstance(smoothed_wlc, str):
        try:
            smoothed_wlc = np.load(smoothed_wlc)
        except (ValueError, FileNotFoundError):
            fancyprint('Light curve file cannot be opened. It will be '
                       'estimated from current data.', msg_type='WARNING')
            smoothed_wlc = None
    # If no lightcurve is provided, estimate it from the current data.
    if smoothed_wlc is None:
        postage = cube[:, -1, 20:60, 1500:1550]
        timeseries = np.nansum(postage, axis=(1, 2))
        timeseries = timeseries / np.nanmedian(timeseries[baseline_ints])
        # Smooth the time series on a timescale of roughly 2%.
        smoothed_wlc = median_filter(timeseries,
                                     int(0.02*np.shape(cube)[0]))

    # Background must be subtracted to accurately subtract off the target
    # trace and isolate 1/f noise. However, the background flux must also be
    # corrected for non-linearity. Therefore, it should be added back after
    # the 1/f is subtracted, in order to be re-subtracted later.
    if background is not None:
        if isinstance(background, str):
            background = np.load(background)

    # Individually treat each segment.
    corrected_rampmodels = []
    current_int = 0
    for n, currentfile in enumerate(datafiles):
        currentfile = datamodels.open(currentfile)
        fancyprint('Starting segment {} of {}.'.format(n+1, len(datafiles)))

        # Define the readout setup - can be 4D (recommended) or 3D.
        if np.ndim(currentfile.data) == 4:
            nint, ngroup, dimy, dimx = np.shape(currentfile.data)
        else:
            nint, dimy, dimx = np.shape(currentfile.data)

        # Read in the outlier map - a (nints, dimy, dimx) 3D cube
        if pixel_masks is None:
            fancyprint('No outlier maps passed, ignoring outliers.')
            outliers = np.zeros((nint, dimy, dimx))
        else:
            fancyprint('Using outlier map {}'.format(pixel_masks[n]))
            outliers = fits.getdata(pixel_masks[n])
            # If the outlier map is 2D (dimy, dimx) extend to int dimension.
            if np.ndim(outliers) == 2:
                outliers = np.repeat(outliers, nint)
                outliers = outliers.reshape((dimy, dimx, nint))
                outliers = outliers.transpose(2, 0, 1)
        # The outlier map is 0 where good and >0 otherwise. As this will be
        # applied multiplicatively, replace 0s with 1s and others with NaNs.
        outliers = np.where(outliers == 0, 1, np.nan)

        # Initialize output storage arrays.
        corr_data = np.copy(currentfile.data)
        # Loop over all integrations to determine the 1/f noise level via a
        # difference image, and correct it.
        for i in tqdm(range(nint)):
            # i counts ints in this particular segment, whereas ii counts
            # ints from the start of the exposure.
            ii = current_int + i
            # Create the difference image.
            sub = currentfile.data[i] - deepstack * smoothed_wlc[ii]
            # Apply the outlier mask.
            sub *= outliers[i, :, :]
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                if even_odd_rows is True:
                    # Calculate 1/f scaling seperately for even and odd rows.
                    # This *should* be taken care of by the RefPixStep, but it
                    # doesn't hurt to do it again.
                    dc = np.zeros_like(sub)
                    # For group-level corrections.
                    if np.ndim(currentfile.data) == 4:
                        dc[:, ::2] = bn.nanmedian(sub[:, ::2], axis=1)[:, None, :]
                        dc[:, 1::2] = bn.nanmedian(sub[:, 1::2], axis=1)[:, None, :]
                    # For integration-level corrections.
                    else:
                        dc[::2] = bn.nanmedian(sub[::2], axis=0)[None, :]
                        dc[1::2] = bn.nanmedian(sub[1::2], axis=0)[None, :]
                else:
                    # Single 1/f scaling for all rows.
                    dc = np.zeros_like(sub)
                    # For group-level corrections.
                    if np.ndim(currentfile.data == 4):
                        dc[:, :, :] = bn.nanmedian(sub, axis=1)[:, None, :]
                    # For integration-level corrections.
                    else:
                        dc[:, :] = bn.nanmedian(sub, axis=0)[None, :]
            # Make sure no NaNs are in the DC map
            dc = np.where(np.isfinite(dc), dc, 0)
            corr_data[i] -= dc
        current_int += nint  # Increment the total integration counter.

        # Add back the zodi background.
        if background is not None:
            corr_data += background

        # Store results.
        newfile = currentfile.copy()
        currentfile.close()
        newfile.data = corr_data
        corrected_rampmodels.append(newfile)

        # Save the results if requested.
        if save_results is True:
            suffix = 'oneoverfstep.fits'
            newfile.write(output_dir + fileroots[n] + suffix)

    return corrected_rampmodels


def oneoverfstep_nirspec(datafiles, even_odd_rows=True, output_dir=None,
                         save_results=True, pixel_masks=None, fileroots=None):
    """Custom 1/f correction routine to be applied at the group level. The
    median level of each detector column is subtracted off while masking
    outlier pixels and the target trace.

    Parameters
    ----------
    datafiles : array-like[str], array-like[RampModel], array-like[CubeModel]
        List of paths to data files, or datamodels themselves for each segment
        of the TSO. Should be 4D ramps, but 3D rate files are also accepted.
    even_odd_rows : bool
        If True, calculate 1/f noise seperately for even and odd numbered rows.
    output_dir : str, None
        Directory to which to save results. Only necessary if saving results.
    save_results : bool
        If True, save results to disk.
    pixel_masks : array-like[str], None
        List of paths to maps of pixels to mask for each data segment. Can be
        3D (nints, dimy, dimx), or 2D (dimy, dimx).
    fileroots : array-like[str], None
        Root names for output files. Only necessary if saving results.

    Returns
    -------
    corrected_rampmodels : array-like[CubeModel]
        RampModels for each segment, corrected for 1/f noise.
    """

    fancyprint('Starting 1/f correction step.')

    # If saving results, ensure output directory and fileroots are provided.
    if save_results is True:
        assert output_dir is not None
        assert fileroots is not None
        # Output directory formatting.
        if output_dir[-1] != '/':
            output_dir += '/'

    datafiles = np.atleast_1d(datafiles)
    # If outlier maps are passed, ensure that there is one for each segment.
    if pixel_masks is not None:
        pixel_masks = np.atleast_1d(pixel_masks)
        if len(pixel_masks) == 1:
            pixel_masks = [pixel_masks[0] for d in datafiles]

    # Individually treat each segment.
    corrected_rampmodels = []
    for n, currentfile in enumerate(datafiles):
        currentfile = datamodels.open(currentfile)
        fancyprint('Starting segment {} of {}.'.format(n+1, len(datafiles)))

        # Define the readout setup - can be 4D (recommended) or 3D.
        if np.ndim(currentfile.data) == 4:
            nint, ngroup, dimy, dimx = np.shape(currentfile.data)
        else:
            nint, dimy, dimx = np.shape(currentfile.data)

        # Read in the outlier map - a (nints, dimy, dimx) 3D cube
        if pixel_masks is None:
            fancyprint('No outlier maps passed, ignoring outliers.')
            outliers = np.zeros((nint, dimy, dimx))
        else:
            fancyprint('Using outlier map {}'.format(pixel_masks[n]))
            outliers = fits.getdata(pixel_masks[n])
            # If the outlier map is 2D (dimy, dimx) extend to int dimension.
            if np.ndim(outliers) == 2:
                outliers = np.repeat(outliers, nint)
                outliers = outliers.reshape((dimy, dimx, nint))
                outliers = outliers.transpose(2, 0, 1)
        # The outlier map is 0 where good and >0 otherwise. As this will be
        # applied multiplicatively, replace 0s with 1s and others with NaNs.
        outliers = np.where(outliers == 0, 1, np.nan)

        # Initialize output storage arrays.
        corr_data = np.copy(currentfile.data)
        # Loop over all integrations to determine the 1/f noise level and
        # correct it.

        diff_im = np.abs(np.diff(currentfile.data, axis=0))
        for i in tqdm(range(nint)):
            # For group-level corrections.
            if np.ndim(currentfile.data) == 4:
                for g in range(ngroup):
                    this_outlier = np.copy(outliers[i])
                    # Also mask pixels which are very deviant from surrounding
                    # pixels in time.
                    j = np.min([i, nint-2])
                    ii = np.where(diff_im[j, g] > np.nanpercentile(diff_im[j, g], 90))
                    this_outlier[ii] = np.nan
                    # Apply mask.
                    masked = currentfile.data[i, g] * this_outlier
                    # Find and subtract the 1/f level.
                    if even_odd_rows is True:
                        corr_data[i, g, :, ::2] -= bn.nanmedian(masked[:, ::2], axis=0)
                        corr_data[i, g, :, 1::2] -= bn.nanmedian(masked[:, 1::2], axis=0)
                    else:
                        corr_data[i, g] -= bn.nanmedian(masked, axis=0)
            else:
                # Also mask pixels which are very deviant from surrounding
                # pixels in time.
                j = np.min([i, nint - 2])
                ii = np.where(diff_im[j] > np.nanpercentile(diff_im[j], 90))
                outliers[i][ii] = np.nan
                # Apply mask.
                masked = currentfile.data[i] * outliers[i]
                # Find and subtract the 1/f level.
                if even_odd_rows is True:
                    corr_data[i, :, ::2] -= bn.nanmedian(masked[:, ::2], axis=0)
                    corr_data[i, :, 1::2] -= bn.nanmedian(masked[:, 1::2], axis=0)
                else:
                    corr_data[i] -= bn.nanmedian(masked, axis=0)

        # Store results.
        newfile = currentfile.copy()
        currentfile.close()
        newfile.data = corr_data
        corrected_rampmodels.append(newfile)

        # Save the results if requested.
        if save_results is True:
            suffix = 'oneoverfstep.fits'
            newfile.write(output_dir + fileroots[n] + suffix)

    return corrected_rampmodels


def run_stage1(results, background_model, baseline_ints=None,
               smoothed_wlc=None, save_results=True, pixel_masks=None,
               force_redo=False, deepframe=None, rejection_threshold=15,
               flag_in_time=False, time_rejection_threshold=10,
               root_dir='./', output_tag='', skip_steps=None, do_plot=False,
               show_plot=False, **kwargs):
    """Run the supreme-SPOON Stage 1 pipeline: detector level processing,
    using a combination of official STScI DMS and custom steps. Documentation
    for the official DMS steps can be found here:
    https://jwst-pipeline.readthedocs.io/en/latest/jwst/pipeline/calwebb_detector1.html

    Parameters
    ----------
    results : array-like[str]
        List of paths to input uncalibrated datafiles for all segments in an
        exposure.
    background_model : array-like[float]
        SOSS background model.
    baseline_ints : array-like[int]
        Integration numbers for transit ingress and egress.
    smoothed_wlc : array-like[float], None
        Estimate of the normalized light curve.
    save_results : bool
        If True, save results of each step to file.
    pixel_masks : array-like[str], None
        For improved 1/f noise corecton. List of paths to outlier maps for each
        data segment. Can be 3D (nints, dimy, dimx), or 2D (dimy, dimx) files.
    force_redo : bool
        If True, redo steps even if outputs files are already present.
    deepframe : str, None
        Path to deep stack, such as one produced by BadPixStep.
    rejection_threshold : int
        For jump detection; sigma threshold for a pixel to be considered an
        outlier.
    flag_in_time : bool
        If True, flag cosmic rays temporally in addition to the default
        up-the-ramp jump detection.
    time_rejection_threshold : int
        Sigma threshold to flag outliers in temporal flagging.
    root_dir : str
        Directory from which all relative paths are defined.
    output_tag : str
        Name tag to append to pipeline outputs directory.
    skip_steps : array-like[str], None
        Step names to skip (if any).
    do_plot : bool
        If True, make step diagnostic plots.
    show_plot : bool
        Only necessary if do_plot is True. Show the diagnostic plots in
        addition to/instead of saving to file.

    Returns
    -------
    results : array-like[RampModel]
        Datafiles for each segment processed through Stage 1.
    """

    # ============== DMS Stage 1 ==============
    # Detector level processing.
    fancyprint('**Starting supreme-SPOON Stage 1**')
    fancyprint('Detector level processing')

    if output_tag != '':
        output_tag = '_' + output_tag
    # Create output directories and define output paths.
    utils.verify_path(root_dir + 'pipeline_outputs_directory' + output_tag)
    outdir = root_dir + 'pipeline_outputs_directory' + output_tag + '/Stage1/'
    utils.verify_path(outdir)

    if skip_steps is None:
        skip_steps = []

    # ===== Group Scale Step =====
    # Default DMS step.
    if 'GroupScaleStep' not in skip_steps:
        if 'GroupScaleStep' in kwargs.keys():
            step_kwargs = kwargs['GroupScaleStep']
        else:
            step_kwargs = {}
        step = GroupScaleStep(results, output_dir=outdir)
        results = step.run(save_results=save_results, force_redo=force_redo,
                           **step_kwargs)

    # ===== Data Quality Initialization Step =====
    # Default/Custom DMS step.
    if 'DQInitStep' not in skip_steps:
        if 'DQInitStep' in kwargs.keys():
            step_kwargs = kwargs['DQInitStep']
        else:
            step_kwargs = {}
        step = DQInitStep(results, output_dir=outdir, deepframe=deepframe)
        results = step.run(save_results=save_results, force_redo=force_redo,
                           **step_kwargs)

    # ===== Saturation Detection Step =====
    # Default DMS step.
    if 'SaturationStep' not in skip_steps:
        if 'SaturationStep' in kwargs.keys():
            step_kwargs = kwargs['SaturationStep']
        else:
            step_kwargs = {}
        step = SaturationStep(results, output_dir=outdir)
        results = step.run(save_results=save_results, force_redo=force_redo,
                           **step_kwargs)

    # ===== Superbias Subtraction Step =====
    # Default DMS step.
    if 'SuperBiasStep' not in skip_steps:
        if 'SuperBiasStep' in kwargs.keys():
            step_kwargs = kwargs['SuperBiasStep']
        else:
            step_kwargs = {}
        step = SuperBiasStep(results, output_dir=outdir)
        results = step.run(save_results=save_results, force_redo=force_redo,
                           do_plot=do_plot, show_plot=show_plot, **step_kwargs)

    if 'OneOverFStep' not in skip_steps:
        # ===== Background Subtraction Step =====
        # Custom DMS step - imported from Stage2.
        step = BackgroundStep(results, background_model=background_model,
                              output_dir=outdir)
        results = step.run(save_results=save_results, force_redo=force_redo)
        results, background_model = results

        # ===== 1/f Noise Correction Step =====
        # Custom DMS step.
        if 'OneOverFStep' in kwargs.keys():
            step_kwargs = kwargs['OneOverFStep']
        else:
            step_kwargs = {}
        step = OneOverFStep(results, baseline_ints=baseline_ints,
                            output_dir=outdir, pixel_masks=pixel_masks,
                            smoothed_wlc=smoothed_wlc,
                            background=background_model)
        results = step.run(save_results=save_results, force_redo=force_redo,
                           do_plot=do_plot, show_plot=show_plot, **step_kwargs)

    # ===== Linearity Correction Step =====
    # Default DMS step.
    if 'LinearityStep' not in skip_steps:
        if 'LinearityStep' in kwargs.keys():
            step_kwargs = kwargs['LinearityStep']
        else:
            step_kwargs = {}
        step = LinearityStep(results, output_dir=outdir)
        results = step.run(save_results=save_results, force_redo=force_redo,
                           do_plot=do_plot, show_plot=show_plot, **step_kwargs)

    # ===== Jump Detection Step =====
    # Default/Custom DMS step.
    if 'JumpStep' not in skip_steps:
        if 'JumpStep' in kwargs.keys():
            step_kwargs = kwargs['JumpStep']
        else:
            step_kwargs = {}
        step = JumpStep(results, output_dir=outdir)
        results = step.run(save_results=save_results, force_redo=force_redo,
                           rejection_threshold=rejection_threshold,
                           flag_in_time=flag_in_time,
                           time_rejection_threshold=time_rejection_threshold,
                           do_plot=do_plot, show_plot=show_plot, **step_kwargs)

    # ===== Ramp Fit Step =====
    # Default DMS step.
    if 'RampFitStep' not in skip_steps:
        if 'RampFitStep' in kwargs.keys():
            step_kwargs = kwargs['RampFitStep']
        else:
            step_kwargs = {}
        step = RampFitStep(results, output_dir=outdir)
        results = step.run(save_results=save_results, force_redo=force_redo,
                           **step_kwargs)

    # ===== Gain Scale Correcton Step =====
    # Default DMS step.
    if 'GainScaleStep' not in skip_steps:
        if 'GainScaleStep' in kwargs.keys():
            step_kwargs = kwargs['GainScaleStep']
        else:
            step_kwargs = {}
        step = GainScaleStep(results, output_dir=outdir)
        results = step.run(save_results=save_results, force_redo=force_redo,
                           **step_kwargs)

    return results
