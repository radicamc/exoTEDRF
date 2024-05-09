#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thurs Jul 21 17:30 2022

@author: MCR

Custom JWST DMS pipeline steps for Stage 1 (detector level processing).
"""

from astropy.io import fits
import bottleneck as bn
import copy
import glob
import numpy as np
import os
import pandas as pd
from scipy.interpolate import griddata
from scipy.ndimage import median_filter
from scipy.signal import medfilt
from tqdm import tqdm
import warnings

from jwst import datamodels
from jwst.pipeline import calwebb_detector1
from jwst.pipeline import calwebb_spec2

import exotedrf.stage2 as stage2
from exotedrf import utils, plotting
from exotedrf.utils import fancyprint


class DQInitStep:
    """Wrapper around default calwebb_detector1 Data Quality Initialization
    step with additional hot pixel flagging.
    """

    def __init__(self, input_data, output_dir, deepframe=None):
        """Step initializer.

        Parameters
        ----------
        input_data : array-like(str), array-like(datamodel)
            List of paths to input data or the input data itself.
        output_dir : str
            Path to directory to which to save outputs.
        deepframe : str, np.ndarray(float)
            Path to observation deepstack or the deepstack itself.
        """

        # Set up easy attributes.
        self.tag = 'dqinitstep.fits'
        self.output_dir = output_dir

        # Unpack input data files.
        datafiles = utils.sort_datamodels(input_data)
        self.datafiles = []
        for file in datafiles:
            self.datafiles.append(utils.open_filetype(file))
        self.fileroots = utils.get_filename_root(self.datafiles)

        # Unpack deepframe.
        if isinstance(deepframe, str):
            # Want deepframe extension before bad pixel interpolation.
            fancyprint('Reading deepframe file: {}...'.format(deepframe))
            self.deepframe = fits.getdata(deepframe, 2)
        elif isinstance(deepframe, np.ndarray) or deepframe is None:
            self.deepframe = deepframe
        else:
            msg = 'Invalid type for deepframe: {}'.format(type(deepframe))
            raise ValueError(msg)

    def run(self, save_results=True, force_redo=False, **kwargs):
        """Method to run the step.

        Parameters
        ----------
        save_results : bool
            If True, save results.
        force_redo : bool
            If True, run step even if output files are detected.
        kwargs : dict
            Keyword arguments for calwebb_detector1.dq_init_step.DQInitStep.

        Returns
        -------
        results : list(datamodel)
            Input data files processed through the step.
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
                res = datamodels.open(expected_file)
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

        Parameters
        ----------
        input_data : array-like(str), array-like(datamodel)
            List of paths to input data or the input data itself.
        output_dir : str
            Path to directory to which to save outputs.
        """

        # Set up easy attributes.
        self.tag = 'saturationstep.fits'
        self.output_dir = output_dir

        # Unpack input data files.
        datafiles = utils.sort_datamodels(input_data)
        self.datafiles = []
        for file in datafiles:
            self.datafiles.append(utils.open_filetype(file))
        self.fileroots = utils.get_filename_root(self.datafiles)

    def run(self, save_results=True, force_redo=False, **kwargs):
        """Method to run the step.

        Parameters
        ----------
        save_results : bool
            If True, save results.
        force_redo : bool
            If True, run step even if output files are detected.
        kwargs : dict
            Keyword arguments for
            calwebb_detector1.saturation_step.SaturationStep.

        Returns
        -------
        results : list(datamodel)
            Input data files processed through the step.
        """

        results = []
        all_files = glob.glob(self.output_dir + '*')
        for i, segment in enumerate(self.datafiles):
            # If an output file for this segment already exists, skip the step.
            expected_file = self.output_dir + self.fileroots[i] + self.tag
            if expected_file in all_files and force_redo is False:
                fancyprint('File {} already exists.'.format(expected_file))
                fancyprint('Skipping Saturation Detection Step.')
                res = datamodels.open(expected_file)
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

        Parameters
        ----------
        input_data : array-like(str), array-like(datamodel)
            List of paths to input data or the input data itself.
        output_dir : str
            Path to directory to which to save outputs.
        """

        # Set up easy attributes.
        self.tag = 'superbiasstep.fits'
        self.output_dir = output_dir

        # Unpack input data files.
        datafiles = utils.sort_datamodels(input_data)
        self.datafiles = []
        for file in datafiles:
            self.datafiles.append(utils.open_filetype(file))
        self.fileroots = utils.get_filename_root(self.datafiles)

        self.instrument = utils.get_instrument_name(self.datafiles[0])

    def run(self, save_results=True, force_redo=False, do_plot=False,
            show_plot=False, **kwargs):
        """Method to run the step.

        Parameters
        ----------
        save_results : bool
            If True, save results.
        force_redo : bool
            If True, run step even if output files are detected.
        do_plot : bool
            If True, do step diagnostic plot.
        show_plot : bool
            If True, show the step diagnostic plot.
        kwargs : dict
            Keyword arguments for
            calwebb_detector1.superbias_step.SuperBiasStep.

        Returns
        -------
        results : list(datamodel)
            Input data files processed through the step.
        """

        results = []
        all_files = glob.glob(self.output_dir + '*')
        for i, segment in enumerate(self.datafiles):
            # If an output file for this segment already exists, skip the step.
            expected_file = self.output_dir + self.fileroots[i] + self.tag
            if expected_file in all_files and force_redo is False:
                fancyprint('File {} already exists.'.format(expected_file))
                fancyprint('Skipping Superbias Subtraction Step.')
                res = datamodels.open(expected_file)
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
                plot_file = self.output_dir + self.tag.replace('.fits', '.png')
                if self.instrument == 'NIRSPEC':
                    det = utils.get_detector_name(self.datafiles[0])
                    plot_file = plot_file.replace('.png',
                                                  '_{}.png'.format(det))
            else:
                plot_file = None
            plotting.make_superbias_plot(results, outfile=plot_file,
                                         show_plot=show_plot)

        return results


class RefPixStep:
    """Wrapper around default calwebb_detector1 Reference Pixel Correction
    step.
    """

    def __init__(self, input_data, output_dir):
        """Step initializer.

        Parameters
        ----------
        input_data : array-like(str), array-like(datamodel)
            List of paths to input data or the input data itself.
        output_dir : str
            Path to directory to which to save outputs.
        """

        # Set up easy attributes.
        self.tag = 'refpixstep.fits'
        self.output_dir = output_dir

        # Unpack input data files.
        datafiles = utils.sort_datamodels(input_data)
        self.datafiles = []
        for file in datafiles:
            self.datafiles.append(utils.open_filetype(file))
        self.fileroots = utils.get_filename_root(self.datafiles)

    def run(self, save_results=True, force_redo=False, **kwargs):
        """Method to run the step.

        Parameters
        ----------
        save_results : bool
            If True, save results.
        force_redo : bool
            If True, run step even if output files are detected.
        kwargs : dict
            Keyword arguments for calwebb_detector1.refpix_step.RefPixStep.

        Returns
        -------
        results : list(datamodel)
            Input data files processed through the step.
        """

        results = []
        all_files = glob.glob(self.output_dir + '*')
        for i, segment in enumerate(self.datafiles):
            # If an output file for this segment already exists, skip the step.
            expected_file = self.output_dir + self.fileroots[i] + self.tag
            if expected_file in all_files and force_redo is False:
                fancyprint('File {} already exists.'.format(expected_file))
                fancyprint('Skipping Reference Pixel Correction Step.')
                res = datamodels.open(expected_file)
            # If no output files are detected, run the step.
            else:
                step = calwebb_detector1.refpix_step.RefPixStep()
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


class DarkCurrentStep:
    """Wrapper around default calwebb_detector1 Dark Current Subtraction step.
    """

    def __init__(self, input_data, output_dir):
        """Step initializer.

        Parameters
        ----------
        input_data : array-like(str), array-like(datamodel)
            List of paths to input data or the input data itself.
        output_dir : str
            Path to directory to which to save outputs.
        """

        # Set up easy attributes.
        self.tag = 'darkcurrentstep.fits'
        self.output_dir = output_dir

        # Unpack input data files.
        datafiles = utils.sort_datamodels(input_data)
        self.datafiles = []
        for file in datafiles:
            self.datafiles.append(utils.open_filetype(file))
        self.fileroots = utils.get_filename_root(self.datafiles)

    def run(self, save_results=True, force_redo=False, **kwargs):
        """Method to run the step.

        Parameters
        ----------
        save_results : bool
            If True, save results.
        force_redo : bool
            If True, run step even if output files are detected.
        kwargs : dict
            Keyword arguments for
            calwebb_detector1.dark_current_step.DarkCurrentStep.

        Returns
        -------
        results : list(datamodel)
            Input data files processed through the step.
        """

        results = []
        all_files = glob.glob(self.output_dir + '*')
        for i, segment in enumerate(self.datafiles):
            # If an output file for this segment already exists, skip the step.
            expected_file = self.output_dir + self.fileroots[i] + self.tag
            if expected_file in all_files and force_redo is False:
                fancyprint('File {} already exists.'.format(expected_file))
                fancyprint('Skipping Dark Current Subtraction Step.')
                res = datamodels.open(expected_file)
            # If no output files are detected, run the step.
            else:
                step = calwebb_detector1.dark_current_step.DarkCurrentStep()
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


class OneOverFStep:
    """Wrapper around custom 1/f Correction Step.
    """

    def __init__(self, input_data, output_dir, baseline_ints=None,
                 pixel_masks=None, centroids=None, soss_background=None,
                 method='scale-achromatic', soss_timeseries=None,
                 soss_timeseries_o2=None):
        """Step initializer.

        Parameters
        ----------
        input_data : array-like(str), array-like(datamodel)
            List of paths to input data or the input data itself.
        baseline_ints : array-like(int), None
            Integration number(s) to use as ingress and/or egress.
        output_dir : str
            Path to directory to which to save outputs.
        method : str
            Correction method. SOSS options are "scale-chromatic",
            "scale-achromatic", "scale-achromatic-window", or "solve".
            NIRSpec options are "median" or "slope".
        soss_timeseries : np.ndarray(float), str, None
            Path to a file containing light curve(s) for order 1, or the
            light curve(s) themselves.
        soss_timeseries_o2 : np.ndarray(float), str, None
            Path to a file containing light curves for order 2, or the
            light curves themselves.
        pixel_masks : array-like(str), np.ndarray(float), None
            List of paths to maps of pixels to mask for each data segment or
            the masks themselves. Should be 3D (nints, dimy, dimx).
        soss_background : np.ndarray(float), str, None
            Model of background flux.
        centroids : str, dict, None
            Path to file containing trace positions for each order or the
            centroids dictionary itself.
        """

        # Set up easy attributes.
        self.tag = 'oneoverfstep.fits'
        self.output_dir = output_dir
        self.method = method

        # Unpack input data files.
        datafiles = utils.sort_datamodels(input_data)
        self.datafiles = []
        for file in datafiles:
            self.datafiles.append(utils.open_filetype(file))
        self.fileroots = utils.get_filename_root(self.datafiles)

        # Unpack centroids.
        if isinstance(centroids, str):
            fancyprint('Reading centroids file: {}...'.format(centroids))
            self.centroids = pd.read_csv(centroids, comment='#')
        else:
            self.centroids = centroids

        # Unpack timeseries.
        if isinstance(soss_timeseries, str):
            fancyprint('Reading timeseries file: {}...'.format(soss_timeseries))
            self.timeseries = np.load(soss_timeseries)
        elif isinstance(soss_timeseries, np.ndarray) or soss_timeseries is None:
            self.timeseries = soss_timeseries
        else:
            msg = 'Invalid type for timeseries: {}'.format(type(soss_timeseries))
            raise ValueError(msg)

        # Unpack timeseries for order 2.
        if isinstance(soss_timeseries_o2, str):
            fancyprint('Reading order 2timeseries file: {}...'
                       ''.format(soss_timeseries_o2))
            self.timeseries_o2 = np.load(soss_timeseries_o2)
        elif isinstance(soss_timeseries_o2, np.ndarray) or soss_timeseries_o2 is None:
            self.timeseries_o2 = soss_timeseries_o2
        else:
            msg = 'Invalid type for timeseries_o2: {}' \
                  ''.format(type(soss_timeseries_o2))
            raise ValueError(msg)

        # Unpack pixel masks.
        if pixel_masks is not None:
            pixel_masks = np.atleast_1d(pixel_masks)
            self.pixel_masks = []
            for mask in pixel_masks:
                if isinstance(mask, str):
                    fancyprint('Reading pixel mask file: {}...'.format(mask))
                    self.pixel_masks.append(fits.getdata(mask))
                elif isinstance(mask, np.ndarray):
                    self.pixel_masks.append(mask)
                else:
                    msg = 'Invalid type for pixel_masks: {}'.format(type(mask))
                    raise ValueError(msg)
            assert len(self.pixel_masks) == len(self.datafiles)
        else:
            self.pixel_masks = pixel_masks

        # Unpack background.
        if isinstance(soss_background, str):
            fancyprint('Reading background file: {}...'.format(soss_background))
            self.background = np.load(soss_background)
        elif isinstance(soss_background, np.ndarray) or soss_background is None:
            self.background = soss_background
        else:
            msg = 'Invalid type for background: {}'.format(type(soss_background))
            raise ValueError(msg)

        # Get instrument.
        self.instrument = utils.get_instrument_name(self.datafiles[0])
        if self.instrument == 'NIRISS':
            assert baseline_ints is not None
            self.baseline_ints = baseline_ints
        # Some attributes that are not needed for NIRSpec, but expected for
        # plotting purposes.
        if self.instrument == 'NIRSPEC':
            nint = self.datafiles[0].meta.exposure.nints
            self.smoothed_wlc = np.ones(nint)
            self.baseline_ints = int(0.25 * nint)
            # Default to median method for NIRSpec.
            if self.method == 'scale-achromatic':
                self.method = 'median'

    def run(self, soss_inner_mask_width=40, soss_outer_mask_width=70,
            nirspec_mask_width=16, save_results=True, force_redo=False,
            do_plot=False, show_plot=False, **kwargs):
        """Method to run the step.

        Parameters
        ----------
        soss_inner_mask_width : int
            Inner full-width (in pixels) around the target trace to mask for
            SOSS.
        soss_outer_mask_width : int
            Outer full-width (in pixels) around the target trace to mask for
            SOSS.
        nirspec_mask_width : int
            Full-width (in pixels) around the target trace to mask for NIRSpec.
        save_results : bool
            If True, save results.
        force_redo : bool
            If True, run step even if output files are detected.
        do_plot : bool
            If True, do step diagnostic plot.
        show_plot : bool
            If True, show the step diagnostic plot.
        kwargs : dict
            Keyword arguments for stage1.oneoverfstep_scale,
            stage1.oneoverfstep_solve, or stage1.oneoverfstep_nirspec.

        Returns
        -------
        results : list(datamodel)
            Input data files processed through the step.
        """

        fancyprint('OneOverFStep instance created.')

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
                results.append(datamodels.open(expected_file))
        if do_step == 1 and force_redo is False:
            fancyprint('Output files already exist.')
            fancyprint('Skipping 1/f Correction Step.')

            return results

        # If no output files are detected, run the step.
        if self.instrument == 'SOSS':
            if self.method in ['scale-chromatic', 'scale-achromatic',
                               'scale-achromatic-window']:
                # To use "reference files" to calculate 1/f noise.
                method = self.method.split('scale-')[-1]
                results = oneoverfstep_scale(self.datafiles,
                                             baseline_ints=self.baseline_ints,
                                             inner_mask_width=soss_inner_mask_width,
                                             outer_mask_width=soss_outer_mask_width,
                                             background=self.background,
                                             timeseries=self.timeseries,
                                             timeseries_o2=self.timeseries_o2,
                                             output_dir=self.output_dir,
                                             save_results=save_results,
                                             pixel_masks=self.pixel_masks,
                                             fileroots=self.fileroots,
                                             method=method,
                                             centroids=self.centroids,
                                             **kwargs)
            elif self.method == 'solve':
                # To use MLE to solve for the 1/f noise.
                results = oneoverfstep_solve(self.datafiles,
                                             baseline_ints=self.baseline_ints,
                                             trace_width=soss_outer_mask_width,
                                             background=self.background,
                                             output_dir=self.output_dir,
                                             save_results=save_results,
                                             pixel_masks=self.pixel_masks,
                                             fileroots=self.fileroots,
                                             do_plot=do_plot,
                                             show_plot=show_plot,
                                             centroids=self.centroids)
            else:
                # Raise error otherwise.
                msg = 'Unrecognized 1/f correction: {}' \
                      ''.format(self.method)
                raise ValueError(msg)
        else:
            if self.method not in ['median', 'slope']:
                # Raise error for bad method.
                msg = 'Unrecognized 1/f correction: {}' \
                      ''.format(self.method)
                raise ValueError(msg)

            results = oneoverfstep_nirspec(self.datafiles,
                                           output_dir=self.output_dir,
                                           save_results=save_results,
                                           pixel_masks=self.pixel_masks,
                                           fileroots=self.fileroots,
                                           mask_width=nirspec_mask_width,
                                           centroids=self.centroids,
                                           method=self.method)

        # Do step plots if requested.
        if do_plot is True:
            if save_results is True:
                plot_file1 = self.output_dir + self.tag.replace(
                    '.fits', '_1.png')
                plot_file2 = self.output_dir + self.tag.replace(
                    '.fits', '_2.png')
                if self.instrument == 'NIRSPEC':
                    det = utils.get_detector_name(self.datafiles[0])
                    plot_file1 = plot_file1.replace('_1.png',
                                                    '_1_{}.png'.format(det))
                    plot_file2 = plot_file2.replace('_2.png',
                                                    '_2_{}.png'.format(det))
            else:
                plot_file1, plot_file2 = None, None

            # make sure we have the correct frame time.
            if self.instrument == 'NIRSPEC':
                tframe = 0.902
            else:
                tframe = 5.494

            # For scale-chromatic correction, collapse the 2D timeseries to
            # 1D for plotting purposes.
            if self.method == 'scale-chromatic':
                this_ts = np.nanmedian(self.timeseries, axis=1)
            else:
                this_ts = self.timeseries
            plotting.make_oneoverf_plot(results,
                                        timeseries=this_ts,
                                        baseline_ints=self.baseline_ints,
                                        outfile=plot_file1,
                                        show_plot=show_plot)
            plotting.make_oneoverf_psd(results, self.datafiles,
                                       timeseries=this_ts,
                                       baseline_ints=self.baseline_ints,
                                       pixel_masks=self.pixel_masks,
                                       outfile=plot_file2,
                                       show_plot=show_plot, tframe=tframe)

        fancyprint('Step OneOverFStep done.')

        return results


class LinearityStep:
    """Wrapper around default calwebb_detector1 Linearity Correction step.
    """

    def __init__(self, input_data, output_dir):
        """Step initializer.

        Parameters
        ----------
        input_data : array-like(str), array-like(datamodel)
            List of paths to input data or the input data itself.
        output_dir : str
            Path to directory to which to save outputs.
        """

        # Set up easy attributes.
        self.tag = 'linearitystep.fits'
        self.output_dir = output_dir

        # Unpack input data files.
        datafiles = utils.sort_datamodels(input_data)
        self.datafiles = []
        for file in datafiles:
            self.datafiles.append(utils.open_filetype(file))
        self.fileroots = utils.get_filename_root(self.datafiles)

    def run(self, save_results=True, force_redo=False, do_plot=False,
            show_plot=False, **kwargs):
        """Method to run the step.

        Parameters
        ----------
        save_results : bool
            If True, save results.
        force_redo : bool
            If True, run step even if output files are detected.
        do_plot : bool
            If True, do step diagnostic plot.
        show_plot : bool
            If True, show the step diagnostic plot.
        kwargs : dict
            Keyword arguments for
            calwebb_detector1.linearity_step.LinearityStep.

        Returns
        -------
        results : list(datamodel)
            Input data files processed through the step.
        """

        results = []
        all_files = glob.glob(self.output_dir + '*')
        for i, segment in enumerate(self.datafiles):
            # If an output file for this segment already exists, skip the step.
            expected_file = self.output_dir + self.fileroots[i] + self.tag
            if expected_file in all_files and force_redo is False:
                fancyprint('File {} already exists.'.format(expected_file))
                fancyprint('Skipping Linearity Correction Step.')
                res = datamodels.open(expected_file)
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
                plot_file1 = self.output_dir + self.tag.replace('.fits',
                                                                '_1.png')
                plot_file2 = self.output_dir + self.tag.replace('.fits',
                                                                '_2.png')
            else:
                plot_file1, plot_file2 = None, None
            plotting.make_linearity_plot(results, self.datafiles,
                                         outfile=plot_file1,
                                         show_plot=show_plot)
            plotting.make_linearity_plot2(results, self.datafiles,
                                          outfile=plot_file2,
                                          show_plot=show_plot)

        return results


class JumpStep:
    """Wrapper around default calwebb_detector1 Jump Detection step with some
    custom modifications.
    """

    def __init__(self, input_data, output_dir):
        """Step initializer.

        Parameters
        ----------
        input_data : array-like(str), array-like(datamodel)
            List of paths to input data or the input data itself.
        output_dir : str
            Path to directory to which to save outputs.
        """

        # Set up easy attributes.
        self.tag = 'jump.fits'
        self.output_dir = output_dir

        # Unpack input data files.
        datafiles = utils.sort_datamodels(input_data)
        self.datafiles = []
        for file in datafiles:
            self.datafiles.append(utils.open_filetype(file))
        self.fileroots = utils.get_filename_root(self.datafiles)

    def run(self, save_results=True, force_redo=False, flag_up_ramp=False,
            rejection_threshold=15, flag_in_time=True,
            time_rejection_threshold=10, time_window=5, do_plot=False,
            show_plot=False, **kwargs):
        """Method to run the step.

        Parameters
        ----------
        save_results : bool
            If True, save results.
        force_redo : bool
            If True, run step even if output files are detected.
        flag_up_ramp : bool
            If True, run up-the-ramp jump flagging.
        rejection_threshold : int
            Sigma threshold for an outlier to be considered a jump for
            up-the-ramp flagging.
        flag_in_time : bool
            If True, run time-domain flagging.
        time_rejection_threshold : int
            Sigma threshold for an outlier to be considered a jump for
            time-domain flagging.
        time_window : int
            Integration window to consider for time-domain flagging.
        do_plot : bool
            If True, do step diagnostic plot.
        show_plot : bool
            If True, show the step diagnostic plot.
        kwargs : dict
            Keyword arguments for calwebb_detector1.jump_step.JumpStep.

        Returns
        -------
        results : list(datamodel)
            Input data files processed through the step.
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
                # Get number of groups in the observation --- ngroup=2 must be
                # treated in a special way as the default pipeline JumpStep
                # will fail.
                # Also need to set minimum_sigclip_groups to something >nints,
                # else the up-the-ramp jump detection will be replaced by a
                # time-domain sigma clipping.
                testfile = datamodels.open(self.datafiles[0])
                ngroups = testfile.meta.exposure.ngroups
                i_start = testfile.meta.exposure.integration_start
                i_end = testfile.meta.exposure.integration_end
                nints = i_end - i_start
                testfile.close()
                # For ngroup > 2, default JumpStep can be used.
                if ngroups > 2 and flag_up_ramp is True:
                    step = calwebb_detector1.jump_step.JumpStep()
                    res = step.call(segment, output_dir=self.output_dir,
                                    save_results=save_results,
                                    rejection_threshold=rejection_threshold,
                                    maximum_cores='quarter',
                                    minimum_sigclip_groups=nints+100,
                                    **kwargs)
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
                plot_file = self.output_dir + self.tag.replace('.fits', '.png')
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

        Parameters
        ----------
        input_data : array-like(str), array-like(datamodel)
            List of paths to input data or the input data itself.
        output_dir : str
            Path to directory to which to save outputs.
        """

        # Set up easy attributes.
        self.tag = 'rampfitstep.fits'
        self.output_dir = output_dir

        # Unpack input data files.
        datafiles = utils.sort_datamodels(input_data)
        self.datafiles = []
        for file in datafiles:
            self.datafiles.append(utils.open_filetype(file))
        self.fileroots = utils.get_filename_root(self.datafiles)

    def run(self, save_results=True, force_redo=False, **kwargs):
        """Method to run the step.

        Parameters
        ----------
        save_results : bool
            If True, save results.
        force_redo : bool
            If True, run step even if output files are detected.
        kwargs : dict
            Keyword arguments for calwebb_detector1.ramp_fit_step.RampFitStep.

        Returns
        -------
        results : list(datamodel)
            Input data files processed through the step.
        """

        results = []
        all_files = glob.glob(self.output_dir + '*')
        for i, segment in enumerate(self.datafiles):
            # If an output file for this segment already exists, skip the step.
            expected_file = self.output_dir + self.fileroots[i] + self.tag
            if expected_file in all_files and force_redo is False:
                fancyprint('File {} already exists.'.format(expected_file))
                fancyprint('Skipping Ramp Fit Step.')
                res = datamodels.open(expected_file)
            # If no output files are detected, run the step.
            else:
                step = calwebb_detector1.ramp_fit_step.RampFitStep()
                res = step.call(segment, output_dir=self.output_dir,
                                save_results=save_results,
                                maximum_cores='quarter', **kwargs)[1]
                # From jwst v1.9.0-1.11.0 ramp fitting algorithm was changed to
                # make all pixels with DO_NOT_USE DQ flags be NaN after ramp
                # fitting. These pixels are marked, ignored and interpolated
                # anyways, so this does not change any actual functionality,
                # but cosmetcically this annoys me, as now plots look terrible.
                # Just griddata interpolate all NaNs so things look better.
                # Note this does not supercede any interpolation done later in
                # Stage 2.
                nint, dimy, dimx = res.data.shape
                px, py = np.meshgrid(np.arange(dimx), np.arange(dimy))
                fancyprint('Doing cosmetic NaN interpolation.')
                for j in range(nint):
                    ii = np.where(np.isfinite(res.data[j]))
                    res.data[j] = griddata(ii, res.data[j][ii], (py, px),
                                           method='nearest')
                if save_results is True:
                    res.save(self.output_dir + res.meta.filename)

                if save_results is True:
                    # Store flags for use in 1/f correction.
                    flags = res.dq
                    flags[flags != 0] = 1  # Convert to binary mask.
                    # Mask detector reset artifact.
                    int_start = res.meta.exposure.integration_start
                    int_end = np.min([res.meta.exposure.integration_end, 256])
                    # Artifact only affects first 256 integrations.
                    if int_start < 255:
                        for j, jj in enumerate(range(int_start, int_end)):
                            # j counts ints from start of this segment, jj is
                            # integrations from start of exposure (1-indexed).
                            # Mask rows from jj to jj+3 for detector reset
                            # artifact.
                            min_row = np.max([256-(jj+3), 0])
                            max_row = np.min([(258 - jj), 256])
                            flags[j, min_row:max_row, :] = 1
                    # Save flags to file.
                    hdu = fits.PrimaryHDU()
                    hdu1 = fits.ImageHDU(flags)
                    hdul = fits.HDUList([hdu, hdu1])
                    outfile = self.output_dir + self.fileroots[i] + 'pixelflags.fits'
                    hdul.writeto(outfile, overwrite=True)
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

        Parameters
        ----------
        input_data : array-like(str), array-like(datamodel)
            List of paths to input data or the input data itself.
        output_dir : str
            Path to directory to which to save outputs.
        """

        # Set up easy attributes.
        self.tag = 'gainscalestep.fits'
        self.output_dir = output_dir

        # Unpack input data files.
        datafiles = utils.sort_datamodels(input_data)
        self.datafiles = []
        for file in datafiles:
            self.datafiles.append(utils.open_filetype(file))
        self.fileroots = utils.get_filename_root(self.datafiles)

    def run(self, save_results=True, force_redo=False, **kwargs):
        """Method to run the step.

        Parameters
        ----------
        save_results : bool
            If True, save results.
        force_redo : bool
            If True, run step even if output files are detected.
        kwargs : dict
            Keyword arguments for
            calwebb_detector1.gain_scale_step.GainScaleStep.

        Returns
        -------
        results : list(datamodel)
            Input data files processed through the step.
        """

        results = []
        all_files = glob.glob(self.output_dir + '*')
        for i, segment in enumerate(self.datafiles):
            # If an output file for this segment already exists, skip the step.
            expected_file = self.output_dir + self.fileroots[i] + self.tag
            if expected_file in all_files and force_redo is False:
                fancyprint('File {} already exists.'.format(expected_file))
                fancyprint('Skipping Gain Scale Correction Step.')
                res = datamodels.open(expected_file)
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
    result : jwst.datamodel
        Input datamodel.
    deepframe : array-like(float)
        Deep stack of the time series.
    box_size : int
        Size of box around each pixel to consider.
    thresh : int
        Sigma threshold above which a pixel will be flagged.
    hot_pix : array-like(bool), None
        Map of pixels to flag.

    Returns
    -------
    result : jwst.datamodel
        Input datamodel with newly flagged pixels added to pixeldq extension.
    hot_pix : np.ndarray(bool)
        Map of new flagged pixels.
    """

    fancyprint('Identifying additional unflagged hot pixels...')

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
                                                dimx)
                # Ensure that the median and std dev extracted are good.
                # If not, increase the box size until they are.
                while np.any(np.isnan(box_prop)):
                    box_size_i += 1
                    box_prop = utils.get_interp_box(deepframe, box_size_i, i,
                                                    j, dimx)
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
    datafile : RampModel
        RampModel for a segment of the TSO.
    window : int
        Number of integrations to use for cosmic ray flagging. Must be odd.
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
    # Mask the detector reset artifact which is picked up by this flagging.
    int_start = datafile.meta.exposure.integration_start
    int_end = np.min([datafile.meta.exposure.integration_end, 256])
    # Artifact only affects first 256 integrations.
    artifact = np.zeros((nints, dimy, dimx)).astype(int)
    if int_start < 255:
        for j, jj in enumerate(range(int_start, int_end)):
            # j counts ints from start of this segment, jj is
            # integrations from start of exposure (1-indexed).
            # Mask rows from jj to jj+3 for detector reset
            # artifact.
            min_row = np.max([256 - (jj + 3), 0])
            max_row = np.min([(258 - jj), 256])
            artifact[j, min_row:max_row, :] = 1

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
        ii = (scale >= thresh) & (cube[:, g] > np.nanpercentile(cube, 10)) & (artifact == 0)

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


def oneoverfstep_nirspec(datafiles, output_dir=None, save_results=True,
                         pixel_masks=None, fileroots=None, mask_width=16,
                         centroids=None, method='median'):
    """Custom 1/f correction routine to be applied at the group level. The
    median level of each detector column is subtracted off while masking
    outlier pixels and the target trace.

    Parameters
    ----------
    datafiles : array-like[str], array-like[RampModel], array-like[CubeModel]
        List of paths to data files, or datamodels themselves for each segment
        of the TSO. Should be 4D ramps, but 3D rate files are also accepted.
    output_dir : str, None
        Directory to which to save results. Only necessary if saving results.
    save_results : bool
        If True, save results to disk.
    pixel_masks : array-like[str], None
        List of paths to maps of pixels to mask for each data segment. Can be
        3D (nints, dimy, dimx), or 2D (dimy, dimx).
    fileroots : list[str], None
        Root names for output files. Only necessary if saving results.
    mask_width : int
        Full width in pixels to mask around the trace.
    centroids : dict, None
        Dictionary containing trace positions for each order.
    method : str
        1/f correction method. Options are "median" or "slope".

    Returns
    -------
    results : list(CubeModel)
        RampModels for each segment, corrected for 1/f noise.
    """

    fancyprint('Starting 1/f correction step using the {} method.'
               ''.format(method))

    # If saving results, ensure output directory and fileroots are provided.
    if save_results is True:
        assert output_dir is not None
        assert fileroots is not None
        # Output directory formatting.
        if output_dir[-1] != '/':
            output_dir += '/'

    # Unpack data files into data cube.
    datafiles = np.atleast_1d(datafiles)
    for i, file in enumerate(datafiles):
        with utils.open_filetype(file) as currentfile:
            if i == 0:
                cube = currentfile.data
            else:
                cube = np.concatenate([cube, currentfile.data], axis=0)

    # Define the readout setup - can be 4D or 3D.
    if np.ndim(cube) == 4:
        nint, ngroup, dimy, dimx = np.shape(cube)
    else:
        nint, dimy, dimx = np.shape(cube)

    # Get the trace centroids.
    if centroids is None:
        # If no centroids file is provided, get the trace positions from the
        # data now.
        fancyprint('No centroids provided, locating trace positions.')
        # Create deepstack.
        if np.ndim(cube) == 4:
            # Only need last group.
            thiscube = cube[:, -1]
        else:
            thiscube = cube
        deepstack = bn.nanmedian(thiscube, axis=0)
        # Get detector to determine x limits.
        det = utils.get_detector_name(datafiles[0])
        if det == 'nrs1':
            xstart = 500
        else:
            xstart = 0
        centroids = utils.get_centroids_nirspec(deepstack, xstart=xstart,
                                                save_results=False)
        xpos, ypos = centroids[0], centroids[1]
    else:
        # Unpack centroids if provided.
        fancyprint('Unpacking centroids.')
        xpos, ypos = centroids['xpos'], centroids['ypos']

    # Read in the outlier maps - (nints, dimy, dimx) 3D cubes.
    if pixel_masks is None:
        fancyprint('No outlier maps passed, ignoring outliers.',
                   msg_type='WARNING')
        outliers = np.zeros((nint, dimy, dimx)).astype(bool)
    else:
        fancyprint('Constructing outlier map.')
        for i, badpix in enumerate(pixel_masks):
            # Create mask cubes.
            if i == 0:
                outliers = badpix.astype(bool)
            else:
                outliers = np.concatenate([outliers, badpix.astype(bool)])

    # Construct trace masks.
    fancyprint('Constructing trace mask.')
    low = np.max([np.zeros_like(ypos),
                  ypos - mask_width / 2], axis=0).astype(int)
    up = np.min([dimy * np.ones_like(ypos),
                 ypos + mask_width / 2], axis=0).astype(int)
    tracemask = np.zeros((dimy, dimx))
    for i, x in enumerate(xpos):
        tracemask[low[i]:up[i], int(x)] = 1
    # Add the trace mask to the outliers cube.
    outliers = (outliers.astype(bool) | tracemask.astype(bool)).astype(int)

    # Identify and mask any potential jumps that are not flagged.
    fancyprint('Flagging additional outliers.')
    if np.ndim(cube) == 4:
        thiscube = cube[:, -1]
    else:
        thiscube = cube
    cube_filt = medfilt(thiscube, (5, 1, 1))
    # Calculate the point-to-point scatter along the temporal axis.
    scatter = np.median(np.abs(0.5 * (thiscube[0:-2] + thiscube[2:]) -
                               thiscube[1:-1]), axis=0)
    scatter = np.where(scatter == 0, np.inf, scatter)
    # Find pixels which deviate more than 10 sigma.
    scale = np.abs(thiscube - cube_filt) / scatter
    ii = np.where(scale > 10)
    outliers[ii] = 1

    # The outlier map is 0 where good and >0 otherwise. As this
    # will be applied multiplicatively, replace 0s with 1s and
    # others with NaNs.
    outliers = np.where(outliers == 0, 1, np.nan)

    # Loop over all integrations to determine the 1/f noise level and correct
    # it.
    fancyprint('Starting full frame correction.')
    cube_corr = copy.deepcopy(cube)
    for i in tqdm(range(nint)):
        # Apply the outlier mask.
        cube[i] *= outliers[i, :, :]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            # Single 1/f scaling for all rows.
            dc = np.zeros_like(cube[i])
            # First method: simply take the column-wise median of all unmasked
            # pixels as the 1/f + background level.
            if method == 'median':
                # For group-level corrections.
                if np.ndim(cube == 4):
                    dc[:, :, :] = bn.nanmedian(cube[i], axis=1)[:, None, :]
                # For integration-level corrections.
                else:
                    dc[:, :] = bn.nanmedian(cube[i], axis=0)[None, :]

            # Second method: fit a linear slope to each column as subtract
            # that as the 1/f + background level.
            elif method == 'slope':
                # For group-level corrections.
                if np.ndim(cube == 4):
                    for xx in range(dimx):
                        xpos = np.arange(dimy)
                        ypos = cube[i][:, :, xx]
                        if np.all(np.isnan(ypos[-1])):
                            continue
                        xxpos = xpos[~np.isnan(ypos[-1])]
                        yypos = ypos[~np.isnan(ypos)].reshape(ngroup, len(xxpos))
                        pp = np.polyfit(xxpos, yypos.T, 1)
                        dc[:, :, xx] = np.polyval(pp, np.repeat(xpos[:, np.newaxis], ngroup, axis=1)).T
                # For integration-level corrections.
                else:
                    for xx in range(dimx):
                        xpos = np.arange(dimy)
                        ypos = cube[i][:, xx]
                        if np.all(np.isnan(ypos)):
                            continue
                        xxpos = xpos[~np.isnan(ypos)]
                        yypos = ypos[~np.isnan(ypos)]
                        pp = np.polyfit(xxpos, yypos, 1)
                        dc[:, xx] = np.polyval(pp, xpos)
            else:
                msg = 'Unrecognized 1/f method {}'.format(method)
                raise ValueError(msg)

        # Make sure no NaNs are in the DC map
        dc = np.where(np.isfinite(dc), dc, 0)
        # Subtract the 1/f map.
        cube_corr[i] -= dc

    results, current_int = [], 0
    # Save 1/f corrected data.
    for n, file in enumerate(datafiles):
        with utils.open_filetype(file) as thisfile:
            currentfile = thisfile.copy()
            nints = np.shape(currentfile.data)[0]
            currentfile.data = cube_corr[current_int:(current_int + nints)]
            current_int += nints
            if save_results is True:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    currentfile.write(
                        output_dir + fileroots[n] + 'oneoverfstep.fits')
            results.append(currentfile)

    return results


def oneoverfstep_scale(datafiles, baseline_ints, inner_mask_width=40,
                       outer_mask_width=70, even_odd_rows=True,
                       background=None, timeseries=None, timeseries_o2=None,
                       output_dir=None, save_results=True, pixel_masks=None,
                       fileroots=None, method='achromatic', centroids=None):
    """Custom 1/f correction routine to be applied at the group or
    integration level. A median stack is constructed using all out-of-transit
    integrations and subtracted from each individual integration. The
    column-wise median of this difference image is then subtracted from the
    original frame to correct 1/f noise. Outlier pixels, background
    contaminants, and the target trace itself can (should) be masked to
    improve the estimation.

    Parameters
    ----------
    datafiles : array-like[RampModel], array-like[CubeModel]
        List of datamodels for each segment of the TSO. Should be 4D ramps,
        but 3D rate files are also accepted.
    baseline_ints : array-like[int]
        Integration numbers of ingress and egress.
    inner_mask_width : int
        Width around the trace to mask. For windowed methods, defines the
        inner window edge.
    outer_mask_width : int
        For windowed methods, the outer edge of the window.
    even_odd_rows : bool
        If True, calculate 1/f noise seperately for even and odd numbered rows.
    background : array-like[float], None
        Model of background flux.
    timeseries : array-like[float], None
        Estimate of normalized light curve(s).
    timeseries_o2 : array-like[float], None
        Estimate of normalized light curve(s) for order 2. Only necessary if
        method is chromatic.
    output_dir : str, None
        Directory to which to save results. Only necessary if saving results.
    save_results : bool
        If True, save results to disk.
    pixel_masks : array-like[float], None
        Maps of pixels to mask for each data segment. Should be 3D (nints,
        dimy, dimx).
    fileroots : list[str], None
        Root names for output files. Only necessary if saving results.
    method : str
        Options are "chromatic", "achromatic", or "achromatic-window".
    centroids : dict, None
        Dictionary containing trace positions for each order.

    Returns
    -------
    results : list[CubeModel]
        RampModels for each segment, corrected for 1/f noise.
    """

    fancyprint('Starting 1/f correction step using the scale-{} '
               'method.'.format(method))

    # If saving results, ensure output directory and fileroots are provided.
    if save_results is True:
        assert output_dir is not None
        assert fileroots is not None
        # Output directory formatting.
        if output_dir[-1] != '/':
            output_dir += '/'

    # Ensure method is correct.
    if method not in ['chromatic', 'achromatic', 'achromatic-window']:
        msg = 'Method must be one of "chromatic", "achromatic", or ' \
              '"achromatic-window".'
        raise ValueError(msg)

    # Format the baseline frames.
    baseline_ints = utils.format_out_frames(baseline_ints)

    # Load in datamodels from all segments.
    datafiles = np.atleast_1d(datafiles)
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

    # Define the readout setup - can be 4D or 3D.
    if np.ndim(cube) == 4:
        nint, ngroup, dimy, dimx = np.shape(cube)
    else:
        nint, dimy, dimx = np.shape(cube)
    if dimy == 256:
        subarray = 'SUBSTRIP256'
    else:
        subarray = 'SUBSTRIP96'

    # Generate the 3D deep stack (ngroup, dimy, dimx) using only
    # baseline integrations.
    fancyprint('Generating a deep stack for each group using baseline '
               'integrations.')
    deepstack = utils.make_deepstack(cube[baseline_ints])

    # Get the trace centroids.
    if centroids is None:
        # If no centroids file is provided, get the trace positions from the
        # data now.
        fancyprint('No centroids provided, locating trace positions.')
        step = calwebb_spec2.extract_1d_step.Extract1dStep()
        tracetable = step.get_reference_file(datafiles[0], 'spectrace')
        if np.ndim(cube) == 4:
            thisdeep = deepstack[-1]
        else:
            thisdeep = deepstack
        centroids = utils.get_centroids_soss(thisdeep, tracetable, subarray,
                                             save_results=False)
        x1, y1 = centroids[0][0], centroids[0][1]
        x2, y2 = centroids[1][0], centroids[1][1]
        x3, y3 = centroids[2][0], centroids[2][1]
    else:
        # Unpack centroids if provided.
        fancyprint('Unpacking centroids.')
        x1, y1 = centroids['xpos'], centroids['ypos o1']
        x2, y2 = centroids['xpos'], centroids['ypos o2']
        x3, y3 = centroids['xpos'], centroids['ypos o3']

    # Read in the outlier maps - (nints, dimy, dimx) 3D cubes.
    if pixel_masks is None:
        fancyprint('No outlier maps passed, ignoring outliers.',
                   msg_type='WARNING')
        outliers1 = np.zeros((nint, dimy, dimx)).astype(bool)
    else:
        fancyprint('Constructing outlier map.')
        for i, badpix in enumerate(pixel_masks):
            # Create mask cubes.
            if i == 0:
                outliers1 = badpix.astype(bool)
            else:
                outliers1 = np.concatenate([outliers1, badpix.astype(bool)])
    outliers2 = np.copy(outliers1)

    # Construct trace masks.
    fancyprint('Constructing trace mask.')
    # Order 1 necessary for all subarrays.
    low_in1 = np.max([np.zeros_like(y1),
                      y1 - inner_mask_width / 2], axis=0).astype(int)
    up_in1 = np.min([dimy * np.ones_like(y1),
                     y1 + inner_mask_width / 2], axis=0).astype(int)
    low_out1 = np.max([np.zeros_like(y1),
                       y1 - outer_mask_width / 2], axis=0).astype(int)
    up_out1 = np.min([dimy * np.ones_like(y1),
                      y1 + outer_mask_width / 2], axis=0).astype(int)
    # Create inner and outer masks.
    mask1_in, mask1_out = np.zeros((dimy, dimx)), np.zeros((dimy, dimx))
    for i, x in enumerate(x1):
        mask1_in[low_in1[i]:up_in1[i], int(x)] = 1
        mask1_out[low_out1[i]:up_out1[i], int(x)] = 1

    # Include orders 2 and 3 for SUBSTRIP256.
    if subarray != 'SUBSTRIP96':
        # Order 2 -- inner and outer masks.
        low_in2 = np.max([np.zeros_like(y2),
                          y2 - inner_mask_width / 2], axis=0).astype(int)
        up_in2 = np.min([dimy * np.ones_like(y2),
                         y2 + inner_mask_width / 2], axis=0).astype(int)
        low_out2 = np.max([np.zeros_like(y2),
                           y2 - outer_mask_width / 2], axis=0).astype(int)
        up_out2 = np.min([dimy * np.ones_like(y2),
                          y2 + outer_mask_width / 2], axis=0).astype(int)
        mask2_in, mask2_out = np.zeros((dimy, dimx)), np.zeros((dimy, dimx))
        for i, x in enumerate(x2):
            mask2_in[low_in2[i]:up_in2[i], int(x)] = 1
            mask2_out[low_out2[i]:up_out2[i], int(x)] = 1
        # Order 3 -- only need inner mask.
        low3 = np.max([np.zeros_like(y3),
                       y3 - inner_mask_width / 2], axis=0).astype(int)
        up3 = np.min([dimy * np.ones_like(y3),
                      y3 + inner_mask_width / 2], axis=0).astype(int)
        mask3 = np.zeros((dimy, dimx))
        for i, x in enumerate(x3):
            mask3[low3[i]:up3[i], int(x)] = 1
    # But not for SUBSTRIP96.
    else:
        mask2_in = np.zeros_like(mask1_in)
        mask2_out = np.zeros_like(mask1_in)
        mask3 = np.zeros_like(mask1_in)

    # Add the appropriate trace mask to the outliers cube for the selected 1/f
    # method.
    tracemask = mask1_in.astype(bool) | mask2_in.astype(bool) | mask3.astype(bool)
    # For the scale-achromatic, just need to mask the cores of each trace,
    # defined by inner_mask_width.
    if method == 'achromatic':
        outliers1 = (outliers1 | tracemask).astype(int)
    # For the windowed corrections, construct a window around each order
    # defined by inner_mask_width and outer_mask_width.
    else:
        window1 = ~(mask1_out - mask1_in).astype(bool)
        window2 = ~(mask2_out - mask2_in).astype(bool)
        outliers1 = (outliers1 | window1 | tracemask).astype(int)
        outliers2 = (outliers2 | window2 | tracemask).astype(int)

    # Identify and mask any potential jumps that are not flagged.
    fancyprint('Flagging additional outliers.')
    if np.ndim(cube) == 4:
        thiscube = cube[:, -1]
    else:
        thiscube = cube
    cube_filt = medfilt(thiscube, (5, 1, 1))
    # Calculate the point-to-point scatter along the temporal axis.
    scatter = np.median(np.abs(0.5 * (thiscube[0:-2] + thiscube[2:]) -
                               thiscube[1:-1]), axis=0)
    scatter = np.where(scatter == 0, np.inf, scatter)
    # Find pixels which deviate more than 10 sigma.
    scale = np.abs(thiscube - cube_filt) / scatter
    ii = np.where(scale > 10)
    outliers1[ii] = 1
    outliers2[ii] = 1

    # The outlier map is 0 where good and >0 otherwise. As this
    # will be applied multiplicatively, replace 0s with 1s and
    # others with NaNs.
    if method in ['chromatic', 'achromatic-window']:
        outliers1 = np.where(outliers1 == 0, 1, np.nan)
        outliers2 = np.where(outliers2 == 0, 1, np.nan)
        # Also cut everything redder than ~0.9µm in order 2.
        outliers2[:, :, :1100] = np.nan
    else:
        outliers1 = np.where(outliers1 == 0, 1, np.nan)

    # In order to subtract off the trace as completely as possible, the median
    # stack must be scaled, via the transit curve, to the flux level of each
    # integration. This can be done via two methods: using the white light
    # curve (i.e., assuming the scaling is not wavelength dependent), or using
    # extracted 2D light curves, such that the scaling is wavelength dependent.
    # Get light curve. If not, estimate it (1D only) from data.
    if timeseries is None:
        if method == 'achromatic':
            fancyprint('No timeseries passed. It will be estimated from '
                       'current data.', msg_type='WARNING')
            # If no lightcurve is provided, estimate it from the current data.
            if np.ndim(cube) == 4:
                postage = cube[:, -1, 20:60, 1500:1550]
            else:
                postage = cube[:, 20:60, 1500:1550]
            timeseries = np.nansum(postage, axis=(1, 2))
            timeseries = timeseries / np.nanmedian(timeseries[baseline_ints])
            # Smooth the time series on a timescale of roughly 2%.
            timeseries = median_filter(timeseries,
                                       int(0.02 * np.shape(cube)[0]))
        else:
            msg = '2D light curves must be provided to use chromatic ' \
                  'method.'
            raise ValueError(msg)

    # If passed light curve is 1D, extend to 2D.
    if np.ndim(timeseries) == 1:
        # If 1D timeseries is passed cannot do chromatic correction.
        if method == 'chromatic':
            msg = '2D light curves are required for chromatic correction, ' \
                  'but 1D ones were passed.'
            raise ValueError(msg)
        else:
            timeseries = np.repeat(timeseries[:, np.newaxis], dimx, axis=1)
    # Get timeseries for order 2.
    if method == 'chromatic':
        if timeseries_o2 is None:
            msg = '2D light curves for order 2 must be provided to use ' \
                  'chromatic method.'
            raise ValueError(msg)
        if np.ndim(timeseries_o2) == 1:
            # If 1D timeseries is passed cannot do chromatic correction.
            msg = '2D light curves are required for chromatic correction,' \
                  ' but 1D ones were passed.'
            raise ValueError(msg)

    # Set up things that are needed for the 1/f correction with each method.
    if method == 'achromatic':
        # Orders to correct.
        orders = [1]
        # Pixel masks.
        outliers = [outliers1]
        # Timerseries.
        timeseries = [timeseries]
    elif method == 'achromatic-window':
        orders = [1, 2]
        outliers = [outliers1, outliers2]
        timeseries = [timeseries, timeseries]
    else:
        orders = [1, 2]
        outliers = [outliers1, outliers2]
        timeseries = [timeseries, timeseries_o2]

    # For chromatic or windowed corrections, need to treat order 1 and
    # order 2 seperately.
    cube_corr = copy.deepcopy(cube)
    for order, outlier, ts in zip(orders, outliers, timeseries):
        if method != 'achromatic':
            fancyprint('Starting order {} correction.'.format(order))
        else:
            fancyprint('Starting full frame correction.')

        # Loop over all integrations to determine the 1/f noise level via a
        # difference image, and correct it.
        for i in tqdm(range(nint)):
            # Create the difference image.
            if np.ndim(cube) == 4:
                sub = cube[i] - deepstack * ts[i, None, None, :]
            else:
                sub = cube[i] - deepstack * ts[i, None, :]
            # Apply the outlier mask.
            sub *= outlier[i, :, :]
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                if even_odd_rows is True:
                    # Calculate 1/f scaling seperately for even and odd rows.
                    dc = np.zeros_like(sub)
                    # For group-level corrections.
                    if np.ndim(cube) == 4:
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
                    if np.ndim(cube == 4):
                        dc[:, :, :] = bn.nanmedian(sub, axis=1)[:, None, :]
                    # For integration-level corrections.
                    else:
                        dc[:, :] = bn.nanmedian(sub, axis=0)[None, :]
            # Make sure no NaNs are in the DC map
            dc = np.where(np.isfinite(dc), dc, 0)
            # Subtract the 1/f map.
            if method == 'achromatic':
                # For achromatic method, just subtract the calculated 1/f
                # values from the whole frame.
                cube_corr[i] -= dc
            else:
                if order == 1:
                    # For order 1, subtract 1/f values from whole frame.
                    cube_corr[i] = cube_corr[i] - dc
                else:
                    # For order 2, subtract in a window around the trace.
                    cube_corr[i] = cube_corr[i] - dc * mask2_out[None, :, :]

        # Rebuild the cube and deepframe.
        if order == 1 and method != 'achromatic':
            deepstack = utils.make_deepstack(cube_corr[baseline_ints])
            cube = copy.deepcopy(cube_corr)

    # Background must be subtracted to accurately subtract off the target
    # trace and isolate 1/f noise. However, the background flux must also be
    # corrected for non-linearity. Therefore, it should be added back after
    # the 1/f is subtracted, in order to be re-subtracted later.
    # Note: only relevant for group-level corrections.
    if background is not None:
        # Add back the zodi background.
        cube_corr += background

    results, current_int = [], 0
    # Save 1/f corrected data.
    for n, file in enumerate(datafiles):
        with utils.open_filetype(file) as thisfile:
            currentfile = thisfile.copy()
            nints = np.shape(currentfile.data)[0]
            currentfile.data = cube_corr[current_int:(current_int + nints)]
            current_int += nints
            if save_results is True:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    currentfile.write(output_dir + fileroots[n] + 'oneoverfstep.fits')
            results.append(currentfile)

    return results


def oneoverfstep_solve(datafiles, baseline_ints, trace_width=70,
                       background=None, output_dir=None, save_results=True,
                       pixel_masks=None,  fileroots=None, do_plot=False,
                       show_plot=False, centroids=None):
    """Custom 1/f correction routine to be applied at the group or
    integration level. 1/f noise level and median frame scaling is calculated
    independently for each pixel column. Outlier pixels and background
    contaminants can (should) be masked to improve the estimation.

    Parameters
    ----------
    datafiles : array-like[RampModel], array-like[CubeModel]
        List of datamodels for each segment of the TSO. Should be 4D ramps,
        but 3D rate files are also accepted.
    baseline_ints : array-like[int]
        Integration numbers of ingress and egress.
    trace_width : int
        Defines the width around the trace to consider for MLE solving.
    background : array-like[float], None
        Model of background flux.
    output_dir : str, None
        Directory to which to save results. Only necessary if saving results.
    save_results : bool
        If True, save results to disk.
    pixel_masks : array-like[float], None
        Maps of pixels to mask for each data segment. Can be 3D (nints, dimy,
        dimx), or 2D (dimy, dimx).
    fileroots : list[str], None
        Root names for output files. Only necessary if saving results.
    do_plot : bool
        If True, do the step diagnostic plot.
    show_plot : bool
        If True, show the step diagnostic plot instead of/in addition to
        saving it to file.
    centroids : dict, None
        Dictionary containing trace positions for each order.

    Returns
    -------
    corrected_rampmodels : list(CubeModel)
        RampModels for each segment, corrected for 1/f noise.
    """

    fancyprint('Starting 1/f correction step using the solve method.')

    # If saving results, ensure output directory and fileroots are provided.
    if save_results is True:
        assert output_dir is not None
        assert fileroots is not None
        # Output directory formatting.
        if output_dir[-1] != '/':
            output_dir += '/'

    # Format the baseline frames.
    baseline_ints = utils.format_out_frames(baseline_ints)

    # Load in datamodels from all segments.
    for i, file in enumerate(datafiles):
        with utils.open_filetype(file) as currentfile:
            # FULL frame uses multiple amplifiers and probably has to be
            # treated differently. Break if we encounter a FULL frame exposure.
            if currentfile.meta.subarray.name == 'FULL':
                raise NotImplementedError
            # To create the deepstack, join all segments together.
            if i == 0:
                cube = np.copy(currentfile.data)
                if np.ndim(cube) == 4:
                    grpdq = currentfile.groupdq
                    pixdq = currentfile.pixeldq
                    dqcube = grpdq + pixdq
                else:
                    dqcube = currentfile.dq
            else:
                cube = np.concatenate([cube, currentfile.data], axis=0)
                if np.ndim(cube) == 4:
                    thisdq = currentfile.groupdq + currentfile.pixeldq
                    dqcube = np.concatenate([dqcube, thisdq], axis=0)
                else:
                    dqcube = np.concatenate([dqcube, currentfile.dq], axis=0)
    # Set errors to variance of data along integration axis.
    err1 = np.nanstd(cube[baseline_ints], axis=0)
    err1 = np.repeat(err1[np.newaxis], cube.shape[0], axis=0)
    err2 = copy.deepcopy(err1)

    # Define the readout setup - can be 4D (recommended) or 3D.
    if np.ndim(cube) == 4:
        nint, ngroup, dimy, dimx = np.shape(cube)
    else:
        nint, dimy, dimx = np.shape(cube)
        ngroup = 0
    if dimy == 256:
        subarray = 'SUBSTRIP256'
    else:
        subarray = 'SUBSTRIP96'

    # Get outlier masks.
    # Ideally, for this algorithm, we only want to consider pixels quite near
    # to the trace.
    if pixel_masks is None:
        fancyprint('No outlier maps passed, ignoring outliers.',
                   msg_type='WARNING')
        outliers1 = np.zeros((nint, dimy, dimx))
        outliers2 = np.zeros((nint, dimy, dimx))
    else:
        # Get bad pixel map.
        fancyprint('Creating outlier map.')
        for i, mask in enumerate(pixel_masks):
            if i == 0:
                outliers1 = mask
            else:
                outliers1 = np.concatenate([outliers1, mask])
        outliers2 = np.copy(outliers1)

    # Get the trace centroids.
    if centroids is None:
        # If no centroids are provided, get the trace positions from the data.
        fancyprint('No centroids provided, locating trace positions.')
        step = calwebb_spec2.extract_1d_step.Extract1dStep()
        tracetable = step.get_reference_file(datafiles[0], 'spectrace')
        # Get deepstack
        if np.ndim(cube) == 4:
            deepstack = utils.make_deepstack(cube[baseline_ints, -1])
        else:
            deepstack = utils.make_deepstack(cube[baseline_ints])
        centroids = utils.get_centroids_soss(deepstack, tracetable, subarray,
                                             save_results=False)
        x1, y1 = centroids[0][0], centroids[0][1]
        x2, y2 = centroids[1][0], centroids[1][1]
    else:
        # Unpack centroids if provided.
        fancyprint('Unpacking centroids.')
        x1, y1 = centroids['xpos'], centroids['ypos o1']
        x2, y2 = centroids['xpos'], centroids['ypos o2']

    # Construct trace masks.
    # Order 1 necessary for all subarrays.
    low1 = np.max([np.zeros_like(y1),
                   y1 - trace_width / 2], axis=0).astype(int)
    up1 = np.min([dimy * np.ones_like(y1),
                  y1 + trace_width / 2], axis=0).astype(int)
    # Create mask.
    mask1 = np.ones((dimy, dimx))
    for i, x in enumerate(x1):
        mask1[low1[i]:up1[i], int(x)] = 0

    # Include order 2 for SUBSTRIP256.
    if subarray != 'SUBSTRIP96':
        # Order 2.
        low2 = np.max([np.zeros_like(y2),
                       y2 - trace_width / 2], axis=0).astype(int)
        up2 = np.min([dimy * np.ones_like(y2),
                      y2 + trace_width / 2], axis=0).astype(int)
        mask2 = np.ones((dimy, dimx))
        for i, x in enumerate(x2):
            mask2[low2[i]:up2[i], int(x)] = 0
    # But not for SUBSTRIP96.
    else:
        mask2 = np.ones_like(mask1)
    # Add trace mask to outliers
    outliers1 = (outliers1.astype(bool) | mask1.astype(bool)).astype(int)
    outliers2 = (outliers2.astype(bool) | mask2.astype(bool)).astype(int)
    # Also mask O2 redwards of ~0.9µm (x<~1100).
    outliers2[:, :, :1100] = 1

    # Mask any pixels with non-zero dq flags.
    ii = np.where((dqcube != 0))
    err1[ii] = np.inf
    err2[ii] = np.inf
    err1[err1 == 0] = np.inf
    err2[err2 == 0] = np.inf
    # Apply the outlier mask.
    ii = np.where(outliers1 != 0)
    ii2 = np.where(outliers2 != 0)
    if ngroup == 0:
        err1[ii] = np.inf
        err2[ii2] = np.inf
    else:
        for g in range(ngroup):
            err1[:, g][ii] = np.inf
            err2[:, g][ii2] = np.inf

    # If no outlier masks were provided and correction is at group level, mask
    # detector reset artifact.
    if pixel_masks is None and np.ndim(cube) == 4:
        for j in range(256):
            # Mask rows from j to j+3 for detector reset artifact.
            max_row = np.min([(258 - j), 256])
            min_row = np.max([258 - (j + 4), 0])
            err1[j, min_row:max_row, :] = np.inf
            err2[j, min_row:max_row, :] = np.inf

    # Calculate 1/f noise using a wavelength-dependent scaling.
    for order, err in zip([1, 2], [err1, err2]):
        fancyprint('Starting order {}.'.format(order))
        # Don't do anything for order 2 if SUBSTRIP96.
        if order == 2 and dimy == 96:
            continue
        # Generate the 3D deep stack (ngroup, dimy, dimx) using only
        # baseline integrations.
        fancyprint('Generating a deep stack for each group using baseline '
                   'integrations.')
        deepstack = utils.make_deepstack(cube[baseline_ints])
        deepstack = np.repeat(deepstack[np.newaxis], cube.shape[0], axis=0)

        slopes_e, oofs_e, slopes_o, oofs_o = [], [], [], []
        if ngroup == 0:
            # Integration-level correction.
            oof = np.zeros_like(cube)
            # Mask any potential jumps.
            cube_filt = medfilt(cube, (5, 1, 1))
            # Calculate the point-to-point scatter along the temporal axis.
            scatter = np.median(np.abs(0.5 * (cube[0:-2] + cube[2:]) -
                                       cube[1:-1]), axis=0)
            scatter = np.where(scatter == 0, np.inf, scatter)
            # Find pixels which deviate more than 10 sigma.
            scale = np.abs(cube - cube_filt) / scatter
            ii = np.where(scale > 10)
            err[ii] = np.inf

            # Do the chromatic 1/f calculation.
            m_e, b_e, m_o, b_o = utils.line_mle(deepstack, cube, err)
            oof[:, ::2, :] = b_e[:, None, :]
            oof[:, 1::2, :] = b_o[:, None, :]
            scaling_e = m_e
            scaling_o = m_o
            slopes_e.append(m_e)
            slopes_o.append(m_o)
            oofs_e.append(b_e)
            oofs_o.append(b_o)
            plot_group = 1
        else:
            # Group-level correction.
            oof = np.zeros_like(cube)
            scaling_o = np.zeros((nint, ngroup, dimx))
            scaling_e = np.zeros((nint, ngroup, dimx))
            # Treat each group individually.
            for g in tqdm(range(ngroup)):
                # Mask any potential jumps.
                cube_filt = medfilt(cube[:, g], (5, 1, 1))
                # Calculate the point-to-point scatter along the temporal axis.
                scatter = np.median(np.abs(0.5 * (cube[0:-2, g] + cube[2:, g])
                                           - cube[1:-1, g]), axis=0)
                scatter = np.where(scatter == 0, np.inf, scatter)
                # Find pixels which deviate more than 10 sigma.
                scale = np.abs(cube[:, g] - cube_filt) / scatter
                ii = np.where(scale > 10)
                err[:, g][ii] = np.inf

                # Do the chromatic 1/f calculation.
                m_e, b_e, m_o, b_o = utils.line_mle(deepstack[:, g],
                                                    cube[:, g], err[:, g])
                oof[:, g, ::2] = b_e[:, None, :]
                oof[:, g, 1::2] = b_o[:, None, :]
                scaling_e[:, g] = m_e
                scaling_o[:, g] = m_o
                slopes_e.append(m_e)
                slopes_o.append(m_o)
                oofs_e.append(b_e)
                oofs_o.append(b_o)
            plot_group = ngroup

        # Replace any NaNs (that could happen if an entire column is masked)
        # with zeros.
        oof[np.isnan(oof)] = 0
        oof[np.isinf(oof)] = 0
        # Subtract the 1/f contribution.
        if order == 1:
            # For order 1, subtract the 1/f value from the whole column.
            cube_corr = cube - oof
            cube = copy.deepcopy(cube_corr)
        else:
            # For order 2, only subtract it from around the order 2 trace.
            trace2 = np.where(mask2 == 1, 0, 1)
            if ngroup != 0:
                cube_corr = cube_corr - oof * trace2[None, None, :, :]
            else:
                cube_corr = cube_corr - oof * trace2[None, :, :]

        # Do step plot if requested.
        if do_plot is True:
            outfile = output_dir + 'oneoverfstep_o{}_3.png'.format(order)
            plotting.make_oneoverf_chromatic_plot(slopes_e, slopes_o, oofs_e,
                                                  oofs_o, plot_group,
                                                  outfile=outfile,
                                                  show_plot=show_plot)
        # Also save 2D scaling.
        if save_results is True:
            outfile = output_dir + fileroots[0][:-12] + \
                '_nis_oofscaling_even_order{}.npy'.format(order)
            np.save(outfile, scaling_e)
            outfile = output_dir + fileroots[0][:-12] + \
                '_nis_oofscaling_odd_order{}.npy'.format(order)
            np.save(outfile, scaling_o)

    # Add back the zodi background.
    if background is not None:
        cube_corr += background

    # Save the corrected data.
    corrected_rampmodels = []
    current_int = 0
    for n, file in enumerate(datafiles):
        file = utils.open_filetype(file)
        nint = np.shape(file.data)[0]
        newfile = file.copy()
        file.close()
        newfile.data = cube_corr[current_int:(current_int+nint)]
        corrected_rampmodels.append(newfile)
        current_int += nint

        # Save the results if requested.
        if save_results is True:
            suffix = 'oneoverfstep.fits'
            newfile.write(output_dir + fileroots[n] + suffix)

    return corrected_rampmodels


def run_stage1(results, mode, soss_background_model=None, baseline_ints=None,
               oof_method='scale-achromatic', soss_timeseries=None,
               soss_timeseries_o2=None, save_results=True, pixel_masks=None,
               force_redo=False, deepframe=None, flag_up_ramp=False,
               rejection_threshold=15, flag_in_time=True,
               time_rejection_threshold=10, root_dir='./', output_tag='',
               skip_steps=None, do_plot=False, show_plot=False,
               soss_inner_mask_width=40, soss_outer_mask_width=70,
               centroids=None, nirspec_mask_width=16, **kwargs):
    """Run the exoTEDRF Stage 1 pipeline: detector level processing,
    using a combination of official STScI DMS and custom steps. Documentation
    for the official DMS steps can be found here:
    https://jwst-pipeline.readthedocs.io/en/latest/jwst/pipeline/calwebb_detector1.html

    Parameters
    ----------
    results : array-like(str)
        List of paths to input uncalibrated datafiles for all segments in an
        exposure.
    mode : str
        Instrument mode which produced the data being analyzed.
    soss_background_model : str, array-like(float), None
        SOSS background model or path to a file containing it.
    baseline_ints : array-like(int)
        Integration numbers for transit ingress and egress.
    oof_method : str
        1/f correction method. Options are "scale-chromatic",
        "scale-achromatic", "scale-achromatic-window", or "solve".
    soss_timeseries : array-like(float), str, None
        Estimate of the normalized light curve, either 1D or 2D, or path to
        a file containing it.
    soss_timeseries_o2 : array-like(float), str, None
        Estimate of the normalized light curve for order 2, either 1D or 2D,
        or path to a file containing it.
    save_results : bool
        If True, save results of each step to file.
    pixel_masks : array-like(str), None
        For improved 1/f noise corecton. List of paths to outlier maps for each
        data segment. Can be 3D (nints, dimy, dimx), or 2D (dimy, dimx) files.
    force_redo : bool
        If True, redo steps even if outputs files are already present.
    deepframe : str, None
        Path to deep stack, such as one produced by BadPixStep.
    flag_up_ramp : bool
        Whether to flag jumps up the ramp. This is the default flagging in the
        STScI pipeline. Note that this is broken as of jwst v1.12.5.
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
    skip_steps : list(str), None
        Step names to skip (if any).
    do_plot : bool
        If True, make step diagnostic plots.
    show_plot : bool
        Only necessary if do_plot is True. Show the diagnostic plots in
        addition to/instead of saving to file.
    soss_inner_mask_width : int
        For SOSS 1/f correction. For scale-achromatic, defines the width
        around the trace to mask. For windowed methods, defines the inner edge
        of the window.
    soss_outer_mask_width : int
        For SOSS 1/f correction. For windowed methods, defines the outer edge
        of the window. For solve, defines the width around the trace to use.
    centroids : str, None
        Path to file containing trace positions for each order.
    nirspec_mask_width : int
        Full-width (in pixels) around the target trace to mask for NIRSpec.

    Returns
    -------
    results : list[RampModel]
        Datafiles for each segment processed through Stage 1.
    """

    # ============== DMS Stage 1 ==============
    # Detector level processing.
    fancyprint('**Starting exoTEDRF Stage 1**')
    fancyprint('Detector level processing')

    if output_tag != '':
        output_tag = '_' + output_tag
    # Create output directories and define output paths.
    utils.verify_path(root_dir + 'pipeline_outputs_directory' + output_tag)
    outdir = root_dir + 'pipeline_outputs_directory' + output_tag + '/Stage1/'
    utils.verify_path(outdir)

    if skip_steps is None:
        skip_steps = []

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

    # ===== Reference Pixel Correction Step =====
    # Default DMS step.
    if 'RefPixStep' not in skip_steps:
        if 'RefPixStep' in kwargs.keys():
            step_kwargs = kwargs['RefPixStep']
        else:
            step_kwargs = {}
        step = RefPixStep(results, output_dir=outdir)
        results = step.run(save_results=save_results, force_redo=force_redo,
                           **step_kwargs)

    # ===== Dark Current Subtraction Step =====
    # Default DMS step.
    if 'DarkCurrentStep' not in skip_steps:
        if 'DarkCurrentStep' in kwargs.keys():
            step_kwargs = kwargs['DarkCurrentStep']
        else:
            step_kwargs = {}
        step = DarkCurrentStep(results, output_dir=outdir)
        results = step.run(save_results=save_results, force_redo=force_redo,
                           **step_kwargs)

    if 'OneOverFStep' not in skip_steps:
        if mode == 'NIRISS/SOSS':
            # ===== Background Subtraction Step =====
            # Custom DMS step - imported from Stage2.
            if 'BackgroundStep' in kwargs.keys():
                step_kwargs = kwargs['BackgroundStep']
            else:
                step_kwargs = {}
            step = stage2.BackgroundStep(results,
                                         baseline_ints=baseline_ints,
                                         background_model=soss_background_model,
                                         output_dir=outdir)
            results = step.run(save_results=save_results,
                               force_redo=force_redo, do_plot=do_plot,
                               show_plot=show_plot, **step_kwargs)
            results, soss_background_model = results

        # ===== 1/f Noise Correction Step =====
        # Custom DMS step.
        if 'OneOverFStep' in kwargs.keys():
            step_kwargs = kwargs['OneOverFStep']
        else:
            step_kwargs = {}
        step = OneOverFStep(results, output_dir=outdir,
                            baseline_ints=baseline_ints,
                            pixel_masks=pixel_masks, centroids=centroids,
                            soss_background=soss_background_model,
                            method=oof_method, soss_timeseries=soss_timeseries,
                            soss_timeseries_o2=soss_timeseries_o2)
        results = step.run(soss_inner_mask_width=soss_inner_mask_width,
                           soss_outer_mask_width=soss_outer_mask_width,
                           save_results=save_results, force_redo=force_redo,
                           do_plot=do_plot, show_plot=show_plot,
                           nirspec_mask_width=nirspec_mask_width,
                           **step_kwargs)

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
                           flag_up_ramp=flag_up_ramp,
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
