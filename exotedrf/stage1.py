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
    """Wrapper around default calwebb_detector1 Data Quality Initialization step with additional
    hot pixel flagging.
    """

    def __init__(self, input_data, output_dir, hot_pixel_map=None):
        """Step initializer.

        Parameters
        ----------
        input_data : array-like(str), array-like(datamodel)
            List of paths to input data or the input data itself.
        output_dir : str
            Path to directory to which to save outputs.
        hot_pixel_map : str, np.ndarray(float)
            Path to a custom hot pixel map, or the map itself.
        """

        # Set up easy attributes.
        self.tag = 'dqinitstep.fits'
        self.output_dir = output_dir

        # Unpack input data files.
        self.datafiles = utils.sort_datamodels(input_data)
        self.fileroots = utils.get_filename_root(self.datafiles)

        # Unpack deepframe.
        if isinstance(hot_pixel_map, str):
            # Want deepframe extension before bad pixel interpolation.
            fancyprint('Reading hot pixel map file: {}...'.format(hot_pixel_map))
            self.hotpix = np.load(hot_pixel_map).astype(bool)
        elif isinstance(hot_pixel_map, np.ndarray) or hot_pixel_map is None:
            self.hotpix = hot_pixel_map
        else:
            raise ValueError('Invalid type for hot_pixel_map: {}'.format(type(hot_pixel_map)))

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

        # Warn user that datamodels will be returned if not saving results.
        if save_results is False:
            fancyprint('Setting "save_results=False" can be memory intensive.', msg_type='WARNING')

        results = []
        all_files = glob.glob(self.output_dir + '*')
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
                res = step.call(segment, output_dir=self.output_dir, save_results=save_results,
                                **kwargs)
                # Flag additional hot pixels not in the default map.
                if self.hotpix is not None:
                    res = flag_hot_pixels(res, hot_pix=self.hotpix)[0]
                    # Overwite the previous edition.
                    res.save(expected_file)
                # Verify that filename is correct.
                if save_results is True:
                    current_name = self.output_dir + res.meta.filename
                    if expected_file != current_name:
                        res.close()
                        os.rename(current_name, expected_file)
                        thisfile = fits.open(expected_file)
                        thisfile[0].header['FILENAME'] = self.fileroots[i] + self.tag
                        thisfile.writeto(expected_file, overwrite=True)
                    res = expected_file
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
        self.datafiles = utils.sort_datamodels(input_data)
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

        # Warn user that datamodels will be returned if not saving results.
        if save_results is False:
            fancyprint('Setting "save_results=False" can be memory intensive.', msg_type='WARNING')

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
                res = step.call(segment, output_dir=self.output_dir, save_results=save_results,
                                **kwargs)
                # Verify that filename is correct.
                if save_results is True:
                    current_name = self.output_dir + res.meta.filename
                    if expected_file != current_name:
                        res.close()
                        os.rename(current_name, expected_file)
                        thisfile = fits.open(expected_file)
                        thisfile[0].header['FILENAME'] = self.fileroots[i] + self.tag
                        thisfile.writeto(expected_file, overwrite=True)
                    res = expected_file
            results.append(res)

        return results


class SuperBiasStep:
    """Wrapper around default calwebb_detector1 Super Bias Subtraction step with some custom
    modifications.
    """

    def __init__(self, input_data, output_dir, centroids=None, method='crds'):
        """Step initializer.

        Parameters
        ----------
        input_data : array-like(str), array-like(datamodel)
            List of paths to input data or the input data itself.
        output_dir : str
            Path to directory to which to save outputs.
        centroids : str, dict, None
            Path to file containing trace positions for each order or the centroids dictionary
            itself.
        method : str
            Method via which to calculate the superbias level. Options are 'crds', 'custom', or
            'custom-rescale'. NIRSpec only.
        """

        # Set up easy attributes.
        self.tag = 'superbiasstep.fits'
        self.output_dir = output_dir
        self.method = method

        # Unpack input data files.
        self.datafiles = utils.sort_datamodels(input_data)
        self.fileroots = utils.get_filename_root(self.datafiles)
        self.fileroot_noseg = utils.get_filename_root_noseg(self.fileroots)

        # Unpack centroids.
        if isinstance(centroids, str):
            fancyprint('Reading centroids file: {}...'.format(centroids))
            self.centroids = pd.read_csv(centroids, comment='#')
        else:
            self.centroids = centroids

        # Get instrument.
        self.instrument = utils.get_instrument_name(self.datafiles[0])

        # Make sure instrument is compatible with the superbias method.
        if self.instrument == 'NIRISS' and method != 'crds':
            fancyprint('Custom bias subtractions are not available for {} observations. Changing '
                       'method to crds'.format(self.instrument), msg_type='WARNING')
            self.method = 'crds'

    def run(self, save_results=True, force_redo=False, do_plot=False, show_plot=False, **kwargs):
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
            Keyword arguments for calwebb_detector1.superbias_step.SuperBiasStep.

        Returns
        -------
        results : list(datamodel)
            Input data files processed through the step.
        """

        # Warn user that datamodels will be returned if not saving results.
        if save_results is False:
            fancyprint('Setting "save_results=False" can be memory intensive.', msg_type='WARNING')

        # If not using the crds reference file, make the custom superbias now.
        if self.method != 'crds':
            fancyprint('Generating a custom superbias frame from 0th group.')
            superbias = utils.make_custom_superbias(self.datafiles)
            # Save superbias frame.
            if save_results is True:
                filename = self.fileroot_noseg + 'superbias.npy'
                np.save(self.output_dir + filename, superbias)

        results = []
        first_time = True
        all_files = glob.glob(self.output_dir + '*')
        for i, segment in enumerate(self.datafiles):
            # If an output file for this segment already exists, skip the step.
            expected_file = self.output_dir + self.fileroots[i] + self.tag
            if expected_file in all_files and force_redo is False:
                fancyprint('File {} already exists.'.format(expected_file))
                fancyprint('Skipping Superbias Subtraction Step.')
                res = expected_file
                do_plot, show_plot = False, False
            # If no output files are detected, run the step.
            else:
                # To subtract the default crds superbias reference file.
                if self.method == 'crds':
                    step = calwebb_detector1.superbias_step.SuperBiasStep()
                    res = step.call(segment, output_dir=self.output_dir, save_results=save_results,
                                    **kwargs)
                # To calculate the superbias level directly from the data.
                else:
                    if self.method == 'custom':
                        method = 'constant'
                    else:
                        method = 'rescale'
                    res = subtract_custom_superbias(segment, superbias, method=method,
                                                    centroids=self.centroids,
                                                    output_dir=self.output_dir,
                                                    save_results=save_results,
                                                    fileroot=self.fileroots[i], **kwargs)
                    res, scale_factor = res
                    # For rescaling method, want to plot the timeseries of scale factors. So keep
                    # track of this.
                    if method == 'rescale':
                        if first_time is True:
                            scale_factors = scale_factor
                            first_time = False
                        else:
                            scale_factors = np.concatenate([scale_factors, scale_factor])

                # Verify that filename is correct.
                if save_results is True:
                    if isinstance(res, str):
                        current_name = res
                    else:
                        current_name = self.output_dir + res.meta.filename
                    if expected_file != current_name:
                        res.close()
                        os.rename(current_name, expected_file)
                        thisfile = fits.open(expected_file)
                        thisfile[0].header['FILENAME'] = self.fileroots[i] + self.tag
                        thisfile.writeto(expected_file, overwrite=True)
                    res = expected_file
            results.append(res)

        # Do step plot if requested.
        if do_plot is True:
            if save_results is True:
                plot_file1 = self.output_dir + self.tag.replace('.fits', '_1.png')
                plot_file2 = self.output_dir + self.tag.replace('.fits', '_2.png')
                if self.instrument == 'NIRSPEC':
                    det = utils.get_nrs_detector_name(self.datafiles[0])
                    plot_file1 = plot_file1.replace('.png', '_{}.png'.format(det))
                    plot_file2 = plot_file2.replace('.png', '_{}.png'.format(det))
            else:
                plot_file1, plot_file2 = None, None
            plotting.make_superbias_plot(results, outfile=plot_file1, show_plot=show_plot)
            if self.method == 'custom-rescale':
                plotting.make_superbias_scale_plot(scale_factors, outfile=plot_file2,
                                                   show_plot=show_plot)

        return results


class RefPixStep:
    """Wrapper around default calwebb_detector1 Reference Pixel Correction step.
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
        self.datafiles = utils.sort_datamodels(input_data)
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

        # Warn user that datamodels will be returned if not saving results.
        if save_results is False:
            fancyprint('Setting "save_results=False" can be memory intensive.', msg_type='WARNING')

        results = []
        all_files = glob.glob(self.output_dir + '*')
        for i, segment in enumerate(self.datafiles):
            # If an output file for this segment already exists, skip the step.
            expected_file = self.output_dir + self.fileroots[i] + self.tag
            if expected_file in all_files and force_redo is False:
                fancyprint('File {} already exists.'.format(expected_file))
                fancyprint('Skipping Reference Pixel Correction Step.')
                res = expected_file
            # If no output files are detected, run the step.
            else:
                step = calwebb_detector1.refpix_step.RefPixStep()
                res = step.call(segment, output_dir=self.output_dir, save_results=save_results,
                                **kwargs)
                # Verify that filename is correct.
                if save_results is True:
                    current_name = self.output_dir + res.meta.filename
                    if expected_file != current_name:
                        res.close()
                        os.rename(current_name, expected_file)
                        thisfile = fits.open(expected_file)
                        thisfile[0].header['FILENAME'] = self.fileroots[i] + self.tag
                        thisfile.writeto(expected_file, overwrite=True)
                    res = expected_file
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
        self.datafiles = utils.sort_datamodels(input_data)
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
            Keyword arguments for calwebb_detector1.dark_current_step.DarkCurrentStep.

        Returns
        -------
        results : list(datamodel)
            Input data files processed through the step.
        """

        # Warn user that datamodels will be returned if not saving results.
        if save_results is False:
            fancyprint('Setting "save_results=False" can be memory intensive.', msg_type='WARNING')

        results = []
        all_files = glob.glob(self.output_dir + '*')
        for i, segment in enumerate(self.datafiles):
            # If an output file for this segment already exists, skip the step.
            expected_file = self.output_dir + self.fileroots[i] + self.tag
            if expected_file in all_files and force_redo is False:
                fancyprint('File {} already exists.'.format(expected_file))
                fancyprint('Skipping Dark Current Subtraction Step.')
                res = expected_file
            # If no output files are detected, run the step.
            else:
                step = calwebb_detector1.dark_current_step.DarkCurrentStep()
                res = step.call(segment, output_dir=self.output_dir, save_results=save_results,
                                **kwargs)
                # Verify that filename is correct.
                if save_results is True:
                    current_name = self.output_dir + res.meta.filename
                    if expected_file != current_name:
                        res.close()
                        os.rename(current_name, expected_file)
                        thisfile = fits.open(expected_file)
                        thisfile[0].header['FILENAME'] = self.fileroots[i] + self.tag
                        thisfile.writeto(expected_file, overwrite=True)
                    res = expected_file
            results.append(res)

        return results


class OneOverFStep:
    """Wrapper around custom 1/f Correction Step.
    """

    def __init__(self, input_data, output_dir, baseline_ints=None, pixel_masks=None, centroids=None,
                 soss_background=None, method='scale-achromatic', soss_timeseries=None,
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
            Correction method. SOSS options are "scale-chromatic", "scale-achromatic",
            "scale-achromatic-window", or "solve". NIRSpec options are "median" or "slope".
        soss_timeseries : np.ndarray(float), str, None
            Path to a file containing light curve(s) for order 1, or the light curve(s) themselves.
        soss_timeseries_o2 : np.ndarray(float), str, None
            Path to a file containing light curves for order 2, or the light curves themselves.
        pixel_masks : array-like(str), np.ndarray(float), None
            List of paths to maps of pixels to mask for each data segment or the masks themselves.
            Should be 3D (nints, dimy, dimx).
        soss_background : np.ndarray(float), str, None
            Model of background flux.
        centroids : str, dict, None
            Path to file containing trace positions for each order or the centroids dictionary
            itself.
        """

        # Set up easy attributes.
        self.tag = 'oneoverfstep.fits'
        self.output_dir = output_dir
        self.method = method

        # Unpack input data files.
        self.datafiles = utils.sort_datamodels(input_data)
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
            raise ValueError('Invalid type for timeseries: {}'.format(type(soss_timeseries)))

        # Unpack timeseries for order 2.
        if isinstance(soss_timeseries_o2, str):
            fancyprint('Reading order 2 timeseries file: {}...'.format(soss_timeseries_o2))
            self.timeseries_o2 = np.load(soss_timeseries_o2)
        elif (isinstance(soss_timeseries_o2, np.ndarray) or
              soss_timeseries_o2 is None):
            self.timeseries_o2 = soss_timeseries_o2
        else:
            raise ValueError('Invalid type for timeseries_o2: {}'.format(type(soss_timeseries_o2)))

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
                    raise ValueError('Invalid type for pixel_masks: {}'.format(type(mask)))
            assert len(self.pixel_masks) == len(self.datafiles)
        else:
            self.pixel_masks = pixel_masks

        # Unpack background.
        if isinstance(soss_background, str):
            fancyprint('Reading background file: {}...'.format(soss_background))
            self.background = np.load(soss_background)
        elif (isinstance(soss_background, np.ndarray) or
              soss_background is None):
            self.background = soss_background
        else:
            raise ValueError('Invalid type for background: {}'.format(type(soss_background)))

        # Get instrument.
        self.instrument = utils.get_instrument_name(self.datafiles[0])
        if self.instrument == 'NIRISS':
            assert baseline_ints is not None
            self.baseline_ints = baseline_ints
        # Some attributes that are not needed for NIRSpec, but expected for plotting purposes.
        if self.instrument == 'NIRSPEC':
            if isinstance(self.datafiles[0], str):
                nint = fits.getheader(self.datafiles[0])['NINTS']
            else:
                nint = self.datafiles[0].meta.exposure.nints
            self.smoothed_wlc = np.ones(nint)
            self.baseline_ints = int(0.25 * nint)
            # Default to median method for NIRSpec.
            if self.method == 'scale-achromatic':
                self.method = 'median'

    def run(self, soss_inner_mask_width=40, soss_outer_mask_width=70, nirspec_mask_width=16,
            smoothing_scale=None, save_results=True, force_redo=False, do_plot=False,
            show_plot=False, **kwargs):
        """Method to run the step.

        Parameters
        ----------
        soss_inner_mask_width : int
            Inner full-width (in pixels) around the target trace to mask for SOSS.
        soss_outer_mask_width : int
            Outer full-width (in pixels) around the target trace to mask for SOSS.
        nirspec_mask_width : int
            Full-width (in pixels) around the target trace to mask for NIRSpec.
        smoothing_scale : int, None
            If no timseries is provided, the scale (in number of integrations) on which to smooth
            the self-extracted timseries.
        save_results : bool
            If True, save results.
        force_redo : bool
            If True, run step even if output files are detected.
        do_plot : bool
            If True, do step diagnostic plot.
        show_plot : bool
            If True, show the step diagnostic plot.
        kwargs : dict
            Keyword arguments for stage1.oneoverfstep_scale, stage1.oneoverfstep_solve, or
            stage1.oneoverfstep_nirspec.

        Returns
        -------
        results : list(datamodel), list(str)
            Input data files processed through the step.
        """

        # Warn user that datamodels will be returned if not saving results.
        if save_results is False:
            fancyprint('Setting "save_results=False" can be memory intensive.', msg_type='WARNING')

        fancyprint('OneOverFStep instance created.')

        all_files = glob.glob(self.output_dir + '*')
        results = []
        first_time = True
        for i, segment in enumerate(self.datafiles):
            # If an output file for this segment already exists, skip the step.
            expected_file = self.output_dir + self.fileroots[i] + self.tag
            if expected_file in all_files and force_redo is False:
                fancyprint('File {} already exists.'.format(expected_file))
                fancyprint('Skipping 1/f Correction Step.')
                res = expected_file
                # Do not do plots if skipping step.
                do_plot, show_plot = False, False
            # If no output files are detected, run the step.
            else:
                # Get outlier mask for the current segment.
                thismask = None
                if self.pixel_masks is not None:
                    thismask = self.pixel_masks[i]

                # Generate some necessary quantities -- only do this for the first segment.
                if first_time:
                    # Quantity #1: deep stack.
                    fancyprint('Creating reference deep stack.')
                    deepstack = utils.make_baseline_stack_general(datafiles=self.datafiles,
                                                                  baseline_ints=self.baseline_ints)

                    # Quantity #2: centroids (if not provided).
                    if self.centroids is None:
                        fancyprint('No centroids provided, locating trace positions.')
                        self.centroids = {}
                        if self.instrument == 'NIRISS':
                            # Define the readout setup.
                            subarray = utils.get_soss_subarray(self.datafiles[0])
                            step = calwebb_spec2.extract_1d_step.Extract1dStep()
                            tracetable = step.get_reference_file(self.datafiles[0], 'spectrace')
                            if np.ndim(deepstack) == 3:
                                thisdeep = deepstack[-1]
                            else:
                                thisdeep = deepstack
                            cens = utils.get_centroids_soss(thisdeep, tracetable, subarray,
                                                            save_results=False)
                            self.centroids['xpos'] = cens[0][0]
                            self.centroids['ypos o1'] = cens[0][1]
                            self.centroids['ypos o2'] = cens[1][1]
                            self.centroids['ypos o3'] = cens[2][1]
                        else:
                            # Get detector to determine x limits.
                            det = utils.get_nrs_detector_name(self.datafiles[0])
                            if det == 'nrs1':
                                xstart = 500
                            else:
                                xstart = 0
                            cens = utils.get_centroids_nirspec(deepstack, xstart=xstart,
                                                               save_results=False)
                            self.centroids['xpos'], self.centroids['ypos'] = cens[0], cens[1]

                    # Quantity #3: storage arrays for NIRISS solving method.
                    if self.method == 'solve':
                        mle_results = []

                    first_time = False

                # Start the corrections.
                if self.instrument == 'NIRISS':
                    # Trim timeseries to match integrations of current segment.
                    thistso, thistso_o2 = None, None
                    if self.timeseries is not None:
                        # For fits file inputs.
                        if isinstance(segment, str):
                            with fits.open(segment)as file:
                                istart = file[0].header['INTSTART'] - 1
                                iend = file[0].header['INTEND']
                        # For datamodel inputs to not break jwst pipeline.
                        else:
                            with utils.open_filetype(segment) as file:
                                istart = file.meta.exposure.integration_start - 1
                                iend = file.meta.exposure.integration_end
                        thistso = self.timeseries[istart:iend]
                    if self.timeseries_o2 is not None:
                        thistso_o2 = self.timeseries_o2[istart:iend]

                    if self.method in ['scale-chromatic', 'scale-achromatic',
                                       'scale-achromatic-window']:
                        # To use "reference files" to calculate 1/f noise.
                        method = self.method.split('scale-')[-1]
                        res = oneoverfstep_scale(segment, deepstack=deepstack,
                                                 inner_mask_width=soss_inner_mask_width,
                                                 outer_mask_width=soss_outer_mask_width,
                                                 background=self.background,
                                                 timeseries=thistso, timeseries_o2=thistso_o2,
                                                 output_dir=self.output_dir,
                                                 save_results=save_results, pixel_mask=thismask,
                                                 fileroot=self.fileroots[i], method=method,
                                                 centroids=self.centroids,
                                                 smoothing_scale=smoothing_scale, **kwargs)
                    elif self.method == 'solve':
                        # To use MLE to solve for the 1/f noise.
                        res = oneoverfstep_solve(datafile=segment, deepstack=deepstack,
                                                 trace_width=soss_outer_mask_width,
                                                 background=self.background,
                                                 output_dir=self.output_dir,
                                                 save_results=save_results,
                                                 pixel_mask=thismask, fileroot=self.fileroots[i],
                                                 centroids=self.centroids)
                        res, calc_vals = res
                        mle_results.append(calc_vals)
                    else:
                        # Raise error otherwise.
                        raise ValueError('Unrecognized 1/f correction: {}'.format(self.method))
                else:
                    if self.method not in ['median', 'slope']:
                        # Raise error for bad method.
                        raise ValueError('Unrecognized 1/f correction: {}'.format(self.method))

                    res = oneoverfstep_nirspec(segment, output_dir=self.output_dir,
                                               save_results=save_results, pixel_mask=thismask,
                                               fileroot=self.fileroots[i],
                                               mask_width=nirspec_mask_width,
                                               centroids=self.centroids, method=self.method)
            results.append(res)

        # Save 2D scaling determined by solving method.
        if save_results is True and self.method == 'solve':
            fancyprint('Saving MLE-determined light curves.')

            for o in range(len(mle_results[0].keys())):
                order = o + 1
                for s in range(len(mle_results)):
                    if s == 0:
                        scale_e = mle_results[s]['o{}'.format(order)]['scale_e']
                        scale_o = mle_results[s]['o{}'.format(order)]['scale_o']
                    else:
                        this_e = mle_results[s]['o{}'.format(order)]['scale_e']
                        this_o = mle_results[s]['o{}'.format(order)]['scale_o']
                        scale_e = np.concatenate([scale_e, this_e])
                        scale_o = np.concatenate([scale_o, this_o])

                outfile_e = (self.output_dir + self.fileroots[0][:-12] +
                             '_nis_oofscaling_even_order{}.npy'.format(order))
                np.save(outfile_e, scale_e)
                outfile_o = (self.output_dir + self.fileroots[0][:-12] +
                             '_nis_oofscaling_odd_order{}.npy'.format(order))
                np.save(outfile_o, scale_o)
                fancyprint('Light curves saved to files {0} and {1}'.format(outfile_e, outfile_o))

        # Do step plots if requested.
        if do_plot is True:
            if save_results is True:
                plot_file1 = self.output_dir + self.tag.replace('.fits', '_1.png')
                plot_file2 = self.output_dir + self.tag.replace('.fits', '_2.png')
                plot_file3 = self.output_dir + self.tag.replace('.fits', '_o1_3.png')
                if self.instrument == 'NIRSPEC':
                    det = utils.get_nrs_detector_name(self.datafiles[0])
                    plot_file1 = plot_file1.replace('_1.png', '_1_{}.png'.format(det))
                    plot_file2 = plot_file2.replace('_2.png', '_2_{}.png'.format(det))
            else:
                plot_file1, plot_file2, plot_file3 = None, None, None

            # Make sure we have the correct frame time.
            if self.instrument == 'NIRSPEC':
                tframe = 0.902
            else:
                tframe = 5.494

            # For scale-chromatic correction, collapse 2D timeseries to 1D for plotting purposes.
            if self.method == 'scale-chromatic':
                this_ts = np.nanmedian(self.timeseries, axis=1)
            else:
                this_ts = self.timeseries

            # Make a deep stack of corrected obervations.
            deepstack_new = utils.make_baseline_stack_general(results, self.baseline_ints)
            plotting.make_oneoverf_plot(results, timeseries=this_ts, deepstack=deepstack_new,
                                        outfile=plot_file1, show_plot=show_plot)
            plotting.make_oneoverf_psd(results, self.datafiles, timeseries=this_ts,
                                       deepstack=deepstack_new, old_deepstack=deepstack,
                                       pixel_masks=self.pixel_masks, outfile=plot_file2,
                                       show_plot=show_plot, tframe=tframe)

            # Plot MLE results if solving method was used.
            if self.method == 'solve':
                for o in range(len(mle_results[0].keys())):
                    order = o + 1
                    # Unpack the additive and multiplicative factors from the
                    # MLE.
                    for s in range(len(mle_results)):
                        if s == 0:
                            slopes_e = mle_results[s]['o{}'.format(order)]['scale_e']
                            slopes_o = mle_results[s]['o{}'.format(order)]['scale_o']
                            if np.ndim(deepstack) == 3:
                                oofs_e = mle_results[s]['o{}'.format(order)]['oof'][:, :, 10]
                                oofs_o = mle_results[s]['o{}'.format(order)]['oof'][:, :, 11]
                            else:
                                oofs_e = mle_results[s]['o{}'.format(order)]['oof'][:, 10]
                                oofs_o = mle_results[s]['o{}'.format(order)]['oof'][:, 11]
                        else:
                            this_e = mle_results[s]['o{}'.format(order)]['scale_e']
                            this_o = mle_results[s]['o{}'.format(order)]['scale_o']
                            slopes_e = np.concatenate([slopes_e, this_e])
                            slopes_o = np.concatenate([slopes_o, this_o])
                            if np.ndim(deepstack) == 3:
                                this_e = mle_results[s]['o{}'.format(order)]['oof'][:, :, 10]
                                this_o = mle_results[s]['o{}'.format(order)]['oof'][:, :, 11]
                            else:
                                this_e = mle_results[s]['o{}'.format(order)]['oof'][:, 10]
                                this_o = mle_results[s]['o{}'.format(order)]['oof'][:, 11]
                            oofs_e = np.concatenate([oofs_e, this_e])
                            oofs_o = np.concatenate([oofs_o, this_o])
                    if order == 2:
                        plot_file3 = plot_file3[:-7] + '2_3.png'
                    if np.ndim(deepstack) == 3:
                        plot_group = np.shape(deepstack)[0]
                    else:
                        plot_group = 1
                    plotting.make_oneoverf_chromatic_plot(slopes_e, slopes_o, oofs_e, oofs_o,
                                                          plot_group, outfile=plot_file3,
                                                          show_plot=show_plot)

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
        self.datafiles = utils.sort_datamodels(input_data)
        self.fileroots = utils.get_filename_root(self.datafiles)

        # Get instrument.
        self.instrument = utils.get_instrument_name(self.datafiles[0])

    def run(self, save_results=True, force_redo=False, do_plot=False, show_plot=False, **kwargs):
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
            Keyword arguments for calwebb_detector1.linearity_step.LinearityStep.

        Returns
        -------
        results : list(datamodel)
            Input data files processed through the step.
        """

        # Warn user that datamodels will be returned if not saving results.
        if save_results is False:
            fancyprint('Setting "save_results=False" can be memory intensive.', msg_type='WARNING')

        results = []
        all_files = glob.glob(self.output_dir + '*')
        for i, segment in enumerate(self.datafiles):
            # If an output file for this segment already exists, skip the step.
            expected_file = self.output_dir + self.fileroots[i] + self.tag
            if expected_file in all_files and force_redo is False:
                fancyprint('File {} already exists.'.format(expected_file))
                fancyprint('Skipping Linearity Correction Step.')
                res = expected_file
                do_plot, show_plot = False, False
            # If no output files are detected, run the step.
            else:
                step = calwebb_detector1.linearity_step.LinearityStep()
                res = step.call(segment, output_dir=self.output_dir, save_results=save_results,
                                **kwargs)
                # Verify that filename is correct.
                if save_results is True:
                    current_name = self.output_dir + res.meta.filename
                    if expected_file != current_name:
                        res.close()
                        os.rename(current_name, expected_file)
                        thisfile = fits.open(expected_file)
                        thisfile[0].header['FILENAME'] = self.fileroots[i] + self.tag
                        thisfile.writeto(expected_file, overwrite=True)
                    res = expected_file
            results.append(res)
        # Do step plot if requested.
        if do_plot is True:
            if save_results is True:
                plot_file1 = self.output_dir + self.tag.replace('.fits', '_1.png')
                plot_file2 = self.output_dir + self.tag.replace('.fits', '_2.png')
                if self.instrument == 'NIRSPEC':
                    det = utils.get_nrs_detector_name(self.datafiles[0])
                    plot_file1 = plot_file1.replace('.png', '_{}.png'.format(det))
                    plot_file2 = plot_file2.replace('.png', '_{}.png'.format(det))
            else:
                plot_file1, plot_file2 = None, None
            plotting.make_linearity_plot(results, self.datafiles, outfile=plot_file1,
                                         show_plot=show_plot)
            plotting.make_linearity_plot2(results, self.datafiles, outfile=plot_file2,
                                          show_plot=show_plot)

        return results


class JumpStep:
    """Wrapper around default calwebb_detector1 Jump Detection step with some custom modifications.
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
        self.datafiles = utils.sort_datamodels(input_data)
        self.fileroots = utils.get_filename_root(self.datafiles)

        # Get instrument.
        self.instrument = utils.get_instrument_name(self.datafiles[0])

    def run(self, save_results=True, force_redo=False, flag_up_ramp=False,
            rejection_threshold=15, flag_in_time=True, time_rejection_threshold=10, time_window=5,
            do_plot=False, show_plot=False, **kwargs):
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
            Sigma threshold for an outlier to be considered a jump for up-the-ramp flagging.
        flag_in_time : bool
            If True, run time-domain flagging.
        time_rejection_threshold : int
            Sigma threshold for an outlier to be considered a jump for time-domain flagging.
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
                results.append(expected_file)
                do_plot, show_plot = False, False
            # If no output files are detected, proceed.
            else:
                # Get number of groups in the observation --- ngroup=2 must be treated in a special
                # way as the default pipeline JumpStep will fail.
                # Also need to set minimum_sigclip_groups to something >nints, else the up-the-ramp
                # jump detection will be replaced by a time-domain sigma clipping.
                if isinstance(self.datafiles[0], str):
                    ngroups = fits.getheader(self.datafiles[0], 0)['NGROUPS']
                else:
                    with datamodels.open(self.datafiles[0]) as file:
                        ngroups = file.meta.exposure.ngroups
                # For ngroup > 2, default JumpStep can be used.
                if ngroups > 2 and flag_up_ramp is True:
                    step = calwebb_detector1.jump_step.JumpStep()
                    res = step.call(segment, output_dir=self.output_dir, save_results=save_results,
                                    rejection_threshold=rejection_threshold,
                                    maximum_cores='quarter', minimum_sigclip_groups=1e6, **kwargs)
                # Time domain jump step must be run for ngroup=2.
                else:
                    res = segment
                    flag_in_time = True
                # Do time-domain flagging.
                if flag_in_time is True:
                    res = jumpstep_in_time(res, window=time_window, thresh=time_rejection_threshold,
                                           fileroot=self.fileroots[i], save_results=save_results,
                                           output_dir=self.output_dir)
                # Verify that filename is correct.
                if save_results is True:
                    if isinstance(res, str):
                        current_name = res
                    else:
                        current_name = self.output_dir + res.meta.filename
                    if expected_file != current_name:
                        res.close()
                        os.rename(current_name, expected_file)
                        thisfile = fits.open(expected_file)
                        thisfile[0].header['FILENAME'] = self.fileroots[i] + self.tag
                        thisfile.writeto(expected_file, overwrite=True)
                    res = expected_file
                results.append(res)

        # Do step plot if requested.
        if do_plot is True:
            if save_results is True:
                plot_file = self.output_dir + self.tag.replace('.fits', '.png')
                if self.instrument == 'NIRSPEC':
                    det = utils.get_nrs_detector_name(self.datafiles[0])
                    plot_file = plot_file.replace('.png', '_{}.png'.format(det))
            else:
                plot_file = None
            plotting.make_jump_location_plot(results, outfile=plot_file, show_plot=show_plot)

        fancyprint('Step JumpStep done.')

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
        self.datafiles = utils.sort_datamodels(input_data)
        self.fileroots = utils.get_filename_root(self.datafiles)

        # Get instrument.
        self.instrument = utils.get_instrument_name(self.datafiles[0])

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
                res = expected_file
            # If no output files are detected, run the step.
            else:
                step = calwebb_detector1.ramp_fit_step.RampFitStep()
                res = step.call(segment, output_dir=self.output_dir, save_results=save_results,
                                maximum_cores='quarter', **kwargs)[1]
                # From jwst v1.9.0-1.11.0 ramp fitting algorithm was changed to make all pixels
                # with DO_NOT_USE DQ flags be NaN after ramp fitting. These pixels are marked,
                # ignored and interpolated anyways, so this does not change any actual
                # functionality, but cosmetcically this annoys me, as now plots look terrible.
                # Just griddata interpolate all NaNs so things look better. Note this does not
                # supercede any interpolation done later in Stage 2.
                nint, dimy, dimx = res.data.shape
                px, py = np.meshgrid(np.arange(dimx), np.arange(dimy))
                fancyprint('Doing cosmetic NaN interpolation.')
                for j in range(nint):
                    ii = np.where(np.isfinite(res.data[j]))
                    res.data[j] = griddata(ii, res.data[j][ii], (py, px), method='nearest')
                if save_results is True:
                    res.save(self.output_dir + res.meta.filename)

                # Store pixel flags for use in 1/f correction.
                if save_results is True:
                    flags = res.dq
                    flags[flags != 0] = 1  # Convert to binary mask.
                    # NIRISS observations have a line of bright pixels that move down the detector
                    # one row at a time each integration. IDK why exactly, its a "detector reset
                    # artifact" according to Loïc. It needs to be masked.
                    if self.instrument == 'NIRISS':
                        artifact = utils.mask_reset_artifact(res.data)
                        flags = (flags.astype(bool) | artifact.astype(bool)).astype(int)

                    # Save flags to file.
                    hdu = fits.PrimaryHDU()
                    hdu1 = fits.ImageHDU(flags)
                    hdul = fits.HDUList([hdu, hdu1])
                    outfile = (self.output_dir + self.fileroots[i] + 'pixelflags.fits')
                    hdul.writeto(outfile, overwrite=True)

                    # Remove rate file because we don't need it and I don't like having extra files.
                    rate = res.meta.filename.replace('_1_ramp', '_0_ramp')
                    os.remove(self.output_dir + rate)

                    # Verify that filename is correct.
                    current_name = self.output_dir + res.meta.filename
                    if expected_file != current_name:
                        res.close()
                        os.rename(current_name, expected_file)
                        thisfile = fits.open(expected_file)
                        thisfile[0].header['FILENAME'] = self.fileroots[i] + self.tag
                        thisfile.writeto(expected_file, overwrite=True)
                    res = expected_file
            results.append(res)

        fancyprint('Step RampFitStep really done.')

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
        self.datafiles = utils.sort_datamodels(input_data)
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
            Keyword arguments for calwebb_detector1.gain_scale_step.GainScaleStep.

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
                        thisfile = fits.open(expected_file)
                        thisfile[0].header['FILENAME'] = self.fileroots[i] + self.tag
                        thisfile.writeto(expected_file, overwrite=True)
                    res = expected_file
            results.append(res)

        return results


def flag_hot_pixels(result, deepframe=None, box_size=10, thresh=15, hot_pix=None):
    """Identify and flag additional hot pixels in a SOSS TSO which are not already in the default
    pipeline flags.

    Parameters
    ----------
    result : jwst.datamodel
        Input datamodel.
    deepframe : array-like(float), None
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

    # Get location of all pixels already flagged as warm or hot.
    hot = utils.get_dq_flag_metrics(result.pixeldq, ['HOT', 'WARM'])

    if hot_pix is not None:
        hot_pix = hot_pix
        fancyprint('Using provided hot pixel map...')
        result.pixeldq[hot_pix] += 2048

    else:
        assert deepframe is not None

        dimy, dimx = np.shape(deepframe)
        all_med = np.nanmedian(deepframe)
        hot_pix = np.zeros_like(deepframe).astype(bool)
        for i in tqdm(range(4, dimx - 4)):
            for j in range(dimy):
                box_size_i = box_size
                box_prop = utils.get_interp_box(deepframe, box_size_i, i, j, dimx)
                # Ensure that the median and std dev extracted are good. If not, increase the box
                # size until they are.
                while np.any(np.isnan(box_prop)):
                    box_size_i += 1
                    box_prop = utils.get_interp_box(deepframe, box_size_i, i, j, dimx)
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


def jumpstep_in_time(datafile, window=5, thresh=10, fileroot=None, save_results=True,
                     output_dir=None):
    """Jump detection step in the temporal domain. This algorithm is based off of Nikolov+ (2014)
    and identifies cosmic ray hits in the temporal domain. All jumps for ngroup<=2 are replaced
    with the median of surrounding integrations, whereas jumps for ngroup>3 are flagged.

    Parameters
    ----------
    datafile : str, RampModel
        RampModel for a segment of the TSO or path to one.
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
    if isinstance(datafile, str):
        datafile = fits.open(datafile)
        cube = datafile[1].data
        dqcube = datafile[3].data
        filename = datafile[0].header['FILENAME']
    else:
        datafile = utils.open_filetype(datafile)
        cube = datafile.data
        dqcube = datafile.groupdq
        filename = datafile.meta.filename
    fancyprint('Processing file {}.'.format(filename))

    nints, ngroups, dimy, dimx = np.shape(cube)

    # Mask the detector reset artifact which is picked up by this flagging.
    # Artifact only affects first 256 integrations for SOSS and first 60 for NIRSpec
    artifact = utils.mask_reset_artifact(datafile)

    # Jump detection algorithm based on Nikolov+ (2014). For each integration, create a difference
    # image using the median of surrounding integrations. Flag all pixels with deviations more
    # than X-sigma as comsic rays hits.
    count, interp = 0, 0
    for g in tqdm(range(ngroups)):
        # Filter the data using the specified window
        cube_filt = medfilt(cube[:, g], (window, 1, 1))
        # Calculate the point-to-point scatter along the temporal axis.
        scatter = np.median(np.abs(0.5 * (cube[0:-2, g] + cube[2:, g]) - cube[1:-1, g]), axis=0)
        scatter = np.where(scatter == 0, np.inf, scatter)
        # Find pixels which deviate more than the specified threshold.
        scale = np.abs(cube[:, g] - cube_filt) / scatter
        ii = ((scale >= thresh) & (cube[:, g] > np.nanpercentile(cube, 10)) & (artifact == 0))

        # If ngroup<=2, replace the pixel with the stack median so that a ramp can still be fit.
        if ngroups <= 2:
            # Do not want to interpolate pixels which are flagged for another reason, so only
            # select good pixels or those which are flagged for jumps.
            jj = (dqcube[:, g] == 0) | (dqcube[:, g] == 4)
            # Replace these pixels with the stack median and remove the dq flag.
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

    if save_results is True:
        datafile[1].data = cube
        datafile[3].data = dqcube
        outfile = output_dir + fileroot + 'jump.fits'
        datafile[0].header['FILENAME'] = fileroot + 'jump.fits'
        datafile.writeto(outfile, overwrite=True)
        datafile = outfile
    else:
        datafile.data = cube
        datafile.groupdq = dqcube

    fancyprint('Done')

    return datafile


def oneoverfstep_nirspec(datafile, output_dir=None, save_results=True, pixel_mask=None,
                         fileroot=None, mask_width=16, centroids=None, method='median',
                         override_centroids=False):
    """Custom 1/f correction routine to be applied at the group level. The median level of each
    detector column is subtracted off while masking outlier pixels and the target trace.

    Parameters
    ----------
    datafile : str, RampModel, CubeModel
        Path to data files, or datamodel itself for a segment of the TSO. Should be 4D ramps,
        but 3D rate files are also accepted.
    output_dir : str, None
        Directory to which to save results. Only necessary if saving results.
    save_results : bool
        If True, save results to disk.
    pixel_mask : array-like[float], None
        Maps of pixels to mask. Should be 3D (nints, dimy, dimx).
    fileroot : str, None
        Root name for output files. Only necessary if saving results.
    mask_width : int
        Full width in pixels to mask around the trace.
    centroids : dict
        Dictionary containing trace positions for each order.
    method : str
        1/f correction method. Options are "median" or "slope".
    override_centroids : bool
        If True, when passed centroids do not match the shape of the data frame, use passed
        centroids and do not recalculate.

    Returns
    -------
    result : str, CubeModel
        RampModel for the segment, corrected for 1/f noise.
    """

    fancyprint('Starting 1/f correction step using the {} method.'.format(method))

    # If saving results, ensure output directory and fileroots are provided.
    if save_results is True:
        assert output_dir is not None
        assert fileroot is not None
        # Output directory formatting.
        if output_dir[-1] != '/':
            output_dir += '/'

    # Load in data, accept both strings (open as fits) and datamodels.
    if isinstance(datafile, str):
        cube = fits.getdata(datafile, 1)
        filename = fits.getheader(datafile, 0)['FILENAME']
    else:
        with utils.open_filetype(datafile) as thisfile:
            cube = thisfile.data
            filename = thisfile.meta.filename
    fancyprint('Processing file {}.'.format(filename))

    # Define the readout setup - can be 4D or 3D.
    if np.ndim(cube) == 4:
        nint, ngroup, dimy, dimx = np.shape(cube)
    else:
        nint, dimy, dimx = np.shape(cube)

    # Unpack trace centroids.
    fancyprint('Unpacking centroids.')
    xpos, ypos = centroids['xpos'], centroids['ypos']
    # If centroids on trimmed slit data frame are passed to be used on full frame data,
    # recalculate centroids on data, unless overridden.
    # This is mostly just cosmetic as we only care about the "in slit" data, however, I like my
    # plots to look nice and not have part of the detector corrected and part not.
    if len(xpos) != dimx and override_centroids is False:
        fancyprint('Dimension of passed centroids do not match data frame dimensions. New '
                   'centroids will be calculated.', msg_type='WARNING')
        # Create deepstack.
        if np.ndim(cube) == 4:
            # Only need last group.
            thiscube = cube[:, -1]
        else:
            thiscube = cube
        deepstack = bn.nanmedian(thiscube, axis=0)
        # Get detector to determine x limits.
        det = utils.get_nrs_detector_name(datafile)
        if det == 'nrs1':
            xstart = 500
        else:
            xstart = 0
        centroids = utils.get_centroids_nirspec(deepstack, xstart=xstart, save_results=False)
        xpos, ypos = centroids[0], centroids[1]

    # Read in the outlier maps - (nints, dimy, dimx) 3D cubes.
    if pixel_mask is None:
        fancyprint('No outlier map passed, ignoring outliers.', msg_type='WARNING')
        outliers = np.zeros((nint, dimy, dimx)).astype(bool)
    else:
        fancyprint('Constructing outlier map.')
        # If the correction is at the integration level after performing the Extract2D step,
        # the detector size will be limited to that illuminated by the slit.
        # Trim pixel masks to match.
        with utils.open_filetype(datafile) as thisfile:
            if isinstance(thisfile, datamodels.SlitModel):
                xstart = thisfile.xstart - 1  # 1-indexed.
                xend = xstart + thisfile.xsize
                ystart = thisfile.ystart - 1
                yend = ystart + thisfile.ysize
                outliers = pixel_mask[:, ystart:yend, xstart:xend].astype(bool)
            else:
                outliers = pixel_mask.astype(bool)

    # Construct trace masks.
    fancyprint('Constructing trace mask.')
    low = np.max([np.zeros_like(ypos), ypos - mask_width / 2], axis=0).astype(int)
    up = np.min([dimy * np.ones_like(ypos), ypos + mask_width / 2], axis=0).astype(int)
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
    scatter = np.median(np.abs(0.5 * (thiscube[0:-2] + thiscube[2:]) - thiscube[1:-1]), axis=0)
    scatter = np.where(scatter == 0, np.inf, scatter)
    # Find pixels which deviate more than 10 sigma.
    scale = np.abs(thiscube - cube_filt) / scatter
    ii = np.where(scale > 10)
    outliers[ii] = 1

    # The outlier map is 0 where good and >0 otherwise. As this will be applied multiplicatively,
    # replace 0s with 1s and others with NaNs.
    outliers = np.where(outliers == 0, 1, np.nan)

    # Loop over all integrations to determine the 1/f noise level and correct it.
    fancyprint('Starting full frame correction.')
    cube_corr = copy.deepcopy(cube)
    for i in tqdm(range(nint)):
        # Apply the outlier mask.
        cube[i] *= outliers[i, :, :]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            # Single 1/f scaling for all rows.
            dc = np.zeros_like(cube[i])
            # First method: simply take the column-wise median of all unmasked pixels as the 1/f +
            # background level.
            if method == 'median':
                # For group-level corrections.
                if np.ndim(cube) == 4:
                    dc[:, :, :] = bn.nanmedian(cube[i], axis=1)[:, None, :]
                # For integration-level corrections.
                else:
                    dc[:, :] = bn.nanmedian(cube[i], axis=0)[None, :]

            # Second method: fit a linear slope to each column as subtract that as the 1/f +
            # background level.
            elif method == 'slope':
                # For group-level corrections.
                if np.ndim(cube) == 4:
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
                raise ValueError('Unrecognized 1/f method {}'.format(method))

        # Make sure no NaNs are in the DC map
        dc = np.where(np.isfinite(dc), dc, 0)
        # Subtract the 1/f map.
        cube_corr[i] -= dc

    # Save 1/f corrected data.
    if save_results is True:
        thisfile = fits.open(datafile)
        thisfile[1].data = cube_corr
        # Save corrected data.
        result = output_dir + fileroot + 'oneoverfstep.fits'
        thisfile[0].header['FILENAME'] = fileroot + 'oneoverfstep.fits'
        thisfile.writeto(result, overwrite=True)
        fancyprint('File saved to: {}.'.format(result))
    # If not saving results, need to work in datamodels to not break interoperability with stsci
    # pipeline.
    else:
        result = utils.open_filetype(datafile)
        result.data = cube_corr

    return result


def oneoverfstep_scale(datafile, deepstack, inner_mask_width=40, outer_mask_width=70,
                       even_odd_rows=True, background=None, timeseries=None, timeseries_o2=None,
                       output_dir=None, save_results=True, pixel_mask=None, fileroot=None,
                       method='achromatic', centroids=None, smoothing_scale=None):
    """Custom 1/f correction routine to be applied at the group or integration level. A median
    stack is constructed using all out-of-transit integrations and subtracted from each individual
    integration. The column-wise median of this difference image is then subtracted from the
    original frame to correct 1/f noise. Outlier pixels, background contaminants, and the target
    trace itself can (should) be masked to improve the estimation.

    Parameters
    ----------
    datafile : str, RampModel, CubeModel
        Datamodel for a segment of the TSO, or a path to one. Should be 4D ramps, but 3D rate
        files are also accepted.
    deepstack : array-like[float]
        Median stack of the baseline integrations.
    inner_mask_width : int
        Width around the trace to mask. For windowed methods, defines the inner window edge.
    outer_mask_width : int
        For windowed methods, the outer edge of the window.
    even_odd_rows : bool
        If True, calculate 1/f noise seperately for even and odd numbered rows.
    background : array-like[float], None
        Model of background flux.
    timeseries : array-like[float], None
        Estimate of normalized light curve(s).
    timeseries_o2 : array-like[float], None
        Estimate of normalized light curve(s) for order 2. Only necessary if method is chromatic.
    output_dir : str, None
        Directory to which to save results. Only necessary if saving results.
    save_results : bool
        If True, save results to disk.
    pixel_mask : array-like[float], None
        Maps of pixels to mask for each data segment. Should be 3D (nints, dimy, dimx).
    fileroot : str, None
        Root name for output file. Only necessary if saving results.
    method : str
        Options are "chromatic", "achromatic", or "achromatic-window".
    centroids : dict
        Dictionary containing trace positions for each order.
    smoothing_scale : int, None
        If no timseries is provided, the scale (in number of integrations) on which to smooth the
        self-extracted timseries.

    Returns
    -------
    results : CubeModel, RampModel, str
        RampModel for the segment, corrected for 1/f noise.
    """

    fancyprint('Starting 1/f correction step using the scale-{} method.'.format(method))

    # If saving results, ensure output directory and fileroots are provided.
    if save_results is True:
        assert output_dir is not None
        assert fileroot is not None
        # Output directory formatting.
        if output_dir[-1] != '/':
            output_dir += '/'

    # Ensure method is correct.
    if method not in ['chromatic', 'achromatic', 'achromatic-window']:
        raise ValueError('Method must be one of "chromatic", "achromatic", or "achromatic-window".')

    # Load in data, accept both strings (open as fits) and datamodels.
    if isinstance(datafile, str):
        cube = fits.getdata(datafile, 1)
        filename = fits.getheader(datafile, 0)['FILENAME']
    else:
        with utils.open_filetype(datafile) as thisfile:
            cube = thisfile.data
            filename = thisfile.meta.filename
    fancyprint('Processing file {}.'.format(filename))

    # Define the readout setup - can be 4D or 3D.
    if np.ndim(cube) == 4:
        nint, ngroup, dimy, dimx = np.shape(cube)
    else:
        nint, dimy, dimx = np.shape(cube)
    if dimy == 256:
        subarray = 'SUBSTRIP256'
    else:
        subarray = 'SUBSTRIP96'

    # Unpack trace centroids.
    fancyprint('Unpacking centroids.')
    x1, y1 = centroids['xpos'], centroids['ypos o1']
    y2, y3 = centroids['ypos o2'], centroids['ypos o3']
    x2, x3 = x1[:len(y2)], x1[:len(y3)]  # Trim o1 x positions to match o2 & 03.

    # Read in the outlier maps - (nints, dimy, dimx) 3D cubes.
    if pixel_mask is None:
        fancyprint('No outlier maps passed, ignoring outliers.', msg_type='WARNING')
        outliers1 = np.zeros((nint, dimy, dimx)).astype(bool)
    else:
        fancyprint('Constructing outlier map.')
        outliers1 = pixel_mask.astype(bool)
    outliers2 = np.copy(outliers1)

    # Construct trace masks.
    fancyprint('Constructing trace mask.')

    # Order 1 necessary for all subarrays -- inner and outer masks.
    mask1_in = utils.make_soss_tracemask(x1, y1, inner_mask_width, dimy, dimx)
    mask1_out = utils.make_soss_tracemask(x1, y1, outer_mask_width, dimy, dimx)

    # Include orders 2 and 3 for SUBSTRIP256.
    if subarray != 'SUBSTRIP96':
        # Order 2 -- inner and outer masks.
        mask2_in = utils.make_soss_tracemask(x2, y2, inner_mask_width, dimy, dimx)
        mask2_out = utils.make_soss_tracemask(x2, y2, outer_mask_width, dimy, dimx)
        # Order 3 -- only need inner mask.
        mask3 = utils.make_soss_tracemask(x3, y3, inner_mask_width, dimy, dimx)
    # But not for SUBSTRIP96.
    else:
        mask2_in = np.zeros_like(mask1_in)
        mask2_out = np.zeros_like(mask1_in)
        mask3 = np.zeros_like(mask1_in)

    # Add the appropriate trace mask to the outliers cube for the selected 1/f method.
    tracemask = (mask1_in.astype(bool) | mask2_in.astype(bool) | mask3.astype(bool))
    # For the scale-achromatic, just need to mask the cores of each trace, defined by
    # inner_mask_width.
    if method == 'achromatic':
        outliers1 = (outliers1 | tracemask).astype(int)
    # For the windowed corrections, construct a window around each order defined by
    # inner_mask_width and outer_mask_width.
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
    cube_filt[-2:], cube_filt[2:] = cube_filt[-3], cube_filt[3]
    # Calculate the point-to-point scatter along the temporal axis.
    scatter = np.median(np.abs(0.5 * (thiscube[0:-2] + thiscube[2:]) - thiscube[1:-1]), axis=0)
    scatter = np.where(scatter == 0, np.inf, scatter)
    # Find pixels which deviate more than 10 sigma.
    scale = np.abs(thiscube - cube_filt) / scatter
    ii = np.where(scale > 10)
    outliers1[ii] = 1
    outliers2[ii] = 1

    # The outlier map is 0 where good and >0 otherwise. As this will be applied multiplicatively,
    # replace 0s with 1s and others with NaNs.
    if method in ['chromatic', 'achromatic-window']:
        outliers1 = np.where(outliers1 == 0, 1, np.nan)
        outliers2 = np.where(outliers2 == 0, 1, np.nan)
        # Also cut everything redder than ~0.9µm in order 2.
        outliers2[:, :, :1100] = np.nan
    else:
        outliers1 = np.where(outliers1 == 0, 1, np.nan)

    # In order to subtract off the trace as completely as possible, the median stack must be
    # scaled, via the transit curve, to the flux level of each integration. This can be done via
    # two methods: using the white light curve (i.e., assuming the scaling is not wavelength
    # dependent), or using extracted 2D light curves, such that the scaling is wavelength dependent.
    # Get light curve. If not, estimate it (1D only) from data.
    if timeseries is None:
        if method == 'achromatic':
            fancyprint('No timeseries passed. It will be estimated from current data.',
                       msg_type='WARNING')
            # If no lightcurve is provided, estimate it from the current data.
            if np.ndim(cube) == 4:
                postage = cube[:, -1, 20:60, 1500:1550]
                zero_point = deepstack[-1, 20:60, 1500:1550]
            else:
                postage = cube[:, 20:60, 1500:1550]
                zero_point = deepstack[20:60, 1500:1550]
            timeseries = np.nansum(postage, axis=(1, 2))
            timeseries /= np.nansum(zero_point)
            if smoothing_scale is None:
                # If no timescale provided, smooth the time series on a timescale of ~2%.
                smoothing_scale = 0.02 * np.shape(cube)[0]
            fancyprint('Smoothing self-calibrated timeseries on a scale of '
                       '{} integrations.'.format(int(smoothing_scale)))
            timeseries = median_filter(timeseries, int(smoothing_scale))
        else:
            raise ValueError('2D light curves must be provided to use chromatic method.')

    # If passed light curve is 1D, extend to 2D.
    if np.ndim(timeseries) == 1:
        # If 1D timeseries is passed cannot do chromatic correction.
        if method == 'chromatic':
            raise ValueError('2D light curves are required for chromatic correction, but 1D ones '
                             'were passed.')
        else:
            timeseries = np.repeat(timeseries[:, np.newaxis], dimx, axis=1)
    # Get timeseries for order 2.
    if method == 'chromatic':
        if timeseries_o2 is None:
            raise ValueError('2D light curves for order 2 must be provided to use chromatic '
                             'method.')
        if np.ndim(timeseries_o2) == 1:
            # If 1D timeseries is passed cannot do chromatic correction.
            raise ValueError('2D light curves are required for chromatic correction, but 1D ones '
                             'were passed.')

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
                # For achromatic method, subtract the calculated 1/f values from the whole frame.
                cube[i] -= dc
            else:
                if order == 1:
                    # For order 1, subtract 1/f values in window around trace.
                    cube[i] -= (dc * mask1_in[None, :, :])
                else:
                    # For order 2, subtract in a window around the trace.
                    cube[i] -= (dc * mask2_in[None, :, :])

    # Background must be subtracted to accurately subtract off the target trace and isolate 1/f
    # noise. However, the background flux must also be corrected for non-linearity. Therefore, it
    # should be added back after the 1/f is subtracted, in order to be re-subtracted later.
    # Note: only relevant for group-level corrections.
    if background is not None:
        # Add back the zodi background.
        cube += background

    # Save 1/f corrected data.
    if save_results is True:
        thisfile = fits.open(datafile)
        thisfile[1].data = cube
        # Save corrected data.
        result = output_dir + fileroot + 'oneoverfstep.fits'
        thisfile[0].header['FILENAME'] = fileroot + 'oneoverfstep.fits'
        thisfile.writeto(result, overwrite=True)
        fancyprint('File saved to: {}.'.format(result))
    # If not saving results, need to work in datamodels to not break interoperability with stsci
    # pipeline.
    else:
        result = utils.open_filetype(datafile)
        result.data = cube

    return result


def oneoverfstep_solve(datafile, deepstack, trace_width=70, background=None, output_dir=None,
                       save_results=True, pixel_mask=None, fileroot=None, centroids=None):
    """Custom 1/f correction routine to be applied at the group or integration level. 1/f noise
    level and median frame scaling is calculated independently for each pixel column. Outlier
    pixels and background contaminants can (should) be masked to improve the estimation.

    Parameters
    ----------
    datafile : str, RampModel, CubeModel
        Datamodel for a segment of the TSO, or path to one. Should be 4D ramp, but 3D rate files
        are also accepted.
    deepstack : array-like[float]
        Median stack of the baseline integrations.
    trace_width : int
        Defines the width around the trace to consider for MLE solving.
    background : array-like[float], None
        Model of background flux.
    output_dir : str, None
        Directory to which to save results. Only necessary if saving results.
    save_results : bool
        If True, save results to disk.
    pixel_mask : array-like[float], None
        Maps of pixels to mask. Can be 3D (nints, dimy, dimx), or 2D (dimy, dimx).
    fileroot : str, None
        Root name for output files. Only necessary if saving results.
    centroids : dict
        Dictionary containing trace positions for each order.

    Returns
    -------
    corrected_rampmodel : str, CubeModel
        RampModel for the segment, corrected for 1/f noise.
    calc_vals : dict
        Dictionary of calculated light curve scaling and 1/f noise values.
    """

    fancyprint('Starting 1/f correction step using the solve method.')

    # If saving results, ensure output directory and fileroots are provided.
    if save_results is True:
        assert output_dir is not None
        assert fileroot is not None
        # Output directory formatting.
        if output_dir[-1] != '/':
            output_dir += '/'

    # Load in data, accept both strings (open as fits) and datamodels.
    if isinstance(datafile, str):
        cube = fits.getdata(datafile, 1)
        if np.ndim(cube) == 4:
            grpdq = fits.getdata(datafile, 2)
            pixdq = fits.getdata(datafile, 3)
            dqcube = grpdq + pixdq
        else:
            dqcube = fits.getdata(datafile, 3)
        filename = fits.getheader(datafile, 0)['FILENAME']
    else:
        with utils.open_filetype(datafile) as thisfile:
            cube = thisfile.data
            if np.ndim(cube) == 4:
                grpdq = thisfile.groupdq
                pixdq = thisfile.pixeldq
                dqcube = grpdq + pixdq
            else:
                dqcube = thisfile.dq
            filename = thisfile.meta.filename
    fancyprint('Processing file {}.'.format(filename))

    # Set errors to variance of data along integration axis.
    err1 = np.nanstd(cube, axis=0)
    err1 = np.repeat(err1[np.newaxis], cube.shape[0], axis=0)
    err2 = copy.deepcopy(err1)

    # Define the readout setup - can be 4D (recommended) or 3D.
    if np.ndim(cube) == 4:
        nint, ngroup, dimy, dimx = np.shape(cube)
    else:
        nint, dimy, dimx = np.shape(cube)
        ngroup = 0
    subarray = utils.get_soss_subarray(datafile)

    # Get outlier masks.
    # Ideally, for this algorithm, we only want to consider pixels quite near to the trace.
    if pixel_mask is None:
        fancyprint('No outlier maps passed, ignoring outliers.', msg_type='WARNING')
        outliers1 = np.zeros((nint, dimy, dimx)).astype(bool)
    else:
        fancyprint('Constructing outlier map.')
        outliers1 = pixel_mask.astype(bool)
    outliers2 = np.copy(outliers1)

    # Unpack trace centroids.
    fancyprint('Unpacking centroids.')
    x1, y1 = centroids['xpos'], centroids['ypos o1']
    y2 = centroids['ypos o2']
    x2 = x1[:len(y2)]  # Trim o1 x positions to match o2.

    # Construct trace masks.
    # Order 1 necessary for all subarrays.
    mask1 = utils.make_soss_tracemask(x1, y1, trace_width, dimy, dimx, invert=True)
    trace1 = np.where(mask1 == 1, 0, 1)
    # Include order 2 for SUBSTRIP256.
    if subarray != 'SUBSTRIP96':
        # Order 2.
        mask2 = utils.make_soss_tracemask(x2, y2, trace_width, dimy, dimx, invert=True)
    # But not for SUBSTRIP96.
    else:
        mask2 = np.ones_like(mask1)
    trace2 = np.where(mask2 == 1, 0, 1)
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

    # Identify and mask any potential jumps that are not already flagged.
    fancyprint('Flagging additional outliers.')
    if np.ndim(cube) == 4:
        thiscube = cube[:, -1]
    else:
        thiscube = cube
    cube_filt = medfilt(thiscube, (5, 1, 1))
    cube_filt[-2:], cube_filt[2:] = cube_filt[-3], cube_filt[3]
    # Calculate the point-to-point scatter along the temporal axis.
    scatter = np.median(np.abs(0.5 * (thiscube[0:-2] + thiscube[2:]) - thiscube[1:-1]), axis=0)
    scatter = np.where(scatter == 0, np.inf, scatter)
    # Find pixels which deviate more than 10 sigma.
    scale = np.abs(thiscube - cube_filt) / scatter
    ii = np.where(scale > 10)
    if ngroup == 0:
        err1[ii] = np.inf
        err2[ii2] = np.inf
    else:
        for g in range(ngroup):
            err1[:, g][ii] = np.inf
            err2[:, g][ii2] = np.inf

    # If no outlier masks were provided and correction is at group level, mask detector reset
    # artifact. Only necessary for first 256 integrations.
    if pixel_mask is None and np.ndim(cube) == 4:
        artifact = utils.mask_reset_artifact(datafile)
        ii = np.where(artifact == 1)
        err1[ii] = np.inf
        err2[ii] = np.inf

    # Calculate 1/f noise using a wavelength-dependent scaling.
    calc_vals = {}
    for order, err in zip([1, 2], [err1, err2]):
        fancyprint('Starting order {}.'.format(order))
        # Don't do anything for order 2 if SUBSTRIP96.
        if order == 2 and dimy == 96:
            continue
        calc_vals['o{}'.format(order)] = {}

        # Loop over all integrations to determine the 1/f noise level via MLE estimation, and
        # correct it.
        if ngroup == 0:
            scaling_e = np.zeros((nint, dimx))
            scaling_o = np.zeros((nint, dimx))
        else:
            scaling_e = np.zeros((nint, ngroup, dimx))
            scaling_o = np.zeros((nint, ngroup, dimx))
        oof = np.zeros_like(cube)
        for i in tqdm(range(nint)):
            # Integration-level correction.
            if ngroup == 0:
                # Do the chromatic 1/f calculation.
                m_e, b_e, m_o, b_o = utils.line_mle(deepstack, cube[i], err[i])
                oof[i, ::2, :] = b_e[None, :]
                oof[i, 1::2, :] = b_o[None, :]
                scaling_e[i] = m_e
                scaling_o[i] = m_o

                # Replace any NaNs (that could happen if an entire column is masked) with zeros.
                oof[np.isnan(oof)] = 0
                oof[np.isinf(oof)] = 0
                # Subtract the 1/f contribution.
                if order == 1:
                    # For order 1, subtract the 1/f value in a window around the trace.
                    cube[i] -= (oof[i] * trace1)
                else:
                    # For order 2, only subtract it from around the order 2 trace.
                    cube[i] -= (oof[i] * trace2)

            else:
                # Group-level correction.
                for g in range(ngroup):
                    # Do the chromatic 1/f calculation.
                    m_e, b_e, m_o, b_o = utils.line_mle(deepstack[g], cube[i, g], err[i, g])
                    oof[i, g, ::2] = b_e[None, :]
                    oof[i, g, 1::2] = b_o[None, :]
                    scaling_e[i, g] = m_e
                    scaling_o[i, g] = m_o

                    # Replace any NaNs (that could happen if an entire column is masked) with zeros.
                    oof[np.isnan(oof)] = 0
                    oof[np.isinf(oof)] = 0
                    # Subtract the 1/f contribution.
                    if order == 1:
                        # For order 1, subtract the 1/f value in a window around the trace.
                        cube[i, g] -= (oof[i, g] * trace1)
                    else:
                        # For order 2, only subtract it from around the order 2 trace.
                        cube[i, g] -= (oof[i, g] * trace2)

        calc_vals['o{}'.format(order)]['oof'] = oof
        calc_vals['o{}'.format(order)]['scale_o'] = scaling_o
        calc_vals['o{}'.format(order)]['scale_e'] = scaling_e

    # Add back the zodi background.
    if background is not None:
        cube += background

    # Save 1/f corrected data.
    if save_results is True:
        thisfile = fits.open(datafile)
        thisfile[1].data = cube
        # Save corrected data.
        result = output_dir + fileroot + 'oneoverfstep.fits'
        thisfile[0].header['FILENAME'] = fileroot + 'oneoverfstep.fits'
        thisfile.writeto(result, overwrite=True)
        fancyprint('File saved to: {}.'.format(result))
    # If not saving results, need to work in datamodels to not break interoperability with stsci
    # pipeline.
    else:
        result = utils.open_filetype(datafile)
        result.data = cube

    return result, calc_vals


def subtract_custom_superbias(datafile, superbias, method='constant', centroids=None, mask_width=10,
                              output_dir=None, save_results=True, fileroot=None,
                              override_centroids=False):
    """Perform a custom superbias subtraction on NIRSpec data where the superbias frame is
    calculated using the 0th group data from the observation itself. The superbias can either be
    subtracted as is from the data, or rescaled to account for minor bias level variations which
    cannot be captured by reference pixels with NIRSpec subarrays.

    Parameters
    ----------
    datafile : str, CubeModel
        Path to data files, or datamodel itself for a segment of the TSO.
    superbias : ndarray(float)
        Superbias frame.
    method : str
        Superbias subtraction method. Either 'constant' or 'rescale'.
    centroids : dict, None
        Dictionary containing trace positions for each order.
    mask_width : int
        Full width in pixels to mask around the trace.
    output_dir : str, None
        Directory to which to save results. Only necessary if saving results.
    save_results : bool
        If True, save results to disk.
    fileroot : str, None
        Root name for output files. Only necessary if saving results.
    override_centroids : bool
        If True, when passed centroids do not match the shape of the data frame, use passed
        centroids and do not recalculate.

    Returns
    -------
    result : RampModel, str
        RampModel for the segment, corrected for the superbias.
    scale_factors : ndarray(float), None
        Time series of superbias scale factors for each integration.
    """

    fancyprint('Starting superbias subraction step using the custom-{} method.'.format(method))

    # If saving results, ensure output directory and fileroots are provided.
    if save_results is True:
        assert output_dir is not None
        assert fileroot is not None
        # Output directory formatting.
        if output_dir[-1] != '/':
            output_dir += '/'

    # Load in data, accept both strings (open as fits) and datamodels.
    if isinstance(datafile, str):
        cube = fits.getdata(datafile, 1)
        filename = fits.getheader(datafile, 0)['FILENAME']
        dq = fits.getdata(datafile, 2).astype(bool)
    else:
        with utils.open_filetype(datafile) as thisfile:
            cube = thisfile.data
            filename = thisfile.meta.filename
            dq = thisfile.pixeldq.astype(bool)
    fancyprint('Processing file {}.'.format(filename))

    # Define the readout setup
    nint, ngroup, dimy, dimx = np.shape(cube)

    # Subtract the superbias.
    # Subtract the custom superbias from each integration.
    if method == 'constant':
        cube_corr = cube - superbias
        scale_factors = None
    # Rescale the custom superbias to match each integration before subtracting.
    else:
        # Get trace centroids to mask.
        if centroids is not None:
            # Unpack centroids if provided.
            fancyprint('Unpacking centroids.')
            xpos, ypos = centroids['xpos'], centroids['ypos']
            # If centroids on trimmed slit data frame are passed to be used on full frame data,
            # recalculate centroids on data, unless overridden.
            if len(xpos) != dimx and override_centroids is False:
                fancyprint('Dimension of passed centroids do not match data frame dimensions. New '
                           'centroids will be calculated.', msg_type='WARNING')
                centroids = None
        if centroids is None:
            # If no centroids file is provided, get the trace positions from the data now.
            fancyprint('No centroids provided, locating trace positions.')
            # Create deepstack.
            if np.ndim(cube) == 4:
                # Only need last group.
                thiscube = cube[:, -1]
            else:
                thiscube = cube
            deepstack = bn.nanmedian(thiscube, axis=0)
            # Get detector to determine x limits.
            det = utils.get_nrs_detector_name(datafile)
            if det == 'nrs1':
                xstart = 500
            else:
                xstart = 0
            centroids = utils.get_centroids_nirspec(deepstack, xstart=xstart, save_results=False)
            xpos, ypos = centroids[0], centroids[1]

        # Construct trace masks.
        fancyprint('Constructing trace mask.')
        low = np.max([np.zeros_like(ypos), ypos - mask_width / 2], axis=0).astype(int)
        up = np.min([dimy * np.ones_like(ypos), ypos + mask_width / 2], axis=0).astype(int)
        tracemask = np.ones((dimy, dimx))
        for i, x in enumerate(xpos):
            tracemask[low[i]:up[i], int(x)] = 0
        # Combine with dq flags.
        mask = ~dq | tracemask.astype(bool)
        # Replace bad pixels with nans.
        mask = np.where(mask == 0, np.nan, mask)
        # Apply mask to 0th group frames.
        group0 = mask * cube[:, 0]

        # Calculate the superbias scaling relative to each integration.
        scale_factors = np.nanmedian(group0 / superbias, axis=(1, 2))
        # Rescale custom superbias and subtract from each integration.
        cube_corr = (cube - scale_factors[:, None, None, None] * superbias[None, None, :, :])

    # Save superbias corrected data.
    if save_results is True:
        thisfile = fits.open(datafile)
        thisfile[1].data = cube_corr
        # Save corrected data.
        result = output_dir + fileroot + 'superbiasstep.fits'
        thisfile[0].header['FILENAME'] = fileroot + 'superbiasstep.fits'
        thisfile.writeto(result, overwrite=True)
        fancyprint('File saved to: {}.'.format(result))
    # If not saving results, need to work in datamodels to not break interoperability with stsci
    # pipeline.
    else:
        result = utils.open_filetype(datafile)
        result.data = cube_corr

    return result, scale_factors


def run_stage1(results, mode, soss_background_model=None, baseline_ints=None,
               oof_method='scale-achromatic', superbias_method='crds',
               soss_timeseries=None, soss_timeseries_o2=None, save_results=True, pixel_masks=None,
               force_redo=False, hot_pixel_map=None, flag_up_ramp=False, rejection_threshold=15,
               flag_in_time=True, time_rejection_threshold=10, root_dir='./', output_tag='',
               skip_steps=None, do_plot=False, show_plot=False, soss_inner_mask_width=40,
               soss_outer_mask_width=70, centroids=None, nirspec_mask_width=16, **kwargs):
    """Run the exoTEDRF Stage 1 pipeline: detector level processing, using a combination of
    official STScI DMS and custom steps. Documentation for the official DMS steps can be found here:
    https://jwst-pipeline.readthedocs.io/en/latest/jwst/pipeline/calwebb_detector1.html

    Parameters
    ----------
    results : array-like(str)
        List of paths to input uncalibrated datafiles for all segments in an exposure.
    mode : str
        Instrument mode which produced the data being analyzed.
    soss_background_model : str, array-like(float), None
        SOSS background model or path to a file containing it.
    baseline_ints : array-like(int)
        Integration numbers for transit ingress and egress.
    oof_method : str
        1/f correction method. Options are "scale-chromatic", "scale-achromatic",
        "scale-achromatic-window", or "solve".
    superbias_method : str
        NIRSpec superbias correction method. Options are "crds", "custom", and "custom-rescale".
    soss_timeseries : array-like(float), str, None
        Estimate of the normalized light curve, either 1D or 2D, or path to a file containing it.
    soss_timeseries_o2 : array-like(float), str, None
        Estimate of the normalized light curve for order 2, either 1D or 2D, or path to a file
        containing it.
    save_results : bool
        If True, save results of each step to file.
    pixel_masks : array-like(str), None
        For improved 1/f noise corecton. List of paths to outlier maps for each data segment. Can
        be 3D (nints, dimy, dimx), or 2D (dimy, dimx) files.
    force_redo : bool
        If True, redo steps even if outputs files are already present.
    hot_pixel_map : str, None
        Path to a hot pixel map, such as one produced by BadPixStep.
    flag_up_ramp : bool
        Whether to flag jumps up the ramp. This is the default flagging in the STScI pipeline. Note
        that this is broken as of jwst v1.12.5.
    rejection_threshold : int
        For jump detection; sigma threshold for a pixel to be considered an outlier.
    flag_in_time : bool
        If True, flag cosmic rays temporally in addition to the default up-the-ramp jump detection.
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
        Only necessary if do_plot is True. Show the diagnostic plots in addition to/instead of
        saving to file.
    soss_inner_mask_width : int
        For SOSS 1/f correction. For scale-achromatic, defines the width around the trace to mask.
        For windowed methods, defines the inner edge of the window.
    soss_outer_mask_width : int
        For SOSS 1/f correction. For windowed methods, defines the outer edge of the window. For
        solve, defines the width around the trace to use.
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
        step = DQInitStep(results, output_dir=outdir, hot_pixel_map=hot_pixel_map)
        results = step.run(save_results=save_results, force_redo=force_redo, **step_kwargs)

    # ===== Saturation Detection Step =====
    # Default DMS step.
    if 'SaturationStep' not in skip_steps:
        if 'SaturationStep' in kwargs.keys():
            step_kwargs = kwargs['SaturationStep']
        else:
            step_kwargs = {}
        step = SaturationStep(results, output_dir=outdir)
        results = step.run(save_results=save_results, force_redo=force_redo, **step_kwargs)

    # ===== Superbias Subtraction Step =====
    # Default/Custom DMS step.
    if 'SuperBiasStep' not in skip_steps:
        if 'SuperBiasStep' in kwargs.keys():
            step_kwargs = kwargs['SuperBiasStep']
        else:
            step_kwargs = {}
        step = SuperBiasStep(results, output_dir=outdir, centroids=centroids,
                             method=superbias_method)
        results = step.run(save_results=save_results, force_redo=force_redo, do_plot=do_plot,
                           show_plot=show_plot, **step_kwargs)

    # ===== Reference Pixel Correction Step =====
    # Default DMS step.
    if 'RefPixStep' not in skip_steps:
        if 'RefPixStep' in kwargs.keys():
            step_kwargs = kwargs['RefPixStep']
        else:
            step_kwargs = {}
        step = RefPixStep(results, output_dir=outdir)
        results = step.run(save_results=save_results, force_redo=force_redo, **step_kwargs)

    # ===== Dark Current Subtraction Step =====
    # Default DMS step.
    if 'DarkCurrentStep' not in skip_steps:
        if 'DarkCurrentStep' in kwargs.keys():
            step_kwargs = kwargs['DarkCurrentStep']
        else:
            step_kwargs = {}
        step = DarkCurrentStep(results, output_dir=outdir)
        results = step.run(save_results=save_results, force_redo=force_redo, **step_kwargs)

    if 'OneOverFStep' not in skip_steps:
        if mode == 'NIRISS/SOSS':
            # ===== Background Subtraction Step =====
            # Custom DMS step - imported from Stage2.
            if 'BackgroundStep' in kwargs.keys():
                step_kwargs = kwargs['BackgroundStep']
            else:
                step_kwargs = {}
            step = stage2.BackgroundStep(input_data=results, baseline_ints=baseline_ints,
                                         background_model=soss_background_model, output_dir=outdir)
            results = step.run(save_results=save_results, force_redo=force_redo, do_plot=do_plot,
                               show_plot=show_plot, **step_kwargs)
            results, soss_background_model = results

        # ===== 1/f Noise Correction Step =====
        # Custom DMS step.
        if 'OneOverFStep' in kwargs.keys():
            step_kwargs = kwargs['OneOverFStep']
        else:
            step_kwargs = {}
        step = OneOverFStep(results, output_dir=outdir, baseline_ints=baseline_ints,
                            pixel_masks=pixel_masks, centroids=centroids,
                            soss_background=soss_background_model, method=oof_method,
                            soss_timeseries=soss_timeseries, soss_timeseries_o2=soss_timeseries_o2)
        results = step.run(soss_inner_mask_width=soss_inner_mask_width,
                           soss_outer_mask_width=soss_outer_mask_width, save_results=save_results,
                           force_redo=force_redo, do_plot=do_plot, show_plot=show_plot,
                           nirspec_mask_width=nirspec_mask_width, **step_kwargs)

    # ===== Linearity Correction Step =====
    # Default DMS step.
    if 'LinearityStep' not in skip_steps:
        if 'LinearityStep' in kwargs.keys():
            step_kwargs = kwargs['LinearityStep']
        else:
            step_kwargs = {}
        step = LinearityStep(results, output_dir=outdir)
        results = step.run(save_results=save_results, force_redo=force_redo, do_plot=do_plot,
                           show_plot=show_plot, **step_kwargs)

    # ===== Jump Detection Step =====
    # Default/Custom DMS step.
    if 'JumpStep' not in skip_steps:
        if 'JumpStep' in kwargs.keys():
            step_kwargs = kwargs['JumpStep']
        else:
            step_kwargs = {}
        step = JumpStep(results, output_dir=outdir)
        results = step.run(save_results=save_results, force_redo=force_redo,
                           rejection_threshold=rejection_threshold, flag_in_time=flag_in_time,
                           flag_up_ramp=flag_up_ramp,
                           time_rejection_threshold=time_rejection_threshold, do_plot=do_plot,
                           show_plot=show_plot, **step_kwargs)

    # ===== Ramp Fit Step =====
    # Default DMS step.
    if 'RampFitStep' not in skip_steps:
        if 'RampFitStep' in kwargs.keys():
            step_kwargs = kwargs['RampFitStep']
        else:
            step_kwargs = {}
        step = RampFitStep(results, output_dir=outdir)
        results = step.run(save_results=save_results, force_redo=force_redo, **step_kwargs)

    # ===== Gain Scale Correcton Step =====
    # Default DMS step.
    if 'GainScaleStep' not in skip_steps:
        if 'GainScaleStep' in kwargs.keys():
            step_kwargs = kwargs['GainScaleStep']
        else:
            step_kwargs = {}
        step = GainScaleStep(results, output_dir=outdir)
        results = step.run(save_results=save_results, force_redo=force_redo, **step_kwargs)

    return results
