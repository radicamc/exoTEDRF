#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thurs Jul 21 17:33 2022

@author: MCR

Custom JWST DMS pipeline steps for Stage 3 (1D spectral extraction).
"""

from astropy.io import fits
import glob
import numpy as np
import os
import pandas as pd
import pastasoss
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit, least_squares
from scipy.signal import butter, filtfilt, correlate
import spectres
from spectres.spectral_resampling import make_bins
from tqdm import tqdm

from applesoss import applesoss

from jwst import datamodels
from jwst.pipeline import calwebb_spec2

from exotedrf import utils, plotting
from exotedrf.utils import fancyprint


class SpecProfileStep:
    """Wrapper around custom SpecProfile Reference Construction step.
    """

    def __init__(self, input_data, output_dir='./'):
        """Step initializer.

        Parameters
        ----------
        input_data : array-like(str), array-like(datamodel)
            List of paths to input data or the input data itself.
        output_dir : str
            Path to directory to which to save outputs.
        """

        # Set up easy attribute.
        self.output_dir = output_dir

        # Unpack input data files.
        self.datafiles = utils.sort_datamodels(input_data)

        # Get subarray identifier.
        if isinstance(self.datafiles[0], str):
            self.subarray = fits.getheader(self.datafiles[0])['SUBARRAY']
        else:
            self.subarray = self.datafiles[0].meta.subarray.name

    def run(self, force_redo=False, empirical=True):
        """Method to run the step.

        Parameters
        ----------
        force_redo : bool
            If True, run step even if output files are detected.
        empirical : bool
            If True, run APPLESOSS in empirical mode.

        Returns
        -------
        specprofile : str
            Path to file containing the 2D PSF model for each order.
        """

        all_files = glob.glob(self.output_dir + '*')
        # If an output file for this segment already exists, skip the step.
        expected_file = (self.output_dir + 'APPLESOSS_ref_2D_profile_{}_os1_pad20.fits'.format(self.subarray))
        if expected_file in all_files and force_redo is False:
            fancyprint('File {} already exists.'.format(expected_file))
            fancyprint('Skipping SpecProfile Reference Construction Step.')
            specprofile = expected_file
        # If no output files are detected, run the step.
        else:
            specprofile = specprofilestep(self.datafiles, output_dir=self.output_dir,
                                          empirical=empirical)
            specprofile = self.output_dir + specprofile

        return specprofile


class Extract1DStep:
    """Wrapper around default calwebb_spec2 1D Spectral Extraction step, with
    custom modifications.
    """

    def __init__(self, input_data, extract_method, st_teff=None, st_logg=None, st_met=None,
                 planet_letter='b', output_dir='./'):
        """Step initializer.

        Parameters
        ----------
        input_data : array-like(str), array-like(datamodel)
            List of paths to input data or the input data itself.
        extract_method : str
            1D extraction method to use; either "box", "optimal", or "atoca".
        st_teff : float
            Stellar effective temperature.
        st_logg : float
            Stellar log gravity.
        st_met : float
            Stellar metallicity.
        planet_letter : str
            Planet's letter designation.
        output_dir : str
            Path to directory to which to save outputs.
        """

        # Set up easy attributes.
        self.extract_method = extract_method
        self.tag = 'extract1dstep_{}.fits'.format(extract_method)
        self.output_dir = output_dir

        # Unpack input data files.
        self.datafiles = utils.sort_datamodels(input_data)
        self.fileroots = utils.get_filename_root(self.datafiles)
        self.fileroot_noseg = utils.get_filename_root_noseg(self.fileroots)

        # Set planet and star attributes.
        with utils.open_filetype(self.datafiles[0]) as datamodel:
            self.target_name = datamodel.meta.target.catalog_name
        self.pl_name = self.target_name + ' ' + planet_letter
        self.stellar_params = [st_teff, st_logg, st_met]

        # Get instrument.
        self.instrument = utils.get_instrument_name(self.datafiles[0])
        if self.instrument != 'NIRISS' and extract_method == 'atoca':
            fancyprint('ATOCA extraction selected but observation does not use NIRISS/SOSS. '
                       'Switching to box extraction.', msg_type='WARNING')
            self.extract_method = 'box'
        if extract_method in ['doublegauss', 'decontam']:
            raise ValueError('{} extraction is not supported in this branch.'
                             .format(extract_method))
        if self.instrument == 'NIRISS' and extract_method == 'optimal':
            fancyprint('Optimal extraction not available for NIRISS/SOSS. '
                       'Switching to box extraction.', msg_type='WARNING')
            self.extract_method = 'box'

    def run(self, extract_width=40, extract_width_soss2=None, soss_specprofile=None, centroids=None,
            save_results=True, force_redo=False, do_plot=False, show_plot=False, deepframe=None,
            use_pastasoss=False, soss_estimate=None, opt_max_iter=25, opt_var_thresh=25,
            allow_miri_slope=False, saturation_rescue=False, mask_do_not_use_pixels=True):
        """Method to run the step.

        Parameters
        ----------
        extract_width : int, tuple(float, float)
            Full width of extraction aperture to use. A two-element tuple is interpreted as an
            asymmetric `(lower_width, upper_width)` aperture for box extraction.
        extract_width_soss2 : int, tuple(float, float), None
            Full width of extraction aperture to use for SOSS order 2. A two-element tuple is
            interpreted as an asymmetric `(lower_width, upper_width)` aperture for box extraction.
        soss_specprofile : str, None
            Path to specprofile file.
        centroids : str, None
            Path to file containing centroids for each order.
        save_results : bool
            If True, save results.
        force_redo : bool
            If True, run step even if output files are detected.
        do_plot : bool
            If True, do step diagnostic plot.
        show_plot : bool
            If True, show the step diagnostic plot.
        deepframe : str, None
            Path to file containing a median stack of the observations.
        use_pastasoss : bool
            If True, use pastasoss to esimate trace positions and wavelength solution.
        soss_estimate : str, None
            Path to file containing the soss_estimate for atoca extractions.
        opt_max_iter : int
            Maximum number of outlier rejection iterations to perform during optimal extraction.
        opt_var_thresh : int
            Variance threshold for a pixel to be flagged as an outlier during optimal exraction.
        allow_miri_slope : bool
            If True, allow the MIRI centroids to be sloped.
        saturation_rescue : bool
            If True for NIRISS/SOSS box extraction, keep post-RampFit pixels whose ramps were only
            partially saturated so RampFit's pre-saturation slope estimate can be extracted.
        mask_do_not_use_pixels : bool
            If True, NaN DO_NOT_USE pixels before box extraction in addition to saturation handling.


        Returns
        -------
        spectra : dict
            1D stellar spectra at the native detector resolution.
        """

        fancyprint('Starting 1D extraction using the {} method.'.format(self.extract_method))

        # Initialize loop and storange variables.
        all_files = glob.glob(self.output_dir + '*')
        expected_file = (self.output_dir + self.target_name + '_' + self.extract_method +
                         '_spectra_fullres.fits')
        # If an output file already exists, skip the step.
        if expected_file in all_files and force_redo is False:
            fancyprint('File {} already exists.'.format(expected_file))
            fancyprint('Skipping Extract 1D Step.')
            spectra = expected_file
        # If no output file is detected, run the step.
        else:
            # Option 1: ATOCA extraction - SOSS only.
            if self.extract_method == 'atoca':
                if soss_specprofile is None:
                    raise ValueError('specprofile reference file must be provided for ATOCA '
                                     'extraction.')
                if extract_width == 'optimize':
                    raise ValueError('Aperture optimization not possible with ATOCA extraction.')
                if extract_width_soss2 is not None:
                    fancyprint('Order 2 cannot use a different width for ATOCA extraction.',
                               msg_type='WARNING')

                results = atoca_extract_soss(self.datafiles, soss_specprofile,
                                             output_dir=self.output_dir, save_results=save_results,
                                             extract_width=extract_width, fileroots=self.fileroots,
                                             soss_estimate=soss_estimate)

            # Option 2: Simple aperture extraction - any instrument.
            elif self.extract_method == 'box':
                # We need a deepframe here.
                if deepframe is None:
                    raise ValueError('Deepframe must be provided for box extraction.')
                # If file path is passed, open it.
                if isinstance(deepframe, str):
                    deepframe = fits.getdata(deepframe)

                # Need to make sure that we have the centroids. Passed centroids always take
                # precedence.
                if centroids is None:
                    centroids = trace_spectrum(self.datafiles, deepframe=deepframe,
                                               output_dir=self.output_dir,
                                               save_results=save_results,
                                               fileroot_noseg=self.fileroot_noseg,
                                               allow_miri_slope=allow_miri_slope,
                                               extract_width=extract_width,
                                               extract_width_soss2=extract_width_soss2,
                                               do_plot=do_plot, show_plot=show_plot)
                # If file path is passed, open it.
                if isinstance(centroids, str):
                    centroids = pd.read_csv(centroids, comment='#')

                mask_saturated_pixels = True
                if self.instrument == 'NIRISS' and saturation_rescue is True:
                    fancyprint('NIRISS saturation rescue enabled; keeping post-RampFit pixels '
                               'with SATURATED DQ flags for box extraction.')
                    mask_saturated_pixels = False

                if self.instrument == 'NIRISS':
                    results = box_extract_soss(self.datafiles, centroids, extract_width,
                                               soss_width_o2=extract_width_soss2, do_plot=do_plot,
                                               show_plot=show_plot, save_results=save_results,
                                               output_dir=self.output_dir,
                                               mask_saturated_pixels=mask_saturated_pixels,
                                               mask_do_not_use_pixels=mask_do_not_use_pixels)
                elif self.instrument == 'NIRSPEC':
                    results = box_extract_nirspec(self.datafiles, centroids, extract_width,
                                                  do_plot=do_plot, show_plot=show_plot,
                                                  save_results=save_results,
                                                  output_dir=self.output_dir,
                                                  mask_saturated_pixels=mask_saturated_pixels,
                                                  mask_do_not_use_pixels=mask_do_not_use_pixels)
                else:
                    results = box_extract_miri(self.datafiles, centroids, extract_width,
                                               do_plot=do_plot, show_plot=show_plot,
                                               save_results=save_results,
                                               output_dir=self.output_dir,
                                               mask_saturated_pixels=mask_saturated_pixels,
                                               mask_do_not_use_pixels=mask_do_not_use_pixels)
                if extract_width == 'optimize':
                    # Get optimized width.
                    extract_width = int(results[-1])
                results = results[:-1]

            # Option 3: Optimal extraction - NIRSpec or MIRI.
            elif self.extract_method == 'optimal':
                # We need a deepframe here.
                if deepframe is None:
                    raise ValueError('Deepframe must be provided for optimal extraction.')
                # If file path is passed, open it.
                if isinstance(deepframe, str):
                    deepframe = fits.getdata(deepframe)

                # Need to make sure that we have the centroids. Passed centroids always take
                # precedence.
                if centroids is None:
                    centroids = trace_spectrum(self.datafiles, deepframe=deepframe,
                                               output_dir=self.output_dir,
                                               save_results=save_results,
                                               fileroot_noseg=self.fileroot_noseg,
                                               allow_miri_slope=allow_miri_slope,
                                               extract_width=extract_width,
                                               extract_width_soss2=extract_width_soss2,
                                               do_plot=do_plot, show_plot=show_plot)
                # If file path is passed, open it.
                if isinstance(centroids, str):
                    centroids = pd.read_csv(centroids, comment='#')

                if self.instrument == 'NIRSPEC':
                    results = optimal_extract_nirspec(self.datafiles, deepframe, centroids,
                                                      extract_width, max_iter=opt_max_iter,
                                                      var_thresh=opt_var_thresh)
                else:
                    results = optimal_extract_miri(self.datafiles, deepframe, centroids,
                                                   extract_width, max_iter=opt_max_iter,
                                                   var_thresh=opt_var_thresh)

                extract_width = 'N/A'

            # Raise exception otherwise.
            else:
                raise ValueError('Invalid extraction method')

            # Do step plot if requested - only for atoca.
            if do_plot is True and self.extract_method == 'atoca':
                if save_results is True:
                    plot_file = self.output_dir + self.tag.replace('.fits', '.png')
                else:
                    plot_file = None
                models = []
                for name in self.fileroots:
                    models.append(self.output_dir + name + 'SossExtractModel.fits')
                plotting.make_decontamination_plot(self.datafiles, models, outfile=plot_file,
                                                   show_plot=show_plot)

            # Save the final extraction parameters.
            extract_params = {'extract_width': _format_extract_width(extract_width),
                              'method': self.extract_method}
            # Get timestamps and pupil wheel position.
            for i, datafile in enumerate(self.datafiles):
                with utils.open_filetype(datafile) as file:
                    # Pipeline data says timestamps are BJD, but they are actually MJD (I think).
                    this_time = file.int_times['int_mid_BJD_TDB']
                if i == 0:
                    times = this_time
                    pwcpos = file.meta.instrument.pupil_position
                else:
                    times = np.concatenate([times, this_time])

            # Clip outliers, refine wavelength solution, and format extracted
            # spectra.
            st_teff, st_logg, st_met = self.stellar_params
            if self.instrument == 'NIRISS':
                spectra = format_soss_spectra(results, times, extract_params, self.pl_name,
                                              st_teff, st_logg, st_met, pwcpos=pwcpos,
                                              output_dir=self.output_dir, save_results=save_results,
                                              use_pastasoss=use_pastasoss)
            elif self.instrument == 'NIRSPEC':
                detector = utils.get_nrs_detector_name(self.datafiles[0])
                spectra = format_nirspec_spectra(results, times, extract_params, self.pl_name,
                                                 detector, st_teff, st_logg, st_met,
                                                 output_dir=self.output_dir,
                                                 save_results=save_results)
            else:
                spectra = format_miri_spectra(results, times, extract_params, self.pl_name,
                                              st_teff, st_logg, st_met, output_dir=self.output_dir,
                                              save_results=save_results)

        return spectra


def specprofilestep(datafiles, empirical=True, output_dir='./'):
    """Wrapper around the APPLESOSS module to construct a specprofile reference file tailored
    to the particular TSO being analyzed.

    Parameters
    ----------
    datafiles : array-like[str], array-like[jwst.RampModel]
        Input datamodels or paths to datamodels for each segment.
    empirical : bool
        If True, construct profiles using only the data. If False, fall back on WebbPSF for the
        trace wings. Note: The current WebbPSF wings are known to not accurately match observations.
        This mode is therefore not advised.
    output_dir : str
        Directory to which to save outputs.

    Returns
    -------
    filename : str
        Name of the output file.
    """

    fancyprint('Starting SpecProfile Construction Step.')
    datafiles = np.atleast_1d(datafiles)

    # Get the most up to date trace table file.
    step = calwebb_spec2.extract_1d_step.Extract1dStep()
    tracetable = step.get_reference_file(datafiles[0], 'spectrace')
    # Get the most up to date 2D wavemap file.
    step = calwebb_spec2.extract_1d_step.Extract1dStep()
    wavemap = step.get_reference_file(datafiles[0], 'wavemap')

    # Create a new deepstack but using all integrations, not just the baseline.
    for i, file in enumerate(datafiles):
        if isinstance(file, str):
            data = fits.getdata(file)
            if i == 0:
                cube = data
            else:
                cube = np.concatenate([cube, data])
        else:
            data = datamodels.open(file)
            if i == 0:
                cube = data.data
            else:
                cube = np.concatenate([cube, data.data])
            data.close()
    deepstack = utils.make_deepstack(cube)

    # Initialize and run the APPLESOSS module with the median stack.
    spat_prof = applesoss.EmpiricalProfile(deepstack, tracetable=tracetable, wavemap=wavemap,
                                           pad=20)
    if empirical is False:
        # Get the date of the observations to use the calculated WFE models from that time.
        obs_date = fits.getheader(datafiles[0])['DATE-OBS']
        spat_prof.build_empirical_profile(verbose=0, empirical=False, wave_increment=0.1,
                                          obs_date=obs_date)
    else:
        spat_prof.build_empirical_profile(verbose=0)

    # Save results to file (non-optional).
    if np.shape(deepstack)[0] == 96:
        subarray = 'SUBSTRIP96'
    else:
        subarray = 'SUBSTRIP256'
    filename = spat_prof.write_specprofile_reference(subarray, output_dir=output_dir)

    return filename


def atoca_extract_soss(datafiles, specprofile, output_dir='./', save_results=True, extract_width=40,
                       soss_estimate=None, fileroots=None):
    """Perform an extraction of SOSS observations using the ATOCA algorithm.

    Parameters
    ----------
    datafiles : array-like(datamodel)
        Input data models.
    specprofile : str
        Path to specprofile reference file generated with APPLESOSS.
    output_dir : str
        Directory to which to save outputs.
    save_results : bool
        If True, save results to file.
    extract_width : int
        Full extraction width, in pixels.
    soss_estimate : str, None
        Path to soss estimate file.
    fileroots : array-like(str), None
        Filename roots.

    Returns
    -------
    results : list(datamodel)
        ATOCA extracted spectra.
    """

    results = []
    to_extract = {}
    first_time = True
    for i, file in enumerate(datafiles):
        to_extract['{}'.format(i)] = file
    while len(to_extract) != 0:
        extracted = []
        for i in to_extract.keys():
            segment = to_extract[i]
            # Initialize extraction parameters for ATOCA.
            soss_modelname = fileroots[int(i)][:-1]
            # Perform the extraction.
            step = calwebb_spec2.extract_1d_step.Extract1dStep()
            try:
                res = step.call(segment, output_dir=output_dir, save_results=save_results,
                                subtract_background=False,
                                soss_bad_pix='model', soss_width=extract_width,
                                soss_modelname=soss_modelname, override_specprofile=specprofile,
                                soss_estimate=soss_estimate)
                results.append(res)
                # Note that this segment was extracted correctly.
                extracted.append(i)
                # The first time that an extraction is successful, create a soss_estimate if one
                # does not already exist.
                if first_time is True and soss_estimate is None:
                    atoca_spectra = output_dir + fileroots[int(i)] + 'AtocaSpectra.fits'
                    soss_estimate = get_soss_estimate(atoca_spectra, output_dir=output_dir)
                    first_time = False
            # When using ATOCA, sometimes a very specific error pops up when an initial estimate of
            # the stellar spectrum cannot be obtained. This is needed to establish the wavelength
            # grid (which has a varying resolution to better capture sharp features in stellar
            # spectra). In these cases, the SOSS estimate provides information to create a
            # wavelength grid.
            except Exception as err:
                if str(err) == '(m>k) failed for hidden m: fpcurf0:m=0':
                    # If every segment has been tested and none work, just fail.
                    if int(i) == len(datafiles) and len(extracted) == 0:
                        fancyprint('No segments could be properly extracted.', msg_type='Error')
                        raise err
                    # If there's still hope, then just skip this segment for now and move onto the
                    # next one.
                    else:
                        fancyprint('Initial flux estimate failed, and no soss estimate provided. '
                                   'Moving to next segment.', msg_type='WARNING')
                        continue
                # If any other error pops up, raise it.
                else:
                    raise err
        # Remove the extracted segments from the list of ones to extract.
        for seg in extracted:
            to_extract.pop(seg)

    # Sort the segments in chronological order, in case they were processed out of order.
    seg_nums = [seg.meta.exposure.segment_number for seg in results]
    ii = np.argsort(seg_nums)
    results = np.array(results)[ii]

    return results


def _get_dq_mask(dq, data_shape, bits):
    """Return a boolean mask for selected DQ bits, broadcast to the science data shape."""

    if dq is None:
        return None

    dq = np.asarray(dq)
    bitmask = np.uint32(0)
    for bit in np.atleast_1d(bits):
        bitmask = np.bitwise_or(bitmask, np.uint32(bit))
    dq_flagged = (dq.astype(np.uint32) & bitmask) != 0

    if dq_flagged.shape == data_shape:
        return dq_flagged

    if len(data_shape) == 3:
        if dq_flagged.ndim == 2 and dq_flagged.shape == data_shape[-2:]:
            return np.broadcast_to(dq_flagged[np.newaxis, :, :], data_shape)
        if dq_flagged.ndim == 3 and dq_flagged.shape[0] == data_shape[0]:
            if dq_flagged.shape[-2:] == data_shape[-2:]:
                return dq_flagged
        if dq_flagged.ndim == 4 and dq_flagged.shape[0] == data_shape[0]:
            if dq_flagged.shape[-2:] == data_shape[-2:]:
                return np.any(dq_flagged, axis=1)

    return None


def _mask_dq_pixels(data, err, dq, source_label, mask_saturated_pixels=True,
                    mask_do_not_use_pixels=True):
    """NaN unusable DQ pixels so box extraction ignores them."""

    data = np.array(data, dtype=float, copy=True)
    err = np.array(err, dtype=float, copy=True)

    # Mask only the requested DQ classes. Saturated pixels can be kept for rescue runs.
    bits = []
    if mask_do_not_use_pixels is True:
        bits.append(1)
    if mask_saturated_pixels is True:
        bits.append(2)
    if len(bits) == 0:
        return data, err, 0

    bad = _get_dq_mask(dq, data.shape, bits)
    if bad is None:
        return data, err, 0

    count = int(np.sum(bad))
    if count > 0:
        data[bad] = np.nan
        err[bad] = np.nan
        do_not_use = _get_dq_mask(dq, data.shape, [1])
        saturated = _get_dq_mask(dq, data.shape, [2])
        do_not_use_count = int(np.sum(do_not_use)) if do_not_use is not None else 0
        saturated_count = int(np.sum(saturated)) if saturated is not None else 0
        if mask_saturated_pixels is True:
            fancyprint('Masked {} DQ pixels for box extraction in {} '
                       '({} DO_NOT_USE, {} SATURATED).'
                       .format(count, source_label, do_not_use_count, saturated_count))
        else:
            fancyprint('Masked {} DQ pixels for box extraction in {} '
                       '({} DO_NOT_USE; {} SATURATED kept for rescue).'
                       .format(count, source_label, do_not_use_count, saturated_count))

    return data, err, count


def _load_box_extraction_cubes(datafiles, mask_saturated_pixels=True,
                               mask_do_not_use_pixels=True):
    """Load science/error cubes and NaN selected DQ pixels before box extraction."""

    datafiles = np.atleast_1d(datafiles)
    total_saturated = 0
    for i, file in enumerate(datafiles):
        if isinstance(file, str):
            data = fits.getdata(file)
            err = fits.getdata(file, 2)
            try:
                dq = fits.getdata(file, 3)
            except (IndexError, KeyError, OSError):
                dq = None
            source_label = os.path.basename(file)
        else:
            with utils.open_filetype(file) as datamodel:
                data = datamodel.data
                err = datamodel.err
                dq = getattr(datamodel, 'dq', None)
                if dq is None:
                    dq = getattr(datamodel, 'groupdq', None)
            source_label = 'datamodel segment {}'.format(i)

        data, err, count = _mask_dq_pixels(
            data, err, dq, source_label, mask_saturated_pixels=mask_saturated_pixels,
            mask_do_not_use_pixels=mask_do_not_use_pixels
        )
        total_saturated += count
        if i == 0:
            cube = data
            ecube = err
        else:
            cube = np.concatenate([cube, data])
            ecube = np.concatenate([ecube, err])

    if total_saturated > 0:
        fancyprint('Box extraction will ignore {} total DQ pixels.'
                   .format(total_saturated))

    return cube, ecube


def box_extract_miri(datafiles, centroids, extract_width, do_plot=False, show_plot=False,
                     save_results=True, output_dir='./', mask_saturated_pixels=True,
                     mask_do_not_use_pixels=True):
    """Perform a simple box aperture extraction on MIRI.

    Parameters
    ----------
    datafiles : array-like[str], array-like[jwst.RampModel]
        Input datamodels or paths to datamodels for each segment.
    centroids : dict
        Dictionary of centroid positions for all SOSS orders.
    extract_width : int, tuple(float, float), str
        Width of extraction box. Or 'optimize'. A two-element tuple is interpreted as an
        asymmetric `(lower_width, upper_width)` aperture.
    do_plot : bool
        If True, do the step diagnostic plot.
    show_plot : bool
        If True, show the step diagnostic plot instead of/in addition to
        saving it to file.
    output_dir : str
        Directory to which to output results.
    save_results : bool
        If True, save results to file.

    Returns
    -------
    wave : ndarray[float]
        2D wavelength solution.
    flux : ndarray[float]
        2D extracted flux.
    ferr: ndarray[float]
        2D flux errors.
    extract_width : int
        Optimized aperture width.
    """

    datafiles = np.atleast_1d(datafiles)
    cube, ecube = _load_box_extraction_cubes(datafiles,
                                             mask_saturated_pixels=mask_saturated_pixels,
                                             mask_do_not_use_pixels=mask_do_not_use_pixels)

    # Get centroid positions.
    x1, y1 = centroids['xpos'].values, centroids['ypos'].values

    # ===== Optimize Aperture Width =====
    if extract_width == 'optimize':
        fancyprint('Optimizing extraction width...')
        # Extract with a variety of widths and find the one that minimizes the white light curve
        # scatter.
        scatter = []
        for w in tqdm(range(2, 13)):
            flux = do_box_extraction(cube.transpose(0, 2, 1), ecube.transpose(0, 2, 1), x1,
                                     width=w, progress=False, extract_start=int(np.min(y1)),
                                     extract_end=int(np.max(y1)))[0]
            wlc = np.nansum(flux, axis=1)
            s = np.median(np.abs(0.5*(wlc[0:-2] + wlc[2:]) - wlc[1:-1]))
            scatter.append(s)
        scatter = np.array(scatter)
        # Find the width that minimizes the scatter.
        ii = np.argmin(scatter)
        extract_width = np.linspace(2, 12, 11)[ii]
        fancyprint('Using width of {} pxiels.'.format(int(extract_width)))

        # Do diagnostic plot if requested.
        if do_plot is True:
            if save_results is True:
                outfile = output_dir + 'aperture_optimization.png'
            else:
                outfile = None
            plotting.make_soss_width_plot(np.linspace(2, 12, 11), scatter, ii, outfile=outfile,
                                          show_plot=show_plot)

    # ===== Extraction ======
    # Do the extraction.
    fancyprint('Performing simple aperture extraction.')
    flux, ferr = do_box_extraction(cube.transpose(0, 2, 1), ecube.transpose(0, 2, 1), x1,
                                   width=extract_width, extract_start=int(np.min(y1)),
                                   extract_end=int(np.max(y1)))

    # Get default 2D wavelength solution.
    wave = get_wave_miri(datafiles[0], centroids, cube.shape[0], cube.shape[1])

    return wave, flux, ferr, extract_width


def box_extract_nirspec(datafiles, centroids, extract_width, do_plot=False, show_plot=False,
                        save_results=True, output_dir='./', mask_saturated_pixels=True,
                        mask_do_not_use_pixels=True):
    """Perform a simple box aperture extraction on NIRSpec.

    Parameters
    ----------
    datafiles : array-like[str], array-like[jwst.RampModel]
        Input datamodels or paths to datamodels for each segment.
    centroids : dict
        Dictionary of centroid positions for all SOSS orders.
    extract_width : int, tuple(float, float), str
        Width of extraction box. Or 'optimize'. A two-element tuple is interpreted as an
        asymmetric `(lower_width, upper_width)` aperture.
    do_plot : bool
        If True, do the step diagnostic plot.
    show_plot : bool
        If True, show the step diagnostic plot instead of/in addition to
        saving it to file.
    output_dir : str
        Directory to which to output results.
    save_results : bool
        If True, save results to file.

    Returns
    -------
    wave : ndarray[float]
        2D wavelength solution.
    flux : ndarray[float]
        2D extracted flux.
    ferr: ndarray[float]
        2D flux errors.
    extract_width : int
        Optimized aperture width.
    """

    datafiles = np.atleast_1d(datafiles)
    det = utils.get_nrs_detector_name(datafiles[0])
    cube, ecube = _load_box_extraction_cubes(datafiles,
                                             mask_saturated_pixels=mask_saturated_pixels,
                                             mask_do_not_use_pixels=mask_do_not_use_pixels)

    # Get centroid positions.
    x1, y1 = centroids['xpos'].values, centroids['ypos'].values

    # ===== Optimize Aperture Width =====
    if extract_width == 'optimize':
        fancyprint('Optimizing extraction width...')
        # Extract with a variety of widths and find the one that minimizes the white light curve
        # scatter.
        scatter = []
        if det == 'nrs1':
            grating = utils.get_nrs_grating(datafiles[0])
            if grating == 'G395H':
                xstart = 500  # Trace starts at pixel ~500 for G395M
            elif grating == 'G395M':
                xstart = 200  # Trace starts at pixel ~200 for G395M
            elif grating == 'PRISM':
                xstart = 14  # Trace starts at pixel ~14 for PRISM
            else:
                raise ValueError('Unknown NIRSpec grating used...')
        else:
            xstart = 0
        for w in tqdm(range(1, 12)):
            flux = do_box_extraction(cube, ecube, y1, width=w, progress=False,
                                     extract_start=xstart)[0]
            wlc = np.nansum(flux, axis=1)
            s = np.median(np.abs(0.5*(wlc[0:-2] + wlc[2:]) - wlc[1:-1]))
            scatter.append(s)
        scatter = np.array(scatter)
        # Find the width that minimizes the scatter.
        ii = np.argmin(scatter)
        extract_width = np.linspace(1, 11, 11)[ii]
        fancyprint('Using width of {} pxiels.'.format(int(extract_width)))

        # Do diagnostic plot if requested.
        if do_plot is True:
            if save_results is True:
                outfile = output_dir + 'aperture_optimization.png'
            else:
                outfile = None
            plotting.make_soss_width_plot(np.linspace(1, 11, 11), scatter, ii, outfile=outfile,
                                          show_plot=show_plot)

    # ===== Extraction ======
    # Do the extraction.
    fancyprint('Performing simple aperture extraction.')
    det = utils.get_nrs_detector_name(datafiles[0])
    subarray = utils.get_soss_subarray(datafiles[0])
    grating = utils.get_nrs_grating(datafiles[0])
    xstart = utils.get_nrs_trace_start(det, subarray, grating)
    flux, ferr = do_box_extraction(cube, ecube, y1, width=extract_width, extract_start=xstart)

    # Get default 2D wavelength solution.
    wave = get_wave_nirspec(datafiles[0], centroids, cube.shape[0], cube.shape[2])

    return wave, flux, ferr, extract_width


def double_gaussian_extract_nirspec(datafiles, centroids, extract_width, separation_guess=4.0,
                                    fit_background=True, main_component=1, deepframe=None,
                                    do_plot=False, show_plot=False, save_results=True,
                                    output_dir='./'):
    """Extract both members of an overlapping NIRSpec binary with a two-Gaussian profile fit.

    Parameters
    ----------
    datafiles : array-like[str], array-like[jwst.RampModel]
        Input datamodels or paths to datamodels for each segment.
    centroids : dict
        Trace centroids. The supplied `ypos` trace is treated as the midpoint between the two
        stellar traces.
    extract_width : int, tuple(float, float), dict
        Width of the extraction aperture around the midpoint trace.
    separation_guess : float
        Initial guess for the separation between the lower and upper Gaussian components, in
        pixels.
    fit_background : bool
        If True, include a constant background term in the spatial profile fit.
    main_component : int
        Which fitted component to treat as the primary extracted target. `1` selects the lower
        trace and `2` selects the upper trace.
    deepframe : array-like[float], None
        Median-combined 2D frame to use for plotting diagnostics. If None and `do_plot` is True,
        a median frame will be built from the extracted cube.
    do_plot : bool
        If True, generate the double-Gaussian extraction diagnostic plot.
    show_plot : bool
        If True, show the diagnostic plot instead of/in addition to saving it.
    save_results : bool
        If True, save the diagnostic plot to file when plotting is requested.
    output_dir : str
        Directory to which diagnostic products should be written.

    Returns
    -------
    result : dict
        Dictionary containing the primary and companion spectra and fit diagnostics.
    """

    if main_component not in [1, 2]:
        raise ValueError('main_component must be either 1 (lower trace) or 2 (upper trace).')

    datafiles = np.atleast_1d(datafiles)
    det = utils.get_nrs_detector_name(datafiles[0])
    for i, file in enumerate(datafiles):
        if isinstance(file, str):
            data = fits.getdata(file)
            err = fits.getdata(file, 2)
        else:
            with utils.open_filetype(file) as datamodel:
                data = datamodel.data
                err = datamodel.err
        if i == 0:
            cube = data
            ecube = err
        else:
            cube = np.concatenate([cube, data])
            ecube = np.concatenate([ecube, err])

    x1, y1 = centroids['xpos'].values, centroids['ypos'].values
    subarray = utils.get_soss_subarray(datafiles[0])
    grating = utils.get_nrs_grating(datafiles[0])
    xstart = utils.get_nrs_trace_start(det, subarray, grating)

    fancyprint('Performing double-Gaussian NIRSpec extraction.')
    flux1, ferr1, flux2, ferr2, profile_params = do_two_gaussian_extraction(
        cube, ecube, y1, width=extract_width, extract_start=xstart,
        separation_guess=separation_guess, fit_background=fit_background
    )

    if main_component == 1:
        flux, ferr = flux1, ferr1
        flux_companion, ferr_companion = flux2, ferr2
    else:
        flux, ferr = flux2, ferr2
        flux_companion, ferr_companion = flux1, ferr1

    wave = get_wave_nirspec(datafiles[0], centroids, cube.shape[0], cube.shape[2])

    if do_plot is True:
        if deepframe is None:
            deepframe = np.nanmedian(cube, axis=0)
        if save_results is True:
            outfile = output_dir + 'doublegauss_diagnostics_{}.png'.format(det)
        else:
            outfile = None
        plotting.make_doublegauss_nirspec_plot(deepframe, x1, y1, extract_width,
                                               profile_params, extract_start=xstart,
                                               outfile=outfile, show_plot=show_plot)

    return {'wave': wave, 'flux': flux, 'ferr': ferr,
            'flux_companion': flux_companion, 'ferr_companion': ferr_companion,
            'profile': profile_params}


def _default_nirspec_decontam_separation(detector, grating=None):
    """Return notebook-inspired default separations for NIRSpec contamination modeling."""

    detector = str(detector).lower()
    if detector == 'nrs1':
        return 3.7
    if detector == 'nrs2':
        return 3.2
    if grating in ['G395H', 'G395M']:
        return 3.5
    return 3.0


def _get_notebook_nirspec_decontam_setup(detector, separation_guess=None, oversample=10,
                                         min_separation=3.0):
    """Return detector-specific defaults used by the contamination notebook."""

    detector = str(detector).lower()
    if separation_guess in [None, 'None', 'null', '']:
        if detector == 'nrs1':
            separation_guess = 3.7
        elif detector == 'nrs2':
            separation_guess = 3.2
        else:
            separation_guess = _default_nirspec_decontam_separation(detector)

    fit_start_col = 500 if detector == 'nrs1' else 0
    return {
        'fit_start_col': fit_start_col,
        'amp_target_0': 3000.0,
        'amp_companion_0': 500.0,
        'sigma_0_os': 1.0 * oversample,
        'separation_guess_os': float(separation_guess) * oversample,
        'min_separation_os': float(min_separation) * oversample,
    }


def _prefer_nirspec_pcareconstruct_files(datafiles):
    """Prefer PCA-reconstructed files for notebook-style decontamination when available."""

    preferred = []
    used_pca = False
    for datafile in np.atleast_1d(datafiles):
        if isinstance(datafile, str) is not True:
            preferred.append(datafile)
            continue

        replacement = None
        if '_badpixstep.fits' in datafile:
            candidate = datafile.replace('_badpixstep.fits', '_pcareconstructstep.fits')
            if os.path.exists(candidate):
                replacement = candidate
        if replacement is None:
            replacement = datafile
        elif replacement != datafile:
            used_pca = True
        preferred.append(replacement)

    if used_pca is True:
        fancyprint('Using *_pcareconstructstep.fits inputs to match the contamination notebook.')

    return np.asarray(preferred, dtype=object)


def _fit_nirspec_contamination_profile_notebook(profile_native, primary_guess, detector,
                                                separation_guess, oversample=10,
                                                min_separation=3.0):
    """Literal shared-sigma two-Gaussian column fit used in the contamination notebook."""

    profile_native = np.asarray(profile_native, dtype=float)
    if np.any(np.isfinite(profile_native)) is not True:
        return None

    dimy = len(profile_native)
    y_native = np.arange(dimy, dtype=float)
    x_fit = np.arange((dimy - 1) * oversample + 1, dtype=float)
    profile_os = np.interp(x_fit / oversample, y_native, np.nan_to_num(profile_native, nan=0.0))

    primary_guess_os = float(primary_guess) * oversample
    setup = _get_notebook_nirspec_decontam_setup(
        detector,
        separation_guess=separation_guess, oversample=oversample,
        min_separation=min_separation
    )
    min_sep_os = setup['min_separation_os']

    def notebook_model(x, amp1, mu1, sigma1, amp2, mu2):
        assert mu2 >= mu1 + min_sep_os
        return (amp1 * np.exp(-(x - mu1) ** 2 / (2. * sigma1 ** 2)) +
                amp2 * np.exp(-(x - mu2) ** 2 / (2. * sigma1 ** 2)))

    p0 = [
        setup['amp_target_0'],
        primary_guess_os,
        setup['sigma_0_os'],
        setup['amp_companion_0'],
        primary_guess_os + setup['separation_guess_os']
    ]

    try:
        coeff, _ = curve_fit(notebook_model, x_fit, profile_os, p0=p0, maxfev=10000)
    except (RuntimeError, ValueError, AssertionError):
        return None

    model_os = notebook_model(x_fit, *coeff)
    rms = np.sqrt(np.nanmean((profile_os - model_os) ** 2))
    return {
        'coeff': np.asarray(coeff, dtype=float),
        'profile_os': profile_os,
        'model_os': model_os,
        'fit_rms': rms,
        'x_fit': x_fit,
    }


def _fit_nirspec_contamination_profile(y_os, profile_os, primary_guess, separation_guess,
                                       min_separation=3.0, max_separation=5.5,
                                       oversample=10, prev_params=None, fit_background=False,
                                       primary_tolerance=1.5):
    """Fit the notebook-style target+contaminant profile on an oversampled NIRSpec column."""

    mask = np.isfinite(y_os) & np.isfinite(profile_os)
    if np.sum(mask) < 6:
        return None

    yy = y_os[mask]
    pp = profile_os[mask]
    background0 = float(np.nanmedian(np.concatenate([pp[:2], pp[-2:]]))) if len(pp) >= 4 else 0.0
    primary_guess_os = float(primary_guess) * oversample
    sep_guess_os = float(separation_guess) * oversample
    min_sep_os = float(min_separation) * oversample
    max_sep_os = float(max_separation) * oversample
    tol_os = float(primary_tolerance) * oversample

    if fit_background is not True:
        def notebook_model(x, amp1, mu1, sigma1, amp2, mu2):
            assert mu2 >= mu1 + min_sep_os
            return (amp1 * np.exp(-(x - mu1) ** 2 / (2. * sigma1 ** 2)) +
                    amp2 * np.exp(-(x - mu2) ** 2 / (2. * sigma1 ** 2)))

        amp1_0 = max(float(np.nanmax(pp)), 1.0)
        amp2_0 = max(0.15 * amp1_0, 1.0)
        p0 = [amp1_0, primary_guess_os, 1.0 * oversample, amp2_0,
              primary_guess_os + sep_guess_os]

        try:
            coeff, _ = curve_fit(notebook_model, yy, pp, p0=p0, maxfev=10000)
        except (RuntimeError, ValueError, AssertionError):
            return None

        if coeff[4] < coeff[1] + min_sep_os:
            return None

        return np.array([coeff[0], coeff[1], coeff[2], coeff[3], coeff[4] - coeff[1]],
                        dtype=float)

    lower_bounds = [0, primary_guess_os - tol_os, 0.4 * oversample, 0, min_sep_os]
    upper_bounds = [np.inf, primary_guess_os + tol_os, 3.5 * oversample, np.inf, max_sep_os]
    if fit_background is True:
        lower_bounds.append(-np.inf)
        upper_bounds.append(np.inf)

    if prev_params is None:
        amp1_0 = max(float(np.nanmax(pp)) - background0, 0)
        companion_side = yy >= primary_guess_os + min_sep_os / 2
        if np.any(companion_side):
            amp2_0 = max(float(np.nanmax(pp[companion_side])) - background0, 0)
        else:
            amp2_0 = max(amp1_0 * 0.15, 0)
        sigma_0 = 1.0 * oversample
        p0 = [amp1_0, primary_guess_os, sigma_0, amp2_0, sep_guess_os]
        if fit_background is True:
            p0.append(background0)
    else:
        p0 = np.array(prev_params, dtype=float)

    p0 = np.clip(np.asarray(p0, dtype=float), lower_bounds, upper_bounds)

    def residuals(params):
        mu_target = params[1]
        sigma = params[2]
        mu_comp = mu_target + params[4]
        model = (params[0] * _gaussian_profile(yy, mu_target, sigma) +
                 params[3] * _gaussian_profile(yy, mu_comp, sigma))
        if fit_background is True:
            model += params[5]
        return pp - model

    try:
        fit = least_squares(residuals, p0, bounds=(lower_bounds, upper_bounds))
    except ValueError:
        return None

    if fit.success is not True:
        return None

    return np.array(fit.x, dtype=float)


def _build_nirspec_contamination_model(deepframe, xtrace, ypos, width,
                                       separation_guess=None, min_separation=3.0,
                                       max_separation=5.5, oversample=10,
                                       fit_background=False, primary_tolerance=1.5,
                                       detector='nrs1'):
    """Build a companion-only contamination image from the notebook-style deepframe fit."""

    dimy, dimx = np.shape(deepframe)
    if separation_guess in [None, 'None', 'null', '']:
        separation_guess = 3.5

    xtrace = np.asarray(np.rint(xtrace), dtype=int)
    ypos = np.asarray(ypos, dtype=float)
    contamination_image = np.zeros_like(deepframe, dtype=float)
    model_params = {
        'amp_target': np.full(dimx, np.nan),
        'mu_target': np.full(dimx, np.nan),
        'sigma': np.full(dimx, np.nan),
        'amp_companion': np.full(dimx, np.nan),
        'mu_companion': np.full(dimx, np.nan),
        'background': np.full(dimx, np.nan),
        'model_rms': np.full(dimx, np.nan),
        'target_flux_in_aperture': np.full(dimx, np.nan),
        'companion_flux_in_aperture': np.full(dimx, np.nan),
        'companion_fraction': np.full(dimx, np.nan),
    }

    edge_low, edge_up, _, _ = _get_extraction_edges(ypos, dimy, width)
    y_native = np.arange(dimy, dtype=float)
    x_os = np.arange((dimy - 1) * oversample + 1, dtype=float)
    notebook_setup = _get_notebook_nirspec_decontam_setup(
        detector, separation_guess=separation_guess, oversample=oversample,
        min_separation=min_separation
    )

    prev_params = None
    for xx, x in enumerate(xtrace):
        if x < notebook_setup['fit_start_col'] or x < 0 or x >= dimx:
            continue

        profile_native = np.asarray(deepframe[:, x], dtype=float)
        if not np.any(np.isfinite(profile_native)):
            continue

        if fit_background is not True:
            fit_result = _fit_nirspec_contamination_profile_notebook(
                profile_native, ypos[xx], detector, separation_guess,
                oversample=oversample, min_separation=min_separation
            )
            if fit_result is None:
                continue
            params = fit_result['coeff']
            profile_os = fit_result['profile_os']
            model_os = fit_result['model_os']
            model_rms = fit_result['fit_rms']
        else:
            profile_os = np.interp(x_os / oversample, y_native,
                                   np.nan_to_num(profile_native, nan=0.0))
            params = _fit_nirspec_contamination_profile(
                x_os, profile_os, ypos[xx], separation_guess,
                min_separation=min_separation, max_separation=max_separation,
                oversample=oversample, prev_params=prev_params,
                fit_background=fit_background, primary_tolerance=primary_tolerance
            )
            if params is None:
                continue
            prev_params = np.array(params, dtype=float)
            model_rms = np.sqrt(np.nanmean((
                profile_os - (
                    params[0] * _gaussian_profile(x_os, params[1], params[2]) +
                    params[3] * _gaussian_profile(x_os, params[1] + params[4], params[2]) +
                    params[5]
                )
            ) ** 2))

        mu_target = params[1] / oversample
        sigma = params[2] / oversample
        mu_comp = params[4] / oversample if fit_background is not True else (params[1] + params[4]) / oversample
        background = params[5] if fit_background is True else 0.0
        target_native = params[0] * _gaussian_profile(y_native, mu_target, sigma)
        companion_native = params[3] * _gaussian_profile(y_native, mu_comp, sigma)
        model_native = target_native + companion_native + background

        contamination_image[:, x] = companion_native
        model_params['amp_target'][x] = params[0]
        model_params['mu_target'][x] = mu_target
        model_params['sigma'][x] = sigma
        model_params['amp_companion'][x] = params[3]
        model_params['mu_companion'][x] = mu_comp
        model_params['background'][x] = background
        model_params['model_rms'][x] = model_rms

        rows, weights = _get_aperture_pixels(edge_low[xx], edge_up[xx], dimy)
        if len(rows) > 0:
            ygrid = rows + 0.5
            target_box = np.sum(params[0] * _gaussian_profile(ygrid, mu_target, sigma) * weights)
            comp_box = np.sum(params[3] * _gaussian_profile(ygrid, mu_comp, sigma) * weights)
            model_params['target_flux_in_aperture'][x] = target_box
            model_params['companion_flux_in_aperture'][x] = comp_box
            if target_box > 0:
                model_params['companion_fraction'][x] = comp_box / target_box

    return contamination_image, model_params


def _save_decontaminated_segment_copy(datafile, contamination_image, output_dir):
    """Save one decontaminated copy of a Stage 2 NIRSpec segment."""

    if isinstance(datafile, str) is not True:
        return None

    with fits.open(datafile) as hdul:
        sci_idx = None
        for idx, hdu in enumerate(hdul):
            data = getattr(hdu, 'data', None)
            if data is None:
                continue
            if np.ndim(data) >= 2 and np.shape(data)[-2:] == np.shape(contamination_image):
                sci_idx = idx
                break
        if sci_idx is None:
            return None

        hdu_data = np.array(hdul[sci_idx].data, dtype=float)
        if np.ndim(hdu_data) == 3:
            hdul[sci_idx].data = hdu_data - contamination_image[None, :, :]
        elif np.ndim(hdu_data) == 2:
            hdul[sci_idx].data = hdu_data - contamination_image
        else:
            return None

        basename = os.path.basename(datafile).replace('.fits', '_decontaminated.fits')
        outfile = os.path.join(output_dir, basename)
        hdul.writeto(outfile, overwrite=True)

    return outfile


def decontaminate_nirspec_then_box_extract(datafiles, centroids, extract_width, deepframe=None,
                                           separation_guess=None, min_separation=3.0,
                                           max_separation=5.5, oversample=10,
                                           fit_background=False, primary_tolerance=1.5,
                                           do_plot=False, show_plot=False, save_results=True,
                                           output_dir='./', save_decontaminated_files=True):
    """Model and subtract contamination, then perform a standard NIRSpec box extraction."""

    datafiles = _prefer_nirspec_pcareconstruct_files(np.atleast_1d(datafiles))
    det = utils.get_nrs_detector_name(datafiles[0])
    grating = utils.get_nrs_grating(datafiles[0])
    if separation_guess in [None, 'None', 'null', '']:
        separation_guess = _default_nirspec_decontam_separation(det, grating=grating)

    for i, file in enumerate(datafiles):
        if isinstance(file, str):
            data = fits.getdata(file)
            err = fits.getdata(file, 2)
        else:
            with utils.open_filetype(file) as datamodel:
                data = datamodel.data
                err = datamodel.err
        if i == 0:
            cube = data
            ecube = err
        else:
            cube = np.concatenate([cube, data])
            ecube = np.concatenate([ecube, err])

    if deepframe is None:
        deepframe = np.nanmedian(cube, axis=0)

    x1, y1 = centroids['xpos'].values, centroids['ypos'].values
    xtrace = np.asarray(np.rint(x1), dtype=int)
    finite_x = xtrace[np.isfinite(xtrace)]
    if len(finite_x) == 0:
        raise ValueError('No finite centroid x-positions available for decontamination.')
    extract_start = int(np.nanmin(finite_x))

    fancyprint('Building notebook-style contamination model.')
    contamination_image, model_params = _build_nirspec_contamination_model(
        deepframe, x1, y1, extract_width, separation_guess=separation_guess,
        min_separation=min_separation, max_separation=max_separation, oversample=oversample,
        fit_background=fit_background, primary_tolerance=primary_tolerance,
        detector=det
    )

    deepframe_decont = deepframe - contamination_image
    cube_decont = cube - contamination_image[None, :, :]

    saved_decontaminated_files = []
    if save_results is True and save_decontaminated_files is True:
        for file in datafiles:
            outfile = _save_decontaminated_segment_copy(file, contamination_image, output_dir)
            if outfile is not None:
                saved_decontaminated_files.append(outfile)

    fancyprint('Performing box extraction on decontaminated data.')
    if len(saved_decontaminated_files) == len(datafiles):
        wave, flux, ferr, _ = box_extract_nirspec(
            saved_decontaminated_files, centroids, extract_width,
            do_plot=False, show_plot=False, save_results=False, output_dir=output_dir
        )
    else:
        flux, ferr = do_box_extraction(cube_decont, ecube, y1, width=extract_width,
                                       extract_start=extract_start)
        wave = get_wave_nirspec(datafiles[0], centroids, cube.shape[0], cube.shape[2])

    if do_plot is True:
        outfile = None
        if save_results is True:
            outfile = output_dir + 'decontam_diagnostics_{}.png'.format(det)
        plotting.make_nirspec_decontamination_plot(
            deepframe, contamination_image, deepframe_decont, x1, y1, extract_width,
            model_params, extract_start=extract_start, outfile=outfile, show_plot=show_plot
        )

    return {'wave': wave, 'flux': flux, 'ferr': ferr, 'contamination_image': contamination_image,
            'decontaminated_deepframe': deepframe_decont, 'profile': model_params,
            'decontaminated_files': saved_decontaminated_files}


def box_extract_soss(datafiles, centroids, soss_width, soss_width_o2=None, do_plot=False,
                     show_plot=False, save_results=True, output_dir='./',
                     mask_saturated_pixels=True, mask_do_not_use_pixels=True):
    """Perform a simple box aperture extraction on SOSS orders 1 and 2.

    Parameters
    ----------
    datafiles : array-like[str], array-like[jwst.RampModel]
        Input datamodels or paths to datamodels for each segment.
    centroids : dict
        Dictionary of centroid positions for all SOSS orders.
    soss_width : int, tuple(float, float), str
        Width of extraction box for order 1. Or 'optimize'. A two-element tuple is interpreted as
        an asymmetric `(lower_width, upper_width)` aperture.
    soss_width_o2 : int, tuple(float, float), str, None
        Width of extraction box for order 2. Or 'optimize'. If None, will use the same aperture as
        order 1. A two-element tuple is interpreted as an asymmetric `(lower_width, upper_width)`
        aperture.
    do_plot : bool
        If True, do the step diagnostic plot.
    show_plot : bool
        If True, show the step diagnostic plot instead of/in addition to saving it to file.
    output_dir : str
        Directory to which to output results.
    save_results : bool
        If True, save results to file.

    Returns
    -------
    wave_o1 : array_like[float]
        2D wavelength solution for order 1.
    flux_o1 : array_like[float]
        2D extracted flux for order 1.
    ferr_o1: array_like[float]
        2D flux errors for order 1.
    wave_o2 : array_like[float]
        2D wavelength solution for order 2.
    flux_o2 : array_like[float]
        2D extracted flux for order 2.
    ferr_o2 : array_like[float]
        2D flux errors for order 2.
    soss_width : int
        Optimized aperture width for order 1.
    """

    datafiles = np.atleast_1d(datafiles)
    cube, ecube = _load_box_extraction_cubes(datafiles,
                                             mask_saturated_pixels=mask_saturated_pixels,
                                             mask_do_not_use_pixels=mask_do_not_use_pixels)

    # Get centroid positions.
    x1 = centroids['xpos'].values
    y1, y2 = centroids['ypos o1'].values, centroids['ypos o2'].values
    ii = np.where(np.isfinite(y2))
    x2, y2 = x1[ii], y2[ii]

    fancyprint('Performing simple aperture extraction.')
    for order, width, y in zip([1, 2], [soss_width, soss_width_o2], [y1, y2]):
        # Optimize aperture width.
        if width == 'optimize':
            fancyprint('Optimizing extraction width for order {}...'.format(order))
            # Extract with different widths and find the one that minimizes the white light scatter.
            scatter = []
            for w in tqdm(range(10, 61)):
                flux = do_box_extraction(cube, ecube, y, width=w, extract_end=len(y),
                                         progress=False)[0]
                wlc = np.nansum(flux, axis=1)
                s = np.median(np.abs(0.5*(wlc[0:-2] + wlc[2:]) - wlc[1:-1]))
                scatter.append(s)
            scatter = np.array(scatter)
            # Find the width that minimizes the scatter.
            ii = np.argmin(scatter)
            width = np.linspace(10, 60, 51)[ii]
            # For order 1, save this value for possible later use.
            if order == 1:
                soss_width = width
            fancyprint('Using width of {0} pxiels for order {1}.'.format(int(width), order))

            # Do diagnostic plot if requested.
            if do_plot is True:
                if save_results is True:
                    outfile = output_dir + 'aperture_optimization_order{}.png'.format(order)
                else:
                    outfile = None
                plotting.make_soss_width_plot(np.linspace(10, 60, 51), scatter, ii, outfile=outfile,
                                              show_plot=show_plot)

        # Do the extraction.
        fancyprint('Extracting Order {}'.format(order))
        if order == 2:
            # If None is passed for the order 2 extraction width, just use the same as order 1.
            if width is None:
                width = soss_width
            flux_o2, ferr_o2 = do_box_extraction(cube, ecube, y, width=width, extract_end=len(y))
        else:
            flux_o1, ferr_o1 = do_box_extraction(cube, ecube, y, width=width)

    # Get default wavelength solution.
    wave_o1, wave_o2 = get_wave_soss(datafiles[0])

    return wave_o1, flux_o1, ferr_o1, wave_o2, flux_o2, ferr_o2, soss_width


def double_gaussian_extract_soss(datafiles, centroids, soss_width, soss_width_o2=None,
                                 separation_guess=4.0, separation_guess_o2=None,
                                 fit_background=True, main_component=1):
    """Extract both members of an overlapping SOSS binary with a two-Gaussian profile fit.

    Parameters
    ----------
    datafiles : array-like[str], array-like[jwst.RampModel]
        Input datamodels or paths to datamodels for each segment.
    centroids : dict
        Dictionary of centroid positions for all SOSS orders. The supplied centroids are treated as
        the midpoint between the two stellar traces for each order.
    soss_width : int, tuple(float, float), dict
        Width of extraction box for order 1.
    soss_width_o2 : int, tuple(float, float), dict, None
        Width of extraction box for order 2. If None, order 1 is reused.
    separation_guess : float
        Initial guess for the separation between the lower and upper Gaussian components in order
        1, in pixels.
    separation_guess_o2 : float, None
        Initial guess for the separation between the lower and upper Gaussian components in order
        2, in pixels. If None, the order 1 value is reused.
    fit_background : bool
        If True, include a constant background term in the spatial profile fit.
    main_component : int
        Which fitted component to treat as the primary extracted target. `1` selects the lower
        trace and `2` selects the upper trace.

    Returns
    -------
    result : dict
        Dictionary containing the primary and companion spectra for SOSS orders 1 and 2.
    """

    if main_component not in [1, 2]:
        raise ValueError('main_component must be either 1 (lower trace) or 2 (upper trace).')

    datafiles = np.atleast_1d(datafiles)
    for i, file in enumerate(datafiles):
        if isinstance(file, str):
            data = fits.getdata(file)
            err = fits.getdata(file, 2)
        else:
            with utils.open_filetype(file) as datamodel:
                data = datamodel.data
                err = datamodel.err
        if i == 0:
            cube = data
            ecube = err
        else:
            cube = np.concatenate([cube, data])
            ecube = np.concatenate([ecube, err])

    x1 = centroids['xpos'].values
    y1, y2 = centroids['ypos o1'].values, centroids['ypos o2'].values
    ii = np.where(np.isfinite(y2))
    x2, y2 = x1[ii], y2[ii]

    if soss_width_o2 is None:
        soss_width_o2 = soss_width
    if separation_guess_o2 is None:
        separation_guess_o2 = separation_guess

    fancyprint('Performing double-Gaussian SOSS extraction.')
    flux1_o1, ferr1_o1, flux2_o1, ferr2_o1, prof_o1 = do_two_gaussian_extraction(
        cube, ecube, y1, width=soss_width, separation_guess=separation_guess,
        fit_background=fit_background
    )
    flux1_o2, ferr1_o2, flux2_o2, ferr2_o2, prof_o2 = do_two_gaussian_extraction(
        cube, ecube, y2, width=soss_width_o2, extract_end=len(x2),
        separation_guess=separation_guess_o2, fit_background=fit_background
    )

    if main_component == 1:
        flux_o1, ferr_o1 = flux1_o1, ferr1_o1
        flux_o2, ferr_o2 = flux1_o2, ferr1_o2
        flux_o1_comp, ferr_o1_comp = flux2_o1, ferr2_o1
        flux_o2_comp, ferr_o2_comp = flux2_o2, ferr2_o2
    else:
        flux_o1, ferr_o1 = flux2_o1, ferr2_o1
        flux_o2, ferr_o2 = flux2_o2, ferr2_o2
        flux_o1_comp, ferr_o1_comp = flux1_o1, ferr1_o1
        flux_o2_comp, ferr_o2_comp = flux1_o2, ferr1_o2

    wave_o1, wave_o2 = get_wave_soss(datafiles[0])

    return {'wave_o1': wave_o1, 'flux_o1': flux_o1, 'ferr_o1': ferr_o1,
            'wave_o2': wave_o2, 'flux_o2': flux_o2, 'ferr_o2': ferr_o2,
            'flux_o1_companion': flux_o1_comp, 'ferr_o1_companion': ferr_o1_comp,
            'flux_o2_companion': flux_o2_comp, 'ferr_o2_companion': ferr_o2_comp,
            'profile_o1': prof_o1, 'profile_o2': prof_o2}


def _format_extract_width(width):
    """Convert extraction width metadata into a FITS-header-safe value."""

    if isinstance(width, str) or np.isscalar(width):
        return width
    if isinstance(width, dict):
        if 'lower' in width and 'upper' in width:
            return 'lower={}, upper={}'.format(width['lower'], width['upper'])
        return str(width)

    try:
        lower_width, upper_width = width
    except (TypeError, ValueError):
        return str(width)

    return 'lower={}, upper={}'.format(lower_width, upper_width)


def _parse_extraction_width(width, lower_width=None, upper_width=None):
    """Normalize symmetric and asymmetric aperture definitions into half-widths."""

    if isinstance(width, str):
        raise ValueError('String widths are not supported by the low-level extraction helpers.')

    if lower_width is not None or upper_width is not None:
        if lower_width is None or upper_width is None:
            raise ValueError('Both lower_width and upper_width must be provided.')
        lower_half = float(lower_width)
        upper_half = float(upper_width)
    elif isinstance(width, dict):
        if 'lower' not in width or 'upper' not in width:
            raise ValueError('Width dictionaries must contain "lower" and "upper" keys.')
        lower_half = float(width['lower'])
        upper_half = float(width['upper'])
    elif np.isscalar(width):
        lower_half = float(width) / 2
        upper_half = float(width) / 2
    else:
        try:
            lower_half, upper_half = width
        except (TypeError, ValueError):
            raise ValueError('width must be a scalar full width or a two-element '
                             '(lower_width, upper_width) pair.')
        lower_half = float(lower_half)
        upper_half = float(upper_half)

    if lower_half <= 0 or upper_half <= 0:
        raise ValueError('Extraction widths must be strictly positive.')

    return lower_half, upper_half


def _get_extraction_edges(ypos, dimy, width, lower_width=None, upper_width=None):
    """Determine the lower and upper edges of an extraction aperture."""

    lower_half, upper_half = _parse_extraction_width(width, lower_width=lower_width,
                                                     upper_width=upper_width)
    ypos = np.asarray(ypos, dtype=float)
    edge_up = np.min([ypos + upper_half, np.ones_like(ypos, dtype=float) * dimy], axis=0)
    edge_low = np.max([ypos - lower_half, np.zeros_like(ypos, dtype=float)], axis=0)

    return edge_low, edge_up, lower_half, upper_half


def _get_aperture_pixels(edge_low, edge_up, dimy):
    """Return detector rows and fractional pixel overlaps for one aperture."""

    row_start = max(int(np.floor(edge_low)), 0)
    row_end = min(int(np.ceil(edge_up)), dimy)
    rows = np.arange(row_start, row_end)
    if len(rows) == 0:
        return rows.astype(int), np.array([], dtype=float)

    weights = np.minimum(rows + 1, edge_up) - np.maximum(rows, edge_low)
    weights = np.clip(weights, 0, 1)
    ii = np.where(weights > 0)[0]

    return rows[ii].astype(int), weights[ii]


def _gaussian_profile(y, mu, sigma):
    """Evaluate a unit-amplitude Gaussian profile."""

    return np.exp(-0.5 * ((y - mu) / sigma) ** 2)


def _initial_peak_guess(y, profile, default):
    """Estimate a Gaussian center from the strongest local signal."""

    if len(y) == 0 or not np.any(np.isfinite(profile)):
        return default

    return y[np.nanargmax(profile)]


def _fit_two_gaussian_profile(y, profile, profile_err, midpoint, edge_low, edge_up, lower_half,
                              upper_half, separation_guess=None, prev_params=None,
                              fit_background=True):
    """Fit a two-Gaussian profile to a single spatial cut."""

    min_points = 5 if fit_background is True else 4
    mask = (np.isfinite(y) & np.isfinite(profile) & np.isfinite(profile_err) &
            (profile_err > 0))
    if np.sum(mask) < min_points:
        return None

    yy = y[mask]
    pp = profile[mask]
    ee = profile_err[mask]
    divider = np.clip(midpoint, edge_low + 0.25, edge_up - 0.25)
    if divider <= edge_low or divider >= edge_up:
        return None

    sigma_max = max(lower_half + upper_half, 1.5)
    lower_bounds = [0, edge_low, 0.3, 0, divider, 0.3]
    upper_bounds = [np.inf, divider, sigma_max, np.inf, edge_up, sigma_max]
    if fit_background is True:
        lower_bounds.append(-np.inf)
        upper_bounds.append(np.inf)

    if prev_params is None:
        lower_side = yy <= divider
        upper_side = yy >= divider
        if len(pp) >= 4:
            background0 = float(np.nanmedian(np.concatenate([pp[:2], pp[-2:]])))
        else:
            background0 = float(np.nanmedian(pp))

        lower_default = divider - max(lower_half / 2, 0.5)
        upper_default = divider + max(upper_half / 2, 0.5)
        if separation_guess is not None:
            lower_default = divider - separation_guess / 2
            upper_default = divider + separation_guess / 2

        mu1_0 = _initial_peak_guess(yy[lower_side], pp[lower_side], lower_default)
        mu2_0 = _initial_peak_guess(yy[upper_side], pp[upper_side], upper_default)
        mu1_0 = np.clip(mu1_0, edge_low + 0.1, divider - 0.1)
        mu2_0 = np.clip(mu2_0, divider + 0.1, edge_up - 0.1)

        amp1_0 = max(float(np.nanmax(pp[lower_side])) - background0, 0) if np.any(lower_side) else 0
        amp2_0 = max(float(np.nanmax(pp[upper_side])) - background0, 0) if np.any(upper_side) else 0
        sigma1_0 = np.clip(max(lower_half / 3, 0.8), 0.3, sigma_max)
        sigma2_0 = np.clip(max(upper_half / 3, 0.8), 0.3, sigma_max)
        p0 = [amp1_0, mu1_0, sigma1_0, amp2_0, mu2_0, sigma2_0]
        if fit_background is True:
            p0.append(background0)
    else:
        p0 = np.array(prev_params, dtype=float)

    p0 = np.clip(np.asarray(p0, dtype=float), lower_bounds, upper_bounds)

    def residuals(params):
        model = (params[0] * _gaussian_profile(yy, params[1], params[2]) +
                 params[3] * _gaussian_profile(yy, params[4], params[5]))
        if fit_background is True:
            model += params[6]
        return (pp - model) / ee

    try:
        fit = least_squares(residuals, p0, bounds=(lower_bounds, upper_bounds))
    except ValueError:
        return None

    if fit.success is not True:
        return None

    params = np.array(fit.x, dtype=float)
    if params[1] > params[4]:
        params = np.array([params[3], params[4], params[5],
                           params[0], params[1], params[2],
                           params[6] if fit_background is True else 0], dtype=float)
        if fit_background is not True:
            params = params[:6]

    return params


def do_box_extraction(cube, err, ypos, width, extract_start=0, extract_end=None, progress=True,
                      lower_width=None, upper_width=None):
    """Do intrapixel aperture extraction.

    Parameters
    ----------
    cube : array-like(float)
        Data cube.
    err : array-like(float)
        Error cube.
    ypos : array-like(float)
        Detector Y-positions to extract.
    width : int, tuple(float, float)
        Full-width of the extraction aperture to use. A two-element tuple is interpreted as an
        asymmetric `(lower_width, upper_width)` aperture.
    extract_start : int
        Detector X-position at which to start extraction.
    extract_end : int, None
        Detector X-position at which to end extraction.
    progress : bool
        if True, show extraction progress bar.
    lower_width : float, None
        Distance from the centroid to the lower aperture edge. If provided together with
        `upper_width`, overrides the symmetric interpretation of `width`.
    upper_width : float, None
        Distance from the centroid to the upper aperture edge. If provided together with
        `lower_width`, overrides the symmetric interpretation of `width`.

    Returns
    -------
    f : np.array(float)
        Extracted flux values.
    ferr : np.array(float)
         Extracted error values.
    """

    # Ensure data and errors are the same shape.
    assert np.shape(cube) == np.shape(err)
    nint, dimy, dimx = np.shape(cube)

    # If extraction end is not specified, extract the whole frame.
    if extract_end is None:
        extract_end = dimx

    # Initialize output arrays.
    f, ferr = np.zeros((nint, dimx)), np.zeros((nint, dimx))

    # Determine the upper and lower edges of the extraction region. Cut at detector edges if
    # necessary.
    edge_low, edge_up, _, _ = _get_extraction_edges(ypos, dimy, width,
                                                    lower_width=lower_width,
                                                    upper_width=upper_width)

    # Loop over all columns and sum flux within the extraction aperture.
    for x in tqdm(range(extract_start, extract_end), disable=not progress):
        xx = x - extract_start
        rows, weights = _get_aperture_pixels(edge_low[xx], edge_up[xx], dimy)
        if len(rows) == 0:
            continue

        weighted_cube = cube[:, rows, x] * weights[None, :]
        weighted_err = err[:, rows, x] * weights[None, :]
        f[:, x] = np.nansum(weighted_cube, axis=1)
        ferr[:, x] = np.sqrt(np.nansum(weighted_err**2, axis=1))

    return f, ferr


def do_two_gaussian_extraction(cube, err, ypos, width, extract_start=0, extract_end=None,
                               progress=True, lower_width=None, upper_width=None,
                               separation_guess=None, fit_background=True):
    """Fit and extract two overlapping Gaussian traces inside one aperture.

    Parameters
    ----------
    cube : array-like(float)
        Data cube.
    err : array-like(float)
        Error cube.
    ypos : array-like(float)
        Detector Y-positions of the midpoint between the two traces.
    width : int, tuple(float, float)
        Full-width of the extraction aperture to use. A two-element tuple is interpreted as an
        asymmetric `(lower_width, upper_width)` aperture.
    extract_start : int
        Detector X-position at which to start extraction.
    extract_end : int, None
        Detector X-position at which to end extraction.
    progress : bool
        If True, show extraction progress bar.
    lower_width : float, None
        Distance from the midpoint trace to the lower aperture edge. If provided together with
        `upper_width`, overrides the symmetric interpretation of `width`.
    upper_width : float, None
        Distance from the midpoint trace to the upper aperture edge. If provided together with
        `lower_width`, overrides the symmetric interpretation of `width`.
    separation_guess : float, None
        Initial separation between the two Gaussian centroids, in pixels.
    fit_background : bool
        If True, include a constant background term in the profile fit.

    Returns
    -------
    f1 : np.array(float)
        Extracted flux for the lower trace.
    ferr1 : np.array(float)
        Extracted error for the lower trace.
    f2 : np.array(float)
        Extracted flux for the upper trace.
    ferr2 : np.array(float)
        Extracted error for the upper trace.
    profile_params : dict
        Column-by-column Gaussian profile parameters.
    """

    assert np.shape(cube) == np.shape(err)
    nint, dimy, dimx = np.shape(cube)

    if extract_end is None:
        extract_end = dimx

    f1 = np.zeros((nint, dimx))
    ferr1 = np.zeros((nint, dimx))
    f2 = np.zeros((nint, dimx))
    ferr2 = np.zeros((nint, dimx))
    profile_params = {'amp1': np.full(dimx, np.nan), 'mu1': np.full(dimx, np.nan),
                      'sigma1': np.full(dimx, np.nan), 'amp2': np.full(dimx, np.nan),
                      'mu2': np.full(dimx, np.nan), 'sigma2': np.full(dimx, np.nan),
                      'reduced_chi2': np.full(dimx, np.nan)}
    if fit_background is True:
        profile_params['background'] = np.full(dimx, np.nan)

    edge_low, edge_up, lower_half, upper_half = _get_extraction_edges(ypos, dimy, width,
                                                                      lower_width=lower_width,
                                                                      upper_width=upper_width)
    prev_params = None
    for x in tqdm(range(extract_start, extract_end), disable=not progress):
        xx = x - extract_start
        rows, weights = _get_aperture_pixels(edge_low[xx], edge_up[xx], dimy)
        if len(rows) < 4:
            continue

        ygrid = rows + 0.5
        col_cube = cube[:, rows, x] * weights[None, :]
        col_err = err[:, rows, x] * weights[None, :]
        profile = np.nanmedian(col_cube, axis=0)
        profile_err = np.sqrt(np.nanmedian(col_err**2, axis=0))

        params = _fit_two_gaussian_profile(ygrid, profile, profile_err, ypos[xx], edge_low[xx],
                                           edge_up[xx], lower_half, upper_half,
                                           separation_guess=separation_guess,
                                           prev_params=prev_params,
                                           fit_background=fit_background)
        if params is None:
            params = prev_params
        if params is None:
            continue
        prev_params = np.array(params, dtype=float)

        g1 = _gaussian_profile(ygrid, params[1], params[2])
        g2 = _gaussian_profile(ygrid, params[4], params[5])
        design = [g1, g2]
        if fit_background is True:
            design.append(np.ones_like(g1))
        design = np.column_stack(design)

        background = params[6] if fit_background is True else 0.0
        model_profile = params[0] * g1 + params[3] * g2 + background
        good = np.isfinite(profile) & np.isfinite(profile_err) & (profile_err > 0)
        dof = np.sum(good) - design.shape[1]

        profile_params['amp1'][x] = params[0]
        profile_params['mu1'][x] = params[1]
        profile_params['sigma1'][x] = params[2]
        profile_params['amp2'][x] = params[3]
        profile_params['mu2'][x] = params[4]
        profile_params['sigma2'][x] = params[5]
        if dof > 0:
            chi2 = np.nansum(((profile[good] - model_profile[good]) / profile_err[good]) ** 2)
            profile_params['reduced_chi2'][x] = chi2 / dof
        if fit_background is True:
            profile_params['background'][x] = background

        component_scale1 = np.sum(g1 * weights)
        component_scale2 = np.sum(g2 * weights)
        for i in range(nint):
            data = col_cube[i]
            sigma = col_err[i]
            mask = np.isfinite(data) & np.isfinite(sigma) & (sigma > 0)
            if np.sum(mask) < design.shape[1]:
                continue

            aw = design[mask] / sigma[mask, None]
            bw = data[mask] / sigma[mask]
            coeffs, _, _, _ = np.linalg.lstsq(aw, bw, rcond=None)
            covariance = np.linalg.pinv(aw.T @ aw)

            f1[i, x] = coeffs[0] * component_scale1
            f2[i, x] = coeffs[1] * component_scale2
            ferr1[i, x] = np.sqrt(np.clip(covariance[0, 0], 0, None)) * component_scale1
            ferr2[i, x] = np.sqrt(np.clip(covariance[1, 1], 0, None)) * component_scale2

    return f1, ferr1, f2, ferr2, profile_params


def do_ccf(wave, flux, mod_flux, oversample=5):
    """Perform a cross-correlation analysis between an extracted and model stellar spectrum to
    determine the appropriate wavelength shift between the two.

    Parameters
    ----------
    wave : array-like[float]
        Wavelength axis.
    flux : array-like[float]
        Extracted spectrum.
    mod_flux : array-like[float]
        Model spectrum.
    oversample : int
        Degree of oversampling for the cross correlation.

    Returns
    -------
    shift_wave : float
        Wavelength shift between the model and extracted spectrum in microns.
    """

    def highpass_filter(signal, order=3, freq=0.05):
        """High pass filter."""
        b, a = butter(order, freq, btype='high')
        signal_filt = filtfilt(b, a, signal)
        return signal_filt

    # Ensure wavelengths are in ascending order
    ii = np.argsort(wave)
    thiswave = wave[ii]
    thisflux = flux[ii]
    thismod = mod_flux[ii]

    # Interpolte both model and data onto a finer wavelength grid.
    if oversample != 1:
        new_wave = []
        for i in range(len(thiswave)):
            new_wave.append(thiswave[i])
            if i < len(thiswave) - 1:
                step = thiswave[i + 1] - thiswave[i]
                step /= oversample
                for s in range(1, oversample):
                    new_wave.append(thiswave[i] + s * step)
        thisflux = np.interp(new_wave, thiswave, thisflux)
        thismod = np.interp(new_wave, thiswave, thismod)
    else:
        new_wave = thiswave

    # Remove any nan pixels.
    ii = np.where(np.isnan(thisflux))
    thisflux = np.delete(thisflux, ii)
    thismod = np.delete(thismod, ii)

    # Cross-correlate the model and observed stellar spectrum.
    ccf = correlate(highpass_filter(thisflux), highpass_filter(thismod))
    # Determine how many wavelength steps corresponds to the CCF peak.
    ll = len(thisflux)
    steps = np.linspace(-ll+1, ll-1, 2*ll-1)
    shift_steps = steps[np.argmax(ccf)]

    # And get the wavelength shift.
    shift_wave = -1*shift_steps*np.median(np.diff(new_wave))

    return shift_wave


def do_optimal_extraction(cube, deepframe, ymin=0, ymax=None, xmin=0, xmax=None, max_iter=25,
                          var_thresh=25):
    """Optimally extract stellar spectra following the Horne 1986 algorithm.

    Parameters
    ----------
    cube : ndarray(float)
        Stack datacube for the observation.
    deepframe : ndarray(float)
        Median stack of the observation.
    ymin : int, ndarray(int)
        Minimum row number to extract.
    ymax : int, ndarray(int), None
        Maximum row number to extract.
    xmin : int, ndarray(int)
        Minimum column number to extract.
    xmax : int, None
        Maximum column number to extract.
    max_iter : int
        Maximum number of outlier rejection iterations to do.
    var_thresh : int
        Variance threshold for a pixel to be considered an outlier.

    Returns
    -------
    f_opt : ndarray(float)
        Optimally extracted flux.
    var_opt : ndarray(float)
        Variance in optimally extracted flux.
    """

    nint, dimy, dimx = np.shape(cube)
    if ymax is None:
        ymax = dimy
    if xmax is None:
        xmax = dimx

    if isinstance(ymin, int):
        ymin = np.ones(xmax - xmin).astype(int) * ymin
    if isinstance(ymax, int):
        ymax = np.ones(xmax - xmin).astype(int) * ymax

    # Get initial flux estimate - Step 4.
    flux = np.zeros((nint, dimx))
    for x in range(xmin, xmax):
        xx = x - xmin
        flux[:, x] = np.nansum(cube[:, ymin[xx]:ymax[xx], x], axis=1)

    # Get initial variance estimate - Step 4.
    var_0 = np.nanstd(cube, axis=0)**2
    var = np.ones_like(cube)
    for i in range(cube.shape[0]):
        var[i] = var_0

    # Get normalized spatial profile - Step 5.
    prof = get_spatial_prof_opt(deepframe, ymin=ymin, ymax=ymax, xmin=xmin, xmax=xmax)

    # Revise variance estimate - Step 6.
    # Assuming 0 for background flux and 1 for gain.
    for x in range(xmin, xmax):
        xx = x - xmin
        var[:, ymin[xx]:ymax[xx], x] += (np.abs(flux[:, None, x] * prof[None, ymin[xx]:ymax[xx], x] + 0)) / 1

    # Optimal extraction - Step 8.
    f_opt, var_opt = extract_optimal(prof, cube, var, ymin=ymin, ymax=ymax, xmin=xmin, xmax=xmax)

    # Loop steps 6 - 8, iteratively clipped outliers.
    fancyprint('Doing iterative outlier clipping.')
    num_clipped = []
    for n_iter in tqdm(range(max_iter)):
        # Break if maximum number of iterations.
        if n_iter > max_iter:
            fancyprint('Maximum number of iterations ({}) exceeded.'.format(max_iter))
            break
        # Find and clip outliers.
        cube_filt, lls = median_filter(cube, (11, 1, 1)), []
        for i in range(cube.shape[0]):
            c, f = cube[i, :, xmin:xmax], f_opt[i, None, xmin:xmax]
            v, p = var[i, :, xmin:xmax], prof[:, xmin:xmax]
            ii = np.where(np.abs(c - f * p)**2 / v > var_thresh)
            cube[i, :, xmin:xmax][ii] = cube_filt[i, :, xmin:xmax][ii]
            lls.append(len(ii[0]))
        npix = np.sum(lls)
        num_clipped.append(npix)

        # Revise variance estimate.
        var_0 = np.nanstd(cube, axis=0)**2
        var = np.ones_like(cube)
        for i in range(cube.shape[0]):
            var[i] = var_0
        var[var == 0] = np.inf

        # Do optimal extraction.
        f_opt, var_opt = extract_optimal(prof, cube, var, ymin=ymin, ymax=ymax, xmin=xmin,
                                         xmax=xmax)

        # Break if we've hit a floor in the number of clipped pixels but haven't exceeded the
        # maximum iteration count.
        if n_iter != 0 and npix == num_clipped[n_iter - 1]:
            fancyprint('Outlier floor reached.')
            break
        # Or break if there are no more outliers left.
        if npix == 0:
            fancyprint('All outliers masked.')
            break

    return f_opt, np.sqrt(var_opt)


def extract_optimal(prof, data, var, ymin=0, ymax=None, xmin=0, xmax=None):
    """Perform the optimal extraction following formula in Step 8 of Horne 1986.

    Parameters
    ----------
    prof : ndarray(float)
        Normalized 2D spatial profile.
    data : ndarray(float)
        Stacked datacube of the observations.
    var : ndarray(float)
        Variance of the data.
    ymin : int, ndarray(int)
        Minimum row number to extract.
    ymax : int, ndarray(int), None
        Maximum row number to extract.
    xmin : int, ndarray(int)
        Minimum column number to extract.
    xmax : int, None
        Maximum column number to extract.

    Returns
    -------
    f_opt : ndarray(float)
        Optimally extracted flux.
    var_opt : ndarray(float)
        Variance in optimally extracted flux.
    """

    nint, dimy, dimx = np.shape(data)
    if ymax is None:
        ymax = dimy
    if xmax is None:
        xmax = dimx
    ymax = np.atleast_1d(ymax)
    ymin = np.atleast_1d(ymin)

    f_opt, var_opt = np.zeros((nint, dimx)), np.zeros((nint, dimx))

    # If the y-bounds are constant.
    if len(ymin) == 1:
        assert len(ymax) == 1
        ymin, ymax = ymin[0], ymax[0]
        p = prof[None, ymin:ymax, xmin:xmax]
        d = data[:, ymin:ymax, xmin:xmax]
        v = var[:, ymin:ymax, xmin:xmax]
        f_opt[:, xmin:xmax] = np.nansum(p * d / v, axis=1) / np.nansum(p ** 2 / v, axis=1)
        var_opt[:, xmin:xmax] = np.nansum(p, axis=1) / np.nansum(p ** 2 / v, axis=1)
    # For non-constant y-bounds.
    else:
        xdim_trim = prof[:, xmin:xmax].shape[1]
        assert len(ymax) == xdim_trim
        assert len(ymin) == xdim_trim
        for x in range(xmin, xmax):
            xx = x - xmin
            p = prof[None, ymin[xx]:ymax[xx], x]
            d = data[:, ymin[xx]:ymax[xx], x]
            v = var[:, ymin[xx]:ymax[xx], x]
            f_opt[:, x] = np.nansum(p * d / v, axis=1) / np.nansum(p ** 2 / v, axis=1)
            var_opt[:, x] = np.nansum(p, axis=1) / np.nansum(p ** 2 / v, axis=1)

    return f_opt, var_opt


def flux_calibrate(spectrum_file):
    """Perform the flux calibration (to erg/s/cm^2/µm) for extracted NIRSpec or MIRI spectra.

    Parameters
    ----------
    spectrum_file : str
        Path to extracted stellar spectra.
    """

    fancyprint('Starting flux calibration.')

    # Get the extracted spectra and erorrs.
    spec = fits.open(spectrum_file)
    # Convert to erg/s/cm2/µm.
    spec[3].data = utils.convert_flux_units(spec[1].data, spec[3].data)
    spec[4].data = utils.convert_flux_units(spec[1].data, spec[4].data)
    spec[3].header['UNITS'] = 'erg/s/cm2/um'
    spec[4].header['UNITS'] = 'erg/s/cm2/um'

    newfile = spectrum_file[:-5] + '_FluxCalibrated.fits'
    fancyprint('Flux calibrated spectra saved to {}'.format(newfile))
    spec.writeto(newfile, overwrite=True)

    return None


def flux_calibrate_soss(spectrum_file, pwcpos, photom_path, spectrace_path, orders=[1, 2]):
    """Perform the flux calibration (to erg/s/cm^2/µm) for extracted SOSS spectra. Note that the
    spectra must have been extracted with a box width of 0 pixels, and also that the rev2 photom
    reference file produced by Kevin Volk during commissioning should be used instead of the
    default one.

    Parameters
    ----------
    spectrum_file : str
        Path to extracted stellar spectra.
    pwcpos : float
        Observation pupil wheel position.
    photom_path : str
        Path to photom reference file.
    spectrace_path : str
        Path to spectra reference file.
    orders : list(int)
        SOSS order(s) to calibrate.
    """

    fancyprint('Starting SOSS flux calibration.')
    fancyprint('Flux calibration is only valid for spectra extracted using an aperture width of '
               '40 pixels!', msg_type='WARNING')
    fancyprint('Ensure to use the rev2 photom file and not the default crds reference!',
               msg_type='WARNING')

    # Get the extracted spectra and erorrs.
    spec = fits.open(spectrum_file)
    for order in orders:
        if order == 1:
            wave = spec[1].data
            fi, ei = 3, 4
        else:
            wave = spec[5].data
            fi, ei = 7, 8

        # Calculate the ADU/s to Jy flux calibration.
        flux_scaling = wave_and_flux_calibrations(pwcpos=pwcpos, obs_x_pixel=np.arange(2048)[::-1],
                                                  photom_path=photom_path,
                                                  spectrace_path=spectrace_path, order=order)
        # Apply the flux calibration.
        spec[fi].data *= flux_scaling
        spec[ei].data *= flux_scaling
        # Convert to erg/s/cm2/µm.
        spec[fi].data = utils.convert_flux_units(wave, spec[fi].data/1e6)  # Convert Jy to MJy
        spec[fi].header['UNITS'] = 'erg/s/cm2/um'
        spec[ei].data = utils.convert_flux_units(wave, spec[ei].data/1e6)  # Convert Jy to MJy
        spec[ei].header['UNITS'] = 'erg/s/cm2/um'

    newfile = spectrum_file[:-5] + '_FluxCalibrated.fits'
    fancyprint('Flux calibrated spectra saved to {}'.format(newfile))
    spec.writeto(newfile, overwrite=True)

    return None


def format_miri_spectra(datafiles, times, extract_params, target_name, st_teff=None,
                        st_logg=None, st_met=None, output_dir='./', save_results=True):
    """Unpack the outputs of the 1D extraction and format them into
    lightcurves at the native detector resolution.

    Parameters
    ----------
    datafiles : array-like[str], array-like[MultiSpecModel], tuple
        Input extract1d data files.
    times : array-like[float]
        Time stamps corresponding to each integration.
    output_dir : str
        Directory to which to save outputs.
    save_results : bool
        If True, save outputs to file.
    extract_params : dict
        Dictonary of parameters used for the 1D extraction.
    target_name : str
        Name of the target.
    st_teff : float, None
        Stellar effective temperature.
    st_logg : float, None
        Stellar log surface gravity.
    st_met : float, None
        Stellar metallicity as [Fe/H].

    Returns
    -------
    spectra : dict
        1D stellar spectra at the native detector resolution.
    """

    fancyprint('Formatting extracted 1d spectra.')
    # Box extract outputs will just be a tuple of arrays.
    wave1d = datafiles[0][0]
    flux = datafiles[1]
    ferr = datafiles[2]

    if st_teff is not None or st_logg is not None or st_met is not None:
        fancyprint('Wavelength calibration not implemented for MIRI.', msg_type='WARNING')
        fancyprint('Using the default wavelength solution.', msg_type='WARNING')
    # Remove any NaN pixels --- important for NIRSpec NRS1.
    # ii = np.where(np.isfinite(wave1d))[0]
    # wave1d_trim = wave1d[ii]

    # Now cross-correlate with stellar model --- skip for MIRI for now.
    # if None in [st_teff, st_logg, st_met]:
    #     fancyprint('Stellar parameters not provided. Using default wavelength solution.',
    #                msg_type='WARNING')
    # else:
    #     fancyprint('Refining the wavelength calibration.')
    #     # Create a grid of stellar parameters, and download PHOENIX spectra for each grid point.
    #     thisout = output_dir + 'phoenix_models'
    #     utils.verify_path(thisout)
    #     res = utils.download_stellar_spectra(st_teff, st_logg, st_met, outdir=thisout)
    #     wave_file, flux_files = res
    #     # Interpolate model grid to correct stellar parameters.
    #     # Reverse direction of both arrays since SOSS is extracted red to blue.
    #     mod_flux = utils.interpolate_stellar_model_grid(flux_files, st_teff, st_logg, st_met)
    #     mod_wave = fits.getdata(wave_file) / 1e4
    #
    #     # Bin model down to data wavelengths.
    #     mod_flux = spectres.spectres(wave1d_trim, mod_wave, mod_flux)
    #
    #     # Cross-correlate extracted spectrum with model to refine wavelength calibration.
    #     x1d_flux = np.nansum(flux, axis=0)[ii]
    #     wave_shift = do_ccf(wave1d_trim, x1d_flux, mod_flux, oversample=1)
    #     fancyprint('Found a wavelength shift of {}um'.format(wave_shift))
    #     wave1d += wave_shift

    # Clip remaining 3-sigma outliers.
    flux_clip = utils.sigma_clip_lightcurves(flux, window=11, thresh=3)

    # Pack the lightcurves into the output format.
    # Put 1D extraction parameters in the output file header.
    filename = (output_dir + target_name[:-2] + '_' + extract_params['method'] +
                '_spectra_fullres.fits')
    header_dict, header_comments = utils.get_default_header()
    header_dict['Target'] = target_name[:-2]
    header_dict['Contents'] = 'Full resolution stellar spectra'
    header_dict['Method'] = extract_params['method']
    header_dict['Width'] = extract_params['extract_width']
    # Calculate the limits of each wavelength bin.
    half_width = make_bins(wave1d)[1] / 2

    # Pack the stellar spectra and save to file if requested.
    data = [wave1d, np.abs(half_width), flux_clip, ferr, times]
    names = ['Wave', 'Wave Err', 'Flux', 'Flux Err', 'Time']
    units = ['Micron', 'Micron', 'e/s', 'e/s', 'MJD_TDB']
    spectra = utils.save_extracted_spectra(filename, data, names, units, header_dict,
                                           header_comments, save_results=save_results)

    return spectra


def format_nirspec_spectra(datafiles, times, extract_params, target_name, detector, st_teff=None,
                           st_logg=None, st_met=None, output_dir='./', save_results=True):
    """Unpack the outputs of the 1D extraction and format them into
    lightcurves at the native detector resolution.

    Parameters
    ----------
    datafiles : array-like[str], array-like[MultiSpecModel], tuple
        Input extract1d data files.
    times : array-like[float]
        Time stamps corresponding to each integration.
    output_dir : str
        Directory to which to save outputs.
    save_results : bool
        If True, save outputs to file.
    extract_params : dict
        Dictonary of parameters used for the 1D extraction.
    target_name : str
        Name of the target.
    detector : str
        Detector name.
    st_teff : float, None
        Stellar effective temperature.
    st_logg : float, None
        Stellar log surface gravity.
    st_met : float, None
        Stellar metallicity as [Fe/H].

    Returns
    -------
    spectra : dict
        1D stellar spectra at the native detector resolution.
    """

    fancyprint('Formatting extracted 1d spectra.')
    # Box extract outputs will just be a tuple of arrays.
    wave1d = datafiles[0][0]
    flux = datafiles[1]
    ferr = datafiles[2]

    # Remove any NaN pixels --- important for NIRSpec NRS1.
    ii = np.where(np.isfinite(wave1d))[0]
    wave1d_trim = wave1d[ii]

    # Now cross-correlate with stellar model.
    # If one or more of the stellar parameters are not provided, use the wavelength solution from
    # pastasoss.
    if None in [st_teff, st_logg, st_met]:
        fancyprint('Stellar parameters not provided. Using default wavelength solution.',
                   msg_type='WARNING')
    else:
        fancyprint('Refining the wavelength calibration.')
        # Create a grid of stellar parameters, and download PHOENIX spectra for each grid point.
        thisout = output_dir + 'phoenix_models'
        utils.verify_path(thisout)
        res = utils.download_stellar_spectra(st_teff, st_logg, st_met, outdir=thisout)
        wave_file, flux_files = res
        # Interpolate model grid to correct stellar parameters.
        # Reverse direction of both arrays since SOSS is extracted red to blue.
        mod_flux = utils.interpolate_stellar_model_grid(flux_files, st_teff, st_logg, st_met)
        mod_wave = fits.getdata(wave_file) / 1e4

        # Bin model down to data wavelengths.
        mod_flux = spectres.spectres(wave1d_trim, mod_wave, mod_flux)

        # Cross-correlate extracted spectrum with model to refine wavelength calibration.
        x1d_flux = np.nansum(flux, axis=0)[ii]
        wave_shift = do_ccf(wave1d_trim, x1d_flux, mod_flux, oversample=1)
        fancyprint('Found a wavelength shift of {}um'.format(wave_shift))
        wave1d += wave_shift

    # Clip remaining 3-sigma outliers.
    flux_clip = utils.sigma_clip_lightcurves(flux, window=11, thresh=3)

    # Pack the lightcurves into the output format.
    # Put 1D extraction parameters in the output file header.
    filename = (output_dir + target_name[:-2] + '_' + detector + '_' + extract_params['method'] +
                '_spectra_fullres.fits')
    header_dict, header_comments = utils.get_default_header()
    header_dict['Target'] = target_name[:-2]
    header_dict['Contents'] = 'Full resolution stellar spectra'
    header_dict['Method'] = extract_params['method']
    header_dict['Width'] = extract_params['extract_width']
    # Calculate the limits of each wavelength bin.
    half_width = make_bins(wave1d)[1] / 2

    # Pack the stellar spectra and save to file if requested.
    data = [wave1d, np.abs(half_width), flux_clip, ferr, times]
    names = ['Wave', 'Wave Err', 'Flux', 'Flux Err', 'Time']
    units = ['Micron', 'Micron', 'e/s', 'e/s', 'MJD_TDB']
    spectra = utils.save_extracted_spectra(filename, data, names, units, header_dict,
                                           header_comments, save_results=save_results)

    return spectra


def format_soss_spectra(datafiles, times, extract_params, target_name, st_teff=None, st_logg=None,
                        st_met=None, pwcpos=None, output_dir='./', save_results=True,
                        use_pastasoss=False):
    """Unpack the outputs of the 1D extraction and format them into lightcurves at the native
    detector resolution.

    Parameters
    ----------
    datafiles : list(MultiSpecModel), tuple
        Input extract1d data files.
    times : array-like(float)
        Time stamps corresponding to each integration.
    output_dir : str
        Directory to which to save outputs.
    save_results : bool
        If True, save outputs to file.
    extract_params : dict
        Dictonary of parameters used for the 1D extraction.
    target_name : str
        Name of the target.
    st_teff : float, None
        Stellar effective temperature.
    st_logg : float, None
        Stellar log surface gravity.
    st_met : float, None
        Stellar metallicity as [Fe/H].
    pwcpos : float
        Filter wheel position. Only necessary is use_pastasoss is True.
    use_pastasoss : bool
        If True, use pastasoss package to predict wavelength solution based on pupil wheel position.
        Note that this will only allow the extraction of order 2 from 0.6 - 0.85µm.

    Returns
    -------
    spectra : dict
        1D stellar spectra at the native detector resolution.
    """

    fancyprint('Formatting extracted 1d spectra.')
    companion_data = None
    # Box and double-Gaussian extract outputs are local arrays.
    if isinstance(datafiles, dict):
        wave1d_o1 = datafiles['wave_o1']
        flux_o1 = datafiles['flux_o1']
        ferr_o1 = datafiles['ferr_o1']
        wave1d_o2 = datafiles['wave_o2']
        flux_o2 = datafiles['flux_o2']
        ferr_o2 = datafiles['ferr_o2']
        if 'flux_o1_companion' in datafiles:
            companion_data = {'flux_o1': datafiles['flux_o1_companion'],
                              'ferr_o1': datafiles['ferr_o1_companion'],
                              'flux_o2': datafiles['flux_o2_companion'],
                              'ferr_o2': datafiles['ferr_o2_companion']}
    elif isinstance(datafiles, tuple):
        wave1d_o1 = datafiles[0]
        flux_o1 = datafiles[1]
        ferr_o1 = datafiles[2]
        wave1d_o2 = datafiles[3]
        flux_o2 = datafiles[4]
        ferr_o2 = datafiles[5]

    # Whereas ATOCA extract outputs are in the atoca extract1dstep format.
    else:
        # Open the datafiles, and pack the wavelength, flux, and flux error information into data
        # cubes.
        datafiles = np.atleast_1d(datafiles)
        for i, file in enumerate(datafiles):
            segment = utils.unpack_atoca_spectra(file)
            if i == 0:
                wave2d_o1 = segment[1]['WAVELENGTH']
                flux_o1 = segment[1]['FLUX']
                ferr_o1 = segment[1]['FLUX_ERROR']
                wave2d_o2 = segment[2]['WAVELENGTH']
                flux_o2 = segment[2]['FLUX']
                ferr_o2 = segment[2]['FLUX_ERROR']
            else:
                wave2d_o1 = np.concatenate([wave2d_o1, segment[1]['WAVELENGTH']])
                flux_o1 = np.concatenate([flux_o1, segment[1]['FLUX']])
                ferr_o1 = np.concatenate([ferr_o1, segment[1]['FLUX_ERROR']])
                wave2d_o2 = np.concatenate([wave2d_o2, segment[2]['WAVELENGTH']])
                flux_o2 = np.concatenate([flux_o2, segment[2]['FLUX']])
                ferr_o2 = np.concatenate([ferr_o2, segment[2]['FLUX_ERROR']])
        # Create 1D wavelength axes from the 2D wavelength solution.
        wave1d_o1, wave1d_o2 = wave2d_o1[0], wave2d_o2[0]

    # Refine wavelength solution.
    if use_pastasoss is True:
        # Use PASTASOSS to predict wavelength solution from pupil wheel position.
        # Note that PASTASOSS only predicts positions and thus wavelengths for order 2 bluewards of
        # ~0.9µm. Therefore, the whole frame cannot be extracted for order 2. PASTASOSS also does
        # not take into account any TA inaccuracies resulting in the position of the target trace
        # not being in the center of the frame - which will effect the resulting wavelength
        # solution.
        fancyprint('Using PASTASOSS to predict wavelength solution.')
        wave1d_o1 = pastasoss.get_soss_traces(pwcpos=pwcpos, order='1', interp=True).wavelength
        soln_o2 = pastasoss.get_soss_traces(pwcpos=pwcpos, order='2', interp=True)
        xpos_o2, wave1d_o2 = soln_o2.x.astype(int), soln_o2.wavelength
        # Trim extracted quantities to match shapes of pastasoss quantities.
        flux_o1 = flux_o1[:, 4:-4]
        ferr_o1 = ferr_o1[:, 4:-4]
        flux_o2 = flux_o2[:, xpos_o2]
        ferr_o2 = ferr_o2[:, xpos_o2]
        if companion_data is not None:
            companion_data['flux_o1'] = companion_data['flux_o1'][:, 4:-4]
            companion_data['ferr_o1'] = companion_data['ferr_o1'][:, 4:-4]
            companion_data['flux_o2'] = companion_data['flux_o2'][:, xpos_o2]
            companion_data['ferr_o2'] = companion_data['ferr_o2'][:, xpos_o2]

    # Cross-correlate with stellar model.
    # If one or more of the stellar parameters are not provided, use the existing wavelength
    # solution.
    if None in [st_teff, st_logg, st_met]:
        fancyprint('Stellar parameters not provided. Using default wavelength solution.',
                   msg_type='WARNING')
    else:
        fancyprint('Refining the wavelength calibration.')
        # Create a grid of stellar parameters, and download PHOENIX spectra for each grid point.
        thisout = output_dir + 'phoenix_models'
        utils.verify_path(thisout)
        res = utils.download_stellar_spectra(st_teff, st_logg, st_met, outdir=thisout)
        wave_file, flux_files = res
        # Interpolate model grid to correct stellar parameters.
        # Reverse direction of both arrays since SOSS is extracted red to blue.
        mod_flux = utils.interpolate_stellar_model_grid(flux_files, st_teff, st_logg, st_met)
        mod_wave = fits.getdata(wave_file)/1e4

        # Bin model down to data wavelengths.
        mod_flux = spectres.spectres(wave1d_o1[::-1], mod_wave, mod_flux)[::-1]

        # Cross-correlate extracted spectrum with model to refine wavelength calibration.
        x1d_flux = np.nansum(flux_o1, axis=0)
        wave_shift = do_ccf(wave1d_o1, x1d_flux, mod_flux)
        fancyprint('Found a wavelength shift of {}um'.format(wave_shift))
        wave1d_o1 += wave_shift
        wave1d_o2 += wave_shift

    # Invert so wavelengths are in increasing order.
    wave1d_o1 = wave1d_o1[::-1]
    wave1d_o2 = wave1d_o2[::-1]
    flux_o1 = flux_o1[:, ::-1]
    flux_o2 = flux_o2[:, ::-1]
    ferr_o1 = ferr_o1[:, ::-1]
    ferr_o2 = ferr_o2[:, ::-1]
    if companion_data is not None:
        companion_data['flux_o1'] = companion_data['flux_o1'][:, ::-1]
        companion_data['flux_o2'] = companion_data['flux_o2'][:, ::-1]
        companion_data['ferr_o1'] = companion_data['ferr_o1'][:, ::-1]
        companion_data['ferr_o2'] = companion_data['ferr_o2'][:, ::-1]

    # Clip remaining 5-sigma outliers.
    flux_o1_clip = utils.sigma_clip_lightcurves(flux_o1)
    flux_o2_clip = utils.sigma_clip_lightcurves(flux_o2)
    if companion_data is not None:
        flux_o1_comp_clip = utils.sigma_clip_lightcurves(companion_data['flux_o1'])
        flux_o2_comp_clip = utils.sigma_clip_lightcurves(companion_data['flux_o2'])

    # Pack the lightcurves into the output format.
    # Put 1D extraction parameters in the output file header.
    filename = (output_dir + target_name[:-2] + '_' + extract_params['method'] +
                '_spectra_fullres.fits')
    header_dict, header_comments = utils.get_default_header()
    header_dict['Target'] = target_name[:-2]
    header_dict['Contents'] = 'Full resolution stellar spectra'
    header_dict['Method'] = extract_params['method']
    header_dict['Width'] = extract_params['extract_width']
    # Calculate the limits of each wavelength bin.
    half_width_o1 = make_bins(wave1d_o1)[1] / 2
    half_width_o2 = make_bins(wave1d_o2)[1] / 2

    # Pack the stellar spectra and save to file if requested.
    data = [wave1d_o1, np.abs(half_width_o1), flux_o1_clip, ferr_o1,
            wave1d_o2, np.abs(half_width_o2), flux_o2_clip, ferr_o2, times]
    names = ['Wave O1', 'Wave Err O1', 'Flux O1', 'Flux Err O1',
             'Wave O2', 'Wave Err O2', 'Flux O2', 'Flux Err O2', 'Time']
    units = ['Micron', 'Micron', 'DN/s', 'DN/s',
             'Micron', 'Micron', 'DN/s', 'DN/s', 'MJD_TDB']
    if companion_data is not None:
        data[8:8] = [flux_o1_comp_clip, companion_data['ferr_o1'],
                     flux_o2_comp_clip, companion_data['ferr_o2']]
        names[8:8] = ['Flux O1 Companion', 'Flux Err O1 Companion',
                      'Flux O2 Companion', 'Flux Err O2 Companion']
        units[8:8] = ['DN/s', 'DN/s', 'DN/s', 'DN/s']
    spectra = utils.save_extracted_spectra(filename, data, names, units, header_dict,
                                           header_comments, save_results=save_results)

    return spectra


def get_soss_estimate(atoca_spectra, output_dir):
    """Convert the AtocaSpectra output of ATOCA into the format expected for a soss_estimate.

    Parameters
    ----------
    atoca_spectra : str, MultiSpecModel
        AtocaSpectra datamodel, or path to the datamodel.
    output_dir : str
        Directory to which to save results.

    Returns
    -------
    estimate_filename : str
        Path to soss_estimate file.
    """

    # Open the AtocaSpectra file.
    atoca_spec = datamodels.open(atoca_spectra)
    # Get the spectrum.
    for spec in atoca_spec.spec:
        if spec.meta.soss_extract1d.type == 'OBSERVATION':
            estimate = datamodels.SpecModel(spec_table=spec.spec_table)
            break
    # Save the spectrum as a soss_estimate file.
    estimate_filename = estimate.save(output_dir + 'soss_estimate.fits')

    return estimate_filename


def get_spatial_prof_opt(deepframe, ymin=0, ymax=None, xmin=0, xmax=None):
    """Create a normalized spatial profile from a deep stack for optimal extraction.

    Parameters
    ----------
    deepframe : ndarray(float)
        Median stack of the observations.
    ymin : int, ndarray(int)
        Minimum y value to use for normalization.
    ymax : int, ndarray(int), None
        Maximum y value to use for normalization.
    xmin : int, None
        Minimum x value to consider.
    xmax : int, None
        Maximum x value to consider.

    Returns
    -------
    prof : ndarray(float)
        Column-normalized spatial profile.
    """

    if ymax is None:
        ymax = np.shape(deepframe)[0]
    if xmax is None:
        xmax = np.shape(deepframe)[1]
    ymax = np.atleast_1d(ymax)
    ymin = np.atleast_1d(ymin)

    deepframe[deepframe < 0] = 0  # Ensure positivity
    deepframe[:, :5] = deepframe[:, 5][:, None]  # Interpolate edge columns
    deepframe[:, -6:] = deepframe[:, -6][:, None]

    # Do the profile normalization
    # If the y-bounds are constant.
    prof = np.zeros_like(deepframe)
    if len(ymin) == 1:
        assert len(ymax) == 1
        ymin, ymax = ymin[0], ymax[0]
        prof[:, xmin:xmax] = deepframe[:, xmin:xmax] / np.nansum(deepframe[ymin:ymax, xmin:xmax],
                                                                 axis=0)
    # For non-constant y-bounds.
    else:
        xdim = deepframe[:, xmin:xmax].shape[1]
        assert len(ymax) == xdim
        assert len(ymin) == xdim
        for x in range(xmin, xmax):
            xx = x - xmin
            prof[:, x] = deepframe[:, x] / np.nansum(deepframe[ymin[xx]:ymax[xx], x], axis=0)

    return prof


def get_wave_miri(datafile, centroids, nint, nwave):
    """Get the default MIRI wavelngth solution.

    Parameters
    ----------
    datafile : str
        Datafile from the observation.
    centroids : dict
        Centroids dictionary.
    nint : int
        Number of integrations in the observation.
    nwave : int
        Number of wavelength channels in the observation.

    Returns
    -------
    wave : ndarray[float]
        2D wavelength solution.
    """

    with datamodels.open(datafile) as d:
        wave2d = d.wavelength
    # Get 1D wavelengths at the locations of the trace centroids.
    wave1d = np.ones(nwave) * np.nan
    x1, y1 = centroids['xpos'].values, centroids['ypos'].values
    for x, y in zip(x1, y1):
        wave1d[int(y)] = wave2d[int(y), int(x)]

    wave = np.repeat(wave1d[np.newaxis, :], nint, axis=0)

    return wave


def get_wave_nirspec(datafile, centroids, nint, nwave):
    """Get the default NIRSpec wavelngth solution.

    Parameters
    ----------
    datafile : str
        Datafile from the observation.
    centroids : dict
        Centroids dictionary.
    nint : int
        Number of integrations in the observation.
    nwave : int
        Number of wavelength channels in the observation.

    Returns
    -------
    wave : ndarray[float]
        2D wavelength solution.
    """

    # Get default 2D wavelength solution.
    with datamodels.open(datafile) as d:
        wave2d = d.wavelength
    # Get 1D wavelengths at the locations of the trace centroids.
    wave1d = np.ones(nwave) * np.nan
    x1, y1 = centroids['xpos'].values, centroids['ypos'].values
    for x, y in zip(x1, y1):
        wave1d[int(x)] = wave2d[int(y), int(x)]

    wave = np.repeat(wave1d[np.newaxis, :], nint, axis=0)

    return wave


def get_wave_soss(datafile):
    """Get the default NIRISS wavelngth solution.

    Parameters
    ----------
    datafile : str
        Datafile from the observation.

    Returns
    -------
    wave_o1 : ndarray[float]
        2D wavelength solution for order 1
    wave_o2 : ndarray[float]
        2D wavelength solution for order 2
    """

    step = calwebb_spec2.extract_1d_step.Extract1dStep()
    wavemap = step.get_reference_file(datafile, 'wavemap')
    # Remove 20 pixel padding that is there for some reason.
    wave_o1 = np.mean(fits.getdata(wavemap, 1)[20:-20, 20:-20], axis=0)
    wave_o2 = np.mean(fits.getdata(wavemap, 2)[20:-20, 20:-20], axis=0)

    return wave_o1, wave_o2


def optimal_extract_miri(datafiles, deepframe, centroids, extract_width=None, max_iter=25,
                         var_thresh=25):
    """Perform am optimal extraction on MIRI.

    Parameters
    ----------
    datafiles : array-like[str], array-like[jwst.RampModel]
        Input datamodels or paths to datamodels for each segment.
    deepframe : array-like(float)
        Median stack of the observation.
    centroids : dict
        Dictionary of centroid positions.
    extract_width : int, None
        Width of extraction box.
    max_iter : int
        Maximum number of outlier rejection iterations to perform during extraction.
    var_thresh : int
        Variance threshold for a pixel to be flagged as an outlier.


    Returns
    -------
    wave : ndarray[float]
        2D wavelength solution.
    flux : ndarray[float]
        2D extracted flux.
    ferr: ndarray[float]
        2D flux errors.
    extract_width : int
        Optimized aperture width.
    """

    datafiles = np.atleast_1d(datafiles)
    # Get flux to extract.
    for i, file in enumerate(datafiles):
        if isinstance(file, str):
            data = fits.getdata(file)
        else:
            with utils.open_filetype(file) as datamodel:
                data = datamodel.data
        if i == 0:
            cube = data
        else:
            cube = np.concatenate([cube, data])

    # Get centroid positions.
    nint, dimy, dimx = np.shape(cube)
    x1, y1 = centroids['xpos'].values, centroids['ypos'].values
    # If an extraction width is provided, only extract over this region.
    if extract_width is not None:
        ymax = np.round(np.min([x1 + extract_width/2, np.ones_like(x1) * dimx], axis=0), 0).astype(int)
        ymin = np.round(np.max([x1 - extract_width/2, np.zeros_like(x1)], axis=0), 0).astype(int)
    # If not, extract the entire frame.
    else:
        ymin = 0
        ymax = dimx

    # ===== Extraction ======
    # Do the extraction.
    fancyprint('Performing optimal extraction.')
    flux, ferr = do_optimal_extraction(cube.transpose(0, 2, 1), deepframe.transpose(1, 0), ymin,
                                       ymax, xmin=int(np.min(y1)), xmax=int(np.max(y1)+1),
                                       max_iter=max_iter, var_thresh=var_thresh)

    # Get default 2D wavelength solution.
    wave = get_wave_miri(datafiles[0], centroids, cube.shape[0], cube.shape[1])

    return wave, flux, ferr, extract_width


def optimal_extract_nirspec(datafiles, deepframe, centroids, extract_width=None, max_iter=25,
                            var_thresh=25):
    """Perform am optimal extraction on NIRSpec.

    Parameters
    ----------
    datafiles : array-like[str], array-like[jwst.RampModel]
        Input datamodels or paths to datamodels for each segment.
    deepframe : array-like(float)
        Median stack of the observation.
    centroids : dict
        Dictionary of centroid positions for all SOSS orders.
    extract_width : int, None
        Width of extraction box.
    max_iter : int
        Maximum number of outlier rejection iterations to perform during extraction.
    var_thresh : int
        Variance threshold for a pixel to be flagged as an outlier.

    Returns
    -------
    wave : ndarray[float]
        2D wavelength solution.
    flux : ndarray[float]
        2D extracted flux.
    ferr: ndarray[float]
        2D flux errors.
    """

    datafiles = np.atleast_1d(datafiles)
    # Get flux to extract.
    for i, file in enumerate(datafiles):
        if isinstance(file, str):
            data = fits.getdata(file)
        else:
            with utils.open_filetype(file) as datamodel:
                data = datamodel.data
        if i == 0:
            cube = data
        else:
            cube = np.concatenate([cube, data])

    # Get centroid positions.
    nint, dimy, dimx = np.shape(cube)
    x1, y1 = centroids['xpos'].values, centroids['ypos'].values
    # If an extraction width is provided, only extract over this region.
    if extract_width is not None:
        ymax = np.round(np.min([y1 + extract_width/2, np.ones_like(y1) * dimy], axis=0), 0).astype(int)
        ymin = np.round(np.max([y1 - extract_width/2, np.zeros_like(y1)], axis=0), 0).astype(int)
    # If not, extract the entire frame.
    else:
        ymin = 0
        ymax = dimy

    # ===== Extraction ======
    # Do the extraction.
    fancyprint('Performing optimal extraction.')
    det = utils.get_nrs_detector_name(datafiles[0])
    subarray = utils.get_soss_subarray(datafiles[0])
    grating = utils.get_nrs_grating(datafiles[0])
    xstart = utils.get_nrs_trace_start(det, subarray, grating)
    flux, ferr = do_optimal_extraction(cube, deepframe, ymin, ymax, xmin=xstart, max_iter=max_iter,
                                       var_thresh=var_thresh)

    # Get default 2D wavelength solution.
    wave = get_wave_nirspec(datafiles[0], centroids, cube.shape[0], cube.shape[2])

    return wave, flux, ferr


def trace_spectrum(datafiles, deepframe, output_dir='./', save_results=True, fileroot_noseg='',
                   do_plot=False, show_plot=False, allow_miri_slope=False, extract_width=None,
                   extract_width_soss2=None):
    """Trace the 2D spectrum on the detector.

    Parameters
    ----------
    datafiles : array-like(RampModel), array-like(str)
        Datamodels for each segment of the TSO.
    deepframe : ndarray(float)
        Deep stack for the TSO. Should be 2D (dimy, dimx).
    output_dir : str
        Directory to which to save outputs.
    save_results : bool
        If Tre, save results to file.
    fileroot_noseg : str
        Root file name with no segment information.
    do_plot : bool
        If True, do the step diagnostic plot.
    show_plot : bool
        If True, show the step diagnostic plot instead of/in addition to saving it to file.
    allow_miri_slope : bool
        If True, allow the MIRI centroids to be sloped.
    extract_width : int, tuple(float, float), None
        Extraction full width. A two-element tuple is interpreted as an asymmetric
        `(lower_width, upper_width)` aperture for box extraction.
    extract_width_soss2 : int, tuple(float, float), None
        Extraction full width for SOSS order 2. A two-element tuple is interpreted as an
        asymmetric `(lower_width, upper_width)` aperture for box extraction.

    Returns
    -------
    centroids : np.ndarray(float), str
        Trace centroids for all orders, or path to centroids file.
    """

    datafiles = np.atleast_1d(datafiles)

    # Get centroids for orders one to three
    fancyprint('Finding trace centroids.')
    instrument = utils.get_instrument_name(datafiles[0])
    if instrument == 'NIRISS':
        subarray = utils.get_soss_subarray(datafiles[0])
        # Get the most up to date trace table file.
        step = calwebb_spec2.extract_1d_step.Extract1dStep()
        tracetable = step.get_reference_file(datafiles[0], 'spectrace')
        # Get centroids via the edgetrigger method.
        save_filename = output_dir + fileroot_noseg
        centroids = utils.get_centroids_soss(deepframe, tracetable, subarray,
                                             save_results=save_results, save_filename=save_filename)
    elif instrument == 'NIRSPEC':
        # Get centroids via the edgetrigger method.
        save_filename = output_dir + fileroot_noseg
        det = utils.get_nrs_detector_name(datafiles[0])
        subarray = utils.get_soss_subarray(datafiles[0])
        grating = utils.get_nrs_grating(datafiles[0])
        xstart = utils.get_nrs_trace_start(det, subarray, grating)
        centroids = utils.get_centroids_nirspec(deepframe, xstart=xstart, save_results=save_results,
                                                save_filename=save_filename)
    else:
        # Get centroids via the edgetrigger method.
        save_filename = output_dir + fileroot_noseg
        centroids = utils.get_centroids_miri(deepframe, ystart=50, save_results=save_results,
                                             save_filename=save_filename,
                                             allow_slope=allow_miri_slope)

    # Do diagnostic plot if requested.
    if do_plot is True:
        if save_results is True:
            if instrument == 'NIRSPEC':
                outfile = output_dir + 'centroiding_{}.png'.format(det)
            else:
                outfile = output_dir + 'centroiding.png'
        else:
            outfile = None
        miri_scale = False
        if instrument == 'MIRI':
            miri_scale = True
        plotting.make_centroiding_plot(deepframe, centroids, instrument, show_plot=show_plot,
                                       outfile=outfile, miri_scale=miri_scale,
                                       extract_width=extract_width,
                                       extract_width_soss2=extract_width_soss2)

    if save_results is True:
        centroids = save_filename + 'centroids.csv'

    return centroids


def wave_and_flux_calibrations(pwcpos, obs_x_pixel, photom_path, spectrace_path, order=1):
    """This function wavelength and flux calibrates an input spectrum expressed in adu per sec
    sampled at a detector x position in pixels. This uses the DMS reference files  (assuming they
    are correct) and corrects  them using the pwcpos keyword to shift the pixels by -11 pixel/degre.
    An important limitation is that the reference flux calibration is applicable to spectra
    extracted using a box aperture of 40 pixels. So the obs_flux_adusec must have been extracted
    using that same box size otherwise the flux may be systematically off.
    Function originally by Loïc Albert and adapted by MCR.

    Parameters
    ----------
    pwcpos : float
        Observation pupil wheel position.
    obs_x_pixel : array-like(float)
        Pixel x-indices.
    photom_path : str
        Path to photom rev2 file
    spectrace_path : str
        Path to spectrace reference file.
    order : int
        SOSS order to calibrate

    Returns
    -------
    obs_flux_scaling : array-like(float)
        The ADU/s to Jy flux scaling.
    """

    # The jwst_niriss_photom_rev2.fits calibration was obtained for the PID 1091 obs 2 data set
    # (BD+601758). So, return results based on that.
    pwcpos_fluxcal = 245.7909

    # The empirically determined relation between PWCPOS and movement of the traces along the x
    # axis was obtained using the Tilt test fro CV3 observations with a span of +/- 10 degrees of
    # PWCPOS.
    xoffset_perdeg = -11.0

    # Use the photomstep ref file rev2 from Kevin
    if order == 3:
        m = 3
    elif order == 2:
        m = 2
    else:
        m = 1

    # Read the flux scaling that needs to be applied to the uncalibrated input
    # flux in adu/sec to output the calibrated flux in Jy.
    # This scaling applies for an extraction aperture of 40 pixels.
    # Its PWCPOS is 245.7909
    hdu = fits.open(photom_path)
    # Scaling to convert from adu/sec to Jy
    w = hdu[1].data[m - 1]['wavelength']
    scaling = hdu[1].data[m - 1]['relresponse'] * hdu[1].data[m - 1]['photmj']
    # Remove zeros from both
    ind = (w != 0) | (scaling != 0)
    fluxcal_wave_micron = w[ind]
    fluxcal_scaling = scaling[ind]
    # The wavelength sampling is almost constant. It's bimodal alternating
    # between 0.97 nm and 0.98 nm with excursions of 0.001 or 0.002 nm around
    # each value. The sampling jumps from one mode the the other between
    # consecutive samples. Weird. The sampling does not quite match the pixel
    # sampling which is on everage ~0.97 nm/pixel but with gradual changes.

    # The wavelength calibration reference file samples the wavelength at
    # every 1 nm and gives the corresponding x pixel positions.
    hdu = fits.open(spectrace_path)
    wavecal_wave_micron = hdu[m].data['wavelength']
    wavecal_x_pixel = hdu[m].data['x']

    # Based on the current pwcpos, shift the wavelength solution by
    # some x pixel offset. The pwcpos offset is relative to that of the flux
    # calibration data set of BD+60.1758 with which these 2 calibrations
    # have been made.
    pwcpos_offset = pwcpos - pwcpos_fluxcal
    x_offset = xoffset_perdeg * pwcpos_offset

    # Directly interpolate the calibrations at the requested sampling
    # Wavelength calibration
    ind = np.argsort(wavecal_x_pixel)
    obs_wave_micron = np.interp(obs_x_pixel, wavecal_x_pixel[ind] + x_offset,
                                wavecal_wave_micron[ind])
    # Flux calibration
    ind = np.argsort(fluxcal_wave_micron)
    obs_flux_scaling = np.interp(obs_wave_micron, fluxcal_wave_micron[ind],
                                 fluxcal_scaling[ind])

    return obs_flux_scaling


def run_stage3(results, save_results=True, root_dir='./', force_redo=False, extract_method='box',
               soss_specprofile=None, centroids=None, extract_width=40, extract_width_soss2=None,
               st_teff=None, st_logg=None, st_met=None, planet_letter='b', output_tag='',
               do_plot=False, show_plot=False, opt_max_iter=25, opt_var_thresh=25, deepframe=None,
               saturation_rescue=False, mask_do_not_use_pixels=True,
               pipeline_outputs_directory='pipeline_outputs_directory', **kwargs):
    """Run the exoTEDRF Stage 3 pipeline: 1D spectral extraction, using a combination of the
    official STScI DMS and custom steps.

    Parameters
    ----------
    results : array-like(str), array-like(CubeModel)
        exoTEDRF Stage 2 outputs for each segment.
    save_results : bool
        If True, save the results of each step to file.
    root_dir : str
        Directory from which all relative paths are defined.
    force_redo : bool
        If True, redo steps even if outputs files are already present.
    extract_method : str
        Either 'box', 'optimal', or 'atoca'.
    soss_specprofile : str, None
        Specprofile reference file; only neceessary for ATOCA extractions.
    centroids : str, None
        Path to file containing trace positions for each order.
    extract_width : int, tuple(float, float), str
        Width around the trace centroids, in pixels, for the 1D extraction. A two-element tuple is
        interpreted as an asymmetric `(lower_width, upper_width)` aperture for box extraction.
    extract_width_soss2 : int, tuple(float, float), str, None
        Width of extraction box for order 2. If None, will use the same aperture as order 1. A
        two-element tuple is interpreted as an asymmetric `(lower_width, upper_width)` aperture for
        box extraction.
    st_teff : float, None
        Stellar effective temperature.
    st_logg : float, None
        Stellar log surface gravity.
    st_met : float, None
        Stellar metallicity as [Fe/H].
    planet_letter : str
        Letter designation for the planet.
    output_tag : str
        Name tag to append to pipeline outputs directory.
    do_plot : bool
        If True, make step diagnostic plot.
    show_plot : bool
        Only necessary if do_plot is True. Show the diagnostic plots in addition to/instead of
        saving to file.
    opt_max_iter : int
        Maximum number of outlier rejection iterations to perform during optimal extraction.
    opt_var_thresh : int
        Variance threshold for a pixel to be flagged as an outlier during optimal exraction.
    deepframe : str, None
        Path to file containing a median stack of the observation.
    saturation_rescue : bool
        If True for NIRISS/SOSS box extraction, keep post-RampFit pixels whose ramps were only
        partially saturated so RampFit's pre-saturation slope estimate can be extracted.
    mask_do_not_use_pixels : bool
        If True, NaN DO_NOT_USE pixels before box extraction in addition to saturation handling.

    Returns
    -------
    specra : dict
        1D stellar spectra for each wavelength bin at the native detector resolution.
    """

    # ============== DMS Stage 3 ==============
    # 1D spectral extraction.
    fancyprint('**Starting exoTEDRF Stage 3**')
    fancyprint('1D spectral extraction...')

    if output_tag != '':
        output_tag = '_' + output_tag
    # Create output directories and define output paths.
    if os.path.isabs(pipeline_outputs_directory) or pipeline_outputs_directory.startswith('~'):
        base_dir = os.path.expanduser(pipeline_outputs_directory) + output_tag
    else:
        base_dir = os.path.join(root_dir, pipeline_outputs_directory + output_tag)
    utils.verify_path(base_dir)
    utils.verify_path(os.path.join(base_dir, 'Stage3'))
    outdir = os.path.join(base_dir, 'Stage3/')

    # ===== SpecProfile Construction Step =====
    # Custom DMS step
    if extract_method == 'atoca':
        if soss_specprofile is None:
            if 'SpeProfileStep' in kwargs.keys():
                step_kwargs = kwargs['SpeProfileStep']
            else:
                step_kwargs = {}
            step = SpecProfileStep(results, output_dir=outdir)
            soss_specprofile = step.run(force_redo=force_redo, **step_kwargs)

    # ===== 1D Extraction Step =====
    # Custom/default DMS step.
    if 'Extract1dStep' in kwargs.keys():
        step_kwargs = kwargs['Extract1dStep']
    else:
        step_kwargs = {}
    step = Extract1DStep(results, extract_method=extract_method, st_teff=st_teff, st_logg=st_logg,
                         st_met=st_met, planet_letter=planet_letter,  output_dir=outdir)
    spectra = step.run(extract_width=extract_width, extract_width_soss2=extract_width_soss2,
                       soss_specprofile=soss_specprofile, centroids=centroids,
                       save_results=save_results, force_redo=force_redo, do_plot=do_plot,
                       show_plot=show_plot, deepframe=deepframe, opt_max_iter=opt_max_iter,
                       opt_var_thresh=opt_var_thresh, saturation_rescue=saturation_rescue,
                       mask_do_not_use_pixels=mask_do_not_use_pixels, **step_kwargs)

    return spectra
