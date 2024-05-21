#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thurs Jul 21 17:33 2022

@author: MCR

Custom JWST DMS pipeline steps for Stage 3 (1D spectral extraction).
"""

from astropy.convolution import Gaussian1DKernel, convolve
from astropy.io import fits
import glob
import numpy as np
import pandas as pd
import pastasoss
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt
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
        datafiles = utils.sort_datamodels(input_data)
        self.datafiles = []
        for file in datafiles:
            self.datafiles.append(utils.open_filetype(file))

        # Get subarray identifier.
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
        expected_file = self.output_dir + 'APPLESOSS_ref_2D_profile_{}_os1_pad20.fits'.format(self.subarray)
        if expected_file in all_files and force_redo is False:
            fancyprint('File {} already exists.'.format(expected_file))
            fancyprint('Skipping SpecProfile Reference Construction Step.')
            specprofile = expected_file
        # If no output files are detected, run the step.
        else:
            specprofile = specprofilestep(self.datafiles,
                                          output_dir=self.output_dir,
                                          empirical=empirical)
            specprofile = self.output_dir + specprofile

        return specprofile


class Extract1DStep:
    """Wrapper around default calwebb_spec2 1D Spectral Extraction step, with
    custom modifications.
    """

    def __init__(self, input_data, extract_method, st_teff=None, st_logg=None,
                 st_met=None, planet_letter='b', output_dir='./'):
        """Step initializer.

        Parameters
        ----------
        input_data : array-like(str), array-like(datamodel)
            List of paths to input data or the input data itself.
        extract_method : str
            1D extraction method to use; either "box" or "atoca".
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
        datafiles = utils.sort_datamodels(input_data)
        self.datafiles = []
        for file in datafiles:
            self.datafiles.append(utils.open_filetype(file))
        self.fileroots = utils.get_filename_root(self.datafiles)
        self.fileroot_noseg = utils.get_filename_root_noseg(self.fileroots)

        # Set planet and star attributes.
        with utils.open_filetype(self.datafiles[0]) as datamodel:
            self.target_name = datamodel.meta.target.catalog_name
        self.pl_name = self.target_name + ' ' + planet_letter
        self.stellar_params = [st_teff, st_logg, st_met]

        # Get instrument.
        self.instrument = utils.get_instrument_name(self.datafiles[0])
        if self.instrument == 'NIRSPEC' and extract_method == 'atoca':
            fancyprint('ATOCA extraction selected but observation does not '
                       'use NIRISS/SOSS. Switching to box extraction.',
                       msg_type='WARNING')
            self.extract_method = 'box'

    def run(self, extract_width=40, soss_specprofile=None, centroids=None,
            save_results=True, force_redo=False, do_plot=False,
            show_plot=False, use_pastasoss=False, soss_estimate=None):
        """Method to run the step.

        Parameters
        ----------
        extract_width : int
            Full width of extraction aperture to use.
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
        use_pastasoss : bool
            If True, use pastasoss to esimate trace positions and wavelength
            solution.
        soss_estimate : str, None
            Path to file containing the soss_estimate for atoca extractions.

        Returns
        -------
        spectra : dict
            1D stellar spectra at the native detector resolution.
        """

        fancyprint('Starting 1D extraction using the {} '
                   'method.'.format(self.extract_method))

        # Initialize loop and storange variables.
        all_files = glob.glob(self.output_dir + '*')
        expected_file = self.output_dir + self.target_name + '_' + \
            self.extract_method + '_spectra_fullres.fits'
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
                    raise ValueError('specprofile reference file must be '
                                     'provided for ATOCA extraction.')

                results = atoca_extract_soss(self.datafiles, soss_specprofile,
                                             output_dir=self.output_dir,
                                             save_results=save_results,
                                             extract_width=extract_width,
                                             soss_estimate=soss_estimate,
                                             fileroots=self.fileroots)

            # Option 2: Simple aperture extraction - any instrument.
            elif self.extract_method == 'box':
                if centroids is None:
                    raise ValueError('Centroids must be provided for box '
                                     'extraction.')
                # If file path is passed, open it.
                if isinstance(centroids, str):
                    centroids = pd.read_csv(centroids, comment='#')
                if self.instrument == 'NIRISS':
                    results = box_extract_soss(self.datafiles, centroids,
                                               extract_width, do_plot=do_plot,
                                               show_plot=show_plot,
                                               save_results=save_results,
                                               output_dir=self.output_dir)
                else:
                    results = box_extract_nirspec(self.datafiles, centroids,
                                                  extract_width,
                                                  do_plot=do_plot,
                                                  show_plot=show_plot,
                                                  save_results=save_results,
                                                  output_dir=self.output_dir)
                if extract_width == 'optimize':
                    # Get optimized width.
                    extract_width = int(results[-1])
                results = results[:-1]

            # Raise exception otherwise.
            else:
                raise ValueError('Invalid extraction method')

            # Do step plot if requested - only for atoca.
            if do_plot is True and self.extract_method == 'atoca':
                if save_results is True:
                    plot_file = self.output_dir + self.tag.replace('.fits',
                                                                   '.png')
                else:
                    plot_file = None
                models = []
                for name in self.fileroots:
                    models.append(self.output_dir + name + 'SossExtractModel.fits')
                plotting.make_decontamination_plot(self.datafiles, models,
                                                   outfile=plot_file,
                                                   show_plot=show_plot)

            # Save the final extraction parameters.
            extract_params = {'extract_width': extract_width,
                              'method': self.extract_method}
            # Get timestamps and pupil wheel position.
            for i, datafile in enumerate(self.datafiles):
                with utils.open_filetype(datafile) as file:
                    this_time = file.int_times['int_mid_BJD_TDB']
                if i == 0:
                    times = this_time
                    pwcpos = file.meta.instrument.pupil_position
                else:
                    times = np.concatenate([times, this_time])

            # Clip outliers and format extracted spectra.
            st_teff, st_logg, st_met = self.stellar_params
            if self.instrument == 'NIRISS':
                # Get throughput data.
                step = calwebb_spec2.extract_1d_step.Extract1dStep()
                thpt = step.get_reference_file(self.datafiles[0], 'spectrace')

                spectra = format_soss_spectra(results, times, extract_params,
                                              self.pl_name, st_teff, st_logg,
                                              st_met, throughput=thpt,
                                              pwcpos=pwcpos,
                                              output_dir=self.output_dir,
                                              save_results=save_results,
                                              use_pastasoss=use_pastasoss)
            else:
                thpt = ''
                detector = utils.get_detector_name(self.datafiles[0])
                spectra = format_nirspec_spectra(results, times,
                                                 extract_params, self.pl_name,
                                                 detector, st_teff, st_logg,
                                                 st_met, throughput=thpt,
                                                 output_dir=self.output_dir,
                                                 save_results=save_results)

        return spectra


def specprofilestep(datafiles, empirical=True, output_dir='./'):
    """Wrapper around the APPLESOSS module to construct a specprofile
    reference file tailored to the particular TSO being analyzed.

    Parameters
    ----------
    datafiles : array-like[str], array-like[jwst.RampModel]
        Input datamodels or paths to datamodels for each segment.
    empirical : bool
        If True, construct profiles using only the data. If False, fall back
        on WebbPSF for the trace wings. Note: The current WebbPSF wings are
        known to not accurately match observations. This mode is therefore not
        advised.
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
        data = datamodels.open(file)
        if i == 0:
            cube = data.data
        else:
            cube = np.concatenate([cube, data.data])
        data.close()
    deepstack = utils.make_deepstack(cube)

    # Initialize and run the APPLESOSS module with the median stack.
    spat_prof = applesoss.EmpiricalProfile(deepstack, tracetable=tracetable,
                                           wavemap=wavemap, pad=20)
    if empirical is False:
        # Get the date of the observations to use the calculated WFE models
        # from that time.
        obs_date = fits.getheader(datafiles[0])['DATE-OBS']
        spat_prof.build_empirical_profile(verbose=0, empirical=False,
                                          wave_increment=0.1,
                                          obs_date=obs_date)
    else:
        spat_prof.build_empirical_profile(verbose=0)

    # Save results to file (non-optional).
    if np.shape(deepstack)[0] == 96:
        subarray = 'SUBSTRIP96'
    else:
        subarray = 'SUBSTRIP256'
    filename = spat_prof.write_specprofile_reference(subarray,
                                                     output_dir=output_dir)

    return filename


def atoca_extract_soss(datafiles, specprofile, output_dir='./',
                       save_results=True, extract_width=40,
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
    fileroots : list(str), None
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
                res = step.call(segment,
                                output_dir=output_dir,
                                save_results=save_results,
                                soss_transform=[0, 0, 0],
                                subtract_background=False,
                                soss_bad_pix='model',
                                soss_width=extract_width,
                                soss_modelname=soss_modelname,
                                override_specprofile=specprofile,
                                soss_estimate=soss_estimate)
                results.append(res)
                # Note that this segment was extracted correctly.
                extracted.append(i)
                # The first time that an extraction is successful,
                # create a soss_estimate if one does not already
                # exist.
                if first_time is True and soss_estimate is None:
                    atoca_spectra = output_dir + fileroots[int(i)] + 'AtocaSpectra.fits'
                    soss_estimate = get_soss_estimate(atoca_spectra,
                                                      output_dir=output_dir)
                    first_time = False
            # When using ATOCA, sometimes a very specific error
            # pops up when an initial estimate of the stellar
            # spectrum cannot be obtained. This is needed to
            # establish the wavelength grid (which has a varying
            # resolution to better capture sharp features in
            # stellar spectra). In these cases, the SOSS estimate
            # provides information to create a wavelength grid.
            except Exception as err:
                if str(err) == '(m>k) failed for hidden m: fpcurf0:m=0':
                    # If every segment has been tested and none
                    # work, just fail.
                    if int(i) == len(datafiles) and len(extracted) == 0:
                        msg = 'No segments could be properly ' \
                              'extracted.'
                        fancyprint(msg, msg_type='Error')
                        raise err
                    # If there's still hope, then just skip this
                    # segment for now and move onto the next one.
                    else:
                        msg = 'Initial flux estimate failed, ' \
                              'and no soss estimate provided. ' \
                              'Moving to next segment.'
                        fancyprint(msg, msg_type='WARNING')
                        continue
                # If any other error pops up, raise it.
                else:
                    raise err
        # Remove the extracted segments from the list of ones
        # to extract.
        for seg in extracted:
            to_extract.pop(seg)

    # Sort the segments in chronological order, in case they were
    # processed out of order.
    seg_nums = [seg.meta.exposure.segment_number for seg in
                results]
    ii = np.argsort(seg_nums)
    results = np.array(results)[ii]

    return results


def box_extract_nirspec(datafiles, centroids, extract_width, do_plot=False,
                        show_plot=False, save_results=True, output_dir='./'):
    """Perform a simple box aperture extraction on NIRSpec.

    Parameters
    ----------
    datafiles : array-like[str], array-like[jwst.RampModel]
        Input datamodels or paths to datamodels for each segment.
    centroids : dict
        Dictionary of centroid positions for all SOSS orders.
    extract_width : int, str
        Width of extraction box. Or 'optimize'.
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
    det = utils.get_detector_name(datafiles[0])
    # Get flux and errors to extract.
    for i, file in enumerate(datafiles):
        with utils.open_filetype(file) as datamodel:
            if i == 0:
                cube = datamodel.data
                ecube = datamodel.err
            else:
                cube = np.concatenate([cube, datamodel.data])
                ecube = np.concatenate([ecube, datamodel.err])

    # Get centroid positions.
    x1, y1 = centroids['xpos'].values, centroids['ypos'].values

    # ===== Optimize Aperture Width =====
    if extract_width == 'optimize':
        fancyprint('Optimizing extraction width...')
        # Extract with a variety of widths and find the one that minimizes
        # the white light curve scatter.
        scatter = []
        if det == 'nrs1':
            xstart = 500
        else:
            xstart = 0
        for w in tqdm(range(1, 12)):
            flux = do_box_extraction(cube, ecube, y1, width=w,
                                     progress=False, extract_start=xstart)[0]
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
            plotting.make_soss_width_plot(scatter, ii, outfile=outfile,
                                          show_plot=show_plot)

    # ===== Extraction ======
    # Do the extraction.
    fancyprint('Performing simple aperture extraction.')
    if det == 'nrs1':
        xstart = 500
    else:
        xstart = 0
    flux, ferr = do_box_extraction(cube, ecube, y1, width=extract_width,
                                   extract_start=xstart)

    # Get default 2D wavelength solution.
    with datamodels.open(datafiles[0]) as d:
        wave2d = d.wavelength
    # Get 1D wavelengths at the locations of the trace centroids.
    wave1d = np.ones(cube.shape[2]) * np.nan
    for x, y in zip(x1, y1):
        wave1d[int(x)] = wave2d[int(y), int(x)]

    wave = np.repeat(wave1d[np.newaxis, :], np.shape(cube)[0], axis=0)

    return wave, flux, ferr, extract_width


def box_extract_soss(datafiles, centroids, soss_width, do_plot=False,
                     show_plot=False, save_results=True, output_dir='./'):
    """Perform a simple box aperture extraction on SOSS orders 1 and 2.

    Parameters
    ----------
    datafiles : array-like[str], array-like[jwst.RampModel]
        Input datamodels or paths to datamodels for each segment.
    centroids : dict
        Dictionary of centroid positions for all SOSS orders.
    soss_width : int, str
        Width of extraction box. Or 'optimize'.
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
        Optimized aperture width.
    """

    datafiles = np.atleast_1d(datafiles)
    # Get flux and errors to extract.
    for i, file in enumerate(datafiles):
        with utils.open_filetype(file) as datamodel:
            if i == 0:
                cube = datamodel.data
                ecube = datamodel.err
            else:
                cube = np.concatenate([cube, datamodel.data])
                ecube = np.concatenate([ecube, datamodel.err])

    # Get centroid positions.
    x1 = centroids['xpos'].values
    y1, y2 = centroids['ypos o1'].values, centroids['ypos o2'].values
    ii = np.where(np.isfinite(y2))
    x2, y2 = x1[ii], y2[ii]

    # ===== Optimize Aperture Width =====
    if soss_width == 'optimize':
        fancyprint('Optimizing extraction width...')
        # Extract order 1 with a variety of widths and find the one that
        # minimizes the white light curve scatter.
        scatter = []
        for w in tqdm(range(10, 61)):
            flux = do_box_extraction(cube, ecube, y1, width=w,
                                     progress=False)[0]
            wlc = np.nansum(flux, axis=1)
            s = np.median(np.abs(0.5*(wlc[0:-2] + wlc[2:]) - wlc[1:-1]))
            scatter.append(s)
        scatter = np.array(scatter)
        # Find the width that minimizes the scatter.
        ii = np.argmin(scatter)
        soss_width = np.linspace(10, 60, 51)[ii]
        fancyprint('Using width of {} pxiels.'.format(int(soss_width)))

        # Do diagnostic plot if requested.
        if do_plot is True:
            if save_results is True:
                outfile = output_dir + 'aperture_optimization.png'
            else:
                outfile = None
            plotting.make_soss_width_plot(scatter, ii, outfile=outfile,
                                          show_plot=show_plot)

    # ===== Extraction ======
    # Do the extraction.
    fancyprint('Performing simple aperture extraction.')
    fancyprint('Extracting Order 1')
    flux_o1, ferr_o1 = do_box_extraction(cube, ecube, y1, width=soss_width)
    fancyprint('Extracting Order 2')
    flux_o2, ferr_o2 = do_box_extraction(cube, ecube, y2, width=soss_width,
                                         extract_end=len(y2))

    # Get default wavelength solution.
    step = calwebb_spec2.extract_1d_step.Extract1dStep()
    wavemap = step.get_reference_file(datafiles[0], 'wavemap')
    # Remove 20 pixel padding that is there for some reason.
    wave_o1 = np.mean(fits.getdata(wavemap, 1)[20:-20, 20:-20], axis=0)
    wave_o2 = np.mean(fits.getdata(wavemap, 2)[20:-20, 20:-20], axis=0)

    return wave_o1, flux_o1, ferr_o1, wave_o2, flux_o2, ferr_o2, soss_width


def do_box_extraction(cube, err, ypos, width, extract_start=0,
                      extract_end=None, progress=True):
    """Do intrapixel aperture extraction.

    Parameters
    ----------
    cube : array-like(float)
        Data cube.
    err : array-like(float)
        Error cube.
    ypos : array-like(float)
        Detector Y-positions to extract.
    width : int
        Full-width of the extraction aperture to use.
    extract_start : int
        Detector X-position at which to start extraction.
    extract_end : int, None
        Detector X-position at which to end extraction.
    progress : bool
        if True, show extraction progress bar.

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

    # Determine the upper and lower edges of the extraction region. Cut at
    # detector edges if necessary.
    edge_up = np.min([ypos + width / 2, np.ones_like(ypos) * dimy], axis=0)
    edge_low = np.max([ypos - width / 2, np.zeros_like(ypos)], axis=0)

    # Loop over all integrations and sum flux within the extraction aperture.
    for i in tqdm(range(nint), disable=not progress):
        for x in range(extract_start, extract_end):
            xx = x - extract_start
            # First sum the whole pixel components within the aperture.
            up_whole = np.floor(edge_up[xx]).astype(int)
            low_whole = np.ceil(edge_low[xx]).astype(int)
            this_flux = np.sum(cube[i, low_whole:up_whole, x])
            this_err = np.sum(err[i, low_whole:up_whole, x]**2)

            # Now incorporate the partial pixels at the upper and lower edges.
            if edge_up[xx] >= (dimy-1) or edge_low[xx] == 0:
                continue
            else:
                up_part = edge_up[xx] % 1
                low_part = 1 - edge_low[xx] % 1
                this_flux += (up_part * cube[i, up_whole, x] +
                              low_part * cube[i, low_whole, x])
                this_err += (up_part * err[i, up_whole, x]**2 +
                             low_part * err[i, low_whole, x]**2)
                f[i, x] = this_flux
                ferr[i, x] = np.sqrt(this_err)

    return f, ferr


def do_ccf(wave, flux, err, mod_flux, nsteps=1000):
    """Perform a cross-correlation analysis between an extracted and model
    stellar spectrum to determine the appropriate wavelength shift between
    the two.

    Parameters
    ----------
    wave : array-like[float]
        Wavelength axis.
    flux : array-like[float]
        Extracted spectrum.
    err : array-like[float]
        Errors on extracted spectrum.
    mod_flux : array-like[float]
        Model spectrum.
    nsteps : int
        Number of wavelength steps to test.

    Returns
    -------
    shift : float
        Wavelength shift between the model and extracted spectrum in microns.
    """

    def highpass_filter(signal, order=3, freq=0.05):
        """High pass filter."""
        b, a = butter(order, freq, btype='high')
        signal_filt = filtfilt(b, a, signal)
        return signal_filt

    ccf = np.zeros(nsteps)
    # Trim edges off of input data to avoid interplation errors.
    wav_a, flux_a, ferr_a = wave[50:-50], flux[50:-50], err[50:-50]
    # Max-value normalize the model spectrum and initialize interpolation.
    mod_norm = mod_flux / np.max(mod_flux)
    f = interp1d(wave, mod_norm, kind='cubic')
    # Max-value normalize and high-pass filter the data.
    data = highpass_filter(flux_a / np.max(flux_a))
    error = ferr_a / np.max(flux_a)

    # Perform the CCF.
    for j, jj in enumerate(np.linspace(-0.01, 0.01, nsteps)):
        # Calculate new wavelength axis.
        new_wave = wav_a + jj
        # Interpolate model onto new axis and high-pass filter.
        model_interp = f(new_wave)
        model_interp = highpass_filter(model_interp)
        # Calculate the CCF at this step.
        ccf[j] = np.nansum(model_interp * data / error ** 2)

    # Determine the peak of the CCF for each integration to get the
    # best-fitting shift.
    maxval = np.argmax(ccf)
    shift = np.linspace(-0.01, 0.01, nsteps)[maxval]

    return shift


def format_nirspec_spectra(datafiles, times, extract_params, target_name,
                           detector, st_teff=None, st_logg=None, st_met=None,
                           throughput=None, output_dir='./',
                           save_results=True):
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
    throughput : str
        Path to JWST spectrace reference file.

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

    # Now cross-correlate with stellar model.
    # If one or more of the stellar parameters are not provided, use the
    # wavelength solution from pastasoss.
    if None in [st_teff, st_logg, st_met]:
        fancyprint('Stellar parameters not provided. '
                   'Using default wavelength solution.', msg_type='WARNING')
    else:
        fancyprint('Refining the wavelength calibration.')
        fancyprint('... is buggy and so wont be run!')
        # # Create a grid of stellar parameters, and download PHOENIX spectra
        # # for each grid point.
        # thisout = output_dir + 'phoenix_models'
        # utils.verify_path(thisout)
        # res = utils.download_stellar_spectra(st_teff, st_logg, st_met,
        #                                      outdir=thisout)
        # wave_file, flux_files = res
        # # Interpolate model grid to correct stellar parameters.
        # mod_flux = utils.interpolate_stellar_model_grid(flux_files, st_teff,
        #                                                 st_logg, st_met)
        # mod_wave = fits.getdata(wave_file)/1e4
        #
        # # Convolve model to lower resolution and interpolate to data
        # # wavelengths.
        # gauss = Gaussian1DKernel(stddev=500)
        # mod_flux = convolve(mod_flux, gauss)
        # mod_flux = np.interp(wave1d, mod_wave, mod_flux)
        # # Add throuput effects to model.
        # thpt = fits.open(throughput)
        # twave = thpt[1].data['WAVELENGTH']
        # thpt = thpt[1].data['THROUGHPUT']
        # thpt = np.interp(wave1d, twave, thpt)
        # mod_flux *= thpt
        #
        # # Cross-correlate extracted spectrum with model to refine wavelength
        # # calibration.
        # x1d_flux = np.nansum(flux, axis=0)
        # x1d_err = np.sqrt(np.nansum(ferr**2, axis=0))
        # wave_shift = do_ccf(wave1d, x1d_flux, x1d_err, mod_flux)
        # fancyprint('Found a wavelength shift of {}um'.format(wave_shift))
        # wave1d += wave_shift

    # Clip remaining 3-sigma outliers.
    flux_clip = utils.sigma_clip_lightcurves(flux, window=11, thresh=3)

    # Pack the lightcurves into the output format.
    # Put 1D extraction parameters in the output file header.
    filename = output_dir + target_name[:-2] + '_' + detector + '_' + \
        extract_params['method'] + '_spectra_fullres.fits'
    header_dict, header_comments = utils.get_default_header()
    header_dict['Target'] = target_name[:-2]
    header_dict['Contents'] = 'Full resolution stellar spectra'
    header_dict['Method'] = extract_params['method']
    header_dict['Width'] = extract_params['extract_width']
    # Calculate the limits of each wavelength bin.
    nint = np.shape(flux_clip)[0]
    wl, wu = utils.get_wavebin_limits(wave1d)
    wl = np.repeat(wl[np.newaxis, :], nint, axis=0)
    wu = np.repeat(wu[np.newaxis, :], nint, axis=0)

    # Pack the stellar spectra and save to file if requested.
    data = [wl, wu, flux_clip, ferr, times]
    names = ['Wave Low', 'Wave Up', 'Flux', 'Flux Err', 'Time']
    units = ['Micron', 'Micron', 'e/s', 'e/s', 'BJD']
    spectra = utils.save_extracted_spectra(filename, data, names, units,
                                           header_dict, header_comments,
                                           save_results=save_results)

    return spectra


def format_soss_spectra(datafiles, times, extract_params, target_name,
                        st_teff=None, st_logg=None, st_met=None,
                        throughput=None, pwcpos=None, output_dir='./',
                        save_results=True, use_pastasoss=False):
    """Unpack the outputs of the 1D extraction and format them into
    lightcurves at the native detector resolution.

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
    throughput : str
        Path to JWST spectrace reference file.
    pwcpos : float
        Filter wheel position. Only necessary is use_pastasoss is True.
    use_pastasoss : bool
        If True, use pastasoss package to predict wavelength solution based on
        pupil wheel position. Note that this will only allow the extraction of
        order 2 from 0.6 - 0.85µm.

    Returns
    -------
    spectra : dict
        1D stellar spectra at the native detector resolution.
    """

    fancyprint('Formatting extracted 1d spectra.')
    # Box extract outputs will just be a tuple of arrays.
    if isinstance(datafiles, tuple):
        wave1d_o1 = datafiles[0]
        flux_o1 = datafiles[1]
        ferr_o1 = datafiles[2]
        wave1d_o2 = datafiles[3]
        flux_o2 = datafiles[4]
        ferr_o2 = datafiles[5]

    # Whereas ATOCA extract outputs are in the atoca extract1dstep format.
    else:
        # Open the datafiles, and pack the wavelength, flux, and flux error
        # information into data cubes.
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
        # Use PASTASOSS to predict wavelength solution from pupil wheel
        # position.
        # Note that PASTASOSS only predicts positions and thus wavelengths for
        # order 2 bluewards of ~0.9µm. Therefore, the whole frame cannot be
        # extracted for order 2. PASTASOSS also does not take into account any
        # TA inaccuracies resulting in the position of the target trace not
        # being in the center of the frame - which will effect the resulting
        # wavelength solution.
        fancyprint('Using PASTASOSS to predict wavelength solution.')
        wave1d_o1 = pastasoss.get_soss_traces(pwcpos=pwcpos, order='1',
                                              interp=True).wavelength
        soln_o2 = pastasoss.get_soss_traces(pwcpos=pwcpos, order='2',
                                            interp=True)
        xpos_o2, wave1d_o2 = soln_o2.x.astype(int), soln_o2.wavelength
        # Trim extracted quantities to match shapes of pastasoss quantities.
        flux_o1 = flux_o1[:, 4:-4]
        ferr_o1 = ferr_o1[:, 4:-4]
        flux_o2 = flux_o2[:, xpos_o2]
        ferr_o2 = ferr_o2[:, xpos_o2]

    # Cross-correlate with stellar model.
    # If one or more of the stellar parameters are not provided, use the
    # existing wavelength solution.
    if None in [st_teff, st_logg, st_met]:
        fancyprint('Stellar parameters not provided. '
                   'Using default wavelength solution.', msg_type='WARNING')
    else:
        fancyprint('Refining the wavelength calibration.')
        # Create a grid of stellar parameters, and download PHOENIX spectra
        # for each grid point.
        thisout = output_dir + 'phoenix_models'
        utils.verify_path(thisout)
        res = utils.download_stellar_spectra(st_teff, st_logg, st_met,
                                             outdir=thisout)
        wave_file, flux_files = res
        # Interpolate model grid to correct stellar parameters.
        # Reverse direction of both arrays since SOSS is extracted red to blue.
        mod_flux = utils.interpolate_stellar_model_grid(flux_files, st_teff,
                                                        st_logg, st_met)
        mod_wave = fits.getdata(wave_file)/1e4

        # Convolve model to lower resolution and interpolate to data
        # wavelengths.
        gauss = Gaussian1DKernel(stddev=500)
        mod_flux = convolve(mod_flux, gauss)
        mod_flux = np.interp(wave1d_o1[::-1], mod_wave, mod_flux)[::-1]
        # Add throuput effects to model.
        thpt = fits.open(throughput)
        twave = thpt[1].data['WAVELENGTH']
        thpt = thpt[1].data['THROUGHPUT']
        thpt = np.interp(wave1d_o1[::-1], twave, thpt)[::-1]
        mod_flux *= thpt

        # Cross-correlate extracted spectrum with model to refine wavelength
        # calibration.
        x1d_flux = np.nansum(flux_o1, axis=0)
        x1d_err = np.sqrt(np.nansum(ferr_o1**2, axis=0))
        wave_shift = do_ccf(wave1d_o1, x1d_flux, x1d_err, mod_flux)
        fancyprint('Found a wavelength shift of {}um'.format(-1*wave_shift))
        wave1d_o1 -= wave_shift
        wave1d_o2 -= wave_shift

    # Clip remaining 5-sigma outliers.
    flux_o1_clip = utils.sigma_clip_lightcurves(flux_o1)
    flux_o2_clip = utils.sigma_clip_lightcurves(flux_o2)

    # Pack the lightcurves into the output format.
    # Put 1D extraction parameters in the output file header.
    filename = output_dir + target_name[:-2] + '_' + extract_params['method'] \
        + '_spectra_fullres.fits'
    header_dict, header_comments = utils.get_default_header()
    header_dict['Target'] = target_name[:-2]
    header_dict['Contents'] = 'Full resolution stellar spectra'
    header_dict['Method'] = extract_params['method']
    header_dict['Width'] = extract_params['extract_width']
    # Calculate the limits of each wavelength bin.
    nint = np.shape(flux_o1_clip)[0]
    wl1, wu1 = utils.get_wavebin_limits(wave1d_o1)
    wl2, wu2 = utils.get_wavebin_limits(wave1d_o2)
    wl1 = np.repeat(wl1[np.newaxis, :], nint, axis=0)
    wu1 = np.repeat(wu1[np.newaxis, :], nint, axis=0)
    wl2 = np.repeat(wl2[np.newaxis, :], nint, axis=0)
    wu2 = np.repeat(wu2[np.newaxis, :], nint, axis=0)

    # Pack the stellar spectra and save to file if requested.
    data = [wl1, wu1, flux_o1_clip, ferr_o1,
            wl2, wu2, flux_o2_clip, ferr_o2, times]
    names = ['Wave Low O1', 'Wave Up O1', 'Flux O1', 'Flux Err O1',
             'Wave Low O2', 'Wave Up O2', 'Flux O2', 'Flux Err O2', 'Time']
    units = ['Micron', 'Micron', 'DN/s', 'DN/s',
             'Micron', 'Micron', 'DN/s', 'DN/s', 'BJD']
    spectra = utils.save_extracted_spectra(filename, data, names, units,
                                           header_dict, header_comments,
                                           save_results=save_results)

    return spectra


def get_soss_estimate(atoca_spectra, output_dir):
    """Convert the AtocaSpectra output of ATOCA into the format expected for a
    soss_estimate.

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


def run_stage3(results, save_results=True, root_dir='./', force_redo=False,
               extract_method='box', soss_specprofile=None, centroids=None,
               extract_width=40, st_teff=None, st_logg=None, st_met=None,
               planet_letter='b', output_tag='', do_plot=False,
               show_plot=False, **kwargs):
    """Run the exoTEDRF Stage 3 pipeline: 1D spectral extraction, using
    a combination of the official STScI DMS and custom steps.

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
        Either 'box' or 'atoca'. Runs the applicable 1D extraction routine.
    soss_specprofile : str, None
        Specprofile reference file; only neceessary for ATOCA extractions.
    centroids : str, None
        Path to file containing trace positions for each order.
    extract_width : int
        Width around the trace centroids, in pixels, for the 1D extraction.
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
        Only necessary if do_plot is True. Show the diagnostic plots in
        addition to/instead of saving to file.

    Returns
    -------
    specra : dict
        1D stellar spectra for each wavelength bin at the native detector
        resolution.
    """

    # ============== DMS Stage 3 ==============
    # 1D spectral extraction.
    fancyprint('**Starting exoTEDRF Stage 3**')
    fancyprint('1D spectral extraction...')

    if output_tag != '':
        output_tag = '_' + output_tag
    # Create output directories and define output paths.
    utils.verify_path(root_dir + 'pipeline_outputs_directory' + output_tag)
    utils.verify_path(root_dir + 'pipeline_outputs_directory' + output_tag + '/Stage3')
    outdir = root_dir + 'pipeline_outputs_directory' + output_tag + '/Stage3/'

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
    step = Extract1DStep(results, extract_method=extract_method,
                         st_teff=st_teff, st_logg=st_logg, st_met=st_met,
                         planet_letter=planet_letter,  output_dir=outdir)
    spectra = step.run(extract_width=extract_width,
                       soss_specprofile=soss_specprofile,
                       centroids=centroids, save_results=save_results,
                       force_redo=force_redo, do_plot=do_plot,
                       show_plot=show_plot, **step_kwargs)

    return spectra
