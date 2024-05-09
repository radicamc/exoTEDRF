#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 08:51 2022

@author: MCR

Miscellaneous pipeline tools.
"""

from astropy.io import fits
import bottleneck as bn
from datetime import datetime
import glob
import numpy as np
import os
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import medfilt
import warnings
import yaml

import applesoss.edgetrigger_centroids as apl
from jwst import datamodels


def do_replacement(frame, badpix_map, dq=None, box_size=5):
    """Replace flagged pixels with the median of a surrounding box.

    Parameters
    ----------
    frame : array-like[float]
        Data frame.
    badpix_map : array-like[bool]
        Map of pixels to be replaced.
    dq : array-like[int]
        Data quality flags.
    box_size : int
        Size of box to consider.

    Returns
    -------
    frame_out : array-like[float]
        Input frame wth pixels interpolated.
    dq_out : array-like[int]
        Input dq map with interpolated pixels set to zero.
    """

    dimy, dimx = np.shape(frame)
    frame_out = np.copy(frame)
    # Get the data quality flags.
    if dq is not None:
        dq_out = np.copy(dq)
    else:
        dq_out = np.zeros_like(frame)

    # Loop over all flagged pixels.
    for i in range(dimx):
        for j in range(dimy):
            if badpix_map[j, i] == 0:
                continue
            # If pixel is flagged, replace it with the box median.
            else:
                med = get_interp_box(frame, box_size, i, j, dimx)[0]
                frame_out[j, i] = med
                # Set dq flag of inerpolated pixel to zero (use the pixel).
                dq_out[j, i] = 0

    return frame_out, dq_out


def download_stellar_spectra(st_teff, st_logg, st_met, outdir):
    """Download a grid of PHOENIX model stellar spectra.

    Parameters
    ----------
    st_teff : float
        Stellar effective temperature.
    st_logg : float
        Stellar log surface gravity.
    st_met : float
        Stellar metallicity as [Fe/H].
    outdir : str
        Output directory.

    Returns
    -------
    wfile : str
        Path to wavelength file.
    ffiles : list[str]
        Path to model stellar spectrum files.
    """

    fpath = 'ftp://phoenix.astro.physik.uni-goettingen.de/'

    # Get wavelength grid.
    wave_file = 'WAVE_PHOENIX-ACES-AGSS-COND-2011.fits'
    wfile = '{}/{}'.format(outdir, wave_file)
    if not os.path.exists(wfile):
        fancyprint('Downloading file {}.'.format(wave_file))
        cmd = 'wget -q -O {0} {1}HiResFITS/{2}'.format(wfile, fpath, wave_file)
        os.system(cmd)
    else:
        fancyprint('File {} already downloaded.'.format(wfile))

    # Get stellar spectrum grid points.
    teffs, loggs, mets = get_stellar_param_grid(st_teff, st_logg, st_met)

    # Construct filenames to retrieve
    ffiles = []
    for teff in teffs:
        for logg in loggs:
            for met in mets:
                if met > 0:
                    basename = 'lte0{0}-{1}0+{2}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'
                elif met == 0:
                    basename = 'lte0{0}-{1}0-{2}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'
                else:
                    basename = 'lte0{0}-{1}0{2}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'
                thisfile = basename.format(teff, logg, met)

                ffile = '{}/{}'.format(outdir, thisfile)
                ffiles.append(ffile)
                if not os.path.exists(ffile):
                    fancyprint('Downloading file {}.'.format(thisfile))
                    if met > 0:
                        cmd = 'wget -q -O {0} {1}HiResFITS/PHOENIX-ACES-AGSS-COND-2011/Z+{2}/{3}'.format(ffile, fpath, met, thisfile)
                    elif met == 0:
                        cmd = 'wget -q -O {0} {1}HiResFITS/PHOENIX-ACES-AGSS-COND-2011/Z-{2}/{3}'.format(ffile, fpath, met, thisfile)
                    else:
                        cmd = 'wget -q -O {0} {1}HiResFITS/PHOENIX-ACES-AGSS-COND-2011/Z{2}/{3}'.format(ffile, fpath, met, thisfile)
                    os.system(cmd)
                else:
                    fancyprint('File {} already downloaded.'.format(ffile))

    return wfile, ffiles


def fancyprint(message, msg_type='INFO'):
    """Fancy printing statement mimicking logging. Basically a hack to get
    around complications with the STScI pipeline logging.

    Parameters
    ----------
    message : str
        Message to print.
    msg_type : str
        Type of message. Mirrors the jwst pipeline logging.
    """

    time = datetime.now().isoformat(sep=' ', timespec='milliseconds')
    print('{} - exoTEDRF - {} - {}'.format(time, msg_type, message))


def format_out_frames(out_frames):
    """Create a mask of baseline flux frames for lightcurve normalization.

    Parameters
    ----------
    out_frames : array-like[int], int
        Integration numbers of ingress and/or egress.

    Returns
    -------
    baseline_ints : array-like[int]
        Array of baseline frames.
    """

    out_frames = np.atleast_1d(out_frames)
    # For baseline just before ingress or after ingress.
    if len(out_frames) == 1:
        if out_frames[0] > 0:
            baseline_ints = np.arange(out_frames[0])
        else:
            out_frames = np.abs(out_frames)
            baseline_ints = np.arange(out_frames[0]) - out_frames[0]
    # If baseline at both ingress and egress to be used.
    elif len(out_frames) == 2:
        out_frames = np.abs(out_frames)
        baseline_ints = np.concatenate([np.arange(out_frames[0]),
                                        np.arange(out_frames[1]) - out_frames[1]])
    else:
        raise ValueError('out_frames must have length 1 or 2.')

    return baseline_ints


def get_default_header():
    """Format the default header for the lightcurve file.

    Returns
    -------
    header_dict : dict
        Header keyword dictionary.
    header_commets : dict
        Header comment dictionary.
    """

    # Header with important keywords.
    header_dict = {'Target': None,
                   'Inst': 'NIRISS/SOSS',
                   'Date': datetime.utcnow().replace(microsecond=0).isoformat(),
                   'Pipeline': 'exoTEDRF',
                   'Author': 'MCR',
                   'Contents': None,
                   'Method': 'Box Extraction',
                   'Width': 25}
    # Explanations of keywords.
    header_comments = {'Target': 'Name of the target',
                       'Inst': 'Instrument used to acquire the data',
                       'Date': 'UTC date file created',
                       'Pipeline': 'Pipeline that produced this file',
                       'Author': 'File author',
                       'Contents': 'Description of file contents',
                       'Method': 'Type of 1D extraction',
                       'Width': 'Box width'}

    return header_dict, header_comments


def get_detector_name(datafile):
    """Get name of detector.

    Parameters
    ----------
    datafile : str, datamodel
        Path to datafile or datafile itself.

    Returns
    -------
    detector : str
        Name of detector.
    """

    with datamodels.open(datafile) as d:
        detector = d.meta.instrument.detector.lower()
    return detector


def get_dq_flag_metrics(dq_map, flags):
    """Take a data quality map and extract a map of pixels which are flagged
    for a specific reason. A list of data quality flags can be found here:
    https://jwst-reffiles.stsci.edu/source/data_quality.html.

    Parameters
    ----------
    dq_map : array-like(float)
        Map of data quality flags.
    flags : list[str], str
        Flag types to find.

    Returns
    -------
    flagged : np.array(bool)
        Boolean map where True values have the applicable flag.
    """

    flags = np.atleast_1d(flags)
    dq_map = np.atleast_3d(dq_map)
    dimy, dimx, nint = np.shape(dq_map)

    # From here: https://jwst-reffiles.stsci.edu/source/data_quality.html
    flags_dict = {'DO_NOT_USE': 0, 'SATURATED': 1, 'JUMP_DET': 2,
                  'DROPOUT': 3, 'OUTLIER': 4, 'PERSISTENCE': 5,
                  'AD_FLOOR': 6, 'RESERVED': 7, 'UNRELIABLE_ERROR': 8,
                  'NON_SCIENCE': 9, 'DEAD': 10, 'HOT': 11, 'WARM': 12,
                  'LOW_QE': 13, 'RC': 14, 'TELEGRAPH': 15, 'NONLINEAR': 16,
                  'BAD_REF_PIXEL': 17, 'NO_FLAT_FIELD': 18,
                  'NO_GAIN_VALUE': 19,
                  'NO_LIN_CORR': 20, 'NO_SAT_CHECK': 21, 'UNRELIABLE_BIAS': 22,
                  'UNRELIABLE_DARK': 23, 'UNRELIABLE_SLOPE': 24,
                  'UNRELIABLE_FLAT': 25, 'OPEN': 26, 'ADJ_OPEN': 27,
                  'UNRELIABLE_RESET': 28, 'MSA_FAILED_OPEN': 29,
                  'OTHER_BAD_PIXEL': 30, 'REFERENCE_PIXEL': 31}

    flagged = np.zeros_like(dq_map).astype(bool)
    # Get bit corresponding to the desired flags.
    flag_bits = []
    for flag in flags:
        flag_bits.append(flags_dict[flag])

    # Find pixels flagged for the selected reasons.
    for i in range(nint):
        for x in range(dimx):
            for y in range(dimy):
                val = np.binary_repr(dq_map[y, x, i], width=32)[::-1]
                for bit in flag_bits:
                    if val[bit] == '1':
                        flagged[y, x, i] = True
    if nint == 1:
        flagged = flagged[:, :, 0]

    return flagged


def get_filename_root(datafiles):
    """Get the file name roots for each segment. Assumes that file names
    follow the default jwst pipeline structure and are in correct segment
    order.

    Parameters
    ----------
    datafiles : array-like[str], array-like[jwst.datamodel]
        Datamodels, or paths to datamodels for each segment.

    Returns
    -------
    fileroots : list[str]
        List of file name roots.
    """

    fileroots = []
    # Open the datamodel.
    if isinstance(datafiles[0], str):
        with datamodels.open(datafiles[0]) as data:
            filename = data.meta.filename  # Get file name.
            seg_start = data.meta.exposure.segment_number  # Starting segment.
    else:
        try:
            filename = datafiles[0].meta.filename
            seg_start = datafiles[0].meta.exposure.segment_number
        except AttributeError:
            msg = 'Unexpected type {}'.format(type(datafiles[0]))
            raise ValueError(msg)

    # Get the last part of the path, and split file name into chunks.
    filename_split = filename.split('/')[-1].split('_')
    fileroot = ''
    # Get the filename before the step info and save.
    for chunk in filename_split[:-1]:
        fileroot += chunk + '_'
    fileroots.append(fileroot)

    # Now assuming everything is in chronological order, just increment the
    # segment number.
    split = fileroot.split('seg')
    for segment in range(seg_start+1, seg_start+len(datafiles)):
        if segment < 10:
            seg_no = 'seg00{}'.format(segment)
        elif 10 <= segment <= 99:
            seg_no = 'seg0{}'.format(segment)
        else:
            seg_no = 'seg{}'.format(segment)
        thisroot = split[0] + seg_no + split[1][3:]
        fileroots.append(thisroot)

    return fileroots


def get_filename_root_noseg(fileroots):
    """Get the file name root for a SOSS TSO with no segment information.

    Parameters
    ----------
    fileroots : array-like[str]
        File root names for each segment.

    Returns
    -------
    fileroot_noseg : str
        File name root with no segment information.
    """

    # Get total file root, with no segment info.
    working_name = fileroots[0]
    if 'seg' in working_name:
        parts = working_name.split('seg')
        part1, part2 = parts[0][:-1], parts[1][3:]
        fileroot_noseg = part1 + part2
    else:
        fileroot_noseg = fileroots[0]

    return fileroot_noseg


def get_instrument_name(datafile):
    """Get name of instrument.

    Parameters
    ----------
    datafile : str, datamodel
        Path to datafile or datafile itself.

    Returns
    -------
    instrument : str
        Name of instrument.
    """

    with datamodels.open(datafile) as d:
        instrument = d.meta.instrument.name
    return instrument


def get_interp_box(data, box_size, i, j, dimx):
    """Get median and standard deviation of a box centered on a specified
    pixel.

    Parameters
    ----------
    data : array-like[float]
        Data frame.
    box_size : int
        Size of box to consider.
    i : int
        X pixel.
    j : int
        Y pixel.
    dimx : int
        Size of x dimension.

    Returns
    -------
    box_properties : array-like
        Median and standard deviation of pixels in the box.
    """

    # Get the box limits.
    low_x = np.max([i - box_size, 0])
    up_x = np.min([i + box_size, dimx - 1])

    # Calculate median and std deviation of box - excluding central pixel.
    box = np.concatenate([data[j, low_x:i], data[j, (i+1):up_x]])
    median = np.nanmedian(box)
    stddev = np.sqrt(outlier_resistant_variance(box))

    # Pack into array.
    box_properties = np.array([median, stddev])

    return box_properties


def get_stellar_param_grid(st_teff, st_logg, st_met):
    """Given a set of stellar parameters, determine the neighbouring grid
    points based on the PHOENIX grid steps.

    Parameters
    ----------
    st_teff : float
        Stellar effective temperature.
    st_logg : float
        Stellar log surface gravity.
    st_met : float
        Stellar metallicity as [Fe/H].

    Returns
    -------
    teffs : list[float]
        Effective temperature grid bounds.
    loggs : list[float]
        Surface gravity grid bounds.
    mets : list[float]
        Metallicity grid bounds.
    """

    # Determine lower and upper teff steps (step size of 100K).
    teff_lw = int(np.floor(st_teff / 100) * 100)
    teff_up = int(np.ceil(st_teff / 100) * 100)
    if teff_lw == teff_up:
        teffs = [teff_lw]
    else:
        teffs = [teff_lw, teff_up]

    # Determine lower and upper logg step (step size of 0.5).
    logg_lw = np.floor(st_logg / 0.5) * 0.5
    logg_up = np.ceil(st_logg / 0.5) * 0.5
    if logg_lw == logg_up:
        loggs = [logg_lw]
    else:
        loggs = [logg_lw, logg_up]

    # Determine lower and upper metallicity steps (step size of 1).
    met_lw, met_up = np.floor(st_met), np.ceil(st_met)
    # Hack to stop met_up being -0.0 if -1<st_met<0.
    if -1 < st_met < 0:
        met_up = 0.0
    if met_lw == met_up:
        mets = [met_lw]
    else:
        mets = [met_lw, met_up]

    return teffs, loggs, mets


def get_centroids_nirspec(deepframe, xstart=0, xend=None, save_results=True,
                          save_filename=''):
    """Get the NIRSpec trace centroids via the edgetrigger method.

    Parameters
    ----------
    deepframe : array-like[float]
        Median stack.
    xstart : int
        Starting x-pixel position on the frame.
    xend : int, None
        Ending x-pixel position on the frame.
    save_results : bool
        If True, save results to file.
    save_filename : str
        Filename of save file.

    Returns
    -------
    cens : array-like[float]
        X and Y centroids.
    """

    dimy, dimx = np.shape(deepframe)
    if xend is None:
        xend = dimx
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        cens = apl.get_centroids_edgetrigger(deepframe[:, xstart:xend],
                                             mode='mean', poly_order=2,
                                             halfwidth=3)

    x1, y1 = cens[0]+xstart, cens[1]
    ii = np.where((x1 >= xstart) & (x1 <= xend - 1))
    # Interpolate onto native pixel grid
    xx1 = np.linspace(xstart, xend-1, (xend-1)-xstart+1)
    yy1 = np.interp(xx1, x1[ii], y1[ii])

    if save_results is True:
        centroids_dict = {'xpos': xx1, 'ypos': yy1}
        df = pd.DataFrame(data=centroids_dict)
        if save_filename[-1] != '_':
            save_filename += '_'
        outfile_name = save_filename + 'centroids.csv'
        outfile = open(outfile_name, 'w')
        outfile.write('# File Contents: Edgetrigger trace centroids\n')
        outfile.write('# File Creation Date: {}\n'.format(
            datetime.utcnow().replace(microsecond=0).isoformat()))
        outfile.write('# File Author: MCR\n')
        df.to_csv(outfile, index=False)
        outfile.close()
        fancyprint('Centroids saved to {}'.format(outfile_name))

    cens = np.array([xx1, yy1])

    return cens


def get_centroids_soss(deepframe, tracetable, subarray, save_results=True,
                       save_filename=''):
    """Get the SOSS trace centroids for all three orders via the edgetrigger
    method.

    Parameters
    ----------
    deepframe : array-like[float]
        Median stack.
    tracetable : str
        Path to SpecTrace reference file.
    subarray : str
        Subarray identifier.
    save_results : bool
        If True, save results to file.
    save_filename : str
        Filename of save file.

    Returns
    -------
    cen_o1 : array-like[float]
        Order 1 X and Y centroids.
    cen_o2 : array-like[float]
        Order 2 X and Y centroids.
    cen_o3 : array-like[float]
        Order 3 X and Y centroids.
    """

    dimy, dimx = np.shape(deepframe)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        cens = apl.get_soss_centroids(deepframe, tracetable,
                                      subarray=subarray)

    x1, y1 = cens['order 1']['X centroid'], cens['order 1']['Y centroid']
    ii = np.where((x1 >= 0) & (y1 <= dimx - 1))
    # Interpolate onto native pixel grid
    xx1 = np.arange(dimx)
    yy1 = np.interp(xx1, x1[ii], y1[ii])

    if subarray != 'SUBSTRIP96':
        x2, y2 = cens['order 2']['X centroid'], cens['order 2']['Y centroid']
        x3, y3 = cens['order 3']['X centroid'], cens['order 3']['Y centroid']
        ii2 = np.where((x2 >= 0) & (x2 <= dimx - 1) & (y2 <= dimy - 1))
        ii3 = np.where((x3 >= 0) & (x3 <= dimx - 1) & (y3 <= dimy - 1))
        # Interpolate onto native pixel grid
        xx2 = np.arange(np.max(np.floor(x2[ii2]).astype(int)))
        yy2 = np.interp(xx2, x2[ii2], y2[ii2])
        xx3 = np.arange(np.max(np.floor(x3[ii3]).astype(int)))
        yy3 = np.interp(xx3, x3[ii3], y3[ii3])
    else:
        xx2, yy2 = xx1, np.ones_like(xx1) * np.nan
        xx3, yy3 = xx1, np.ones_like(xx1) * np.nan

    if save_results is True:
        yyy2 = np.ones_like(xx1) * np.nan
        yyy2[:len(yy2)] = yy2
        yyy3 = np.ones_like(xx1) * np.nan
        yyy3[:len(yy3)] = yy3

        centroids_dict = {'xpos': xx1, 'ypos o1': yy1, 'ypos o2': yyy2,
                          'ypos o3': yyy3}
        df = pd.DataFrame(data=centroids_dict)
        if save_filename[-1] != '_':
            save_filename += '_'
        outfile_name = save_filename + 'centroids.csv'
        outfile = open(outfile_name, 'w')
        outfile.write('# File Contents: Edgetrigger trace centroids\n')
        outfile.write('# File Creation Date: {}\n'.format(
            datetime.utcnow().replace(microsecond=0).isoformat()))
        outfile.write('# File Author: MCR\n')
        df.to_csv(outfile, index=False)
        outfile.close()
        fancyprint('Centroids saved to {}'.format(outfile_name))

    cen_o1 = np.array([xx1, yy1])
    cen_o2 = np.array([xx2, yy2])
    cen_o3 = np.array([xx3, yy3])

    return cen_o1, cen_o2, cen_o3


def get_wavebin_limits(wave):
    """Determine the upper and lower limits of wavelength bins centered on a
    given wavelength axis.

    Parameters
    ----------
    wave : array-like[float]
        Wavelengh array.

    Returns
    -------
    bin_low : array-like[float]
        Lower edge of wavelength bin.
    bin_up : array-like[float]
        Upper edge of wavelength bin.
    """

    # Shift wavelength array by one element forward and backwards, and create
    # 2D stack where each wavelength is sandwiched between its upper or lower
    # neighbour respectively.
    up = np.concatenate([wave[:, None], np.roll(wave, 1)[:, None]], axis=1)
    low = np.concatenate([wave[:, None], np.roll(wave, -1)[:, None]], axis=1)

    # Take the mean in the vertical direction to get the midpoint between the
    # two wavelengths. Use this as the bin limits.
    bin_low = (np.mean(low, axis=1))[:-1]
    bin_low = np.append(bin_low, 2*bin_low[-1] - bin_low[-2])
    bin_up = (np.mean(up, axis=1))[1:]
    bin_up = np.insert(bin_up, 0, 2*bin_up[0] - bin_up[1])

    return bin_low, bin_up


def interpolate_stellar_model_grid(model_files, st_teff, st_logg, st_met):
    """Given a grid of stellar spectrum files, interpolate the model spectra
    to a set of stellar parameters.

    Parameters
    ----------
    model_files : list[str]
        List of paths to stellar spectra at grid points.
    st_teff : float
        Stellar effective temperature.
    st_logg : float
        Stellar log surface gravity.
    st_met : float
        Stellar metallicity as [Fe/H].

    Returns
    -------
    model_interp : array-like(float)
        Model stellar spectrum interpolated to the input paramaters.
    """

    # Get stellar spectrum grid points.
    teffs, loggs, mets = get_stellar_param_grid(st_teff, st_logg, st_met)
    pts = (np.array(teffs), np.array(loggs), np.array(mets))

    # Read in models.
    specs = []
    for model in model_files:
        specs.append(fits.getdata(model))

    # Create stellar model grid
    vals = np.zeros((len(teffs), len(loggs), len(mets), len(specs[0])))
    tot_i = 0
    for i in range(pts[0].shape[0]):
        for j in range(pts[1].shape[0]):
            for k in range(pts[2].shape[0]):
                vals[i, j, k] = specs[tot_i]
                tot_i += 1

    # Interpolate grid
    grid = RegularGridInterpolator(pts, vals)
    planet = [st_teff, st_logg, st_met]
    model_interp = grid(planet)[0]

    return model_interp


def line_mle(x, y, e):
    """Analytical solution for Chi^2 of fitting a straight line to data.
    All inputs are assumed to be 3D (ints, dimy, dimx).

    Parameters
    ----------
    x : array-like[float]
        X-data. Median stack for 1/f correction.
    y : array-like[float]
        Y-data. Data frames for 1/f correction.
    e : array-like[float]
        Errors.

    Returns
    -------
    m_e : np.array(float)
        "Slope" values for even numbered columns.
    b_e : np.array(float)
        "Intercept" values for even numbered columns.
    m_o : np.array(float)
        "Slope" values for odd numbered columns.
    b_o : np.array(float)
        "Intercept" values for odd numbered columns.
    """

    assert np.shape(x) == np.shape(y) == np.shape(e)
    # Following "Numerical recipes in C. The art of scientific computing"
    # Press, William H. (1989)
    # pdf: https://www.grad.hr/nastava/gs/prg/NumericalRecipesinC.pdf S15.2
    sx_e = np.nansum(x[:, ::2] / e[:, ::2]**2, axis=1)
    sxx_e = np.nansum((x[:, ::2] / e[:, ::2])**2, axis=1)
    sy_e = np.nansum(y[:, ::2] / e[:, ::2]**2, axis=1)
    sxy_e = np.nansum(x[:, ::2] * y[:, ::2] / e[:, ::2]**2, axis=1)
    s_e = np.nansum(1 / e[:, ::2]**2, axis=1)

    m_e = (s_e * sxy_e - sx_e * sy_e) / (s_e * sxx_e - sx_e**2)
    b_e = (sy_e - m_e * sx_e) / s_e

    sx_o = np.nansum(x[:, 1::2] / e[:, 1::2]**2, axis=1)
    sxx_o = np.nansum((x[:, 1::2] / e[:, 1::2])**2, axis=1)
    sy_o = np.nansum(y[:, 1::2] / e[:, 1::2]**2, axis=1)
    sxy_o = np.nansum(x[:, 1::2] * y[:, 1::2] / e[:, 1::2]**2, axis=1)
    s_o = np.nansum(1 / e[:, 1::2]**2, axis=1)

    m_o = (s_o * sxy_o - sx_o * sy_o) / (s_o * sxx_o - sx_o**2)
    b_o = (sy_o - m_o * sx_o) / s_o

    return m_e, b_e, m_o, b_o


def make_deepstack(cube):
    """Make deep stack of a TSO.

    Parameters
    ----------
    cube : array-like[float]
        Stack of all integrations in a TSO

    Returns
    -------
    deepstack : array-like[float]
       Median of the input cube along the integration axis.
    """

    # Take median of input cube along the integration axis.
    deepstack = bn.nanmedian(cube, axis=0)

    return deepstack


def open_filetype(datafile):
    """Open a datamodel whether it is a path, or the datamodel itself.

    Parameters
    ----------
    datafile : str, jwst.datamodel
        Datamodel or path to datamodel.

    Returns
    -------
    data : jwst.datamodel
        Opened datamodel.

    Raises
    ------
    ValueError
        If the filetype passed is not str or jwst.datamodel.
    """

    if isinstance(datafile, str):
        data = datamodels.open(datafile)
    elif isinstance(datafile, (datamodels.CubeModel, datamodels.RampModel,
                               datamodels.MultiSpecModel)):
        data = datafile
    else:
        raise ValueError('Invalid filetype: {}'.format(type(datafile)))

    return data


def outlier_resistant_variance(data):
    """Calculate the varaince of some data along the 0th axis in an outlier
    resistant manner.
    """

    var = (bn.nanmedian(np.abs(data - bn.nanmedian(data, axis=0)), axis=0) / 0.6745)**2
    return var


def parse_config(config_file):
    """Parse a yaml config file.

    Parameters
    ----------
    config_file : str
        Path to config file.

    Returns
    -------
    config : dict
        Dictionary of config parameters.
    """

    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    for key in config.keys():
        if config[key] == 'None':
            config[key] = None

    return config


def save_extracted_spectra(filename, wl1, wu1, f1, e1, wl2, wu2, f2, e2, t,
                           header_dict=None, header_comments=None,
                           save_results=True):
    """Pack stellar spectra into a fits file.

    Parameters
    ----------
    filename : str
        File to which to save results.
    wl1 : array-like[float]
        Order 1 wavelength bin lower limits.
    wu1 : array-like[float]
        Order 1 wavelength bin upper limits.
    f1 : array-like[float]
        Order 1 flux.
    e1 : array-like[float]
        Order 1 flux error.
    wl2 : array-like[float]
        Order 2 wavelength bin lower limits.
    wu2 : array-like[float]
        Order 2 wavelength bin upper limits.
    f2 : array-like[float]
        Order 2 flux.
    e2 : array-like[float]
        Order 2 flux error.
    t : array-like[float]
        Time axis.
    header_dict : dict
        Header keywords and values.
    header_comments : dict
        Header comments.
    save_results : bool
        If True, save results to file.

    Returns
    -------
    param_dict : dict
        Lightcurve parameters packed into a dictionary.
    """

    # Initialize the fits header.
    hdr = fits.Header()
    if header_dict is not None:
        for key in header_dict:
            hdr[key] = header_dict[key]
            if key in header_comments.keys():
                hdr.comments[key] = header_comments[key]
    hdu1 = fits.PrimaryHDU(header=hdr)

    # Pack order 1 values.
    hdr = fits.Header()
    hdr['EXTNAME'] = "Wave Low O1"
    hdr['UNITS'] = "Micron"
    hdu2 = fits.ImageHDU(wl1, header=hdr)
    hdr = fits.Header()
    hdr['EXTNAME'] = "Wave Up O1"
    hdr['UNITS'] = "Micron"
    hdu3 = fits.ImageHDU(wu1, header=hdr)
    hdr = fits.Header()
    hdr['EXTNAME'] = "Flux O1"
    hdr['UNITS'] = "DN/s"
    hdu4 = fits.ImageHDU(f1, header=hdr)
    hdr = fits.Header()
    hdr['EXTNAME'] = "Flux Err O1"
    hdr['UNITS'] = "DN/s"
    hdu5 = fits.ImageHDU(e1, header=hdr)

    # Pack order 2 values.
    hdr = fits.Header()
    hdr['EXTNAME'] = "Wave Low O2"
    hdr['UNITS'] = "Micron"
    hdu6 = fits.ImageHDU(wl2, header=hdr)
    hdr = fits.Header()
    hdr['EXTNAME'] = "Wave Up O2"
    hdr['UNITS'] = "Micron"
    hdu7 = fits.ImageHDU(wu2, header=hdr)
    hdr = fits.Header()
    hdr['EXTNAME'] = "Flux O2"
    hdr['UNITS'] = "DN/s"
    hdu8 = fits.ImageHDU(f2, header=hdr)
    hdr = fits.Header()
    hdr['EXTNAME'] = "Flux Err O2"
    hdr['UNITS'] = "DN/s"
    hdu9 = fits.ImageHDU(e2, header=hdr)

    # Pack time axis.
    hdr = fits.Header()
    hdr['EXTNAME'] = "Time"
    hdr['UNITS'] = "MJD"
    hdu10 = fits.ImageHDU(t, header=hdr)

    if save_results is True:
        fancyprint('Spectra saved to {}'.format(filename))
        hdul = fits.HDUList([hdu1, hdu2, hdu3, hdu4, hdu5, hdu6, hdu7, hdu8,
                             hdu9, hdu10])
        hdul.writeto(filename, overwrite=True)

    param_dict = {'Wave Low O1': wl1, 'Wave Up O1': wu1, 'Flux O1': f1,
                  'Flux Err O1': e1, 'Wave Low O2': wl2, 'Wave Up O2': wu2,
                  'Flux O2': f2, 'Flux Err O2': e2, 'Time': t}

    return param_dict


def save_ld_priors(wave, c1, c2, order, target, m_h, teff, logg, outdir):
    """Write model limb darkening parameters to a file to be used as priors
    for light curve fitting.

    Parameters
    ----------
    wave : array-like[float]
        Wavelength axis.
    c1 : array-like[float]
        c1 parameter for two-parameter limb darkening law.
    c2 : array-like[float]
        c2 parameter for two-parameter limb darkening law.
    order : int
        SOSS order.
    target : str
        Name of the target.
    m_h : float
        Host star metallicity.
    teff : float
        Host star effective temperature.
    logg : float
        Host star gravity.
    outdir : str
        Directory to which to save file.
    """

    # Create dictionary with model LD info.
    dd = {'wave': wave, 'c1': c1, 'c2': c2}
    df = pd.DataFrame(data=dd)
    # Remove old LD file if one exists.
    filename = target+'_order' + str(order) + '_exotic-ld_quadratic.csv'
    if os.path.exists(outdir + filename):
        os.remove(outdir + filename)
    # Add header info.
    f = open(outdir + filename, 'a')
    f.write('# Target: {}\n'.format(target))
    f.write('# Instrument: NIRISS/SOSS\n')
    f.write('# Order: {}\n'.format(order))
    f.write('# Author: {}\n'.format(os.environ.get('USER')))
    f.write('# Date: {}\n'.format(datetime.utcnow().replace(microsecond=0).isoformat()))
    f.write('# Stellar M/H: {}\n'.format(m_h))
    f.write('# Stellar log g: {}\n'.format(logg))
    f.write('# Stellar Teff: {}\n'.format(teff))
    f.write('# Algorithm: ExoTiC-LD\n')
    f.write('# Limb Darkening Model: quadratic\n')
    f.write('# Column wave: Central wavelength of bin (micron)\n')
    f.write('# Column c1: Quadratic Coefficient 1\n')
    f.write('# Column c2: Quadratic Coefficient 2\n')
    f.write('#\n')
    df.to_csv(f, index=False)
    f.close()


def sigma_clip_lightcurves(flux, thresh=5, window=5):
    """Sigma clip outliers in time from final lightcurves.

    Parameters
    ----------
    flux : array-like[float]
        Flux array.
    thresh : int
        Sigma level to be clipped.
    window : int
        Window function to calculate median. Must be odd.

    Returns
    -------
    flux_clipped : array-like[float]
        Flux array with outliers
    """

    flux_clipped = np.copy(flux)
    nints, nwaves = np.shape(flux)
    flux_filt = medfilt(flux, (window, 1))

    # Check along the time axis for outlier pixels.
    std_dev = np.median(np.abs(0.5 * (flux[0:-2] + flux[2:]) - flux[1:-1]), axis=0)
    std_dev = np.where(std_dev == 0, np.inf, std_dev)
    scale = np.abs(flux - flux_filt) / std_dev
    ii = np.where((scale > thresh))
    # Replace outliers.
    flux_clipped[ii] = flux_filt[ii]

    fancyprint('{0} pixels clipped ({1:.3f}%)'.format(len(ii[0]), len(ii[0])/nints/nwaves*100))

    return flux_clipped


def sort_datamodels(datafiles):
    """Sort a list of jwst datamodels or filenames in chronological order by
    segment.

    Parameters
    ----------
    datafiles : array-like(str), array-like(datamodel)
        List of jwst datamodels or filenames.

    Returns
    -------
    files_sorted : np.array
        Inputs sorted in chronological order.
    """

    datafiles = np.atleast_1d(datafiles)

    if isinstance(datafiles[0], str):
        # If filenames are passed, just sort.
        files_sorted = np.sort(datafiles)
    else:
        # If jwst datamodels are passed, first get the filenames, then sort.
        files_unsorted = []
        for file in datafiles:
            files_unsorted.append(file.meta.filename)
        sort_inds = np.argsort(files_unsorted)
        files_sorted = datafiles[sort_inds]

    return files_sorted


def unpack_atoca_spectra(datafile,
                         quantities=('WAVELENGTH', 'FLUX', 'FLUX_ERROR')):
    """Unpack useful quantities from extract1d outputs.

    Parameters
    ----------
    datafile : str, MultiSpecModel
        Extract1d output, or path to the file.
    quantities : tuple(str)
        Quantities to unpack.

    Returns
    -------
    all_spec : dict
        Dictionary containing unpacked quantities for each order.
    """

    multi_spec = open_filetype(datafile)

    # Initialize output dictionary.
    all_spec = {sp_ord: {quantity: [] for quantity in quantities}
                for sp_ord in [1, 2, 3]}
    # Unpack desired quantities into dictionary.
    for spec in multi_spec.spec:
        sp_ord = spec.spectral_order
        for quantity in quantities:
            all_spec[sp_ord][quantity].append(spec.spec_table[quantity])
    for sp_ord in all_spec:
        for key in all_spec[sp_ord]:
            all_spec[sp_ord][key] = np.array(all_spec[sp_ord][key])

    multi_spec.close()

    return all_spec


def unpack_input_dir(indir, filetag='', exposure_type='CLEAR'):
    """Get all segment files of a specified exposure type from an input data
     directory.

    Parameters
    ----------
    indir : str
        Path to input directory.
    filetag : str
        File name extension of files to unpack.
    exposure_type : str
        Either 'CLEAR' or 'F277W'; unpacks the corresponding exposure type.

    Returns
    -------
    segments: array-like[str]
        File names of the requested exposure and file tag in chronological
        order.
    """

    if indir[-1] != '/':
        indir += '/'
    all_files = glob.glob(indir + '*')
    segments = []

    # Check all files in the input directory to see if they match the
    # specified exposure type and file tag.
    for file in all_files:
        try:
            header = fits.getheader(file, 0)
        # Skip directories or non-fits files.
        except(OSError, IsADirectoryError):
            continue
        # Keep files of the correct exposure with the correct tag.
        try:
            if header['FILTER'] == exposure_type:
                if filetag in file:
                    segments.append(file)
            else:
                continue
        except KeyError:
            continue

    # Ensure that segments are packed in chronological order
    if len(segments) > 1:
        segments = np.array(segments)
        segment_numbers = []
        for file in segments:
            seg_no = fits.getheader(file, 0)['EXSEGNUM']
            segment_numbers.append(seg_no)
        correct_order = np.argsort(segment_numbers)
        segments = segments[correct_order]

    return segments


def verify_path(path):
    """Verify that a given directory exists. If not, create it.

    Parameters
    ----------
    path : str
        Path to directory.
    """

    if os.path.exists(path):
        pass
    else:
        # If directory doesn't exist, create it.
        os.mkdir(path)
