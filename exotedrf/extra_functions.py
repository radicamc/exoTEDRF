#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 15:08 2023

@author: MCR

Extra functions for use alongside the main pipeline.
"""

from astropy.io import fits
from astroquery.mast import Observations
from itertools import groupby
import numpy as np
import os
from scipy.signal import medfilt
import shutil

from exotedrf import utils
from exotedrf.utils import fancyprint


def download_observations(proposal_id, instrument_name=None, objectname=None,
                          filters=None, visit_nos=None):
    """Directly download uncal files associated with a given observation from
    the MAST archive.

    Parameters
    ----------
    proposal_id : str
        ID for proposal with which the observations are associated.
    instrument_name : str
        Name of instrument data to retrieve. NIRISS/SOSS, NIRSPEC/SLIT
        currently supported. (optional).
    objectname : str
        Name of observational target. (optional).
    filters : str
        Instrument filters to retrieve.
    visit_nos : int, array-like(int)
        For targets with multiple visits, which visits to retrieve.
    """

    # Make sure something is specified to download.
    if proposal_id is None and objectname is None:
        msg = 'At least one of proposal_id or objectname must be specified.'
        raise ValueError(msg)

    # Get observations from MAST.
    obs_table = Observations.query_criteria(proposal_id=proposal_id,
                                            instrument_name=instrument_name,
                                            filters=filters,
                                            objectname=objectname,
                                            radius='.2 deg')
    all_products = Observations.get_product_list(obs_table)

    products = Observations.filter_products(all_products,
                                            dataproduct_type='image',
                                            extension='_uncal.fits',
                                            productType='SCIENCE')

    # If specific visits are specified, retrieve only those files. If not,
    # retrieve all relevant files.
    if visit_nos is not None:
        # Group files by observation number.
        nums = []
        for file in products['productFilename'].data.data:
            nums.append(file.split('_')[0])
        nums = np.array(nums)
        exps = [list(j) for i, j in groupby(nums)]
        fancyprint('Identified {} observations.'.format(len(exps)))

        # Select only files from relevant visits.
        visit_nos = np.atleast_1d(visit_nos)
        if np.max(visit_nos) > len(exps):
            msg = 'You are trying to retrieve visit {0}, but only {1} ' \
                  'visits exist.'.format(np.max(visit_nos), len(exps))
            raise ValueError(msg)
        fancyprint('Retrieving visit(s) {}.'.format(visit_nos))
        for visit in visit_nos:
            ii = np.where(nums == exps[visit - 1][0])[0]
            this_visit = products[ii]
            # Download the relevant files.
            Observations.download_products(this_visit)
    else:
        # Download the relevant files.
        Observations.download_products(products)

    # Unpack auto-generated directories into something better.
    os.mkdir('DMS_uncal')
    for root, _, files in os.walk('mastDownload/', topdown=False):
        for name in files:
            file = os.path.join(root, name)
            shutil.move(file, 'DMS_uncal/.')
    shutil.rmtree('mastDownload/')

    return


def get_throughput_from_photom_file(photom_path):
    """Calculate the throughput based on the photom reference file.
    Function from Loïc, and is apparently the proper way to get the
    throughput? Gives different results from contents of spectra reference
    file.

    Parameters
    ----------
    photom_path : str
        Path to photom reference file

    Returns
    -------
    w1 : np.array(float)
        Order 1 wavelength axis.
    w2 : np.array(float)
        Order 2 wavelength axis.
    thpt1 : np.array(float)
        Order 1 throughput values
    thpt2 : np.array(float)
        Order 2 throughput values
    """

    # From the photom ref file, get conversion factors + wavelength/pixel grid.
    photom = fits.open(photom_path)
    w1 = photom[1].data['wavelength'][0]
    ii = np.where((photom[1].data['wavelength'][0] >= 0.84649785) &
                  (photom[1].data['wavelength'][0] <= 2.83358154))
    w1 = w1[ii]
    scale1 = photom[1].data['relresponse'][0][ii]

    w2 = photom[1].data['wavelength'][1]
    ii = np.where((photom[1].data['wavelength'][1] >= 0.49996997) &
                  (photom[1].data['wavelength'][1] <= 1.40884607))
    w2 = w2[ii]
    scale2 = photom[1].data['relresponse'][1][ii]

    # Calculate throughput from conversion factor.
    thpt1 = 1 / (scale1 * 3e8 / w1 ** 2)
    thpt2 = 1 / (scale2 * 3e8 / w2 ** 2)

    return w1, w2, thpt1, thpt2


def make_smoothed_2d_lightcurve(spec, baseline_ints, nint, dimx, filename,
                                order=1, tscale=3, wscale=9):
    """Smooth extracted 2D SOSS light curves on specified time and wavelength
    scales to use as input for chromatic 1/f correction.

    Parameters
    ----------
    spec : array-like(float)
        Extracted 2D light curves.
    baseline_ints : int, array-like(int)
        Integrations or ingress and/or egress.
    nint : int
        Number of integration in exposure.
    dimx : int
        Number of wavelength bins in exposure.
    filename : str
        File to which to save results.
    order : int
        SOSS diffraction order being considered.
    tscale : int
        Timescale, in integrations, on which to smooth. Must be odd.
    wscale : int
        Timescale, in wavelength bins, on which to smooth. Must be odd.
    """

    # Normalize light curves.
    baseline_ints = utils.format_out_frames(baseline_ints)
    spec /= np.nanmedian(spec[baseline_ints], axis=0)

    # Smooth on desired scale.
    spec_smoothed = medfilt(spec, (tscale, wscale))

    # Put back on full size wavelength axis.
    ref_file = np.ones((nint, dimx))
    if order == 1:
        ref_file = spec_smoothed
    else:
        end = 1206 + spec.shape[1]
        ref_file[:, 1206:end] = spec_smoothed

    # Save file.
    suffix = 'lcestimate_2d_o{}.npy'.format(order)
    np.save(filename + suffix, ref_file)
