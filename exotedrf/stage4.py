#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thurs Jul 21 18:07 2022

@author: MCR

Custom JWST DMS pipeline steps for Stage 4 (lightcurve fitting).
"""

from datetime import datetime
from exotic_ld import StellarLimbDarkening
import juliet
import numpy as np
import os
import pandas as pd
import ray
from tqdm import tqdm

from jwst import datamodels
from exotedrf.utils import fancyprint


def bin_at_pixel(flux, error, wave, npix):
    """Similar to bin_at_resolution, but will bin in widths of a set number of
    pixels instead of at a fixed resolution.

    Parameters
    ----------
    flux : array-like[float]
        Flux values.
    error : array-like[float]
        Flux error values.
    wave : array-like[float]
        Wavelength values.
    npix : int
        Number of pixels per bin.

    Returns
    -------
    wave_bin : np.ndarray[float]
        Central bin wavelength.
    wave_low : np.ndarray[float]
        Lower edge of wavelength bin.
    wave_up : np.ndarray[float]
        Upper edge of wavelength bin.
    dout : np.ndarray[float]
        Binned depth.
    derrout : np.ndarray[float]
        Error on binned depth.
    """

    # Calculate number of bins given wavelength grid and npix value.
    nint, nwave = np.shape(flux)
    # If the number of pixels does not bin evenly, trim from beginning and end.
    if nwave % npix != 0:
        cut = nwave % npix
        cut_s = int(np.floor(cut/2))
        cut_e = -1*(cut - cut_s)
        flux = flux[:, cut_s:cut_e]
        error = error[:, cut_s:cut_e]
        wave = wave[:, cut_s:cut_e]
        nint, nwave = np.shape(flux)
    nbin = int(nwave / npix)

    # Sum flux in bins and calculate resulting errors.
    flux_bin = np.nansum(np.reshape(flux, (nint, nbin, npix)), axis=2)
    err_bin = np.sqrt(np.nansum(np.reshape(error, (nint, nbin, npix))**2,
                                axis=2))
    # Calculate mean wavelength per bin.
    wave_bin = np.nanmean(np.reshape(wave, (nint, nbin, npix)), axis=2)
    # Get bin wavelength limits.
    up = np.concatenate([wave_bin[0][:, None],
                         np.roll(wave_bin[0][:, None], 1)], axis=1)
    lw = np.concatenate([wave_bin[0][:, None],
                         np.roll(wave_bin[0][:, None], -1)], axis=1)
    wave_up = (np.mean(up, axis=1) - wave_bin[0])[:-1]
    wave_up = np.insert(wave_up, -1, wave_up[-1])
    wave_up = np.repeat(wave_up, nint).reshape(nbin, nint).transpose(1, 0)
    wave_up = wave_bin + wave_up
    wave_low = (wave_bin[0] - np.mean(lw, axis=1))[1:]
    wave_low = np.insert(wave_low, 0, wave_low[0])
    wave_low = np.repeat(wave_low, nint).reshape(nbin, nint).transpose(1, 0)
    wave_low = wave_bin - wave_low

    return wave_bin, wave_low, wave_up, flux_bin, err_bin


def bin_at_resolution(inwave_low, inwave_up, flux, flux_err, res,
                      method='sum'):
    """Function that bins input wavelengths and transit depths (or any other
    observable, like flux) to a given resolution "res". Can handle 1D or 2D
    flux arrays.

    Parameters
    ----------
    inwave_low : array-like[float]
        Lower edge of wavelength bin. Must be 1D.
    inwave_up : array-like[float]
        Upper edge of wavelength bin. Must be 1D.
    flux : array-like[float]
        Flux values at each wavelength. Can be 1D or 2D. If 2D, the first axis
        must be the one corresponding to wavelength.
    flux_err : array-like[float]
        Errors corresponding to each flux measurement. Must be the same shape
        as flux.
    res : int
        Target resolution at which to bin.
    method : str
        Method to bin depths. Either "sum" or "average".

    Returns
    -------
    binned_waves : array-like[float]
        Wavelength of the given bin at the desired resolution.
    binned_werr : array-like[float]
        Half-width of the wavelength bin.
    binned_flux : array-like[float]
        Binned flux.
    binned_ferr : array-like[float]
        Error on binned flux.
    """

    def nextstep(w, r):
        return w * (2 * r + 1) / (2 * r - 1)

    # Sort quantities in order of increasing wavelength.
    waves = np.nanmean([inwave_low, inwave_up], axis=0)
    if np.ndim(waves) > 1:
        raise ValueError('Input wavelength array must be 1D.')
    ii = np.argsort(waves)
    waves, flux, flux_err = waves[ii], flux[ii], flux_err[ii]
    inwave_low, inwave_up = inwave_low[ii], inwave_up[ii]
    # Calculate the input resolution and check that we are not trying to bin
    # to a higher R.
    average_input_res = np.mean(waves[1:] / np.diff(waves))
    if res > average_input_res:
        raise ValueError('You are trying to bin at a higher resolution than '
                         'the input.')
    else:
        fancyprint('Binning from an average resolution of '
                   'R={:.0f} to R={}'.format(average_input_res, res))

    # Make the binned wavelength grid.
    outwave_low = []
    w_i, w_ip1 = waves[0], waves[0]
    while w_ip1 < waves[-1]:
        outwave_low.append(w_ip1)
        w_ip1 = nextstep(w_i, res)
        w_i = w_ip1
    outwave_low = np.array(outwave_low)
    outwave_up = np.append(outwave_low[1:], waves[-1])
    binned_waves = np.mean([outwave_low, outwave_up], axis=0)
    binned_werr = (outwave_up - outwave_low) / 2

    # Loop over all wavelengths in the input and bin flux and error into the
    # new wavelength grid.
    ii = 0
    for wl, wu in zip(outwave_low, outwave_up):
        first_time, count = True, 0
        current_flux = np.ones_like(flux[ii]) * np.nan
        current_ferr = np.ones_like(flux_err[ii]) * np.nan
        weight = []
        for i in range(ii, len(waves)):
            # If the wavelength is fully within the bin, append the flux and
            # error to the current bin info.
            if inwave_low[i] >= wl and inwave_up[i] < wu:
                if np.ndim(flux) == 1:
                    current_flux = np.hstack([flux[i], current_flux])
                    current_ferr = np.hstack([flux_err[i], current_ferr])
                else:
                    current_flux = np.vstack([flux[i], current_flux])
                    current_ferr = np.vstack([flux_err[i], current_ferr])
                count += 1
                weight.append(1)
            # For edge cases where one of the input bins falls on the edge of
            # the binned wavelength grid, linearly interpolate the flux into
            # the new bins.
            # Upper edge split.
            elif inwave_low[i] < wu <= inwave_up[i]:
                inbin_width = inwave_up[i] - inwave_low[i]
                in_frac = (inwave_up[i] - wu) / inbin_width
                weight.append(in_frac)
                if np.ndim(flux) == 1:
                    current_flux = np.hstack([flux[i], current_flux])
                    current_ferr = np.hstack([flux_err[i], current_ferr])
                else:
                    current_flux = np.vstack([flux[i], current_flux])
                    current_ferr = np.vstack([flux_err[i], current_ferr])
                count += 1
            # Lower edge split.
            elif inwave_low[i] < wl <= inwave_up[i]:
                inbin_width = inwave_up[i] - inwave_low[i]
                in_frac = (wl - inwave_low[i]) / inbin_width
                weight.append(in_frac)
                if np.ndim(flux) == 1:
                    current_flux = np.hstack([flux[i], current_flux])
                    current_ferr = np.hstack([flux_err[i], current_ferr])
                else:
                    current_flux = np.vstack([flux[i], current_flux])
                    current_ferr = np.vstack([flux_err[i], current_ferr])
                count += 1
            # Since wavelengths are in increasing order, once we exit the bin
            # completely we're done.
            if inwave_low[i] >= wu or i == len(waves)-1:
                if count != 0:
                    # If something was put into this bin, bin it using the
                    # requested method.
                    weight.append(0)
                    weight = np.array(weight)
                    if method == 'sum':
                        if np.ndim(current_flux) != 1:
                            thisflux = np.nansum(current_flux * weight[:, None],
                                                 axis=0)
                        else:
                            thisflux = np.nansum(current_flux * weight, axis=0)
                        thisferr = np.sqrt(np.nansum(current_ferr**2, axis=0))
                    elif method == 'average':
                        if np.ndim(current_flux) != 1:
                            thisflux = np.nansum(current_flux * weight[:, None],
                                                 axis=0)
                            thisflux /= np.nansum(weight)
                        else:
                            thisflux = np.nansum(current_flux * weight, axis=0)
                            thisflux /= np.nansum(weight)
                        thisferr = np.sqrt(np.nansum(current_ferr**2, axis=0))
                        thisferr /= np.nansum(weight)
                    else:
                        raise ValueError('Unknown method.')
                else:
                    # If nothing is in the bin (can happen if the output
                    # reslution is higher than the local input resolution),
                    # append NaNs
                    if np.ndim(flux) == 1:
                        thisflux, thisferr = np.nan, np.nan
                    else:
                        thisflux = np.ones_like(flux[0]) * np.nan
                        thisferr = np.ones_like(flux[0]) * np.nan
                # Store the binned quantities.
                if ii == 0:
                    binned_flux = thisflux
                    binned_ferr = thisferr
                else:
                    binned_flux = np.vstack([binned_flux, thisflux])
                    binned_ferr = np.vstack([binned_ferr, thisferr])
                # Move to the next bin.
                ii = i-1
                break

    # If the input was 1D, reformat to match.
    if np.ndim(flux) == 1:
        binned_flux = binned_flux[:, 0]
        binned_ferr = binned_ferr[:, 0]

    return binned_waves, binned_werr, binned_flux, binned_ferr


@ray.remote
def fit_data(data_dictionary, priors, output_dir, bin_no, num_bins):
    """Functional wrapper around run_juliet to make it compatible for
    multiprocessing with ray.
    """

    fancyprint('Fitting bin {} / {}'.format(bin_no, num_bins))

    # Get key names.
    all_keys = list(data_dictionary.keys())

    # Unpack fitting arrays
    t = {'SOSS': data_dictionary['times']}
    flux = {'SOSS': data_dictionary['flux']}
    flux_err = {'SOSS': data_dictionary['error']}

    # Initialize GP and linear model regressors.
    gp_regressors = None
    linear_regressors = None
    if 'GP_parameters' in all_keys:
        gp_regressors = {'SOSS': data_dictionary['GP_parameters']}
    if 'lm_parameters' in all_keys:
        linear_regressors = {'SOSS': data_dictionary['lm_parameters']}

    fit_results = run_juliet(priors, t, flux, flux_err, output_dir,
                             gp_regressors, linear_regressors)

    return fit_results


def fit_lightcurves(data_dict, prior_dict, order, output_dir, fit_suffix,
                    nthreads=4):
    """Wrapper about both the juliet and ray libraries to parallelize juliet's
    lightcurve fitting functionality.

    Parameters
    ----------
    data_dict : dict
        Dictionary of fitting data: time, flux, and flu error.
    prior_dict : dict
        Dictionary of fitting priors in juliet format.
    order : int
        SOSS diffraction order.
    output_dir : str
        Path to directory to which to save results.
    fit_suffix : str
        String to label the results of this fit.
    nthreads : int
        Number of cores to use for multiprocessing.

    Returns
    -------
    results : juliet.fit object
        The results of the juliet fit.
    """

    # Initialize results dictionary and keynames.
    results = dict.fromkeys(data_dict.keys(), [])
    keynames = list(data_dict.keys())

    # Format output directory
    if output_dir[-1] != '/':
        output_dir += '/'

    # Initialize ray with specified number of threads.
    ray.shutdown()
    ray.init(num_cpus=nthreads)

    # Set juliet fits as remotes to run parallel with ray.
    all_fits = []
    num_bins = np.arange(len(keynames))+1
    for i, keyname in enumerate(keynames):
        outdir = output_dir + 'speclightcurve{2}/order{0}_{1}'.format(order, keyname, fit_suffix)
        all_fits.append(fit_data.remote(data_dict[keyname],
                                        prior_dict[keyname],
                                        output_dir=outdir,
                                        bin_no=num_bins[i],
                                        num_bins=len(num_bins)))
    # Run the fits.
    ray_results = ray.get(all_fits)

    # Reorder the results based on the key name.
    for i in range(len(keynames)):
        keyname = keynames[i]
        results[keyname] = ray_results[i]

    return results


def gen_ld_coefs(wavebin_low, wavebin_up, order, m_h, logg, teff,
                 ld_data_path, spectrace_ref=None, mode=None,
                 model_type='stagger'):
    """Generate estimates of quadratic limb-darkening coefficients using the
    ExoTiC-LD package.

    Parameters
    ----------
    spectrace_ref : str, None
        Path to spectrace reference file.
    wavebin_low : array-like[float]
        Lower edge of wavelength bins.
    wavebin_up: array-like[float]
        Upper edge of wavelength bins.
    order : int
        SOSS diffraction order.
    m_h : float
        Stellar metallicity as [M/H].
    logg : float
        Stellar log gravity.
    teff : float
        Stellar effective temperature in K.
    ld_data_path : str
        Path to ExoTiC-LD model data.
    mode : str
        ExoTiC-LD instrument mode identifier.
    model_type : str
        Identifier for type of stellar model to use. See
        https://exotic-ld.readthedocs.io/en/latest/views/supported_stellar_grids.html
        for supported grids.

    Returns
    -------
    c1s : array-like[float]
        c1 parameter for the quadratic limb-darkening law.
    c2s : array-like[float]
        c2 parameter for the quadratic limb-darkening law.
    """

    # Set up the stellar model parameters - with specified model type.
    sld = StellarLimbDarkening(m_h, teff, logg, model_type, ld_data_path)

    if spectrace_ref is not None:
        # Load the most up to date throughput info for SOSS
        spec_trace = datamodels.SpecTraceModel(spectrace_ref)
        wavelengths = spec_trace.trace[order-1].data['WAVELENGTH'] * 10000
        throughputs = spec_trace.trace[order-1].data['THROUGHPUT']
        # Note that custom throughputs are used.
        mode = 'custom'

    # Compute the LD coefficients over the given wavelength bins.
    c1s, c2s = [], []
    for wl, wu in tqdm(zip(wavebin_low * 10000, wavebin_up * 10000),
                       total=len(wavebin_low)):
        wr = [wl, wu]
        try:
            if spectrace_ref is not None:
                c1, c2 = sld.compute_quadratic_ld_coeffs(wr, mode, wavelengths,
                                                         throughputs)
            else:
                c1, c2 = sld.compute_quadratic_ld_coeffs(wr, mode=mode)
        except ValueError:
            c1, c2 = np.nan, np.nan
        c1s.append(c1)
        c2s.append(c2)
    c1s = np.array(c1s)
    c2s = np.array(c2s)

    return c1s, c2s


def read_ld_coefs(filename, wavebin_low, wavebin_up, ld_model='quadratic'):
    """Unpack limb darkening coefficients and interpolate to the wavelength
    grid of data being fit. File must be comma separated with three columns:
    wavelength c1 and c2.

    Parameters
    ----------
    filename : str
        Path to file containing model limb darkening coefficients.
    wavebin_low : array-like[float]
        Lower edge of wavelength bins being fit.
    wavebin_up : array-like[float]
        Upper edge of wavelength bins being fit.
    ld_model : str
        Limb darkening model.

    Returns
    -------
    prior_q1 : array-like[float]
        Model estimates for q1 parameter.
    prior_q2 : array-like[float]
        Model estimates for q2 parameter.
    """

    # Open the LD model file and convert c1 and c2 parameters to q1 and q2 of
    # the Kipping (2013) parameterization.
    ld = pd.read_csv(filename, comment='#', sep=',')
    q1s, q2s = juliet.reverse_q_coeffs(ld_model, ld['c1'].values,
                                       ld['c2'].values)

    # Get model wavelengths and sort in increasing order.
    waves = ld['wave'].values
    ii = np.argsort(waves)
    waves = waves[ii]
    q1s, q2s = q1s[ii], q2s[ii]

    # Check that coeffs in file span the wavelength range of the observations
    # with at least the same resolution.
    if np.min(waves) < np.min(wavebin_low) or np.max(waves) > np.max(wavebin_up):
        msg = 'LD coefficient file does not span the full wavelength range ' \
              'of the observations.'
        raise ValueError(msg)
    if len(waves) < len(wavebin_low):
        msg = 'LD coefficient file has a coarser wavelength grid than ' \
              'the observations.'
        raise ValueError(msg)

    prior_q1, prior_q2 = [], []
    # Loop over all fitting bins. Calculate mean of model LD coefs within that
    # range.
    for wl, wu in zip(wavebin_low, wavebin_up):
        current_q1, current_q2 = [], []
        for w, q1, q2 in zip(waves, q1s, q2s):
            if wl < w <= wu:
                current_q1.append(q1)
                current_q2.append(q2)
            # Since LD model wavelengths are sorted in increasing order, once
            # we are above the upper edge of the bin, we can stop.
            elif w > wu:
                prior_q1.append(np.nanmean(current_q1))
                prior_q2.append(np.nanmean(current_q2))
                break

    return prior_q1, prior_q2


def run_juliet(priors, t_lc, y_lc, yerr_lc, out_folder,
               gp_regressors_lc, linear_regressors_lc):
    """Wrapper around the lightcurve fitting functionality of the juliet
    package.

    Parameters
    ----------
    priors : dict
        Dictionary of fitting priors.
    t_lc : dict
        Time axis.
    y_lc : dict
        Normalized lightcurve flux values.
    yerr_lc : dict
        Errors associated with each flux value.
    out_folder : str
        Path to folder to which to save results.
    gp_regressors_lc : dict
        GP regressors to fit, if any.
    linear_regressors_lc : dict
        Linear model regressors, if any.

    Returns
    -------
    res : juliet.fit object
        Results of juliet fit.
    """

    if np.isfinite(y_lc['SOSS']).all():
        # Load in all priors and data to be fit.
        dataset = juliet.load(priors=priors, t_lc=t_lc, y_lc=y_lc,
                              yerr_lc=yerr_lc, out_folder=out_folder,
                              GP_regressors_lc=gp_regressors_lc,
                              linear_regressors_lc=linear_regressors_lc)
        kwargs = {'print_progress': False}

        # Run the fit.
        try:
            res = dataset.fit(sampler='dynesty', **kwargs)
        except KeyboardInterrupt as err:
            raise err
        except:
            fancyprint('Exception encountered.', msg_type='WARNING')
            fancyprint('Skipping bin.', msg_type='WARNING')
            res = None
    else:
        fancyprint('NaN bin encountered.', msg_type='WARNING')
        fancyprint('Skipping bin.', msg_type='WARNING')
        res = None

    return res


def save_transmission_spectrum(wave, wave_err, dppm, dppm_err, order, outdir,
                               filename, target, extraction_type, resolution,
                               fit_meta='', occultation_type='transit'):
    """Write a transmission/emission spectrum to file.

    Parameters
    ----------
    wave : array-like[float]
        Wavelength values.
    wave_err : array-like[float]
        Bin half-widths for each wavelength bin.
    dppm : array-like[float]
        Transit/eclipse depth in each bin.
    dppm_err : array-like[float]
        Error on the transit/eclipse depth in each bin.
    order : array-like[int]
        SOSS order corresponding to each bin.
    outdir : str
        Firectory to whch to save outputs.
    filename : str
        Name of the file to which to save spectra.
    target : str
        Target name.
    extraction_type : str
        Type of extraction: either box or atoca.
    resolution: int, str
        Spectral resolution of spectrum.
    fit_meta: str
        Fitting metadata.
    occultation_type : str
        Type of occultation; either 'transit' or 'eclipse'.
    """

    # Hack to fix weird end bin errors.
    ii = np.where(wave_err < 0)
    wave_err[ii] = np.median(wave_err)

    # Pack the quantities into a dictionary.
    dd = {'wave': wave,
          'wave_err': wave_err,
          'dppm': dppm,
          'dppm_err': dppm_err,
          'order': order}
    # Save the dictionary as a csv.
    df = pd.DataFrame(data=dd)
    if os.path.exists(outdir + filename):
        os.remove(outdir + filename)

    # Re-open the csv and append some critical info the header.
    f = open(outdir + filename, 'w')
    f.write('# Target: {}\n'.format(target))
    f.write('# Instrument: NIRISS/SOSS\n')
    f.write('# Pipeline: exoTEDRF\n')
    f.write('# 1D Extraction: {}\n'.format(extraction_type))
    f.write('# Spectral Resolution: {}\n'.format(resolution))
    f.write('# Author: {}\n'.format(os.environ.get('USER')))
    f.write('# Date: {}\n'.format(datetime.utcnow().replace(microsecond=0).isoformat()))
    f.write(fit_meta)
    f.write('# Column wave: Central wavelength of bin (micron)\n')
    f.write('# Column wave_err: Wavelength bin halfwidth (micron)\n')
    if occultation_type == 'transit':
        f.write('# Column dppm: (Rp/R*)^2 (ppm)\n')
        f.write('# Column dppm_err: Error in (Rp/R*)^2 (ppm)\n')
    else:
        f.write('# Column dppm: (Fp/F*) (ppm)\n')
        f.write('# Column dppm_err: Error in (Fp/F*) (ppm)\n')
    f.write('# Column order: SOSS diffraction order\n')
    f.write('#\n')
    df.to_csv(f, index=False)
    f.close()
