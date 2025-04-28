#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 14:35 2022

@author: MCR

Script to fit simple transit and eclipse light curves with exoUPRF.
"""

from astropy.io import fits
import copy
from datetime import datetime
import exouprf.light_curve_models as model
from exouprf.plotting import make_lightcurve_plot
import matplotlib.backends.backend_pdf
import numpy as np
import os
import pandas as pd
import shutil
import sys

from exotedrf import stage4
from exotedrf import plotting, utils
from exotedrf.utils import fancyprint, verify_path

# Read config file.
try:
    config_file = sys.argv[1]
except IndexError:
    raise FileNotFoundError('Config file must be provided')
config = utils.parse_config(config_file)


def fit_lightcurves(config):
    """Do light curve fitting.
    """

    if config['output_tag'] != '':
        config['output_tag'] = '_' + config['output_tag']
    # Create output directories and define output paths.
    utils.verify_path('pipeline_outputs_directory' + config['output_tag'])
    utils.verify_path('pipeline_outputs_directory' + config['output_tag'] + '/Stage4')
    outdir = 'pipeline_outputs_directory' + config['output_tag'] + '/Stage4/'

    # Tag for this particular fit.
    if config['fit_suffix'] != '':
        fit_suffix = '_' + config['fit_suffix']
    else:
        fit_suffix = config['fit_suffix']
    # Add resolution/binning info to the fit tag.
    if config['res'] is not None:
        if config['res'] == 'pixel':
            fit_suffix += '_pixel'
            res_str = 'pixel resolution'
        elif config['res'] == 'prebin':
            fit_suffix += '_prebin'
            res_str = 'custom prebinned resolution'
        else:
            fit_suffix += '_R{}'.format(config['res'])
            res_str = 'R = {}'.format(config['res'])
    elif config['npix'] is not None:
        fit_suffix += '_{}pix'.format(config['npix'])
        res_str = 'npix = {}'.format(config['npix'])
    else:
        raise ValueError('Number of columns to bin or spectral resolution must be provided.')

    # Save a copy of the config file.
    root_dir = 'pipeline_outputs_directory' + config['output_tag'] + '/config_files'
    verify_path(root_dir)
    i = 0
    copy_config = root_dir + '/' + config_file
    root = copy_config.split('.yaml')[0]
    copy_config = root + '{}.yaml'.format(fit_suffix)
    while os.path.exists(copy_config):
        i += 1
        copy_config = root_dir + '/' + config_file
        root = copy_config.split('.yaml')[0]
        copy_config = root + '{0}_{1}.yaml'.format(fit_suffix, i)
    shutil.copy(config_file, copy_config)
    # Append time at which it was run.
    f = open(copy_config, 'a')
    time = datetime.utcnow().isoformat(sep=' ', timespec='minutes')
    f.write('\nRun at {}.'.format(time))
    f.close()

    # Formatted parameter names for plotting.
    formatted_names = {'per_p1': r'$P$', 't0_p1': r'$T_0$', 'rp_p1_inst': r'$R_p/R_*$',
                       'inc_p1': r'$i$', 'u1_inst': r'$u_1$', 'u2_inst': r'$u_2$',
                       'ecc_p1': r'$e$', 'w_p1': r'$\Omega$', 'a_p1': r'$a/R_*$',
                       'sigma_inst': r'$\sigma$', 'zero_inst': 'zero point',
                       'theta1_inst': r'$\theta_1$', 'theta2_inst': r'$\theta_2$',
                       'theta3_inst': r'$\theta_3$', 'theta4_inst': r'$\theta_4$',
                       'theta5_inst': r'$\theta_5$', 'GP_sigma_inst': r'$GP \sigma$',
                       'GP_rho_inst': r'$GP \rho$', 'GP_S0_inst': r'$GP S0$',
                       'GO_omega0_inst': r'$GP \Omega_0$', 'GP_Q_inst': r'$GP Q$', 'rho': r'$\rho$',
                       'tsec_p1': r'$T_{sec}$', 'fp_p1_inst': r'$F_p/F_*$'}

    # === Get Detrending Quantities ===
    # Get time axis
    if config['observing_mode'] == 'NIRISS/SOSS':
        t = fits.getdata(config['infile'], 9)
    else:
        t = fits.getdata(config['infile'], 5)
    # Quantities against which to linearly detrend.
    if config['lm_file'] is not None:
        lm_data = pd.read_csv(config['lm_file'], comment='#')
        lm_quantities = np.zeros((len(config['lm_parameters']), len(t)))
        for i, key in enumerate(config['lm_parameters']):
            lm_param = lm_data[key]
            lm_quantities[i] = (lm_param - np.mean(lm_param)) / np.std(lm_param)

    # Quantity on which to train GP.
    if config['gp_file'] is not None:
        gp_data = pd.read_csv(config['gp_file'], comment='#')
        gp_quantities = gp_data[config['gp_parameter']].values

    # Format the baseline frames - either out-of-transit or in-eclipse.
    baseline_ints = utils.format_out_frames(config['baseline_ints'])

    # === Fit Light Curves ===
    # Start the light curve fitting.
    results_dict = {}
    model_file_names = []
    if config['observing_mode'] != 'NIRISS/SOSS':
        config['orders'] = [1]
    for order in config['orders']:
        if config['do_plots'] is True:
            if config['observing_mode'] == 'NIRISS/SOSS':
                expected_file = outdir + 'speclightcurve{1}/lightcurve_fit_order{0}.pdf'.format(
                    order, fit_suffix)
                outpdf = matplotlib.backends.backend_pdf.PdfPages(expected_file)
            elif 'NIRSPEC' in config['observing_mode']:
                expected_file = outdir + 'speclightcurve{1}/lightcurve_fit_{0}.pdf'.format(
                    config['detector'], fit_suffix)
                outpdf = matplotlib.backends.backend_pdf.PdfPages(expected_file)
            else:
                expected_file = outdir + 'speclightcurve{}/lightcurve_fit.pdf'.format(fit_suffix)
                outpdf = matplotlib.backends.backend_pdf.PdfPages(expected_file)
        else:
            outpdf = None

        # === Set Up Priors and Fit Parameters ===
        if config['observing_mode'] == 'NIRISS/SOSS':
            fancyprint('Fitting order {} at {}.'.format(order, res_str))
        elif 'NIRSPEC' in config['observing_mode']:
            fancyprint('Fitting detector {} at {}.'.format(config['detector'], res_str))
        else:
            fancyprint('Fitting light curves at {}.'.format(res_str))
        # Unpack wave, flux and error.
        wave = fits.getdata(config['infile'], 1 + 4 * (order - 1))
        wave_err = fits.getdata(config['infile'], 2 + 4 * (order - 1))
        flux = fits.getdata(config['infile'], 3 + 4 * (order - 1))
        err = fits.getdata(config['infile'], 4 + 4 * (order - 1))
        # Cut reference pixel columns if data is not prebinned.
        if config['res'] != 'prebin':
            wave = wave[5:-5]
            wave_err = wave_err[5:-5]
            flux = flux[:, 5:-5]
            err = err[:, 5:-5]

        # For order 2, only fit wavelength bins between 0.6 and 0.85µm.
        if order == 2:
            ii = np.where((wave >= 0.6) & (wave <= 0.85))[0]
            flux, err = flux[:, ii], err[:, ii]
            wave, wave_err = wave[ii], wave_err[ii]
        # For NRS1, only fit wavelengths larger than blue cutoff.
        if config['detector'] == 'NRS1' and config['observing_mode'].upper() == 'NIRSPEC/G395H':
            if config['res'] != 'prebin':
                ii = np.where(wave >= config['nrs1_blue'])[0]
                flux, err = flux[:, ii], err[:, ii]
                wave, wave_err = wave[ii], wave_err[ii]
        # For MIRI, only fit wavelengths in given bounds.
        if config['observing_mode'].upper() == 'MIRI/LRS':
            if config['res'] != 'prebin':
                ii = np.where((wave >= config['lrs_blue']) & (wave <= config['lrs_red']))[0]
                flux, err = flux[:, ii], err[:, ii]
                wave, wave_err = wave[ii], wave_err[ii]

        # Bin input spectra to desired resolution.
        if config['res'] is not None:
            if config['res'] == 'pixel' or config['res'] == 'prebin':
                pass
            else:
                binned_vals = stage4.bin_at_resolution(wave, flux.T, err.T, res=config['res'])
                wave, wave_err, flux, err = binned_vals
                flux, err = flux.T, err.T
        else:
            binned_vals = stage4.bin_at_pixel(wave, flux, err, npix=config['npix'])
            wave, wave_err, flux, err = binned_vals
        nints, nbins = np.shape(flux)

        # Sort input arrays in order of increasing wavelength.
        ii = np.argsort(wave)
        wave, wave_err = wave[ii], wave_err[ii]
        flux, err = flux[:, ii], err[:, ii]

        # Normalize flux and error by the baseline.
        baseline = np.median(flux[baseline_ints], axis=0)
        norm_flux = flux / baseline
        norm_err = err / baseline

        # Set up priors
        priors = {}
        for param, dist, hyperp in zip(config['params'], config['dists'], config['values']):
            priors[param] = {}
            priors[param]['distribution'] = dist
            priors[param]['value'] = hyperp

        # For transit fits, calculate LD coefficients from stellar models.
        if config['lc_model_type'] == 'transit' and config['ld_fit_type'] != 'free':
            calculate = True
            # Convert wave and wave_err to wavebin edges.
            wave_low, wave_up = wave - wave_err, wave + wave_err
            # Check if LD coefficient files have been provided.
            if config['ldcoef_file{}'.format(order)] is not None:
                calculate = False
                fancyprint('Reading limb-darkening coefficient file.')
                try:
                    ld = stage4.read_ld_coefs(config['ldcoef_file{}'.format(order)],
                                              wave_low, wave_up)
                    u1 = np.array(ld[0])
                    if config['ld_model_type'] != 'linear':
                        u2 = np.array(ld[1])
                    if config['ld_model_type'] == 'nonlinear':
                        u3 = np.array(ld[2])
                        u4 = np.array(ld[3])
                except ValueError:
                    fancyprint('LD coefficient file could not be correctly parsed. Falling back '
                               'onto LD calculation.', msg_type='WARNING')
                    calculate = True
            if calculate is True:
                # Calculate LD coefficients on specified wavelength grid.
                fancyprint('Calculating limb-darkening coefficients.')
                m_h, logg, teff = config['m_h'], config['logg'], config['teff']
                msg = 'All stellar parameters needed to calculate limb-darkening coefficients.'
                assert np.all(np.array([m_h, logg, teff]) != None), msg
                # Get ExoTiC-LD instrument mode identifier.
                modes = {'NIRISS/SOSS1': 'JWST_NIRISS_SOSSo1',
                         'NIRISS/SOSS2': 'JWST_NIRISS_SOSSo2',
                         'NIRSpec/PRISM': 'JWST_NIRSpec_Prism',
                         'NIRSpec/G395H': 'JWST_NIRSpec_G395H',
                         'NIRSpec/G395M': 'JWST_NIRSpec_G395M',
                         'NIRSpec/G235H': 'JWST_NIRSpec_G235H',
                         'NIRSpec/G235M': 'JWST_NIRSpec_G235M',
                         'NIRSpec/G140H': 'JWST_NIRSpec_G140H',
                         'NIRSpec/G140M': 'JWST_NIRSpec_G140M',
                         'MIRI/LRS': 'JWST_MIRI_LRS'}
                if config['observing_mode'] != 'NIRISS/SOSS':
                    thismode = config['observing_mode']
                else:
                    thismode = config['observing_mode'] + '{}'.format(order)
                ld = stage4.gen_ld_coefs(wavebin_low=wave_low, wavebin_up=wave_up, m_h=m_h,
                                         logg=logg, teff=teff, ld_data_path=config['ld_data_path'],
                                         stellar_model_type=config['stellar_model_type'],
                                         mode=modes[thismode],
                                         ld_model_type=config['ld_model_type'])
                u1 = np.array(ld[0])
                if config['ld_model_type'] != 'linear':
                    u2 = np.array(ld[1])
                if config['ld_model_type'] == 'nonlinear':
                    u3 = np.array(ld[2])
                    u4 = np.array(ld[3])

                # Save calculated coefficients.
                target = fits.getheader(config['infile'], 0)['TARGET']
                outdir_ld = outdir + 'speclightcurve{}/'.format(fit_suffix)
                utils.verify_path(outdir_ld)
                if config['observing_mode'] != 'NIRISS/SOSS' and config['detector'] == 'NRS2':
                    thisorder = 2
                else:
                    thisorder = order
                utils.save_ld_priors(wave, ld, thisorder, target, m_h, teff, logg,
                                     outdir_ld, config['ld_model_type'], config['observing_mode'])

        # Pack fitting arrays and priors into dictionaries.
        data_dict, prior_dict = {}, {}
        for wavebin in range(nbins):
            # Data dictionaries, including linear model and GP regressors.
            thisbin = 'wavebin' + str(wavebin)
            bin_dict = {'times': t, 'flux': norm_flux[:, wavebin]}
            # If linear models are to be included.
            if config['lm_file'] is not None:
                bin_dict['lm_parameters'] = lm_quantities
            # If GPs are to be inclided.
            if config['gp_file'] is not None:
                bin_dict['GP_parameters'] = gp_quantities
            data_dict[thisbin] = bin_dict

            # Prior dictionaries.
            prior_dict[thisbin] = copy.deepcopy(priors)
            # For transit only; update the LD prior for this bin if available.
            # For prior: set prior width to 0.2 around the model value - based on findings of
            # Patel & Espinoza 2022.
            if config['lc_model_type'] == 'transit' and config['ld_fit_type'] != 'free':
                if config['ld_model_type'] == 'quadratic-kipping':
                    low_lim = 0.0
                else:
                    low_lim = -1.0
                if np.isfinite(u1[wavebin]):
                    if config['ld_fit_type'] == 'prior':
                        dist = 'truncated_normal'
                        vals = [u1[wavebin], 0.2, low_lim, 1.0]
                    elif config['ld_fit_type'] == 'fixed':
                        dist = 'fixed'
                        vals = u1[wavebin]
                    prior_dict[thisbin]['u1_inst']['distribution'] = dist
                    prior_dict[thisbin]['u1_inst']['value'] = vals
                    if config['ld_model_type'] != 'linear':
                        if config['ld_fit_type'] == 'prior':
                            dist = 'truncated_normal'
                            vals = [u2[wavebin], 0.2, low_lim, 1.0]
                        elif config['ld_fit_type'] == 'fixed':
                            dist = 'fixed'
                            vals = u2[wavebin]
                        prior_dict[thisbin]['u2_inst']['distribution'] = dist
                        prior_dict[thisbin]['u2_inst']['value'] = vals
                    if config['ld_model_type'] == 'nonlinear':
                        if config['ld_fit_type'] == 'prior':
                            dist = 'truncated_normal'
                            vals3 = [u3[wavebin], 0.2, low_lim, 1.0]
                            vals4 = [u4[wavebin], 0.2, low_lim, 1.0]
                        elif config['ld_fit_type'] == 'fixed':
                            dist = 'fixed'
                            vals3 = u3[wavebin]
                            vals4 = u4[wavebin]
                        prior_dict[thisbin]['u3_inst']['distribution'] = dist
                        prior_dict[thisbin]['u3_inst']['value'] = vals3
                        prior_dict[thisbin]['u4_inst']['distribution'] = dist
                        prior_dict[thisbin]['u4_inst']['value'] = vals4

        # Get information about custom light curve model, if being used.
        if 'custom' in config['lc_model_type']:
            if config['custom_lc_model'] is None:
                # No custom light curve function passed, raise error.
                raise AttributeError('Light curve model type is custom, but no custom model '
                                     'passed.')
            elif isinstance(config['custom_lc_model'], str):
                try:
                    custom_lc_function = getattr(model, config['custom_lc_model'])
                except AttributeError:
                    all_models = utils.get_exouprf_built_in_models(model)
                    raise AttributeError('exoUPRF has no built-in light curve model {0}. Available '
                                         'models are: {1}'.format(config['custom_lc_model'],
                                                                  all_models))
            else:
                thistype = type(config['custom_lc_model'])
                raise ValueError('Object passed to custom_lc_model of type {} is not a string or '
                                 'None.'.format(thistype))
        else:
            # No custom light curve function to be used, all is good.
            custom_lc_function = None

        # === Do the Fit ===
        # Fit each light curve
        if config['observing_mode'].split('/')[0].upper() == 'NIRSPEC':
            thisorder = int(config['detector'][-1])
        else:
            thisorder = order
        fit_results = stage4.fit_lightcurves(data_dict=data_dict, prior_dict=prior_dict,
                                             order=thisorder, output_dir=outdir,
                                             nthreads=config['ncores'], fit_suffix=fit_suffix,
                                             observing_mode=config['observing_mode'],
                                             lc_model_type=config['lc_model_type'],
                                             ld_model=config['ld_model_type'],
                                             custom_lc_function=custom_lc_function,
                                             debug=config['debug'])

        # === Summarize Fit Results ===
        # Loop over results for each wavebin, extract best-fitting parameters and make summary
        # plots.
        fancyprint('Summarizing fit results.')
        data = np.ones((nints, nbins)) * np.nan
        models = np.ones((4, nints, nbins)) * np.nan
        residuals = np.ones((nints, nbins)) * np.nan
        order_results = {'dppm': [], 'dppm_err': [], 'wave': wave, 'wave_err': wave_err}
        for i, wavebin in enumerate(fit_results.keys()):
            # Make note if something went wrong with this bin.
            skip = False
            if fit_results[wavebin] is None:
                skip = True

            # Pack best fit Rp/R* into a dictionary. Append NaNs if the bin was skipped.
            if skip is True:
                order_results['dppm'].append(np.nan)
                order_results['dppm_err'].append(np.nan)
            # If not skipped, append median and 1-sigma bounds.
            else:
                this_result = fit_results[wavebin].get_results_from_fit()
                if 'transit' in config['lc_model_type']:
                    md = this_result['rp_p1_inst']['median']
                    up = this_result['rp_p1_inst']['up_1sigma']
                    lw = this_result['rp_p1_inst']['low_1sigma']
                    order_results['dppm'].append((md ** 2) * 1e6)
                    err_low = (md ** 2 - (md - lw) ** 2) * 1e6
                    err_up = ((up + md) ** 2 - md ** 2) * 1e6
                else:
                    md = this_result['fp_p1_inst']['median']
                    up = this_result['fp_p1_inst']['up_1sigma']
                    lw = this_result['fp_p1_inst']['low_1sigma']
                    order_results['dppm'].append(md * 1e6)
                    err_low = (md ** 2 - (md - lw) ** 2) * 1e6
                    err_up = ((up + md) ** 2 - md ** 2) * 1e6
                order_results['dppm_err'].append(np.max([err_up, err_low]))

            # Summarize fits and make plots if necessary.
            if skip is False:
                # Get dictionary of best-fitting parameters from fit.
                param_dict = fit_results[wavebin].get_param_dict_from_fit()
                # Calculate best-fitting light curve model.
                thislm, thisgp = None, None
                if config['lm_file'] is not None:
                    thislm = {'inst': data_dict[wavebin]['lm_parameters']}
                if config['gp_file'] is not None:
                    thisgp = {'inst': data_dict[wavebin]['GP_parameters']}
                thislcmod = {'inst': {'p1': config['lc_model_type']}}
                thislcfunc = {'inst': {'p1': custom_lc_function}}
                result = model.LightCurveModel(
                    input_parameters=param_dict,
                    t={'inst': data_dict[wavebin]['times']},
                    linear_regressors=thislm, gp_regressors=thisgp,
                    observations={'inst': {'flux': data_dict[wavebin]['flux']}},
                    ld_model=config['ld_model_type'], silent=True
                )
                result.compute_lightcurves(lc_model_type=thislcmod, lc_model_functions=thislcfunc)

                # Plot transit model and residuals.
                scatter = param_dict['sigma_inst']['value']
                nfit = len(np.where(config['dists'] != 'fixed')[0])
                t0_loc = np.where(np.array(config['params']) == 't0_p1')[0][0]
                if config['dists'][t0_loc] == 'fixed':
                    t0 = config['values'][t0_loc]
                else:
                    t0 = param_dict['t0_p1']['value']

                # Get systematics and transit models.
                transit_model = result.flux['inst']
                systematics = None
                gp_model, lm_model = None, None
                if config['lm_file'] is not None or config['gp_file'] is not None:
                    systematics = np.zeros_like(transit_model)
                    if config['lm_file'] is not None:
                        lm_model = result.flux_decomposed['inst']['lm']['total']
                        systematics += lm_model
                    if config['gp_file'] is not None:
                        gp_model = result.flux_decomposed['inst']['gp']['total']
                        systematics += gp_model

                if config['do_plots'] is True:
                    make_lightcurve_plot(t=(t - t0) * 24, data=norm_flux[:, i], model=transit_model,
                                         scatter=scatter, errors=norm_err[:, i], outpdf=outpdf,
                                         title='bin {0} | {1:.3f}µm'.format(i, wave[i]),
                                         systematics=systematics, nfit=nfit, nbin=int(len(t) / 35),
                                         rasterized=True)
                    # Corner plot for fit.
                    if config['include_corner'] is True:
                        posterior_names = []
                        for param in prior_dict[wavebin].keys():
                            dist = prior_dict[wavebin][param]['distribution']
                            if dist != 'fixed':
                                if param in formatted_names.keys():
                                    posterior_names.append(formatted_names[param])
                                else:
                                    posterior_names.append(param)
                        fit_results[wavebin].make_corner_plot(labels=posterior_names, outpdf=outpdf)

                data[:, i] = norm_flux[:, i]
                models[0, :, i] = transit_model
                if systematics is not None:
                    models[1, :, i] = systematics
                if gp_model is not None:
                    models[2, :, i] = gp_model
                models[3, :, i] = norm_flux[:, i]
                residuals[:, i] = norm_flux[:, i] - transit_model

        results_dict['order {}'.format(order)] = order_results

        # Save best-fitting light curve models.
        if config['observing_mode'].upper() == 'NIRISS/SOSS':
            order_txt = 'order{}'.format(order)
        elif 'NIRSPEC' in config['observing_mode'].upper():
            order_txt = config['detector']
        else:
            order_txt = 'LRS'
        filename = outdir + 'speclightcurve{0}/_models_{1}.npy'.format(fit_suffix, order_txt)
        np.save(filename, models)
        model_file_names.append(filename)

        # Plot 2D lightcurves.
        if config['do_plots'] is True:
            plotting.make_2d_lightcurve_plot(wave, data, outpdf=outpdf,
                                             title='Normalized Lightcurves')
            plotting.make_2d_lightcurve_plot(wave, models[0], outpdf=outpdf,
                                             title='Model Lightcurves')
            plotting.make_2d_lightcurve_plot(wave, residuals, outpdf=outpdf, title='Residuals')
            outpdf.close()
            fancyprint('Plot saved to {}'.format(expected_file))

    # === Transmission Spectrum ===
    # Save the transmission spectrum.
    fancyprint('Writing spectrum to file.')
    for order in ['1', '2']:
        if 'order ' + order not in results_dict.keys():
            order_results = {'dppm': [], 'dppm_err': [], 'wave': [], 'wave_err': []}
            results_dict['order ' + order] = order_results

    # Concatenate transit depths, wavelengths, and associated errors from both
    # orders.
    depths = np.concatenate([results_dict['order 2']['dppm'], results_dict['order 1']['dppm']])
    errors = np.concatenate([results_dict['order 2']['dppm_err'],
                             results_dict['order 1']['dppm_err']])
    waves = np.concatenate([results_dict['order 2']['wave'], results_dict['order 1']['wave']])
    wave_errors = np.concatenate([results_dict['order 2']['wave_err'],
                                  results_dict['order 1']['wave_err']])
    orders = np.concatenate([2 * np.ones_like(results_dict['order 2']['dppm']),
                             np.ones_like(results_dict['order 1']['dppm'])]).astype(int)

    # Get target/reduction metadata.
    infile_header = fits.getheader(config['infile'], 0)
    extract_type = infile_header['METHOD']
    target = infile_header['TARGET'] + config['planet_letter']
    if 'transit' in config['lc_model_type']:
        spec_type = 'transmission'
    else:
        spec_type = 'emission'
    inst = '_'
    for chunk in config['observing_mode'].split('/'):
        inst += '{}_'.format(chunk)
    if config['observing_mode'][-1] == 'H':
        inst += '{}_'.format(config['detector'])
    filename = target + inst + spec_type + '_spectrum' + fit_suffix + '.csv'
    # Get fit metadata.
    # Include fixed parameter values.
    fit_metadata = '#\n# Fit Metadata\n'
    for param, dist, value in zip(config['params'], config['dists'], config['values']):
        if dist == 'fixed':
            try:
                fit_metadata += '# {}: {}\n'.format(formatted_names[param], value)
            except KeyError:
                fit_metadata += '# {}: {}\n'.format(param, value)
    # Append info on detrending via linear models or GPs.
    if len(config['lm_parameters']) != 0:
        fit_metadata += '# Linear Model: '
        for i, param in enumerate(config['lm_parameters']):
            if i == 0:
                fit_metadata += param
            else:
                fit_metadata += ', {}'.format(param)
        fit_metadata += '\n'
    if config['gp_parameter'] != '':
        fit_metadata += '# Gaussian Process: '
        fit_metadata += config['gp_parameter']
        fit_metadata += '\n'
    fit_metadata += '#\n'

    # Save spectrum.
    stage4.save_transmission_spectrum(wave=waves, wave_err=wave_errors, dppm=depths,
                                      dppm_err=errors, order=orders, outdir=outdir,
                                      filename=filename, target=target,
                                      extraction_type=extract_type, resolution=config['res'],
                                      fit_meta=fit_metadata,
                                      occultation_type=config['lc_model_type'],
                                      observing_mode=config['observing_mode'])
    fancyprint('{0} spectrum saved to {1}'.format(spec_type, outdir + filename))

    # === Covariance Matrix ===
    # Get the covariance matrix for the residuals.
    fancyprint('Calculating covariance matrix.')
    cov_matrix = stage4.calculate_residual_covariance(model_file_names)
    # Save covariance matrix.
    if config['observing_mode'].upper() in ['NIRISS/SOSS', 'MIRI/LRS']:
        order_txt = ''
    else:
        order_txt = '_{}'.format(config['detector'])
    filename = outdir + 'speclightcurve{0}/_covariance_matrix{1}.npy'.format(fit_suffix, order_txt)
    np.save(filename, cov_matrix)

    # Do the covariance matrix plot if requested.
    if config['do_plots'] is True:
        plotfile = outdir + 'speclightcurve{1}/covariance_matrix{0}.pdf'.format(order_txt,
                                                                                fit_suffix)
        plotting.make_lightcurve_covariance_plot(cov_matrix, waves, outfile=plotfile)

    fancyprint('Done')

    return


# ===== Do Stuff =====
if __name__ == '__main__':
    fit_lightcurves(config)
