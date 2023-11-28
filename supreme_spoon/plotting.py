#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 14:02 2022

@author: MCR

Plotting routines.
"""

from astropy.io import fits
from astropy.timeseries import LombScargle
import bottleneck as bn
import corner
import matplotlib.backends.backend_pdf
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter
from tqdm import tqdm
import warnings

from supreme_spoon import utils
from supreme_spoon.utils import fancyprint


def make_background_plot(results, outfile=None, show_plot=True):
    """Nine-panel plot for background subtraction results.
    """
    kwargs = {'max_percentile': 70}
    basic_nine_panel_plot(results, outfile=outfile, show_plot=show_plot,
                          **kwargs)


def make_badpix_plot(deep, hotpix, otherpix, outfile=None, show_plot=True):
    """Show locations of interpolated pixels.
    """

    fancyprint('Doing diagnostic plot.')
    # Plot the location of all jumps and hot pixels.
    fig, ax = plt.subplots(figsize=(8, 5), facecolor='white')
    plt.imshow(deep, aspect='auto', origin='lower', vmin=0,
               vmax=np.nanpercentile(deep, 85))

    # Show hot pixel locations.
    first_time = True
    for ypos, xpos in zip(hotpix[0], hotpix[1]):
        if first_time is True:
            marker = Ellipse((xpos, ypos), 21, 3, color='red',
                             fill=False, label='Hot Pixel')
            ax.add_patch(marker)
            first_time = False
        else:
            marker = Ellipse((xpos, ypos), 21, 3, color='red',
                             fill=False)
            ax.add_patch(marker)

    # Show 'other' locations.
    first_time = True
    for ypos, xpos in zip(otherpix[0], otherpix[1]):
        if first_time is True:
            marker = Ellipse((xpos, ypos), 21, 3, color='green',
                             fill=False, label='Other')
            ax.add_patch(marker)
            first_time = False
        else:
            marker = Ellipse((xpos, ypos), 21, 3, color='green',
                             fill=False)
            ax.add_patch(marker)

    plt.yticks(fontsize=10)
    plt.xticks(fontsize=10)
    plt.legend(loc=1)

    if outfile is not None:
        plt.savefig(outfile, bbox_inches='tight')
        fancyprint('Plot saved to {}'.format(outfile))
    if show_plot is False:
        plt.close()
    else:
        plt.show()


def make_corner_plot(fit_params, results, posterior_names=None, outpdf=None,
                     truths=None):
    """Make corner plot for lightcurve fitting.
    """

    first_time = True
    for param in fit_params:
        if first_time:
            pos = results.posteriors['posterior_samples'][param]
            first_time = False
        else:
            pos = np.vstack((pos, results.posteriors['posterior_samples'][param]))

    figure = corner.corner(pos.T, labels=posterior_names, color='black',
                           show_titles=True, title_fmt='.3f',
                           label_kwargs=dict(fontsize=14), truths=truths,
                           facecolor='white')
    if outpdf is not None:
        if isinstance(outpdf, matplotlib.backends.backend_pdf.PdfPages):
            outpdf.savefig(figure)
        else:
            figure.savefig(outpdf)
        figure.clear()
        plt.close(figure)
    else:
        plt.show()


def make_decontamination_plot(results, models, outfile=None, show_plot=True):
    """Nine-pixel plot for ATOCA decontamination.
    """

    fancyprint('Doing diagnostic plot.')
    results = np.atleast_1d(results)
    for i, file in enumerate(results):
        with utils.open_filetype(file) as datamodel:
            if i == 0:
                cube = datamodel.data
                ecube = datamodel.err
            else:
                cube = np.concatenate([cube, datamodel.data])
                ecube = np.concatenate([ecube, datamodel.err])

    models = np.atleast_1d(models)
    for i, model in enumerate(models):
        if i == 0:
            order1 = fits.getdata(model, 2)
            order2 = fits.getdata(model, 3)
        else:
            order1 = np.concatenate([order1, fits.getdata(model, 2)])
            order2 = np.concatenate([order2, fits.getdata(model, 3)])

    ints = np.random.randint(0, np.shape(cube)[0], 9)
    to_plot, to_write = [], []
    for i in ints:
        to_plot.append((cube[i] - order1[i] - order2[i]) / ecube[i])
        to_write.append('({0})'.format(i))
    kwargs = {'vmin': -5, 'vmax': 5}
    nine_panel_plot(to_plot, to_write, outfile=outfile, show_plot=show_plot,
                    **kwargs)
    if outfile is not None:
        fancyprint('Plot saved to {}'.format(outfile))


def make_jump_location_plot(results, outfile=None, show_plot=True):
    """Show locations of detected jumps.
    """

    fancyprint('Doing diagnostic plot.')
    results = np.atleast_1d(results)
    for i, file in enumerate(results):
        with utils.open_filetype(file) as datamodel:
            if i == 0:
                cube = datamodel.data
                dqcube = datamodel.groupdq
            else:
                cube = np.concatenate([cube, datamodel.data])
                dqcube = np.concatenate([dqcube, datamodel.groupdq])
            pixeldq = datamodel.pixeldq
    nint, ngroup, dimy, dimx = np.shape(cube)

    # Plot the location of all jumps and hot pixels.
    plt.figure(figsize=(15, 9), facecolor='white')
    gs = GridSpec(3, 3)

    for k in range(3):
        for j in range(3):
            # Get random group and integration.
            i = np.random.randint(nint)
            g = np.random.randint(1, ngroup)

            # Get location of all hot pixels and jump detections.
            hot = utils.get_dq_flag_metrics(pixeldq, ['HOT', 'WARM'])
            jump = utils.get_dq_flag_metrics(dqcube[i, g], ['JUMP_DET'])
            hot = np.where(hot != 0)
            jump = np.where(jump != 0)

            ax = plt.subplot(gs[k, j])
            diff = cube[i, g] - cube[i, g-1]
            plt.imshow(diff, aspect='auto', origin='lower', vmin=0,
                       vmax=np.nanpercentile(diff, 85))

            # Show hot pixel locations.
            first_time = True
            for ypos, xpos in zip(hot[0], hot[1]):
                if first_time is True:
                    marker = Ellipse((xpos, ypos), 21, 3, color='blue',
                                     fill=False, label='Hot Pixel')
                    ax.add_patch(marker)
                    first_time = False
                else:
                    marker = Ellipse((xpos, ypos), 21, 3, color='blue',
                                     fill=False)
                    ax.add_patch(marker)

            # Show jump locations.
            first_time = True
            for ypos, xpos in zip(jump[0], jump[1]):
                if first_time is True:
                    marker = Ellipse((xpos, ypos), 21, 3, color='red',
                                     fill=False, label='Cosmic Ray')
                    ax.add_patch(marker)
                    first_time = False
                else:
                    marker = Ellipse((xpos, ypos), 21, 3, color='red',
                                     fill=False)
                    ax.add_patch(marker)

            ax.text(30, 230, '({0}, {1})'.format(i, g), c='white', fontsize=12)
            if j != 0:
                ax.yaxis.set_major_formatter(plt.NullFormatter())
            else:
                plt.yticks(fontsize=10)
            if k != 2:
                ax.xaxis.set_major_formatter(plt.NullFormatter())
            else:
                plt.xticks(fontsize=10)
            if k == 0 and j == 0:
                plt.legend(loc=1)

    if outfile is not None:
        plt.savefig(outfile, bbox_inches='tight')
        fancyprint('Plot saved to {}'.format(outfile))
    if show_plot is False:
        plt.close()
    else:
        plt.show()


def make_lightcurve_plot(t, data, model, scatter, errors, nfit, outpdf=None,
                         title=None, systematics=None, rasterized=False,
                         nbin=10):
    """Plot results of lightcurve fit.
    """

    def gaus(x, m, s):
        return np.exp(-0.5*(x - m)**2/s**2)/np.sqrt(2*np.pi*s**2)

    def chi2(o, m, e):
        return np.nansum((o - m)**2/e**2)

    if systematics is not None:
        fig = plt.figure(figsize=(13, 9), facecolor='white',
                         rasterized=rasterized)
        gs = GridSpec(5, 1, height_ratios=[3, 3, 1, 0.3, 1])
    else:
        fig = plt.figure(figsize=(13, 7), facecolor='white',
                         rasterized=rasterized)
        gs = GridSpec(4, 1, height_ratios=[3, 1, 0.3, 1])

    # Light curve with full systematics + astrophysical model.
    ax1 = plt.subplot(gs[0])
    assert len(data) == len(model)
    nint = len(data)  # Total number of data points
    # Full dataset
    ax1.errorbar(t, data, yerr=scatter*1e-6, fmt='o', capsize=0,
                 color='royalblue', ms=5, alpha=0.25)
    # Binned points
    rem = nint % nbin
    if rem != 0:
        trim_i = np.random.randint(0, rem)
        trim_e = -1*(rem-trim_i)
        t_bin = t[trim_i:trim_e].reshape((nint-rem)//nbin, nbin)
        d_bin = data[trim_i:trim_e].reshape((nint-rem)//nbin, nbin)
    else:
        t_bin = t.reshape((nint-rem)//nbin, nbin)
        d_bin = data.reshape((nint-rem)//nbin, nbin)
    t_bin = np.nanmean(t_bin, axis=1)
    d_bin = np.nanmean(d_bin, axis=1)
    ax1.errorbar(t_bin, d_bin, yerr=scatter*1e-6/np.sqrt(nbin), fmt='o',
                 mfc='blue', mec='white', ecolor='blue', ms=8, alpha=1,
                 zorder=11)
    # Other stuff.
    ax1.plot(t, model, color='black', zorder=10)
    ax1.set_ylabel('Relative Flux', fontsize=18)
    ax1.set_xlim(np.min(t), np.max(t))
    ax1.xaxis.set_major_formatter(plt.NullFormatter())
    chi2_v = chi2(data*1e6, model*1e6, errors*1e6) / (len(t) - nfit)
    mean_err = np.nanmean(errors)
    err_mult = scatter / (mean_err*1e6)
    ax1.text(t[2], np.min(model),
             r'$\chi_\nu^2 = {:.2f}$''\n'r'$\sigma={:.2f}$ppm''\n'r'$e={:.2f}$'.format(
                 chi2_v, mean_err*1e6, err_mult),
             fontsize=14)
    ax1.tick_params(axis='x', labelsize=12)
    ax1.tick_params(axis='y', labelsize=12)

    if title is not None:
        plt.title(title, fontsize=16)

    # Detrended Light curve.
    if systematics is not None:
        ax2 = plt.subplot(gs[1])
        assert len(model) == len(systematics)
        model_detrended = model - systematics
        data_detrended = data - systematics
        # Full dataset.
        ax2.errorbar(t, data_detrended, yerr=scatter*1e-6, fmt='o',
                     capsize=0, color='salmon', ms=5, alpha=0.25)
        # Binned points.
        if rem != 0:
            d_bin = data_detrended[trim_i:trim_e].reshape((nint-rem)//nbin, nbin)
        else:
            d_bin = data_detrended.reshape((nint-rem)//nbin, nbin)
        d_bin = np.nanmean(d_bin, axis=1)
        ax2.errorbar(t_bin, d_bin, yerr=scatter*1e-6/np.sqrt(nbin), fmt='o',
                     mfc='red', mec='white', ecolor='red', ms=8, alpha=1,
                     zorder=11)
        # Other stuff.
        ax2.plot(t, model_detrended, color='black', zorder=10)
        ax2.set_ylabel('Relative Flux\n(Detrended)', fontsize=18)
        ax2.set_xlim(np.min(t), np.max(t))
        ax2.xaxis.set_major_formatter(plt.NullFormatter())
        ax2.tick_params(axis='x', labelsize=12)
        ax2.tick_params(axis='y', labelsize=12)

    # Residuals.
    if systematics is not None:
        ax3 = plt.subplot(gs[2])
    else:
        ax3 = plt.subplot(gs[1])
    # Full dataset.
    res = (data - model)*1e6
    ax3.errorbar(t, res, yerr=scatter, alpha=0.25, ms=5,
                 c='royalblue', fmt='o', zorder=10)
    # Binned points.
    if rem != 0:
        r_bin = res[trim_i:trim_e].reshape((nint-rem)//nbin, nbin)
    else:
        r_bin = res.reshape((nint-rem)//nbin, nbin)
    r_bin = np.nanmean(r_bin, axis=1)
    ax3.errorbar(t_bin, r_bin, yerr=scatter/np.sqrt(nbin), fmt='o',
                 mfc='blue', mec='white', ecolor='blue', ms=8, alpha=1,
                 zorder=11)
    # Other stuff.
    ax3.axhline(0, ls='--', c='black')
    xpos = np.percentile(t, 1)
    plt.text(xpos, np.max((data - model)*1e6),
             r'{:.2f}$\,$ppm'.format(scatter))
    ax3.fill_between(t, -scatter, scatter, color='black', alpha=0.1)
    ax3.set_xlim(np.min(t), np.max(t))
    ax3.set_ylabel('Residuals\n(ppm)', fontsize=18)
    ax3.set_xlabel('Time from Transit Midpoint [hrs]', fontsize=18)
    ax3.tick_params(axis='x', labelsize=12)
    ax3.tick_params(axis='y', labelsize=12)

    # Histogram of residuals.
    if systematics is not None:
        ax4 = plt.subplot(gs[4])
    else:
        ax4 = plt.subplot(gs[3])
    bins = np.linspace(-10, 10, 41) + 0.25
    hist = ax4.hist(res/scatter, edgecolor='grey', color='lightgrey', bins=bins)
    area = np.sum(hist[0] * np.diff(bins))
    ax4.plot(np.linspace(-15, 15, 500),
             gaus(np.linspace(-15, 15, 500), 0, 1) * area, c='black')
    ax4.set_ylabel('Counts', fontsize=18)
    ax4.set_xlabel('Residuals/Scatter', fontsize=18)
    ax4.set_xlim(-5, 5)
    ax4.tick_params(axis='x', labelsize=12)
    ax4.tick_params(axis='y', labelsize=12)

    if outpdf is not None:
        if isinstance(outpdf, matplotlib.backends.backend_pdf.PdfPages):
            outpdf.savefig(fig)
        else:
            fig.savefig(outpdf)
        fig.clear()
        plt.close(fig)
    else:
        plt.show()


def make_linearity_plot(results, old_results, outfile=None, show_plot=True):
    """Plot group differences before and after linearity correction.
    """

    fancyprint('Doing diagnostic plot.')
    results = np.atleast_1d(results)
    old_results = np.atleast_1d(old_results)
    for i, file in enumerate(results):
        with utils.open_filetype(file) as datamodel:
            if i == 0:
                cube = datamodel.data
            else:
                cube = np.concatenate([cube, datamodel.data])
    for i, file in enumerate(old_results):
        with utils.open_filetype(file) as datamodel:
            if i == 0:
                old_cube = datamodel.data
            else:
                old_cube = np.concatenate([old_cube, datamodel.data])

    nint, ngroup, dimy, dimx = np.shape(cube)
    # Get bright pixels in the trace.
    stack = bn.nanmedian(cube[np.random.randint(0, nint, 25), -1], axis=0)
    ii = np.where((stack >= np.nanpercentile(stack, 80)) &
                  (stack < np.nanpercentile(stack, 99)))

    # Compute group differences in these pixels.
    new_diffs = np.zeros((ngroup-1, len(ii[0])))
    old_diffs = np.zeros((ngroup-1, len(ii[0])))
    num_pix = 20000
    if len(ii[0]) < 20000:
        num_pix = len(ii[0])
    for it in range(num_pix):
        ypos, xpos = ii[0][it], ii[1][it]
        # Choose a random integration.
        i = np.random.randint(0, nint)
        # Calculate the group differences.
        new_diffs[:, it] = np.diff(cube[i, :, ypos, xpos])
        old_diffs[:, it] = np.diff(old_cube[i, :, ypos, xpos])

    new_med = np.mean(new_diffs, axis=1)
    old_med = np.mean(old_diffs, axis=1)

    # Plot up mean group differences before and after linearity correction.
    plt.figure(figsize=(5, 3))
    plt.plot(np.arange(len(new_med)), new_med - np.mean(new_med),
             label='After Correction', c='blue', lw=2)
    plt.plot(np.arange(len(new_med)), old_med - np.mean(old_med),
             label='Before Correction', c='red', lw=2)
    plt.axhline(0, ls='--', c='black', zorder=0)
    plt.xlabel(r'Groups', fontsize=12)
    locs = np.arange(ngroup-1).astype(int)
    labels = []
    for i in range(ngroup-1):
        labels.append('{0}-{1}'.format(i+2, i+1))
    if 25 < len(locs) < 50:
        plt.xticks(locs[::2], labels[::2], rotation=45)
    elif len(locs) > 50:
        plt.xticks(locs[::3], labels[::3], rotation=45)
    else:
        plt.xticks(locs, labels, rotation=45)
    plt.ylabel('Differences [DN]', fontsize=12)
    plt.ylim(1.1*np.min(old_med - np.mean(old_med)),
             1.1*np.max(old_med - np.mean(old_med)))
    plt.legend()

    if outfile is not None:
        plt.savefig(outfile, bbox_inches='tight')
        fancyprint('Plot saved to {}'.format(outfile))
    if show_plot is False:
        plt.close()
    else:
        plt.show()


def make_oneoverf_plot(results, baseline_ints, timeseries=None,
                       outfile=None, show_plot=True):
    """make nine-panel plot of dataframes after 1/f correction.
    """

    fancyprint('Doing diagnostic plot 1.')
    results = np.atleast_1d(results)
    for i, file in enumerate(results):
        with utils.open_filetype(file) as datamodel:
            if i == 0:
                cube = datamodel.data
            else:
                cube = np.concatenate([cube, datamodel.data])

    # Format the baseline frames - either out-of-transit or in-eclipse.
    baseline_ints = utils.format_out_frames(baseline_ints)
    # Make deepstack using baseline integrations.
    deep = utils.make_deepstack(cube[baseline_ints])

    # Get smoothed light curve.
    if isinstance(timeseries, str):
        try:
            timeseries = np.load(timeseries)
        except (ValueError, FileNotFoundError):
            timeseries = None
    # If no lightcurve is provided, use array of ones.
    if timeseries is None:
        timeseries = np.ones(np.shape(cube[0]))

    if np.ndim(cube) == 4:
        nint, ngroup, dimy, dimx = np.shape(cube)
        ints = np.random.randint(0, nint, 9)
        grps = np.random.randint(0, ngroup, 9)
        to_plot, to_write = [], []
        kwargs = {'vmin': -50, 'vmax': 50}
        for i, g in zip(ints, grps):
            to_plot.append(cube[i, g] - deep[g] * timeseries[i])
            to_write.append('({0}, {1})'.format(i, g))
    else:
        nint, dimy, dimx = np.shape(cube)
        ints = np.random.randint(0, nint, 9)
        to_plot, to_write = [], []
        kwargs = {'vmin': -50, 'vmax': 50}
        for i in ints:
            to_plot.append(cube[i] - deep * timeseries[i])
            to_write.append('({0})'.format(i))
    nine_panel_plot(to_plot, to_write, outfile=outfile, show_plot=show_plot,
                    **kwargs)
    if outfile is not None:
        fancyprint('Plot saved to {}'.format(outfile))


def make_oneoverf_psd(results, old_results, timeseries, baseline_ints,
                      nsample=25,  pixel_masks=None, tframe=5.494, tpix=1e-5,
                      tgap=1.2e-4, outfile=None, show_plot=True):
    """Make a PSD plot to see PSD of background before and after 1/f removal.
    """

    fancyprint('Doing diagnostic plot 2.')

    results = np.atleast_1d(results)
    old_results = np.atleast_1d(old_results)
    for i, file in enumerate(results):
        with utils.open_filetype(file) as datamodel:
            if i == 0:
                cube = datamodel.data
            else:
                cube = np.concatenate([cube, datamodel.data])
    cube = np.where(np.isnan(cube), np.nanmedian(cube), cube)
    for i, file in enumerate(old_results):
        with utils.open_filetype(file) as datamodel:
            if i == 0:
                old_cube = datamodel.data
            else:
                old_cube = np.concatenate([old_cube, datamodel.data])
    old_cube = np.where(np.isnan(old_cube), np.nanmedian(old_cube), old_cube)
    if pixel_masks is not None:
        for i, file in enumerate(pixel_masks):
            if i == 0:
                mask_cube = fits.getdata(file)
            else:
                mask_cube = np.concatenate([mask_cube, fits.getdata(file)])
    else:
        mask_cube = None

    if np.ndim(cube) == 4:
        nints, ngroups, dimy, dimx = np.shape(cube)
    else:
        nints, dimy, dimx = np.shape(cube)
    baseline_ints = utils.format_out_frames(baseline_ints)
    old_deep = bn.nanmedian(old_cube[baseline_ints], axis=0)
    deep = bn.nanmedian(cube[baseline_ints], axis=0)

    # Get smoothed light curve.
    if isinstance(timeseries, str):
        try:
            timeseries = np.load(timeseries)
        except (ValueError, FileNotFoundError):
            timeseries = None
    # If no lightcurve is provided, use array of ones.
    if timeseries is None:
        timeseries = np.ones(np.shape(cube[0]))

    # Generate array of timestamps for each pixel
    pixel_ts = []
    time1 = 0
    for p in range(dimy * dimx):
        ti = time1 + tpix
        # If column is done, add gap time.
        if p % dimy == 0 and p != 0:
            ti += tgap
        pixel_ts.append(ti)
        time1 = ti

    # Generate psd frequency array
    freqs = np.logspace(np.log10(1 / tframe), np.log10(1 / tpix), 100)
    pwr_old = np.zeros((nsample, len(freqs)))
    pwr = np.zeros((nsample, len(freqs)))
    # Select nsample random frames and compare PSDs before and after 1/f
    # removal.
    for s in tqdm(range(nsample)):
        if np.ndim(cube) == 4:
            # Get random groups and ints.
            i, g = np.random.randint(nints), np.random.randint(ngroups)
            # Get difference images before and after 1/f removal.
            diff_old = (old_cube[i, g] - old_deep[g] * timeseries[i]).flatten('F')[::-1]
            diff = (cube[i, g] - deep[g] * timeseries[i]).flatten('F')[::-1]
        else:
            # Get random ints.
            i = np.random.randint(nints)
            # Get difference images before and after 1/f removal.
            diff_old = (old_cube[i] - old_deep * timeseries[i]).flatten('F')[::-1]
            diff = (cube[i] - deep * timeseries[i]).flatten('F')[::-1]
        # Mask pixels which are not part of the background
        if mask_cube is None:
            # If no pixel/trace mask, discount pixels above a threshold.
            bad = np.where(np.abs(diff) > 100)
        else:
            # Mask flagged pixels.
            bad = np.where(mask_cube[i] != 0)
        diff, diff_old = np.delete(diff, bad), np.delete(diff_old, bad)
        this_t = np.delete(pixel_ts, bad)
        # Calculate PSDs
        pwr_old[s] = LombScargle(this_t, diff_old).power(freqs, normalization='psd')
        pwr[s] = LombScargle(this_t, diff).power(freqs, normalization='psd')

    # Make the plot.
    plt.figure(figsize=(7, 3))
    # Individual power series.
    for i in range(nsample):
        plt.plot(freqs[:-1], pwr_old[i, :-1], c='salmon', alpha=0.1)
        plt.plot(freqs[:-1], pwr[i, :-1], c='royalblue', alpha=0.1)
    # Median trends.
    # Aprox white noise level
    plt.plot(freqs[:-1], np.nanmedian(pwr_old, axis=0)[:-1], c='red', lw=2,
             label='Before Correction')
    plt.plot(freqs[:-1], np.nanmedian(pwr, axis=0)[:-1], c='blue', lw=2,
             label='After Correction')

    plt.xscale('log')
    plt.xlabel('Frequency [Hz]', fontsize=12)
    plt.yscale('log')
    plt.ylim(np.nanpercentile(pwr, 0.1), np.nanmax(pwr_old))
    plt.ylabel('PSD', fontsize=12)
    plt.legend(loc=1)

    if outfile is not None:
        plt.savefig(outfile, bbox_inches='tight')
        fancyprint('Plot saved to {}'.format(outfile))
    if show_plot is False:
        plt.close()
    else:
        plt.show()


def make_pca_plot(pcs, var, projections, show_plot=False, outfile=None):
    """Plot of PCA results and reprojections.
    """

    fancyprint('Plotting PCA outputs.')
    var_no1 = var / np.nansum(var[1:])

    plt.figure(figsize=(12, 15), facecolor='white')
    gs = GridSpec(len(var), 2)

    for i in range(len(var)):
        ax1 = plt.subplot(gs[i, 0])
        if i == 0:
            label = '{0:.2e}'.format(var[i])
        else:
            label = '{0:.2e} | {1:.2f}'.format(var[i], var_no1[i])
        plt.plot(pcs[i], c='black', label=label)

        ax1.legend(handlelength=0, handletextpad=0, fancybox=True)

        ax2 = plt.subplot(gs[i, 1])
        plt.imshow(projections[i], aspect='auto', origin='lower',
                   vmin=np.nanpercentile(projections[i], 1),
                   vmax=np.nanpercentile(projections[i], 99))

        if i != len(var) - 1:
            ax1.xaxis.set_major_formatter(plt.NullFormatter())
            ax2.xaxis.set_major_formatter(plt.NullFormatter())
        else:
            ax1.set_xlabel('Integration Number', fontsize=14)
            ax2.set_xlabel('Spectral Pixel', fontsize=14)

    gs.update(hspace=0.1, wspace=0.1)

    if outfile is not None:
        plt.savefig(outfile, bbox_inches='tight')
        fancyprint('Plot saved to {}'.format(outfile))
    if show_plot is False:
        plt.close()
    else:
        plt.show()


def make_superbias_plot(results, outfile=None, show_plot=True):
    """Nine-panel plot for superbias subtraction results.
    """
    basic_nine_panel_plot(results, outfile=outfile, show_plot=show_plot)


def make_2d_lightcurve_plot(wave1, flux1, wave2=None, flux2=None, outpdf=None,
                            title='', **kwargs):
    """Plot 2D spectroscopic light curves.
    """

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')

        if 'vmax' not in kwargs:
            kwargs['vmax'] = np.nanpercentile(flux1, 95)
        if 'vmin' not in kwargs:
            kwargs['vmin'] = np.nanpercentile(flux1, 5)

        if title != '':
            title = ': ' + title

        fig = plt.figure(figsize=(12, 5), facecolor='white')
        gs = GridSpec(1, 2, width_ratios=[1, 1])

        ax1 = fig.add_subplot(gs[0, 0])
        pp = ax1.imshow(flux1.T, aspect='auto', origin='lower',
                        extent=(0, flux1.shape[0]-1, wave1[0], wave1[-1]),
                        **kwargs)
        if wave2 is None:
            cax = ax1.inset_axes([1.05, 0.005, 0.03, 0.99],
                                 transform=ax1.transAxes)
            cb = fig.colorbar(pp, ax=ax1, cax=cax)
            cb.set_label('Normalized Flux', labelpad=15, rotation=270,
                         fontsize=16)
        ax1.set_ylabel('Wavelength [µm]', fontsize=16)
        ax1.set_xlabel('Integration Number', fontsize=16)
        plt.title('Order 1' + title, fontsize=18)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        if wave2 is not None:
            ax2 = fig.add_subplot(gs[0, 1])
            pp = ax2.imshow(flux2.T, aspect='auto', origin='lower',
                            extent=(0, flux2.shape[0]-1, wave2[0], wave2[-1]),
                            **kwargs)
            cax = ax2.inset_axes([1.05, 0.005, 0.03, 0.99],
                                 transform=ax2.transAxes)
            cb = fig.colorbar(pp, ax=ax2, cax=cax)
            cb.set_label('Normalized Flux', labelpad=15, rotation=270,
                         fontsize=16)
            ax2.set_xlabel('Integration Number', fontsize=16)
            plt.title('Order 2' + title, fontsize=18)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)

            gs.update(wspace=0.15)

    if outpdf is not None:
        if isinstance(outpdf, matplotlib.backends.backend_pdf.PdfPages):
            outpdf.savefig(fig)
        else:
            fig.savefig(outpdf)
        fig.clear()
        plt.close(fig)
    else:
        plt.show()


def basic_nine_panel_plot(results, outfile=None, show_plot=True, **kwargs):
    """Do general nine-panel plot of either 4D or 3D data.
    """

    fancyprint('Doing diagnostic plot.')
    results = np.atleast_1d(results)
    for i, file in enumerate(results):
        with utils.open_filetype(file) as datamodel:
            if i == 0:
                cube = datamodel.data
            else:
                cube = np.concatenate([cube, datamodel.data])

    if np.ndim(cube) == 4:
        nint, ngroup, dimy, dimx = np.shape(cube)
        grps = np.random.randint(0, ngroup, 9)
    else:
        nint, dimy, dimx = np.shape(cube)
        ngroup = 0
    ints = np.random.randint(0, nint, 9)

    to_plot, to_write = [], []
    if ngroup != 0:
        for i, g in zip(ints, grps):
            to_plot.append(cube[i, g])
            to_write.append('({0}, {1})'.format(i, g))
    else:
        for i in ints:
            to_plot.append(cube[i])
            to_write.append('({0})'.format(i))
    nine_panel_plot(to_plot, to_write, outfile=outfile, show_plot=show_plot,
                    **kwargs)
    if outfile is not None:
        fancyprint('Plot saved to {}'.format(outfile))


def nine_panel_plot(data, text=None, outfile=None, show_plot=True, **kwargs):
    """Basic setup for nine panel plotting.
    """

    plt.figure(figsize=(15, 9), facecolor='white')
    gs = GridSpec(3, 3)

    frame = 0
    for i in range(3):
        for j in range(3):
            ax = plt.subplot(gs[i, j])
            if 'vmin' not in kwargs.keys():
                vmin = 0
            else:
                vmin = kwargs['vmin']
            if 'vmax' not in kwargs.keys():
                if 'max_percentile' not in kwargs.keys():
                    max_percentile = 85
                else:
                    max_percentile = kwargs['max_percentile']
                vmax = np.nanpercentile(data[frame], max_percentile)
                while vmax <= vmin:
                    max_percentile += 5
                    vmax = np.nanpercentile(data[frame], max_percentile)
            else:
                vmax = kwargs['vmax']
            ax.imshow(data[frame], aspect='auto', origin='lower', vmin=vmin,
                      vmax=vmax)
            if text is not None:
                ax.text(30, 0.9*np.shape(data[frame])[0], text[frame],
                        c='white', fontsize=12)
            if j != 0:
                ax.yaxis.set_major_formatter(plt.NullFormatter())
            else:
                plt.yticks(fontsize=10)
            if i != 2:
                ax.xaxis.set_major_formatter(plt.NullFormatter())
            else:
                plt.xticks(fontsize=10)
            frame += 1

    gs.update(hspace=0.05, wspace=0.05)

    if outfile is not None:
        plt.savefig(outfile, bbox_inches='tight')
    if show_plot is False:
        plt.close()
    else:
        plt.show()
