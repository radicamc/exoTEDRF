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

from jwst import datamodels

from exotedrf import utils
from exotedrf.utils import fancyprint


def make_background_plot(results, outfile=None, show_plot=True):
    """Nine-panel plot for background subtraction results.
    """
    kwargs = {'max_percentile': 70}
    basic_nine_panel_plot(results, outfile=outfile, show_plot=show_plot,
                          **kwargs)


def make_background_row_plot(before, after, background_model, row_start=230,
                             row_end=251, f=1, outfile=None, show_plot=True):
    """Plot rows after background subtraction.
    """

    # Open files.
    with utils.open_filetype(before) as file:
        bf = file.data
    with utils.open_filetype(after) as file:
        af = file.data
    if isinstance(background_model, str):
        bkg = np.load(background_model)
    else:
        bkg = background_model

    # If SUBSTRIP96, change rows to use.
    if np.shape(af)[-2] == 96:
        sub96 = True
        row_start = 5
        row_end = 21
    else:
        sub96 = False

    # Create medians.
    if np.ndim(af) == 4:
        before = bn.nanmedian(bf[:, -1], axis=0)
        after = bn.nanmedian(af[:, -1], axis=0)
        bbkg = np.nanmedian(bkg[-1, row_start:row_end], axis=0)
    else:
        before = bn.nanmedian(bf, axis=0)
        after = bn.nanmedian(af, axis=0)
        bbkg = np.nanmedian(bkg[0, row_start:row_end], axis=0)
    bbefore = np.nanmedian(before[row_start:row_end], axis=0)
    aafter = np.nanmedian(after[row_start:row_end], axis=0)

    plt.figure(figsize=(5, 3))
    plt.plot(bbefore)
    plt.plot(np.arange(2048)[:700], bbkg[:700], c='black', ls='--')
    plt.plot(aafter)

    bkg_scale = f * (bbkg[700:] - bbkg[700]) + bbkg[700]
    plt.plot(np.arange(2048)[700:], bkg_scale, c='black', ls='--')
    plt.plot(np.arange(2048)[700:], bbefore[700:] - bkg_scale)

    plt.axvline(700, ls=':', c='grey')
    plt.axhline(0, ls=':', c='grey')
    if sub96 is True:
        plt.ylim(np.min([np.nanmin(aafter), np.nanmin(bbefore[700:] - bkg_scale)]),
                 np.nanpercentile(bbefore, 75))
    else:
        plt.ylim(np.min([np.nanmin(aafter), np.nanmin(bbefore[700:] - bkg_scale)]),
                 np.nanpercentile(bbefore, 95))
    plt.xlabel('Spectral Pixel', fontsize=12)
    plt.ylabel('Counts', fontsize=12)

    if outfile is not None:
        plt.savefig(outfile, bbox_inches='tight')
        fancyprint('Plot saved to {}'.format(outfile))
    if show_plot is False:
        plt.close()
    else:
        plt.show()


def make_badpix_plot(deep, hotpix, nanpix, otherpix, outfile=None,
                     show_plot=True):
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

    # Show negative locations.
    first_time = True
    for ypos, xpos in zip(nanpix[0], nanpix[1]):
        if first_time is True:
            marker = Ellipse((xpos, ypos), 21, 3, color='blue',
                             fill=False, label='Negative')
            ax.add_patch(marker)
            first_time = False
        else:
            marker = Ellipse((xpos, ypos), 21, 3, color='blue',
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


def make_compare_spectra_plot(spec1, spec2, title=None):
    """Make plot comparing two spectra.
    """

    # Get maximum error of the two spectra.
    emax = np.sqrt(np.sum([spec1['dppm_err'].values**2,
                           spec2['dppm_err'].values**2], axis=0))
    # Find where spectra deviate by multiples of emax.
    i1 = np.where(np.abs(spec1['dppm'].values - spec2['dppm'].values) / emax > 1)[0]
    i2 = np.where(np.abs(spec1['dppm'].values - spec2['dppm'].values) / emax > 2)[0]
    i3 = np.where(np.abs(spec1['dppm'].values - spec2['dppm'].values) / emax > 3)[0]

    f = plt.figure(figsize=(10, 7))
    gs = GridSpec(4, 1, height_ratios=[1, 1, 1, 1])

    # Spectrum #1.
    ax1 = f.add_subplot(gs[0])
    ax1.errorbar(spec1['wave'].values, spec1['dppm'].values,
                 yerr=spec1['dppm_err'].values, fmt='o', mec='black',
                 mfc='white', ecolor='black', label=r'1$\sigma$')
    ax1.errorbar(spec1['wave'].values[i1], spec1['dppm'].values[i1],
                 yerr=spec1['dppm_err'].values[i1], fmt='o', mec='blue',
                 mfc='white', ecolor='blue', label=r'2$\sigma$')
    ax1.errorbar(spec1['wave'].values[i2], spec1['dppm'].values[i2],
                 yerr=spec1['dppm_err'].values[i2], fmt='o', mec='orange',
                 mfc='white', ecolor='orange', label=r'3$\sigma$')
    ax1.errorbar(spec1['wave'].values[i3], spec1['dppm'].values[i3],
                 yerr=spec1['dppm_err'].values[i3], fmt='o', mec='green',
                 mfc='white', ecolor='green', label=r'>3$\sigma$')
    # Show spectrum #2 in faded points.
    ax1.errorbar(spec1['wave'].values, spec2['dppm'], yerr=spec2['dppm_err'],
                 fmt='o', mec='black', mfc='white', ecolor='black', alpha=0.1)
    plt.legend(ncol=2)
    ax1.set_xscale('log')
    ax1.set_xlim(0.58, 2.9)
    plt.xticks([0.6, 0.8, 1.0, 1.5, 2.0, 2.5], ['', '', '', '', '', ''])
    ax1.set_ylabel(r'(R$_p$/R$_*)^2$ [ppm]', fontsize=12)

    # Spectrum #2.
    ax2 = f.add_subplot(gs[1])
    ax2.errorbar(spec2['wave'].values, spec2['dppm'],
                 yerr=spec2['dppm_err'], fmt='o', mec='black',
                 mfc='white', ecolor='black', label=r'1$\sigma$')
    ax2.errorbar(spec2['wave'].values[i1], spec2['dppm'].values[i1],
                 yerr=spec2['dppm_err'].values[i1], fmt='o', mec='blue',
                 mfc='white', ecolor='blue', label=r'2$\sigma$')
    ax2.errorbar(spec2['wave'].values[i2], spec2['dppm'].values[i2],
                 yerr=spec2['dppm_err'].values[i2], fmt='o', mec='orange',
                 mfc='white', ecolor='orange', label=r'3$\sigma$')
    ax2.errorbar(spec2['wave'].values[i3], spec2['dppm'].values[i3],
                 yerr=spec2['dppm_err'].values[i3], fmt='o', mec='green',
                 mfc='white', ecolor='green', label=r'>3$\sigma$')
    # Show spectrum #1 in faded points.
    ax2.errorbar(spec2['wave'].values, spec1['dppm'].values,
                 yerr=spec1['dppm_err'].values, fmt='o', mec='black',
                 mfc='white', ecolor='black', alpha=0.1)
    ax2.set_xscale('log')
    ax2.set_xlim(0.58, 2.9)
    plt.xticks([0.6, 0.8, 1.0, 1.5, 2.0, 2.5], ['', '', '', '', '', ''])
    ax2.set_ylabel(r'(R$_p$/R$_*)^2$ [ppm]', fontsize=12)

    # Differences in multiples of emax.
    ax3 = f.add_subplot(gs[2])
    dev = (spec1['dppm'].values - spec2['dppm'].values) / emax
    ax3.errorbar(spec2['wave'].values, dev,
                 fmt='o', mec='black', mfc='white', ecolor='black')
    ax3.errorbar(spec2['wave'].values[i1], dev[i1],
                 fmt='o', mec='blue', mfc='white', ecolor='blue')
    ax3.errorbar(spec2['wave'].values[i2], dev[i2],
                 fmt='o', mec='orange', mfc='white', ecolor='orange')
    ax3.errorbar(spec2['wave'].values[i3], dev[i3],
                 fmt='o', mec='green', mfc='white', ecolor='green')
    ax3.plot(spec2['wave'].values, median_filter(dev, int(0.1 * len(dev))),
             c='red', zorder=100, ls='--')
    plt.axhline(0, ls='--', c='grey', zorder=99)

    maxy = np.ceil(np.max(dev)).astype(int)
    miny = np.floor(np.min(dev)).astype(int)
    ypoints = np.linspace(miny, maxy, (maxy - miny) + 1).astype(int)
    ax3.set_xscale('log')
    ax3.set_xlim(0.58, 2.9)
    plt.xticks([0.6, 0.8, 1.0, 1.5, 2.0, 2.5], ['', '', '', '', '', ''])
    ax3.text(0.6, maxy - 0.25 * maxy,
             r'$\bar\sigma$={:.2f}'.format(np.sum(np.abs(dev)) / len(dev)))
    ax3.axhspan(-1, 1, color='grey', alpha=0.2)
    ax3.axhspan(-0.5, 0.5, color='grey', alpha=0.2)
    plt.yticks(ypoints, ypoints.astype(str))
    ax3.set_ylabel(r'$\Delta$ [$\sigma$]', fontsize=14)

    # Differences in ppm.
    ax4 = f.add_subplot(gs[3])
    dev = (spec1['dppm'].values - spec2['dppm'].values)
    ax4.errorbar(spec2['wave'].values, dev,
                 fmt='o', mec='black', mfc='white', ecolor='black')
    ax4.errorbar(spec2['wave'].values[i1], dev[i1],
                 fmt='o', mec='blue', mfc='white', ecolor='blue')
    ax4.errorbar(spec2['wave'].values[i2], dev[i2],
                 fmt='o', mec='orange', mfc='white', ecolor='orange')
    ax4.errorbar(spec2['wave'].values[i3], dev[i3],
                 fmt='o', mec='green', mfc='white', ecolor='green')
    ax4.plot(spec2['wave'].values, median_filter(dev, int(0.1 * len(dev))),
             c='red', zorder=100, ls='--')
    ax4.axhline(0, ls='--', c='grey', zorder=99)

    maxy = np.ceil(np.max(dev)).astype(int)
    ax4.set_xscale('log')
    ax4.set_xlim(0.58, 2.9)
    plt.xticks([0.6, 0.8, 1.0, 1.5, 2.0, 2.5],
               ['0.6', '0.8', '1.0', '1.5', '2.0', '2.5'])
    ax4.text(0.6, maxy - 0.25 * maxy,
             r'$\bar\sigma$={:.2f}'.format(np.sum(np.abs(dev)) / len(dev)))
    ax4.axhspan(-1, 1, color='grey', alpha=0.2)
    ax4.axhspan(-0.5, 0.5, color='grey', alpha=0.2)
    ax4.set_ylabel(r'$\Delta$ [ppm]', fontsize=14)
    ax4.set_xlabel('Wavelength [µm]', fontsize=14)

    gs.update(hspace=0.1)
    if title is not None:
        ax1.set_title(title, fontsize=18)
    plt.show()

    return dev


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

            if dimy == 96:
                ax.text(30, 80, '({0}, {1})'.format(i, g), c='white',
                        fontsize=12)
            else:
                ax.text(30, 230, '({0}, {1})'.format(i, g), c='white',
                        fontsize=12)
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
             r'$\chi_\nu^2 = {:.2f}$''\n'r'$\sigma={:.2f}$ppm''\n'r'$e={:.2f}$'.format(chi2_v, mean_err*1e6, err_mult),
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
            d_bin = data_detrended[trim_i:trim_e].reshape((nint-rem)//nbin,
                                                          nbin)
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
    hist = ax4.hist(res/scatter, edgecolor='grey', color='lightgrey',
                    bins=bins)
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

    fancyprint('Doing diagnostic plot 1.')
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
    plt.xticks(locs, labels, rotation=45)
    plt.ylabel('Differences [DN]', fontsize=12)
    plt.ylim(1.1*np.nanmin(old_med - np.nanmean(old_med)),
             1.1*np.nanmax(old_med - np.nanmean(old_med)))
    plt.legend()

    if outfile is not None:
        plt.savefig(outfile, bbox_inches='tight')
        fancyprint('Plot saved to {}'.format(outfile))
    if show_plot is False:
        plt.close()
    else:
        plt.show()


def make_linearity_plot2(results, old_results, outfile=None, show_plot=True):
    """Plot residuals to flat line before and after linearity correction.
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
    for i, file in enumerate(old_results):
        with utils.open_filetype(file) as datamodel:
            if i == 0:
                old_cube = datamodel.data
            else:
                old_cube = np.concatenate([old_cube, datamodel.data])

    nint, ngroup, dimy, dimx = np.shape(cube)
    # Get bright pixels in the trace.
    stack = bn.nanmedian(cube[:, -1], axis=0)
    ii = np.where((stack >= np.nanpercentile(stack, 80)) &
                  (stack < np.nanpercentile(stack, 99)))
    jj = np.random.randint(0, len(ii[0]), 1000)
    y = ii[0][jj]
    x = ii[1][jj]
    i = np.random.randint(0, nint, 1000)

    oold = np.zeros((1000, ngroup))
    nnew = np.zeros((1000, ngroup))
    for j in range(1000):
        o = old_cube[i[j], :, y[j], x[j]]
        ol = np.linspace(np.min(o), np.max(o), ngroup)
        oold[j] = (o - ol) / np.max(o) * 100
        n = cube[i[j], :, y[j], x[j]]
        nl = np.linspace(np.min(n), np.max(n), ngroup)
        nnew[j] = (n - nl) / np.max(n) * 100

    # Plot up mean group differences before and after linearity correction.
    plt.figure(figsize=(5, 3))
    plt.plot(np.arange(ngroup)+1, np.nanmedian(oold, axis=0),
             label='Before Correction', c='blue')
    plt.plot(np.arange(ngroup)+1, np.nanmedian(nnew, axis=0),
             label='After Correction', c='red')
    plt.axhline(0, ls='--', c='black')
    plt.xticks(np.arange(ngroup)+1, (np.arange(ngroup)+1).astype(str))
    plt.xlabel('Group Number', fontsize=12)
    plt.ylabel('Residual [%]', fontsize=12)
    plt.legend()

    if outfile is not None:
        plt.savefig(outfile, bbox_inches='tight')
        fancyprint('Plot saved to {}'.format(outfile))
    if show_plot is False:
        plt.close()
    else:
        plt.show()


def make_oneoverf_chromatic_plot(m_e, m_o, b_e, b_o, ngroup, outfile=None,
                                 show_plot=True):
    """Make plot of chromatic 1/f slope and intercept values.
    """

    fancyprint('Doing diagnostic plot 3.')

    to_plot = [m_e, m_o, b_e, b_o]
    for obj in to_plot:
        obj = np.array(obj)
        obj[np.isnan(obj)] = 0
    texts = ['m even', 'm odd', 'b even', 'b odd']

    fig = plt.figure(figsize=(8, 2*ngroup), facecolor='white')
    gs = GridSpec(ngroup+1, 4, width_ratios=[1, 1, 1, 1])

    for i in range(ngroup):
        for j, obj in enumerate(to_plot):
            ax = fig.add_subplot(gs[i, j])
            if j < 2:
                plt.imshow(obj[i], aspect='auto', origin='lower',
                           vmin=np.nanpercentile(obj[i], 5),
                           vmax=np.nanpercentile(obj[i], 95))
            else:
                plt.imshow(obj[i], aspect='auto', origin='lower',
                           vmin=np.nanpercentile(obj[i], 25),
                           vmax=np.nanpercentile(obj[i], 75))
            if j != 0:
                ax.yaxis.set_major_formatter(plt.NullFormatter())
            if i != ngroup-1:
                ax.xaxis.set_major_formatter(plt.NullFormatter())
            if i == ngroup-1:
                plt.xlabel('Spectral Pixel', fontsize=10)
            if i == 0:
                plt.title(texts[j], fontsize=12)
            if j == 0:
                plt.ylabel('Integration', fontsize=10)

    gs.update(hspace=0.1, wspace=0.1)
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
        timeseries = np.ones(np.shape(cube)[0])

    if np.ndim(cube) == 4:
        nint, ngroup, dimy, dimx = np.shape(cube)
        ints = np.random.randint(0, nint, 9)
        grps = np.random.randint(0, ngroup, 9)
        to_plot, to_write = [], []
        for i, g in zip(ints, grps):
            diff = cube[i, g] - deep[g] * timeseries[i]
            to_plot.append(diff)
            to_write.append('({0}, {1})'.format(i, g))
    else:
        nint, dimy, dimx = np.shape(cube)
        ints = np.random.randint(0, nint, 9)
        to_plot, to_write = [], []
        for i in ints:
            diff = cube[i] - deep * timeseries[i]
            to_plot.append(diff)
            to_write.append('({0})'.format(i))
    kwargs = {'vmin': np.nanpercentile(diff, 5),
              'vmax': np.nanpercentile(diff, 95)}
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
    mask_cube = None
    if pixel_masks is not None:
        for i, data in enumerate(pixel_masks):
            if i == 0:
                mask_cube = data
            else:
                mask_cube = np.concatenate([mask_cube, data])

    if np.ndim(cube) == 4:
        nints, ngroups, dimy, dimx = np.shape(cube)
    else:
        nints, dimy, dimx = np.shape(cube)
    baseline_ints = utils.format_out_frames(baseline_ints)
    old_deep = bn.nanmedian(old_cube[baseline_ints], axis=0)
    deep = bn.nanmedian(cube[baseline_ints], axis=0)

    # Mask trace.
    det = utils.get_detector_name(results[0])
    if det == 'nrs1' and not isinstance(results[0], datamodels.SlitModel):
        xstart = 500
    else:
        xstart = 0
    if np.ndim(cube) == 4:
        thisdeep = deep[:, -1]
    else:
        thisdeep = deep
    centroids = utils.get_centroids_nirspec(thisdeep, xstart=xstart,
                                            save_results=False)
    xpos, ypos = centroids[0], centroids[1]

    inst = utils.get_instrument_name(results[0])
    if inst == 'NIRISS':
        mask_width = 45
    else:
        mask_width = 16
    low = np.max([np.zeros_like(ypos),
                  ypos - mask_width / 2], axis=0).astype(int)
    up = np.min([dimy * np.ones_like(ypos),
                 ypos + mask_width / 2], axis=0).astype(int)
    tracemask = np.zeros((dimy, dimx))
    for i, x in enumerate(xpos):
        tracemask[low[i]:up[i], int(x)] = 1
    # Add the trace mask to the outliers cube.
    if mask_cube is not None:
        mask_cube = (mask_cube.astype(bool) | tracemask.astype(bool)).astype(int)
    else:
        mask_cube = np.repeat(tracemask[np.newaxis], nints, axis=0).astype(int)

    # Get smoothed light curve.
    if isinstance(timeseries, str):
        try:
            timeseries = np.load(timeseries)
        except (ValueError, FileNotFoundError):
            timeseries = None
    # If no lightcurve is provided, use array of ones.
    if timeseries is None:
        timeseries = np.ones(np.shape(cube)[0])

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


def make_photon_noise_plot(spectrum_files, ngroup, baseline_ints, order=1,
                           labels=None, tframe=5.494, gain=1.6):
    """Make plot comparing lightcurve precision to photon noise.
    """

    spectrum_files = np.atleast_1d(spectrum_files)
    base = utils.format_out_frames(baseline_ints)

    plt.figure(figsize=(7, 2))
    for j, spectrum_file in enumerate(spectrum_files):
        with fits.open(spectrum_file) as spectrum:
            if order == 1:
                spec = spectrum[3].data
            else:
                spec = spectrum[7].data
            spec *= tframe * gain * ngroup
            if order == 1:
                wave = np.mean([spectrum[1].data[0], spectrum[2].data[0]],
                               axis=0)
                ii = np.ones_like(wave)
            else:
                wave = np.mean([spectrum[5].data[0], spectrum[6].data[0]],
                               axis=0)
                ii = np.where((wave >= 0.6) & (wave < 0.85))[0]

        scatter = []
        for i in range(len(ii)):
            wlc = spec[:, i]
            noise = 0.5 * (wlc[0:-2] + wlc[2:]) - wlc[1:-1]
            noise = np.median(np.abs(noise))
            scatter.append(noise / np.median(wlc[base]))
        scatter = np.array(scatter)
        if labels is not None:
            label = labels[j]
        else:
            label = None
        plt.plot(wave, median_filter(scatter, 10) * 1e6, label=label)

    phot = np.sqrt(np.median(spec[base], axis=0)) / np.median(spec[base],
                                                              axis=0)
    plt.plot(wave, median_filter(phot, 10) * 1e6, c='black')
    plt.plot(wave, 2 * median_filter(phot, 10) * 1e6, c='black')

    plt.ylabel('Precision [ppm]', fontsize=14)

    if labels is not None:
        plt.legend(ncol=2)
    plt.show()


def make_soss_width_plot(scatter, min_width, outfile=None, show_plot=True):
    """Make plot showing optimization of extraction box.
    """

    plt.figure(figsize=(8, 5))
    plt.plot(np.linspace(10, 60, 51), scatter, c='royalblue')
    plt.scatter(np.linspace(10, 60, 51)[min_width], scatter[min_width],
                marker='*', c='red', s=100, zorder=2)

    plt.xlabel('Aperture Width', fontsize=14)
    plt.ylabel('Scatter', fontsize=14)
    plt.yticks(fontsize=10)
    plt.xticks(fontsize=10)

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
            cax = ax1.inset_axes((1.05, 0.005, 0.03, 0.99),
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
            cax = ax2.inset_axes((1.05, 0.005, 0.03, 0.99),
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
