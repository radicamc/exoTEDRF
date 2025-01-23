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
import matplotlib.backends.backend_pdf
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter
import warnings

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

    # Open files -- assume before and after are same file type.
    if isinstance(before, str):
        bf = fits.getdata(before, 1)
        af = fits.getdata(after, 1)
    else:
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


def make_centroiding_plot(deepframe, centroids, outfile=None, show_plot=True):
    """Make plot showing results of centroiding.
    """

    dimy, dimx = np.shape(deepframe)

    plt.figure(figsize=(8, 3))
    plt.imshow(deepframe, aspect='auto', origin='lower', vmin=0,
               vmax=np.nanpercentile(deepframe, 80))

    for key in centroids.keys():
        if 'ypos' in key:
            plt.plot(centroids['xpos'], centroids[key], ls='--', c='red')

    plt.ylim(0, dimy - 1)
    plt.xlim(0, dimx - 1)
    plt.xlabel('X Pixel', fontsize=12)
    plt.ylabel('Y Pixel', fontsize=12)

    if outfile is not None:
        plt.savefig(outfile, bbox_inches='tight')
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
    files = np.random.randint(0, len(results), 9)

    # Plot the location of all jumps and hot pixels.
    plt.figure(figsize=(15, 9), facecolor='white')
    gs = GridSpec(3, 3)

    counter = 0
    for k in range(3):
        for j in range(3):
            thisfile = files[counter]
            counter += 1
            if isinstance(results[thisfile], str):
                cube = fits.getdata(results[thisfile], 1)
                pixeldq = fits.getdata(results[thisfile], 2)
                dqcube = fits.getdata(results[thisfile], 3)
            else:
                with utils.open_filetype(results[thisfile]) as datamodel:
                    cube = datamodel.data
                    pixeldq = datamodel.pixeldq
                    dqcube = datamodel.groupdq
            nint, ngroup, _, _ = np.shape(cube)

            # Get random group and integration.
            i = np.random.randint(0, nint)
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

            ax.text(30, 0.9 * np.shape(cube)[-2], '({0}, {1})'.format(i, g),
                    c='white', fontsize=12)
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


def make_linearity_plot(results, old_results, outfile=None, show_plot=True):
    """Plot group differences before and after linearity correction.
    """

    fancyprint('Doing diagnostic plot 1.')
    results = np.atleast_1d(results)
    old_results = np.atleast_1d(old_results)
    # Just use first segment for a quick look.
    if isinstance(results[0], str):
        cube = fits.getdata(results[0], 1)
        old_cube = fits.getdata(old_results[0], 1)
    else:
        with utils.open_filetype(results[0]) as datamodel:
            cube = datamodel.data
        with utils.open_filetype(old_results[0]) as datamodel:
            old_cube = datamodel.data

    nint, ngroup, dimy, dimx = np.shape(cube)

    # Get bright pixels in the trace.
    stack = bn.nanmedian(cube[np.random.randint(0, nint, 25), -1], axis=0)
    ii = np.where((stack >= np.nanpercentile(stack, 80)) &
                  (stack < np.nanpercentile(stack, 99)))

    # Compute group differences in these pixels.
    new_diffs = np.zeros((ngroup-1, len(ii[0])))
    old_diffs = np.zeros((ngroup-1, len(ii[0])))
    num_pix = 10000
    if len(ii[0]) < 10000:
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
    # Just use first segment for a quick look.
    if isinstance(results[0], str):
        cube = fits.getdata(results[0], 1)
        old_cube = fits.getdata(old_results[0], 1)
    else:
        with utils.open_filetype(results[0]) as datamodel:
            cube = datamodel.data
        with utils.open_filetype(old_results[0]) as datamodel:
            old_cube = datamodel.data

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
    texts = ['m even', 'm odd', 'b even', 'b odd']

    fig = plt.figure(figsize=(8, 2*ngroup), facecolor='white')
    gs = GridSpec(ngroup+1, 4, width_ratios=[1, 1, 1, 1])

    for i in range(ngroup):
        for j, obj in enumerate(to_plot):
            obj[obj == 0] = np.nan
            if ngroup == 1:
                nint, dimx = np.shape(obj)
                obj = np.reshape(obj, (nint, 1, dimx))
            ax = fig.add_subplot(gs[i, j])
            if j < 2:
                plt.imshow(obj[:, i], aspect='auto', origin='lower',
                           vmin=np.nanpercentile(obj[:, i], 5),
                           vmax=np.nanpercentile(obj[:, i], 95))
            else:
                plt.imshow(obj[:, i], aspect='auto', origin='lower',
                           vmin=np.nanpercentile(obj[:, i], 25),
                           vmax=np.nanpercentile(obj[:, i], 75))
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


def make_oneoverf_plot(results, deepstack, timeseries=None, outfile=None,
                       show_plot=True):
    """make nine-panel plot of dataframes after 1/f correction.
    """

    fancyprint('Doing diagnostic plot 1.')
    results = np.atleast_1d(results)
    # Format baseline frame integrations.
    if isinstance(results[0], str):
        nints = fits.getheader(results[0], 0)['NINTS']
    else:
        with utils.open_filetype(results[0]) as currentfile:
            nints = currentfile.meta.exposure.nints

    # Get smoothed light curve.
    if isinstance(timeseries, str):
        try:
            timeseries = np.load(timeseries)
        except (ValueError, FileNotFoundError):
            timeseries = None
    # If no lightcurve is provided, use array of ones.
    if timeseries is None:
        timeseries = np.ones(nints)

    to_plot, to_write = [], []
    files = np.random.randint(0, len(results), 9)
    for f in files:
        thisfile = results[f]
        if isinstance(thisfile, str):
            cube = fits.getdata(thisfile, 1)
            ngroup = fits.getheader(thisfile, 0)['NGROUPS']
            istart = fits.getheader(thisfile, 0)['INTSTART']
            thisi = fits.getheader(thisfile, 0)['INTEND'] - istart
        else:
            cube = thisfile.data
            ngroup = thisfile.meta.exposure.ngroups
            istart = thisfile.meta.exposure.integration_start
            thisi = thisfile.meta.exposure.integration_end - istart

        i = np.random.randint(0, thisi)
        if np.ndim(cube) == 4:
            g = np.random.randint(0, ngroup)
            diff = cube[i, g] - deepstack[g] * timeseries[i+istart]
            to_plot.append(diff)
            to_write.append('({0}, {1})'.format(i+istart, g))
        else:
            diff = cube[i] - deepstack * timeseries[i+istart]
            to_plot.append(diff)
            to_write.append('({0})'.format(i+istart))

    kwargs = {'vmin': np.nanpercentile(diff, 5),
              'vmax': np.nanpercentile(diff, 95)}
    nine_panel_plot(to_plot, to_write, outfile=outfile, show_plot=show_plot,
                    **kwargs)
    if outfile is not None:
        fancyprint('Plot saved to {}'.format(outfile))


def make_oneoverf_psd(results, old_results, deepstack, old_deepstack,
                      timeseries, nsample=10,  pixel_masks=None, tframe=5.494,
                      tpix=1e-5, tgap=1.2e-4, outfile=None, show_plot=True):
    """Make a PSD plot to see PSD of background before and after 1/f removal.
    """

    fancyprint('Doing diagnostic plot 2.')

    results = np.atleast_1d(results)
    old_results = np.atleast_1d(old_results)

    # Get smoothed light curve.
    if isinstance(timeseries, str):
        try:
            timeseries = np.load(timeseries)
        except (ValueError, FileNotFoundError):
            timeseries = None
    # If no lightcurve is provided, use array of ones.
    if timeseries is None:
        if isinstance(results[0], str):
            nints = fits.getheader(results[0], 0)['NINTS']
        else:
            with utils.open_filetype(results[0]) as currentfile:
                nints = currentfile.meta.exposure.nints
        timeseries = np.ones(nints)

    # Generate psd frequency array
    freqs = np.logspace(np.log10(1 / tframe), np.log10(1 / tpix), 100)
    pwr_old = np.zeros((nsample, len(freqs)))
    pwr = np.zeros((nsample, len(freqs)))

    # Select nsample random frames and compare PSDs before and after 1/f
    # removal.
    files = np.random.randint(0, len(results), nsample)
    for s, f in enumerate(files):
        thisfile = results[f]
        oldfile = old_results[f]
        if isinstance(thisfile, str):
            cube = fits.getdata(thisfile, 1)
            ngroup = fits.getheader(thisfile, 0)['NGROUPS']
            istart = fits.getheader(thisfile, 0)['INTSTART']
            thisi = fits.getheader(thisfile, 0)['INTEND'] - istart
            old_cube = fits.getdata(oldfile, 1)
        else:
            cube = thisfile.data
            ngroup = thisfile.meta.exposure.ngroups
            istart = thisfile.meta.exposure.integration_start
            thisi = thisfile.meta.exposure.integration_end - istart
            old_cube = oldfile.data
        i = np.random.randint(0, thisi)

        # Get data frame dimensions.
        if np.ndim(cube) == 4:
            _, _, dimy, dimx = np.shape(cube)
        else:
            _, dimy, dimx = np.shape(cube)

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

        # Bad pixel maps
        if pixel_masks is not None:
            mask_cube = pixel_masks[f]
        else:
            if np.ndim(cube) == 4:
                mask_cube = np.zeros_like(cube[:, 0])
            else:
                mask_cube = np.zeros_like(cube)

        if np.ndim(cube) == 4:
            g = np.random.randint(0, ngroup)
            # Get difference images before and after 1/f removal.
            diff_old = (old_cube[i, g] - old_deepstack[g] * timeseries[i+istart]).flatten('F')[::-1]
            diff = (cube[i, g] - deepstack[g] * timeseries[i+istart]).flatten('F')[::-1]
        else:
            # Get difference images before and after 1/f removal.
            diff_old = (old_cube[i] - old_deepstack * timeseries[i+istart]).flatten('F')[::-1]
            diff = (cube[i] - deepstack * timeseries[i+istart]).flatten('F')[::-1]

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


def make_superbias_scale_plot(scale_factors, outfile=None, show_plot=True):
    """ Make a plot showing the custom superbias scale factors.
    """

    plt.plot(scale_factors)
    plt.xlabel('Integration No.', fontsize=12)
    plt.ylabel('Scale Factor', fontsize=12)

    if outfile is not None:
        plt.savefig(outfile, bbox_inches='tight')
        fancyprint('Plot saved to {}'.format(outfile))
    if show_plot is False:
        plt.close()
    else:
        plt.show()


def make_2d_lightcurve_plot(wave1, flux1, wave2=None, flux2=None, outpdf=None,
                            title='', instrument='NIRISS', **kwargs):
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
        if instrument.upper() == 'NIRISS':
            plt.title('Order 1' + title, fontsize=18)
        elif instrument.upper() == 'NIRSPEC':
            plt.title('NRS1' + title, fontsize=18)
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
            if instrument.upper() == 'NIRISS':
                plt.title('Order 2' + title, fontsize=18)
            elif instrument.upper() == 'NIRSPEC':
                plt.title('NRS2' + title, fontsize=18)
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
    files = np.random.randint(0, len(results), 9)

    to_plot, to_write = [], []
    for f in files:
        thisfile = results[f]
        if isinstance(thisfile, str):
            cube = fits.getdata(thisfile, 1)
            ngroup = fits.getheader(thisfile, 0)['NGROUPS']
            istart = fits.getheader(thisfile, 0)['INTSTART']
            thisi = fits.getheader(thisfile, 0)['INTEND'] - istart
        else:
            cube = thisfile.data
            ngroup = thisfile.meta.exposure.ngroups
            istart = thisfile.meta.exposure.integration_start
            thisi = thisfile.meta.exposure.integration_end - istart

        i = np.random.randint(0, thisi)
        if np.ndim(cube) == 4:
            g = np.random.randint(0, ngroup)
            to_plot.append(cube[i, g])
            to_write.append('({0}, {1})'.format(i+istart, g))
        else:
            to_plot.append(cube[i])
            to_write.append('({0})'.format(i+istart))

    nine_panel_plot(to_plot, to_write, outfile=outfile, show_plot=show_plot,
                    **kwargs)
    if outfile is not None:
        fancyprint('Plot saved to {}'.format(outfile))


def plot_quicklook_lightcurve(datafiles):
    """Quick and dirty light curve plot, mostly for validation of
    observations.
    """

    datafiles = np.atleast_1d(datafiles)

    # Stack the TSO into a cube.
    for i, file in enumerate(datafiles):
        fancyprint('Reading file {}.'.format(file))
        if i == 0:
            cube = fits.getdata(file)[:, -1]
        else:
            cube = np.concatenate([cube, fits.getdata(file)[:, -1]])

    # Quick sum of flux near throughput peak.
    instrument = utils.get_instrument_name(datafiles[0])
    if instrument == 'NIRSPEC':
        det = utils.get_detector_name(datafiles[0])
        if det == 'NRS1':
            postage = cube[:, 12:17, 900:1500]
        else:
            postage = cube[:, 6:10, :500]
    else:
        postage = cube[:, 20:60, 1500:1550]
    timeseries = np.nansum(postage, axis=(1, 2))

    # Make plot.
    plt.figure(figsize=(6, 4))
    plt.errorbar(np.arange(len(timeseries)),
                 timeseries / np.nanmedian(timeseries[:20]),
                 fmt='o', mfc='white', mec='royalblue', ms=3)
    plt.xlabel('Integration No.', fontsize=12)
    plt.ylabel('Normalized Flux', fontsize=12)
    plt.show()


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
