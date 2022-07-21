#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thurs Jul 21 17:30 2022

@author: MCR

Custom JWST DMS pipeline steps for Stage 1 (detector level processing).
"""

from astropy.io import fits
import numpy as np
from tqdm import tqdm
import warnings

from jwst import datamodels

from supreme_spoon import utils


def oneoverfstep(datafiles, output_dir=None, save_results=True,
                 outlier_maps=None, trace_mask=None, use_dq=True):
    """Custom 1/f correction routine to be applied at the group level.

    Parameters
    ----------
    datafiles : list[str], or list[RampModel]
        List of paths to data files, or RampModels themselves for each segment
        of the TSO. Should be 4D ramps and not rate files.
    output_dir : str
        Directory to which to save results.
    save_results : bool
        If True, save results to disk.
    outlier_maps : list[str], None
        List of paths to outlier maps for each data segment. Can be
        3D (nints, dimy, dimx), or 2D (dimy, dimx) files.
    trace_mask : str, None
        Path to trace mask file. Should be 2D (dimy, dimx).
    use_dq : bool
        If True, also mask all pixels flagged in the DQ array.

    Returns
    -------
    corrected_rampmodels : list
        Ramp models for each segment corrected for 1/f noise.
    """

    print('Starting custom 1/f correction step.')

    # Output directory formatting.
    if output_dir is not None:
        if output_dir[-1] != '/':
            output_dir += '/'

    datafiles = np.atleast_1d(datafiles)
    # If outlier maps are passed, ensure that there is one for each segment.
    if outlier_maps is not None:
        outlier_maps = np.atleast_1d(outlier_maps)
        if len(outlier_maps) == 1:
            outlier_maps = [outlier_maps[0] for d in datafiles]

    data, fileroots = [], []
    # Load in datamodels from all segments.
    for i, file in enumerate(datafiles):
        if isinstance(file, str):
            currentfile = datamodels.open(file)
        else:
            currentfile = file
        data.append(currentfile)
        # Hack to get filename root.
        filename_split = currentfile.meta.filename.split('/')[-1].split('_')
        fileroot = ''
        for seg, segment in enumerate(filename_split):
            if seg == len(filename_split) - 1:
                break
            segment += '_'
            fileroot += segment
        fileroots.append(fileroot)

        # To create the deepstack, join all segments together.
        if i == 0:
            cube = currentfile.data
        else:
            cube = np.concatenate([cube, currentfile.data], axis=0)

    # Get total file root, with no segment info.
    working_name = fileroots[0]
    if 'seg' in working_name:
        parts = working_name.split('seg')
        part1, part2 = parts[0][:-1], parts[1][3:]
        fileroot_noseg = part1+part2
    else:
        fileroot_noseg = fileroots[0]

    # Generate the deep stack and rms of it. Both 3D (ngroup, dimy, dimx).
    print('Generating a deep stack for each frame using all integrations...')
    deepstack, rms = utils.make_deepstack(cube, return_rms=True)
    # Save these to disk if requested.
    if save_results is True:
        hdu = fits.PrimaryHDU(deepstack)
        hdu.writeto(output_dir+fileroot_noseg+'oneoverfstep_deepstack.fits',
                    overwrite=True)
        hdu = fits.PrimaryHDU(rms)
        hdu.writeto(output_dir+fileroot_noseg+'oneoverfstep_rms.fits',
                    overwrite=True)

    corrected_rampmodels = []
    for n, datamodel in enumerate(data):
        print('Starting segment {} of {}.'.format(n + 1, len(data)))

        # The readout setup
        ngroup = datamodel.meta.exposure.ngroups
        nint = np.shape(datamodel.data)[0]
        dimx = np.shape(datamodel.data)[-1]
        dimy = np.shape(datamodel.data)[-2]
        # Also open the data quality flags if requested.
        if use_dq is True:
            print(' Considering data quality flags.')
            dq = datamodel.groupdq
            # Mask will be applied multiplicatively.
            dq = np.where(dq == 0, 1, np.nan)
        else:
            dq = np.ones_like(datamodel.data)

        # Weighted average to determine the 1/f DC level
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            w = 1 / rms

        # Read in the outlier map -- a (nints, dimy, dimx) 3D cube
        if outlier_maps is None:
            msg = ' No outlier maps passed, ignoring outliers.'
            print(msg)
            outliers = np.zeros((nint, dimy, dimx))
        else:
            print(' Using outlier map {}'.format(outlier_maps[n]))
            outliers = fits.getdata(outlier_maps[n])
            # If the outlier map is 2D (dimy, dimx) extend to int dimension.
            if np.ndim(outliers) == 2:
                outliers = np.repeat(outliers, nint).reshape(dimy, dimx, nint)
                outliers = outliers.transpose(2, 0, 1)

        # Read in the trace mask -- a (dimy, dimx) data frame.
        if trace_mask is None:
            msg = ' No trace mask passed, ignoring the trace.'
            print(msg)
            tracemask = np.zeros((dimy, dimx))
        else:
            print(' Using trace mask {}'.format(trace_mask))
            tracemask = fits.getdata(trace_mask)

        # The outlier map is 0 where good and >0 otherwise. As this will be
        # applied multiplicatively, replace 0s with 1s and others with NaNs.
        outliers = np.where(outliers == 0, 1, np.nan)
        # Same thing with the trace mask.
        tracemask = np.where(tracemask == 0, 1, np.nan)
        tracemask = np.repeat(tracemask, nint).reshape(dimy, dimx, nint)
        tracemask = tracemask.transpose(2, 0, 1)
        # Combine the two masks.
        outliers = (outliers + tracemask) // 2

        dcmap = np.copy(datamodel.data)
        subcorr = np.copy(datamodel.data)
        sub, sub_m = np.copy(datamodel.data), np.copy(datamodel.data)
        for i in tqdm(range(nint)):
            # Create two difference images; one to be masked and one not.
            sub[i] = datamodel.data[i] - deepstack
            sub_m[i] = datamodel.data[i] - deepstack
            for g in range(ngroup):
                # Add in DQ mask.
                current_outlier = (outliers[i, :, :] + dq[i, g, :, :]) // 2
                # Apply the outlier mask.
                sub_m[i, g, :, :] *= current_outlier
                # Make sure to not subtract an overall bias
                sub[i, g, :, :] -= np.nanmedian(sub[i, g, :, :])
                sub_m[i, g, :, :] -= np.nanmedian(sub[i, g, :, :])
            if datamodel.meta.subarray.name == 'SUBSTRIP256':
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', category=RuntimeWarning)
                    dc = np.nansum(w * sub_m[i], axis=1)
                    dc /= np.nansum(w * current_outlier, axis=1)
                # make sure no NaN will corrupt the whole column
                dc = np.where(np.isfinite(dc), dc, 0)
                # dc is 2D - expand to the 3rd (columns) dimension
                dc3d = np.repeat(dc, 256).reshape((ngroup, 2048, 256))
                dcmap[i, :, :, :] = dc3d.swapaxes(1, 2)
                subcorr[i, :, :, :] = sub[i, :, :, :] - dcmap[i, :, :, :]
            elif datamodel.meta.subarray.name == 'SUBSTRIP96':
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', category=RuntimeWarning)
                    dc = np.nansum(w * sub_m[i], axis=1)
                    dc /= np.nansum(w * current_outlier, axis=1)
                # make sure no NaN will corrupt the whole column
                dc = np.where(np.isfinite(dc), dc, 0)
                # dc is 2D - expand to the 3rd (columns) dimension
                dc3d = np.repeat(dc, 256).reshape((ngroup, 2048, 256))
                dcmap[i, :, :, :] = dc3d.swapaxes(1, 2)
                subcorr[i, :, :, :] = sub[i, :, :, :] - dcmap[i, :, :, :]
            elif datamodel.meta.subarray.name == 'FULL':
                for amp in range(4):
                    yo = amp*512
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', category=RuntimeWarning)
                        dc = np.nansum(w[:, :, yo:yo+512, :] * sub_m[:, :, yo:yo+512, :], axis=1)
                        dc /= (np.nansum(w[:, :, yo:yo+512, :] * current_outlier, axis=1))
                    # make sure no NaN will corrupt the whole column
                    dc = np.where(np.isfinite(dc), dc, 0)
                    # dc is 2D - expand to the 3rd (columns) dimension
                    dc3d = np.repeat(dc, 512).reshape((ngroup, 2048, 512))
                    dcmap[i, :, yo:yo+512, :] = dc3d.swapaxes(1, 2)
                    subcorr[i, :, yo:yo+512, :] = sub[i, :, yo:yo+512, :] - dcmap[i, :, yo:yo+512, :]

        # Make sure no NaNs are in the DC map
        dcmap = np.where(np.isfinite(dcmap), dcmap, 0)

        # Subtract the DC map from a copy of the data model
        rampmodel_corr = datamodel.copy()
        rampmodel_corr.data = datamodel.data - dcmap

        # Save results to disk if requested.
        if save_results is True:
            hdu = fits.PrimaryHDU(sub)
            hdu.writeto(output_dir+fileroots[n]+'oneoverfstep_diffim.fits',
                        overwrite=True)
            hdu = fits.PrimaryHDU(subcorr)
            hdu.writeto(output_dir+fileroots[n]+'oneoverfstep_diffimcorr.fits',
                        overwrite=True)
            hdu = fits.PrimaryHDU(dcmap)
            hdu.writeto(output_dir+fileroots[n]+'oneoverfstep_noisemap.fits',
                        overwrite=True)
            corrected_rampmodels.append(rampmodel_corr)
            rampmodel_corr.write(output_dir+fileroots[n]+'oneoverfstep.fits')

        datamodel.close()

    return corrected_rampmodels