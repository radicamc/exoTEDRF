#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thurs Jul 21 17:33 2022

@author: MCR

Custom JWST DMS pipeline steps for Stage 2 (Spectroscopic processing).
"""

from astropy.io import fits
import bottleneck as bn
import glob
import more_itertools as mit
import numpy as np
import os
import pandas as pd
from sklearn.decomposition import PCA
from scipy.ndimage import median_filter
from scipy.signal import medfilt
from tqdm import tqdm
import warnings

from jwst import datamodels
from jwst.extract_1d.soss_extract import soss_boxextract
from jwst.pipeline import calwebb_spec2

from supreme_spoon import utils, plotting
from supreme_spoon.utils import fancyprint
from supreme_spoon.stage3 import do_box_extraction


class AssignWCSStep:
    """Wrapper around default calwebb_spec2 Assign WCS step.
    """

    def __init__(self, datafiles, output_dir='./'):
        """Step initializer.
        """

        self.tag = 'assignwcsstep.fits'
        self.output_dir = output_dir
        self.datafiles = np.atleast_1d(datafiles)
        self.fileroots = utils.get_filename_root(self.datafiles)

    def run(self, save_results=True, force_redo=False, **kwargs):
        """Method to run the step.
        """

        results = []
        all_files = glob.glob(self.output_dir + '*')
        for i, segment in enumerate(self.datafiles):
            # If an output file for this segment already exists, skip the step.
            expected_file = self.output_dir + self.fileroots[i] + self.tag
            if expected_file in all_files and force_redo is False:
                fancyprint('File {} already exists.'.format(expected_file))
                fancyprint('Skipping Assign WCS Step.')
                res = expected_file
            # If no output files are detected, run the step.
            else:
                step = calwebb_spec2.assign_wcs_step.AssignWcsStep()
                res = step.call(segment, output_dir=self.output_dir,
                                save_results=save_results, **kwargs)
                # Verify that filename is correct.
                if save_results is True:
                    current_name = self.output_dir + res.meta.filename
                    if expected_file != current_name:
                        res.close()
                        os.rename(current_name, expected_file)
                        res = datamodels.open(expected_file)
            results.append(res)

        return results


class SourceTypeStep:
    """Wrapper around default calwebb_spec2 Source Type Determination step.
    """

    def __init__(self, datafiles, output_dir='./'):
        """Step initializer.
        """

        self.tag = 'sourcetypestep.fits'
        self.output_dir = output_dir
        self.datafiles = np.atleast_1d(datafiles)
        self.fileroots = utils.get_filename_root(self.datafiles)

    def run(self, save_results=True, force_redo=False, **kwargs):
        """Method to run the step.
        """

        results = []
        all_files = glob.glob(self.output_dir + '*')
        for i, segment in enumerate(self.datafiles):
            # If an output file for this segment already exists, skip the step.
            expected_file = self.output_dir + self.fileroots[i] + self.tag
            if expected_file in all_files and force_redo is False:
                fancyprint('File {} already exists.'.format(expected_file))
                fancyprint('Skipping Source Type Determination Step.')
                res = expected_file
            # If no output files are detected, run the step.
            else:
                step = calwebb_spec2.srctype_step.SourceTypeStep()
                res = step.call(segment, output_dir=self.output_dir,
                                save_results=save_results, **kwargs)
                # Verify that filename is correct.
                if save_results is True:
                    current_name = self.output_dir + res.meta.filename
                    if expected_file != current_name:
                        res.close()
                        os.rename(current_name, expected_file)
                        res = datamodels.open(expected_file)
            results.append(res)

        return results


class BackgroundStep:
    """Wrapper around custom Background Subtraction step.
    """

    def __init__(self, input_data, background_model, output_dir='./'):
        """Step initializer.
        """

        self.tag = 'backgroundstep.fits'
        self.background_model = background_model
        self.output_dir = output_dir
        self.datafiles = utils.sort_datamodels(input_data)
        self.fileroots = utils.get_filename_root(self.datafiles)
        self.fileroot_noseg = utils.get_filename_root_noseg(self.fileroots)

    def run(self, save_results=True, force_redo=False, do_plot=False,
            show_plot=False, **kwargs):
        """Method to run the step.
        """

        all_files = glob.glob(self.output_dir + '*')
        do_step = 1
        results, background_models = [], []
        for i in range(len(self.datafiles)):
            # If an output file for this segment already exists, skip the step.
            expected_file = self.output_dir + self.fileroots[i] + self.tag
            expected_bkg = self.output_dir + self.fileroot_noseg + 'background.npy'
            if expected_file not in all_files or expected_bkg not in all_files:
                do_step = 0
                break
            else:
                results.append(expected_file)
                background_models.append(expected_bkg)
        if do_step == 1 and force_redo is False:
            fancyprint('Output files already exist.')
            fancyprint('Skipping Background Subtraction Step.')
        # If no output files are detected, run the step.
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                scale, background_coords = None, None
                if 'scale' in kwargs.keys():
                    scale = kwargs['scale']
                if 'background_coords' in kwargs.keys():
                    background_coords = kwargs['background_coords']
                step_results = backgroundstep(self.datafiles,
                                              self.background_model,
                                              output_dir=self.output_dir,
                                              save_results=save_results,
                                              fileroots=self.fileroots,
                                              fileroot_noseg=self.fileroot_noseg,
                                              scale=scale,
                                              background_coords=background_coords)
                results, background_models = step_results

            # Do step plot if requested.
            if do_plot is True:
                if save_results is True:
                    plot_file = self.output_dir + self.tag.replace('fits',
                                                                   'pdf')
                else:
                    plot_file = None
                plotting.make_background_plot(results, outfile=plot_file,
                                              show_plot=show_plot)

        return results, background_models


class FlatFieldStep:
    """Wrapper around default calwebb_spec2 Flat Field Correction step.
    """

    def __init__(self, datafiles, output_dir='./'):
        """Step initializer.
        """

        self.tag = 'flatfieldstep.fits'
        self.output_dir = output_dir
        self.datafiles = np.atleast_1d(datafiles)
        self.fileroots = utils.get_filename_root(self.datafiles)

    def run(self, save_results=True, force_redo=False, **kwargs):
        """Method to run the step.
        """

        results = []
        all_files = glob.glob(self.output_dir + '*')
        for i, segment in enumerate(self.datafiles):
            # If an output file for this segment already exists, skip the step.
            expected_file = self.output_dir + self.fileroots[i] + self.tag
            if expected_file in all_files and force_redo is False:
                fancyprint('File {} already exists.'.format(expected_file))
                fancyprint('Skipping Flat Field Correction Step.')
                res = expected_file
            # If no output files are detected, run the step.
            else:
                step = calwebb_spec2.flat_field_step.FlatFieldStep()
                res = step.call(segment, output_dir=self.output_dir,
                                save_results=save_results, **kwargs)
                # Verify that filename is correct.
                if save_results is True:
                    current_name = self.output_dir + res.meta.filename
                    if expected_file != current_name:
                        res.close()
                        os.rename(current_name, expected_file)
                        res = datamodels.open(expected_file)
            results.append(res)

        return results


class BadPixStep:
    """Wrapper around custom Bad Pixel Correction Step.
    """

    def __init__(self, input_data, baseline_ints, output_dir='./'):
        """Step initializer.
        """

        self.tag = 'badpixstep.fits'
        self.output_dir = output_dir
        self.baseline_ints = baseline_ints
        self.datafiles = utils.sort_datamodels(input_data)
        self.fileroots = utils.get_filename_root(self.datafiles)
        self.fileroot_noseg = utils.get_filename_root_noseg(self.fileroots)

    def run(self, space_thresh=15, time_thresh=10, box_size=5,
            save_results=True, force_redo=False, do_plot=False,
            show_plot=False):
        """Method to run the step.
        """

        all_files = glob.glob(self.output_dir + '*')
        do_step = 1
        results = []
        for i in range(len(self.datafiles)):
            # If an output file for this segment already exists, skip the step.
            expected_file = self.output_dir + self.fileroots[i] + self.tag
            expected_deep = self.output_dir + self.fileroot_noseg + 'deepframe.fits'
            if expected_file not in all_files or expected_deep not in all_files:
                do_step = 0
                break
            else:
                results.append(datamodels.open(expected_file))
                deepframe = fits.getdata(expected_deep, 1)
        if do_step == 1 and force_redo is False:
            fancyprint('Output files already exist.')
            fancyprint('Skipping Bad Pixel Correction Step.')
        # If no output files are detected, run the step.
        else:
            step_results = badpixstep(self.datafiles,
                                      baseline_ints=self.baseline_ints,
                                      output_dir=self.output_dir,
                                      save_results=save_results,
                                      fileroots=self.fileroots,
                                      fileroot_noseg=self.fileroot_noseg,
                                      space_thresh=space_thresh,
                                      time_thresh=time_thresh,
                                      box_size=box_size,
                                      do_plot=do_plot, show_plot=show_plot)
            results, deepframe = step_results

        return results, deepframe


class TracingStep:
    """Wrapper around custom Tracing Step.
    """

    def __init__(self, input_data, deepframe, output_dir='./'):
        """Step initializer.
        """

        self.output_dir = output_dir
        self.deepframe = deepframe
        self.datafiles = utils.sort_datamodels(input_data)
        self.fileroots = utils.get_filename_root(self.datafiles)
        self.fileroot_noseg = utils.get_filename_root_noseg(self.fileroots)
        self.instrument = utils.get_instrument_name(self.datafiles[0])

    def run(self, generate_tracemask=True, mask_width=45, pixel_flags=None,
            generate_order0_mask=True, f277w=None,
            calculate_stability=True, pca_components=10,
            save_results=True, force_redo=False, generate_lc=True,
            baseline_ints=None, smoothing_scale=None, do_plot=False,
            show_plot=False):
        """Method to run the step.
        """

        all_files = glob.glob(self.output_dir + '*')
        # If an output file for this segment already exists, skip the step.
        suffix = 'centroids.csv'
        expected_file = self.output_dir + self.fileroot_noseg + suffix
        if expected_file in all_files and force_redo is False:
            fancyprint('Main output file already exists.')
            fancyprint('If you wish to still produce secondary outputs, run '
                       'with force_redo=True.')
            fancyprint('Skipping Tracing Step.')
            centroids = pd.read_csv(expected_file, comment='#')
            tracemask, order0mask, smoothed_lc = None, None, None
        # If no output files are detected, run the step.
        else:
            # Attempt to find pixel mask files if none were passed.
            if pixel_flags is None and generate_tracemask is True:
                new_pixel_flags = []
                s1_dir = self.output_dir.replace('Stage2', 'Stage1')
                stage1_files = glob.glob(s1_dir + '*')
                for root in self.fileroots:
                    expected_file = s1_dir + root + 'pixelflags.fits'
                    if expected_file in stage1_files:
                        new_pixel_flags.append(expected_file)
                # If some files were found, pass them to the function call.
                if len(new_pixel_flags) > 0:
                    pixel_flags = new_pixel_flags

            if self.instrument == 'NIRISS':
                step_res = tracingstep_soss(self.datafiles, self.deepframe,
                                            calculate_stability=calculate_stability,
                                            pca_components=pca_components,
                                            generate_tracemask=generate_tracemask,
                                            mask_width=mask_width,
                                            pixel_flags=pixel_flags,
                                            generate_order0_mask=generate_order0_mask,
                                            f277w=f277w,
                                            generate_lc=generate_lc,
                                            baseline_ints=baseline_ints,
                                            smoothing_scale=smoothing_scale,
                                            output_dir=self.output_dir,
                                            save_results=save_results,
                                            fileroot_noseg=self.fileroot_noseg,
                                            do_plot=do_plot,
                                            show_plot=show_plot)
                centroids, tracemask, order0mask, smoothed_lc = step_res
            else:
                step_res = tracingstep_nirspec(self.datafiles, self.deepframe,
                                               calculate_stability=calculate_stability,
                                               pca_components=pca_components,
                                               generate_tracemask=generate_tracemask,
                                               mask_width=mask_width,
                                               pixel_flags=pixel_flags,
                                               generate_lc=generate_lc,
                                               baseline_ints=baseline_ints,
                                               smoothing_scale=smoothing_scale,
                                               output_dir=self.output_dir,
                                               save_results=save_results,
                                               fileroot_noseg=self.fileroot_noseg,
                                               do_plot=do_plot,
                                               show_plot=show_plot)
                centroids, tracemask, smoothed_lc = step_res
                order0mask = None

        return centroids, tracemask, order0mask, smoothed_lc


def backgroundstep(datafiles, background_model, output_dir='./',
                   save_results=True, fileroots=None, fileroot_noseg='',
                   scale=None, background_coords=None):
    """Background subtraction must be carefully treated with SOSS observations.
    Due to the extent of the PSF wings, there are very few, if any,
    non-illuminated pixels to serve as a sky region. Furthermore, the zodi
    background has a unique stepped shape, which would render a constant
    background subtraction ill-advised. Therefore, a background subtracton is
    performed by scaling a model background to the counts level of a median
    stack of the exposure. This scaled model background is then subtracted
    from each integration.

    Parameters
    ----------
    datafiles : array-like[str], array-like[CubeModel]
        Paths to data segments for a SOSS exposure, or the datamodels
        themselves.
    background_model : array-like[float]
        Background model. Should be 2D (dimy, dimx)
    output_dir : str
        Directory to which to save outputs.
    save_results : bool
        If True, save outputs to file.
    fileroots : array-like[str]
        Root names for output files.
    fileroot_noseg : str
        Root name with no segment information.
    scale : float, array-like[float], None
        Scaling value to apply to background model to match data. Will take
        precedence over calculated scaling value. If applied at group level,
        length of scaling array must equal ngroup.
    background_coords : array-like[int], None
        Region of frame to use to estimate the background. Must be 1D:
        [x_low, x_up, y_low, y_up].

    Returns
    -------
    results : array-like[CubeModel]
        Input data segments, corrected for the background.
    model_scaled : array-like[float]
        Background model, scaled to the flux level of each group median.
    """

    fancyprint('Starting background subtraction step.')
    # Output directory formatting.
    if output_dir is not None:
        if output_dir[-1] != '/':
            output_dir += '/'

    datafiles = np.atleast_1d(datafiles)
    # Load in each of the datafiles.
    for i, file in enumerate(datafiles):
        with utils.open_filetype(file) as currentfile:
            # To create the deepstack, join all segments together.
            if i == 0:
                cube = currentfile.data
            else:
                cube = np.concatenate([cube, currentfile.data], axis=0)

    # Make median stack of all integrations to use for background scaling.
    # This is to limit the influence of cosmic rays, which can greatly effect
    # the background scaling factor calculated for an individual inegration.
    fancyprint('Generating a median stack using all integrations.')
    stack = utils.make_deepstack(cube)
    # If applied at the integration level, reshape median stack to 3D.
    if np.ndim(stack) != 3:
        dimy, dimx = np.shape(stack)
        stack = stack.reshape(1, dimy, dimx)
    ngroup, dimy, dimx = np.shape(stack)
    # Ensure if user-defined scalings are provided that there is one per group.
    if scale is not None:
        scale = np.atleast_1d(scale)
        assert len(scale) == ngroup

    fancyprint('Calculating background model scaling.')
    model_scaled = np.zeros_like(stack)
    first_time = True
    for i in range(ngroup):
        if scale is None:
            if background_coords is None:
                # If region to estimate background is not provided, use a
                # default region.
                if dimy == 96:
                    # Use area in bottom left corner for SUBSTRIP96.
                    xl, xu = 5, 21
                    yl, yu = 5, 401
                else:
                    # Use area in the top left corner for SUBSTRIP256
                    xl, xu = 210, 250
                    yl, yu = 250, 500
            else:
                # Use user-defined background scaling region.
                assert len(background_coords) == 4
                # Convert to int if not already.
                background_coords = np.array(background_coords).astype(int)
                xl, xu, yl, yu = background_coords
            bkg_ratio = stack[i, xl:xu, yl:yu] / background_model[xl:xu, yl:yu]
            # Instead of a straight median, use the median of the 2nd quartile
            # to limit the effect of any remaining illuminated pixels.
            q1 = np.nanpercentile(bkg_ratio, 25)
            q2 = np.nanpercentile(bkg_ratio, 50)
            ii = np.where((bkg_ratio > q1) & (bkg_ratio < q2))
            scale_factor = np.nanmedian(bkg_ratio[ii])
            if scale_factor < 0:
                scale_factor = 0
            fancyprint('Using calculated background scale factor: '
                       '{:.5f}'.format(scale_factor))
            model_scaled[i] = background_model * scale_factor
        else:
            # If using a user specified scaling for the whole frame.
            fancyprint('Using user-defined background scaling: '
                       '{:.5f}'.format(scale[i]))
            model_scaled[i] = background_model * scale[i]

    # Loop over all segments in the exposure and subtract the background from
    # each of them.
    results = []
    for i, file in enumerate(datafiles):
        with utils.open_filetype(file) as currentfile:
            # Subtract the scaled background model.
            data_backsub = currentfile.data - model_scaled
            currentfile.data = data_backsub

            # Save the results to file if requested.
            if save_results is True:
                if first_time is True:
                    # Scaled model background.
                    np.save(output_dir + fileroot_noseg + 'background.npy',
                            model_scaled)
                    first_time = False
                # Background subtracted data.
                currentfile.write(output_dir + fileroots[i] + 'backgroundstep.fits')

            results.append(currentfile)

    return results, model_scaled


def badpixstep(datafiles, baseline_ints, space_thresh=15, time_thresh=10,
               box_size=5, output_dir='./', save_results=True,
               fileroots=None, fileroot_noseg='', do_plot=False,
               show_plot=False):
    """Identify and correct outlier pixels remaining in the dataset, using
    both a spatial and temporal approach. First, find spatial outlier pixels
    in the median stack and correct them in each integration via the median of
    a box of surrounding pixels. Then flag outlier pixels in the temporal
    direction and again replace with the surrounding median in time.

    Parameters
    ----------
    datafiles : array-like[str], array-like[RampModel]
        List of paths to datafiles for each segment, or the datamodels
        themselves.
    baseline_ints : array-like[int]
        Integrations of ingress and egress.
    space_thresh : int
        Sigma threshold for a deviant pixel to be flagged spatially.
    time_thresh : int
        Sigma threshold for a deviant pixel to be flagged temporally.
    box_size : int
        Size of box around each pixel to test for deviations.
    output_dir : str
        Directory to which to output results.
    save_results : bool
        If True, save results to file.
    fileroots : array-like[str], None
        Root names for output files.
    fileroot_noseg : str
        Root file name with no segment information.
    do_plot : bool
        If True, do the step diagnostic plot.
    show_plot : bool
        If True, show the step diagnostic plot instead of/in addition to
        saving it to file.

    Returns
    -------
    data : list[CubeModel]
        Input datamodels for each segment, corrected for outlier pixels.
    deepframe : array-like[float]
        Final median stack of all outlier corrected integrations.
    """

    fancyprint('Starting outlier pixel interpolation step.')

    # Output directory formatting.
    if output_dir is not None:
        if output_dir[-1] != '/':
            output_dir += '/'

    datafiles = np.atleast_1d(datafiles)
    # Format the baseline frames.
    baseline_ints = utils.format_out_frames(baseline_ints)

    # Load in datamodels from all segments.
    for i, file in enumerate(datafiles):
        with utils.open_filetype(file) as currentfile:
            # To create the deepstack, join all segments together.
            # Also stack all the dq arrays from each segement.
            if i == 0:
                cube = currentfile.data
                err_cube = currentfile.err
                dq_cube = currentfile.dq
            else:
                cube = np.concatenate([cube, currentfile.data])
                err_cube = np.concatenate([err_cube, currentfile.err])
                dq_cube = np.concatenate([dq_cube, currentfile.dq])

    # Initialize starting loop variables.
    newdata = np.copy(cube)
    newdq = np.copy(dq_cube)

    # ===== Spatial Outlier Flagging ======
    fancyprint('Starting spatial outlier flagging...')
    # Generate the deepstack.
    deepframe_itl = utils.make_deepstack(newdata[baseline_ints])

    # Get locations of all hot pixels.
    hot_pix = utils.get_dq_flag_metrics(dq_cube[0], ['DO_NOT_USE', 'HOT',
                                                     'WARM'])

    hotpix = np.zeros_like(deepframe_itl)
    otherpix = np.zeros_like(deepframe_itl)
    hot, other = 0, 0
    nint, dimy, dimx = np.shape(newdata)

    # Set all nans/negatives to zero.
    newdata[newdata < 0] = 0
    newdata[np.isnan(newdata)] = 0

    # Loop over whole deepstack and flag deviant pixels.
    for i in tqdm(range(4, dimx - 4)):
        # TODO: Set to dimy for NIRSpec
        for j in range(dimy - 4):
            # If the pixel is known to be hot, add it to list to interpolate.
            if hot_pix[j, i]:
                hotpix[j, i] = 1
                hot += 1
            # If not already flagged, double check that the pixel isn't
            # deviant in some other manner.
            else:
                box_size_i = box_size
                box_prop = utils.get_interp_box(deepframe_itl, box_size_i,
                                                i, j, start=4, end=dimx-4)
                # Ensure that the median and std dev extracted are good.
                # If not, increase the box size until they are.
                while np.any(np.isnan(box_prop)):
                    box_size_i += 1
                    box_prop = utils.get_interp_box(deepframe_itl, box_size_i,
                                                    i, j, start=4, end=dimx-4)
                med, std = box_prop[0], box_prop[1]

                # If central pixel is too deviant flag it.
                if np.abs(deepframe_itl[j, i] - med) >= (space_thresh * std):
                    otherpix[j, i] = 1
                    other += 1

    # Combine all flagged pixel maps.
    badpix = hotpix.astype(bool) | otherpix.astype(bool)
    badpix = badpix.astype(int)

    fancyprint('{0} hot and {1} deviant pixels identified.'.format(hot, other))
    # Replace the flagged pixels in each integration.
    fancyprint('Doing pixel replacement...')
    for i in tqdm(range(nint)):
        newdata[i], thisdq = utils.do_replacement(newdata[i], badpix,
                                                  dq=np.ones_like(newdata[i]),
                                                  box_size=box_size, start=4,
                                                  end=dimx-4)
        # Set DQ flags for these pixels to zero (use the pixel).
        thisdq = ~thisdq.astype(bool)
        newdq[:, thisdq] = 0

    # ===== Temporal Outlier Flagging =====
    fancyprint('Starting temporal outlier flagging...')
    # Median filter the data.
    cube_filt = medfilt(newdata, (5, 1, 1))

    # Check along the time axis for outlier pixels.
    std_dev = np.median(np.abs(0.5*(newdata[0:-2] + newdata[2:]) - newdata[1:-1]), axis=0)
    std_dev = np.where(std_dev == 0, np.inf, std_dev)
    scale = np.abs(newdata - cube_filt) / std_dev
    # Filter out some noise.
    # TODO: swap to 50 for NIRSpec
    ii = np.where((scale > time_thresh) & (cube_filt > np.nanpercentile(newdata, 25)))
    fancyprint('{} outliers detected.'.format(len(ii[0])))
    # Replace the flagged pixels in each integration.
    fancyprint('Doing pixel replacement...')
    newdata[ii] = cube_filt[ii]
    newdq[ii] = 0

    # Lastly, do a final check for any remaining invalid flux or error values.
    ii = np.where(np.isnan(newdata))
    newdata[ii] = cube_filt[ii]
    ii = np.where(np.isnan(err_cube))
    err_cube[ii] = np.nanmedian(err_cube)
    # And replace any negatives with zeros.
    newdata[newdata < 0] = 0

    # Make a final, corrected deepframe for the baseline intergations.
    deepframe_fnl = utils.make_deepstack(newdata[baseline_ints])

    results = []
    current_int = 0
    # Save interpolated data.
    for n, file in enumerate(datafiles):
        with utils.open_filetype(file) as currentfile:
            currentdata = currentfile.data
            nints = np.shape(currentdata)[0]
            currentfile.data = newdata[current_int:(current_int + nints)]
            currentfile.err = err_cube[current_int:(current_int + nints)]
            currentfile.dq = newdq[current_int:(current_int + nints)]
            current_int += nints
            if save_results is True:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    currentfile.write(output_dir + fileroots[n] + 'badpixstep.fits')

            results.append(currentfile)

    if save_results is True:
        # Save deep frame before and after interpolation.
        hdu1 = fits.PrimaryHDU()
        hdr = fits.Header()
        hdr['EXTNAME'] = 'Interpolated'
        hdu2 = fits.ImageHDU(deepframe_fnl, header=hdr)
        hdr = fits.Header()
        hdr['EXTNAME'] = 'Uninterpolated'
        hdu3 = fits.ImageHDU(deepframe_itl, header=hdr)

        hdul = fits.HDUList([hdu1, hdu2, hdu3])
        hdul.writeto(output_dir + fileroot_noseg + 'deepframe.fits',
                     overwrite=True)

    if do_plot is True:
        if save_results is True:
            outfile = output_dir + 'badpixstep.pdf'
        else:
            outfile = None
        hotpix = np.where(hotpix != 0)
        otherpix = np.where(otherpix != 0)
        plotting.make_badpix_plot(deepframe_itl, hotpix, otherpix,
                                  outfile=outfile, show_plot=show_plot)

    return results, deepframe_fnl


def tracingstep_nirspec(datafiles, deepframe=None, calculate_stability=True,
                        pca_components=10, generate_tracemask=True,
                        mask_width=20, pixel_flags=None, generate_lc=True,
                        baseline_ints=None, smoothing_scale=None,
                        output_dir='./', save_results=True, fileroot_noseg='',
                        do_plot=False, show_plot=False):
    """A multipurpose step to perform some initial analysis of the 2D
    dataframes and produce products which can be useful in further reduction
    iterations. The four functionalities are detailed below:
    1. Locate the centroids of the trace via the edgetrigger algorithm.
    2. (optional) Generate a mask of the target trace.
    3. (optional) Calculate the stability of the trace over the course of
    the TSO.
    4. (optional) Create a smoothed estimate of the white light curve.

    Parameters
    ----------
    datafiles : array-like[str], array-like[RampModel]
        List of paths to datafiles for each segment, or the datamodels
        themselves.
    deepframe : str, array-like[float], None
        Path to median stack file, or the median stack itself. Should be 2D
        (dimy, dimx). If None is passed, one will be generated.
    calculate_stability : bool
        If True, calculate the stabilty of the SOSS trace over the TSO using a
        PCA method.
    pca_components : int
        Number of PCA stability components to calcaulte.
    generate_tracemask : bool
        If True, generate a mask of the target diffraction orders.
    mask_width : int
        Mask width, in pixels, around the trace centroids. Only necesssary if
        generate_tracemask is True.
    pixel_flags: None, str, array-like[str]
        Paths to files containing existing pixel flags to which the trace mask
        should be added. Only necesssary if generate_tracemask is True.
    generate_lc : bool
        If True, also produce a smoothed order 1 white light curve.
    baseline_ints : array-like[int]
        Integrations of ingress and egress. Only necessary if generate_lc=True.
    smoothing_scale : int, None
        Timescale on which to smooth the lightcurve. Only necessary if
        generate_lc=True.
    output_dir : str
        Directory to which to save outputs.
    save_results : bool
        If Tre, save results to file.
    fileroot_noseg : str
        Root file name with no segment information.
    do_plot : bool
        If True, do the step diagnostic plot.
    show_plot : bool
        If True, show the step diagnostic plot instead of/in addition to
        saving it to file.

    Returns
    -------
    centroids : array-like[float]
        Trace centroids for all three orders.
    tracemask : array-like[bool], None
        If requested, the trace mask.
    smoothed_lc : array-like[float], None
        If requested, the smoothed order 1 white light curve.
    """

    fancyprint('Starting Tracing Step.')

    datafiles = np.atleast_1d(datafiles)
    # If no deepframe is passed, construct one. Also generate a datacube for
    # later white light curve or stability calculations.
    if deepframe is None or np.any([generate_lc, calculate_stability]) == True:
        # Construct datacube from the data files.
        for i, file in enumerate(datafiles):
            if isinstance(file, str):
                this_data = fits.getdata(file, 1)
            else:
                this_data = file.data
            if i == 0:
                cube = this_data
            else:
                cube = np.concatenate([cube, this_data], axis=0)
        deepframe = utils.make_deepstack(cube)
    elif isinstance(deepframe, str):
        deepframe = fits.getdata(deepframe)
    # Get the subarray dimensions.
    dimy, dimx = np.shape(deepframe)

    # ===== PART 1: Get trace centroids =====
    fancyprint('Finding trace centroids.')
    # Get centroids via the edgetrigger method.
    save_filename = output_dir + fileroot_noseg
    # Determine which detector.
    det = utils.get_detector_name(datafiles[0])
    if det == 'nrs1':
        xstart = 500
    else:
        xstart = 0
    centroids = utils.get_centroids(deepframe, xstart=xstart,
                                    save_results=save_results,
                                    save_filename=save_filename)
    x1, y1 = centroids[0], centroids[1]

    # ===== PART 2: Create a trace mask =====
    # If requested, create a trace mask.
    tracemask = None
    if generate_tracemask is True:
        fancyprint('Generating trace masks.')
        tracemask = np.zeros_like(deepframe).astype(bool)
        for x, y in zip(x1, y1):
            edge_up = np.ceil(np.min([y + mask_width/2, dimy], axis=0)).astype(int)
            edge_low = np.floor(np.max([y - mask_width/2, 0], axis=0)).astype(int)
            tracemask[edge_low:edge_up, int(x)] = True

        # Save the trace mask to file if requested.
        if save_results is True:
            # If we are to combine the trace mask with existing pixel mask.
            if pixel_flags is not None:
                pixel_flags = np.atleast_1d(pixel_flags)
                # Ensure there is one pixel flag file per data file
                assert len(pixel_flags) == len(datafiles)
                # Combine tracemask with existing flags and overwrite old file.
                parts = pixel_flags[0].split('seg')
                outfile = parts[0] + 'seg' + 'XXX' + parts[1][3:]
                fancyprint('Trace mask added to {}'.format(outfile))
                for flag_file in pixel_flags:
                    old_flags = fits.getdata(flag_file)
                    new_flags = old_flags.astype(bool) | tracemask
                    hdu = fits.PrimaryHDU(new_flags.astype(int))
                    hdu.writeto(flag_file, overwrite=True)
            else:
                hdu = fits.PrimaryHDU(tracemask.astype(int))
                suffix = 'tracemask_width{}.fits'.format(mask_width)
                outfile = output_dir + fileroot_noseg + suffix
                hdu.writeto(outfile, overwrite=True)
                fancyprint('Trace mask saved to {}'.format(outfile))

    # ===== PART 3: Calculate the trace stability =====
    # If requested, calculate the stability of the trace over the course
    # of the TSO using PCA.
    if calculate_stability is True:
        fancyprint('Calculating trace stability using the PCA method...'
                   ' This might take a while.')
        assert save_results is True, 'save_results must be True to run ' \
                                     'calculate_stability.'

        # Calculate the trace stability using PCA.
        outfile = output_dir + '{}_stability_pca.pdf'.format(det)
        pcs, var = soss_stability_pca(cube, n_components=pca_components,
                                      outfile=outfile, do_plot=do_plot,
                                      show_plot=show_plot)
        stability_results = {}
        for i, pc in enumerate(pcs):
            stability_results['Component {}'.format(i+1)] = pc
        # Save stability results.
        suffix = 'soss_stability.csv'
        if os.path.exists(output_dir + fileroot_noseg + suffix):
            old_data = pd.read_csv(output_dir + fileroot_noseg + suffix,
                                   comment='#')
            for key in stability_results.keys():
                old_data[key] = stability_results[key]
            os.remove(output_dir + fileroot_noseg + suffix)
            old_data.to_csv(output_dir + fileroot_noseg + suffix, index=False)
        else:
            df = pd.DataFrame(data=stability_results)
            df.to_csv(output_dir + fileroot_noseg + suffix, index=False)

    # ===== PART 4: Calculate a smoothed light curve =====
    # If requested, generate a smoothed estimate of the white light curve.
    smoothed_lc = None
    if generate_lc is True:
        fancyprint('Generating a smoothed light curve')
        # Format the baseline frames.
        assert baseline_ints is not None
        baseline_ints = utils.format_out_frames(baseline_ints)

        # Do quick and dirty extraction.
        flux = do_box_extraction(cube, np.ones_like(cube), ypos=y1, width=5,
                                 extract_start=xstart)[0]
        timeseries = np.nansum(flux, axis=1)
        # Normalize by the baseline flux level.
        timeseries = timeseries / np.nanmedian(timeseries[baseline_ints])
        # If not smoothing scale is provided, smooth the time series on a
        # timescale of roughly 2% of the total length.
        if smoothing_scale is None:
            smoothing_scale = int(0.02 * np.shape(cube)[0])
        smoothed_lc = median_filter(timeseries, smoothing_scale)

        if save_results is True:
            outfile = output_dir + fileroot_noseg + 'lcestimate.npy'
            fancyprint('Smoothed light curve saved to {}'.format(outfile))
            np.save(outfile, smoothed_lc)

    return centroids, tracemask, smoothed_lc


def tracingstep_soss(datafiles, deepframe=None, calculate_stability=True,
                     pca_components=10, generate_tracemask=True, mask_width=45,
                     pixel_flags=None, generate_order0_mask=False, f277w=None,
                     generate_lc=True, baseline_ints=None,
                     smoothing_scale=None, output_dir='./', save_results=True,
                     fileroot_noseg='', do_plot=False, show_plot=False):
    """A multipurpose step to perform some initial analysis of the 2D
    dataframes and produce products which can be useful in further reduction
    iterations. The five functionalities are detailed below:
    1. Locate the centroids of all three SOSS orders via the edgetrigger
    algorithm.
    2. (optional) Generate a mask of the target diffraction orders.
    3. (optional) Generate a mask of order 0 contaminants from background
    stars.
    4. (optional) Calculate the stability of the SOSS traces over the course
    of the TSO.
    5. (optional) Create a smoothed estimate of the order 1 white light curve.

    Parameters
    ----------
    datafiles : array-like[str], array-like[RampModel]
        List of paths to datafiles for each segment, or the datamodels
        themselves.
    deepframe : str, array-like[float], None
        Path to median stack file, or the median stack itself. Should be 2D
        (dimy, dimx). If None is passed, one will be generated.
    calculate_stability : bool
        If True, calculate the stabilty of the SOSS trace over the TSO using a
        PCA method.
    pca_components : int
        Number of PCA stability components to calcaulte.
    generate_tracemask : bool
        If True, generate a mask of the target diffraction orders.
    mask_width : int
        Mask width, in pixels, around the trace centroids. Only necesssary if
        generate_tracemask is True.
    pixel_flags: None, str, array-like[str]
        Paths to files containing existing pixel flags to which the trace mask
        should be added. Only necesssary if generate_tracemask is True.
    generate_order0_mask : bool
        If True, generate a mask of order 0 cotaminants using an F277W filter
        exposure.
    f277w : None, str, array-like[float]
        F277W filter exposure which has been superbias and background
        corrected. Only necessary if generate_order0_mask is True.
    generate_lc : bool
        If True, also produce a smoothed order 1 white light curve.
    baseline_ints : array-like[int]
        Integrations of ingress and egress. Only necessary if generate_lc=True.
    smoothing_scale : int, None
        Timescale on which to smooth the lightcurve. Only necessary if
        generate_lc=True.
    output_dir : str
        Directory to which to save outputs.
    save_results : bool
        If Tre, save results to file.
    fileroot_noseg : str
        Root file name with no segment information.
    do_plot : bool
        If True, do the step diagnostic plot.
    show_plot : bool
        If True, show the step diagnostic plot instead of/in addition to
        saving it to file.

    Returns
    -------
    centroids : array-like[float]
        Trace centroids for all three orders.
    tracemask : array-like[bool], None
        If requested, the trace mask.
    order0mask : array-like[bool], None
        If requested, the order 0 mask.
    smoothed_lc : array-like[float], None
        If requested, the smoothed order 1 white light curve.
    """

    fancyprint('Starting Tracing Step.')

    datafiles = np.atleast_1d(datafiles)
    # If no deepframe is passed, construct one. Also generate a datacube for
    # later white light curve or stability calculations.
    if deepframe is None or np.any([generate_lc, calculate_stability]) == True:
        # Construct datacube from the data files.
        for i, file in enumerate(datafiles):
            if isinstance(file, str):
                this_data = fits.getdata(file, 1)
            else:
                this_data = file.data
            if i == 0:
                cube = this_data
            else:
                cube = np.concatenate([cube, this_data], axis=0)
        deepframe = utils.make_deepstack(cube)
    elif isinstance(deepframe, str):
        deepframe = fits.getdata(deepframe)
    # Get the subarray dimensions.
    dimy, dimx = np.shape(deepframe)
    if dimy == 256:
        subarray = 'SUBSTRIP256'
    elif dimy == 96:
        subarray = 'SUBSTRIP96'
    else:
        raise NotImplementedError

    # ===== PART 1: Get centroids for orders one to three =====
    fancyprint('Finding trace centroids.')
    # Get the most up to date trace table file.
    step = calwebb_spec2.extract_1d_step.Extract1dStep()
    tracetable = step.get_reference_file(datafiles[0], 'spectrace')
    # Get centroids via the edgetrigger method.
    save_filename = output_dir + fileroot_noseg
    centroids = utils.get_soss_centroids(deepframe, tracetable, subarray,
                                         save_results=save_results,
                                         save_filename=save_filename)
    x1, y1 = centroids[0][0], centroids[0][1]
    x2, y2 = centroids[1][0], centroids[1][1]
    x3, y3 = centroids[2][0], centroids[2][1]

    # ===== PART 2: Create trace masks for each order =====
    # If requested, create a trace mask for each order.
    tracemask = None
    if generate_tracemask is True:
        fancyprint('Generating trace masks.')
        tracemask = np.zeros_like(deepframe).astype(bool)
        if subarray != 'SUBSTRIP96':
            xcens, ycens = [x1, x2, x3], [y1, y2, y3]
        else:
            xcens, ycens = [x1], [y1]
        for order in range(len(xcens)):
            for x, y in zip(xcens[order], ycens[order]):
                edge_up = np.ceil(np.min([y + mask_width/2, dimy], axis=0)).astype(int)
                edge_low = np.floor(np.max([y - mask_width/2, 0], axis=0)).astype(int)
                tracemask[edge_low:edge_up, x] = True

        # Save the trace mask to file if requested.
        if save_results is True:
            # If we are to combine the trace mask with existing pixel mask.
            if pixel_flags is not None:
                pixel_flags = np.atleast_1d(pixel_flags)
                # Ensure there is one pixel flag file per data file
                assert len(pixel_flags) == len(datafiles)
                # Combine tracemask with existing flags and overwrite old file.
                parts = pixel_flags[0].split('seg')
                outfile = parts[0] + 'seg' + 'XXX' + parts[1][3:]
                fancyprint('Trace mask added to {}'.format(outfile))
                for flag_file in pixel_flags:
                    old_flags = fits.getdata(flag_file)
                    new_flags = old_flags.astype(bool) | tracemask
                    hdu = fits.PrimaryHDU(new_flags.astype(int))
                    hdu.writeto(flag_file, overwrite=True)
            else:
                hdu = fits.PrimaryHDU(tracemask.astype(int))
                suffix = 'tracemask_width{}.fits'.format(mask_width)
                outfile = output_dir + fileroot_noseg + suffix
                hdu.writeto(outfile, overwrite=True)
                fancyprint('Trace mask saved to {}'.format(outfile))

    # ===== PART 3: Create order 0 background contamination mask =====
    # If requested, create a mask for all background order 0 contaminants.
    order0mask = None
    if generate_order0_mask is True:
        fancyprint('Generating background order 0 mask.')
        if isinstance(f277w, str):
            try:
                f277w = np.load(f277w)
            except (ValueError, FileNotFoundError):
                fancyprint('F277W filter exposure file cannot be opened.',
                           msg_type='WARNING')
                f277w = None
        if f277w is None:
            fancyprint('No F277W filter exposure provided. Skipping the order '
                       '0 mask.', msg_type='WARNING')
        else:
            order0mask = make_order0_mask_from_f277w(f277w)

            # Save the order 0 mask to file if requested.
            if save_results is True:
                # If we are to combine the trace mask with existing pixel mask.
                if pixel_flags is not None:
                    pixel_flags = np.atleast_1d(pixel_flags)
                    # Ensure there is one pixel flag file per data file
                    assert len(pixel_flags) == len(datafiles)
                    # Combine with existing flags and overwrite old file.
                    parts = pixel_flags[0].split('seg')
                    outfile = parts[0] + 'seg' + 'XXX' + parts[1][3:]
                    fancyprint('Order 0 mask added to {}'.format(outfile))
                    for flag_file in pixel_flags:
                        old_flags = fits.getdata(flag_file)
                        new_flags = old_flags.astype(bool) | order0mask.astype(bool)
                        hdu = fits.PrimaryHDU(new_flags.astype(int))
                        hdu.writeto(flag_file, overwrite=True)
                else:
                    hdu = fits.PrimaryHDU(order0mask)
                    suffix = 'order0_mask.fits'
                    outfile = output_dir + fileroot_noseg + suffix
                    hdu.writeto(outfile, overwrite=True)
                    fancyprint('Order 0 mask saved to {}'.format(outfile))

    # ===== PART 4: Calculate the trace stability =====
    # If requested, calculate the stability of the SOSS trace over the course
    # of the TSO using PCA.
    if calculate_stability is True:
        fancyprint('Calculating trace stability using the PCA method...'
                   ' This might take a while.')
        assert save_results is True, 'save_results must be True to run ' \
                                     'soss_stability.'

        # Calculate the trace stability using PCA.
        outfile = output_dir + 'soss_stability_pca.pdf'
        pcs, var = soss_stability_pca(cube, n_components=pca_components,
                                      outfile=outfile, do_plot=do_plot,
                                      show_plot=show_plot)
        stability_results = {}
        for i, pc in enumerate(pcs):
            stability_results['Component {}'.format(i+1)] = pc
        # Save stability results.
        suffix = 'soss_stability.csv'
        if os.path.exists(output_dir + fileroot_noseg + suffix):
            old_data = pd.read_csv(output_dir + fileroot_noseg + suffix,
                                   comment='#')
            for key in stability_results.keys():
                old_data[key] = stability_results[key]
            os.remove(output_dir + fileroot_noseg + suffix)
            old_data.to_csv(output_dir + fileroot_noseg + suffix, index=False)
        else:
            df = pd.DataFrame(data=stability_results)
            df.to_csv(output_dir + fileroot_noseg + suffix, index=False)

    # ===== PART 5: Calculate a smoothed light curve =====
    # If requested, generate a smoothed estimate of the order 1 white light
    # curve.
    smoothed_lc = None
    if generate_lc is True:
        fancyprint('Generating a smoothed light curve')
        # Format the baseline frames.
        assert baseline_ints is not None
        baseline_ints = utils.format_out_frames(baseline_ints)

        # Use an area centered on the peak of the order 1 blaze to estimate the
        # photometric light curve.
        postage = cube[:, 20:60, 1500:1550]
        timeseries = np.nansum(postage, axis=(1, 2))
        # Normalize by the baseline flux level.
        timeseries = timeseries / np.nanmedian(timeseries[baseline_ints])
        # If not smoothing scale is provided, smooth the time series on a
        # timescale of roughly 2% of the total length.
        if smoothing_scale is None:
            smoothing_scale = int(0.02 * np.shape(cube)[0])
        smoothed_lc = median_filter(timeseries, smoothing_scale)

        if save_results is True:
            outfile = output_dir + fileroot_noseg + 'lcestimate.npy'
            fancyprint('Smoothed light curve saved to {}'.format(outfile))
            np.save(outfile, smoothed_lc)

    return centroids, tracemask, order0mask, smoothed_lc


def make_order0_mask_from_f277w(f277w, thresh_std=1, thresh_size=10):
    """Locate order 0 contaminants from background stars using an F277W filter
     exposure data frame.

    Parameters
    ----------
    f277w : array-like[float]
        An F277W filter exposure, superbias and background subtracted.
    thresh_std : int
        Threshold above which a group of pixels will be flagged.
    thresh_size : int
        Size of pixel group to be considered an order 0.

    Returns
    -------
    mask : array-like[int]
        Frame with locations of order 0 contaminants.
    """

    dimy, dimx = np.shape(f277w)
    mask = np.zeros_like(f277w)

    # Loop over all columns and find groups of pixels which are significantly
    # above the column median.
    # Start at column 700 as that is ~where pickoff mirror effects start.
    for col in range(700, dimx):
        # Subtract median from column and get the standard deviation
        diff = f277w[:, col] - np.nanmedian(f277w[:, col])
        dev = np.nanstd(diff)
        # Find pixels which are deviant.
        vals = np.where(np.abs(diff) > thresh_std * dev)[0]
        # Mark consecutive groups of pixels found above.
        for group in mit.consecutive_groups(vals):
            group = list(group)
            if len(group) > thresh_size:
                mask[group, col] = 1

    return mask


def soss_stability_pca(cube, n_components=10, outfile=None, do_plot=False,
                       show_plot=False):
    """Calculate the stability of the SOSS trace over the course of a TSO
    using a PCA method.

    Parameters
    ----------
    cube : array-like[float]
        Cube of TSO data.
    n_components : int
        Maximum number of principle components to calcaulte.
    outfile : None, str
        File to which to save plot.
    do_plot : bool
        If True, do the step diagnostic plot.
    show_plot : bool
        If True, show the step diagnostic plot instead of/in addition to
        saving it to file.

    Returns
    -------
    pcs : array-like[float]
        Extracted principle components.
    var : array-like[float]
        Explained variance of each principle component.
    """

    # Flatten cube along frame direction.
    nints, dimy, dimx = np.shape(cube)
    cube = np.reshape(cube, (nints, dimx * dimy))

    # Replace any remaining nan-valued pixels.
    cube2 = np.reshape(np.copy(cube), (nints, dimy*dimx))
    ii = np.where(np.isnan(cube2))
    med = bn.nanmedian(cube2)
    cube2[ii] = med

    # Do PCA.
    pca = PCA(n_components=n_components)
    thispca = pca.fit(cube2.transpose())

    # Get PCA results.
    pcs = pca.components_
    var = pca.explained_variance_ratio_

    if do_plot is True:
        # Reproject PCs onto data.
        projection = pca.transform(cube2.transpose())
        projection = np.reshape(projection, (dimy, dimx, n_components))
        # Do plot.
        plotting.make_pca_plot(pcs, var, projection.transpose(2, 0, 1),
                               outfile=outfile, show_plot=show_plot)

    return pcs, var


def run_stage2(results, background_model, baseline_ints, save_results=True,
               force_redo=False, space_thresh=15, time_thresh=15,
               calculate_stability=True, pca_components=10,
               root_dir='./', output_tag='', smoothing_scale=None,
               skip_steps=None, generate_lc=True, generate_tracemask=True,
               mask_width=45, pixel_flags=None, generate_order0_mask=True,
               f277w=None, do_plot=False, show_plot=False, **kwargs):
    """Run the supreme-SPOON Stage 2 pipeline: spectroscopic processing,
    using a combination of official STScI DMS and custom steps. Documentation
    for the official DMS steps can be found here:
    https://jwst-pipeline.readthedocs.io/en/latest/jwst/pipeline/calwebb_spec2.html

    Parameters
    ----------
    results : array-like[str], array-like[CubeModel]
        supreme-SPOON Stage 1 output files.
    background_model : array-like[float]
        SOSS background model.
    baseline_ints : array-like[int]
        Integrations of ingress and egress.
    save_results : bool
        If True, save results of each step to file.
    force_redo : bool
        If True, redo steps even if outputs files are already present.
    space_thresh : int
        Sigma threshold for pixel to be flagged as an outlier spatially.
    time_thresh : int
        Sigma threshold for pixel to be flagged as an outlier temporally.
    calculate_stability : bool
        If True, calculate the stability of the SOSS trace over the course of
        the TSO using a PCA method.
    pca_components : int
        Number of PCA components to calculate.
    root_dir : str
        Directory from which all relative paths are defined.
    output_tag : str
        Name tag to append to pipeline outputs directory.
    smoothing_scale : int, None
        Timescale on which to smooth the lightcurve.
    skip_steps : array-like[str], None
        Step names to skip (if any).
    generate_lc : bool
        If True, produce a smoothed order 1 white light curve.
    generate_tracemask : bool
        If True, generate a mask of the target diffraction orders.
    mask_width : int
        Mask width, in pixels, around the trace centroids. Only necesssary if
        generate_tracemask is True.
    pixel_flags: None, str, array-like[str]
        Paths to files containing existing pixel flags to which the trace mask
        should be added. Only necesssary if generate_tracemask is True.
    generate_order0_mask : bool
        If True, generate a mask of order 0 cotaminants using an F277W filter
        exposure.
    f277w : None, str, array-like[float]
        F277W filter exposure which has been superbias and background
        corrected. Only necessary if generate_order0_mask is True.
    do_plot : bool
        If True, make step diagnostic plots.
    show_plot : bool
        Only necessary if do_plot is True. Show the diagnostic plots in
        addition to/instead of saving to file.

    Returns
    -------
    results : array-like[CubeModel]
        Datafiles for each segment processed through Stage 2.
    """

    # ============== DMS Stage 2 ==============
    # Spectroscopic processing.
    fancyprint('**Starting supreme-SPOON Stage 2**')
    fancyprint('Spectroscopic processing')

    if output_tag != '':
        output_tag = '_' + output_tag
    # Create output directories and define output paths.
    utils.verify_path(root_dir + 'pipeline_outputs_directory' + output_tag)
    utils.verify_path(root_dir + 'pipeline_outputs_directory' + output_tag + '/Stage2')
    outdir = root_dir + 'pipeline_outputs_directory' + output_tag + '/Stage2/'

    if skip_steps is None:
        skip_steps = []

    # ===== Assign WCS Step =====
    # Default DMS step.
    if 'AssignWCSStep' not in skip_steps:
        if 'AssignWCSStep' in kwargs.keys():
            step_kwargs = kwargs['AssignWCSStep']
        else:
            step_kwargs = {}
        step = AssignWCSStep(results, output_dir=outdir)
        results = step.run(save_results=save_results, force_redo=force_redo,
                           **step_kwargs)

    # ===== Source Type Determination Step =====
    # Default DMS step.
    if 'SourceTypeStep' not in skip_steps:
        if 'SourceTypeStep' in kwargs.keys():
            step_kwargs = kwargs['SourceTypeStep']
        else:
            step_kwargs = {}
        step = SourceTypeStep(results, output_dir=outdir)
        results = step.run(save_results=save_results, force_redo=force_redo,
                           **step_kwargs)

    # ===== Flat Field Correction Step =====
    # Default DMS step.
    if 'FlatFieldStep' not in skip_steps:
        if 'FlatFieldStep' in kwargs.keys():
            step_kwargs = kwargs['FlatFieldStep']
        else:
            step_kwargs = {}
        step = FlatFieldStep(results, output_dir=outdir)
        results = step.run(save_results=save_results, force_redo=force_redo,
                           **step_kwargs)

    # ===== Background Subtraction Step =====
    # Custom DMS step.
    if 'BackgroundStep' not in skip_steps:
        if 'BackgroundStep' in kwargs.keys():
            step_kwargs = kwargs['BackgroundStep']
        else:
            step_kwargs = {}
        step = BackgroundStep(results, background_model=background_model,
                              output_dir=outdir)
        results = step.run(save_results=save_results, force_redo=force_redo,
                           do_plot=do_plot, show_plot=show_plot,
                           **step_kwargs)[0]

    # ===== Bad Pixel Correction Step =====
    # Custom DMS step.
    if 'BadPixStep' not in skip_steps:
        if 'BadPixStep' in kwargs.keys():
            step_kwargs = kwargs['BadPixStep']
        else:
            step_kwargs = {}
        step = BadPixStep(results, baseline_ints=baseline_ints,
                          output_dir=outdir)
        step_results = step.run(save_results=save_results,
                                space_thresh=space_thresh,
                                time_thresh=time_thresh, force_redo=force_redo,
                                do_plot=do_plot, show_plot=show_plot,
                                **step_kwargs)
        results, deepframe = step_results
    else:
        deepframe = None

    # ===== Tracing Step =====
    # Custom DMS step.
    if 'TracingStep' not in skip_steps:
        step = TracingStep(results, deepframe=deepframe, output_dir=outdir)
        step_results = step.run(calculate_stability=calculate_stability,
                                pca_components=pca_components,
                                generate_tracemask=generate_tracemask,
                                mask_width=mask_width, pixel_flags=pixel_flags,
                                generate_order0_mask=generate_order0_mask,
                                f277w=f277w,
                                generate_lc=generate_lc,
                                baseline_ints=baseline_ints,
                                smoothing_scale=smoothing_scale,
                                save_results=save_results, do_plot=do_plot,
                                show_plot=show_plot, force_redo=force_redo)

    return results
