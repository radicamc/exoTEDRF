exoTEDRF Usage Guide
====================

Currently supported instruments/modes: **NIRISS/SOSS**, **NIRSpec/BOTS**, and **MIRI/LRS**.

The pipeline is divided into four stages, which closely mirror the STScI JWST calibration pipeline:

 * Stage 1: Detector-Level Processing
 * Stage 2: Spectroscopic Processing
 * Stage 3: 1D Spectral Extraction
 * Stage 4: Light Curve Fitting (optional)


Tutorial Notebooks
------------------

Below are several tutorials that will walk you through the basics of JWST data analysis using exoTEDRF.

Note that for NIRSpec, the notebook example features the G395H grating, but you can follow basically the same procedure for PRISM any of the other gratings (and keep in mind that PRISM or any of the "M" gratings only use NRS1!).

.. note::
    Be aware that TSO data is large, and reducing it is both memory and storage intensive, particularly for NIRISS.
    For smaller datasets such as those featured in the tutorial notebooks, a full reduction is possible on a laptop, but it is generally preferrable to reduce your observations on a cluster.
    For example: running the NIRISS tutorial notebook requires ~30Gb of memory, and ~200Gb of storage (to save all outputs).

.. toctree::
   :maxdepth: 2

   notebooks/tutorial_niriss-soss
   notebooks/tutorial_nirspec-g395h
   notebooks/tutorial_miri-lrs

exoTEDRF also has the capabilities for basic transit and eclipse light curve fitting in Stage 4 through the exoUPRF library.
If you are looking for some guidance on basic light curve fitting for JWST data, you can check out the exoUPRF tutorial notebooks
`here <https://exouprf.readthedocs.io/en/latest/content/usage.html#tutorial-notebooks>`_.


A Note on 1/f Correction Methods
--------------------------------

exoTEDRF offers multiple possible methods of correcting 1/f noise in TSOs. The 1/f-correction method is controlled via the ``method`` argument in the ``OneOverFStep`` or the ``oof_method`` parameter in run_DMS.yaml.
Below, you will find a brief description of what how each method works.

**NIRSpec**

The NIRSpec target trace is sufficiently thin on the detector that there are generally a sufficient amount of unilluminated pixels to directly estimate the 1/f noise from a given frame.
(Note though that this assumption may not hold if using e.g., the SUB512S subarray).
The NIRSpec 1/f correction also serves as the background subtraction!

    - ``median``: Use the median of all (unmasked) pixels in a column :math:`\pm` X pixels :sup:`[1]` away from the target trace as the 1/f value for that column.
    - ``slope``: Fit a line to all (unmasked) pixels in a column :math:`\pm` X pixels :sup:`[1]` away from the target trace, and subtract this as the 1/f value.

:sup:`[1]`: X is user-defined with ``nirspec_mask_width`` in run_DMS.yaml or the ``OneOverFstep``.

**NIRISS**

NIRISS/SOSS observations are, unfortunately, more complicated to deal with than NIRSpec. Due to the defocusing lens, the target trace is so wide on the detector that there are virtually no unilluminated pixels to use similar 1/f correction methods to NIRSpec.
As a result, we create *difference images* to identify the 1/f contributions. This generally involves subtracting some sort of scaled median stack of the TSO from each frame to remove the target trace and reveal the 1/f noise.
See `Radica et al. (2023) <https://ui.adsabs.harvard.edu/abs/2023MNRAS.524..835R/abstract>`_ for some helpful visuals and a more in-depth discussion of the nuances introduced by this.
The four methods below are essentially different ideologies for accomplishing this.

    - ``scale-achromatic``: Create the difference images using a median stack scaled by an estimate of the white light curve :sup:`[1]`. Use the median of all (unmasked) pixels in a column :math:`\pm` X pixels :sup:`[2]` away from the target traces as the 1/f value for that column. The same 1/f value is used for the entire column.
    - ``scale-achromatic-window``: Create the difference images using a median stack scaled by an estimate of the white light curve :sup:`[1]`. Use the median of all (unmasked) pixels withint a window with an inner width of :math:`\pm` X pixels and outer width :math:`\pm` Y pixels :sup:`[3]` around each order as the 1/f value for that order. Different 1/f values are used for each order within a given column.
    - ``scale-chromatic``: Create the difference images using a median stack scaled by an estimate of the extracted spectroscopic light curves :sup:`[4]`. Use the median of all (unmasked) pixels withint a window with an inner width of :math:`\pm` X pixels and outer width :math:`\pm` Y pixels :sup:`[3]` around each order as the 1/f value for that order. Different 1/f values are used for each order within a given column.
    - ``solve``: Make no assumptions about the underlying scaling of the median stack, and for each frame, column, and order simultaneously solve for the 1/f noise and the factor multiplying the median stack such that :math:`Data = A*MedianStack + 1/f`.

:sup:`[1]`: White light curve estimate passed as ``soss_timeseres`` in run_DMS.yaml or the ``OneOverFstep``

:sup:`[2]`: X is user-defined with ``soss_inner_mask_width`` in run_DMS.yaml or the ``OneOverFstep``.

:sup:`[3]`: Y is user-defined with ``soss_outer_mask_width`` in run_DMS.yaml or the ``OneOverFstep``.

:sup:`[4]`: Light curve estimates passed as ``soss_timeseres`` and ``soss_timeseries_o2`` in run_DMS.yaml or the ``OneOverFstep``

Additionally, in all cases, the 1/f correction can be done either at the group-level (that is, before fitting the ramp) or at the integration-level (that is, after fitting the ramp).


A Note on PCA Reconstruction
----------------------------

exoTEDRF offers a unique correction as the final step of Stage 2 inspired by similar techniques used in the analysis of ground-based high spectral resolution observations:
reconstruction of the TSO data cube via principle component analysis (PCA).
During the ``PCAReconstructStep`` PCA will be used to decompose the TSO cube into the "eigenimages" that best explain the variance in the observation, as well as their corresponding eigenvalue time series
(see the tutorial notebooks for an example of what these look like).

In general, the first principle component, which explains the largest portion of the variance will be the band-integrated (i.e., white) light curve
of whatever transit, eclipse, or phase curve was observed. However, some the higher order terms are generally due to "detector noise", e.g., tilt events, sub-pixel drifts of the position of the spectral trace on the detector,
and particularly with SOSS, a beating pattern from JWST's thermal control system. All of these effects can add excess noise to the extracted light curves!

One option is to attempt to correct for these effects at the light curve fiting stage. However, an alternate option is to use PCA magic to remove these eigenimage timeseries from the TSO data frames themselves, before the exraction.
That way, you can be sure that no adverse detector effects impact the light curves!

It is important to note, though, that this application of PCA is different than that commonly used for high resolution analyses. Here, we are generally going to keep some of the lower order components (which contain the broadband astrophysical
signals), and remove higher-order ones. We are also not looking to "optimize" the removal of components to "maximize" the detection of an injected atmosphere signal. The goal here is simply to remove components that track detector noise and thereby
decrease the precision of the extracted spectra.

Finally, due to the nature of PCA, this step needs to be run on the entire TSO data cube simultaneously, unlike all other steps which work internally on an individual segment at a time. This means that it can be very memory intensive, particularly
for large datasets. If you run into memory issues, the auxiliary outputs from the step (i.e., the observation deeo stack) can still be produced by specifying ``skip_pca=True``. This will produce all auxiliary files, but skip the PCA component of
the step, which will not have a major impact on the final data quality. As mentioned above, detector effects can be fit out at the light curve level, or mitigated by a judicious choice of extraction aperture
(e.g., see Figures C1 & C2 in `Radica et al. 2023 <https://ui.adsabs.harvard.edu/abs/2023MNRAS.524..835R/abstract>`_).


Scripting
---------

Alternatively, exoTEDRF can be run in script form. This is not recommended for a first pass, or deep dive into a
particular dataset, but it can be useful for a quick look or to easily test the impact of tweaking aspects
of the analysis. Stages 1 to 3 can be run at once via the run_DMS.py script as follows:

#. Copy the run_DMS.py script and the run_DMS.yaml config file into your working directory.
#. Fill out the yaml file with the appropriate inputs.
#. Once happy with the input parameters, enter the follwing in your terminal:

    .. code-block:: bash

        python run_DMS.py run_DMS.yaml

To use the light curve fitting capabilities (if installed), simply follow the same procedure with the fit_lightcurves.py and .yaml files.