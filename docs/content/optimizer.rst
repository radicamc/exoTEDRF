Optimizing a Reduction
======================

Every reduction involves a whole host of small choices: how wide to make the 1/f mask, how aggressively to clip cosmic rays and bad pixels, how wide an extraction aperture to use, etc.
The defaults in exoTEDRF are sensible starting points, but the best value for any of these will vary from dataset to dataset. Rather than tweaking them all by hand, exoTEDRF includes an optimizer
which will sweep through a grid of values for each parameter and pick the ones that produce the cleanest light curves.

Below is a quick walkthrough of how to use it. Like the run_DMS.py script, the optimizer is not really meant to replace a careful look at your data (so make sure to work through one of the
tutorial notebooks first!), but it is a handy way to tune a reduction once you know what you're working with.


How it Works
------------

The optimizer runs in two phases:

 * **Phase 1**: Using only the *first segment* of the TSO, the optimizer sweeps through each parameter in turn, in the order that they appear in the pipeline. For each trial value, it reruns the pipeline from the relevant step, does a quick box extraction, and computes a cost.
   Once a sweep is finished, the value with the lowest cost is locked in and used for all subsequent sweeps. Any parameter which hasn't been swept yet is held at the middle value of its grid.
 * **Phase 2**: The full TSO (all segments) is then run through Stages 1 to 3 with the winning parameters. If requested, the extraction width is optimized here on the full dataset.

The cost itself is a measure of the point-to-point scatter in the light curves. Specifically, it is the median absolute second difference of the normalized flux, i.e., how much each integration
deviates from the average of its two neighbours. Using the second difference means that smooth signals like the transit or eclipse itself, or slow systematic trends, don't contribute much,
while white noise does. There are two terms which can be weighted using the ``w1`` and ``w2`` parameters:

    - ``w1``: The scatter of the white light curve.
    - ``w2``: The median scatter of the spectroscopic light curves, computed only using the out-of-transit (or eclipse) integrations defined by ``baseline_ints``, and over the wavelength range set by ``wave_range``.

By default, only the spectroscopic term is used (``w1: 0``, ``w2: 1``). In general, this is what we care about since it is the precision of the transmission or emission spectrum we are trying to maximize!

.. note::
    Since each trial value requires rerunning part of the pipeline, the optimizer can take quite a bit longer than a single reduction. Phase 1 only uses the first segment of the TSO to keep things manageable,
    but it's still a good idea to run the optimizer on a cluster.


Running the Optimizer
---------------------

#. Copy the run_optimize.py script and the run_optimize.yaml config file into your working directory.
#. Fill out the top part of the yaml file just as you would for run_DMS.yaml (input directory, observing mode, which steps to run, etc.).
#. Set ``baseline_ints`` to cover the out-of-transit integrations of your observation. These are used both in the reduction and for computing the cost.
#. Choose which parameters to optimize (see below).
#. Once happy with the inputs, enter the following in your terminal:

    .. code-block:: bash

        python run_optimize.py run_optimize.yaml


Choosing Parameters to Sweep
----------------------------

Every parameter that can be optimized comes in a pair in run_optimize.yaml: an ``optimize_`` flag, and the parameter itself. To sweep a parameter, set its flag to ``True`` and pass a list of values to try.
To hold it fixed, set the flag to ``False`` and pass a single value. For example:

.. code-block:: yaml

    # Sweep the time-domain jump threshold...
    optimize_time_jump_threshold : True
    time_jump_threshold : [5,6,7,8,9,10]

    # ...but keep the window fixed.
    optimize_time_window : False
    time_window : 5

The template config is set up for NIRISS/SOSS by default, but the comments next to each setting show the appropriate alternatives for NIRSpec and MIRI. The parameters which can be optimized are:

**Stage 1**

    - ``soss_inner_mask_width``, ``soss_outer_mask_width``: Width of the 1/f masks around the target traces (NIRISS only).
    - ``nirspec_mask_width``: Width of the 1/f and background mask around the target trace (NIRSpec only).
    - ``time_jump_threshold``, ``time_window``: Sigma threshold and window size for time-domain cosmic ray flagging.

**Stage 2**

    - ``miri_trace_width``, ``miri_background_width``: Trace mask width and background region width for the background subtraction (MIRI only).
    - ``space_outlier_threshold``, ``time_outlier_threshold``, ``box_size``, ``window_size``: Thresholds and window sizes for the spatial and temporal bad pixel flagging.

**Stage 3**

    - ``extract_width``: Width of the extraction aperture. Since the aperture width can have a large impact on the final precision, this is optimized in Phase 2 on the full dataset rather than on a single segment.

Make sure that any instrument-specific parameters that don't apply to your observing mode have their ``optimize_`` flags set to ``False`` (e.g., don't try to sweep ``soss_inner_mask_width`` for a NIRSpec observation!).

.. note::
    For NIRSpec and MIRI, the quick extraction used in Phase 1 does not yet have access to a wavelength solution, so the Phase 1 cost is calculated over the full spectrum, and ``wave_range`` only comes into play when optimizing the extraction width in Phase 2.
    If no ``wave_range`` is provided, the optimizer will default to a high signal-to-noise region for each instrument (1--2µm for NIRISS, 3--3.5µm for NRS1, 4--4.5µm for NRS2, and 5--10µm for MIRI).


Outputs
-------

All optimizer outputs are saved to the Optimizer_Files subdirectory of the pipeline_outputs_directory, tagged with the ``run_name`` tag given in the config file:

    - ``Cost_Summary_<name_tag>.txt``: A table of every trial, with the parameter values used, the runtime, and the resulting cost.
    - ``Cost_Summary_<name_tag>.png``: A summary plot of the cost for each sweep (normalized within each sweep), with the best value highlighted, along with a table of the winning parameters.
    - ``LightCurve_Scatter_<name_tag>.txt`` and ``LightCurve_Scatter_Plot_<name_tag>.png``: The scatter as a function of wavelength for each trial. These are useful to check that the improvement is not driven by just a handful of wavelength channels.

The final, optimized Stage 1 to 3 products are saved to the usual Stage1, Stage2, and Stage3 directories, just as with run_DMS.py, and the winning parameters are printed at the end of the run.
These can then be copied into run_DMS.yaml if you'd like to reproduce, or further tweak, the reduction later on.

.. note::
    The optimizer will happily return the best value on your grid, even if that is one of the edges! If the winning value for a parameter is the first or last value you tried, it's worth extending the grid and running again.


Shortcuts
---------

Sometimes you might only want to rerun part of the optimization. There are a couple of options at the bottom of run_optimize.yaml for this:

    - ``optimize_extract_width_only``: Skip straight to Stage 3 using existing Stage 2 outputs, and only optimize the extraction width.
    - ``from_pca_only``: Restart from existing ``BadPixStep`` outputs, rerun the PCA reconstruction (e.g., with a new set of ``remove_components``), and then continue on into Stage 3.
