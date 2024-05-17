exoTEDRF Usage Guide
====================

Currently supported instruments/modes: **NIRISS/SOSS** and **NIRSpec/BOTS**. **MIRI/LRS** is in development.

The pipeline is divided into four stages, which closely mirror the STScI pipeline:

 * Stage 1: Detector-Level Processing
 * Stage 2: Spectroscopic Processing
 * Stage 3: 1D Spectral Extraction
 * Stage 4: Light Curve Fitting (optional)


Tutorial Notebooks
------------------

Below are several tutorials that will walk you through the basics of JWST data analysis using exoTEDRF.

.. toctree::
   :maxdepth: 2

   notebooks/tutorial_niriss-soss
   notebooks/tutorial_nirspec-g395h
   notebooks/tutorial_light-curve-fitting


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