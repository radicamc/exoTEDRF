exoTEDRF Usage Guide
====================

Currently supported instruments/modes: **NIRISS/SOSS**, **NIRSpec/BOTS** (coming soon), **MIRI/LRS** (in development).

The pipeline is divided into four stages, which closely mirror the STScI pipeline:

 * Stage 1: Detector-Level Processing
 * Stage 2: Spectroscopic Processing
 * Stage 3: 1D Spectral Extraction
 * Stage 4: Light Curve Fitting (optional)

Below are several tutorials that will walk you through the basics of JWST data analysis using exoTEDRF.

.. toctree::
   :maxdepth: 1
   :hidden:

   content/tutorials

Alternatively, exoTEDRF can be run in script form. This is not recommended for a first pass, or deep dive into a
particular dataset, but it can be useful for a quick look or to easily test the impact of tweaking aspects
of the analysis.
Below you'll learn how to run exoTEDRF in script mode.

.. toctree::
   :maxdepth: 1
   :hidden:

   content/scripting