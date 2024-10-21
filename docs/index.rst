exoTEDRF's Documentation
==================================

exoTEDRF: the EXOplanet Transit and Eclipse Data Reduction Framework!

**exoTEDRF** (formerly supreme-SPOON) is an end-to-end pipeline for JWST exoplanet time series observations.

Data analysis is a challenging process that is encountered by all observational studies. Ensuring that the resulting
atmosphere spectra are robust against particular choices made in the reduction process is critical, especially as we push to characterize the atmospheres of small rocky planets.
The modularity and tunability of **exoTEDRF** make it easy to run multiple reductions of a given dataset, and therefore robustly ascertain whether the spectral features driving atmosphere inferences are robust, or sensitive to the peculiarities of a given reduction.

**exoTEDRF** also has full support for TSOs with NIRISS/SOSS, an observing mode which is underserved by the current ecosystem of JWST reduction tools, including being the only pipeline with the ability to run the **ATOCA** extraction algorithm to explicitly model the SOSS order overlap.

Currently supported instruments/modes are: **NIRISS/SOSS** and **NIRSpec/BOTS**. **MIRI/LRS** is in development.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   content/installation
   content/usage
   content/citations
   content/contributions
   api/api

*Happy Analyzing!*
