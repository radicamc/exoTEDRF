# Changelog
All notable changes to this project will be documented in this file.

### [2.1.0] -- 2025-02-03
#### Added
- Massive improvements to speed and memory usage.
- Expicit support for NIRSpec/G395M.
- Custom superbias subtractions for NIRSpec.
- Functionalities for SOSS flux calibration.
- PCAReconstructStep for PCA reconstruction of datacubes. 
- Misc. bug fixes.

#### Removed
- Deepframe functionalities in BadPixStep.
- PCA functionalities in TracingStep.

### [2.0.0] -- 2024-07-05
#### Added
- Support for NIRSpec/BOTS observations. 
- Functions to refine SOSS timestamps for limb asymmetry studies.
- Light curve fitting with exoUPRF.
- Updates to documentation.
- NIRSpec/G395H reduction tutorial. 
- Misc. bug fixes.

#### Removed
- Light curve plotting routines
- juliet dependencies. 
- Light curve fitting tutorial.

### [1.4.0] -- 2024-03-08
#### Added
- Package name changed to exoTEDRF.

### [1.3.2] -- 2024-03-08
#### Added
- Allow the trace mask width to be specified directly during 1/f correction.
- More support for SUBSTRIP96, particularly in plotting routines. 
- Add possibility for flux offset in background correction.
- Best-fitting models are now automatically saved in Stage 4. 
- Patch of ATOCA implementation to circumvent occasional error in estimation of wavelength grid.
- supreme-SPOON end-of-life notices.
- Misc. bug fixes. 

#### Removed
- Trace masking functionality in TracingStep.

### [1.3.1] -- 2024-01-19
#### Added
- Fixed implementation of STScI up-the-ramp JumpStep which would not run consistently.
- Made it easier to select stellar models to use for limb-darkening calculation in Stage 4. 

### [1.3.0] -- 2023-12-21
#### Added
- Compatibility with v1.12.5 of the STScI jwst package.
- Variety of utility functions to supplement pipeline functionalities.
- Further improvements in bad pixel correction.
- Updated intrapixel box extraction algorithm.
- Optimization of box extraction width.
- Multiple new 1/f corrections, including chromatic and windowed methods.
- Dark current subtraction step in Stage 1.
- Re-addition of reference pixel correction in Stage 1. 
- Some improvements to diagnostic plotting.
- Automated differential background scaling.
- Misc. bug fixes. 

#### Removed
- Useless GroupScaleStep.
- Several unneeded functions.

### [1.2.0] -- 2023-11-21
#### Added
- Best fitting transit models are saved by default.
- Limb-darkening coefficients are automatically calculated at desired binning during light curve fitting.
- Streamline process for fitting (free or prior) or fixing LD coefficients during fits. 
- Incorporate PASTASOSS to calculate SOSS wavelength solution based on pupil wheel position. 
- Option to specify background estimation region.
- Improvements to bad pixel interpolation.
- extra_functions.py for potentially helpful functions not directly used by the pipeline.
- Misc. bug fixes. 

#### Removed
- Stability calculations via the cross-correlation method.

### [1.1.7] -- 2023-07-28
#### Added
- Streamline installation process.

### [1.1.2] -- 2023-07-24
#### Added
- Swap order of FlatFieldStep and BackgroundStep, which were somehow backwards.
- Corrected calculation of errors in transit/eclipse fitting.
- Misc. bug fixes. 

#### Removed
- RefPixStep. Functionality is redundant. 
- Buggy automatic stellar parameter searching.

### [1.1.1] -- 2023-06-01
#### Added
- Support for eclipse fitting.
- Misc. bug fixes. 

### [1.1.0] -- 2023-05-04
#### Added
- Compatibility with jwst v1.8.5.
- Major updates in nearly every step for speed and self-consistency.
- Vastly simplified ATOCA extraction. 
- Added plotting capabilities to most steps.
- Added PCA method for assessing trace stability.
- Automatic detection and masking of Order 0 contaminants if an F277W filter exposure is available.
- New box extraction routine as jwst Extract1dStep has removed this functionality.
- Refinement of wavelength solution.
- When running pipeline via script, a copy of the config file will now be saved in the output directory.
- Misc. bug fixes.

#### Removed
- LightCurveEstimateStep. Functionality now added to TracingStep.
- Large wrapper around ATOCA extraction is removed.
- SOSSSolverStep.
- LightCurveStep. Functionality now added to Extract1dStep
- Various now outdated utility functions.

### [1.0.0] -- 2023-03-13
#### Added
- Preparation for first release.
- New fancy printing.
- Misc. bug fixes.

### [0.4.0] -- 2023-03-02
#### Added
- Correction of bug pointing to throughputs for wrong order when calculating LD coefficients.
- Switch default stellar models to 3D grid for LD calculation.
- Simplify LD coefficient calculation.
- Misc. bug fixes. 

#### Removed
- Unnecessary functionality to calculate LD coefficients on grid of stellar models to derive errors.  

### [0.3.0] -- 2022-12-22
#### Added
- Add time-domain jump detection for ngroup=2 observations.
- Allow for integration-level correction of 1/f noise.
- Paralellization of stability parameter calculations.
- Allow for piecewise subtraction of background.

### [0.2.0] -- 2022-10-28
#### Added
- Updated run_DMS script to accept yaml config file.
- TracingStep can now calculate changes in x and y position of the trace as well as the FWHM.
- BackgroundStep has been moved to Stage 2. 
- Background is re-added after 1/f noise correction during Stage 1 such that it is properly treated by the non-linearity correction.
- Speed improvements for 1/f and hot pixel corrections.
- stage4.py: code including routines for light curve binning and parallelized fitting using ray and juliet (transit only currently).
- fit_lightcurves.py: script to fit spectrophotometric light curves.
- fit_lightcurves.yaml: config file for fit_lightcurves.py.

#### Removed
- Redundant plotting routines.
- Multiple depreciated utility functions.

### [0.1.0] -- 2022-08-29
#### Added
- Base prerelease code for Stages 1 - 3.