---
title: 'exoTEDRF: An EXOplanet Transit and Eclipse Data Reduction Framework'
tags:
  - Python
  - Astronomy
  - JWST
  - Exoplanets
  - Teddy Bears
authors:
  - name: Michael Radica
    orcid: 0000-0002-3328-1203
    affiliation: 1
affiliations:
 - name: Trottier Institute for Research on Exoplanets (iREx), Université de Montréal, Montréal, Canada
   index: 1
date: 05 July 2024
bibliography: paper.bib
---

# Summary
Since the start of science operations in July 2022, JWST has delivered groundbreaking results on a regular basis. 
For spectroscopic studies of exoplanet atmospheres the process of extracting robust and reliable atmosphere 
spectra from the raw JWST observations is critical. Especially as the field pushes to detect the signatures of secondary atmospheres on rocky Earth-like planets
it is imperative to ensure that the spectral features that drive atmosphere inferences are robust against the particular choices made during the data reduction process. 

Here, I present the community with `exoTEDRF` (EXOplanet Transit and Eclipse Data Reduction Framework; formerly known as 
`supreme-SPOON`), an end-to-end pipeline for data reduction and light curve analysis of time series observations (TSOs) of transiting exoplanets with JWST. The pipeline is highly modular and designed to produce reliable spectra from raw JWST exposures. 
`exoTEDRF` (pronounced exo-tedorf) consists of four stages, each of which are further subdivided into a series of steps. These steps can either be run individually, for example in a Jupyter notebook, or via the command line using the provided configuration files.
The steps are highly tunable, allowing full control over every parameter in the reduction. Each step also produces diagnostic plots to allow the user to verify their results at each intermediate stage, and compare outputs with other pipelines if so desired.
Finally, `exoTEDRF` has also been designed to be run in "batch" mode: simultaneously running multiple reductions, each tweaking a subset of parameters, to understand any impacts on the resulting atmosphere spectrum.


# Overview of exoTEDRF Stages
Like similar pipelines [e.g. `Eureka!`, @bell_eureka_2022; `jwst`, @bushouse_howard_2022_7038885]
`exoTEDRF` is divided up into four major stages which are summarized below:

- Stage 1, Detector-level processing: Converts raw, 4D (integrations, groups, $x$-pixel, $y$-pixel) data frames to 3D (integrations, $x$-pixel, $y$-pixel) slope images. Steps include superbias subtractions, correction of 1/$f$ noise, ramp fitting, etc. 
- Stage 2, Spectroscopic processing: Performs additional calibrations to prepare slope images for spectral extraction. Steps include, flat field correction, background subtraction, etc. 
- Stage 3, Spectral extraction: Extract the 2D stellar spectra from the 3D slope images.
- Stage 4, Light curve fitting: An optional stage for the fitting of extracted light curves.

In `exoTEDRF`, Stage 4 is an optional installation which is currently built around the `exoUPRF` library [@michael_radica_2024_12628066], and incorporates tools such as `ExoTiC-LD` [@david_grant_2022_7437681] for the estimation of stellar limb-darkening parameters. 
In certain places (e.g., superbias subtraction, flat field correction), `exoTEDRF` simply provides a wrapper around the existing functionalities of the `jwst` package maintained by the Space Telescope Science Institute. 


# Statement of Need
Data analysis is a challenging process that is encountered by all observational studies. Ensuring that the resulting 
atmosphere spectra are robust against particular choices made in the reduction process is critical, especially as we push to characterize the atmospheres of small rocky planets. 
The modularity and tunability of `exoTEDRF` make it easy to run multiple reductions of a given dataset, and therefore robustly ascertain whether the spectral features driving atmosphere inferences are robust, or sensitive to the peculiarities of a given reduction.
Moreover, `exoTEDRF` has full support for TSOs with NIRISS/SOSS [@Albert2023], an observing mode which is underserved by the current ecosystem of JWST reduction tools, including being the only pipeline with the ability to run the `ATOCA` extraction algorithm [@Darveau-Bernier2022; @Radica2022] to explicitly model the SOSS order overlap. 


# Documentation
Documentation for `exoTEDRF`, including example notebooks, is available at [https://exotedrf.readthedocs.io/en/latest/](https://exotedrf.readthedocs.io/en/latest/). 


# Uses of exoTEDRF in Current Literature
`exoTEDRF` (particularly in its previous life as `supreme-SPOON`) has been widely applied to exoplanet TSOs. 
A list of current literature which has made use of `exoTEDRF` includes: 
@Feinstein2023, @Coulombe2023, @Radica2023, @Albert2023, @Lim2023, @Radica2024, @Fournier-Tondreau2024, @Benneke2024, and @Cadieux2024.


# Future Developments
The current release of `exoTEDRF` (v2.0.0) currently supports the reduction of TSOs observed with JWST NIRISS/SOSS as well as NIRSpec/BOTS. 
Support for observations MIRI/LRS is in development and will be added in the coming months.
`exoTEDRF` has also been applied to exoplanet observations from the Hubble Space Telescope using the UVIS mode (Radica et al., 2024b, in prep). This functionality will also be made available to the public.

Suggestions for additional features are always welcome!


# Similar Tools
The following is a list of other open source pipelines tailored to exoplanet observations with JWST, some of which general purpose, and others which are more tailored to specific instruments:

- General purpose: `Eureka!` [@bell_eureka_2022], `jwst` [@bushouse_howard_2022_7038885], `transitspectroscopy` [@espinoza_nestor_2022_6960924]
- NIRISS specific: `nirHiss` [@nirhiss2022]
- NIRCam specific: `tshirt` [@tshirt2022]
- MIRI specific:  `PACMAN` [@pacman2022], `ExoTiC-MIRI` [@grant_david_2023_8211207]
- NIRSpec specific: `ExoTiC-JEDI` [@jedi2022]

Packages like `exoplanet` [@exoplanet:joss], `Eureka!` [@bell_eureka_2022], and `juliet` [@espinoza_juliet_2019] also enable similar light curve fitting. 


# Acknowledgements
The foundations of `exoTEDRF` are built upon many wonderful Python libraries, including `numpy` [@harris2020array], `scipy` [@2020SciPy-NMeth], `astropy` [@astropy:2013; @astropy:2018], and `matplotlib` [@Hunter:2007].

MR acknowledges funding from the Natural Sciences and Engineering Research Council of Canada,
the Fonds de Recherche du Québec -- Nature et Technologies, and the Trottier Institute for Research on Exoplanets. 
He would also like to thank the JWST Transiting Exoplanet Community Early Release Science program for providing the 
forum where much of the development of this pipeline occurred, and in particular, Adina Feinstein, Louis-Philippe 
Coulombe, Néstor Espinoza, and Lili Alderson for many helpful conversations. 


# References
