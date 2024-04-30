---
title: 'exoTEDRF: An EXOplanet Transit and Eclipse Data Reduction Framework'
tags:
  - Python
  - Astronomy
  - JWST
  - Exoplanets
authors:
  - name: Michael Radica
    orcid: 0000-0002-3328-1203
    affiliation: 1
affiliations:
 - name: Trottier Institute for Research on Exoplanets (iREx), Université de Montréal, Canada
   index: 1
date: 30 April 2024
bibliography: paper.bib
---

# Summary
`exoTEDRF` (EXOplanet Transit and Eclipse Data Reduction Framework; formerly known as 
`supreme-SPOON`) is an end-to-end pipeline for the data reduction and light curve analysis
of exoplanet time series observations (TSOs) with JWST.


# Overview of exoTEDRF Stages
Like similar pipelines (`Eureka!` [@bell_eureka_2022], `jwst` [@bushouse_howard_2022_7038885], etc.)
`exoTEDRF` is divided up into four major stages which are summarized below:

- Stage 1, Detector-level processing: Converts raw, 4D (integrations, groups, x-pixel, y-pixel) data frame to 3D (integrations, x-pixel, y-pixel) slope images. Steps include superbias subtractions, correction of 1/$f$ noise, ramp fitting, etc. 
- Stage 2, Spectroscopic processing: Performs additional calibrations to prepare slope images for spectral extraction. Steps include, flat field correction, background subtraction, etc. 
- Stage 3, Spectral extraction: Extract the 2D stellar spectra from the 3D slope images.
- Stage 4, Light curve fitting: An optional stage for the fitting of extracted light curves.

In `exoTEDRF` Stage 4 is an optional installation, which is currently built around the excellent `juliet` library [@espinoza_juliet_2019], and incorporates tools such as
`exoTiC-LD` [@david_grant_2022_7437681] fof the estimation of stellar limb darkening parameters. 


# Statement of Need
Is good pipeline.


# Documentation
Documentation for `exoTEDRF`, including example notebooks, is available at [https://exotedrf.readthedocs.io/en/latest/](https://exotedrf.readthedocs.io/en/latest/). 


# Uses of exoTEDRF in Current Literature
`exoTEDRF` (particulalry in its previous life as `supreme-SPOON`) has been widely applied to exoplanet TSOs. 
A list of current literature which has made use of `exoTEDRF` includes: 
@Feinstein2023, @Coulombe2023, @Radica2023, @Albert2023, @Lim2023, @Radica2024, @Fournier-Tondreau2024, and @Benneke2024.


# Future Developments
The current release of `exoTEDRF` (v1.4.0) currently supports the reduction of TSOs observed with JWST NIRISS/SOSS. 
Support for observations with NIRSpec and MIRI/LRS are in development and will be added in the coming months.
`exoTEDRF` has also been applied to exoplanet observations from the Hubble Space Telescope using the UVIS mode (Radica et al., 2024, in prep).
This functionality will also be made available to the public.
Finally, updates to the light curve fitting functionalities are underway to allow for more flexibility for the fitting of both astrophysical and systematics models.
Suggestions for additional features are always welcome!


# Similar Tools
The following is a list of other open source pipelines tailored to exoplanet observations with JWST:
`Eureka!` @bell_eureka_2022, `jwst` @bushouse_howard_2022_7038885, `tshirt` @tshirt2022, `PACMAN` @pacman2022,
`nirHiss` @nirHiss2022, `ExoTiC-JEDI` @jedi2022, `ExoTiC-MIRI` @grant_david_2023_8211207, 
and `transitspectroscopy` @espinoza_nestor_2022_6960924.
Packages like `exoplanet` @exoplanet:joss and `Eureka!` @bell_eureka_2022 also enable simmilar light curve fitting. 


# Acknowledgements
MR acknowledges funding from the Natural Sciences and Engineering Research Council of Canada,
the Fonds de Recherche du Québec - Nature et Technologies, and the Trottier Institute for Research on Exoplanets. 
MR would also like to thank the JWST Transiting Exoplanet Community Early Release Science program for providing the 
forum where much of the development of this pipeline occured, and in particular, Adina Feinstein, Louis-Philippe 
Coulombe, Néstor Espinoza, and Lili Alderson for many helpful conversations. 


# References