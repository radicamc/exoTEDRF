Scripting
=========

The **exoTEDRF** pipeline can be run in a similar fashion to the STScI jwst pipeline, by individually calling each step.
Alternatively, Stages 1 to 3 can be run at once via the run_DMS.py script as follows:

#. Copy the run_DMS.py script and the run_DMS.yaml config file into your working directory.
#. Fill out the yaml file with the appropriate inputs.
#. Once happy with the input parameters, enter the follwing in your terminal:

    .. code-block:: bash

        python run_DMS.py run_DMS.yaml

To use the light curve fitting capabilities (if installed), simply follow the same procedure with the fit_lightcurves.py and .yaml files.