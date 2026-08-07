Installation
============

The latest release of **exoTEDRF** can be downloaded from PyPI by running:

.. code-block:: bash

    pip install exotedrf

.. attention::
    Depending on the operating system, the package jmespath may fail to install. In this case, run

    .. code-block:: bash

        pip install jmespath

    and then proceed with the **exoTEDRF** installation.


The default pip installation only includes Stages 1 to 3. Stage 4 can be included via specifying the following option during installation:

.. code-block:: bash

    pip install exotedrf[stage4]

If you plan on using ExoTiC-LD to derive limb darkening coefficients, you will also need to install its reference files manually.
See the `ExoTiC-LD documentation <https://exotic-ld.readthedocs.io/en/latest/views/installation.html>`_ for information on how to do this.
Moreover, if you intend to use the power2 limb darkening law (recommended) you will need to install the GitHub version of ExoTiC-LD as this functionality is not yet available via pip.

.. attention::
    At time of writing the pip installable version of ExoTiC-LD is using a deprecated call to pkg_resources, which will throw an error when ExoTiC-LD is imported.
    This is fixed on the GitHub version, so it is recommended to install from GitHub (>=3.2.2) in all cases.


Alternatively, **exoTEDRF** can be grabbed from GitHub (inlcludes all pipeline stages as well as tutorial notebooks, etc.) via:

.. code-block:: bash

    git clone https://github.com/radicamc/exoTEDRF.git
    cd exoTEDRF
    python setup.py install


.. note::
    **exoTEDRF** is currently compatible with python 3.13.14 and v3.0.0 of the `STScI JWST pipeline <https://github.com/spacetelescope/jwst>`_. If you wish to run a
    different version of jwst, certain functionalities of **exoTEDRF** may not work.