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


Alternatively, the package can be grabbed from GitHub (inlcludes all pipeline stages as well as tutorial notebooks, etc.) via:

.. code-block:: bash

    git clone https://github.com/radicamc/exoTEDRF.git
    cd exoTEDRF
    python setup.py install


.. note::
    **exoTEDRF** is currently compatible with python 3.10.4 and v1.12.5 of the `STScI JWST pipeline <https://github.com/spacetelescope/jwst>`_. If you wish to run a
    different version of jwst, certain functionalities of **exoTEDRF** may not work.