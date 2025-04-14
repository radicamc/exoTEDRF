#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

setup(name='exotedrf',
      version='2.2.0',
      license='MIT',
      author='Michael Radica',
      author_email='radicamc@uchicago.edu',
      packages=['exotedrf'],
      include_package_data=True,
      url='https://github.com/radicamc/exoTEDRF',
      description='Tools for Reduction of JWST TSOs',
      package_data={'': ['README.md', 'LICENSE']},
      install_requires=['applesoss==2.1.0', 'astropy', 'astroquery',
                        'bottleneck', 'corner', 'jwst==1.17.1', 'matplotlib',
                        'more_itertools', 'numpy', 'pandas', 'ray',
                        'scikit-learn', 'scipy', 'spectres', 'tqdm',
                        'pastasoss', 'pyyaml'],
      extras_require={'stage4': ['exotedrf', 'exouprf', 'exotic_ld', 'h5py'],
                      'webbpsf': ['exotedrf', 'webbpsf>=1.1.1']},
      classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.10',
        ],
      )
