#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

setup(name='supreme_spoon',
      version='1.2.0',
      license='MIT',
      author='Michael Radica',
      author_email='michael.radica@umontreal.ca',
      packages=['supreme_spoon'],
      include_package_data=True,
      url='https://github.com/radicamc/supreme-spoon',
      description='Tools for Reduction of NIRISS/SOSS TSOs',
      package_data={'': ['README.md', 'LICENSE']},
      install_requires=['applesoss==2.1.0', 'astropy', 'astroquery',
                        'bottleneck', 'corner', 'jwst==1.8.5', 'matplotlib',
                        'more_itertools', 'numpy', 'pandas', 'opencv-python',
                        'ray', 'scikit-learn', 'scipy', 'tqdm', 'pastasoss',
                        'pyyaml'],
      extras_require={'stage4': ['supreme_spoon', 'cython', 'dynesty==1.2.2',
                                 'exotic_ld', 'h5py', 'juliet'],
                      'webbpsf': ['supreme_spoon', 'webbpsf>=1.1.1']},
      classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.10',
        ],
      )
