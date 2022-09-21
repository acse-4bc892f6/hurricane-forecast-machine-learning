#!/usr/bin/env python

from setuptools import setup, Extension

setup(name='Hugo',
      version='1.0',
      description='Hurricane Forecast Tool',
      author='ACSE project',
      packages=['HurricaneForecast'],
      package_dir = {'HurricaneForecast': 'HurricaneForecast'},
      include_package_data=False
     )
