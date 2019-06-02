#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 09:50:51 2019

@author: julia
"""

import xarray as xr

# xr.open_mfdataset
# xr.open_dataset

path = "/Volumes/Elements/2019_Sonne/"
ceilo = xr.open_dataset(path + "ceilometer/20190407_RV Sonne_CHM188105_000.nc")
ceilo.cbh # layer 1

