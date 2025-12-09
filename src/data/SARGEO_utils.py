import os
import logging
import argparse

import cv2
import pyproj
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(asctime)s %(levelname)s %(name)s %(lineno)d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

def proj(values, src_xax, src_yax, src_proj, dst_proj, dst_xax, dst_yax, undef=-999.0):
    """
    Parameters
    ----------
    values : np.ndarray
        2d array of the data to be projected.
    src_xax : np.ndarray
        1d array of the x-axis coordinates for 2nd dimension of source image
    src_yax : np.ndarray
        1d array of the y-axis coordinates for 1nd dimension of source image
    src_proj : pyproj.Proj
        pyproj.Proj object for the source image
    dst_xax : np.ndarray
        1d array of the x-axis coordinates for 2nd dimension of destination image
    dst_yax : np.ndarray
        1d array of the y-axis coordinates for 1nd dimension of destination image
    dst_proj : pyproj.Proj
        pyproj.Proj object for the destination image
    undef : int or float, default -999.0
        In the projection, the values at out of source extent will be `undef` and converted to np.nan
    
    Returns
    -------
    dst_image : np.ndarray
        2d array projected onto the `dst_proj` projection
    """
    if not isinstance(values, np.ndarray):
        raise ValueError("`values` must be np.ndarray")

    if (src_yax[-1] - src_yax[0]) < 0:
        values = values[::-1, :]
    if (src_xax[-1] - src_xax[0]) < 0:
        values = values[:, ::-1]

    dst_lons, dst_lats = dst_proj(dst_xax, dst_yax, inverse=True)
    dst_xax_on_src, dst_yax_on_src = src_proj(dst_lons, dst_lats)

    x0, y0 = np.min(src_xax), np.min(src_yax)
    x1, y1 = np.max(src_xax), np.max(src_yax)
    xscale = abs(x1-x0)/src_xax.size
    yscale = abs(y1-y0)/src_yax.size
    x_inds = (dst_xax_on_src - min(x0, x1)) / xscale
    y_inds = (dst_yax_on_src - min(y0, y1)) / yscale

    dst = cv2.remap(values.astype(np.float32), x_inds.astype(np.float32), y_inds.astype(np.float32), \
        interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=undef)
    dst = np.where(dst==undef, np.nan, dst)
    return dst

def aeqd(values, src_xax, src_yax, src_proj, clon, clat, max_r, dr, invert_yaxis=True, undef=np.nan, units="km"):
    """
    Parameters
    ----------
    values : np.ndarray
        2d array of the data to be projected.
    src_xax : np.ndarray
        1d array of the x-axis coordinates for 2nd dimension of source image
    src_yax : np.ndarray
        1d array of the y-axis coordinates for 1nd dimension of source image
    src_proj : pyproj.Proj
        pyproj.Proj object for the source image
    clon : int or float (in degree)
        central longitude
    clat : int or float (in degree)
        central latitude
    max_r : int or float (in `units`)
        maximum radius of projection
    dr : int or float (in `units`)
        Δradius
    invert_yaxis : bool, default True
        If True, interprate upper-left as (0,0) origin. If False, interprate lower-left as (0,0) origin.
    undef : int or float, default np.nan
        In the projection, the values at out of source extent will be `undef` and converted to np.nan
    units : str, default "km"
        The unit of the radius. Default is "km"
    
    Returns
    -------
    aeqd : np.ndarray
        2d array projected onto the azimuthal equidistant projection
    aeqd_proj : pyproj.Proj
        The aeqd projection object
    """
    if not isinstance(values, np.ndarray):
        raise ValueError("`values` must be np.ndarray")

    aeqd_proj = pyproj.Proj(proj="aeqd", lat_0=clat, lon_0=clon, ellps="WGS84", units=units)
    
    if invert_yaxis:
        dst_ys, dst_xs = np.mgrid[max_r:-max_r-dr:-dr, -max_r:max_r+dr:dr]
    else:
        dst_ys, dst_xs = np.mgrid[-max_r:max_r+dr:dr, -max_r:max_r+dr:dr]

    aeqd = proj(values, src_xax, src_yax, src_proj, aeqd_proj, dst_xs, dst_ys, undef=undef)
    aeqd = xr.DataArray(aeqd, dims=("y","x"), coords={"y":dst_ys[:,0], "x":dst_xs[0,:]})
    return aeqd, aeqd_proj