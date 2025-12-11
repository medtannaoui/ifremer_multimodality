#%%
import os

import pyproj
import numpy as np
import pandas as pd
import xarray as xr
import pyresample
from pyresample.bilinear import XArrayBilinearResampler


class NESDIS:
    @classmethod
    def get_var_from_ij(cls, sar_org, i, j, varname="lat"):
        """
        Varname is one of the following:
        - lat
        - lon
        - incid
        - rlook
        
        Parameters
        ----------
        sar_org : xarray.Dataset
            Original SAR data from NESDIS
        i : int
            i (x) index
        j : int
            j (y) index
        varname : str, optional
            The variable name to get. Default is "lat".
        """
        var = 0
        for xfit in range(sar_org["xfit"].size):
            coef = sar_org[f"{varname}_coef"].isel(xfit=xfit).item()
            xexp = sar_org[f"{varname}_xexp"].isel(xfit=xfit).item()
            yexp = sar_org[f"{varname}_yexp"].isel(xfit=xfit).item()
            var = var + coef * (i**xexp) * (j**yexp)
        return var
    
    @classmethod
    def get_xy_from_lonlat(cls, sar_org, lon, lat):
        return cls.get_ij_from_lonlat(sar_org, lon, lat)
    
    @classmethod
    def get_ij_from_lonlat(cls, sar_org, lon, lat):
        """
        Get the i (x), j (y) indices from the given longitude and latitude.
        """
        i, j = 0, 0
        for xfit in range(sar_org["xfit"].size):
            coef = sar_org["i_coef"].isel(xfit=xfit).item()
            lonexp = sar_org["i_xexp"].isel(xfit=xfit).item()
            latexp = sar_org["i_yexp"].isel(xfit=xfit).item()
            i = i + coef * (lon**lonexp) * (lat**latexp)

            coef = sar_org["j_coef"].isel(xfit=xfit).item()
            lonexp = sar_org["j_xexp"].isel(xfit=xfit).item()
            latexp = sar_org["j_yexp"].isel(xfit=xfit).item()
            j = j + coef * (lon**lonexp) * (lat**latexp)
        return i, j
    
    @classmethod
    def get_lonlat_from_xy(cls, sar_org, x, y):
        lon = cls.get_var_from_ij(sar_org, x, y, varname="lon")
        lat = cls.get_var_from_ij(sar_org, x, y, varname="lat")
        return lon, lat
    
    @classmethod
    def get_lonlat_from_ij(cls, sar_org, i, j):
        return cls.get_lonlat_from_xy(sar_org, i, j)

    @classmethod
    def aeqd(cls, sar_org, clon, clat, max_r=300, dxy=0.5, kind="nearest", varnames=["sar_wind"], include_org=False, 
             org_varnames=["sar_wind"],match=None):
        """
        Project SAR data on azimuthal equidistant projection centered at (clon, clat)
        
        Parameters
        ----------
        sar_org : xarray.Dataset
            Original SAR data from NESDIS
        clon : float
            Center longitude in degrees
        clat : float 
            Center latitude in degrees
        max_r : float
            Maximum radius for output (km)
        dxy : float
            Output spatial resolution in x/y axes (km)
        kind : str
            Interpolation method. "nearest" or "linear". "nearest" is much faster than "linear".
        varnames : list
            Variable names to be projected
        include_org : bool
            If True, include original variables in the output dataset
        
        Returns
        -------
        xarray.Dataset
            Projected SAR data on azimuthal equidistant projection
        """
        out_nx = out_ny = int(2 * max_r/dxy) + 1
        xax = np.arange(out_nx) * dxy - (out_nx-1)*dxy/2
        yax = -(np.arange(out_ny) * dxy - (out_ny-1)*dxy/2)

        aeqd_proj = pyproj.Proj(proj="aeqd", lat_0=clat, lon_0=clon, ellps="WGS84", units="km")

        aeqd_areadef = pyresample.geometry.create_area_def(
            'AEQD', 
            aeqd_proj.crs, 
            center=(0, 0), 
            units='km',
            shape=(out_ny, out_nx),  
            resolution=(dxy, dxy)
            )
        
        sar_aeqd = xr.Dataset(coords={"x_sar":xax.astype(np.float32), "y_sar":yax.astype(np.float32)})
        sar_aeqd["clon"] = ([], clon, {"standard_name": "clon", "long_name": "center longitude", "units": "degrees"})
        sar_aeqd["clat"] = ([], clat, {"standard_name": "clat", "long_name": "center latitude", "units": "degrees"})
        
        # sar_aeqd["sar_lon"] = (("sar_lon",), sar_org["owiLon"].values)
        # sar_aeqd["sar_lat"] = (("sar_lat",), sar_org["owiLat"].values)
        # sar_aeqd["sar_acquisition_time"] = sar_org["sar_acquisition_time"]
        
        # Resampling
        RADIUS_RESAMPLE = 1000 # m
        NEIGHBORS_RESAMPLE = 16
        swath_longitude, swath_latitude = pyresample.utils.check_and_wrap(sar_org["owiLon"], sar_org["owiLat"])
        swath = pyresample.SwathDefinition(swath_longitude, swath_latitude)

        for varname in varnames:
            if kind == "nearest":
                reso = dxy*1000
                sar_aeqd[varname] = (
                    ["y_sar","x_sar"], 
                    pyresample.kd_tree.resample_nearest(
                        swath, sar_org[varname].values, aeqd_areadef, radius_of_influence=RADIUS_RESAMPLE, fill_value=np.nan
                    ))
            elif kind == "linear":
                bt_resampler = XArrayBilinearResampler(
                    swath, aeqd_areadef, RADIUS_RESAMPLE, neighbours=NEIGHBORS_RESAMPLE
                    )
                sar_aeqd[varname] = bt_resampler.resample(sar_org[varname])

        sar_aeqd["kind"] = ([], kind, {"standard_name": "kind", "long_name": "kind of interpolation in resampling"})
        sar_aeqd["dxy"] = ([], dxy, {"standard_name": "dxy", "long_name": "Spatial resolution in x/y axes", "units": "km"})
        sar_aeqd["max_r"] = ([], max_r, {"standard_name": "max_r", "long_name": "Maximum radius for output", "units": "km"})
        sar_aeqd["x_sar"] = (("x_sar",), sar_aeqd["x_sar"].values)
        sar_aeqd["y_sar"] = (("y_sar",), sar_aeqd["y_sar"].values)

        # Attributes
        sar_aeqd["x_sar"].attrs.update({"standard_name": "x_sar", "long_name": "x from storm center", "units": "km"})
        sar_aeqd["y_sar"].attrs.update({"standard_name": "y_sar", "long_name": "y from storm center", "units": "km"})
        for varname in varnames:
            sar_aeqd[varname].attrs.update({"standard_name": varname+"_500m", "long_name": sar_org[varname].attrs["long_name"]+f" resampled on {reso}-m azimuthal equidistant projection", "units": sar_org[varname].attrs["units"]})

    
        return sar_aeqd, aeqd_proj