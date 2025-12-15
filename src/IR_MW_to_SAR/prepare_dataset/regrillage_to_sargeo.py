
import pyproj
import numpy as np
import pandas as pd
import xarray as xr
import pyresample
from pyresample.bilinear import XArrayBilinearResampler
import pickle as pkl

import src.IR_MW_to_SAR.prepare_dataset.regrillage_to_sargeo as regimwsar
import importlib 
importlib.reload(regimwsar)




def aeqd(data_org, clon, clat, lon, lat,
         max_r=300, dxy=2, kind="nearest", radius_resample=10000):

    out_n = int(2 * max_r / dxy) + 1
    x_aeqd = np.arange(out_n) * dxy - (out_n - 1) * dxy / 2
    y_aeqd = (np.arange(out_n) * dxy - (out_n - 1) * dxy / 2)

    aeqd_proj = pyproj.Proj(
        proj="aeqd",
        lat_0=clat,
        lon_0=clon,
        ellps="WGS84",
        units="km"
    )
    data = np.asarray(data_org.values)
    lon = np.asarray(lon)
    lat = np.asarray(lat)
    
    lon = (lon +180)%360 -180
    clon = (clon + 180)%360 - 180

    # --- Vérification cruciale
    if data.shape != lon.shape or data.shape != lat.shape:
        raise ValueError(
            f"Shape mismatch: data {data.shape}, lon {lon.shape}, lat {lat.shape}"
        )
    area_def = pyresample.geometry.create_area_def(
        area_id="AEQD",
        projection=aeqd_proj.crs,
        center=(0, 0),
        units="km",
        shape=(out_n, out_n),
        resolution=(dxy, dxy)
    )
    
    lon, lat = pyresample.utils.check_and_wrap(lon, lat)
   
    swath = pyresample.geometry.SwathDefinition(lon, lat)
    radius_resample = radius_resample
    if kind == "nearest":
        data_aeqd = pyresample.kd_tree.resample_nearest(
            swath,
            data_org.values,
            area_def,
            radius_of_influence=radius_resample,
            fill_value=np.nan
        )
        data_aeqd = data_aeqd[::-1, :]
    else:
        raise NotImplementedError("Only nearest implemented")

    return data_aeqd.astype(np.float32), x_aeqd.astype(np.float32), y_aeqd.astype(np.float32)




def add_aeqd_fields_to_coloc(coloc, max_r=300, dxy=2):
    
    clon = float(coloc["lon_centre"].values)
    clat = float(coloc["lat_centre"].values)
    ir_aeqd, x_aeqd, y_aeqd = aeqd(
        data_org=coloc["ir_tcprimed"],
        clon=clon,
        clat=clat,
        lon=coloc["lon_ir_primed"],
        lat=coloc["lat_ir_primed"],
        max_r=max_r,
        dxy=dxy,
        radius_resample=10000
    )
    coloc["ir_tcprimed_aeqd"] = (("y_sar", "x_sar"), ir_aeqd)
    mw_vars = [v for v in coloc.data_vars if v.startswith("mw_TB_") and "aeqd" not in v]
    
    for var in mw_vars:
        mw_aeqd, _, _ = aeqd(
            data_org=coloc[var],
            clon=clon,
            clat=clat,
            lon=coloc["lon_mw"],
            lat=coloc["lat_mw"],
            max_r=max_r,
            dxy=dxy,
            radius_resample=15000
        )
        coloc[var + "_aeqd"] = (("y_sar", "x_sar"), mw_aeqd)

    coloc["x_ir_aeqd"] = ("x_sar", x_aeqd)
    coloc["y_ir_aeqd"] = ("y_sar", y_aeqd)
    coloc["x_mw_aeqd"] = ("x_sar", x_aeqd)
    coloc["y_mw_aeqd"] = ("y_sar", y_aeqd)

    return coloc



def process_all_colocs(colocs, max_r=300, dxy=2,save=False):
    out = []
    for i, coloc in enumerate(colocs):
        
        out.append(add_aeqd_fields_to_coloc(coloc, max_r, dxy))
    print(len(out),"colocs regrilled")
    if save : 
        with open("/scale/user/mtannaou/alternance/src/IR_MW_to_SAR/data/ir_mw_sar.pkl","wb") as f:
            pkl.dump(f)
    return out


if __name__ == "__main__":
    pkl_path = "/scale/user/mtannaou/alternance/src/IR_MW_to_SAR/data/coloc_primed_sargeo_v1.pkl"
    with open(pkl_path,"rb") as f:
        colocs = pkl.load(f)
    
    regimwsar.process_all_colocs(colocs=colocs)
    


















































# def aeqd(data_org, clon, clat, lon, lat, max_r=300, dxy=0.5, kind="nearest", radius_resample=4000):
       
#         out_nx = out_ny = int(2 * max_r/dxy) + 1
#         xax = np.arange(out_nx) * dxy - (out_nx-1)*dxy/2
#         yax = -(np.arange(out_ny) * dxy - (out_ny-1)*dxy/2)

#         aeqd_proj = pyproj.Proj(proj="aeqd", lat_0=clat, lon_0=clon, ellps="WGS84", units="km")

#         aeqd_areadef = pyresample.geometry.create_area_def(
#             'AEQD', 
#             aeqd_proj.crs, 
#             center=(0, 0), 
#             units='km',
#             shape=(out_ny, out_nx),  
#             resolution=(dxy, dxy)
#             )
        
#         RADIUS_RESAMPLE = radius_resample # m
#         NEIGHBORS_RESAMPLE = 16
#         swath_longitude, swath_latitude = pyresample.utils.check_and_wrap(lon, lat)
#         swath = pyresample.SwathDefinition(swath_longitude, swath_latitude)

#         if True : 
#             if kind == "nearest":
#                 reso = dxy*1000
#                 data_aeqd = (
#                     ["y_sar","x_sar"], 
#                     pyresample.kd_tree.resample_nearest(
#                         swath, data_org.values, aeqd_areadef, radius_of_influence=RADIUS_RESAMPLE, fill_value=np.nan
#                     ))
#             elif kind == "linear":
#                 bt_resampler = XArrayBilinearResampler(
#                     swath, aeqd_areadef, RADIUS_RESAMPLE, neighbours=NEIGHBORS_RESAMPLE
#                     )
#                 data_aeqd= (1,bt_resampler.resample(data_org))

   

    
#         return data_aeqd