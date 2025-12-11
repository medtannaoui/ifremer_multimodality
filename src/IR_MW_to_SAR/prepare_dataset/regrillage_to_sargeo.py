import pyproj
import pyresample
import numpy as np
import xarray as xr
import pickle as pkl

def regrid_swath_to_aeqd(values, x, y, clon, clat, max_r=300, dxy=2):
    """
    Reproject a swath (x,y,values) already in KM into AEQD 2-km grid.
    """

    out_n = int(2 * max_r / dxy) + 1

    x_grid = np.linspace(-max_r, max_r, out_n)
    y_grid = np.linspace(-max_r, max_r, out_n)

    # AEQD projection centered on cyclone
    proj = pyproj.Proj(proj="aeqd", lat_0=clat, lon_0=clon, ellps="WGS84", units="km")

    area = pyresample.geometry.create_area_def(
        "AEQD",
        proj.crs,
        center=(0,0),
        units="km",
        shape=(out_n,out_n),
        resolution=(dxy,dxy)

    )

  
    swath = pyresample.geometry.SwathDefinition(lons=x, lats=y)

    # nearest neighbor resampling (same as SAR)
    resampled = pyresample.kd_tree.resample_nearest(
        swath,
        values,
        area,
        radius_of_influence=1000,
        fill_value=np.nan
    )

    return resampled.astype(np.float32), x_grid, y_grid


def convert_mw_to_xy(coloc, clon, clat, max_r=300, dxy=2):
    """Reprojette dynamiquement TOUS les champs MW présents dans le coloc."""

    results = {}

    x_mw = coloc["x_mw"].values
    y_mw = coloc["y_mw"].values

    # Détection automatique des variables MW
    mw_vars = [v for v in coloc.data_vars if v.startswith("mw_TB_")]

    if len(mw_vars) == 0:
        print("⚠️ Aucun champ MW trouvé dans ce coloc.")
        return results

    for var in mw_vars:
        

        data = coloc[var].values

        regrid, xg, yg = regrid_swath_to_aeqd(
            values=data,
            x=x_mw,
            y=y_mw,
            clon=clon,
            clat=clat,
            max_r=max_r,
            dxy=dxy
        )

        results[var + "_aeqd"] = (("y_sar", "x_sar"), regrid)

    return results



def convert_ir_primed_to_xy(coloc, clon, clat, max_r=300, dxy=2):
    
    x_ir = coloc["x_ir_primed"].values
    y_ir = coloc["y_ir_primed"].values
    ir_img = coloc["ir_tcprimed"].values

    regrid, xg, yg = regrid_swath_to_aeqd(
        values=ir_img,
        x=x_ir,
        y=y_ir,
        clon=clon,
        clat=clat,
        max_r=max_r,
        dxy=dxy
    )

    return {"ir_tcprimed_aeqd": (("y_sar","x_sar"), regrid)}


def main(pkl_path="/scale/user/mtannaou/alternance/src/sargeo_primed_colocs/coloc_primed_sargeo_v1.pkl"):
    
    with open(pkl_path, "rb") as f:
        colocs = pkl.load(f)

    for coloc in colocs:

        clon = float(coloc["lon_centre"].values)
        clat = float(coloc["lat_centre"].values)

        mw_fields = convert_mw_to_xy(coloc, clon, clat)
        for k, v in mw_fields.items():
            coloc[k] = v

        ir_fields = convert_ir_primed_to_xy(coloc, clon, clat)
        for k, v in ir_fields.items():
            coloc[k] = v

    with open("/scale/user/mtannaou/alternance/src/sargeo_primed_colocs/coloc_primed_sargeo_regridded.pkl","wb") as f:
        pkl.dump(colocs, f)

    print("DONE!")
