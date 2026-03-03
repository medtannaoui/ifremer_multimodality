from pyresample import geometry, kd_tree
import numpy as np
import xarray as xr
import pandas as pd
import logging

def extract_coords_from_point(wkt_point: str):
    """
    Extrait les coordonnées (lon, lat) depuis une chaîne WKT de type 'POINT (x y)'.

    Exemples :
    >>> extract_coords_from_point('POINT (71.84687300692487 -22.71783074014432)')
    (71.84687300692487, -22.71783074014432)
    """
    try:
        # retirer le préfixe 'POINT' et les parenthèses
        content = wkt_point.strip().removeprefix('POINT').strip(' ()')
        # séparer en deux nombres
        x_str, y_str = content.split()
        return float(x_str), float(y_str)
    except Exception as e:
        raise ValueError(f"Format invalide pour un POINT WKT : {wkt_point}") from e


def create_target_grid(center_lon, center_lat, size_km, resolution_km):
    """Creates a target grid for regridding."""
    km_per_deg_lat = 111.0
    km_per_deg_lon = km_per_deg_lat * np.cos(np.deg2rad(center_lat))

    size_deg_lat = size_km / km_per_deg_lat
    size_deg_lon = size_km / km_per_deg_lon

    lat_min = center_lat - size_deg_lat / 2
    lat_max = center_lat + size_deg_lat / 2
    lon_min = center_lon - size_deg_lon / 2
    lon_max = center_lon + size_deg_lon / 2

    resolution_deg_lat = resolution_km / km_per_deg_lat
    resolution_deg_lon = resolution_km / km_per_deg_lon

    n_lat = int(size_km / resolution_km)
    n_lon = int(size_km / resolution_km)

    target_lat = np.linspace(lat_min, lat_max, n_lat)
    target_lon = np.linspace(lon_min, lon_max, n_lon)

    return xr.Dataset(coords={"lat": target_lat, "lon": target_lon})

def regrid_sar(collocs, target_grid, data_vars=["wind_speed"], margin_deg=1):
    """Loads and regrids SAR data using pyresample for irregular grid interpolation."""

    datasets = []
    for cl in collocs:
        data = xr.open_dataset(cl)
        if data is not None:
            datasets.append(data.set_coords(["lon", "lat"]))

    if not datasets:
        return None
    
    target_grid = target_grid.assign_coords(lon=(((target_grid.lon + 180) % 360) - 180))
    
    grid_lons = target_grid.lon.values.copy()


    tgt_lon_vals = target_grid.lon.values.copy()

    # --- Build target definition (regular grid) ---
    tgt_lon2d, tgt_lat2d = np.meshgrid(
        tgt_lon_vals, target_grid.lat.values
    )
    target_def = geometry.GridDefinition(lons=tgt_lon2d, lats=tgt_lat2d)

    regridded_datasets = []

    for sar_data in datasets:

        # Normalise longitude
        sar_data["lon"] = (sar_data["lon"] + 180) % 360 - 180

        valid = np.isfinite(sar_data["wind_speed"])
        sar_data = sar_data.where(valid, drop=True)

        lon_min = np.min(grid_lons) - margin_deg
        lon_max = np.max(grid_lons) + margin_deg
        lat_min = np.min(target_grid.lat.values) - margin_deg
        lat_max = np.max(target_grid.lat.values) + margin_deg

        ds_lons = sar_data.lon.copy()

        lat_mask = (sar_data.lat >= lat_min) & (sar_data.lat <= lat_max)
        lon_mask = (ds_lons >= lon_min) & (ds_lons <= lon_max)

        mask = lat_mask & lon_mask
        sar_data = sar_data.where(mask, drop=True)

        # Must be 1D lon/lat
        if sar_data.lon.ndim != 1 or sar_data.lat.ndim != 1:
            raise ValueError("Regular grid expected: lon and lat must be 1D")

        src_lon_vals = sar_data.lon.values.copy()


        # --- Build source definition (regular grid) ---
        src_lon2d, src_lat2d = np.meshgrid(
            src_lon_vals, sar_data.lat.values
        )
        source_def = geometry.SwathDefinition(lons=src_lon2d, lats=src_lat2d)

        # Prepare output dataset
        dst = xr.Dataset()

        for var in data_vars:
            if var not in sar_data:
                continue

            src = sar_data[var].values

            if np.isfinite(src).sum().item() == 0:
                #print(f"All values are NaN for variable {var}, skipping")
                continue

            # Interpolation
            dst_values = kd_tree.resample_nearest(
                source_def,
                src,
                target_def,
                radius_of_influence=3000,
                fill_value=np.nan,
            )

            dst[var] = (("lat", "lon"), dst_values)
            # Update the variable with regridded values
            
            # Add the target coordinates
            dst = dst.assign_coords(
                    lat=target_grid.lat,
                    lon=target_grid.lon,
                    scat_time=(
                        pd.to_datetime(sar_data.attrs['measurementDate'])
                    ).to_datetime64(),
                )
            
            regridded_datasets.append(dst[data_vars])

    if not regridded_datasets:
        return None

    return xr.concat(regridded_datasets, dim="sar_item")

def regrid_scat(collocs, target_grid, data_vars=["wind_speed"], margin_deg=1):
    """Loads and regrids SCAT data using pyresample for irregular grid interpolation."""
    #if not isinstance(collocs, list):
      #  return None
    datasets = []
    for cl in collocs:
        data = xr.open_dataset(cl)
        if data is not None:
            datasets.append(data.set_coords(["lon", "lat"]))

    if not datasets:
        return None
    
    target_grid = target_grid.assign_coords(lon=(((target_grid.lon + 180) % 360) - 180))

    regridded_datasets = []
    for scat_data in datasets:
        scat_data["lon"] = (scat_data["lon"] + 180) % 360 - 180
        mask_valid = np.isfinite(scat_data['lat']) & np.isfinite(scat_data['lon']) & np.isfinite(scat_data['wind_speed'])
        scat_data = scat_data.where(mask_valid, drop=True)

        grid_lons = target_grid.lon.values.copy()

        tgt_lon_vals = target_grid.lon.values.copy()

        lon_min = np.min(grid_lons) - margin_deg
        lon_max = np.max(grid_lons) + margin_deg
        lat_min = np.min(target_grid.lat.values) - margin_deg
        lat_max = np.max(target_grid.lat.values) + margin_deg

        ds_lons = scat_data.lon.copy()

        lat_mask = (scat_data.lat >= lat_min) & (scat_data.lat <= lat_max)
        lon_mask = (ds_lons >= lon_min) & (ds_lons <= lon_max)

        mask = lat_mask & lon_mask

        scat_data = scat_data.where(mask, drop=True)
        # Extract the irregular grid coordinates
        lons = scat_data.lon.values.copy()
        lats = scat_data.lat.values

        # Create pyresample geometries
        source_def = geometry.SwathDefinition(lons=lons, lats=lats)

        # Create a grid definition for the target based on the target_grid coordinates
        target_lon_2d, target_lat_2d = np.meshgrid(
            tgt_lon_vals, target_grid.lat.values
        )
        target_def = geometry.SwathDefinition(lons=target_lon_2d, lats=target_lat_2d)

        # Perform the resampling
        regridded_data = scat_data.copy(deep=False, data={})

        for var in data_vars:
            if var not in scat_data:
                print(
                    f"No variable named {var} in  scat data among {list(scat_data)}, skipping"
                )
                continue
            source_values = scat_data[var].values
            if np.isfinite(source_values).sum().item() == 0:
                #print(f"All values are NaN for variable {var}, skipping")
                continue
            #print('before', np.isfinite(scat_data[var].values).mean())
            #print(f'{scat_data[var].sizes=}')
            # Resample each data variable
            regridded_values = kd_tree.resample_nearest(
                source_def,
                source_values,
                target_def,
                radius_of_influence=25000,  # 25 km influence radius
                fill_value=np.nan,
            )

            #print('after', np.isfinite(regridded_values).mean())
            # Update the variable with regridded values
            regridded_data[var] = (("lat", "lon"), regridded_values.squeeze())
        
        if scat_data.time.notnull().sum().item()>0:
            d_min_scat = pd.to_datetime(scat_data.time.min(skipna=True).item())
            d_max_scat = pd.to_datetime(scat_data.time.max(skipna=True).item())

            # Add the target coordinates
            regridded_data = regridded_data.assign_coords(
                lat=target_grid.lat,
                lon=target_grid.lon,
                scat_time=(
                    pd.to_datetime(d_min_scat)
                    + (
                        pd.to_datetime(d_max_scat)
                        - pd.to_datetime(d_min_scat)
                    )
                    / 2
                ).to_datetime64(),
            )
            regridded_datasets.append(regridded_data[data_vars])

    if not regridded_datasets:
        return None

    return xr.concat(regridded_datasets, dim="scat_item")
    
def regrid_radiometer(collocs, target_grid, data_vars=["wind_speed"], margin_deg=1):
    """Regrid regular-gridded radiometer data to another regular grid using pyresample."""
    
    datasets = []
    for cl in collocs:
        data = xr.open_dataset(cl)
        if data is not None:
            datasets.append(data.set_coords(["lon", "lat"]))

    if not datasets:
        return None
    target_grid = target_grid.assign_coords(lon=(((target_grid.lon + 180) % 360) - 180))
    # --- Build target definition (regular grid) ---
    tgt_lon2d, tgt_lat2d = np.meshgrid(
        target_grid.lon.values, target_grid.lat.values
    )
    target_def = geometry.GridDefinition(lons=tgt_lon2d, lats=tgt_lat2d)

    regridded_datasets = []

    for radio_data in datasets:

        # Normalise longitude
        radio_data["lon"] = (radio_data["lon"] + 180) % 360 - 180

        # Filter invalids
        valid = (
            np.isfinite(radio_data["lat"])
            & np.isfinite(radio_data["lon"])
            & np.isfinite(radio_data["wind_speed"])
        )
        radio_data = radio_data.where(valid, drop=True)

        lon_min = np.min(target_grid.lon.values) - margin_deg
        lon_max = np.max(target_grid.lon.values) + margin_deg
        lat_min = np.min(target_grid.lat.values) - margin_deg
        lat_max = np.max(target_grid.lat.values) + margin_deg

        mask = (
            (radio_data.lat >= lat_min)
            & (radio_data.lat <= lat_max)
            & (radio_data.lon >= lon_min)
            & (radio_data.lon <= lon_max)
        )
        radio_data = radio_data.where(mask, drop=True)

        # Must be 1D lon/lat
        if radio_data.lon.ndim != 1 or radio_data.lat.ndim != 1:
            raise ValueError("Regular grid expected: lon and lat must be 1D")

        # --- Build source definition (regular grid) ---
        src_lon2d, src_lat2d = np.meshgrid(
            radio_data.lon.values, radio_data.lat.values
        )
        source_def = geometry.GridDefinition(lons=src_lon2d, lats=src_lat2d)

        # Prepare output dataset
        dst = xr.Dataset()

        for var in data_vars:
            if var not in radio_data:
                continue

            src = radio_data[var].values

            # Interpolation
            dst_values = kd_tree.resample_nearest(
                source_def,
                src,
                target_def,
                radius_of_influence=25000,
                fill_value=np.nan,
            )

            dst[var] = (("lat", "lon"), dst_values)

        # Time stamp
        dmin = pd.to_datetime(radio_data.attrs['measurementStartDate'])
        dmax = pd.to_datetime(radio_data.attrs['measurementStopDate'])

        dst = dst.assign_coords(
            lat=target_grid.lat,
            lon=target_grid.lon,
            radio_time=(dmin + (dmax - dmin) / 2).to_datetime64(),
        )

        regridded_datasets.append(dst[data_vars])

    if not regridded_datasets:
        return None

    return xr.concat(regridded_datasets, dim="radio_item")

# def regrid_era5(ds, df, data_vars=["wind_speed"], margin_deg=2):
#     """Loads and regrids ERA5 data using pyresample for grid interpolation."""
    
#     # Normalise longitude once before the loop
#     ds = ds.copy()  # Work on a copy to avoid modifying the original
#     ds["lon"] = (ds["lon"] + 180) % 360 - 180

#     ds = ds.assign_coords(
#     lon=(ds.lon % 360)
# )

#     ds = ds.sortby("lon")

#     regridded_datasets = []

#     for index, row in df.iterrows():
        
#         if not row['bt_pos']:
#             continue
#         grid = row['grids']

#         grid.lon.values[grid.lon.values<-180] += 360
#         grid.lon.values[grid.lon.values>180] -= 360

#         grid_lons = grid.lon.values.copy()

        
#         #print(grid_lons.min(), grid_lons.max())
#         # Detect dateline crossing
#         if np.ptp(grid_lons) > 180:
#             # Shift negative longitudes to make region continuous
#             grid_lons[grid_lons < 0] += 360
#             grid_crosses_dateline = True
#         else:
#             grid_crosses_dateline = False

#         # --- Build target definition (regular grid) ---
#         tgt_lon_vals = grid.lon.values.copy()

#         if grid_crosses_dateline:
#             tgt_lon_vals[tgt_lon_vals < 0] += 360

#         tgt_lon2d, tgt_lat2d = np.meshgrid(
#             tgt_lon_vals,
#             grid.lat.values
#         )

#         print(np.unique(tgt_lon_vals))

#         target_def = geometry.GridDefinition(lons=tgt_lon2d, lats=tgt_lat2d)
        
#         # Crop to region of interest for this timestep
#         lon_min = np.min(grid_lons) - margin_deg
#         lon_max = np.max(grid_lons) + margin_deg

#         lat_min = np.min(grid.lat.values) - margin_deg
#         lat_max = np.max(grid.lat.values) + margin_deg

#         ds_lons = ds.lon.copy()

#         if grid_crosses_dateline:
#             ds_lons = ds_lons.where(ds_lons >= 0, ds_lons + 360)

#         # lat_mask = (ds.lat >= lat_min) & (ds.lat <= lat_max)
#         # lon_mask = (ds_lons >= lon_min) & (ds_lons <= lon_max)

#         # mask = lat_mask & lon_mask
#         # ds_cropped = ds.where(mask, drop=True)

#         ds_cropped = ds.sel(
#     lat=slice(lat_min, lat_max),
#     lon=slice(lon_min, lon_max)
# )

#         # Must be 1D lon/lat
#         if ds_cropped.lon.ndim != 1 or ds_cropped.lat.ndim != 1:
#             raise ValueError("Regular grid expected: lon and lat must be 1D")

#         # --- Build source definition for this timestep ---
#         src_lon_vals = ds_cropped.lon.values.copy()

#         if grid_crosses_dateline:
#             src_lon_vals[src_lon_vals < 0] += 360

#         src_lon2d, src_lat2d = np.meshgrid(
#             src_lon_vals,
#             ds_cropped.lat.values
#         )
#         source_def = geometry.GridDefinition(lons=src_lon2d, lats=src_lat2d)

#         # Prepare output dataset for this timestep
#         dst = xr.Dataset()

#         for var in data_vars:
#             if var not in ds_cropped:
#                 continue

#             time_col = "valid_time" if "valid_time" in ds.coords else "time"

#             # Extract the data for this specific timestep
#             src = ds_cropped[var].isel({time_col:index}).values

#             if np.isfinite(src).sum() == 0:
#                 #print(f"All values are NaN for variable {var} at index {index}, skipping")
#                 continue

#             # Interpolation
#             dst_values = kd_tree.resample_nearest(
#                 source_def,
#                 src,
#                 target_def,
#                 radius_of_influence=31000,
#                 fill_value=np.nan,
#             )

#             dst[var] = (("lat", "lon"), dst_values)
        
#         # Add the target coordinates and expand dims to include era5_time
#         if len(dst.data_vars) > 0:
#             era5_time = pd.to_datetime(ds[time_col][index].values).to_datetime64()
            
#             dst = dst.assign_coords(
#                 lat=grid.lat,
#                 lon=grid.lon,
#                 era5_time=era5_time
#             )
            
#             regridded_datasets.append(dst)

#         #print(grid_crosses_dateline)

#     if not regridded_datasets:
#         return None
    
#     return regridded_datasets

# def regrid_era5(ds, df, data_vars=["wind_speed"], margin_deg=2):  ####CHAT
#     """
#     Regrid ERA5 data using pyresample to match target grids for all timesteps.
#     Uses a fixed cropped ERA5 region to avoid missing pixels progressively.
#     """
#     import numpy as np
#     import pandas as pd
#     import xarray as xr
#     from pyresample import geometry, kd_tree

#     # Normalize ERA5 longitudes once
#     ds = ds.copy()
#     ds["lon"] = (ds["lon"] + 360) % 360
#     ds = ds.sortby("lon")

#     # Compute union bounding box over all valid grids
#     all_lons, all_lats = [], []
#     for grid in df['grids']:
#         if grid is not None:
#             all_lons.append(grid.lon.values)
#             all_lats.append(grid.lat.values)

#     if not all_lons:
#         return None  # No valid grids

#     all_lons = np.concatenate(all_lons)
#     all_lats = np.concatenate(all_lats)

#     lon_min = all_lons.min() - margin_deg
#     lon_max = all_lons.max() + margin_deg
#     lat_min = all_lats.min() - margin_deg
#     lat_max = all_lats.max() + margin_deg

#     # Crop ERA5 once for all timesteps
#     ds_cropped = ds.sel(
#         lat=slice(lat_min, lat_max),
#         lon=slice(lon_min, lon_max)
#     )

#     if ds_cropped.lon.ndim != 1 or ds_cropped.lat.ndim != 1:
#         raise ValueError("ERA5 lon/lat must be 1D")

#     # Source definition for pyresample
#     src_lon2d, src_lat2d = np.meshgrid(ds_cropped.lon.values, ds_cropped.lat.values)
#     source_def = geometry.GridDefinition(lons=src_lon2d, lats=src_lat2d)

#     regridded_datasets = []

#     time_col = "valid_time" if "valid_time" in ds.coords else "time"

#     for index, row in df.iterrows():
#         grid = row['grids']
#         if grid is None:
#             continue

#         tgt_lon2d, tgt_lat2d = np.meshgrid(grid.lon.values, grid.lat.values)
#         target_def = geometry.GridDefinition(lons=tgt_lon2d, lats=tgt_lat2d)

#         dst = xr.Dataset()
#         for var in data_vars:
#             if var not in ds_cropped:
#                 continue

#             src = ds_cropped[var].isel({time_col:index}).values

#             if np.isfinite(src).sum() == 0:
#                 continue

#             dst_values = kd_tree.resample_nearest(
#                 source_def,
#                 src,
#                 target_def,
#                 radius_of_influence=31000,
#                 fill_value=np.nan
#             )

#             dst[var] = (("lat", "lon"), dst_values)

#         if len(dst.data_vars) > 0:
#             era5_time = pd.to_datetime(ds[time_col][index].values).to_datetime64()
#             dst = dst.assign_coords(
#                 lat=grid.lat,
#                 lon=grid.lon,
#                 era5_time=era5_time
#             )
#             regridded_datasets.append(dst)

#     if not regridded_datasets:
#         return None

#     return regridded_datasets


def regrid_era5(ds, df, data_vars=["wind_speed"], margin_deg=2):
    """Regrid ERA5 data to target grids, handling dateline properly."""
    regridded_datasets = []

    for index, row in df.iterrows():
        if row['bt_pos'] is None or row['grids'] is None:
            continue

        grid = row['grids']

        # Target grid coordinates normalized
        tgt_lon_vals = grid.lon.values.copy()
        tgt_lat_vals = grid.lat.values.copy()

        tgt_lon2d, tgt_lat2d = np.meshgrid(tgt_lon_vals, tgt_lat_vals)
        target_def = geometry.GridDefinition(lons=tgt_lon2d, lats=tgt_lat2d)

        # Crop source dataset with safe dateline handling
        lon_min = np.min(tgt_lon_vals) - margin_deg
        lon_max = np.max(tgt_lon_vals) + margin_deg
        lat_min = np.min(tgt_lat_vals) - margin_deg
        lat_max = np.max(tgt_lat_vals) + margin_deg

        ds_lons = ds.lon.copy()

        mask = (
            (ds.lat >= lat_min)
            & (ds.lat <= lat_max)
            & (ds.lon >= lon_min)
            & (ds.lon <= lon_max)
        )

        # Handle dateline crossing: select two slices if necessary
        ds_cropped = ds.where(mask, drop=True)#ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))

        # Build source grid for resampling
        src_lon2d, src_lat2d = np.meshgrid(ds_cropped.lon.values, ds_cropped.lat.values)
        source_def = geometry.GridDefinition(lons=src_lon2d, lats=src_lat2d)

        # Prepare output dataset
        dst = xr.Dataset()
        time_col = "valid_time" if "valid_time" in ds.coords else "time"

        for var in data_vars:
            if var not in ds_cropped:
                continue
            src = ds_cropped[var].isel({time_col:index}).values

            if np.isfinite(src).sum() == 0:
                continue

            dst_values = kd_tree.resample_nearest(
                source_def,
                src,
                target_def,
                radius_of_influence=31000,
                fill_value=np.nan,
            )
            dst[var] = (("lat", "lon"), dst_values)

        if len(dst.data_vars) > 0:
            era5_time = pd.to_datetime(ds[time_col][index].values).to_datetime64()
            dst = dst.assign_coords(
                lat=grid.lat,
                lon=grid.lon,
                era5_time=era5_time
            )

            regridded_datasets.append(dst)
        

    return regridded_datasets if regridded_datasets else None

