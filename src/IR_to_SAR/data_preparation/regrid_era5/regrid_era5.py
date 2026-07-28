import numpy as np
import pandas as pd
from shapely import wkt
from datetime import datetime
import matplotlib.pyplot as plt
import os
import importlib
from pyresample import geometry, kd_tree
import numpy as np
import xarray as xr
import pandas as pd
import src.IR_to_SAR.data_preparation.regrid_era5.regrid as regrid
importlib.reload(regrid)
from pathlib import Path
import math
import posixpath
from urllib.error import HTTPError, URLError
import time

def replace_data_url_by_local_path(df):
    df.data_url = df.data_url.str.replace("https://cyclobs.ifremer.fr/static/sarwing_datarmor/",
                                          "/home/datawork-cersat-public/cache/public/ftp/project/sarwing/")
    df.rename(columns={"data_url":"files"}, inplace=True)
    return df

def count_nans_in_file(file):
    with xr.open_dataset(file) as ds:
        nan_count = np.isnan(ds.wind_speed).sum().item()
    return nan_count

def count_notnans_in_file(file):
    with xr.open_dataset(file) as ds:
        notnans = np.isfinite(ds.wind_speed).sum().item()
    return notnans

def custom_log(level, message, filename):
    with open(filename, "a", encoding="utf-8") as f:
        line = f"{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')} ---- {level} ---- {message}\n"
        f.write(line)

def extract_dates_from_filename(filename):
    
    if posixpath.basename(filename).split("_")[0] in ['scat', 'RSS', 'SM']:
        date_begin = datetime.strptime(posixpath.basename(filename).split("_")[-3], "%Y%m%dT%H%M%S")
        date_end = datetime.strptime(posixpath.basename(filename).split("_")[-2], "%Y%m%dT%H%M%S")
    elif posixpath.basename(filename).split("_")[0] == 'ERA5':
        date_begin = datetime.strptime(posixpath.basename(filename).split("_")[1], "%Y%m%d%H%M%S")
        date_end = date_begin
    else:
        return extract_dates_from_filename_sar(posixpath.basename(filename))
    return date_begin, date_end

def extract_dates_from_filename_sar(filename):
    date_begin = datetime.strptime(str(filename).split("-")[-4], "%Y%m%dT%H%M%S")
    date_end = datetime.strptime(str(filename).split("-")[-3], "%Y%m%dT%H%M%S")
    return date_begin, date_end

def extract_sid_from_filename(filename):
    return filename.split("_")[-1].split(".")[0]

def extract_middle_from_dates(date_begin, date_end):
    delta = date_end - date_begin
    middle_date = date_begin + delta / 2
    return middle_date

def get_center_linear_interpolation(mypath, file_type='scat'):
    '''
    This function extracts files from a given path, retrieves their acquisition times and storm IDs,
    and performs linear interpolation to find the best track position of cyclones at the acquisition times.
    
    :param mypath: Path to the directory containing the colocated files to regrid.
    :param file_type: Type of files to process ('scat' for scatterometers or 'radio' for radiometers, defaults to 'scat').
    :return: DataFrame containing file paths, acquisition times, storm IDs, and best track linearly interpolated positions.
    '''
    fileslist = []
    for root, dirs, files in os.walk(mypath):
        for f in files:
            if file_type == 'scat':
                if f.split("_")[0] == "scat":
                    fileslist.append(os.path.join(root, f))
            elif file_type == 'radio':
                if f.split("_")[0] in ["RSS", "SM"]:
                    fileslist.append(os.path.join(root, f))
    df = pd.DataFrame(dict(files=fileslist)).assign(
        acq_time=lambda x: x['files'].map(lambda f: extract_middle_from_dates(*extract_dates_from_filename(f))),
        sids=lambda x: x['files'].map(extract_sid_from_filename),
        nans=lambda x: x['files'].map(count_nans_in_file),
        notnans=lambda x: x['files'].map(count_notnans_in_file),
        percentage_notnans=lambda x: x['notnans'] / (x['nans'] + x['notnans']) * 100,
        onlynans_bool=lambda x: x['notnans'] == 0,
        ).sort_values('acq_time').reset_index(drop=True)
    bt_pos = []

    for index, row in df.iterrows():
        sid = row.sids
        t_obs = row.acq_time

        df_api = read_track_with_retry(sid, t_obs.year)

        if df_api.empty:
            bt_pos.append(None)
            continue

        # Convert track times
        df_api['date'] = pd.to_datetime(df_api['date'])

        # Convert geometry if WKT
        if isinstance(df_api.geometry.iloc[0], str):
            df_api['geometry'] = df_api['geometry'].apply(wkt.loads)

        # Find the two surrounding points
        # Time difference
        df_api['dt'] = df_api['date'] - t_obs

        # Separate before and after
        before = df_api[df_api['dt'] <= pd.Timedelta(0)]
        after  = df_api[df_api['dt'] >  pd.Timedelta(0)]

        # If no interpolation possible (only one side exists)
        if before.empty or after.empty:
            # fallback to nearest neighbor
            ind_min = df_api['dt'].abs().idxmin()
            bt_pos.append(df_api.loc[ind_min, 'geometry'])
            continue

        # Take the closest before and after
        p_before = before.iloc[-1]   # latest before
        p_after  = after.iloc[0]     # earliest after

        t0 = p_before['date']
        t1 = p_after['date']

        # If same timestamp, avoid division by zero
        if t0 == t1:
            bt_pos.append(p_before['geometry'])
            continue

        # Compute interpolation weight
        # w = fraction of time between t0 and t1
        w = (t_obs - t0) / (t1 - t0)   # value between 0 and 1

        # Extract coordinates
        lon0, lat0 = p_before['geometry'].x, p_before['geometry'].y
        lon1, lat1 = p_after['geometry'].x, p_after['geometry'].y

        # Linear interpolation
        lon_interp = lon0 * (1 - w) + lon1 * w
        lat_interp = lat0 * (1 - w) + lat1 * w

        # Store as shapely point
        from shapely.geometry import Point
        bt_pos.append(Point(lon_interp, lat_interp))
    df['best_track_position'] = bt_pos
    return df

def get_eye_center_or_linear_interpolation(start_date, stop_date):
    '''
    
    '''
    df_api_sar = pd.read_csv(
            f"https://cyclobs.ifremer.fr/app/api/getData?&acquisition_start_time={start_date}&acquisition_stop_time={stop_date}&include_cols=data_url,sid,eye_center"
        )

    fileslist = replace_data_url_by_local_path(df_api_sar)['files'].tolist()
    
    df = pd.DataFrame(dict(files=fileslist)).assign(
        acq_time=lambda x: x['files'].map(lambda f: extract_middle_from_dates(*extract_dates_from_filename_sar(f))),
        sids=df_api_sar.set_index('files').loc[fileslist]['sid'].values,
        eye_center=df_api_sar.set_index('files').loc[fileslist]['eye_center'].values,
        nans=lambda x: x['files'].map(count_nans_in_file),
        notnans=lambda x: x['files'].map(count_notnans_in_file),
        percentage_notnans=lambda x: x['notnans'] / (x['nans'] + x['notnans']) * 100,
        onlynans_bool=lambda x: x['notnans'] == 0,
        ).sort_values('acq_time').reset_index(drop=True)

    bt_pos = []

    for index, row in df.iterrows():
        sid = row.sids
        t_obs = row.acq_time
        if pd.notna(row.eye_center):
            bt_pos.append(wkt.loads(row.eye_center))
        else:

            df_api = pd.read_csv(
                f"https://cyclobs.ifremer.fr/app/api/track?sid={sid}&year={t_obs.year}"
            )

            if df_api.empty:
                bt_pos.append(None)
                continue

            # Convert track times
            df_api['date'] = pd.to_datetime(df_api['date'])

            # Convert geometry if WKT
            if isinstance(df_api.geometry.iloc[0], str):
                df_api['geometry'] = df_api['geometry'].apply(wkt.loads)

            # Find the two surrounding points
            # Time difference
            df_api['dt'] = df_api['date'] - t_obs

            # Separate before and after
            before = df_api[df_api['dt'] <= pd.Timedelta(0)]
            after  = df_api[df_api['dt'] >  pd.Timedelta(0)]

            # If no interpolation possible (only one side exists)
            if before.empty or after.empty:
                # fallback to nearest neighbor
                ind_min = df_api['dt'].abs().idxmin()
                bt_pos.append(df_api.loc[ind_min, 'geometry'])
                continue

            # Take the closest before and after
            p_before = before.iloc[-1]   # latest before
            p_after  = after.iloc[0]     # earliest after

            t0 = p_before['date']
            t1 = p_after['date']

            # If same timestamp, avoid division by zero
            if t0 == t1:
                bt_pos.append(p_before['geometry'])
                continue

            # Compute interpolation weight
            # w = fraction of time between t0 and t1
            w = (t_obs - t0) / (t1 - t0)   # value between 0 and 1

            # Extract coordinates
            lon0, lat0 = p_before['geometry'].x, p_before['geometry'].y
            lon1, lat1 = p_after['geometry'].x, p_after['geometry'].y

            # Linear interpolation
            lon_interp = lon0 * (1 - w) + lon1 * w
            lat_interp = lat0 * (1 - w) + lat1 * w

            # Store as shapely point
            from shapely.geometry import Point
            bt_pos.append(Point(lon_interp, lat_interp))
    df['best_track_position'] = bt_pos
    return df

def regrid_from_center(ds, output_path, size_km=600, resolution_km=25, file_type='scat'):
    '''
    This function regrids data files based on cyclone center positions using a specified grid size and resolution.
    :param ds: DataFrame containing file paths and cyclone center positions.
    :param output_path: Directory where regridded files will be saved.
    :param size_km: Size of the regridded area in kilometers (default is 600 km).
    :param resolution_km: Resolution of the regridded grid in kilometers (default is 25 km).
    :param file_type: Type of files to process ('scat' for scatterometers or 'radio' for radiometers, defaults to 'scat').
    :return: List of paths to the regridded files.
    '''
    n = len(ds)
    i = 0
    regridded_path_list = []

    for index, row in ds.iterrows():

        point = row.best_track_position
        if pd.notnull(point):
            # Converting Point to coordinates
            a, b = point.x, point.y

            # Target grid
            grid = regrid.create_target_grid(a, b, size_km, resolution_km)                

            if file_type == 'scat':
                regridded = regrid.regrid_scat([row['files']], grid)
                # Creating output paths and saving filenames
                output = f"{output_path}/{row['files'].split('/')[-1]}"
            elif file_type == 'radio':
                regridded = regrid.regrid_radiometer([row['files']], grid)
                # Creating output paths and saving filenames
                output = f"{output_path}/{row['files'].split('/')[-1]}"
            elif file_type == 'sar':
                regridded = regrid.regrid_sar([row['files']], grid)
                output = f"{output_path}/{row['files'].split('/')[-1].split('.')[0]}_{row['sids']}.nc"
            if regridded is not None:
                if regridded.wind_speed[0,:,:].notnull().sum().item()>0:
                    regridded_path_list.append(output)

                    # Saving
                    regridded.to_netcdf(output, mode="w")

        else:
            regridded_path_list.append(None)
    return regridded_path_list

_TRACK_CACHE = {}

def read_track_with_retry(sid, year, retries=5, sleep=5, polite_sleep=0.1):
    key = (sid, year)
    if key in _TRACK_CACHE:
        return _TRACK_CACHE[key]

    url = f"https://cyclobs.ifremer.fr/app/api/track?sid={sid}&year={year}"

    last_err = None
    for _ in range(retries):
        try:
            df = pd.read_csv(url)

            if not df.empty:
                df["date"] = pd.to_datetime(df["date"])
                if isinstance(df.geometry.iloc[0], str):
                    df["geometry"] = df["geometry"].apply(wkt.loads)

            _TRACK_CACHE[key] = df

            time.sleep(polite_sleep)
            return df

        except (HTTPError, URLError) as e:
            last_err = e
            time.sleep(sleep)

    raise RuntimeError(
        f"Failed to retrieve track for {sid} ({year})"
    ) from last_err

_TRACK_CACHE_SID = {}

def read_track_with_retry_sid(sid, retries=5, sleep=5, polite_sleep=0.1):
    key = (sid)
    if key in _TRACK_CACHE_SID:
        return _TRACK_CACHE_SID[key]

    url = f"https://cyclobs.ifremer.fr/app/api/track?sid={sid}&freq=60"

    last_err = None
    for _ in range(retries):
        try:
            df = pd.read_csv(url)

            if not df.empty:
                df["date"] = pd.to_datetime(df["date"])
                if isinstance(df.geometry.iloc[0], str):
                    df["geometry"] = df["geometry"].apply(wkt.loads)

            _TRACK_CACHE_SID[key] = df

            time.sleep(polite_sleep)
            return df
        

        except (HTTPError, URLError) as e:
            last_err = e
            time.sleep(sleep)

    raise RuntimeError(
        f"Failed to retrieve track for {sid}"
    ) from last_err

# def regrid_one_file_era5(filename):
#     sid = str(filename).split('_')[-1].split('.')[0]
#     df_api = read_track_with_retry_sid(sid)
#     with xr.open_dataset(filename) as ds:
#         bt_pos = []
#         if "valid_time" in ds.coords:
#             time_col = "valid_time"
#         else:
#             time_col = "time"
#         for i in range(len(ds.wind_speed[time_col])):
#             t_obs = pd.to_datetime(ds.wind_speed[time_col][i].values)
#             if df_api.empty:
#                 bt_pos.append(None)
#                 continue

#             # Convert track times
#             df_api['date'] = pd.to_datetime(df_api['date'])

#             # Convert geometry if WKT
#             if isinstance(df_api.geometry.iloc[0], str):
#                 df_api['geometry'] = df_api['geometry'].apply(wkt.loads)

#             # Find the closest time
#             # Time difference
#             df_api['dt'] = df_api['date'] - t_obs

#             # Find the minimum time difference
#             idx_min = df_api['dt'].abs().idxmin()
#             closest_point = df_api.loc[idx_min, 'geometry']
#             bt_pos.append(closest_point)
            
#         df = pd.DataFrame({"bt_pos": bt_pos, "time": ds[time_col].values})
#         grids = []
#         for index, row in df.iterrows():
#             if row['bt_pos']:
#                 grids.append(regrid.create_target_grid(row['bt_pos'].x, row['bt_pos'].y, 1536, 12))
#         df['grids'] = grids
#     return regrid.regrid_era5(ds, df, data_vars=["wind_speed"]), sid

# def regrid_one_file_era5(filename): ###CHAT
#     """
#     Regrid an ERA5 file for all timesteps using the target grids derived from the cyclone track.
#     Returns the list of regridded datasets and the storm ID (sid).
#     """
#     import pandas as pd
#     import xarray as xr
#     from shapely import wkt
#     import regrid  # Assuming regrid_era5 and create_target_grid are in this module

#     sid = str(filename).split('_')[-1].split('.')[0]
#     df_api = read_track_with_retry_sid(sid)

#     with xr.open_dataset(filename) as ds:
#         # Determine the time coordinate
#         time_col = "valid_time" if "valid_time" in ds.coords else "time"

#         # Normalize ERA5 longitudes once (0–360)
#         ds = ds.copy()
#         ds["lon"] = (ds["lon"] + 360) % 360
#         ds = ds.sortby("lon")

#         bt_pos_list = []
#         for i in range(len(ds[time_col])):
#             t_obs = pd.to_datetime(ds[time_col][i].values)

#             if df_api.empty:
#                 bt_pos_list.append(None)
#                 continue

#             df_api['date'] = pd.to_datetime(df_api['date'])

#             # Convert geometry from WKT if needed
#             if isinstance(df_api.geometry.iloc[0], str):
#                 df_api['geometry'] = df_api['geometry'].apply(wkt.loads)

#             # Find closest track point in time
#             df_api['dt'] = (df_api['date'] - t_obs).abs()
#             idx_min = df_api['dt'].idxmin()
#             closest_point = df_api.loc[idx_min, 'geometry']
#             bt_pos_list.append(closest_point)

#         df = pd.DataFrame({"bt_pos": bt_pos_list, "time": ds[time_col].values})

#         # Create target grids
#         grids = []
#         for _, row in df.iterrows():
#             if row['bt_pos']:
#                 grid = regrid.create_target_grid(row['bt_pos'].x, row['bt_pos'].y, 1536, 12)
#                 # Normalize grid longitudes 0–360
#                 grid = grid.assign_coords(lon=(grid.lon + 360) % 360)
#                 grids.append(grid)
#             else:
#                 grids.append(None)
#         df['grids'] = grids

#     # Call regrid_era5 on the whole file
#     regridded = regrid.regrid_era5(ds, df, data_vars=["wind_speed"])
#     return regridded, sid

def regrid_one_file_era5(filename, resolution_km=2, grid_size_km=300):
    """Regrid a single ERA5 file for all timesteps and return regridded datasets."""
    sid = str(filename).split('_')[-1].split('.')[0]
    df_api = read_track_with_retry_sid(sid)
    if df_api.empty:
        print(f"Empty track for sid {sid} in file {filename}")
        return None, sid

    with xr.open_dataset(filename) as ds:
        # Determine time coordinate
        time_col = "valid_time" if "valid_time" in ds.coords else "time"

        # Normalize source longitudes to 0–360
        ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180))
        ds = ds.sortby("lon")
        #print("min and max of era5 file lon:", (ds.lon.min(), ds.lon.max()))

        # Find closest track points for each timestep
        bt_pos = []
        for i in range(len(ds[time_col])):
            t_obs = pd.to_datetime(ds[time_col][i].values)
            if df_api.empty:
                bt_pos.append(None)
                continue

            df_api['date'] = pd.to_datetime(df_api['date'])

            # Convert WKT geometry if needed
            if isinstance(df_api.geometry.iloc[0], str):
                df_api['geometry'] = df_api['geometry'].apply(wkt.loads)

            # Find closest observation
            df_api['dt'] = df_api['date'] - t_obs
            idx_min = df_api['dt'].abs().idxmin()
            bt_pos.append(df_api.loc[idx_min, 'geometry'])

        df = pd.DataFrame({"bt_pos": bt_pos, "time": ds[time_col].values})

        # Create target grids
        grids = []
        for _, row in df.iterrows():
            if row['bt_pos'] is not None:
                grid = regrid.create_target_grid(row['bt_pos'].x, row['bt_pos'].y, grid_size_km, resolution_km)
                grid = grid.assign_coords(lon=(((grid.lon + 180) % 360) - 180))  # normalize grid
                #print("min and max of target grid lon:", (grid.lon.min(), grid.lon.max()))
                grids.append(grid)
            else:
                grids.append(None)
        
        df['grids'] = grids

        # Regrid using the updated grids
        return regrid.regrid_era5(ds, df, data_vars=["wind_speed"]), sid

def write_regridded_files_era5(regridded, sid, output_path):
    for i in range(len(regridded)):
        output = f"{output_path}/ERA5_{pd.to_datetime(regridded[i].era5_time.values).strftime('%Y%m%d%H%M%S')}_{sid}.nc"
        regridded[i].to_netcdf(output, mode="w")
    

def regrid_and_write_era5_file(filename, output_path, grid_size_km=300, resolution_km=2):
    try:
        regridded, sid = regrid_one_file_era5(filename, grid_size_km=grid_size_km, resolution_km=resolution_km)
        return regridded, sid
        # if regridded is not None:
        #     write_regridded_files_era5(regridded, sid, output_path)
        # else:
        #     print(f"Skipping {filename} - no valid regridded data")
    except Exception as e:
        print(f"Error processing {filename}: {e}")

def regrid_files_era5(file_list, output_path, grid_size_km=300, resolution_km=2, index_hour=None):
    results = []
    for filename in file_list:
        regridded, sid = regrid_and_write_era5_file(filename, output_path, grid_size_km=grid_size_km, resolution_km=resolution_km,
                                                    )
        results.append(regridded[index_hour]["wind_speed"].values)
    return results