import logging
import datetime
import os
import sys
import xarray
import rioxarray as rxr
import pandas as pd
import geopandas as gpd
import numpy as np
import math
# from geo_shapely import shape360
from shapely.wkt import dumps
from shapely.geometry import Polygon
# from pkg_resources import get_distribution

logger = logging.getLogger(__name__)
version_extract_cyclones = "v01"

from functools import wraps

import time
# Initialize memory monitor
mem_monitor = True
try:
    from psutil import Process
except ImportError:
    logger.warning("psutil module not found. Disabling memory monitor")
    mem_monitor = False


def timing(f):
    """provide a @timing decorator for functions, that log time spent in it"""

    @wraps(f)
    def wrapper(*args, **kwargs):
        mem_str = ''
        process = None

        if mem_monitor:
            process = Process(os.getpid())
            startrss = process.memory_info().rss

        starttime = time.time()
        result = f(*args, **kwargs)
        endtime = time.time()

        if mem_monitor:
            endrss = process.memory_info().rss
            mem_str = 'mem: %+.1fMb' % ((endrss - startrss) / (1024 ** 2))

        logger.debug('timing %s : %.1fs. %s' % (f.__name__, endtime - starttime, mem_str))

        return result

    return wrapper


os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'


def deg180(deg360):
    """
    convert [0,360] to [-180,180]
    """
    res = np.copy(deg360)
    res[res > 180] = res[res > 180] - 360
    return res


def deg360(deg180):
    """shapely shape to 0 360 (for shapely.ops.transform)"""
    lon = np.array(deg180) % 360
    return lon


def get_track_points_from_api(date):
    """
    Request cyclobs API for track points between date and date + 2 days.

    Parameters
    ----------
    date : datetime The date used to filter the query.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the API result. Can be empty.

    """

    min_date = date.strftime("%Y-%m-%d")
    max_date = (date + datetime.timedelta(days=2)).strftime("%Y-%m-%d")
    req = f"https://cyclobs.ifremer.fr/app/api/track?min_date={min_date}&max_date={max_date}"
    try:
        df = pd.read_csv(req)
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] >= date]
    except pd.errors.EmptyDataError as e:
        logger.warning(f"Empty data from API (req : {req}")
        df = pd.DataFrame()

    return df


# Selecting best pass
def select_pass(selected_set, file_date, track_point, lon_index, lat_index, time_col_name, is_full_time):
    """
    Some scatterometers may store 2 files per day, one for the ascending passes and another for the descending.
    This function looks into both passes and select the one with the best colocation with the track.
    xarray Dataset having 2 pass must have a "node" dimension with a size of 2 and if the file is correctly formatted
    the lon, lat, time and other variables should depend on that node dimension. (for example

    :param selected_set:
    :param file_date:
    :param track_point:
    :param lon_index:
    :param lat_index:
    :param time_col_name:
    :param is_full_time:
    :return:
    """
    df = selected_set.to_dataframe()
    # Storing the time offset between the data point and the track point
    # The time offset is stored with all the track point data to ease later processing
    for node in df.index:
        if not pd.isna(df[time_col_name][node]):
            if not is_full_time:
                point_date = file_date + df[time_col_name][node]
            else:
                point_date = df[time_col_name][node]
            time_offset = abs(track_point["date"] - point_date)
            logger.debug(f" Data point date: {point_date}, Track point date: {track_point['date']},"
                         f" Time offset: {time_offset}")
            # If this is the first node processed OR that the previous nodes had NAN times OR that previous nodes
            # offset times are greater than this one
            if node == 0 or not "time_offset" in track_point or track_point["time_offset"] > time_offset:
                logger.debug(f"Minute : {df[time_col_name][node]}")
                track_point["time_offset"] = time_offset
                track_point["node"] = node
                track_point["lon_index"] = lon_index
                track_point["lat_index"] = lat_index


def is_full_time(dataset, time_col_name):
    df = dataset.to_dataframe().reset_index()
    df = df.dropna(subset=[time_col_name]).reset_index()
    time = df.loc[0, time_col_name]
    if type(time) == pd._libs.tslibs.timestamps.Timestamp:
        return True
    elif type(time) == pd._libs.tslibs.timedeltas.Timedelta:
        return False
    else:
        logger.error(f"Error : could not detect time type on column {time_col_name}. Exiting.")
        sys.exit(1)


@timing
def get_colocated_track_points(dataset, track_points, file_date, wind_col_name, is_full_time,
                               time_col_name, pass_col_name=None, lon_col_name="lon", lat_col_name="lat",
                               deg_delta=3, time_delta=60):

    track_point_offsets = []
    for index, track_point in track_points.iterrows():
        logger.debug(f"Track point : {track_point}")
        # Selecting point in dataset that is geographically the nearest from the current iterated track point.
        sel = dataset.sel(**{lon_col_name: track_point["lon"], lat_col_name: track_point["lat"], "method": "nearest"})

        # Find the index in the lon and lat array of that nearest pixel.
        lon_index = np.where(dataset[lon_col_name].data == sel[lon_col_name].data)[0][0]
        lat_index = np.where(dataset[lat_col_name].data == sel[lat_col_name].data)[0][0]
        logger.debug(f"Point find : lon: {sel[lon_col_name].values}, lat: {sel[lat_col_name].values},"
                     f" minute: {sel[time_col_name].values},"
                     f" lon_index: {lon_index}, lat_index {lat_index}")

        trk_ptn = track_point.to_dict()

        if pass_col_name:
            # If we are processing a file which has several pass, we select the "best" pass. That is the pass
            # in which the selected pixel is the closest temporally to the track point date.
            select_pass(sel, file_date, trk_ptn, lon_index, lat_index, time_col_name, is_full_time)
        else:
            df = sel.to_dataframe().reset_index()
            if not pd.isna(df.loc[0, time_col_name]):
                if not is_full_time:
                    point_date = file_date + df.loc[0, time_col_name]
                else:
                    point_date = df.loc[0, time_col_name]
                # Computing and storing the time offset between the track point and the pixel time of acquisition.
                # This offset will later be used to decide if that pixel is used as a reference point to extract the data.
                time_offset = abs(track_point["date"] - point_date)
                trk_ptn["time_offset"] = time_offset
                trk_ptn["node"] = None
                trk_ptn["lon_index"] = lon_index
                trk_ptn["lat_index"] = lat_index
        if "time_offset" in trk_ptn:
            track_point_offsets.append(trk_ptn)

    kept_track_points = {}
    # Getting the *best* track point for each sid (storm id).
    # One file can contain several cyclones acquisitions so that is why we are doing this filtering per sid
    # The kept track point are those that have temporal distance below 7 min 30 sec with the associated pixel and
    # if several track points match that criteria, the closest temporally is kept.
    for track_point in track_point_offsets:
        # TODO replace timedelta with database track sampling time
        if track_point["time_offset"] <= datetime.timedelta(minutes=7, seconds=30):
            if track_point["sid"] not in kept_track_points:
                kept_track_points[track_point["sid"]] = track_point
            elif kept_track_points[track_point["sid"]]["time_offset"] > track_point["time_offset"]:
                kept_track_points[track_point["sid"]] = track_point

    return kept_track_points


# Set NaNs where the field "minute" and "wind" of the dataset is outside the +-time_offset
# relative to the cyclone track point date
def set_nan_outside_time_offset(dataset, track_data, time_col_name, wind_col_name, is_full_time, time_offset=30):
    track_timedelta = datetime.timedelta(hours=track_data["date"].hour, minutes=track_data["date"].minute)
    if is_full_time:
        min_time_bound = np.datetime64(track_data["date"] - datetime.timedelta(minutes=time_offset))
        max_time_bound = np.datetime64(track_data["date"] + datetime.timedelta(minutes=time_offset))
    else:
        min_time_bound = np.timedelta64(int((track_timedelta - datetime.timedelta(
            minutes=time_offset)).total_seconds()), 's')
        max_time_bound = np.timedelta64(int((track_timedelta + datetime.timedelta(
            minutes=time_offset)).total_seconds()), 's')

    dataset[time_col_name] = dataset[time_col_name].where((dataset[time_col_name].data < max_time_bound) &
                                                          (dataset[time_col_name].data > min_time_bound))

    # Apply "minute" created NaNs on "wind" field
    dataset[wind_col_name].data[np.isnan(dataset[time_col_name].data)] = np.nan

    return dataset


def transform_polyg_180(p):
    coords = p.exterior.coords
    new_coords = [(c[0] - 360, c[1]) if c[0] > 180 else (c[0], c[1]) for c in coords]

    return Polygon(new_coords)


# Extract footprint (not taking NaN data into account)
def compute_footprint(dataset, wind_field):
    # Using 0:360 range because the -180:180 range does create the right polygon with gdf.unary_union.convex_hull
    lon_atts = dataset["lon"].attrs
    dataset["lon"] = deg360(dataset["lon"])
    dataset["lon"].attrs = lon_atts

    df = dataset.to_dataframe()
    df = df[pd.isna(df[wind_field]) == False]
    df = df.reset_index()
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.lon, df.lat))

    p = gdf.unary_union.convex_hull
    # p = p.simplify(5)
    # Transforming the polygon so its coords are in range -180:180
    # p = transform_polyg_180(p)
    # print(p)
    return p


# Extracting min and max time
def extract_start_stop_measure(dataset, time_col_name):
    df = dataset.to_dataframe().reset_index()
    min_time = df[time_col_name].min(skipna=True)
    max_time = df[time_col_name].max(skipna=True)

    return min_time, max_time


def extract_write_cyclone_data(dataset, kept_track_points, filename, output_path,
                               file_date, source_filename, attrs, attrs_to_del, attrs_rename,
                               var_to_del, lon_col_name,
                               lat_col_name, wind_col_name, time_col_name, pass_col_name,
                               filename_format,
                               is_full_time,
                               extract_size_km=2000,
                               var_attr_edit={},
                               specific_func=None,
                               min_pixel=5):
    for sid, track_point in kept_track_points.items():
        logger.debug(f"Track point : {track_point}")
        # Km per deg in latitude
        km_per_deg_lat = 111
        # Degrees offset per latitude index in data array
        deg_per_index_lat = abs(dataset[lat_col_name][0] - dataset[lat_col_name][1])
        # Km offset per latitude index in data array
        km_per_idx_lat = km_per_deg_lat * deg_per_index_lat
        logger.debug(f"km_per_idx_lat: {km_per_idx_lat}")
        # Number of latitude index to take to have extract_size_km size of data
        nb_idx_lat = round(float(extract_size_km / km_per_idx_lat) / 2)
        logger.debug(f"nb_idx_lat: {nb_idx_lat}")

        if track_point["node"] is not None:
            sel_lat = dataset.isel(**{lat_col_name: slice(track_point["lat_index"] - nb_idx_lat,
                                                          track_point["lat_index"] + nb_idx_lat),
                                      pass_col_name: track_point["node"]})
        else:
            sel_lat = dataset.isel(**{lat_col_name: slice(track_point["lat_index"] - nb_idx_lat,
                                                          track_point["lat_index"] + nb_idx_lat)})

        # In next section, the goal is to compute how many longitude index we must take to have a data section of
        # 'extract_size_km'. As the km offset between each longitude degrees depends of the latitude, we first compute
        # the latitude_average of the current selected data and compute the km_per_deg at that latitude.

        # Latitude average of the current area
        logger.debug(f"sel_lat[lat_col_name] : {sel_lat[lat_col_name]}")
        lat_av = (sel_lat[lat_col_name].max() + sel_lat[lat_col_name].min()) / 2
        logger.debug(f"lat_av: {lat_av}")
        # Earth circumference
        earth_circum = 40000
        # Kilometer per longitude deg at current latitude
        km_per_deg = abs(earth_circum * math.cos(math.radians(lat_av)) / 360)
        logger.debug(f"km_per_deg: {km_per_deg}")
        # Deg offset per index in data
        deg_per_index = abs(sel_lat[lon_col_name][0] - sel_lat[lon_col_name][1])
        # Km offset per index in data
        km_per_idx = km_per_deg * deg_per_index
        logger.debug(f"km_per_idx: {km_per_idx}")
        # Number of longitude index to take to have extract_size_km size of data
        nb_idx_lon = float(extract_size_km / km_per_idx)
        nb_idx_lon_2 = round(nb_idx_lon / 2)
        logger.debug(f"nb_idx_lon_2 : {nb_idx_lon_2}")

        # Selecting area of interest
        if track_point["node"] is not None:
            sel = dataset.isel(**{lon_col_name: slice(track_point["lon_index"] - nb_idx_lon_2,
                                                      track_point["lon_index"] + nb_idx_lon_2),
                                  lat_col_name: slice(track_point["lat_index"] - nb_idx_lat,
                                                      track_point["lat_index"] + nb_idx_lat),
                                  pass_col_name: track_point["node"]}, drop=True)
        else:
            sel = dataset.isel(**{lon_col_name: slice(track_point["lon_index"] - nb_idx_lon_2,
                                                      track_point["lon_index"] + nb_idx_lon_2),
                                  lat_col_name: slice(track_point["lat_index"] - nb_idx_lat,
                                                      track_point["lat_index"] + nb_idx_lat)}, drop=True)

        # Removing data that is on other swath than the one where the cyclone is
        sel = set_nan_outside_time_offset(sel, track_point, time_col_name=time_col_name, wind_col_name=wind_col_name,
                                          is_full_time=is_full_time)

        if specific_func is not None:
            sel = specific_func(sel)

        # if there are less than x points, remove the image
        df = sel.to_dataframe()
        df = df[pd.isna(df[wind_col_name]) == False]
        if len(df.index) <= min_pixel:
            logger.info(f"Data has less than {min_pixel} pixel of data. Not writing to file.")
            return

        sel, start_date, stop_date, year, doy = set_metadata(sel, wind_col_name, time_col_name, file_date, attrs,
                                                        attrs_to_del, attrs_rename, var_to_del, var_attr_edit,
                                                        lon_col_name, source_filename, is_full_time)
        write_ncdf(sel, output_path, filename_format, start_date, stop_date, year, doy, track_point["sid"])


def set_metadata(dataset, wind_col_name, time_col_name, file_date, attrs, attrs_to_del, attrs_rename,
                 var_to_del, var_attr_edit, lon_col_name, source_filename, is_full_time):

    for attr in attrs_rename:
        if attr in dataset.attrs:
            attrs[attrs_rename[attr]] = dataset.attrs[attr]

    # Deleting all attributes
    dataset.attrs.clear()

    # Extracting min and max time to set attributes on NetCDF
    min_time, max_time = extract_start_stop_measure(dataset, time_col_name)
    if max_time - min_time > datetime.timedelta(minutes=30):
        logger.error(f"Time offset between min_time ({min_time}) and max_time ({max_time}) exceeds 30 min.")
        sys.exit(1)

    logger.debug(f"min_time: {min_time}, max_time: {max_time}")
    # Computing acquisition start and stop dates and setting as global attribute
    if is_full_time:
        min_date = min_time
        max_date = max_time
    else:
        # In the case of SMAP, the time dataArray contains only a time (hour, min, sec) and no date.
        # To have a full date, we add the file date.
        min_date = file_date + min_time
        max_date = file_date + max_time

    year = min_date.year
    doy = min_date.strftime('%j')
    # measure_date = min_date + (max_date - min_date) / 2
    dataset.attrs["measurementStartDate"] = min_date.strftime("%Y-%m-%d %H:%M:%S")
    dataset.attrs["measurementStopDate"] = max_date.strftime("%Y-%m-%d %H:%M:%S")

    # Setting new attributes
    for attr in attrs:
        if type(attrs[attr]) == dict:
            for var_attr in attrs[attr]:
                # The _FillValue attribute is in the encoding dict
                if var_attr == "_FillValue":
                    dataset[attr].encoding[var_attr] = attrs[attr][var_attr]
                else:
                    dataset[attr].attrs[var_attr] = attrs[attr][var_attr]
        else:
            dataset.attrs[attr] = attrs[attr]

    # Removing attributes
    # FIXME potentially to remove because shadowed by the above deletion. But for now it still has an effect because
    # it deletes attributes from .encoding too.
    for attr in attrs_to_del:
        if type(attrs_to_del[attr]) == list:
            for var_attr in attrs_to_del[attr]:
                if var_attr in dataset[attr].attrs:
                    del dataset[attr].attrs[var_attr]
                if var_attr in dataset[attr].encoding:
                    del dataset[attr].encoding[var_attr]
        else:
            if attr in dataset.attrs:
                del dataset.attrs[attr]

    for var in var_to_del:
        if var in dataset.keys():
            dataset = dataset.drop_vars(var)

    new_wind_col = "wind_speed"
    dataset = dataset.rename_vars({wind_col_name: new_wind_col})

    footprint_polyg = compute_footprint(dataset, new_wind_col)

    # Converting lon from 0;360 range to -180;180
    lon_attrs = dataset[lon_col_name].attrs
    dataset[lon_col_name] = deg180(dataset[lon_col_name])
    dataset[lon_col_name].attrs = lon_attrs

    dataset.attrs["sourceProduct"] = source_filename
    dataset.attrs["footprint"] = dumps(footprint_polyg)
    # __version__ = get_distribution('extract_cyclones').version
    dataset.attrs["productVersion"] = version_extract_cyclones

    for var in var_attr_edit:
        for attr in var_attr_edit[var]:
            dataset[var].attrs[attr] = var_attr_edit[var][attr]

    return dataset, min_date, max_date, year, doy


# Write the dataset to a netCDF file
def write_ncdf(dataset, output_path, filename_format, start_date, stop_date, year, doy, track_point_sid):
    output_filename = filename_format.replace("<start_date>", start_date.strftime('%Y%m%dT%H%M%S'))
    output_filename = output_filename.replace("<stop_date>", stop_date.strftime('%Y%m%dT%H%M%S'))
    output_filename = output_filename.replace("<sid>", track_point_sid)

    logger.info(f"Writing file {output_filename}")

    dataset = dataset.squeeze()



    dataset.rio.set_spatial_dims('lon', 'lat', inplace=True)
    dataset.rio.write_crs("epsg:4326", inplace=True)

    out = os.path.join(output_path, str(year), doy)
    os.makedirs(out, exist_ok=True)

    dataset.to_netcdf(os.path.join(out, output_filename), format="NETCDF4_CLASSIC")


def process_file(file, output_path, extract_date_func, attrs, var_to_del, wind_col,
                 time_col, lat_col, lon_col, filename_format, pass_width, pass_col=None, attrs_to_del=[],
                 attrs_rename={},
                 var_attr_edit={},
                 specfic_func=None):
    """
    Extract portion on data from file (data corresponding to a cyclone) and write it to a NetCDF file.


    Track points concerning the given file are retrieved from CyclObs API, then it is checked if
    one or several of these track points are actually colocated with data points in the given file.
    If there are, a portion of data around each selected point is extracted from file. These extracted portions
    correspond to interesting areas where a cyclone has been captured. That extracted data is
    saved as a well structured NETCDF file.

    Parameters
    ----------
    file : str Path to SMOS or SMAP file from which to extract the cyclone data
    output_path : str Path in which save the netCDF file
    extract_date_func : function Function used to extract the date using, as parameter, os.path.basename(file)
    attrs : dict Attributes that will be set in the output NETCDF file.
    var_to_del : list of str Variables that will not be included into the resulting netCDF file.
    wind_col : str Name of the wind speed variable in the given file.
    time_col : str Name of the time variable in the given file.
    lat_col : str Name of the latitude variable in the given file.
    lon_col : str Name of the lngitude variable in the given file.
    filename_format : str Format to use to name the resulting NETCDF file. Example : RSS_smap_wind_<start_date>_<stop_date>_<sid>.nc
    pass_width : int Size of the extracted portion.
    pass_col : str or None Variable used to indicate if there is several pass in the given file. If there is (!=None), a string must be given indicating the variable name of the pass.
    attrs_to_del : dict of list Variable attributes to not include into resulting NETCDF. Keys : variable name, Value: list of attribute names
    attrs_rename : dict of str Rename global attributes. Key: attribute value to rename, Value: new attribute name
    var_attr_edit : dict of dict Edit variable attribute values. Format : {"var": {"attr": new_value}}
    specfic_func : function A function taking an xarray dataset as parameter and returning an xarray dataset. That function
    can perform any operation on the data.

    Returns
    -------
    Nothing

    """

    logger.info(f"Processing {file}...")

    filename = os.path.basename(file)
    file_date = extract_date_func(filename)
    logger.debug(f"File date {file_date}")

    # Retrieve track points near that date from CyclObs API
    track_points = get_track_points_from_api(file_date)
    logger.debug(f"Number of track point found : {len(track_points)}")

    # If no track points
    if len(track_points.index) == 0:
        logger.warning("No track points, skipping this file.")
        return

    # Transform lon from -180, 180 to 0, 360
    # track_points["lon"] = track_points["lon"].apply(lambda x: shape360(x, 0)[0])
    track_points["lon"] = track_points["lon"] % 360

    dataset = xarray.open_dataset(file)
    df = dataset.to_dataframe()
    # Not processing file if it contains only NaNs
    if not df[wind_col].isnull().all():
        full_time = is_full_time(dataset, time_col)
        kept_track_points = get_colocated_track_points(dataset, track_points, file_date,
                                                       wind_col_name=wind_col, time_col_name=time_col,
                                                       pass_col_name=pass_col, is_full_time=full_time)
        logger.debug(f"For file {filename} Kept track that will be used to extract cyclone data: {kept_track_points}")

        extract_write_cyclone_data(dataset, kept_track_points, filename, output_path, file_date,
                                   source_filename=filename, wind_col_name=wind_col,
                                   time_col_name=time_col, lat_col_name=lat_col, lon_col_name=lon_col,
                                   pass_col_name=pass_col, attrs=attrs, attrs_to_del=attrs_to_del,
                                   attrs_rename=attrs_rename,
                                   var_to_del=var_to_del,
                                   filename_format=filename_format,
                                   is_full_time=full_time, extract_size_km=pass_width * 2, var_attr_edit=var_attr_edit,
                                   specific_func=specfic_func)
    else:
        logger.warning(f"Filename {filename} contains only NaN values for {wind_col} column. It will not be processed.")
        sys.exit(2)



###################################################################################################################################
###################### ADDING CODE FOR SCATTEROMETER PROCESSING ###################################################################
###################################################################################################################################


def process_file_scat(file, output_path, extract_date_func, attrs, var_to_del, wind_col,
                 time_col, lat_col, lon_col, filename_format, pass_width, pass_col=None, attrs_to_del=[],
                 attrs_rename={},
                 var_attr_edit={},
                 specfic_func=None):
    logger.info(f"Processing {file}...")
    filename = os.path.basename(file)
    file_date = extract_date_func(filename)

    # Retrieve track points near that date from CyclObs API
    track_points = get_track_points_from_api(file_date)

    # If no track points
    if len(track_points.index) == 0:
        print("No track points, skipping this file.")

    # Transform lon from -180, 180 to 0, 360
    # track_points["lon"] = track_points["lon"].apply(lambda x: shape360(x, 0)[0])
    track_points["lon"] = track_points["lon"] % 360

    dataset = xarray.open_dataset(file)
    df = dataset.to_dataframe()
    # Not processing file if it contains only NaNs
    if not df[wind_col].isnull().all():
        full_time = is_full_time(dataset, time_col)
        kept_track_points = new_get_colocated_track_points(dataset, track_points, file_date,
                                                        wind_col_name=wind_col, time_col_name=time_col,
                                                        pass_col_name=pass_col, is_full_time=full_time)

        new_extract_write_cyclone_data(dataset, kept_track_points, filename, output_path, file_date,
                                    source_filename=filename, wind_col_name=wind_col,
                                    time_col_name=time_col, lat_col_name=lat_col, lon_col_name=lon_col,
                                    pass_col_name=pass_col, attrs=attrs, attrs_to_del=attrs_to_del,
                                    attrs_rename=attrs_rename,
                                    var_to_del=var_to_del,
                                    filename_format=filename_format,
                                    is_full_time=full_time, extract_size_km=pass_width * 2, var_attr_edit=var_attr_edit,
                                    specific_func=specfic_func)
        return

def new_get_colocated_track_points(dataset, track_points, file_date, wind_col_name, is_full_time,
                               time_col_name, pass_col_name=None, lon_col_name="lon", lat_col_name="lat",
                               deg_delta=3, time_delta=60):

    track_point_offsets = []
    for index, track_point in track_points.iterrows():
        logger.debug(f"Track point : {track_point}")
        # Selecting point in dataset that is geographically the nearest from the current iterated track point.
        #sel = dataset.sel(**{lon_col_name: track_point["lon"], lat_col_name: track_point["lat"], "method": "nearest"})
        # Computing the distance between each pixel and the track point
        distance = np.sqrt((dataset[lon_col_name] - track_point["lon"])**2 +
                   (dataset[lat_col_name] - track_point["lat"])**2)

        idx = distance.argmin()

        multi_idx = np.unravel_index(idx, distance.shape)

        # Extraction du lon/lat du point le plus proche
        # closest_point_lon = dataset[lon_col_name].values[multi_idx].item()
        # closest_point_lat = dataset[lat_col_name].values[multi_idx].item()

        # # Find the index in the lon and lat array of that nearest pixel.
        # lon_index = np.where(dataset[lon_col_name].data == sel[lon_col_name].data)[0][0]
        # lat_index = np.where(dataset[lat_col_name].data == sel[lat_col_name].data)[0][0]
        # logger.debug(f"Point find : lon: {sel[lon_col_name].values}, lat: {sel[lat_col_name].values},"
        #              f" minute: {sel[time_col_name].values},"
        #              f" lon_index: {lon_index}, lat_index {lat_index}")

        trk_ptn = track_point.to_dict()

        ds_time = dataset[time_col_name].isel(**{dim: multi_idx[i] for i, dim in enumerate(dataset[time_col_name].dims)})
        
        if not pd.isna(ds_time):
            if not is_full_time:
                point_date = file_date + pd.to_datetime(ds_time.item())
            else:
                point_date = pd.to_datetime(ds_time.item())

            time_offset = abs(track_point["date"] - point_date)
            trk_ptn["time_offset"] = time_offset
            trk_ptn["node"] = None
            trk_ptn["multi_idx"] = multi_idx

        if "time_offset" in trk_ptn:
            track_point_offsets.append(trk_ptn)

    kept_track_points = {}
    # Getting the *best* track point for each sid (storm id).
    # One file can contain several cyclones acquisitions so that is why we are doing this filtering per sid
    # The kept track point are those that have temporal distance below 7 min 30 sec with the associated pixel and
    # if several track points match that criteria, the closest temporally is kept.
    for track_point in track_point_offsets:
        # TODO replace timedelta with database track sampling time
        if track_point["time_offset"] <= datetime.timedelta(minutes=7, seconds=30):
            if track_point["sid"] not in kept_track_points:
                kept_track_points[track_point["sid"]] = track_point
            elif kept_track_points[track_point["sid"]]["time_offset"] > track_point["time_offset"]:
                kept_track_points[track_point["sid"]] = track_point

    return kept_track_points

def new_extract_write_cyclone_data(dataset, kept_track_points, filename, output_path,
                               file_date, source_filename, attrs, attrs_to_del, attrs_rename,
                               var_to_del, lon_col_name,
                               lat_col_name, wind_col_name, time_col_name, pass_col_name,
                               filename_format,
                               is_full_time,
                               extract_size_km=2000,
                               var_attr_edit={},
                               specific_func=None,
                               min_pixel=5):
    for sid, track_point in kept_track_points.items():
        track_lon = track_point["lon"]
        track_lat = track_point["lat"]
        #print(f"track lon: {track_lon}, track_lat: {track_lat}")

        # Earth circumference
        earth_circum = 40000

        km_per_deg_lon = abs(earth_circum * math.cos(math.radians(track_lat)) / 360)
        km_per_deg_lat = 111
        lon_offset = (extract_size_km / 2) / km_per_deg_lon
        lat_offset = (extract_size_km / 2) / km_per_deg_lat
        #print(f"lon_offset: {lon_offset}, lat_offset: {lat_offset}")

        min_lon = track_lon - lon_offset
        max_lon = track_lon + lon_offset
        min_lat = track_lat - lat_offset
        max_lat = track_lat + lat_offset
        
        nan_mask = np.isnan(dataset[lon_col_name]) | np.isnan(dataset[lat_col_name])
        dataset = dataset.where(~nan_mask, drop=True)

        mask =  (dataset[lon_col_name] > min_lon) & (dataset[lon_col_name] < max_lon) & (dataset[lat_col_name] > min_lat) & (dataset[lat_col_name] < max_lat)

        dataset_filtered = dataset.isel(
            {
                dim: mask.any(dim=[d for d in mask.dims if d != dim])
                for dim in mask.dims
            }
        )
       
        sel = set_nan_outside_time_offset(dataset_filtered, track_point, time_col_name=time_col_name, wind_col_name=wind_col_name,
                                          is_full_time=is_full_time)
        

        if specific_func is not None:
            sel = specific_func(sel)

        # if there are less than x points, remove the image
        df = sel.to_dataframe()
        df = df[pd.isna(df[wind_col_name]) == False]
        if len(df.index) <= min_pixel:
            logger.info(f"Data has less than {min_pixel} pixel of data. Not writing to file.")
            return

        sel, start_date, stop_date, year, doy, resolution_km = new_set_metadata(sel, wind_col_name, time_col_name, file_date, attrs,
                                                        attrs_to_del, attrs_rename, var_to_del, var_attr_edit,
                                                        lon_col_name, source_filename, is_full_time)
        
        new_write_ncdf(sel, output_path, filename_format, start_date, stop_date, year, doy, track_point["sid"], resolution_km)

def new_compute_footprint(dataset, wind_field):
    # Using 0:360 range because the -180:180 range does create the right polygon with gdf.unary_union.convex_hull
    lon_atts = dataset["lon"].attrs
    dataset["lon"] = dataset["lon"] % 360
    dataset["lon"].attrs = lon_atts

    df = dataset.to_dataframe()
    df = df[pd.isna(df[wind_field]) == False]
    df = df.reset_index()
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.lon, df.lat))

    p = gdf.unary_union.convex_hull
    # p = p.simplify(5)
    # Transforming the polygon so its coords are in range -180:180
    # p = transform_polyg_180(p)
    # print(p)
    return p

def new_write_ncdf(dataset, output_path, filename_format, start_date, stop_date, year, doy, track_point_sid, resolution_km):
    """
    Écrit un dataset xarray en NetCDF, gérant correctement les dims spatiales et les coordonnées 2D.
    Compatible 0D, 1D et 2D.
    """
    # Construire le nom de fichier
    output_filename = filename_format.replace("<start_date>", start_date.strftime('%Y%m%dT%H%M%S'))
    output_filename = output_filename.replace("<stop_date>", stop_date.strftime('%Y%m%dT%H%M%S'))
    output_filename = output_filename.replace("<sid>", track_point_sid)
    output_filename = output_filename.replace("<resolution_km>", resolution_km)


    logger.info(f"Writing file {output_filename}")

    # Supprimer les dimensions de taille 1
    dataset = dataset.squeeze()

    # Créer le dossier de sortie
    out = os.path.join(output_path, str(year), doy)
    os.makedirs(out, exist_ok=True)

    # Check if dataset is 2D or more to set spatial dims with rioxarray
    if len(dataset.dims) >= 2 and 'lat' in dataset.coords and 'lon' in dataset.coords:
        # Convert dim in list to be able to slice
        dims_list = list(dataset.dims)
        y_dim, x_dim = dims_list[:2]  # taking the first two dimensions as spatial dims

        if x_dim not in ('x', 'lon'):
            dataset = dataset.rename({x_dim: 'x'})
            x_dim = 'x'
        if y_dim not in ('y', 'lat'):
            dataset = dataset.rename({y_dim: 'y'})
            y_dim = 'y'

        dataset.rio.set_spatial_dims(x_dim=x_dim, y_dim=y_dim, inplace=True)
        dataset.rio.write_crs("epsg:4326", inplace=True)
    else:
        logger.debug("Dataset is 0D or 1D, writing NetCDF without rioxarray.")

    # Écriture finale en NetCDF
    dataset.to_netcdf(os.path.join(out, output_filename), format="NETCDF4_CLASSIC")

def new_set_metadata(dataset, wind_col_name, time_col_name, file_date, attrs, attrs_to_del, attrs_rename,
                 var_to_del, var_attr_edit, lon_col_name, source_filename, is_full_time):
    
    resolution_km = dataset.attrs['pixel_size_on_horizontal'].split(' ')[0]

    for attr in attrs_rename:
        if attr in dataset.attrs:
            attrs[attrs_rename[attr]] = dataset.attrs[attr]

    # Deleting all attributes
    dataset.attrs.clear()

    # Extracting min and max time to set attributes on NetCDF
    min_time, max_time = extract_start_stop_measure(dataset, time_col_name)
    if max_time - min_time > datetime.timedelta(minutes=30):
        logger.error(f"Time offset between min_time ({min_time}) and max_time ({max_time}) exceeds 30 min.")
        sys.exit(1)

    logger.debug(f"min_time: {min_time}, max_time: {max_time}")
    # Computing acquisition start and stop dates and setting as global attribute
    if is_full_time:
        min_date = min_time
        max_date = max_time
    else:
        # In the case of SMAP, the time dataArray contains only a time (hour, min, sec) and no date.
        # To have a full date, we add the file date.
        min_date = file_date + min_time
        max_date = file_date + max_time

    year = min_date.year
    doy = min_date.strftime('%j')
    # measure_date = min_date + (max_date - min_date) / 2
    dataset.attrs["measurementStartDate"] = min_date.strftime("%Y-%m-%d %H:%M:%S")
    dataset.attrs["measurementStopDate"] = max_date.strftime("%Y-%m-%d %H:%M:%S")

    # Setting new attributes
    for attr in attrs:
        if type(attrs[attr]) == dict:
            for var_attr in attrs[attr]:
                # The _FillValue attribute is in the encoding dict
                if var_attr == "_FillValue":
                    dataset[attr].encoding[var_attr] = attrs[attr][var_attr]
                else:
                    dataset[attr].attrs[var_attr] = attrs[attr][var_attr]
        else:
            dataset.attrs[attr] = attrs[attr]

    # Removing attributes
    # FIXME potentially to remove because shadowed by the above deletion. But for now it still has an effect because
    # it deletes attributes from .encoding too.
    for attr in attrs_to_del:
        if type(attrs_to_del[attr]) == list:
            for var_attr in attrs_to_del[attr]:
                if var_attr in dataset[attr].attrs:
                    del dataset[attr].attrs[var_attr]
                if var_attr in dataset[attr].encoding:
                    del dataset[attr].encoding[var_attr]
        else:
            if attr in dataset.attrs:
                del dataset.attrs[attr]

    for var in var_to_del:
        if var in dataset.keys():
            dataset = dataset.drop_vars(var)

    new_wind_col = "wind_speed"
    dataset = dataset.rename_vars({wind_col_name: new_wind_col})

    footprint_polyg = new_compute_footprint(dataset, new_wind_col)

    
    lon_attrs = dataset[lon_col_name].attrs
    
    # Converting lon from 0;360 range to -180;180
    dataset[lon_col_name] = (dataset[lon_col_name] + 180) % 360 - 180
    dataset[lon_col_name].attrs = lon_attrs

    dataset.attrs["sourceProduct"] = source_filename
    dataset.attrs["footprint"] = dumps(footprint_polyg)
    # __version__ = get_distribution('extract_cyclones').version
    dataset.attrs["productVersion"] = version_extract_cyclones

    for var in var_attr_edit:
        for attr in var_attr_edit[var]:
            dataset[var].attrs[attr] = var_attr_edit[var][attr]

    return dataset, min_date, max_date, year, doy, resolution_km

def process_file_era5(file, output_path, extract_date_func, attrs, var_to_del, wind_cols,
                 time_col, lat_col, lon_col, filename_format, pass_width, pass_col=None, attrs_to_del=[],
                 attrs_rename={},
                 var_attr_edit={},
                 specfic_func=None):
    """
    Extract portion on data from file (data corresponding to a cyclone) and write it to a NetCDF file.


    Track points concerning the given file are retrieved from CyclObs API, then it is checked if
    one or several of these track points are actually colocated with data points in the given file.
    If there are, a portion of data around each selected point is extracted from file. These extracted portions
    correspond to interesting areas where a cyclone has been captured. That extracted data is
    saved as a well structured NETCDF file.

    Parameters
    ----------
    file : str Path to SMOS or SMAP file from which to extract the cyclone data
    output_path : str Path in which save the netCDF file
    extract_date_func : function Function used to extract the date using, as parameter, os.path.basename(file)
    attrs : dict Attributes that will be set in the output NETCDF file.
    var_to_del : list of str Variables that will not be included into the resulting netCDF file.
    wind_cols : str Name of the wind speed variables in the given file.
    time_col : str Name of the time variable in the given file.
    lat_col : str Name of the latitude variable in the given file.
    lon_col : str Name of the lngitude variable in the given file.
    filename_format : str Format to use to name the resulting NETCDF file. Example : RSS_smap_wind_<start_date>_<stop_date>_<sid>.nc
    pass_width : int Size of the extracted portion.
    pass_col : str or None Variable used to indicate if there is several pass in the given file. If there is (!=None), a string must be given indicating the variable name of the pass.
    attrs_to_del : dict of list Variable attributes to not include into resulting NETCDF. Keys : variable name, Value: list of attribute names
    attrs_rename : dict of str Rename global attributes. Key: attribute value to rename, Value: new attribute name
    var_attr_edit : dict of dict Edit variable attribute values. Format : {"var": {"attr": new_value}}
    specfic_func : function A function taking an xarray dataset as parameter and returning an xarray dataset. That function
    can perform any operation on the data.

    Returns
    -------
    Nothing

    """

    logger.info(f"Processing {file}...")

    filename = os.path.basename(file)
    file_date = extract_date_func(filename)
    logger.debug(f"File date {file_date}")

    # Retrieve track points near that date from CyclObs API
    track_points = get_track_points_from_api(file_date)
    logger.debug(f"Number of track point found : {len(track_points)}")

    # If no track points
    if len(track_points.index) == 0:
        logger.warning("No track points, skipping this file.")
        return

    # Transform lon from -180, 180 to 0, 360
    # track_points["lon"] = track_points["lon"].apply(lambda x: shape360(x, 0)[0])
    track_points["lon"] = track_points["lon"] % 360

    # dataset = xarray.open_dataset(file)
    with xarray.open_dataset(file) as dataset:
        # df = dataset.drop_vars(["d2m", "t2m", "msl", "sst", "sp", "u100", "v100", "hcc", "lcc", "tcc", "tclw", "blh", "tcw", "tcwv", "iews", "inss", "skt", "rsn", "sd", "stl1", "stl2", "stl3", "stl4", "swvl1", "swvl2", "swvl3", "swvl4", "zust", "z", "lsm", "siconc"])
        #["u100", "v100", "d2m", "t2m", "blh", "hcc", "lcc", "mdww", "mpww", "msl", "mwd", "p140122", "p1ps", "mwp", "pp1d", "r", "siconc", "sst", "swh", "shww", "p140121", "slhf", "ssr", "str", "sp", "sshf", "tcc", "tclw", "tcw", "tcwv", "tp"])
        LISTE = ["d2m", "t2m", "msl", "sst", "sp", "u100", "v100", "hcc", "lcc", "tcc", "tclw", "blh", "tcw", "tcwv", "iews", "inss", "skt", "rsn", "sd", "stl1", "stl2", "stl3", "stl4", "swvl1", "swvl2", "swvl3", "swvl4", "zust", "z", "lsm", "siconc"]
        to_drop = [v for v in LISTE if v in dataset.variables]
        dataset = dataset.drop_vars(to_drop)
        #dataset = df.drop_dims(["longitude050", "latitude050"])
        df = dataset.to_dataframe()
        # Not processing file if it contains only NaNs
        if not df[wind_cols[0]].isnull().all():
            full_time = is_full_time(dataset, time_col)
            kept_track_points = get_colocated_track_points(dataset, track_points, file_date, lon_col_name=lon_col,
                                                        lat_col_name=lat_col, wind_col_name=wind_cols[0],
                                                        time_col_name=time_col, pass_col_name=pass_col, is_full_time=full_time)
            logger.debug(f"For file {filename} Kept track that will be used to extract cyclone data: {kept_track_points}")

            extract_write_cyclone_data_era5(dataset, kept_track_points, filename, output_path, file_date,
                                    source_filename=filename, wind_col_names=wind_cols,
                                    time_col_name=time_col, lat_col_name=lat_col, lon_col_name=lon_col,
                                    pass_col_name=pass_col, attrs=attrs, attrs_to_del=attrs_to_del,
                                    attrs_rename=attrs_rename,
                                    var_to_del=var_to_del,
                                    filename_format=filename_format,
                                    is_full_time=full_time, extract_size_km=pass_width * 2, var_attr_edit=var_attr_edit,
                                    specific_func=specfic_func)
        else:
            logger.warning(f"Filename {filename} contains only NaN values for {wind_cols[0]} column. It will not be processed.")
            sys.exit(2)

def extract_write_cyclone_data_era5(dataset, kept_track_points, filename, output_path,
                               file_date, source_filename, attrs, attrs_to_del, attrs_rename,
                               var_to_del, lon_col_name,
                               lat_col_name, wind_col_names, time_col_name, pass_col_name,
                               filename_format,
                               is_full_time,
                               extract_size_km=2000,
                               var_attr_edit={},
                               specific_func=None,
                               min_pixel=5):
    for sid, track_point in kept_track_points.items():
        logger.debug(f"Track point : {track_point}")
        # Km per deg in latitude
        km_per_deg_lat = 111
        # Degrees offset per latitude index in data array
        deg_per_index_lat = abs(dataset[lat_col_name][0] - dataset[lat_col_name][1])
        # Km offset per latitude index in data array
        km_per_idx_lat = km_per_deg_lat * deg_per_index_lat
        logger.debug(f"km_per_idx_lat: {km_per_idx_lat}")
        # Number of latitude index to take to have extract_size_km size of data
        nb_idx_lat = round(float(extract_size_km / km_per_idx_lat) / 2)
        logger.debug(f"nb_idx_lat: {nb_idx_lat}")

        if track_point["node"] is not None:
            sel_lat = dataset.isel(**{lat_col_name: slice(track_point["lat_index"] - nb_idx_lat,
                                                          track_point["lat_index"] + nb_idx_lat),
                                      pass_col_name: track_point["node"]})
        else:
            sel_lat = dataset.isel(**{lat_col_name: slice(track_point["lat_index"] - nb_idx_lat,
                                                          track_point["lat_index"] + nb_idx_lat)})

        # In next section, the goal is to compute how many longitude index we must take to have a data section of
        # 'extract_size_km'. As the km offset between each longitude degrees depends of the latitude, we first compute
        # the latitude_average of the current selected data and compute the km_per_deg at that latitude.

        # Latitude average of the current area
        logger.debug(f"sel_lat[lat_col_name] : {sel_lat[lat_col_name]}")
        lat_av = (sel_lat[lat_col_name].max() + sel_lat[lat_col_name].min()) / 2
        logger.debug(f"lat_av: {lat_av}")
        # Earth circumference
        earth_circum = 40000
        # Kilometer per longitude deg at current latitude
        km_per_deg = abs(earth_circum * math.cos(math.radians(lat_av)) / 360)
        logger.debug(f"km_per_deg: {km_per_deg}")
        # Deg offset per index in data
        deg_per_index = abs(sel_lat[lon_col_name][0] - sel_lat[lon_col_name][1])
        # Km offset per index in data
        km_per_idx = km_per_deg * deg_per_index
        logger.debug(f"km_per_idx: {km_per_idx}")
        # Number of longitude index to take to have extract_size_km size of data
        nb_idx_lon = float(extract_size_km / km_per_idx)
        nb_idx_lon_2 = round(nb_idx_lon / 2)
        logger.debug(f"nb_idx_lon_2 : {nb_idx_lon_2}")

        # Selecting area of interest
        if track_point["node"] is not None:
            sel = dataset.isel(**{lon_col_name: slice(track_point["lon_index"] - nb_idx_lon_2,
                                                      track_point["lon_index"] + nb_idx_lon_2),
                                  lat_col_name: slice(track_point["lat_index"] - nb_idx_lat,
                                                      track_point["lat_index"] + nb_idx_lat),
                                  pass_col_name: track_point["node"]}, drop=True)
        else:
            sel = dataset.isel(**{lon_col_name: slice(track_point["lon_index"] - nb_idx_lon_2,
                                                      track_point["lon_index"] + nb_idx_lon_2),
                                  lat_col_name: slice(track_point["lat_index"] - nb_idx_lat,
                                                      track_point["lat_index"] + nb_idx_lat)}, drop=True)

        # Removing data that is on other swath than the one where the cyclone is
        # sel = set_nan_outside_time_offset(sel, track_point, time_col_name=time_col_name, wind_col_name=wind_col_name,
        #                                   is_full_time=is_full_time)

        if specific_func is not None:
            sel = specific_func(sel)

        # if there are less than x points, remove the image
        df = sel.to_dataframe()
        wind_col_name = wind_col_names[0]  # Taking the first wind col name to check for data presence
        df = df[pd.isna(df[wind_col_name]) == False]
        if len(df.index) <= min_pixel:
            logger.info(f"Data has less than {min_pixel} pixel of data. Not writing to file.")
            return

        sel, start_date, stop_date, year, doy = set_metadata_era5(sel, wind_col_names, time_col_name, file_date, attrs,
                                                        attrs_to_del, attrs_rename, var_to_del, var_attr_edit,
                                                        lon_col_name, lat_col_name, source_filename, is_full_time)
        write_ncdf(sel, output_path, filename_format, start_date, stop_date, year, doy, track_point["sid"])

def set_metadata_era5(dataset, wind_col_names, time_col_name, file_date, attrs, attrs_to_del, attrs_rename,
                 var_to_del, var_attr_edit, lon_col_name, lat_col_name, source_filename, is_full_time):

    for attr in attrs_rename:
        if attr in dataset.attrs:
            attrs[attrs_rename[attr]] = dataset.attrs[attr]

    # Deleting all attributes
    dataset.attrs.clear()

    # Extracting min and max time to set attributes on NetCDF
    min_time, max_time = extract_start_stop_measure(dataset, time_col_name)
    # if max_time - min_time > datetime.timedelta(minutes=30):
    #     logger.error(f"Time offset between min_time ({min_time}) and max_time ({max_time}) exceeds 30 min.")
    #     sys.exit(1)

    logger.debug(f"min_time: {min_time}, max_time: {max_time}")
    # Computing acquisition start and stop dates and setting as global attribute
    if is_full_time:
        min_date = min_time
        max_date = max_time
    else:
        # In the case of SMAP, the time dataArray contains only a time (hour, min, sec) and no date.
        # To have a full date, we add the file date.
        min_date = file_date + min_time
        max_date = file_date + max_time

    year = min_date.year
    doy = min_date.strftime('%j')
    # measure_date = min_date + (max_date - min_date) / 2
    dataset.attrs["measurementStartDate"] = min_date.strftime("%Y-%m-%d %H:%M:%S")
    dataset.attrs["measurementStopDate"] = max_date.strftime("%Y-%m-%d %H:%M:%S")

    # Setting new attributes
    for attr in attrs:
        if type(attrs[attr]) == dict:
            for var_attr in attrs[attr]:
                # The _FillValue attribute is in the encoding dict
                if var_attr == "_FillValue":
                    dataset[attr].encoding[var_attr] = attrs[attr][var_attr]
                else:
                    dataset[attr].attrs[var_attr] = attrs[attr][var_attr]
        else:
            dataset.attrs[attr] = attrs[attr]

    # Removing attributes
    # FIXME potentially to remove because shadowed by the above deletion. But for now it still has an effect because
    # it deletes attributes from .encoding too.
    for attr in attrs_to_del:
        if type(attrs_to_del[attr]) == list:
            for var_attr in attrs_to_del[attr]:
                if var_attr in dataset[attr].attrs:
                    del dataset[attr].attrs[var_attr]
                if var_attr in dataset[attr].encoding:
                    del dataset[attr].encoding[var_attr]
        else:
            if attr in dataset.attrs:
                del dataset.attrs[attr]

    for var in var_to_del:
        if var in dataset.keys():
            dataset = dataset.drop_vars(var)

    # Computing the wind speed from the given wind components
    u_comp = dataset[wind_col_names[0]]
    v_comp = dataset[wind_col_names[1]]
    wind_speed = np.sqrt(u_comp**2 + v_comp**2)
    dataset["wind_speed"] = wind_speed

    # Deleting the original wind component variables
    for wind_col in wind_col_names:
        dataset = dataset.drop_vars(wind_col)

    new_lon_col_name = "lon"
    new_lat_col_name = "lat"

    dataset = dataset.rename({lat_col_name: new_lat_col_name, lon_col_name: new_lon_col_name})

    #print("Dataset variables: ", dataset.data_vars)

    footprint_polyg = compute_footprint(dataset, "wind_speed")

    # Converting lon from 0;360 range to -180;180
    lon_attrs = dataset[new_lon_col_name].attrs
    dataset[new_lon_col_name] = deg180(dataset[new_lon_col_name])
    dataset[new_lon_col_name].attrs = lon_attrs

    dataset.attrs["sourceProduct"] = source_filename
    dataset.attrs["footprint"] = dumps(footprint_polyg)
    # __version__ = get_distribution('extract_cyclones').version
    dataset.attrs["productVersion"] = version_extract_cyclones

    for var in var_attr_edit:
        for attr in var_attr_edit[var]:
            dataset[var].attrs[attr] = var_attr_edit[var][attr]

    return dataset, min_date, max_date, year, doy