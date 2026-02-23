#!/usr/bin/env python

import argparse
import logging
import datetime
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from extract_cyclones import process_file_era5

#logging.basicConfig()
logger = logging.getLogger(os.path.basename(__file__))
#logger.setLevel(logging.INFO)


def extract_date_from_filename(filename):
    logger.debug(f"Filename: {filename}")
    #filename example: '/home/ref-ecmwf/ERA5T/2025/01/era5_single-levels_inst_20250101.nc'
    date_part = filename.split("_")[-1].split(".")[0]
    return datetime.datetime.strptime(date_part, "%Y%m%d")


if __name__ == "__main__":
    description = """
        Read ERA5 netCDF files from a directory, extract the wind data of cyclones and save it into a new netCDF file.
        """
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("--debug", action="store_true", default=False, help="Run in debug mode (verbose output)")
    parser.add_argument("-i", "--input", action="store", type=str, required=True, nargs="+",
                        help="Input files from which extract cyclone data")
    parser.add_argument("-o", "--output", action="store", type=str, default="./output_test",
                        help="Output path where files will be written")
    args = parser.parse_args()

    if sys.gettrace():
        logger.setLevel(logging.DEBUG)
        logging.basicConfig(level=logging.DEBUG)

    if args.debug:
        logger.setLevel(logging.DEBUG)
        logging.basicConfig(level=logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
        logging.basicConfig(level=logging.INFO)

    from extract_cyclones import process_file

    #attrs_to_del = ["aprrox_local_equatorial_crossing_time", "swath_sector", "first_orbit", "last_orbit"]

    #var_attr_edit = {"lon": {"valid_min": -180, "valid_max": 180}}

    # Attributes to add to the era5 netcdf
    attrs = {"Conventions": "CF-1.6",
             "title": "ERA5 ocean surface wind speed (10m above surface) cropped around Tropical Cyclone track",
             "institution": "ECMWF",
             "reference": "https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5",
             "sourceReference": ("European Centre for Medium-Range Weather Forecasts (ECMWF), 2017: ERA5: Fifth generation of ECMWF atmospheric reanalyses of the global climate. ECMWF, Reading, United Kingdom. Available online at https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5"),
             "missionName": "ERA5"
             }
    attrs_to_del = {}#"wind": ["missing_value"]}
    attrs_rename = {"version": "sourceProductVersion"}

    for f in args.input:
        # TODO generate filename_format depending on input filename
        process_file_era5(file=f, output_path=args.output, extract_date_func=extract_date_from_filename,
                     var_to_del=["e"],#["u100", "v100", "d2m", "t2m", #"msl", "sst", "sp", "hcc", "lcc", "tcc", "tclw", "blh", "tcw", "tcwv", "iews", "inss", "skt", "rsn", "sd", "stl1", "stl2", "stl3", "stl4", "swvl1", "swvl2", "swvl3", "swvl4", "zust", "z", "lsm", "siconc"],
                                                                #"blh", "hcc", "lcc", "mdww", "mpww", "msl", "mwd", "p140122", "p1ps", "mwp", "pp1d", "r", "siconc", "sst", "swh", "shww", "p140121", "slhf", "ssr", "str", "sp", "sshf", "tcc", "tclw", "tcw", "tcwv", "tp"],
                     attrs=attrs,
                     attrs_to_del=attrs_to_del,
                     attrs_rename=attrs_rename,
                     wind_cols=["u", "v"], time_col="valid_time",
                     lat_col="latitude",
                     lon_col="longitude", pass_width=2000,
                     filename_format="ERA5_<start_date>_<stop_date>_<sid>.nc")