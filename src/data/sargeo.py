# this script is used to create sargeo database on csv format

#creer la base de données excel du SARGEO 
import os
import xarray as xr
import pandas as pd


run = True


sargeo_path = "/scale/project/ifremer-isi-jumeaunumerique/SARGEO/prototype/v01r02/cyclobs" 

# cyclones list
cyclones = os.listdir(sargeo_path)

infos = []

for cyclone in cyclones:
    cyclone_path = os.path.join(sargeo_path, cyclone)
    if not os.path.isdir(cyclone_path):
        continue

    # IRWIn and WV existing check as a subfolders
    for subdir in ["IRWIN", "WV"]:
        sub_path = os.path.join(cyclone_path, subdir)
        if not os.path.exists(sub_path):
            # print(f"  ⚠️ No folder {subdir} of {cyclone}")
            continue

        # netcdf files list
        nc_files = [f for f in os.listdir(sub_path) if f.endswith(".nc")]
        # print(f"  📁 {subdir} → {len(nc_files)} files founded")

        for nc_file in nc_files:
            file_path = os.path.join(sub_path, nc_file)
            try:
                ds = xr.open_dataset(file_path)

                # existing path check
                lat_centre = float(ds["storm_latitude"].values[4]) if "storm_latitude" in ds else None
                lon_centre = float(ds["storm_longitude"].values[4]) if "storm_longitude" in ds else None
                time = str(ds["sar_acquisition_time"].values) if "sar_acquisition_time" in ds else None

                infos.append({
                    "cyclone": cyclone,
                    "canal": subdir,
                    "fichier": nc_file,
                    "lat_centre": lat_centre,
                    "lon_centre": lon_centre,
                    "sar_acquisition_time": time
                })

                ds.close()

            except Exception as e:
                print(f"  ⚠️ error with  {file_path} : {e}")


def main(output_dir = "/scale/user/mtannaou/alternance/excels/sargeo.csv"):
    # Convert to dataframe
    df = pd.DataFrame(infos)

    # save on csv file
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    df.to_csv(output_dir, index=False)

    print(f"\n File saved on : {output_dir}")
    print(f"{len(df)} files")


if run :
    main()