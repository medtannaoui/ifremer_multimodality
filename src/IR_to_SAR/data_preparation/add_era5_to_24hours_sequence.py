import pickle as pkl
import os 
print(os.getcwd())
import numpy as np
import src.IR_to_SAR.data_preparation.regrid_era5.regrid_era5 as regrid_colocs
os.chdir("/scale/user/mtannaou/alternance")

from tqdm import tqdm


def add_era5():
    # données pour era5
    era5_path = "/scale/user/mtannaou/alternance/src/extract_cyclones_era5/era5_single_levels"
    janvier, mars, mai, juillet, aout, octobre, decembre = (
        np.arange(1, 32, 1),
        np.arange(1, 32, 1),
        np.arange(1, 32, 1),
        np.arange(1, 32, 1),
        np.arange(1, 32, 1),
        np.arange(1, 32, 1),
        np.arange(1, 32, 1),
    )
    avril, juin, septembre, novembre = (
        np.arange(1, 31, 1),
        np.arange(1, 31, 1),
        np.arange(1, 31, 1),
        np.arange(1, 31, 1),
    )

    with open("src/IR_to_SAR/ML_IR_SAR/csv_data/one_day_24hours_data_with_infos_stride1.pkl","rb") as f:
        data = pkl.load(f)
    
    count = 0
    for ind in tqdm(range(len(data)), desc="adding era5 ...", total=len(data)-365):
        sample = data[ind+365]
        cyclone_id  = sample["cyc_id"]
        er = 0
        try :
            for sequence in sample["sequence"] : 
                sequence["era5"] = None
                date = str(sequence["target_date"])
                year = date[0:4]
                month = date[5:7]
                day = date[8:10]
                hour = date[11:13]
                minute = date[14:16]
                year_path = os.path.join(era5_path, str(year))
                fevrier = np.arange(1, 29, 1) if int(year) % 4 != 0 else np.arange(1, 30, 1)
                months = [
                    janvier, fevrier, mars, avril, mai, juin,
                    juillet, aout, septembre, octobre, novembre, decembre,
                ]
                ndays = 0
                for i in range(int(month) - 1):
                    ndays += len(months[i])
                ndays += int(day)
                ndays_str = "0" + str(ndays) if len(str(ndays)) < 3 else str(ndays)
                dayera5_path = os.path.join(year_path, ndays_str)
                nc_path = ""
                for nc_file in os.listdir(dayera5_path):
                    if cyclone_id in nc_file:
                        nc_path = os.path.join(dayera5_path, nc_file)
                        break
                reg_era5 = regrid_colocs.regrid_files_era5(
                    [nc_path],
                    "/scale/user/mtannaou/alternance/src/IR_to_SAR/data_preparation/"
                    "regrid_era5/regridded_era5",
                    resolution_km=2,
                    grid_size_km=512,
                    index_hour=int(hour) - 1 + int(minute) // 30,
                )[0]  
                sequence["era5"] = reg_era5
                er += 1 
            if er == 12:
                sample["all_era5"] = 1
                count += 1
            else : 
                sample["all_era5"] = 0

        except : 
            sample["all_era5"] = 0
            continue                                                                  ## 300km , resolution 2km

    with open("/scale/user/mtannaou/alternance/src/IR_to_SAR/ML_IR_SAR/csv_data/one_day_24hours_with_era5.pkl","wb") as f:
        pkl.dump(data, f)

    print(count)

if __name__ == "__main__":
    add_era5()


