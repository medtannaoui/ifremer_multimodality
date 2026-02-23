#!/bin/bash

GETOPT=$(getopt -n $0 -o d:o:h:: --long smos,smap,scat,era5,help -- "$@")
eval set -- "$GETOPT"

echo $(pwd)
while true; do
    case "$1" in
        -d) days=$2 ; shift 2 ;;
        -o) output_path=$2 ; shift 2 ;;
        --smos) smos=True ; shift ;;
        --smap) smap=True ; shift ;;
        --scat) scat=True ; shift ;;
        --era5) era5=True ; shift ;;
        -- ) shift ; break ;;
        * ) echo "unknown opt $1" >&2 ; exit 1 ;;
    esac
done

script_dir=$(dirname $(readlink -f "$0")) # ie $CONDA_PREFIX/bin
project_root=$(readlink -f "$script_dir/..")
export PYTHONPATH="$project_root:${PYTHONPATH}"
# Activate our conda environment
if [ -z "$CONDA_PREFIX" ] ; then
    conda_dir=$(readlink -f $script_dir/..)
    . "$conda_dir/bin/activate $conda_dir"
fi

if [ -z "$output_path" ] ; then
  echo "The output path is not defined, define it with -o <output_path>"
  exit 1
fi

smap_paths=("/home/datawork-cersat-public/provider/remss/satellite/l3/smap/smap/wind/v1.0") # includes daily and daily_nrt
smos_paths=("/home/ref-smoswind-public/data/v3.0/l3/data")
scat_paths=(
   #"/home/datawork-cersat-public/provider/knmi/satellite/l2b/hy-2b/hscat/25km/data"
           #"/home/datawork-cersat-public/provider/knmi/satellite/l2b/hy-2c/hscat/25km/data"
          #"/home/datawork-cersat-public/provider/knmi/satellite/l2b/metop-b/ascat/25km/data"
          "/home/datawork-cersat-public/provider/knmi/satellite/l2b/metop-b/ascat/12.5km/data"
          # "/home/datawork-cersat-public/provider/knmi/satellite/l2b/metop-b/ascat/5.7km/data"
           #"/home/datawork-cersat-public/provider/knmi/satellite/l2b/metop-c/ascat/12.5km/data"

          ) # multiple paths
era5_paths=(
  "/home/ref-ecmwf/ERA5T"
  #"/home/ref-ecmwf/ERA5/2023"
  #"/home/ref-ecmwf/ERA5/2024"
  ) # ERA5T single levels instantaneous data

if [ -z "$smos" ] && [ -z "$smap" ] && [ -z "$scat" ] && [ -z "$era5" ] ; then
  echo "You must chose which archive to update with --smos, --smap, --scat or --era5"
  exit 1
fi

if [ "$smap" == "True" ] && [ "$smos" == "True" ] ; then
  echo "Cannot update SMOS and SMAP at the same time. Only chose --smos or --smap."
  exit 1
elif [ "$smap" == "True" ]; then
  inpaths=("${smap_paths[@]}")
elif [ "$smos" == "True" ]; then
  inpaths=("${smos_paths[@]}")
elif [ "$scat" == "True" ]; then
  inpaths=("${scat_paths[@]}")
elif [ "$era5" == "True" ]; then
  inpaths=("${era5_paths[@]}")
fi

shopt -s globstar
if [[ -n $days ]]; then
    echo "Fetching files for the last $days days. Generated files will be written in $output_path"
else
  echo "Updating whole archive in $output_path"
fi
# Converting the date "$days days ago" to seconds since UNIX-time
sec=$(date --date="$days days ago" +%s)

files_to_process=""

# Keeping only files that are in the requested time interval for SMAP
for p in "${inpaths[@]}"; do
  #echo "$p"

  for file in "$p"/**/*.nc; do
    #echo " $file"

      #echo "################# debug $file $inpath/**/*.nc"
    if [ "$smap" == "True" ] ; then
      # If the current file is an NRT file, checking that the FINAL file does not exist. If it does, skipping the NRT file.
      if [[ "$file" == *"daily_nrt"* ]]; then
        final_file=${file/daily_nrt/daily}
        final_file=${final_file/_NRT/}
        final_dir=$(dirname "$final_file")
        if [ -d "$final_dir" ] && [ -f "$final_file" ]; then
          # Skipping NRT file
          echo "skipping NRT $file"
          continue
        fi
      fi
      secDt=$(basename "$file" | awk -F_ '{print $5"-"$6"-"$7}' | xargs -IDT date -d DT +%s)
    elif [ "$smos" == "True" ] ; then
      if [[ "$file" == *"nrt"* ]]; then
        final_file=${file/nrt/reprocessing}
        final_dir=$(dirname "$final_file")
        if [ -d "$final_dir" ] && [ -f "$final_file" ]; then
          # Skipping NRT file
          echo "skipping NRT $file"
          continue
        fi
      fi
      secDt=$(echo "$file" | awk -F_ '{print $5}' | xargs -IDT date -d DT +%s 2>/dev/null)
    elif [ "$era5" == "True" ] ; then
    # example filename: /home/ref-ecmwf/ERA5/2023/12/era_5-copernicus__20231231.nc
      # secDt=$(basename "$file" | awk -F_ '{print $4}' | sed 's/.nc$//' | xargs -IDT date -d DT +%s 2>/dev/null)
      secDt=$(basename "$file" \
                  | awk -F_ '{d=$NF; sub(/\.nc$/,"",d); print d}' \
                  | xargs -IDT date -d DT +%s 2>/dev/null)
      if [[ -z "$secDt" ]]; then
        echo "Invalid date in filename for $file → skipping"
        continue
      fi
    elif [ "$scat" == "True" ] ; then
      secDt=$(basename "$file" | awk -F_ '{print $2}' | xargs -IDT date -d DT +%s 2>/dev/null)
      if [[ -z "$secDt" ]]; then
        echo "Invalid date in filename for $file → skipping"
        continue
      
      fi
    fi
    #echo "################# debug $secDt > $sec"
    if [[ -z $days ]] || (( $secDt >= $sec )); then
      files_to_process="$files_to_process"$'\n'"$file"
    fi
  done
done

  


# Removing empty lines
files_to_process=$(echo "$files_to_process" | sed -r '/^\s*$/d')

#printf "%s\n $files_to_process"
# Generating cyclone files
if [ "$smap" == "True" ] ; then
  echo "Generating SMAP files..."
  #printf "%s\n $files_to_process" | xargs -r -P 5 extractSmapCyclone.py --dbd "$DATABASE_URL" -o "$output_path" -i
  printf "%s\n $files_to_process" | xargs -r -P 1 -L 1 extractWrapper.sh extractSmapCyclone.py -o "$output_path" -i
elif [ "$smos" == "True" ] ; then
  echo "Generating SMOS files..."
  printf "%s\n $files_to_process" | xargs -r -P 1 -L 1 extractWrapper.sh extractSmosCyclone.py -o "$output_path" -i
elif [ "$scat" == "True" ] ; then
  echo "Generating SCAT files..."
  printf "%s\n $files_to_process" | xargs -r -P 1 -L 1 extractWrapper.sh extractScatCyclone.py -o "$output_path" -i
elif [ "$era5" == "True" ] ; then
  echo "Generating ERA5 files..."
  printf "%s\n" "$files_to_process" | xargs -r -P 1 -L 1 bin/extractWrapper.sh python bin/extractERA5Cyclone.py -o "$output_path" -i
fi