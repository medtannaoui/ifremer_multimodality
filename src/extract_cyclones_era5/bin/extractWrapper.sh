#!/bin/bash
#echo "Starting extractWrapper.sh with args: $@"
cmd=("$@")
output_path=""
file_path=""

# parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    -o) output_path="$2"; shift 2 ;;
    -i) file_path="$2"; shift 2 ;;
    *) shift ;;
  esac
done
filename="${file_path##*/}"
filename_no_ext="${filename%.*}"

# TODO put report file in same dir as generated files

file_report_dir="$output_path/report/$filename_no_ext"
mkdir -p "$file_report_dir"
#echo "Running $1 ${@:2} > $file_report_dir/Extract.log 2>&1"
# $1 "${@:2}" > "$file_report_dir/Extract.log" 2>&1
"${cmd[@]}" > "$file_report_dir/Extract.log" 2>&1
status=$?
echo $status > "$file_report_dir/Extract.status"

cat "$file_report_dir/Extract.log"

exit $status
