#!/bin/bash

[ -z "$2" ] && { echo "Usage: $0 <zip_file_name> <yolov3_dir_path>"; exit 1; } 

set -xe
# if data/customdata_walle does not exist, 
if [ ! -d "$2/data/customdata_walle" ]; then
    # then rename data/customdata to data/customdata_walle
    mv -v "$2/data/customdata" "$2/data/customdata_walle"
    
    # unzip customdata.zip.  this creates data/customdata directory
    cd $2/data
    unzip -q $1 
fi;
