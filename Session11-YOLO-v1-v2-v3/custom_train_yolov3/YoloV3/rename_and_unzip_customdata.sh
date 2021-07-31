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
    
    # copy the 100 custom images
    rsync -avi /content/gdrive/MyDrive/tsai_eva6/session_11_yolo_v1_v2_v3/my_custom_dataset/images/ customdata/images/         # copy your images to the custom dataset's images/  directory
    rsync -avi /content/gdrive/MyDrive/tsai_eva6/session_11_yolo_v1_v2_v3/my_custom_dataset/labels/ customdata/labels/         # copy your labels to the custom dataset's labels/  directory
    cat /content/gdrive/MyDrive/tsai_eva6/session_11_yolo_v1_v2_v3/my_custom_dataset/test.txt       >> customdata/test.txt     # add your test/validation images to custom dataset's test.txt
    cat /content/gdrive/MyDrive/tsai_eva6/session_11_yolo_v1_v2_v3/my_custom_dataset/train.txt       >> customdata/train.txt   # add your train images to custom dataset's train.txt
fi;
