#!/usr/bin/env bash

zenodoaddress="https://zenodo.org/api/records/10288494/files-archive"

if [ ! -f FrozenMeltData.zip ]; then
    echo "Fetching from $zenodoaddress"
    curl $zenodoaddress --output FrozenMeltData.zip
fi

mkdir FrozenMeltData
unzip FrozenMeltData.zip -d FrozenMeltData
unzip FrozenMeltData/PostProcessed.zip
mv PostProcessed/* post_processing/
unzip FrozenMeltData/VTU.zip
mv VTU/* seismic_post_processing/data/

rm -r FrozenMeltData
rm -r VTU
rm -r PostProcessed