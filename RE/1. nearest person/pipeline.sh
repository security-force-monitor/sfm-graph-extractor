#!/bin/bash
# Pipeline from NER to RE

env_name=kge
dataset_path=SFM_STARTER/annotated_sources
output_dir=out
ner_dir=NER
re_dir=RE
re_method="1. nearest person"

cd ../..
# ========================== NER ==========================
# conda activate $env_name
# cd $ner_dir
# mkdir $output_dir
# cp $dataset_path/*.txt $output_dir
# rm $output_dir/*_meta.txt
#
# doc_num=0
# for entry in $output_dir/*.txt; do
#   doc_num=$((doc_num+1))
# done
#
# echo $doc_num
# progress_idx=0
# for entry in $output_dir/*.txt; do
#   python ner.py "$entry" > /dev/null
#   progress_idx=$((progress_idx+1))
#   echo "Progress " $progress_idx / $doc_num ": $entry"
# done
#
# cd ..



# ========================== RE ==========================
cd $re_dir
cd $re_method
cp -r ../../$ner_dir/$output_dir ./

# entry=361de417-fe5f-4a76-9637-144f104713db.txt
for path_to_entry in $output_dir/*.txt; do
  entry=$(echo "$path_to_entry" | cut -f 2 -d '/')
  id=$(echo "$entry" | cut -f 1 -d '.')
  python relation_np.py $output_dir/$entry $output_dir/$id.ann
done

cd ../..
