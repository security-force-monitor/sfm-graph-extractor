#!/bin/bash
# Pipeline from NER to RE

output_dir="../4. dep+np/out"
comp_dir="./ann/no_mod"

for path_to_pred in $output_dir/*.txt; do
  entry=$(echo "$path_to_pred" | cut -f 4 -d '/')
  id=$(echo "$entry" | cut -f 1 -d '.')

  pred_ann=$output_dir/$id.ann
  comp_ann=$comp_dir/$id.ann

  echo "============================================"
  echo $id
  # echo $path_to_pred
  # echo $pred_ann
  # echo $comp_ann

  diff -u <(sort $pred_ann) <(sort $comp_ann)

done
