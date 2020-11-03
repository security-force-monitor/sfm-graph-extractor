#!/bin/bash
# Pipeline from NER to RE

if [ -z "$1" ]; then

  echo "Enter an output directory name."

else

  env_name=kge
  dataset_path=../SFM_STARTER/annotated_sources
  output_dir=$1
  ner_dir=NER
  re_dir=RE
  re_method="2. dep"
  dep_dir=jPTDP-master
  jPTDP_buffer=buffer

  cd ../..
  # ========================== NER ==========================
  # echo "Results in directory: $output_dir"
  #
  # conda activate $env_name
  # cd $ner_dir
  # # mkdir $output_dir
  # # cp $dataset_path/*.txt $output_dir
  # # rm $output_dir/*_meta.txt
  #
  # doc_num=0
  # for entry in $output_dir/*.txt; do
  #   doc_num=$((doc_num+1))
  # done
  #
  # echo "Number of documents: $doc_num"
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
  conda activate $env_name

  for path_to_entry in $output_dir/*.txt; do
    entry=$(echo "$path_to_entry" | cut -f 2 -d '/')
    id=$(echo "$entry" | cut -f 1 -d '.')

    # Run dependency parsing
    # cd ../$dep_dir
    # cp ../$re_method/$output_dir/$entry $jPTDP_buffer
    # conda deactivate
    # source .DyNet/bin/activate
    # python parse_script.py $jPTDP_buffer/$entry
    # deactivate
    # cd ../$re_method
    # mkdir $output_dir/$id
    # mv ../$dep_dir/$jPTDP_buffer/* $output_dir/$id
    # conda activate $env_name

    # Use dependency parsing to extract relations
    python relation_dep.py $output_dir/$id $output_dir/$entry $output_dir/$id.ann
  done

  # cd ../..
  # conda deactivate

fi
