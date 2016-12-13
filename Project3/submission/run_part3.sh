#!/bin/bash

# use the two datasets
DATASET_LIST="multisource news"
SUM=0.0

for data in $DATASET_LIST; do 
    echo "current dataset: $data"
    # in/out file
    in_file=datasets+scoring_script/xling_test_gs/STS.input.$data.txt    # the input xlingual file
    es_file=output/part3/ES.$data.txt                                    # es sentences from in_file
    es_file_mask=output/part3/ES.$data.pickle                            # the index of each sentence from pairs
    mt_file=datasets+scoring_script/xling_test_mt/STS.input.$data.MT.txt # thie file include translations
    mg_file=output/part3/STS.input.MT.$data.txt                          # the output of merging the eng with in_file
    out_file=output/part3/out.$data.txt                                  # the output for the predictions
    tru_file=datasets+scoring_script/xling_test_gs/STS.gs.$data.txt      # the Gold standard file
    # do predictions
    python predict_baseline.py $mg_file $out_file
    corr="$(python get_correlation.py $out_file $tru_file)"
    # parse the lines
    SUM="$(bc -l <<< "$SUM+$corr")"
    echo "$data: $corr"
done

# perform the average
AVERAGE="$(bc -l <<< "$SUM/2.0")"
echo "average performance is [$AVERAGE]"