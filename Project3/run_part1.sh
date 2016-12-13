#!/bin/bash

# use the five datasets
DATASET_LIST="answer-answer headlines plagiarism postediting question-question"
SUM=0.0

for data in $DATASET_LIST; do 
    echo "current dataset: $data"
    # in/out file
    input_file=datasets+scoring_script/test/STS2016.input.$data.txt
    output_file=output/part1/out.$data.txt
    # do predictions
    #python predict_baseline.py $input_file $output_file
    corr="$(python get_correlation.py $input_file $output_file)"
	echo "$corr"
    # parse the lines
    SUM="$(bc -l <<< "$SUM+$corr")"
done

# perform the average
AVERAGE="$(bc -l <<< "$SUM/5.0")"
echo "average performance is [$AVERAGE]"

# RESULTS: (0.67397+0.71794+0.82125+0.83391+0.65056)/5 = 0.739526
