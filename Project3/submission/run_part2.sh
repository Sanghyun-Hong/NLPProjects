#!/bin/bash

# use the five datasets
DATASET_LIST="answer-answer headlines plagiarism postediting question-question"
SUM=0.0

for data in $DATASET_LIST; do 
    echo "current dataset: $data"
    # in/out file
    input_file=datasets+scoring_script/test/STS2016.input.$data.txt
    truth_file=datasets+scoring_script/test/STS2016.gs.$data.txt
    output_file=output/part2/out.st.$data.txt
    # do predictions
    python predict_fancy_monolingual.py $input_file $output_file
    corr="$(./datasets+scoring_script/correlation-noconfidence.pl $truth_file $output_file)"
    # print the results
    echo $corr
done
