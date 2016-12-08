# RESULTS: (0.67397+0.71794+0.82125+0.83391+0.65056)/5 = 0.739526

#DATASET=answer-answer # Pearson: 0.67397 ST:Pearson: 0.27136
DATASET=headlines # Pearson: 0.71794
#DATASET=plagiarism # Pearson: 0.82125
#DATASET=postediting # Pearson: 0.83391
#DATASET=question-question # Pearson: 0.65056

echo "Dataset: $DATASET"

IN_FILE=datasets+scoring_script/test/STS2016.input.$DATASET.txt
OUT_FILE=output/out.st.$DATASET.txt

python predict_skip-thoughts.py $IN_FILE $OUT_FILE
#python get_correlation.py $IN_FILE $OUT_FILE
