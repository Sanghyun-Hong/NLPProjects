
DATASET=answer-answer
DATASET=headlines
DATASET=plagiarism
DATASET=postediting
DATASET=question-question

IN_FILE=datasets+scoring_script/test/STS2016.input.$DATASET.txt
OUT_FILE=output/out.$DATASET.txt

python predict_baseline.py $IN_FILE $OUT_FILE
python get_correlation.py $IN_FILE $OUT_FILE
