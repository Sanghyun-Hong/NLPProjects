# RESULTS: (0.88545+0.81606)/2 = 0.850755

#DATASET=multisource # Pearson: 0.81606
#DATASET=news # Pearson: 0.88545

echo "Dataset: $DATASET"

IN_FILE=datasets+scoring_script/xling_test_gs/STS.input.$DATASET.txt # the input xlingual file
ES_FILE=output_xl/ES.$DATASET.txt # this file will contain the spanish sentences from IN_FILE
ES_FILE_MASK=output_xl/ES.$DATASET.pickle # this file will contain the index of each sentence from the sentance pairs in IN_FILE that were added to ES_FILE
MT_FILE=datasets+scoring_script/xling_test_mt/STS.input.$DATASET.MT.txt # the result of translating ES_FILE
MERGE_FILE=output_xl/STS.input.MT.$DATASET.txt # the output of merging MT_FILE with the english sentences from IN_FILE
OUT_CORRELATION_FILE=output_xl/out.$DATASET.txt # the output file for the predictions
GS_CORRELATION_FILE=datasets+scoring_script/xling_test_gs/STS.gs.$DATASET.txt # the gold standard file

#python xl_extract_ES_sentences.py $IN_FILE $ES_FILE


#python xl_lang_detect.py $IN_FILE $ES_FILE $ES_FILE_MASK
#python xl_merge_MT_sentences.py $IN_FILE $MT_FILE $ES_FILE_MASK $MERGE_FILE
python predict_baseline.py $MERGE_FILE $OUT_CORRELATION_FILE
python get_correlation.py $OUT_CORRELATION_FILE $GS_CORRELATION_FILE
