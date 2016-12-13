#!/bin/bash

# create the data folder
mkdir data
echo " .. creates the data directories"

# Download Paragram-Phrase XXL
wget http://ttic.uchicago.edu/~wieting/paragram-phrase-XXL.zip > data/paragram-phrase-XXL.zip
unzip data/paragram-phrase-XXL.zip -d data/
echo " .. download the paragram XXL "

# Download Additional datas
wget http://www.snehesh.com/shared/cmsc723/paragram_300_sl999.zip > data/paragram_300_sl999.zip
unzip data/paragram_300_sl999.zip
mv data/paragram_300_sl999/paragram_300_sl999.txt data/paragram_300_sl999.txt
echo " .. download the paragram SL999 (1.7GB)"

# Download the vocab datas
wget http://www.snehesh.com/shared/cmsc723/skipthought_vocab.json > data/skipthought_vocab.json
echo " .. download the skip-thought vocabs"

# Download the pre-compiled skip_thoughts
wget http://www.snehesh.com/shared/cmsc723/skip_thoughts.zip > skip_thoughts.zip
unzip skip_thoughts.zip
echo " .. download the pre-compiled skip_thought module"

# Download the pre-trained data for skip_thoughts
wget http://www.snehesh.com/shared/cmsc723/st_trained.zip > data/st_trained.zip
unzip data/st_trained.zip
echo " .. download the pre-trained skip_thoughts data"

# Download the provided data
wget http://www.snehesh.com/shared/cmsc723/datasets+scoring_script-1.zip
unzip datasets+scoring_script-1.zip
echo " .. download the provided data"
