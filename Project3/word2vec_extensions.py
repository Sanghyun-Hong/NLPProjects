
# Basic 
import os
import sys
import nltk

import numpy as np
import scipy as sp

# word2vec
from gensim.models import Word2Vec


"""
    Pre-defined variables
"""
MODEL_LOC  = 'models'
# word2vec
MNAME_W2V  = 'word2vec'
MODEL_W2V  = 'GoogleNews-vectors-negative300.bin'
# doc2vec
MNAME_D2V  = 'doc2vec'
MODEL_D2V1 = 'apnews_dbow/doc2vec.bin'
MODEL_D2V2 = 'enwiki_dbow/doc2vec.bin'

EMBED_TYPE = 'sum'  # choose: sum, prod, so on



"""
    Utility Functions
"""
def read_model(modelname):
    # model to store
    cur_model = None
    # word2vec model: Google news (300 dimensions)
    if 'word2vec' in modelname.lower():
        cur_model = Word2Vec.load_word2vec_format(MODEL_LOC + '/' + MODEL_W2V, binary=True)
    else:
        assert False, 'Invalid model name: %s' % (modelname)
    return cur_model

def tokenize_sentence(sentence):
    tokens = nltk.word_tokenize(sentence)
    tokens = map(lambda e: e.strip().lower(), tokens)
    return tokens

def obtain_summation(sentence_token, model):
    # prepare zero array (size = embedding dimension: Google use 300)
    sum_emb = np.zeros(300)
    for token in sentence_token:
        if token in model.vocab:
            token_emb = model[token]
            sum_emb  += token_emb
    return sum_emb

def obtain_product(sentence_token, model):
    # prepare zero array (size = embedding dimension: Google use 300)
    prod_emb = np.zeros(300)
    for token in sentence_token:
        if token in model.vocab:
            token_emb = model[token]
            # FIXME - I have no idea how to deal with negative small numbers. the log will be nan
            prod_emb += np.log(token_emb)
    return prod_emb

def compute_cosine_similarity(embedding1, embedding2):
    # compute cosine similarity btw. two embeddings and normalize
    cos_sim = 1.0 - sp.spatial.distance.cosine(embedding1, embedding2)
    nor_sim = ( (cos_sim - 0.0) / (1.0 - 0.0) ) * (5.0 - 0.0) + 0.0
    return nor_sim


"""
    Main (test)

     - test command: python word2vec_extensions.py datasets+scoring_script/test/STS2016.input.headlines.txt w2v.headlines.txt
     - corr command: python get_correlation.py datasets+scoring_script/test/STS2016.input.headlines.txt w2v.headlines.txt
"""
if __name__ == '__main__':

    # read stored models
    word2vec = read_model(MNAME_W2V)
    print ' .. read the %s model complete ' % (word2vec)

    # do predictions
    if len(sys.argv) != 3:
        print 'Usage: script <input_file> <output_file>'
        exit(1)

    filename_in  = sys.argv[1]
    filename_out = sys.argv[2]
    if not os.path.exists(filename_in):
        print 'Input path does not exist.'
        exit(1)
    print ' .. compute the similarity btw. [%s, %s] ' % (filename_in, filename_out)

    file_in  = open(filename_in, 'rb')
    file_out = open(filename_out,'wb')
    for each_line in file_in:
        sentence1, sentence2 = each_line.split('\t')
        sen1_token = tokenize_sentence(sentence1)
        sen2_token = tokenize_sentence(sentence2)

        if 'sum' in EMBED_TYPE:
            # use various embeddings (summation of word2vec)
            sen1_embed = obtain_summation(sen1_token, word2vec)
            sen2_embed = obtain_summation(sen2_token, word2vec)
            print '  . use the summation of embeddings '
        elif 'prod' in EMBED_TYPE:
            # use various embeddings (product of word2vec)
            sen1_embed = obtain_product(sen1_token, word2vec)
            sen2_embed = obtain_product(sen2_token, word2vec)
            print '  . use the product of embeddings '
        else:
            assert False, 'Invalid usage type of embeddings'

        cos_sim = compute_cosine_similarity(sen1_embed, sen2_embed)
        file_out.write('%0.3lf\n' % cos_sim)

    file_out.close()
    print ' .. similarity computation done '

