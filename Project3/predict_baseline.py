import sys
import os
import re
import nltk
from scipy import spatial

DATA_DIR = '../data/'


def read_embedding_file(model_file,existing_embeddings):
    embeddings = existing_embeddings

    if not os.path.exists(model_file):
        print 'Make sure you download the embeddings file under:',model_file
        exit(1)

    for l in open(model_file,'r'):
        tokens = l.strip().split(' ')
        word = tokens[0].strip().lower()
        weights = map(lambda e: float(e),tokens[1:])
        assert len(weights) == 300
        embeddings[word] = weights
    return embeddings

# read the Paragram embeddings from file
# returns a dictionary {word:embeddings} where each embeddings is a 300x1 vector
def read_word_embeddings():
    embeddings = {}
    model_file = os.path.join(DATA_DIR,'paragram_300_sl999.txt')
    new_embeddings = read_embedding_file(model_file,embeddings)

    print len(embeddings)

    model_file = os.path.join(DATA_DIR,'paragram-phrase-XXL.txt')
    all_embeddings = read_embedding_file(model_file,new_embeddings)

    print len(embeddings)

    return all_embeddings

# use the nltk word tokenizer to tokenize a sentence and return the lower case token list
def tokenize_sentence(sent):
    tokens = nltk.word_tokenize(sent)
    tokens = map(lambda e: e.strip().lower(), tokens)
    return tokens

# return the sentence embedding that is the average of all token embeddings present in the embeddings dictionary
# tokens not found in the embeddings dictionary are ignored
def get_sentence_embeddings(tokenized_sent,embeddings_dict):
    countw = 0
    sumw = [0.0]*len(embeddings_dict[embeddings_dict.keys()[0]])
    for t in tokenized_sent:
        token_embeddings = embeddings_dict.get(t,None)
        if token_embeddings is not None:
            countw += 1
            for i in range(len(token_embeddings)):
                sumw[i] += token_embeddings[i]
        #else:
        #    print '@@',t

    for i in range(len(sumw)):
        sumw[i] /= countw
    return sumw

# compute the cosine similarity between two embeddings as 1 - cosine_distance, scale it from 0-1 to 0-5
def get_cosine_similarity(e1,e2):
    cos_sim = 1.0 - spatial.distance.cosine(e1,e2)
    sim = ( (cos_sim - 0.0) / (1.0 - 0.0) ) * (5.0 - 0.0) + 0.0
    return sim

def get_predictions():

    if len(sys.argv) != 3:
        print 'Usage: script <input_file> <output_file>'
        exit(1)

    fn_in = sys.argv[1]
    fn_out = sys.argv[2]
    if not os.path.exists(fn_in):
        print 'Input path does not exist.'
        exit(1)

    embeddings = read_word_embeddings()

    f_in = open(fn_in,'r')
    f_out = open(fn_out,'w')
    for l in f_in:
        s1,s2 = l.split('\t')
        s1t = tokenize_sentence(s1)
        s2t = tokenize_sentence(s2)

        s1te = get_sentence_embeddings(s1t,embeddings)
        s2te = get_sentence_embeddings(s2t,embeddings)

        sim = get_cosine_similarity(s1te,s2te)
        f_out.write('%0.3lf\n' % sim)


    f_out.close()

get_predictions()
