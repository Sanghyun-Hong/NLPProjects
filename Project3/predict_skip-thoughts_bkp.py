import sys
import os
import re
import nltk
from scipy import spatial
from skip_thoughts import skipthoughts

DATA_DIR = '../data/'


#
def read_word_embeddings():
    model = skipthoughts.load_model()
    return model

# use the nltk word tokenizer to tokenize a sentence and return the lower case token list
def tokenize_sentence(sent):
    tokens = nltk.word_tokenize(sent)
    tokens = map(lambda e: e.strip().lower(), tokens)
    return tokens

#
def get_sentence_embeddings(sentence,embeddings):
    vectors = skipthoughts.encode(embeddings, [sentence])
    '''
    print vectors
    print sentence
    print len(vectors[0])
    exit()
    '''
    return vectors[0]

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
        #s1t = tokenize_sentence(s1)
        #s2t = tokenize_sentence(s2)

        s1te = get_sentence_embeddings(s1,embeddings)
        s2te = get_sentence_embeddings(s2,embeddings)

        sim = get_cosine_similarity(s1te,s2te)
        f_out.write('%0.3lf\n' % sim)


    f_out.close()

#get_predictions()
#exit()
#from skip_thoughts import eval_sick
model_sk = skipthoughts.load_model()
eval_sick.evaluate(model, evaltest=True)
