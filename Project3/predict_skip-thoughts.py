import sys
import os
import re
import nltk
import json
from scipy import spatial
from skip_thoughts import skipthoughts
from skip_thoughts import eval_sick

DATA_DIR = '../data/'

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
def read_word_embeddings(vocab=None):

    if vocab is not None:
        vocab_path = '../data/vocab.json'
        if os.path.exists(vocab_path):
            return json.loads(open(vocab_path).read())

    embeddings = {}
    model_file = os.path.join(DATA_DIR,'paragram_300_sl999.txt')
    new_embeddings = read_embedding_file(model_file,embeddings)
    #new_embeddings = {}
    print len(new_embeddings)

    model_file = os.path.join(DATA_DIR,'paragram-phrase-XXL.txt')
    all_embeddings = read_embedding_file(model_file,new_embeddings)
    print len(all_embeddings)

    if vocab is not None:
        vocab_path = '../data/vocab.json'

        vocab_dict = {}
        for k in vocab:
            emb = all_embeddings.get(k,None)
            if emb is not None:
                vocab_dict[k] = emb
        jv = json.dumps(vocab_dict)
        fo = open(vocab_path,'w')
        fo.write(jv)
        fo.close()
        print len(vocab),len(vocab_dict)
        return vocab_dict
    return all_embeddings

def read_word_embeddings_st(vocab=None):
    vocab_path = '../data/vocab_st.json'
    if vocab is not None:
        if os.path.exists(vocab_path):
            return json.loads(open(vocab_path).read())

    model_sk = skipthoughts.load_model()

    if vocab is not None:

        vocab_dict = {}
        for k in vocab:
            emb = all_embeddings.get(k,None)
            if emb is not None:
                vocab_dict[k] = emb
        jv = json.dumps(vocab_dict)
        fo = open(vocab_path,'w')
        fo.write(jv)
        fo.close()
        print len(vocab),len(vocab_dict)
        return vocab_dict
    return all_embeddings




vocab = []#eval_sick.get_vocabulary()

print 'a'
model_sk = []#skipthoughts.load_model()
model_bl = []#read_word_embeddings(vocab)
print 'b'

model = {}
model['skipthoughts'] = model_sk
model['baseline'] = model_bl

test_dataset = sys.argv[1]
print 'test:',test_dataset
preds = eval_sick.evaluate(model, evaltest=True,test_dataset=test_dataset)
fn_out = sys.argv[2]
f_out = open(fn_out,'w')
for p in preds:
    f_out.write('%lf\n' % p)
