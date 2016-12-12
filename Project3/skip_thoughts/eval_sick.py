'''
Evaluation code for the SICK dataset (SemEval 2014 Task 1)
'''
import numpy as np
import skipthoughts
import copy
from sklearn.metrics import mean_squared_error as mse
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.utils import shuffle

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam


import os
import re
import sys
import nltk
from scipy import spatial



if not os.path.exists('../data/np'):
    os.mkdir('../data/np')

def tokenize_sentence(sent):
    tokens = nltk.word_tokenize(sent)
    tokens = map(lambda e: e.strip().lower(), tokens)
    return tokens

# compute the cosine similarity between two embeddings as 1 - cosine_distance, scale it from 0-1 to 0-5
def get_cosine_similarity(e1,e2):
    cos_sim = 1.0 - spatial.distance.cosine(e1,e2)
    sim = ( (cos_sim - 0.0) / (1.0 - 0.0) ) * (5.0 - 0.0) + 0.0
    return sim

def get_cosine_feature(X,Y):
    feat = []
    for idx in range(len(X)):
        feat.append(get_cosine_similarity(X[idx],Y[idx]))
    return feat


def get_vocabulary():
    rootdir = 'datasets+scoring_script'
    all_tokens = []
    for folder, subs, files in os.walk(rootdir):
        for filename in files:
            if 'input' not in filename:
                continue
            #print filename
            with open(os.path.join(folder, filename), 'r') as src:
                tokens = nltk.word_tokenize(src.read().decode('utf8','ignore'))
                tokens = map(lambda e: e.strip().lower(), tokens)
                all_tokens += tokens
    print len(all_tokens),len(set(all_tokens))
    #print set(all_tokens)
    return list(set(all_tokens))


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


def self_encode(model,Xname,X,verbose=True,use_eos=False):
    xname_path = '../data/np/%s.npy' % (Xname)

    model_skipthoughts = model.get('skipthoughts',None)
    model_baseline = model.get('baseline',None)

    if os.path.exists(xname_path):
        E = np.load(xname_path)
    else:
        E = np.ones((len(X),1))
        if model_skipthoughts is not None:
            Est = skipthoughts.encode(model_skipthoughts, X, verbose, use_eos)
            E = np.concatenate((E,Est),axis=1)
        if model_baseline is not None:
            sentsEmb = []
            for i,s1 in enumerate(X):
                s1t = tokenize_sentence(s1)
                s1te = get_sentence_embeddings(s1t,model_baseline)
                sentsEmb.append(s1te)
            Ebl = np.array(sentsEmb)
            print Ebl.shape,E.shape
            E = np.concatenate((E,Ebl),axis=1)
        np.save(xname_path,E)
    return E


def get_features(tA,tB):
    cossim = get_cosine_feature(tA[:,4801:],tB[:,4801:])
    feats = np.c_[np.abs(tA[:,4801:] - tB[:,4801:]), np.abs(tA[:,:4801] - tB[:,:4801]), get_cosine_feature(tA[:,4801:],tB[:,4801:]), get_cosine_feature(tA[:,:4801],tB[:,:4801])]
    print '!!!!!',feats.shape
    return feats

def evaluate(model, seed=1234, evaltest=False,test_dataset=None):
    """
    Run experiment
    """
    print 'Preparing data...'
    train, dev, test, scores,test_dataset_idx = load_data15(test_dataset=test_dataset)
    test_dataset_idx = (0,len(test[0]))
    # TODO what is this???? shuffling only the list for the first sentence in the pair???
    #train[0], train[1], scores[0] = shuffle(train[0], train[1], scores[0], random_state=seed)

    #print len(train[0]),type(train[0]),train[0][1]
    #print len(train[1])
    #exit()

    print 'Computing training skipthoughts...'
    trainA = self_encode(model, 'trainA',train[0], verbose=False, use_eos=True)
    trainB = self_encode(model, 'trainB',train[1], verbose=False, use_eos=True)

    print 'Computing development skipthoughts...'
    devA = self_encode(model, 'devA',dev[0], verbose=False, use_eos=True)
    devB = self_encode(model, 'devB',dev[1], verbose=False, use_eos=True)

    print 'Computing feature combinations...'
    trainF = get_features(trainA,trainB)#np.c_[np.abs(trainA - trainB), trainA * trainB, get_cosine_feature(trainA,trainB)]
    devF = get_features(devA,devB)#np.c_[np.abs(devA - devB), devA * devB, get_cosine_feature(devA,devB)]

    print 'Encoding labels...'
    trainY = encode_labels(scores[0])
    devY = encode_labels(scores[1])

    print 'Compiling model...'
    lrmodel = prepare_model(ninputs=trainF.shape[1])

    print 'Training...'
    bestlrmodel = train_model(lrmodel, trainF, trainY, devF, devY, scores[1])

    if evaltest:
        print 'Computing test skipthoughts...'
        testA = self_encode(model, 'testA',test[0], verbose=False, use_eos=True)
        testB = self_encode(model, 'testB',test[1], verbose=False, use_eos=True)
        #print testA.shape

        testA = testA[test_dataset_idx[0]:test_dataset_idx[0]+test_dataset_idx[1]]
        testB = testB[test_dataset_idx[0]:test_dataset_idx[0]+test_dataset_idx[1]]
        scores[2] = scores[2][test_dataset_idx[0]:test_dataset_idx[0]+test_dataset_idx[1]]

        print 'Computing feature combinations...'
        #testF = np.c_[np.abs(testA - testB), testA * testB, get_cosine_feature(testA,testB)]
        testF = get_features(testA,testB)

        print 'Evaluating...'
        r = np.arange(1,6)
        yhat = np.dot(bestlrmodel.predict_proba(testF, verbose=2), r)

        print '????SHAPE:',testA.shape
        #cossim = get_cosine_feature(testA[:,4800:],testB[:,4800:])
        #cossim = get_cosine_feature(testA[:,2400:4800],testB[:,2400:4800])
        #print len(cossim),len(list(yhat))
        #yhat = np.array(cossim)#np.mean([cossim,list(yhat)],axis=0)
        print yhat.shape

        pr = pearsonr(yhat, scores[2])[0]
        sr = spearmanr(yhat, scores[2])[0]
        se = mse(yhat, scores[2])
        print 'Test Pearson: ' + str(pr)
        print 'Test Spearman: ' + str(sr)
        print 'Test MSE: ' + str(se)

        return yhat




########################


def prepare_model(ninputs=9600, nclass=5):
    """
    Set up and compile the model architecture (Logistic regression)
    """
    lrmodel = Sequential()
    lrmodel.add(Dense(nclass,input_dim=ninputs))
    lrmodel.add(Activation('softmax'))
    lrmodel.compile(loss='categorical_crossentropy', optimizer='adam')
    return lrmodel


def train_model(lrmodel, X, Y, devX, devY, devscores):
    """
    Train model, using pearsonr on dev for early stopping
    """
    done = False
    best = -1.0
    r = np.arange(1,6)

    while not done:
        # Every 100 epochs, check Pearson on development set
        lrmodel.fit(X, Y, verbose=2, shuffle=False, validation_data=(devX, devY))
        yhat = np.dot(lrmodel.predict_proba(devX, verbose=2), r)
        score = pearsonr(yhat, devscores)[0]
        if score > best:
            print score
            best = score
            bestlrmodel = copy.deepcopy(lrmodel)
        else:
            done = True

    yhat = np.dot(bestlrmodel.predict_proba(devX, verbose=2), r)
    score = pearsonr(yhat, devscores)[0]
    print 'Dev Pearson: ' + str(score)
    return bestlrmodel


def encode_labels(labels, nclass=5):
    """
    Label encoding from Tree LSTM paper (Tai, Socher, Manning)
    """
    Y = np.zeros((len(labels), nclass)).astype('float32')
    for j, y in enumerate(labels):
        for i in range(nclass):
            if i+1 == np.floor(y) + 1:
                Y[j,i] = y - np.floor(y)
            if i+1 == np.floor(y):
                Y[j,i] = np.floor(y) - y + 1
    return Y


def read_folder(full_folder,track_file=None):
    trainA, trainB, trainS = [], [], []
    track_file_idxes = (0,0)
    for fl in os.listdir(full_folder):
        if 'input' not in fl:
            continue
        flg = re.sub('input','gs',fl)
        filei = os.path.join(full_folder,fl)
        fileg = os.path.join(full_folder,flg)
        sents = map(lambda e: e.strip(),open(filei).read().decode('utf8','ignore').strip().split('\n'))
        scores = map(lambda e: e.strip(),open(fileg).read().decode('utf8','ignore').strip().split('\n'))
        assert len(sents) == len(scores)
        if track_file == filei:
            track_file_idxes = (len(trainA),len(sents))
        for i in range(len(sents)):
            s1,s2 = sents[i].split('\t')
            sc = scores[i]
            trainA.append(s1)
            trainB.append(s2)
            trainS.append(sc)
    return trainA,trainB,trainS,track_file_idxes


def read_files(test_dataset):
    trainA, trainB, devA, devB, testA, testB = [],[],[],[],[],[]
    trainS, devS, testS = [],[],[]

    for folder in os.listdir('datasets+scoring_script/train'):
        full_folder = os.path.join('datasets+scoring_script/train',folder)

        trainAt, trainBt, trainSt,_ = read_folder(full_folder)

        if 'train' in folder:
            devA+= trainAt
            devB+= trainBt
            devS+= trainSt
        else:
            trainA+= trainAt
            trainB+= trainBt
            trainS+= trainSt

    full_folder_test = 'datasets+scoring_script/test'
    testA, testB, testS,test_dataset_idx = read_folder(full_folder_test,test_dataset)
    print '>>>',test_dataset_idx

    print len(trainA),len(devA),len(testA)
    return trainA, trainB, devA, devB, testA, testB, trainS, devS, testS,test_dataset_idx


def load_data15(loc='../data/sick/',test_dataset=None):
    trainA, trainB, devA, devB, testA, testB, trainS, devS, testS, test_dataset_idx = read_files(test_dataset)

    trainS = [float(s) for s in trainS]
    devS = [float(s) for s in devS]
    testS = [float(s) for s in testS]

    return [trainA, trainB], [devA, devB], [testA, testB], [trainS, devS, testS],test_dataset_idx


    """
    Load the SICK semantic-relatedness dataset
    """
    trainA, trainB, devA, devB, testA, testB = [],[],[],[],[],[]
    trainS, devS, testS = [],[],[]

    with open(loc + 'SICK_train.txt', 'rb') as f:
        for line in f:
            text = line.strip().split('\t')
            trainA.append(text[1])
            trainB.append(text[2])
            trainS.append(text[3])
    with open(loc + 'SICK_trial.txt', 'rb') as f:
        for line in f:
            text = line.strip().split('\t')
            devA.append(text[1])
            devB.append(text[2])
            devS.append(text[3])
    with open(loc + 'SICK_test_annotated.txt', 'rb') as f:
        for line in f:
            text = line.strip().split('\t')
            testA.append(text[1])
            testB.append(text[2])
            testS.append(text[3])

    trainS = [float(s) for s in trainS[1:]]
    devS = [float(s) for s in devS[1:]]
    testS = [float(s) for s in testS[1:]]

    return [trainA[1:], trainB[1:]], [devA[1:], devB[1:]], [testA[1:], testB[1:]], [trainS, devS, testS]
