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

import os,re

def evaluate(model, seed=1234, evaltest=False):
    """
    Run experiment
    """
    print 'Preparing data...'
    train, dev, test, scores = load_data15()
    train[0], train[1], scores[0] = shuffle(train[0], train[1], scores[0], random_state=seed)

    #print len(train[0]),type(train[0]),train[0][1]
    #print len(train[1])
    #exit()

    print 'Computing training skipthoughts...'
    trainA = skipthoughts.encode(model, train[0], verbose=False, use_eos=True)

    #print trainA[0]
    #print len(trainA),type(trainA),type(trainA[0])
    #exit()
    trainB = skipthoughts.encode(model, train[1], verbose=False, use_eos=True)

    print 'Computing development skipthoughts...'
    devA = skipthoughts.encode(model, dev[0], verbose=False, use_eos=True)
    devB = skipthoughts.encode(model, dev[1], verbose=False, use_eos=True)

    print 'Computing feature combinations...'
    trainF = np.c_[np.abs(trainA - trainB), trainA * trainB]
    devF = np.c_[np.abs(devA - devB), devA * devB]

    print 'Encoding labels...'
    trainY = encode_labels(scores[0])
    devY = encode_labels(scores[1])

    print 'Compiling model...'
    lrmodel = prepare_model(ninputs=trainF.shape[1])

    print 'Training...'
    bestlrmodel = train_model(lrmodel, trainF, trainY, devF, devY, scores[1])

    if evaltest:
        print 'Computing test skipthoughts...'
        testA = skipthoughts.encode(model, test[0], verbose=False, use_eos=True)
        testB = skipthoughts.encode(model, test[1], verbose=False, use_eos=True)

        print 'Computing feature combinations...'
        testF = np.c_[np.abs(testA - testB), testA * testB]

        print 'Evaluating...'
        r = np.arange(1,6)
        yhat = np.dot(bestlrmodel.predict_proba(testF, verbose=2), r)
        pr = pearsonr(yhat, scores[2])[0]
        sr = spearmanr(yhat, scores[2])[0]
        se = mse(yhat, scores[2])
        print 'Test Pearson: ' + str(pr)
        print 'Test Spearman: ' + str(sr)
        print 'Test MSE: ' + str(se)

        return yhat


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


def read_folder(full_folder):
    trainA, trainB, trainS = [], [], []
    for fl in os.listdir(full_folder):
        if 'input' not in fl:
            continue
        flg = re.sub('input','gs',fl)
        filei = os.path.join(full_folder,fl)
        fileg = os.path.join(full_folder,flg)
        sents = map(lambda e: e.strip(),open(filei).read().decode('utf8','ignore').strip().split('\n'))
        scores = map(lambda e: e.strip(),open(fileg).read().decode('utf8','ignore').strip().split('\n'))
        assert len(sents) == len(scores)
        for i in range(len(sents)):
            s1,s2 = sents[i].split('\t')
            sc = scores[i]
            trainA.append(s1)
            trainB.append(s2)
            trainS.append(sc)
    return trainA,trainB,trainS


def read_files():
    trainA, trainB, devA, devB, testA, testB = [],[],[],[],[],[]
    trainS, devS, testS = [],[],[]

    for folder in os.listdir('datasets+scoring_script/train'):
        full_folder = os.path.join('datasets+scoring_script/train',folder)

        trainAt, trainBt, trainSt = read_folder(full_folder)

        if 'train' in folder:
            devA+= trainAt
            devB+= trainBt
            devS+= trainSt
        else:
            trainA+= trainAt
            trainB+= trainBt
            trainS+= trainSt

    full_folder_test = 'datasets+scoring_script/test'
    testA, testB, testS = read_folder(full_folder_test)

    print len(trainA),len(devA),len(testA)
    return trainA, trainB, devA, devB, testA, testB, trainS, devS, testS


def load_data15(loc='../data/sick/'):
    trainA, trainB, devA, devB, testA, testB, trainS, devS, testS = read_files()

    trainS = [float(s) for s in trainS]
    devS = [float(s) for s in devS]
    testS = [float(s) for s in testS]

    return [trainA, trainB], [devA, devB], [testA, testB], [trainS, devS, testS]


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
