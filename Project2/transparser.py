
# basic
import os
import sys
from collections import deque

# advanced
import numpy as np
import networkx as nx

# debug
import matplotlib.pyplot as plt


# the first thing we do is define a Weights class that will store the
# learned weights of the parser. this is just a dictionary that maps
# strings (features) to floats (parameter values).
class Weights(dict):
    # default all unknown feature values to zero
    def __getitem__(self, idx):
        if self.has_key(idx):
            return dict.__getitem__(self, idx)
        else:
            return 0.

    # given a feature vector, compute a dot product
    def dotProduct(self, t, x):
        dot = 0.
        for feat,val in x.iteritems():
            dot += val * self[t, feat]
        return dot

    # given an example _and_ a true label (y is +1 or -1), update the
    # weights according to the perceptron update rule (we assume
    # you've already checked that the classification is incorrect
    def update(self, t, x, y, cnt=1):
        for feat,val in x.iteritems():
            if val != 0.:
                self[t, feat] += y * val * cnt

    # [AVG] perceptron: compute an average of weights
    def average(self, x, cnt):
        avg = {}
        for feat,val in self.iteritems():
            cnt_val   = x[feat]
            avg[feat] = (val - ((1./cnt) * cnt_val))
        return Weights(avg)


# compute number of mistakes
def numMistakes(graph):
    err = 0.
    for nid in graph.nodes():
        true_head = graph.node[nid]['head']
        pred_head = graph.node[nid]['phead']
        if true_head == pred_head: continue   # skip
        err += 1
    return err

# now we can finally put it all together to make a single update on a
# single example
def runOneExample(start_cnt, bias, weights, tmp_bias, tmp_weis, trueGraph, quiet=True):
    
    # [AVG] perceptron
    cnt = 0

    # initialize the configurations
    depStack  = deque()
    depBuffer = deque()
    for nid in trueGraph.nodes():
        depBuffer.append(nid)
    depStack.append(depBuffer.popleft())    # move the *root*

    # ARC-standard transitions (move on, update weights)
    while depBuffer and depStack:
        # [AVG] perceptron
        cnt += 1

        # two nodes, one for each stack/buffer
        stid = depStack[-1]
        qtid = depBuffer[0]
        s0 = trueGraph.node[stid]
        q0 = trueGraph.node[qtid]

        # [DEBUG]
        if not quiet:
            print ' .. Configuration (before): %s | %s' % (depStack, depBuffer)

        # extract features: suggested ones
        feats = { 'w_stop=' + s0['word']: 1.,
                  'w_btop=' + q0['word']: 1.,
                  'cp_stop='+ s0['cpos']: 1.,
                  'cp_btop='+ q0['cpos']: 1.,
                  'w_pair=' + s0['word'] + '_' + q0['word']: 1.,
                  'cp_pair='+ s0['cpos'] + '_' + q0['cpos']: 1. }

        #### [PREDICTED TRANSISION]
        rweight = weights.dotProduct('r', feats) + bias['r']
        lweight = weights.dotProduct('l', feats) + bias['l']
        sweight = weights.dotProduct('s', feats) + bias['s']
        # select the biggest weight
        transps = [rweight, lweight, sweight]
        transidx= np.argmax(transps)
        # predict transition w.r.t. the weight
        pred_trans = -1
        if transidx == 0:   pred_trans = 'r'
        elif transidx == 1: pred_trans = 'l'
        else:               pred_trans = 's'
        ###########################

        #### [ORACLE TRANSISION]
        true_trans = -1
        if trueGraph.node[qtid]['head'] == str(stid):
            # check if it's possible to remove qtid
            possible = True
            for item in depBuffer:
                if trueGraph.node[item]['head'] == str(qtid):
                    possible = False
            for item in depStack:
                if trueGraph.node[item]['head'] == str(qtid):
                    possible = False
            # if possible remove qtid w. right
            if possible: true_trans = 'r'
            else:        true_trans = 's'
        elif trueGraph.node[stid]['head'] == str(qtid):
            # check if it's possible to remove stid
            possible = True
            if stid == 0:
                possible = False
            for item in depBuffer:
                if trueGraph.node[item]['head'] == str(stid):
                    possible = False
            for item in depStack:
                if trueGraph.node[item]['head'] == str(stid):
                    possible = False
            # if possible remove stid w. left
            if possible: true_trans = 'l'
            else:        true_trans = 's'
        else:
            true_trans = 's'
        ########################

        #### [UPDATES]
        if pred_trans != true_trans:
            # update weights ('r', 'l', 's') w.r.t true_trans
            bias[true_trans] += 1
            tmp_bias[true_trans] += 1 * (start_cnt+cnt)
            weights.update( true_trans, feats, 1)
            tmp_weis.update(true_trans, feats, 1, cnt=(start_cnt+cnt))
            # update weights ('r', 'l', 's') w.r.t pred_trans
            bias[pred_trans] += -1
            tmp_bias[pred_trans] += -1 * (start_cnt+cnt)
            weights.update( pred_trans, feats, -1)
            tmp_weis.update(pred_trans, feats, -1, cnt=(start_cnt+cnt))
        ##############

        #### [TRANSITION]
        if true_trans == 'r':
            trueGraph.node[qtid]['phead'] = str(stid)
            depBuffer.popleft()
            depBuffer.appendleft(depStack.pop())
        elif true_trans == 'l':
            trueGraph.node[stid]['phead'] = str(qtid)
            depStack.pop()
        else:
            depStack.append(depBuffer.popleft())
        #################

        # [DEBUG]
        if not quiet:
            print ' .. Configuration (after) : %s | %s' % (depStack, depBuffer)
            print ' .. Gold transition: [%s] btn [%s, %s]' % (transidx, stid, qtid)
    #### end of while

    # compute the error
    err = numMistakes(G)

    # print the predicted tree and error
    if not quiet:
        print 'error =', err, '\tpred =',
        for i,j in G.edges_iter():
            print '(', trueGraph.node[i]['word'], '<->', trueGraph.node[j]['word'], ':', G[i][j]['phead'], ')',
        print ''

    return (cnt, err)


def iterCoNLL(filename):
    h = open(filename, 'r')
    G = None
    nn = 0
    for l in h:
        l = l.strip()
        if l == "":
            if G != None:
                yield G
            G = None
        else:
            if G == None:
                nn = nn + 1
                G = nx.Graph()
                G.add_node(0, {'word': '*root*', 'lemma': '*root*', 'cpos': '*root*',\
                               'pos': '*root*',  'feats': '*root*', 'head': '*root*',\
                               'drel': '*root*', 'phead': '*root*', 'pdrel':'*root' })
                newGraph = False
            [id, word, lemma, cpos, pos, feats, head, drel, phead, pdrel] = l.split('\t')
            G.add_node(int(id), {'word' : word, 'lemma': lemma, 'cpos' : cpos,
                                 'pos'  : pos,  'feats': feats, 'head' : head, 
                                 'drel' : drel, 'phead': phead, 'pdrel': pdrel })
            
            if not head == '_':
                G.add_edge(int(head), int(id), {}) # 'true_rel': drel, 'true_par': int(id)})

    if G != None:
        yield G
    h.close()


"""
    Utility functions (prediction, print, etc.)
"""
def predictOneExampleHeads(bias, weights, testGraph, quiet=True):
    # initialize the configurations
    depStack  = deque()
    depBuffer = deque()
    for nid in testGraph.nodes():
        depBuffer.append(nid)
    depStack.append(depBuffer.popleft())    # move the *root*

    # predict the heads by following the predictions (ARC-Standard)
    while depBuffer and depStack:
        stid = depStack[-1]
        qtid = depBuffer[0]
        s0 = testGraph.node[stid]
        q0 = testGraph.node[qtid]

        # [DEBUG]
        if not quiet:
            print ' .. Configuration (before): %s | %s' % (depStack, depBuffer)

        # extract features: suggested ones
        feats = { 'w_stop=' + s0['word']: 1.,
                  'w_btop=' + q0['word']: 1.,
                  'cp_stop='+ s0['cpos']: 1.,
                  'cp_btop='+ q0['cpos']: 1.,
                  'w_pair=' + s0['word'] + '_' + q0['word']: 1.,
                  'cp_pair='+ s0['cpos'] + '_' + q0['cpos']: 1. }

        # predict the current transition
        rweight = weights.dotProduct('r', feats) + bias['r']
        lweight = weights.dotProduct('l', feats) + bias['l']
        sweight = weights.dotProduct('s', feats) + bias['s']
        # select the biggest weight
        transps = [rweight, lweight, sweight]
        transidx= np.argmax(transps)
        # convert to str
        if transidx == 0:   transition = 'r'
        elif transidx == 1: transition = 'l'
        else:               transition = 's'

        # assign heads w.r.t. the transitions
        if transition == 'r':   # right
            testGraph.node[qtid]['head'] = str(stid)
            depBuffer.popleft()
            depBuffer.appendleft(depStack.pop())
        elif transition == 'l': # left
            testGraph.node[stid]['head'] = str(qtid)
            depStack.pop()
        else:               # shift
            depStack.append(depBuffer.popleft())

        # [DEBUG]
        if not quiet:
            print ' .. Configuration (after) : %s | %s' % (depStack, depBuffer)
            print ' .. Pred transition: [%s] btn [%s, %s]' % (transidx, stid, qtid)


def printPrediction(filename, Graph):
    outfile = open(filename, 'a')
    for nid in Graph.nodes():
        curnode = Graph.node[nid]
        # skip the root (don't need to print)
        if curnode['word'] == '*root*':
            continue
        oneline = [str(nid), curnode['word'], curnode['lemma'], curnode['cpos'],\
                             curnode['pos'],  curnode['feats'], curnode['head'],\
                             curnode['drel'], curnode['phead'], curnode['pdrel']]
        outfile.write('\t'.join(oneline))
        outfile.write('\n')
    outfile.write('\n')



"""
    Main (to evalate)
"""
if __name__ == "__main__":
    # developing case: cmd, train, test, out = ['transparser.py', en.tr100', 'en.dev', 'en.dev.out' ]
    # evaluation case: cmd, train, test, out = ['transparser.py', en.tr100', 'en.tst', 'en.tst.out']
    if len(sys.argv) != 4:
        print 'Error: check the arguments (current: %s)' % (len(sys.argv))
        exit(1)
    cmd, train, test, out = sys.argv

    # control hyper-parameters
    num_epochs = 5

    # weight variables: averaged perceptron
    bias    = Weights()
    weights = Weights()
    tmp_bias= Weights()
    tmp_weis= Weights()

    # from Marine's 2nd present in graphparser.py
    total_cnt = 1   # [AVG] perceptron
    for iteration in range(num_epochs):
        total_err = 0.
        for G in iterCoNLL(train): 
            (cur_cnt, cur_err) = runOneExample(total_cnt, bias, weights, tmp_bias, tmp_weis, G)
            total_err += cur_err
            total_cnt += cur_cnt    # [AVG] perceptron
        print (total_cnt, total_err)

    # [AVG] perceptron: average weights
    avg_bias = bias.average(   tmp_bias, total_cnt)
    avg_weis = weights.average(tmp_weis, total_cnt)

    # remove the output file for a new output
    if os.path.isfile(out):
        os.remove(out)

    # iterate over the test cases and create outputs
    for G in iterCoNLL(test):
        predictOneExampleHeads(avg_bias, avg_weis, G)
        printPrediction(out, G)

