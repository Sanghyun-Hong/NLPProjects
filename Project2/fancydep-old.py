
# basic
import os
import sys
from collections import deque

# advanced
import numpy as np
import networkx as nx

# debug
import matplotlib.pyplot as plt

MAGIC_CHAR = '<N>'

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
    def update(self, t, x, y):
        for feat,val in x.iteritems():
            if val != 0.:
                self[t, feat] += y * val


# we compute the oracle transitions and put them to the out(graph)
# those transitions are used to update weights during traning
def computeOracleTransition(trueGraph, quiet=True):
    # create a training Graph to return
    out = nx.Graph()

    # initialize the configurations
    depStack  = deque()
    depBuffer = deque()
    for nid in trueGraph.nodes():
        depBuffer.append(nid)
    depStack.append(depBuffer.popleft())    # move the *root*

    # compute the oracle transitions (ARC-Standard)
    while depBuffer:
        # prepare the variables to compute features
        stid = depStack[-1]
        qtid = depBuffer[0]
        s0 = trueGraph.node[stid]
        s1 = { 'word': MAGIC_CHAR, 'pos': MAGIC_CHAR }
        s2 = { 'word': MAGIC_CHAR, 'pos': MAGIC_CHAR }
        q0 = trueGraph.node[qtid]
        q1 = { 'word': MAGIC_CHAR, 'pos': MAGIC_CHAR }
        q2 = { 'word': MAGIC_CHAR, 'pos': MAGIC_CHAR }
        # case where it has more word in stack/buffer
        if len(depStack) == 2:
            s1id = depStack[-2]
            s1 = trueGraph.node[s1id]
        elif len(depStack) > 2:
            s1id = depStack[-2]
            s2id = depStack[-3]
            s1 = trueGraph.node[s1id]
            s2 = trueGraph.node[s2id]
        if len(depBuffer) == 2:
            q1id = depBuffer[1]
            q1 = trueGraph.node[q1id]
        elif len(depBuffer) > 2:
            q1id = depBuffer[1]
            q2id = depBuffer[2]
            q1 = trueGraph.node[q1id]
            q2 = trueGraph.node[q2id]

        # [DEBUG]
        if not quiet:
            print ' .. Configuration (before): %s | %s' % (depStack, depBuffer)

        # examine the gold transition
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
            if possible:
                transition = 'r'
                depBuffer.popleft()
                if stid != 0: depBuffer.appendleft(depStack.pop())
            else:
                transition = 's'
                depStack.append(depBuffer.popleft())
        elif trueGraph.node[stid]['head'] == str(qtid):
            # check if it's possible to remove stid
            possible = True
            for item in depBuffer:
                if trueGraph.node[item]['head'] == str(stid):
                    possible = False
            for item in depStack:
                if trueGraph.node[item]['head'] == str(stid):
                    possible = False
            # if possible remove stid w. left
            if possible:
                transition = 'l'
                if stid != 0: depStack.pop()
            # else: does shift
            else:
                transition = 's'
                depStack.append(depBuffer.popleft())
        else:
            transition = 's'
            depStack.append(depBuffer.popleft())

        # compute a new features: intermediates
        #  - distance
        distance = str(abs(stid - qtid))

        # compuate a new features: from Zhang and Nivre's paper
        feats = { 
            # baseline features: sigle words
            'w_stop='  + s0['word']: 1.,
            'p_stop='  + s0['pos' ]: 1.,
            'w_pstop=' + s0['word'] + '_' + s0['pos' ]: 1.,
            'w_btop='  + q0['word']: 1.,
            'p_btop='  + q0['pos' ]: 1.,
            'w_pbtop=' + q0['word'] + '_' + q0['pos' ]: 1.,
            'w_b2nd='  + q1['word']: 1.,
            'p_b2nd='  + q1['pos' ]: 1.,
            'w_pb2nd=' + q1['word'] + '_' + q1['pos' ]: 1.,
            'w_b3rd='  + q2['word']: 1.,
            'p_b3rd='  + q2['pos' ]: 1.,
            'w_pb3rd=' + q2['word'] + '_' + q2['pos' ]: 1.,
            # baseline features: from word pairs
            's0wp_q0wp=' + s0['word'] + '_' + s0['pos' ] + '_' + q0['word'] + '_' + q0['pos' ]: 1.,
            's0wp_q0w='  + s0['word'] + '_' + s0['pos' ] + '_' + q0['word']: 1.,
            's0w_q0wp='  + s0['word'] + '_' + q0['word'] + '_' + q0['pos' ]: 1.,
            's0wp_q0p='  + s0['word'] + '_' + s0['pos' ] + '_' + q0['pos' ]: 1.,
            's0p_q0wp='  + s0['pos' ] + '_' + q0['word'] + '_' + q0['pos' ]: 1.,
            's0w_q0w='   + s0['word'] + '_' + q0['word']: 1.,
            's0p_q0p='   + s0['pos' ] + '_' + q0['pos' ]: 1.,
            'q0p_q1p='   + q0['pos' ] + '_' + q1['pos' ]: 1.,
            # baseline features: from three words
            'q0p_q1p_q2p=' + q0['pos' ] + '_' + q1['pos' ] + '_' + q2['pos' ]: 1.,
            's0p_q0p_q1p=' + s0['pos' ] + '_' + q0['pos' ] + '_' + q1['pos' ]: 1.,
            
            # new feature templates: distance between S0 and N0
            's0_wd=' + s0['word'] + '_' + distance: 1.,
            's0_pd=' + s0['pos' ] + '_' + distance: 1.,
            'q0_wd=' + q0['word'] + '_' + distance: 1.,
            'q0_pd=' + q0['pos' ] + '_' + distance: 1.,
            's0w_q0wd=' + s0['word'] + '_' + q0['word'] + '_' + distance: 1.,
            's0p_q0pd=' + s0['pos' ] + '_' + q0['pos' ] + '_' + distance: 1.,

            # TBA
        }

        # a set of informations at an edge
        info  = dict()
        info['feats']  = feats
        info['otrans'] = transition
        out.add_edge(stid, qtid, info)

        # [DEBUG]
        if not quiet:
            print ' .. Configuration (after) : %s | %s' % (depStack, depBuffer)
            print ' .. Gold transition: [%s] btn [%s, %s]' % (transition, stid, qtid)

    # return the created Graph
    return out


# once we have a graph with weights on the edges, we need to be able
# to make a prediction (i.e., compute the MST):
def updatePredictTransitions(graph):
    for i,j in graph.edges_iter():
        feats = graph[i][j]['feats']
        rweight = weights.dotProduct('r', feats)
        lweight = weights.dotProduct('l', feats)
        sweight = weights.dotProduct('s', feats)
        # select the biggest weight
        transps = [rweight, lweight, sweight]
        transidx= np.argmax(transps)
        # predict transition w.r.t. the weight
        if transidx == 0:   graph[i][j]['ptrans'] = 'r'
        elif transidx == 1: graph[i][j]['ptrans'] = 'l'
        else:               graph[i][j]['ptrans'] = 's'


# compute number of mistakes
def numMistakes(graph):
    err = 0.
    for i,j in graph.edges_iter():
        if graph[i][j]['ptrans'] == graph[i][j]['otrans']: continue # skip
        err += 1
    return err


# now, given a graph (which has features), a true tree and a predicted
# tree, we want to update our weights
def perceptronUpdate(weights, graph):
    # update the weights in cases where 
    # predict transition is not the same as the oracle transition
    for i,j in graph.edges_iter():
        # variables
        feats  = graph[i][j]['feats']
        ptrans = graph[i][j]['ptrans']
        otrans = graph[i][j]['otrans']
        # skip if correct
        if ptrans == otrans : continue # skip
        # update weights ('r', 'l', 's') w.r.t oracles
        if otrans == 'r':
            weights.update('r', feats, 1)
        elif otrans == 'l':
            weights.update('l', feats, 1)
        else:
            weights.update('s', feats, 1)
        # update weights ('r', 'l', 's') w.r.t predictions
        if ptrans == 'r':
            weights.update('r', feats, -1)
        elif ptrans == 'l':
            weights.update('l', feats, -1)
        else:
            weights.update('s', feats, -1)


# now we can finally put it all together to make a single update on a
# single example
def runOneExample(weights, trueGraph, quiet=False):
    # first, compute the full graph and compute edge weights
    G = computeOracleTransition(trueGraph)

    # make a prediction
    updatePredictTransitions(G)

    # compute the error
    err = numMistakes(G)

    # print the predicted tree and error
    if not quiet:
        print 'error =', err, '\tpred =',
        for i,j in G.edges_iter():
            print '(', trueGraph.node[i]['word'], '<->', trueGraph.node[j]['word'], ':', G[i][j]['ptrans'], ')',
        print ''

    # if necessary, make an update
    if err > 0.:
        perceptronUpdate(weights, G)

    return err


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
def predictOneExampleHeads(testGraph, quiet=True):
    # initialize the configurations
    depStack  = deque()
    depBuffer = deque()
    for nid in testGraph.nodes():
        depBuffer.append(nid)
    depStack.append(depBuffer.popleft())    # move the *root*

    # predict the heads by following the predictions (ARC-Standard)
    while depBuffer:
        # prepare the variables to compute features
        stid = depStack[-1]
        qtid = depBuffer[0]
        s0 = testGraph.node[stid]
        s1 = { 'word': MAGIC_CHAR, 'pos': MAGIC_CHAR }
        s2 = { 'word': MAGIC_CHAR, 'pos': MAGIC_CHAR }
        q0 = testGraph.node[qtid]
        q1 = { 'word': MAGIC_CHAR, 'pos': MAGIC_CHAR }
        q2 = { 'word': MAGIC_CHAR, 'pos': MAGIC_CHAR }
        # case where it has more word in stack/buffer
        if len(depStack) == 2:
            s1id = depStack[-2]
            s1 = testGraph.node[s1id]
        elif len(depStack) > 2:
            s1id = depStack[-2]
            s2id = depStack[-3]
            s1 = testGraph.node[s1id]
            s2 = testGraph.node[s2id]
        if len(depBuffer) == 2:
            q1id = depBuffer[1]
            q1 = testGraph.node[q1id]
        elif len(depBuffer) > 2:
            q1id = depBuffer[1]
            q2id = depBuffer[2]
            q1 = testGraph.node[q1id]
            q2 = testGraph.node[q2id]

        # [DEBUG]
        if not quiet:
            print ' .. Configuration (before): %s | %s' % (depStack, depBuffer)
        
        # extract a new features: intermediates
        #  - distance
        distance = str(abs(stid - qtid))

        # extract a new features: from Zhang and Nivre's paper
        feats = { 
            # baseline features: sigle words
            'w_stop='  + s0['word']: 1.,
            'p_stop='  + s0['pos' ]: 1.,
            'w_pstop=' + s0['word'] + '_' + s0['pos' ]: 1.,
            'w_btop='  + q0['word']: 1.,
            'p_btop='  + q0['pos' ]: 1.,
            'w_pbtop=' + q0['word'] + '_' + q0['pos' ]: 1.,
            'w_b2nd='  + q1['word']: 1.,
            'p_b2nd='  + q1['pos' ]: 1.,
            'w_pb2nd=' + q1['word'] + '_' + q1['pos' ]: 1.,
            'w_b3rd='  + q2['word']: 1.,
            'p_b3rd='  + q2['pos' ]: 1.,
            'w_pb3rd=' + q2['word'] + '_' + q2['pos' ]: 1.,
            # baseline features: from word pairs
            's0wp_q0wp=' + s0['word'] + '_' + s0['pos' ] + '_' + q0['word'] + '_' + q0['pos' ]: 1.,
            's0wp_q0w='  + s0['word'] + '_' + s0['pos' ] + '_' + q0['word']: 1.,
            's0w_q0wp='  + s0['word'] + '_' + q0['word'] + '_' + q0['pos' ]: 1.,
            's0wp_q0p='  + s0['word'] + '_' + s0['pos' ] + '_' + q0['pos' ]: 1.,
            's0p_q0wp='  + s0['pos' ] + '_' + q0['word'] + '_' + q0['pos' ]: 1.,
            's0w_q0w='   + s0['word'] + '_' + q0['word']: 1.,
            's0p_q0p='   + s0['pos' ] + '_' + q0['pos' ]: 1.,
            'q0p_q1p='   + q0['pos' ] + '_' + q1['pos' ]: 1.,
            # baseline features: from three words
            'q0p_q1p_q2p=' + q0['pos' ] + '_' + q1['pos' ] + '_' + q2['pos' ]: 1.,
            's0p_q0p_q1p=' + s0['pos' ] + '_' + q0['pos' ] + '_' + q1['pos' ]: 1.,
            
            # new feature templates: distance between S0 and N0
            's0_wd=' + s0['word'] + '_' + distance: 1.,
            's0_pd=' + s0['pos' ] + '_' + distance: 1.,
            'q0_wd=' + q0['word'] + '_' + distance: 1.,
            'q0_pd=' + q0['pos' ] + '_' + distance: 1.,
            's0w_q0wd=' + s0['word'] + '_' + q0['word'] + '_' + distance: 1.,
            's0p_q0pd=' + s0['pos' ] + '_' + q0['pos' ] + '_' + distance: 1.,

            # TBA
        }

        # predict the current transition
        rweight = weights.dotProduct('r', feats)
        lweight = weights.dotProduct('l', feats)
        sweight = weights.dotProduct('s', feats)
        # select the biggest weight
        transps = [rweight, lweight, sweight]
        transidx= np.argmax(transps)
        # convert to str
        if transidx == 0:   transition = 'r'
        elif transidx == 1: transition = 'l'
        else:               transition = 's'
        
        # assign heads w.r.t. the transitions
        if transition == 'r':   # right
            # check if it's possible to remove qtid
            possible = True
            for item in depBuffer:
                if testGraph.node[item]['head'] == str(qtid):
                    possible = False
            for item in depStack:
                if testGraph.node[item]['head'] == str(qtid):
                    possible = False
            # if possible remove qtid w. right
            if possible:
                testGraph.node[qtid]['head'] = str(stid)
                depBuffer.popleft()
                if stid != 0: depBuffer.appendleft(depStack.pop())
            else:
                depStack.append(depBuffer.popleft())
        elif transition == 'l': # left
            # check if it's possible to remove stid
            possible = True
            for item in depBuffer:
                if testGraph.node[item]['head'] == str(stid):
                    possible = False
            for item in depStack:
                if testGraph.node[item]['head'] == str(stid):
                    possible = False
            # if possible remove stid w. left
            if possible:
                testGraph.node[stid]['head'] = str(qtid)
                if stid != 0: depStack.pop()
            # else: does shift
            else:
                depStack.append(depBuffer.popleft())
        else:               # shift
            depStack.append(depBuffer.popleft())

        # [DEBUG]
        if not quiet:
            print ' .. Configuration (after) : %s | %s' % (depStack, depBuffer)
            print ' .. Gold transition: [%s] btn [%s, %s]' % (transidx, stid, qtid)


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

    # from Marine's 2nd present in graphparser.py
    weights = Weights()
    for iteration in range(num_epochs):
        totalErr = 0.
        for G in iterCoNLL(train): 
            totalErr += runOneExample(weights, G, quiet=True)
        print totalErr

    # remove the output file for a new output
    if os.path.isfile(out):
        os.remove(out)

    # iterate over the test cases and create outputs
    for G in iterCoNLL(test):
        predictOneExampleHeads(G)
        printPrediction(out, G)

