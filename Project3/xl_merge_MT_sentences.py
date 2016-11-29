import sys
import os
import re
import nltk
from scipy import spatial
import pickle

def get_predictions():

    if len(sys.argv) != 5:
        print 'Usage: script <input_file> <mt_file> <mask_file> <out_file>'
        exit(1)

    fn_in = sys.argv[1]
    fn_mt = sys.argv[2]
    fn_mask = sys.argv[3]
    fn_out = sys.argv[4]
    if not os.path.exists(fn_in):
        print 'Input path does not exist.'
        exit(1)

    if not os.path.exists(fn_mt):
        print 'MT path does not exist.'
        exit(1)

    if not os.path.exists(fn_mask):
        print 'MASK path does not exist.'
        exit(1)

    f_in = open(fn_in,'r')
    in_sents = map(lambda e: e.strip(), f_in.read().strip().split('\n'))

    f_in_mt = open(fn_mt,'r')
    mted_sents = map(lambda e: e.strip(), f_in_mt.read().strip().split('\n'))

    mask = pickle.load(open(fn_mask,'r'))

    print len(mask),len(in_sents)

    assert len(in_sents) == len(mted_sents) and len(mted_sents) == len(mask)

    f_out = open(fn_out,'w')
    for i in range(len(in_sents)):
        s1,s2 = in_sents[i].split('\t')[:2]
        m = mask[i]
        st = mted_sents[i]
        if m == 2:
            f_out.write('%s\t%s\n' % (s1,st))
        elif m == 1:
            f_out.write('%s\t%s\n' % (st,s2))
        else:
            assert False
    f_out.close()

get_predictions()

print 'Done!'
