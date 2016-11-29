import sys
import os
import re
import nltk
from scipy import spatial
from langdetect import detect
import pickle

def get_predictions():

    if len(sys.argv) != 4:
        print 'Usage: script <input_file> <output_file> <mask_file>'
        exit(1)

    fn_in = sys.argv[1]
    fn_out = sys.argv[2]
    fn_out_mask = sys.argv[3]
    if not os.path.exists(fn_in):
        print 'Input path does not exist.'
        exit(1)

    f_in = open(fn_in,'r')
    f_out = open(fn_out,'w')
    f_out_mask = open(fn_out_mask,'w')
    mask = []
    for i,l in enumerate(f_in):
        #print len(l.split('\t'))
        s1,s2 = l.split('\t')[:2]
        s1 = s1.decode('utf8','ignore')
        s2 = s2.decode('utf8','ignore')

        ds1 = str(detect(s1))
        ds2 = str(detect(s2))

        if ds1 == 'en':
            mask.append(2)
            f_out.write('%s\n' % (s2.encode('utf-8').strip()))
        else:
            mask.append(1)
            f_out.write('%s\n' % (s1.encode('utf-8').strip()))

    f_out.close()
    print mask
    pickle.dump(mask,f_out_mask)
    f_out_mask.close()

get_predictions()
