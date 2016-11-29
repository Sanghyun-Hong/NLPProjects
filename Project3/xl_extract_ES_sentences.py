import sys
import os
import re
import nltk
from scipy import spatial

def get_predictions():

    if len(sys.argv) != 3:
        print 'Usage: script <input_file> <output_file>'
        exit(1)

    fn_in = sys.argv[1]
    fn_out = sys.argv[2]
    if not os.path.exists(fn_in):
        print 'Input path does not exist.'
        exit(1)



    f_in = open(fn_in,'r')
    f_out = open(fn_out,'w')
    for l in f_in:
        s1,s2 = l.split('\t')[:2]
        f_out.write('%s\n' % s1)

    f_out.close()

get_predictions()
