# helper script
# just extract the gold standard scores for the sentences included in the test set used for grading 
import sys
import os
import re
import subprocess



def read_lines(filename):
    lines = open(fi,'r').read().split('\n')
    return map(lambda e: e.strip().lower(),lines)

for f in os.listdir('datasets+scoring_script/test'):
        if 'input' not in f:
            continue
        fi = os.path.join('datasets+scoring_script','test',f)
        f_gs = re.sub('input','gs',f)
        fi_orig = os.path.join('datasets+scoring_script','test_gs',f)
        fi_gs_orig = os.path.join('datasets+scoring_script','test_gs',f_gs)
        assert os.path.exists(fi_gs_orig) and os.path.exists(fi_orig)
        fo_gs = os.path.join('datasets+scoring_script','test',f_gs)

        existing_lines = open(fi,'r').read().split('\n')
        orig_lines = open(fi_orig,'r').read().split('\n')
        orig_lines_gs = open(fi_gs_orig,'r').read().split('\n')

        fo = open(fo_gs,'w')

        '''
        print existing_lines[0]
        print orig_lines[4]
        exit()

        print len(existing_lines),len(orig_lines),len(orig_lines_gs)

        print fi

        existing_lines_gs = []
        for line in existing_lines:
            orig_line_idx = orig_lines.index(line)
            gs = orig_lines_gs[orig_line_idx]
            print gs
        exit()
        '''
        existing_lines_gs = [0]*len(existing_lines)
        existing_lines_idx = 0
        for idx,score in enumerate(orig_lines_gs):
            if len(score.strip()) == 0:
                continue
            #print existing_lines[existing_lines_idx], orig_lines[idx]
            assert existing_lines[existing_lines_idx] in orig_lines[idx]
            existing_lines_gs[existing_lines_idx] = score
            fo.write('%s\n' % score.strip())
            existing_lines_idx += 1


        fo.close()

print 'Done!'
