import sys
import os
import re
import subprocess

correlation_script = 'datasets+scoring_script/correlation-noconfidence.pl'

if not os.path.exists(correlation_script):
    exit(1)

if len(sys.argv) != 3:
    exit(1)

fn_in = sys.argv[1]
fn_in_gs = re.sub('input','gs',fn_in)
fn_out = sys.argv[2]

if not os.path.exists(fn_in):
    exit(1)

if not os.path.exists(fn_in_gs):
    exit(1)

if not os.path.exists(fn_out):
    exit(1)

# preprocess outputs
outstr = subprocess.check_output([correlation_script,fn_out,fn_in_gs])
perfor = outstr.split(':')[1]
print float(perfor)
