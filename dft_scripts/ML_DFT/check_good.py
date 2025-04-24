#!/usr/bin/env python

from glob import glob
import numpy as np
import sys
import os
import subprocess as sbp


run_name = sys.argv[1]

logfiles = glob(os.path.join(run_name, '*.log'))
nruns = len(logfiles)
success_msg = 'Normal termination'
nchars = len(success_msg)

good_runs = np.zeros(nruns, dtype='bool')
for k, lf in enumerate(logfiles):
    print(lf, end =': ')
    last_line = sbp.run(['tail', '-n', '1', lf], capture_output=True).stdout.decode().strip()

    if last_line[:nchars] == success_msg:
        good_runs[k] = True
        print('ツ')
    else:
        print('Fail! Last line = ', last_line)

if np.any(~good_runs):
    ibad = (~good_runs).nonzero()[0]
    outpath = os.path.join(run_name, 'failed_runs.txt')
    print(f'\n *** {ibad.shape[0]} failed runs found ***\nListing problematic logfiles in {outpath}.')
    with open(outpath,'w') as fo:
        for i in ibad:
            fo.write(logfiles[i] + '\n')
else:
    print('\nAll good~')
