#!/usr/bin/env python

from glob import glob
import numpy as np
import sys
import os
import subprocess as sbp

run_name = sys.argv[1]
run_mol_list_file = os.path.join(run_name, 'conf_names.txt')

with open(run_mol_list_file) as fo:
    mol_IDs = [line.split()[-1].strip() for line in fo]

nruns = len(mol_IDs) * 2
success_msg = 'Normal termination'
nchars = len(success_msg)

exit_codes = np.zeros(nruns, dtype='int')
run_IDs = np.empty(nruns, dtype='str')

# good_runs values meaning:
#   0 = success
#   1 = missing comfile
#   2 = missing logfile
#   3 = run did not terminate properly

k = 0
for mol_id in mol_IDs:
    for run_type in ['singlet', 'triplet']:
        run_IDs[k] = f'{mol_id}_{run_type}'
        print(f'{mol_id}_{run_type}: ', end = '')
        comfile = os.path.join(run_name, f'{mol_id}_{run_type}.com')
        logfile = os.path.join(run_name, f'{mol_id}_{run_type}.log')

        if not os.path.exists(comfile):
            print('Fail! Missing comfile.')
            exit_codes[k] = 1

        elif not os.path.exists(logfile):
            print('Fail! Missing logfile.')
            exit_codes[k] = 2

        else:
            last_line = sbp.run(['tail', '-n', '1', logfile], capture_output=True).stdout.decode().strip()
            if last_line[:nchars] == success_msg:
                print('ツ')
            else:
                print('Fail! Last line = ', last_line)
                exit_codes[k] = 3
        k += 1

if np.any(exit_codes > 0):
    ibad = exit_codes.nonzero()[0]
    outpath = os.path.join(run_name, 'failed_runs.txt')
    print(f'\n *** {ibad.shape[0]} failed runs found (including {(exit_codes == 1).sum()} missing comfiles) ***\nListing problematic runs in {outpath}.')
    with open(outpath,'w') as fo:
        for i in ibad:
            fo.write(f'{run_IDs[i]}  {exit_codes[i]}\n')
else:
    print('\nAll good~')
