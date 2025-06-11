#!/usr/bin/env python

import sys
from pathlib import Path
import sqlite3
import re


Ha2eV = 27.2114

def parse_singlet_log(logfile):
    with open(logfile) as fo:
        
        # read total ground state energy
        E0 = read_E0(fo)
        print('E0 = ', E0, flush=True)
        
        eocc = []
        evirt = []
        # Read orbital energy eigenvalues (keeps reading from previous loop left off)
        for line in fo:
            if line.lstrip()[:5] != 'Alpha':
                continue
            split_line = line.strip().split()
            mo_type = split_line[1]
            if mo_type == 'occ.':
                eocc.extend([float(e) * Ha2eV for e in split_line[4:]])
            elif mo_type == 'virt.':
                evirt.extend([float(e) * Ha2eV for e in split_line[4:]])
                break # only need LUMO and LUMO+1

    eocc.sort()
    evirt.sort()
    ehomo = eocc[-1] 
    dh = eocc[-1] - eocc[-2]
    elumo = evirt[0]
    dl = evirt[1] - evirt[0]

    return E0, ehomo, elumo, dh, dl


def parse_triplet_log(logfile):
    with open(logfile) as fo: 
        # read total ground state energy
        E_triplet = read_E0(fo)
    
    return E_triplet


def read_E0(fo):       
    # read total ground state energy
    for line in fo:
        if line[:10] == ' SCF Done:':
            break
    E0 = float(line.split()[4]) * Ha2eV
    return E0

def append_energies(energies, split_line):
    pass


# -------------------- MAIN --------------------

db_path = Path("~/scratch/DFT_OSC.db")
con = sqlite3.connect(db_path.expanduser())
cur = con.cursor()


if len(sys.argv) == 2:
    nn = int(sys.argv[1])
    datadir = Path(f"C{nn}")

else:
    nn = int(sys.argv[1])
    mm = int(sys.argv[2])
    datadir = Path(f"C{nn}/C{nn}_C{mm}")


conf_names = datadir / "conf_names.txt"

# mol_names = [] 
# sing_names = [] 
# trip_names = [] 

data = []

with open(conf_names) as fc:
    for k, line in enumerate(fc):
        split_line = line.strip().split()
        mol_name = split_line[0] + '.molecule'
        sing_name =  datadir / (split_line[2] + '_singlet.log')
        trip_name = datadir / (split_line[2] + '_triplet.log')

        print(sing_name)
        Esinglet, Ehomo, Elumo, dh, dl = parse_singlet_log(sing_name)
        print(trip_name)
        Etriplet = parse_triplet_log(trip_name)
        E31 = Etriplet - Esinglet

        data.append((k, mol_name, Ehomo, Elumo, dh, dl, E31))
        print('\n')

con.execute("BEGIN TRANSACTION")
cur.executemany("INSERT INTO mol_dft_data VALUES (?, ?, ?, ?, ?, ?, ?)", data)
con.commit()
print(f'Added {len(data)} rows to {db_path}')