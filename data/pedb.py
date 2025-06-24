#!/usr/bin/env python

import sys
from pathlib import Path
import sqlite3
import re


# global vars (don't touch)

HA_TO_EV = 27.2114
DB_PATH = Path("~/scratch/DFT_OSC.db")


def update_db(datadir, db_path=DB_PATH):
    """Updates SQLite database at `db_path` using the contents of the Gaussian log files that folder `datadir` contains. 
    Recursively iterates over all of the along with all of the subfolders of `datadir`."""
    
    if not isinstance(db_path, Path): db_path = Path(db_path)
    if not isinstance(datadir, Path): datadir = Path(datadir)

    print(f'********** Working on folder {str(datadir)} **********', flush=True)

    con = sqlite3.connect(db_path.expanduser())
    cur = con.cursor()

    if not any([f.suffix == '.log' for f in datadir.iterdir()]): # Look for Gaussiam output files
        # Recursion over subfolders of datadir
        for d in subdirs(datadir):
            if str(d.name)[0] != str(datadir)[0]: 
                print('Skipped!')
                continue # skips all subfolders with different inital letter than `datadir`
            update_db(d)
    else:
        print('Found Gaussian log files!')

        conf_names = datadir / "conf_names.txt"

        data = []

        cur.execute("SELECT MAX(id) FROM mol_dft_data;")
        max_id = cur.fetchone()[0]
        print(f'**** MAX ID = {max_id} ****\n\n')

        if max_id is None: max_id = 0

        with open(conf_names) as fc:
            for k, line in enumerate(fc):
                split_line = line.strip().split()
                mol_name = datadir / (split_line[0] + '.molecule')
                sing_name =  datadir / (split_line[2] + '_singlet.log')
                trip_name = datadir / (split_line[2] + '_triplet.log')

                #print(sing_name,flush=True)
                Esinglet, Ehomo, Elumo, dh, dl = parse_singlet_log(sing_name)
                #print(trip_name,flush=True)
                Etriplet = parse_triplet_log(trip_name)
                E31 = Etriplet - Esinglet

                data.append((max_id + 1 + k, str(mol_name), Ehomo, Elumo, dh, dl, E31))
                #print('\n')

        # Wrap up INSERT statement into a transaction and a call to `executemany` for greater efficiency
        con.execute("BEGIN TRANSACTION")
        cur.executemany("INSERT INTO mol_dft_data VALUES (?, ?, ?, ?, ?, ?, ?)", data)
        con.commit()
        print(f'Added {len(data)} rows to {db_path}')


def subdirs(d):
    """Looks for subdirectories in directory `d`. In practice this function is used to determine
    if the script should look for Gaussian output files in the current directory, or in its subdirectories.
    It returns a (possibly empty) list of the subfolders contained in `d`."""
    return [entry for entry in d.iterdir() if entry.is_dir()]

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
                extend_energies(eocc, line)
            elif mo_type == 'virt.':
                extend_energies(evirt, line)
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
    E0 = float(line.split()[4]) * HA_TO_EV
    return E0

def extend_energies(energies, line):
    pattern = r'[-]?\d+\.\d{5}'
    matches = re.findall(pattern, line)
    energies.extend([float(e) * HA_TO_EV for e in matches])



    


# -------------------- MAIN --------------------


outdir = sys.argv[1]
update_db(outdir)

