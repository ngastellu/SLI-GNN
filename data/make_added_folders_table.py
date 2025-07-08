#!/usr/bin/env python

import sqlite3
from pathlib import Path
from tqdm import tqdm

db_path = Path("~/scratch/DFT_OSC.db").expanduser()

con = sqlite3.connect(db_path)
cur = con.cursor()

try:
    cur.execute("CREATE TABLE added_folders (folder_id INTEGER PRIMARY KEY, folder_name TEXT);")
except:
    pass

added_folders = set()


nrows = cur.execute("SELECT COUNT(*) FROM mol_dft_data;").fetchone()[0]

cur.execute("SELECT mol_name FROM mol_dft_data;")

for (mol_name,) in tqdm(cur, total=nrows):
    mol_path = Path(mol_name)
    folder = str(mol_path.parent)
    if folder not in added_folders:
        added_folders.add(folder)



print(f'Done looping over rows! Inserting data into added_folders table...', end = ' ')
cur.executemany("INSERT INTO added_folders (folder_name) VALUES (?)", ((folder,) for folder in added_folders))
con.commit()
print('Done!')  