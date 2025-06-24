#!/usr/bin/env python

import sqlite3
from pathlib import Path

db_path = Path("~/scratch/DFT_OSC.db")
con = sqlite3.connect(db_path.expanduser())
cur = con.cursor()

cur.execute("""CREATE TABLE mol_dft_data (
            id INTEGER PRIMARY KEY,
            mol_name TEXT,
            ehomo REAL,
            elumo REAL,
            dhomo REAL,
            dlumo REAL,
            e31 REAL
            ) STRICT;""")
