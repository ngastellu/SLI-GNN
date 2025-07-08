import numpy as np
import sqlite3
from pathlib import Path


def get_batch(db_cur,table_name, ids, target=None):
        # prepare (?, ?, …) placeholders
        placeholders = ','.join('?' for _ in ids)
        if target:
            q = f"SELECT id, mol_name {','+target if target else ''} " \
                f"FROM {table_name} WHERE id IN ({placeholders})"
        else:
            q = f"SELECT * FROM {table_name} WHERE id IN ({placeholders})"
        rows = db_cur.execute(q, ids).fetchall()
        # rows is a list of tuples; turn it into a dict for fast lookup
        lookup = {row[0]: row[1:] for row in rows}
        # return in the same order as `ids`
        out = []
        for mid in ids:
            mol_name, *yvals = lookup[mid]
            y = yvals[0] if target else np.array(yvals)
            out.append((mol_name, y))
        return out


db_path = Path("~/scratch/DFT_OSC.db").expanduser()
table_name = 'mol_dft_data'

con = sqlite3.connect(db_path)
cur = con.cursor()

ids = list(range(1,10))

out_homo = get_batch(cur, table_name, ids, target='ehomo')
for xy in out_homo:
    x,y = xy
    print(f'mol_name = {x} ---> ehomo = {y}')


out_all = get_batch(cur, table_name, ids)
for xy in out_all:
    x,y = xy
    print(f'mol_name = {x} ---> ehomo = {y[0]} ; elumo = {y[1]} ; dhomo = {y[2]} ; dlumo = {y[3]} ; e31 = {y[4]}')
