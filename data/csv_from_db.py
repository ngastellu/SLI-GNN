#!/usr/bin/env python

from ase.db import connect
import csv


db_name = input('Name of ASE database: ')
db_path = f'/Users/nico/Desktop/scripts/OPVGCN/data/{db_name}.db'

db = connect(db_path)

out_csv = f'dataset/targets/{db_name}-targets.csv'

with open(out_csv,'w',newline='') as csvfile:
    kerouac = csv.writer(csvfile)
    for row in db.select():
        kerouac.writerow([row.id, row.data['HOMO']])