#!/usr/bin/env python

import pandas as pd
import csv
from os import path


df = pd.read_excel('data-indoor.xlsx')

moltype = 'Donor'
motype = 'LUMO'

out_csv = f'dataset/targets/data-indoor_{moltype}_{motype}.csv'

seen = []

with open(out_csv,'w',newline='') as csvfile:
    kerouac = csv.writer(csvfile)
    for i, row in df.iterrows():
        name = row[moltype]
        if name in seen:
            continue
        seen.append(name)
        if path.exists(f'dataset/{name}.xyz') or path.exists(f'data/{name}.sdf'):
            energy = row[f'{moltype}_{motype}(ev)']
            print(name,energy)
            kerouac.writerow([name,energy])
        else:
            continue