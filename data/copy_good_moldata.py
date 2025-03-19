import numpy as np
import csv
from os import path

def filter_copy_csv(csv_in,molprop_name):
    '''Copies entries from `csv_in` which have been successfully converted to XYZ into a new csvfile.'''
    datadir = path.dirname(csv_in)
    xyzdir = path.join(datadir,'xyzfiles')

    csv_out_name = path.basename(csv_in).split('.')[0] + f'_xyzexists_{molprop_name}.csv'
    csv_out=path.join(datadir,csv_out_name)


    fin = open(csv_in)
    fout = open(csv_out,'w', newline='')
    rr = csv.DictReader(fin)
    kerouac = csv.DictWriter(fout,fieldnames=['id',molprop_name])
    kerouac.writeheader()
    for row in rr:
        # n+=1
        id = row['id']
        molprop = row[molprop_name]

        if path.exists(path.join(xyzdir, f'{id}.xyz')):
            # kerouac.writerow(row)
            kerouac.writerow({'id': id, molprop_name: molprop})

    
csvfile = 'D:/harvard-cep-dataset-main/Raw-data/moldata.csv'
target_prop = 'e_homo_alpha'
filter_copy_csv(csvfile,target_prop)