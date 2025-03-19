#!/usr/bin/env python

import os
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import pandas as pd
import sys


def write_xyz(coords, symbols, filename, append=False):
    """Writes the coordinates stored in NumPy array to a .xyz file."""

    symbol_field_size = len(sorted(symbols, key=len)[-1])  # Get maximum size of symbols
    if coords.shape[1] == 2:
        new_coords = np.zeros((coords.shape[0], 3), dtype=float)
        new_coords[:, :2] = coords
        coords = new_coords
    iostyle = 'a' if append else 'w'
    with open(filename, iostyle) as fo:
        fo.write(' {:d}\n\n'.format(coords.shape[0]))
        for s, r in zip(symbols, coords):
            x, y, z = r
            fo.write('{0:{width}}\t{1:2.8f}\t{2:2.8f}\t{3:2.8f}\n'.format(s, x, y, z, width=symbol_field_size))


def embed_molecule(hmol,nconfs=10):
    try:
        # First attempt to embed a single conformer
        AllChem.EmbedMolecule(hmol)  # Use EmbedMolecule for a single conformer
        positions = hmol.GetConformer().GetPositions()  # Get positions of the conformer
    except ValueError as e:
        print("Error using single conformer: ", e)
        try:
            # If the first attempt fails, try embedding multiple conformers
            AllChem.EmbedMultipleConfs(hmol, numConfs=nconfs)
            # Get the positions of the first conformer
            positions = hmol.GetConformer().GetPositions()
        except ValueError as e:
            print("Error generating multiple conformers: ", e)
            positions = None
    return positions



# ---------------- MAIN -------------------


#moltype = 'Acceptor' # Donor or Acceptor (must be capitalized!)

# Load the Data_file
excel_file = 'D:\harvard-cep-dataset\raw_data\moldata.csv'
# excel_file = '/Users/emna/Desktop/ML_OSC/data-indoor.xlsx'
df = pd.read_excel(excel_file)

# Initialize a boolean array to track failed runs
nrows = df.shape[0]
failed_runs = np.zeros(nrows, dtype=bool)

# Get the directory
excel_dir = os.path.dirname(excel_file)

# Create the 'xyzfiles' directory
xyz_dir = os.path.join(excel_dir, 'xyzfiles')
os.makedirs(xyz_dir, exist_ok=True)  # Corrected variable name from xyz5_dir to xyz_dir

#seen = ['PC61BM', 'PC71BM', 'Y6', 'BTA3', 'C70','ICBA', 'tPDI2N-EH', 'ITIC-M', 'ITIC-F', 'IT-4F']

#if sys.argv[1] == 'all' or len(sys.argv) == 1:
    #print('*** Looping over entire Excel file ***')

# Iterate over rows and extract SMILES strings
for index, row in df.iterrows():
        # donor_smiles = row['SMILES(donor)']
        molecule_ID = str(row['id'])
        smiles=row['SMILES_str']

        #if name in seen:
            #print(f"Already seen {name}!")
            #continue

        #seen.append(name) # add acceptor name to list of already processed acceptors

        #print(f'\nWorking on {name}...')
        #smiles = row[f'SMILES({moltype.lower()})']

        # Process acceptor SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Invalid SMILES: {smiles} for molecule ID {molecule_ID}")
            #failed_runs[index] = True
            continue
        
        hmol = Chem.AddHs(mol)

        positions = embed_molecule(hmol)

        if positions is not None:
            filename = os.path.join(xyz_dir,f'{molecule_ID}.xyz')
            # If we reach here, we have valid positions
            symbols = [at.GetSymbol() for at in hmol.GetAtoms()]
            # filename = os.path.join(xyz_dir, f'{index}.xyz')
            write_xyz(positions, symbols, filename)
        else:
            print(f"failed to generate coorinates for molecule ID {molecule_ID}")

    # Save the failed runs array
    #np.save(f'failed_runs_{moltype}.npy', failed_runs)

    # Print out the indices of the failed runs
    #failed_indices = np.where(failed_runs)[0]
    #if len(failed_indices) > 0:
        # print(f"Failed runs at indices: {failed_indices.tolist()}")
        #print(f'Failed {moltype}: ')
        #failed_acc = np.unique([df.iloc[i][moltype] for i in failed_indices])
        #for a in failed_acc:
            #print(a)
    #else:
        #print("No failed runs.")

#else:
    #name = sys.argv[1].strip()
    #smile = [smi.strip() for smi in df.loc[df[moltype].str.strip() == name, f'SMILES({moltype.lower()})'].tolist()][0] # keep only the first SMILE string
    #mol = Chem.MolFromSmiles(smile)
    #hmol = Chem.AddHs(mol)
    #positions = embed_molecule(hmol)
    #if positions is not None:
        #filename = os.path.join('dataset',f'{name}.xyz')
        # If we reach here, we have valid positions
        #symbols = [at.GetSymbol() for at in hmol.GetAtoms()]
        # acceptor_filename = os.path.join(xyz_dir, f'acceptor_{index}.xyz')
        #write_xyz(positions, symbols, filename)
        #print('Success! :)')
    #else:
        #print('Did not work :/')


