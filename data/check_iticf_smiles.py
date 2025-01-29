#!/usr/bin/env python

import pandas as pd
import numpy as np

excel_file = 'dataset/data-indoor.xlsx'
df = pd.read_excel(excel_file)

itic_f_smi = np.unique([smi.strip() for smi in df.loc[df['Acceptor'].str.strip() == 'ITIC-F', 'SMILES(acceptor)'].tolist()])
# itic_f_inds =np.array(df.loc[df['Acceptor'].str.strip() == 'ITIC-F', 'SMILES(acceptor)'].index)
itic_4f_smi = np.unique([smi.strip() for smi in df.loc[df['Acceptor'].str.strip() == 'IT-4F', 'SMILES(acceptor)'].tolist()])

print(itic_f_smi)
