#!/usr/bin/env python

import sqlite3
from pathlib import Path

# Paths to databases
db1_path = Path("~/scratch/DFT_OSC.db").expanduser()          # target DB
db2_path = Path("~/scratch/DFT_OSC_emna.db").expanduser()     # source DB

# Open connections
con1 = sqlite3.connect(db1_path)
con2 = sqlite3.connect(db2_path)
cur1 = con1.cursor()
cur2 = con2.cursor()

# 1. Migrate mol_dft_data with ID mapping
id_mapping_mol = {}

rows = cur2.execute("""
    SELECT id, mol_name, ehomo, elumo, dhomo, dlumo, e31 
    FROM mol_dft_data
""").fetchall()

for row in rows:
    old_id = row[0]
    data = row[1:]  # everything except the old id
    cur1.execute("""
        INSERT INTO mol_dft_data (mol_name, ehomo, elumo, dhomo, dlumo, e31)
        VALUES (?, ?, ?, ?, ?, ?)
    """, data)
    new_id = cur1.lastrowid
    id_mapping_mol[old_id] = new_id

# 2. Migrate added_folders with ID mapping
id_mapping_folders = {}

rows = cur2.execute("""
    SELECT id, folder_name FROM added_folders
""").fetchall()

for row in rows:
    old_id = row[0]
    folder_name = row[1]
    cur1.execute("""
        INSERT INTO added_folders (folder_name)
        VALUES (?)
    """, (folder_name,))
    new_id = cur1.lastrowid
    id_mapping_folders[old_id] = new_id

con1.commit()
con1.close()
con2.close()

# 3. Optionally print or save the mappings
print("mol_dft_data ID mapping (old_id → new_id):")
for old_id, new_id in id_mapping_mol.items():
    print(f"{old_id} → {new_id}")

print("\nadded_folders ID mapping (old_id → new_id):")
for old_id, new_id in id_mapping_folders.items():
    print(f"{old_id} → {new_id}")

# Optional: save to file
out_path = Path("~/scratch/id_mapping.txt").expanduser()
with out_path.open("w") as f:
    f.write("mol_dft_data ID mapping:\n")
    for old_id, new_id in id_mapping_mol.items():
        f.write(f"{old_id} → {new_id}\n")
    f.write("\nadded_folders ID mapping:\n")
    for old_id, new_id in id_mapping_folders.items():
        f.write(f"{old_id} → {new_id}\n")

print(f"\nID mappings saved to: {out_path}")
