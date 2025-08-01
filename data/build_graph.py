from pathlib import Path
from time import perf_counter
import sqlite3
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, IterableDataset
from mendeleev import element
from pymatgen.core.structure import IMolecule
from torch_geometric.data import Data


def read_molfile(molfile):
    parent_folder = Path("~/scratch/ML_OSC/").expanduser()
    molfile = parent_folder / molfile
    symbols = []
    pos = []
    with open(molfile) as fo:
        # skip first 2 lines
        fo.readline()
        fo.readline()
        for line in fo:
            if line[0] == '$':
                break
            split_line = line.strip().split()
            symbols.append(split_line[0])
            pos.append([float(coord) for coord in split_line[1:4]])
        
    pos = np.array(pos)
    return IMolecule(symbols, pos)


class DataReader:

    def __init__(self, path, target, table_name='mol_dft_data'):
        self.path = Path(path)
        self.table_name = table_name
        self.target = target
        # print(os.listdir(path))
        assert path.exists(), f'path {str(path)} does not exist!'

        print(f'Reading from database in path: ', path)

        # Connect to database
        self.db_con = sqlite3.connect(path)
        self.db_cur = self.db_con.cursor()

        # store result so that we only need perform this query only once
        query = self.db_cur.execute(f"SELECT MAX(id) FROM {self.table_name};")
        self.ndata = query.fetchone()[0]

        if target is None:
            self.get_id = self._get_id_all_props
        else:
            self.get_id = self._get_id_single_prop

    def _get_id_single_prop(self, material_id):
        query = self.db_cur.execute(f"SELECT mol_name, {self.target} FROM {self.table_name} WHERE id={material_id};")
        mol_name, y = query.fetchone()
        structure = read_molfile(mol_name)
        return structure, [y]
    
    def _get_id_all_props(self, material_id):
        query = self.db_cur.execute(f"SELECT * FROM {self.table_name} WHERE id={material_id};")
        out = query.fetchone()
        mol_name = out[1] #first element of y is the ID; don;'t need it
        structure = read_molfile(mol_name)
        y = out[2:]
        return structure, y 

    def _get_batch(self, ids):
            # prepare (?, ?, …) placeholders
            placeholders = ','.join('?' for _ in ids)
            if self.target:
                q = f"SELECT id, mol_name {','+self.target if self.target else ''} " \
                    f"FROM {self.table_name} WHERE id IN ({placeholders})"
            else:
                q = f"SELECT * FROM {self.table_name} WHERE id IN ({placeholders})"
            rows = self.db_cur.execute(q, ids).fetchall()
            # rows is a list of tuples; turn it into a dict for fast lookup
            lookup = {row[0]: row[1:] for row in rows}
            # return in the same order as `ids`
            out = []
            for mid in ids:
                mol_name, *yvals = lookup[mid]
                mol = read_molfile(mol_name)
                y = [yvals[0]] if self.target else yvals
                out.append((mol, y))
            return out
    

class AtomFeatureEncoder(object):

    def __init__(self, properties_list=None):
        """
        properties = ['atomic_number', 'group_id', 'period', 'nvalence',
                      'en_pauling', 'atomic_radius', 'atomic_volume',
                      'electron_affinity', 'ionenergies']
        """
        self.properties_name = ['N', 'G', 'P', 'NV', 'E', 'R', 'V', 'EA', 'I']
        self.properties_list = properties_list
        self.__atoms = []
        for atom_number in range(1, 101):
            self.__atoms.append(element(atom_number))
        self.__atoms_properties = [list() for i in range(9)]
        self.__atoms_fea = [list() for i in range(9)]
        self.__get_properties()
        self.__disperse()

    def __get_properties(self):
        properties = ['atomic_number', 'group_id', 'period', 'nvalence',
                      'en_pauling', 'atomic_radius', 'atomic_volume',
                      'electron_affinity', 'ionenergies']
        for i in range(len(properties)):
            for atom in self.__atoms:
                fea = getattr(atom, properties[i])
                if properties[i] == 'nvalence':
                    fea = fea()
                elif properties[i] == 'ionenergies':
                    fea = fea[1]
                self.__atoms_properties[i].append(fea if fea is not None else 0)

    def __disperse(self):
        for i in range(4):
            self.__atoms_fea[i] = np.array(self.__atoms_properties[i])
        for i in range(4, 9):
            self.__atoms_fea[i] = pd.cut(self.__atoms_properties[i],
                                         10, right=True, labels=False,
                                         retbins=False, precision=3,
                                         include_lowest=False, duplicates='raise')

    def __get_single_atom_features(self, atom_number):
        single_atom_feature = []
        for i in range(9):
            single_atom_feature.append(self.__atoms_fea[i][atom_number - 1])
        return np.array(single_atom_feature)

    def get_atoms_features(self, atoms):
        if len(self.properties_list) == 1 and self.properties_list[0] == 'N':
            atoms_fea = np.array(atoms).reshape((-1, 1))
        else:
            atoms_fea = []
            for atom_number in atoms:
                atom_fea = []
                single_atom_fea = self.__get_single_atom_features(atom_number)
                for prop in self.properties_list:
                    index = self.properties_name.index(prop)
                    atom_fea.append(single_atom_fea[index])
                atom_fea = np.array(atom_fea)
                atoms_fea.append(atom_fea)
        return torch.LongTensor(atoms_fea)


class BondFeatureEncoder(object):

    def __init__(self, max_num_nbr=12, radius=5, dmin=0, step=0.05):
        self.max_num_nbr, self.radius = max_num_nbr, radius
        self.dmax = self.radius + step
        self.filter = np.arange(dmin, self.dmax + step, step)
        self.num_category = len(self.filter) - 1

    def disperse(self, nbr_fea):
        disperse_nbr_fea = []
        for distances in nbr_fea:
            data = pd.cut(distances, self.filter, labels=False, include_lowest=True)
            disperse_nbr_fea.append(data)
        disperse_nbr_fea = torch.LongTensor(np.array(disperse_nbr_fea))
        return disperse_nbr_fea.view(-1, )

    def format_edges_idx(self, edges_idx):
        size = len(edges_idx)
        src_list = list(range(size))
        all_src_nodes = torch.tensor([[x] * edges_idx.shape[1] for x in src_list]).view(-1).long().unsqueeze(0)
        all_dst_nodes = edges_idx.view(-1).unsqueeze(0)
        return torch.cat((all_src_nodes, all_dst_nodes), dim=0)

    def get_bond_features(self, all_nbrs):
        edges_idx, bond_fea = [], []
        for nbr in all_nbrs:
            if len(nbr) < self.max_num_nbr:
                edges_idx.append(list(map(lambda x: x[2], nbr)) +
                                 [0] * (self.max_num_nbr - len(nbr)))
                bond_fea.append(list(map(lambda x: x[1], nbr)) +
                                [self.dmax] * (self.max_num_nbr - len(nbr)))
            else:
                edges_idx.append(list(map(lambda x: x[2],
                                          nbr[:self.max_num_nbr])))
                bond_fea.append(list(map(lambda x: x[1],
                                         nbr[:self.max_num_nbr])))
        edges_idx, bond_fea = np.array(edges_idx), np.array(bond_fea)
        bond_fea = self.disperse(bond_fea)
        edges_idx = self.format_edges_idx(torch.LongTensor(edges_idx))
        return bond_fea, edges_idx


class AtomBondFactory(object):

    def __init__(self, radius):
        self.radius = radius

    def get_atoms(self, structure):
        atoms = [structure[i].specie.number for i in range(len(structure))]
        return atoms

    def get_edges(self, structure):
        all_nbrs = []
        for atom in structure:
            nbrs = structure.get_neighbors(atom, self.radius)
            all_nbrs.append(nbrs)
        return all_nbrs


class GraphData(Dataset):

    def __init__(self, path, target, table_name, max_num_nbr=12, radius=5, dmin=0, step=0.1, properties_list=None):
        self.data_reader = DataReader(path=path, target=target, table_name=table_name)
        self.atom_bond_factory = AtomBondFactory(radius)
        self.atom_feature_encoder = AtomFeatureEncoder(properties_list=properties_list)
        self.bond_feature_encoder = BondFeatureEncoder(max_num_nbr=max_num_nbr,
                                                       radius=radius, dmin=dmin, step=step)
        self.benchmark_fo = open('graph_data_timing.txt', 'w')
        fo = self.benchmark_fo
        fo.write('# All times are in seconds\n')
        fo.write('Index | DB fetch time | AtomBondFactory.get_atoms | AtomBondFactory.get_edges | AtomFeatureEncoder | BondFeatureEncoder | Data\n')
        fo.flush()


    def __len__(self):
        return self.data_reader.ndata

    def __getitem__(self, idx):
        print(f'\n\n***** GraphData {idx} *****')
        # material_id, target = self.data_reader.id_prop_data[idx]

        # prop_data = self.data_reader.id_prop_data[idx]
        # material_id = prop_data[0]
        # target = prop_data[1:]
        read_start = perf_counter()
        structure, target = self.data_reader.get_id(idx)
        read_end = perf_counter()
        
        read_time = read_end - read_start
        print(f'Fetching from DB = {read_time} seconds',flush=True)

        atoms_start = perf_counter()
        atoms = self.atom_bond_factory.get_atoms(structure)
        atoms_end = perf_counter()
        atoms_time = atoms_end - atoms_start
        print(f'AtomBondFactory.get_atoms = {atoms_time} seconds',flush=True)
        
        
        bonds_start = perf_counter()
        all_nbrs = self.atom_bond_factory.get_edges(structure)
        bonds_end = perf_counter()
        bonds_time = bonds_end - bonds_start
        print(f'AtomBondFactory.get_edges = {bonds_time} seconds',flush=True)

        afe_start = perf_counter()
        atoms_fea = self.atom_feature_encoder.get_atoms_features(atoms)
        afe_end = perf_counter()
        afe_time = afe_end - afe_start
        print(f'AtomFeatureEncoder = {afe_time} seconds',flush=True)

        bfe_start = perf_counter()
        bond_fea, edges_idx = self.bond_feature_encoder.get_bond_features(all_nbrs)
        bfe_end = perf_counter()
        bfe_time = bfe_end - bfe_start
        print(f'bondfeatureencoder = {bfe_time} seconds',flush=True)

        target = torch.Tensor([float(t) for t in target])

        num_atoms = atoms_fea.shape[0]

        dat_start = perf_counter()
        material_graph = Data(x=atoms_fea, edge_index=edges_idx, edge_attr=bond_fea, y=target,
                              material_id=idx, num_atoms=num_atoms)
        dat_end = perf_counter()
        dat_time = dat_end - dat_start
        print(f'Building matl graph = {dat_time} seconds',flush=True)

        self.benchmark_fo.write(f'{idx}\t{read_time}\t{atoms_time}\t{bonds_time}\{afe_time}\t{bfe_time}\t{dat_time}\n')
        self.benchmark_fo.flush()


        return material_graph


class GraphIterableDataset(IterableDataset):
    def __init__(self, path, target, table_name, ids, max_num_nbr=12, radius=5, dmin=0, step=0.1, properties_list=None,batch_size=256):
        self.data_reader = DataReader(path=path, target=target, table_name=table_name)
        self.atom_bond_factory = AtomBondFactory(radius)
        self.atom_feature_encoder = AtomFeatureEncoder(properties_list=properties_list)
        self.bond_feature_encoder = BondFeatureEncoder(max_num_nbr=max_num_nbr,
                                                        radius=radius, dmin=dmin, step=step) 
        self.batch_size = batch_size

        self.ids = ids
    
    def __len__(self):
        return len(self.ids)


    def __iter__(self):
        # chunk the list of ids
        for i in range(0, len(self.ids), self.batch_size):
            batch_ids = self.ids[i:i + self.batch_size]
            molnames_targets = self.data_reader._get_batch(batch_ids)
            data_list = [None] * self.batch_size
            for (structure, target), mid, k in zip(molnames_targets, batch_ids, range(self.batch_size)):
                # same graph‐building logic as __getitem__
                atoms = self.atom_bond_factory.get_atoms(structure)
                nbrs  = self.atom_bond_factory.get_edges(structure)
                atoms_fea = self.atom_feature_encoder.get_atoms_features(atoms)
                bond_fea, edges_idx = self.bond_feature_encoder.get_bond_features(nbrs)
                data = Data(x=atoms_fea,
                            edge_index=edges_idx,
                            edge_attr=bond_fea,
                            y=torch.Tensor([float(t) for t in target]),
                            material_id=mid,
                            num_atoms=atoms_fea.size(0))
                data_list[k] = data
                
                # Sanity checks
                assert isinstance(data, Data), f"Yielded object is not a PyG Data: {type(data)}"
                assert hasattr(data, 'x') and hasattr(data, 'edge_index'), "Data object is malformed"

            yield data_list