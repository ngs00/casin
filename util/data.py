import numpy
import pandas
import torch
import json
import os
from tqdm import tqdm
from rdkit.Chem import MolFromMolFile
from torch_geometric.data import Batch
from util.chem import load_elem_attrs
from util.crystal import get_mol_graph
from util.crystal import get_crystal_graph


class ChemSystem:
    def __init__(self, struct_react, struct_env, sys_id, y=None):
        self.struct_react = struct_react
        self.struct_env = struct_env if isinstance(struct_env, list) else [struct_env]
        self.y = y
        self.sys_id = sys_id


def load_dataset(path_metadata_file, path_structs, idx_target, n_bond_feats, atomic_cutoff=4.0):
    metadata = numpy.array(pandas.read_excel(path_metadata_file))
    elem_attrs_org = load_elem_attrs('res/matscholar-embedding.json')
    elem_attrs_inorg = load_elem_attrs('res/cgcnn-embedding.json')
    rbf_means = numpy.linspace(start=1.0, stop=atomic_cutoff, num=n_bond_feats)
    dataset = list()

    for i in tqdm(range(0, metadata.shape[0])):
        struct_id = metadata[i, 0]
        g_org = get_mol_graph(MolFromMolFile(path_structs + '/' + struct_id + '.mol'), elem_attrs_org)
        g_inorg = get_crystal_graph(path_structs + '/' + struct_id + '.cif',
                                    elem_attrs_inorg,
                                    rbf_means,
                                    atomic_cutoff)

        if g_org is not None and g_inorg is not None:
            dataset.append(ChemSystem(g_org, g_inorg, sys_id=i, y=metadata[i, idx_target]))

    return dataset


def load_dataset_cathub(path_metadata_file, path_structs, idx_target, n_bond_feats, atomic_cutoff=4.0):
    metadata = numpy.array(pandas.read_excel(path_metadata_file))
    elem_attrs_product = load_elem_attrs('res/cgcnn-embedding.json')
    elem_attrs_env_mol = load_elem_attrs('res/matscholar-embedding.json')
    elem_attrs_env_surface = load_elem_attrs('res/cgcnn-embedding.json')
    rbf_means = numpy.linspace(start=1.0, stop=atomic_cutoff, num=n_bond_feats)
    dataset = list()

    for i in tqdm(range(0, metadata.shape[0])):
        sys_id = metadata[i, 0]
        product_id = list(json.loads(metadata[i, 5].replace('\'', '"')).keys())[0]
        env_ids = list(json.loads(metadata[i, 4].replace('\'', '"')).keys())

        fname_product = path_structs + '/' + sys_id + '/' + product_id + '.cif'
        fname_env_mol = path_structs + '/' + sys_id + '/' + env_ids[0] + '.mol'
        fname_env_surface = path_structs + '/' + sys_id + '/' + env_ids[1] + '.cif'

        if not validate_files([fname_product, fname_env_mol, fname_env_surface]):
            continue

        g_product = get_crystal_graph(fname_product, elem_attrs_product, rbf_means, atomic_cutoff)
        g_env_mol = get_mol_graph(MolFromMolFile(fname_env_mol), elem_attrs_env_mol)
        g_env_surface = get_crystal_graph(fname_env_surface, elem_attrs_env_surface, rbf_means, atomic_cutoff)

        if g_product is not None and g_env_mol is not None and g_env_surface is not None:
            dataset.append(ChemSystem(g_product, [g_env_mol, g_env_surface], i, metadata[i, idx_target]))

    return dataset


def split_dataset(dataset, ratio_train, random_seed=None):
    n_train = int(ratio_train * len(dataset))
    n_val = int(0.2 * len(dataset))

    if random_seed is not None:
        numpy.random.seed(random_seed)

    idx_rand = numpy.random.permutation(len(dataset))

    dataset_train = [dataset[idx] for idx in idx_rand[:n_train]]
    dataset_val = [dataset[idx] for idx in idx_rand[n_train:n_train+n_val]]
    dataset_test = [dataset[idx] for idx in idx_rand[n_train+n_val:]]

    return dataset_train, dataset_val, dataset_test


def collate(batch):
    n_env_structs = len(batch[0].struct_env)
    structs_react = list()
    structs_env = [list() for i in range(0, n_env_structs)]
    y = list()

    for b in batch:
        structs_react.append(b.struct_react)
        structs_env.append(b.struct_env)
        y.append(torch.tensor(b.y, dtype=torch.float))

        for i in range(0, n_env_structs):
            structs_env[i].append(b.struct_env[i])

    structs_react = Batch.from_data_list(structs_react)
    structs_env = [Batch.from_data_list(structs_env[i]) for i in range(0, n_env_structs)]
    y = torch.vstack(y)

    return structs_react, structs_env, y


def validate_files(file_names):
    for fname in file_names:
        if not os.path.isfile(fname):
            return False

    return True
