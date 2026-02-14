from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from Bio.PDB import get_surface, PDBParser, ShrakeRupley

import os
from scipy import spatial
import torch
import warnings
from scipy.spatial import KDTree
from ogb.utils.features import (atom_to_feature_vector, bond_to_feature_vector)

biopython_parser = PDBParser()

allowable_features = {
    'possible_lig_atom_list': ['C','N','O','S','P','Br','Cl','F','I','H','Si','B','Na','K','Al','Ca','Sn','As','Hg','Fe','Zn','Cr','Se','Gd','Au','Li','misc'],
    'possible_chirality_list': [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER',
        'CHI_SQUAREPLANAR',
        'CHI_OCTAHEDRAL',
        'CHI_TRIGONALBIPYRAMIDAL'
    ],
    'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
    'possible_numring_list': [0, 1, 2, 3, 4, 5, 6, 'misc'],
    'possible_implicit_valence_list': [0, 1, 2, 3, 4, 5, 6, 'misc'],
    'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'],
    'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    'possible_number_radical_e_list': [0, 1, 2, 3, 4, 'misc'],
    'possible_hybridization_list': [
        'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc'
    ],
    'possible_is_aromatic_list': [False, True],
    'possible_is_in_ring3_list': [False, True],
    'possible_is_in_ring4_list': [False, True],
    'possible_is_in_ring5_list': [False, True],
    'possible_is_in_ring6_list': [False, True],
    'possible_is_in_ring7_list': [False, True],
    'possible_is_in_ring8_list': [False, True],
    'possible_amino_acids': ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET',
                             'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'HIP', 'HIE', 'TPO', 'HID', 'LEV', 'MEU',
                             'PTR', 'GLV', 'CYT', 'SEP', 'HIZ', 'CYM', 'GLM', 'ASQ', 'TYS', 'CYX', 'GLZ', 'misc'],
    'possible_atom_type': ['C', 'CA', 'CB', 'CD', 'CD1', 'CD2', 'CE', 'CE1', 'CE2', 'CE3', 'CG', 'CG1', 'CG2', 'CH2',
                             'CZ', 'CZ2', 'CZ3', 'N', 'ND1', 'ND2', 'NE', 'NE1', 'NE2', 'NH1', 'NH2', 'NZ', 'O', 'OD1',
                             'OD2', 'OE1', 'OE2', 'OG', 'OG1', 'OH', 'OXT', 'SD', 'SG', 'H','misc'],
}


# 使用renumber将原子按smiles的顺序排列
def read_molecule(molecule_file, sanitize=False, calc_charges=False, remove_hs=False):  # equibind-process_mol
    """Load a molecule from a file of format ``.mol2`` or ``.sdf`` or ``.pdbqt`` or ``.pdb``.

    Parameters
    ----------
    molecule_file : str
        Path to file for storing a molecule, which can be of format ``.mol2`` or ``.sdf``
        or ``.pdbqt`` or ``.pdb``.
    sanitize : bool
        Whether sanitization is performed in initializing RDKit molecule instances. See
        https://www.rdkit.org/docs/RDKit_Book.html for details of the sanitization.
        Default to False.
    calc_charges : bool
        Whether to add Gasteiger charges via RDKit. Setting this to be True will enforce
        ``sanitize`` to be True. Default to False.
    remove_hs : bool
        Whether to remove hydrogens via RDKit. Note that removing hydrogens can be quite
        slow for large molecules. Default to False.
    use_conformation : bool
        Whether we need to extract molecular conformation from proteins and ligands.
        Default to True.

    Returns
    -------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule instance for the loaded molecule.
    coordinates : np.ndarray of shape (N, 3) or None
        The 3D coordinates of atoms in the molecule. N for the number of atoms in
        the molecule. None will be returned if ``use_conformation`` is False or
        we failed to get conformation information.
    """
    if molecule_file.endswith('.mol2'):
        mol = Chem.MolFromMol2File(molecule_file, sanitize=False, removeHs=False)
    elif molecule_file.endswith('.sdf'):
        supplier = Chem.SDMolSupplier(molecule_file, sanitize=False, removeHs=False)
        mol = supplier[0]
    elif molecule_file.endswith('.pdbqt'):
        with open(molecule_file) as file:
            pdbqt_data = file.readlines()
        pdb_block = ''
        for line in pdbqt_data:
            pdb_block += '{}\n'.format(line[:66])
        mol = Chem.MolFromPDBBlock(pdb_block, sanitize=False, removeHs=False)
    elif molecule_file.endswith('.pdb'):
        mol = Chem.MolFromPDBFile(molecule_file, sanitize=False, removeHs=False)
    else:
        return ValueError('Expect the format of the molecule_file to be '
                          'one of .mol2, .sdf, .pdbqt and .pdb, got {}'.format(molecule_file))

    try:
        if sanitize or calc_charges:
            Chem.SanitizeMol(mol)

        if calc_charges:
            # Compute Gasteiger charges on the molecule.
            try:
                AllChem.ComputeGasteigerCharges(mol)
            except:
                warnings.warn('Unable to compute charges for the molecule.')

        if remove_hs:
            mol = Chem.RemoveHs(mol, sanitize=sanitize)
    except:
        return None

    return mol


def get_receptor(rec_path, lig, cutoff):  # equibind-process_mols
    #获取lig的原子坐标
    conf = lig.GetConformer()
    lig_coords = conf.GetPositions()
    #get_structure函数返回pdb文件的结构
    structure = biopython_parser.get_structure('random_id', rec_path)  
    rec = structure[0]
    min_distances = []
    coords = []
    valid_chain_ids = []
    lengths = []
    for i, chain in enumerate(rec):
        chain_coords = []  
        count = 0
        invalid_res_ids = []
        for res_idx, residue in enumerate(chain):
            if residue.get_resname() == 'HOH':
                invalid_res_ids.append(residue.get_id())
                continue
            residue_coords = []
            c_alpha, n, c = None, None, None
            for atom in residue:
                if atom.name == 'CA':
                    c_alpha = list(atom.get_vector())
                if atom.name == 'N':
                    n = list(atom.get_vector())
                if atom.name == 'C':
                    c = list(atom.get_vector())
                residue_coords.append(list(atom.get_vector()))
            # TODO: Also include the chain_coords.append(np.array(residue_coords)) for non amino acids such that they can be used when using the atom representation of the receptor
            if c_alpha != None and n != None and c != None:  # only append residue if it is an amino acid and not some weired molecule that is part of the complex
                chain_coords.append(np.array(residue_coords))
                count += 1
            else:
                invalid_res_ids.append(residue.get_id()) #无效的残基
        for res_id in invalid_res_ids: #将无效的残基从链中移除
            chain.detach_child(res_id)
        if len(chain_coords) > 0:
            all_chain_coords = np.concatenate(chain_coords, axis=0) #将链上所有残基中的坐标整合为一个维度
            distances = spatial.distance.cdist(lig_coords, all_chain_coords)
            min_distance = distances.min() #计算lig中某一个原子距离该链上最近的一个原子的距离
        else:
            min_distance = np.inf

        min_distances.append(min_distance)
        lengths.append(count)
        coords.append(chain_coords)
        if min_distance < cutoff:
            valid_chain_ids.append(chain.get_id())
    min_distances = np.array(min_distances)
    if len(valid_chain_ids) == 0:
        valid_chain_ids.append(np.argmin(min_distances))
    valid_coords = []
    valid_lengths = []
    invalid_chain_ids = []
    for i, chain in enumerate(rec):
        if chain.get_id() in valid_chain_ids:
            valid_coords.append(coords[i])
            valid_lengths.append(lengths[i])
        else:
            invalid_chain_ids.append(chain.get_id())
    coords = [item for sublist in valid_coords for item in sublist]  # list with n_residues arrays: [n_atoms, 3]

    for invalid_id in invalid_chain_ids: #将无效的链从rec结构中移除
        rec.detach_child(invalid_id)

    coords = np.concatenate(coords, axis=0)  #将各个残基的坐标整合为整体的坐标，coords中对应存放所有原子对应的坐标

    return rec, coords


def get_pocket_protein(pocket_coords, rec_coords, cutoff):  # 获取给定pocket内的蛋白质原子的索引(由0开始)
    
    pocket_rec_distance = np.sqrt(np.sum(np.asarray(pocket_coords - rec_coords)**2, axis=1)) #计算ligand中心原子距离受体中有效原子的距离
    satisfies_cufoff_mask = np.where(pocket_rec_distance < cutoff)[0] #返回所有距离小于cutoff的受体原子的index
    
    #限制pocket原子的个数
    if len(satisfies_cufoff_mask) > 256: 
        dict_index = {}
        for i, v in enumerate(pocket_rec_distance):
            dict_index[v] = i
        sorted_temp = sorted(dict_index.items())
        satisfies_cufoff_mask = np.array([sorted_temp[i][1] for i in range(len(sorted_temp))][:256])

    return satisfies_cufoff_mask


def get_rec_mask(rec_coords, lig_coords, num_atoms=256):
    #计算ligand原子和蛋白质原子的距离矩阵
    dist_matrix = np.linalg.norm(rec_coords[:, np.newaxis, :] - lig_coords, axis=2)

    candidate_atoms = np.unique(np.where(dist_matrix <= 6)[0])

    min_dist = np.min(dist_matrix[candidate_atoms], axis=1)
    min_ligand = np.argmin(dist_matrix[candidate_atoms], axis=1)

    sorted_atoms = candidate_atoms[np.argsort(min_dist)]

    satisfies_cutoff_mask= sorted_atoms[:num_atoms]
    return satisfies_cutoff_mask


def safe_index(l, e):
    """
    Return index of element e in list l. If e is not present, return the last index
    """
    try:
        return l.index(e)
    except:
        return len(l) - 1


sr = ShrakeRupley(probe_radius=1.4, n_points=100)
def get_rec_feature(rec):
    feature = torch.tensor([])
    sr.compute(rec, level="A") # 分配ASA值的级别，"A"原子级别，"R"残基级别,"C"原子级别...
    for atom in rec.get_atoms():
        atom_name, element = atom.name, atom.element #原子的名称、原子的元素
        if element == 'H':
            atom_name = 'H'
        sasa = atom.sasa # 溶剂可及表面积
        bfactor = atom.bfactor #各向同性B因子
        assert not element == ''
        assert not np.isinf(bfactor) #确保bfactor不是无穷大
        assert not np.isnan(bfactor) #确保bfactor不是空值
        assert not np.isinf(sasa)
        assert not np.isnan(sasa)
        cur = torch.tensor([])
        # atom.get_parent 获取原子对应的残基
        residue_type = torch.tensor([safe_index(allowable_features['possible_amino_acids'], atom.get_parent().get_resname())])
        #残基类型设置为one-hot编码
        cur = torch.cat([cur, torch.nn.functional.one_hot(residue_type, num_classes=len(allowable_features['possible_amino_acids']))],dim=-1)
        atom_type = torch.tensor([safe_index(allowable_features['possible_atom_type'], atom_name)])
        cur = torch.cat([cur, torch.nn.functional.one_hot(atom_type, num_classes=len(allowable_features['possible_atom_type']))],dim=-1)
        #unsqueze函数，dim=-1，表示在最后一个维度插入新的维度
        cur = torch.cat([cur, torch.tensor([sasa]).unsqueeze(0)],dim=-1)
        cur = torch.cat([cur, torch.tensor([bfactor]).unsqueeze(0)],dim=-1)
        feature = torch.cat([feature, cur],dim=0)
    return feature.to(dtype=torch.float32)  # (N_res, 1)

def get_lig_feature(lig):
    conf = lig.GetConformer()
    lig_coords = conf.GetPositions()
    # ComputeGasteigerCharges(lig)
    ringinfo = lig.GetRingInfo() #获得分子的RingInfo对象数量
    atom_features = torch.tensor([])
    # tag = 0
    for idx, atom in enumerate(lig.GetAtoms()):#返回一个包含分子所有原子的只读序列
        # g_charge = atom.GetDoubleProp('_GasteigerCharge')
        if atom.GetSymbol() != 'H':
            cur = torch.tensor([])
            Symbol = torch.tensor([safe_index(allowable_features['possible_lig_atom_list'], atom.GetSymbol())])
            cur = torch.cat(
                [cur, torch.nn.functional.one_hot(Symbol, num_classes=len(allowable_features['possible_lig_atom_list']))],
                dim=-1)

            # atom.GetChiralTag返回原子对应的手性标签，整体返回对应手性的下标
            ChiralTag = torch.tensor([allowable_features['possible_chirality_list'].index(str(atom.GetChiralTag()))])
            cur = torch.cat([cur, torch.nn.functional.one_hot(ChiralTag, num_classes=len(
                allowable_features['possible_chirality_list']))], dim=-1)

            # atom.GetTotalDegree返回原子的总度数（邻居的数量+Hs的数量）
            TotalDegree = torch.tensor([safe_index(allowable_features['possible_degree_list'], atom.GetTotalDegree())])
            cur = torch.cat([cur, torch.nn.functional.one_hot(TotalDegree,
                                                            num_classes=len(allowable_features['possible_degree_list']))],
                            dim=-1)

            # atom.GetFormalCharge返回原子的形式电荷
            FormalCharge = torch.tensor(
                [safe_index(allowable_features['possible_formal_charge_list'], atom.GetFormalCharge())])
            cur = torch.cat([cur, torch.nn.functional.one_hot(FormalCharge, num_classes=len(
                allowable_features['possible_formal_charge_list']))], dim=-1)

            # atom.GetImplictitValence返回原子上隐含的Hs数量
            ImplicitValence = torch.tensor(
                [safe_index(allowable_features['possible_implicit_valence_list'], atom.GetImplicitValence())])
            cur = torch.cat([cur, torch.nn.functional.one_hot(ImplicitValence, num_classes=len(
                allowable_features['possible_implicit_valence_list']))], dim=-1)

            # 返回原子上的Hs总数（显性和隐性）
            TotalNumHs = torch.tensor([safe_index(allowable_features['possible_numH_list'], atom.GetTotalNumHs())])
            cur = torch.cat(
                [cur, torch.nn.functional.one_hot(TotalNumHs, num_classes=len(allowable_features['possible_numH_list']))],
                dim=-1)

            # 获得自由基电子的数目
            NumRadicalElectrons = torch.tensor(
                [safe_index(allowable_features['possible_number_radical_e_list'], atom.GetNumRadicalElectrons())])
            cur = torch.cat([cur, torch.nn.functional.one_hot(NumRadicalElectrons, num_classes=len(
                allowable_features['possible_number_radical_e_list']))], dim=-1)

            # 返回原子的杂化
            Hybridization = torch.tensor(
                [safe_index(allowable_features['possible_hybridization_list'], str(atom.GetHybridization()))])
            cur = torch.cat([cur, torch.nn.functional.one_hot(Hybridization, num_classes=len(
                allowable_features['possible_hybridization_list']))], dim=-1)

            # atom.GetIsAromatic判断原子是否在芳香烃内
            IsAromatic = torch.tensor([allowable_features['possible_is_aromatic_list'].index(atom.GetIsAromatic())])
            cur = torch.cat([cur, torch.nn.functional.one_hot(IsAromatic, num_classes=len(
                allowable_features['possible_is_aromatic_list']))], dim=-1)

            # ringinfo.NumAtomRings返回idx所参与的环的数量
            NumAtomRings = torch.tensor(
                [safe_index(allowable_features['possible_numring_list'], ringinfo.NumAtomRings(idx))])
            cur = torch.cat([cur, torch.nn.functional.one_hot(NumAtomRings, num_classes=len(
                allowable_features['possible_numring_list']))], dim=-1)

            # ringinfo.IsAtomInRingOfSize(idx, 3)返回原子是否在一个大小为3的环中
            IsAtomInRingOfthree = torch.tensor(
                [allowable_features['possible_is_in_ring3_list'].index(ringinfo.IsAtomInRingOfSize(idx, 3))])
            cur = torch.cat([cur, torch.nn.functional.one_hot(IsAtomInRingOfthree, num_classes=len(
                allowable_features['possible_is_in_ring3_list']))], dim=-1)

            IsAtomInRingOffour = torch.tensor(
                [allowable_features['possible_is_in_ring4_list'].index(ringinfo.IsAtomInRingOfSize(idx, 4))])
            cur = torch.cat([cur, torch.nn.functional.one_hot(IsAtomInRingOffour, num_classes=len(
                allowable_features['possible_is_in_ring4_list']))], dim=-1)

            IsAtomInRingOffive = torch.tensor(
                [allowable_features['possible_is_in_ring5_list'].index(ringinfo.IsAtomInRingOfSize(idx, 5))])
            cur = torch.cat([cur, torch.nn.functional.one_hot(IsAtomInRingOffive, num_classes=len(
                allowable_features['possible_is_in_ring5_list']))], dim=-1)

            IsAtomInRingOfsix = torch.tensor(
                [allowable_features['possible_is_in_ring6_list'].index(ringinfo.IsAtomInRingOfSize(idx, 6))])
            cur = torch.cat([cur, torch.nn.functional.one_hot(IsAtomInRingOfsix, num_classes=len(
                allowable_features['possible_is_in_ring6_list']))], dim=-1)

            IsAtomInRingOfseven = torch.tensor(
                [allowable_features['possible_is_in_ring7_list'].index(ringinfo.IsAtomInRingOfSize(idx, 7))])
            cur = torch.cat([cur, torch.nn.functional.one_hot(IsAtomInRingOfseven, num_classes=len(
                allowable_features['possible_is_in_ring7_list']))], dim=-1)

            IsAtomInRingOfeight = torch.tensor(
                [allowable_features['possible_is_in_ring8_list'].index(ringinfo.IsAtomInRingOfSize(idx, 8))])
            cur = torch.cat([cur, torch.nn.functional.one_hot(IsAtomInRingOfeight, num_classes=len(
                allowable_features['possible_is_in_ring8_list']))], dim=-1)
            atom_features = torch.cat([atom_features, cur], dim=0)

    return atom_features

def binarize(x):
    return torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))

#adj - > n_hops connections adj
#torch.eye 返回一个二维张量，其对角线为1，其他地方为0
def n_hops_adj(adj, n_hops):
    adj_mats = [torch.eye(adj.size(0), dtype=torch.long, device=adj.device), binarize(adj + torch.eye(adj.size(0), dtype=torch.long, device=adj.device))]

    for i in range(2, n_hops+1):
        adj_mats.append(binarize(adj_mats[i-1] @ adj_mats[1]))
    extend_mat = torch.zeros_like(adj)

    for i in range(1, n_hops+1):
        extend_mat += (adj_mats[i] - adj_mats[i-1]) * i

    return extend_mat

#LAS矩阵为1表示原子距离在2-hop之内或者在同一个环中，否则为0
def get_LAS_distance_constraint_mask(mol):
    # Get the adj
    h_index = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == 'H']
    for atom in reversed(h_index):
        mol = Chem.RWMol(mol)
        mol.RemoveAtom(atom)
    
    adj = Chem.GetAdjacencyMatrix(mol) 
    adj = torch.from_numpy(adj)
    extend_adj = n_hops_adj(adj,2)
    # add ring
    ssr = Chem.GetSymmSSSR(mol) #
    for ring in ssr:
        # print(ring)
        for i in ring:
            for j in ring:
                if i==j:
                    continue
                else:
                    extend_adj[i][j]+=1
    # turn to mask
    mol_mask = binarize(extend_adj)
    return mol_mask

def mol2graph(mol):
    try:
        #atoms
        # atom_features_list = []
        # for atom in mol.GetAtoms():
        #     atom_features_list.append(atom_to_feature_vector(atom))
        # x = np.array(atom_features_list, dtype=np.int64)

        #bonds
        num_bond_features = 3 #bond type, bond stereo, is_conjugated
        if len(mol.GetBonds()) > 0: #mol has bonds
            edges_list = []
            edge_features_list = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                edge_feature = bond_to_feature_vector(bond)

                # add edges in both directions
                edges_list.append((i, j))
                edge_features_list.append(edge_feature)
                edges_list.append((j, i))
                edge_features_list.append(edge_feature)

            # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
            edge_index = np.array(edges_list, dtype = np.int64).T

            # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
            edge_attr = np.array(edge_features_list, dtype = np.int64)
        else:   # mol has no bonds
            edge_index = np.empty((2, 0), dtype = np.int64)
            edge_attr = np.empty((0, num_bond_features), dtype = np.int64)
        return edge_index, edge_attr
    except:
        return None   

def get_mol_bond(mol):
    h_index = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == 'H']
    for atom in reversed(h_index):
        mol = Chem.RWMol(mol)
        mol.RemoveAtom(atom)
    edge_index, edge_attr = mol2graph(mol)
    edge_index = torch.from_numpy(edge_index).to(torch.int64)
    edge_attr = torch.from_numpy(edge_attr).to(torch.int64)
    N = len([atom for atom in mol.GetAtoms()])
    #返回小分子的边的信息

    atten_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
    atten_edge_type[edge_index[0, :], edge_index[1, :]] = edge_attr + 1

    graph_attn_bias = atten_edge_type
    return graph_attn_bias

























