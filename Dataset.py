from torch.utils.data import Dataset
from joblib.externals.loky import get_reusable_executor
from joblib import cpu_count
import os
import torch
from tqdm import tqdm
import random
from utils import pmap_multi, read_strings_from_txt, Remove_h, numpy_seed, random_rotation_translation
import copy
from rdkit import Chem
import numpy as np
import matplotlib.pyplot as plt
from rdkit.Chem import AllChem
import lmdb

import pickle
import logging

import numpy as np

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Dictionary:
    """A mapping from symbols to consecutive integers"""

    def __init__(
            self,
            *,  # begin keyword-only arguments
            bos="[CLS]",
            pad="[PAD]",
            eos="[SEP]",
            unk="[UNK]",
            extra_special_symbols=None,
    ):
        self.bos_word, self.unk_word, self.pad_word, self.eos_word = bos, unk, pad, eos
        self.symbols = []
        self.count = []
        self.indices = {}
        self.specials = set()
        self.specials.add(bos)
        self.specials.add(unk)
        self.specials.add(pad)
        self.specials.add(eos)

    def __eq__(self, other):
        return self.indices == other.indices

    def __getitem__(self, idx):
        if idx < len(self.symbols):
            return self.symbols[idx]
        return self.unk_word

    def __len__(self):
        """Returns the number of symbols in the dictionary"""
        return len(self.symbols)

    def __contains__(self, sym):
        return sym in self.indices

    def vec_index(self, a):
        return np.vectorize(self.index)(a)

    def index(self, sym):
        """Returns the index of the specified symbol"""
        assert isinstance(sym, str)
        if sym in self.indices:
            return self.indices[sym]
        return self.indices[self.unk_word]

    def special_index(self):
        return [self.index(x) for x in self.specials]

    def add_symbol(self, word, n=1, overwrite=False, is_special=False):
        """Adds a word to the dictionary"""
        if is_special:
            self.specials.add(word)
        if word in self.indices and not overwrite:
            idx = self.indices[word]
            self.count[idx] = self.count[idx] + n
            return idx
        else:
            idx = len(self.symbols)
            self.indices[word] = idx
            self.symbols.append(word)
            self.count.append(n)
            return idx

    def bos(self):
        """Helper to get index of beginning-of-sentence symbol"""
        return self.index(self.bos_word)

    def pad(self):
        """Helper to get index of pad symbol"""
        return self.index(self.pad_word)

    def eos(self):
        """Helper to get index of end-of-sentence symbol"""
        return self.index(self.eos_word)

    def unk(self):
        """Helper to get index of unk symbol"""
        return self.index(self.unk_word)

    @classmethod
    def load(cls, f):
        """Loads the dictionary from a text file with the format:

        ```
        <symbol0> <count0>
        <symbol1> <count1>
        ...
        ```
        """
        d = cls()
        d.add_from_file(f)
        return d

    def add_from_file(self, f):
        """
        Loads a pre-existing dictionary from a text file and adds its symbols
        to this instance.
        """
        if isinstance(f, str):
            try:
                with open(f, "r", encoding="utf-8") as fd:
                    self.add_from_file(fd)
            except FileNotFoundError as fnfe:
                raise fnfe
            except UnicodeError:
                raise Exception(
                    "Incorrect encoding detected in {}, please "
                    "rebuild the dataset".format(f)
                )
            return

        lines = f.readlines()

        for line_idx, line in enumerate(lines):
            try:
                splits = line.rstrip().rsplit(" ", 1)
                line = splits[0]
                field = splits[1] if len(splits) > 1 else str(len(lines) - line_idx)
                if field == "#overwrite":
                    overwrite = True
                    line, field = line.rsplit(" ", 1)
                else:
                    overwrite = False
                count = int(field)
                word = line
                if word in self and not overwrite:
                    logger.info(
                        "Duplicate word found when loading Dictionary: '{}', index is {}.".format(word,
                                                                                                  self.indices[word])
                    )
                else:
                    self.add_symbol(word, n=count, overwrite=overwrite)
            except ValueError:
                raise ValueError(
                    "Incorrect dictionary format, expected '<token> <cnt> [flags]'"
                )


class PDBBind(Dataset):
    def __init__(self, datatype, rank,
                 remove_h=True,
                 seed=42,
                 **kwargs):
        # 训练/验证/测试
        self.datatype = datatype
        self.remove_h = remove_h
        self.device = torch.device("cuda", rank) if torch.cuda.is_available() else torch.device("cpu")
        self.max_atoms = 256
        self.seed = seed
        self.conf_size = 10
        if datatype == 'pose_buster':
            self.data_path = os.path.join("./lmdb/posebuster_428.lmdb")
        elif datatype == 'astex':
            self.data_path = os.path.join("./lmdb/Astex.lmdb")
        elif datatype == 'DockGen':
            self.data_path = os.path.join("./lmdb/DockGen.lmdb")    
        else:
            self.data_path = os.path.join("./lmdb", self.datatype + ".lmdb")

        self.mol_dictionary = Dictionary.load(os.path.join("./lmdb", "dict_mol.txt"))
        self.pocket_dictionary = Dictionary.load(os.path.join("./lmdb", "dict_pkt.txt"))
        self.max_seq_len = 512
        self.total_lmdb_data = self.read_lmdb(self.data_path)
        # 删除total_lmdb_data中的3个无效数据
        if self.datatype == 'train':
            self.total_lmdb_data = [data for data in self.total_lmdb_data if
                                    data['pocket'] not in ['3fxz', '3zp9', '3kck']]
        if self.datatype == 'pose_buster':
            self.processed_dir = f'./lmdb'
            self.total_mol_data = torch.load(os.path.join(self.processed_dir, 'posebuster_mol_data.pt'))
        elif self.datatype == 'astex':
            self.processed_dir = f'./lmdb'
        elif self.datatype == 'DockGen':
            self.processed_dir = f'./lmdb'        
            self.total_mol_data = self.read_lmdb(os.path.join(self.processed_dir, 'DockGen_mol_data.lmdb'))
        else:
            self.processed_dir = f'./lmdb/'
            self.data_root_path = "./lmdb/"
            self.total_mol_data = torch.load(os.path.join(self.processed_dir, 'test_mol_data.pt'))

    def read_lmdb(self, db_path):
        all_data = []
        assert os.path.isfile(db_path), "{} not found".format(db_path)
        env = lmdb.open(
            db_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=256,
        )

        txn = env.begin()
        keys = list(txn.cursor().iternext(values=False))
        for idx in keys:
            datapoint_pickled = txn.get(idx)
            data = pickle.loads(datapoint_pickled)
            all_data.append(data)
        return all_data

    def RemoveHydrogen(self, atoms):
        if self.remove_h:
            mask_hydrogen = atoms != 'H'
        return mask_hydrogen

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        # return len(self.total_lmdb_data) * self.conf_size
        if self.datatype == "train":
            return len(self.total_lmdb_data)
        else:
            return len(self.total_lmdb_data) * self.conf_size

    def __getitem__(self, idx):
        # # 处理ligand相关的随机性
        # smi_idx = idx // self.conf_size
        # coord_idx = idx % self.conf_size
        # self.lmdb_data = self.total_lmdb_data[smi_idx]
        # mol_data = self.total_mol_data[smi_idx]
        # self.lig_coords = self.lmdb_data["coordinates"][coord_idx].astype(np.float32)
        # 处理ligand相关的随机性
        if self.datatype == "train":
            # 读取lmdb文件中的数据
            self.lmdb_data = self.total_lmdb_data[idx]
            mol_data = self.total_mol_data[idx]
            size = len(self.lmdb_data["coordinates"])
            with numpy_seed(self.seed, self.epoch, idx):
                sample_index = np.random.randint(size)
            self.lig_coords = self.lmdb_data["coordinates"][sample_index].astype(np.float32)
        else:
            smi_idx = idx // self.conf_size
            coord_idx = idx % self.conf_size
            self.lmdb_data = self.total_lmdb_data[smi_idx]
            mol_data = self.total_mol_data[smi_idx]
            self.lig_coords = self.lmdb_data["coordinates"][coord_idx].astype(np.float32)

        # tmp = set(self.lmdb_data['residue'])
        self.lig_conformer = self.lmdb_data["mol_list"][0]
        self.lig_name = self.lmdb_data["pocket"]

        # 配体和蛋白质的原子
        self.lig_atom = np.array(self.lmdb_data["atoms"])
        self.rec_atom = np.array([item[0] for item in self.lmdb_data["pocket_atoms"]])

        # 配体和蛋白质的坐标
        self.target_lig_coords = self.lmdb_data["holo_coordinates"][0].astype(np.float32)
        self.rec_coords = self.lmdb_data["pocket_coordinates"][0].astype(np.float32)

        # 去除口袋原子中的氢原子
        mask_hydrogen = self.RemoveHydrogen(self.rec_atom)
        self.rec_atom = self.rec_atom[mask_hydrogen]
        self.rec_coords = self.rec_coords[mask_hydrogen]

        # 去除配体原子中的氢原子
        mask_hydrogen = self.RemoveHydrogen(self.lig_atom)
        self.lig_atom = self.lig_atom[mask_hydrogen]
        self.lig_coords = self.lig_coords[mask_hydrogen]
        self.target_lig_coords = torch.from_numpy(self.target_lig_coords[mask_hydrogen])

        # 处理protein原子，每次取其中的256个原子
        if self.max_atoms and len(self.rec_coords) > self.max_atoms:
            with numpy_seed(self.seed, self.epoch):
                distance = np.linalg.norm(
                    self.rec_coords - self.rec_coords.mean(axis=0), axis=1
                )

                def softmax(x):
                    x -= np.max(x)
                    x = np.exp(x) / np.sum(np.exp(x))
                    return x

                distance += 1  # prevent inf
                weight = softmax(np.reciprocal(distance))
                index = np.random.choice(
                    len(self.rec_coords), self.max_atoms, replace=False, p=weight
                )
            self.rec_coords = self.rec_coords[index]
            self.rec_atom = self.rec_atom[index]

        # 规范配体随机构象原子的坐标和口袋原子的坐标
        # lig_coords = lig_coords - lig_coords.mean(axis=0）
        # rec_coords = rec_coords - rec_coords.meand(axis=0)

        # 配体原子特征
        assert len(self.lig_atom) < self.max_seq_len and len(self.lig_atom) > 0
        self.lig_feature = torch.from_numpy(self.mol_dictionary.vec_index(self.lig_atom)).long()
        # 填充开始和结束符
        # lig_feature = torch.cat([torch.full_like(lig_feature[0], self.mol_dictionary.bos()).unsqueeze(0), lig_feature], dim=0)
        # lig_feature = torch.cat([lig_feature, torch.full_like(lig_feature[0], self.mol_dictionary.eos()).unsqueeze(0)], dim=0)

        self.lig_edge_type = self.lig_feature.view(-1, 1) * len(self.mol_dictionary) + self.lig_feature.view(1, -1)
        self.lig_coords = torch.from_numpy(self.lig_coords)

        # 受体原子特征
        assert len(self.rec_atom) < self.max_seq_len and len(self.rec_atom) > 0
        self.rec_feature = torch.from_numpy(self.pocket_dictionary.vec_index(self.rec_atom)).long()
        # 填充开始和结束符
        # rec_feature = torch.cat([torch.full_like(rec_feature[0], self.pocket_dictionary.bos()).unsqueeze(0), rec_feature], dim=0)
        # rec_feature = torch.cat([rec_feature, torch.full_like(rec_feature[0], self.pocket_dictionary.eos()).unsqueeze(0)], dim=0)

        self.rec_edge_type = self.rec_feature.view(-1, 1) * len(self.pocket_dictionary) + self.rec_feature.view(1, -1)
        self.rec_coords = torch.from_numpy(self.rec_coords)

        self.lig_spatial_pos = mol_data['spatial_pos']
        self.lig_edge_input = mol_data['edge_input']

        return self.lig_conformer, \
               self.lig_name, \
               self.lig_coords, \
               self.lig_feature, \
               self.lig_edge_type, \
               self.rec_coords, \
               self.rec_feature, \
               self.rec_edge_type, \
               self.target_lig_coords, \
               idx, \
               self.lig_spatial_pos, \
               self.lig_edge_input


# train_names = "test"
# traindata = PDBBind(datatype=train_names, rank = 2, seed = 42)
# traindata.set_epoch(0)
# print(traindata[1])

def collate_revised(batch):  # 将各个Batch中的数据分别整合在对应的list中，list的长度就是batch的size
    lig_batch, \
    lig_name_batch, \
    lig_coords_batch, \
    lig_feature_batch, \
    lig_edge_type_batch, \
    rec_coords_batch, \
    rec_feature_batch, \
    rec_edge_type_batch, \
    target_lig_coords_batch, \
    idx_batch, \
    lig_spatial_pos_batch, \
    lig_edge_input_batch \
        = map(list, zip(*batch))

    # target_lig_coords_batch = []  # 获取batch内所有的配体原子坐标做为target
    # for lig in lig_batch:
    #     #删除显示表示的氢原子，不确定是不是合适
    #     h_index = [atom.GetIdx() for atom in lig.GetAtoms() if atom.GetSymbol() == 'H']
    #     for atom in reversed(h_index):
    #         lig = Chem.RWMol(lig)
    #         lig.RemoveAtom(atom)

    #     conf = lig.GetConformer()
    #     coords = conf.GetPositions()
    #     target_lig_coords_batch.append(torch.from_numpy(coords))

    # 归一化随机构象的坐标以及蛋白质原子的坐标
    # for i, coord in enumerate(lig_coords_batch):
    #     rec_center_coordinates = rec_coords_batch[i].mean(axis=0)
    #     lig_center_coordinates = lig_coords_batch[i].mean(axis=0)
    #     lig_coords_batch[i] -= lig_center_coordinates
    #     rec_coords_batch[i] -= rec_center_coordinates
    #     target_lig_coords_batch[i] -= rec_center_coordinates

    return lig_batch, lig_name_batch, lig_coords_batch, \
           lig_feature_batch, lig_edge_type_batch, rec_coords_batch, rec_feature_batch, rec_edge_type_batch, \
           target_lig_coords_batch, idx_batch, lig_spatial_pos_batch, lig_edge_input_batch
