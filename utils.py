import torch
import numpy as np
import scipy
from rdkit.Chem import rdMolTransforms
from rdkit import Chem
from joblib import Parallel, delayed, cpu_count
from tqdm import tqdm
from scipy.spatial.transform import Rotation
import copy
from typing import Optional, Callable, List, Tuple, Sequence
import math
import torch.nn as nn
from scipy.stats import truncnorm
from torch.utils.data import DataLoader
from rdkit.Chem import AllChem
import random
from rdkit.Chem.rdMolAlign import AlignMolConformers
from sklearn.cluster import KMeans
from rdkit.Geometry import Point3D

import contextlib


@contextlib.contextmanager
def numpy_seed(seed, *addl_seeds):
    """Context manager which seeds the NumPy PRNG with the specified seed and
    restores the state afterward"""
    if seed is None:
        yield
        return
    if len(addl_seeds) > 0:
        seed = int(hash((seed, *addl_seeds)) % 1e6)
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def pmap_multi(pickleable_fn, data, n_jobs=None, verbose=1, desc=None, **kwargs):  # 用于并行处理任务
    """
    Parallel map using joblib.

    Parameters
    ----------
    pickleable_fn : callable
        Function to map over data.
    data : iterable
        Data over which we want to parallelize the function call.
    n_jobs : int, optional
        The maximum number of concurrently running jobs. By default, it is one less than
        the number of CPUs.
    verbose: int, optional
        The verbosity level. If nonzero, the function prints the progress messages.
        The frequency of the messages increases with the verbosity level. If above 10,
        it reports all iterations. If above 50, it sends the output to stdout.
    kwargs
        Additional arguments for :attr:`pickleable_fn`.

    Returns
    -------
    list
        The i-th element of the list corresponds to the output of applying
        :attr:`pickleable_fn` to :attr:`data[i]`.
    """
    if n_jobs is None:
        n_jobs = 1  # cpu_count() / 2
    # 结果以列表形式返回
    results = Parallel(n_jobs=n_jobs, verbose=verbose, timeout=None)(
        delayed(pickleable_fn)(*d, **kwargs) for i, d in tqdm(enumerate(data), desc=desc)
    )

    return results


def permute_final_dims(tensor: torch.Tensor, inds: List[int]):
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


def flatten_final_dims(t: torch.Tensor, no_dims: int):
    return t.reshape(t.shape[:-no_dims] + (-1,))


def get_pair_dis_one_hot(pair_dis, device, no_bin=64, min_bin=0, max_bin=30):
    pair_dis[pair_dis > max_bin] = max_bin
    pair_dis = pair_dis.unsqueeze(-1)  # 便于后续比较大小

    # 创建torch.linspace创建一个大小是步长的一维张量，其值从开始到结束是均匀分布的
    bins = torch.linspace(min_bin, max_bin, no_bin, dtype=torch.float, requires_grad=False).to(device)  # 下界
    upper = torch.cat([bins[1:], bins.new_tensor([np.inf])], dim=-1).to(device)  # 上界
    # 不太理解
    pair_dis = ((pair_dis >= bins) * (pair_dis < upper)).type(torch.float32)

    return pair_dis


def _rbf(D, D_min=0., D_max=20., D_count=16, device='cpu'):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design

    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    '''
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])

    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF


def get_mask(ligand_batch_num_nodes, receptor_batch_num_nodes):
    rows = ligand_batch_num_nodes.sum()
    cols = receptor_batch_num_nodes.sum()
    mask = torch.ones(rows, cols)  # 使用masked_fill函数时，1表示被mask，0表示保留，不是特别理解
    partial_l = 0
    partial_r = 0
    for l_n, r_n in zip(ligand_batch_num_nodes, receptor_batch_num_nodes):
        mask[partial_l: partial_l + l_n, partial_r: partial_r + r_n] = 0
        partial_l = partial_l + l_n
        partial_r = partial_r + r_n
    return mask


def get_LAS_dis(lig_coords, LAS_batch):
    num_nodes = lig_coords.shape[0]  # 一个batch中所有ligand原子的个数

    mask = torch.zeros(num_nodes, num_nodes, dtype=torch.int32)
    for idx, LAS_mask in enumerate(LAS_batch):  # 这里的0表示距离不固定，1表示距离固定
        LAS_mask = LAS_mask
        l_n = len(LAS_mask)
        mask[partial_l: partial_l + l_n, partial_l: partial_l + l_n] = LAS_mask
        partial_l = partial_l + l_n
    return mask


def get_Dihedral_atom(mol_list):
    atom_counter = 0
    torsionList = []
    dihedralList = []
    for m in mol_list:
        torsionSmarts = '[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]'  # 没查到是什么意思
        torsionQuery = Chem.MolFromSmarts(torsionSmarts)
        matches = m.GetSubstructMatches(torsionQuery)  # 获取对应所有子结构对相应的原子编号
        conf = m.GetConformer()  # 获得分子的构象
        for match in matches:
            idx2 = match[0]  # 原子的index
            idx3 = match[1]
            bond = m.GetBondBetweenAtoms(idx2, idx3)
            jAtom = m.GetAtomWithIdx(idx2)  # 返回一个特定的原子
            kAtom = m.GetAtomWithIdx(idx3)
            for b1 in jAtom.GetBonds():  # 返回原子键的只读序列
                if (b1.GetIdx() == bond.GetIdx()):  # GetIdx返回键的索引
                    continue
                idx1 = b1.GetOtherAtomIdx(idx2)  # 给定键中一个原子的index，返回另一个原子的index
                for b2 in kAtom.GetBonds():
                    if ((b2.GetIdx() == bond.GetIdx())
                            or (b2.GetIdx() == b1.GetIdx())):
                        continue
                    idx4 = b2.GetOtherAtomIdx(idx3)
                    # skip 3-membered rings
                    if (idx4 == idx1):
                        continue
                    # skip torsions that include hydrogens
                    #                     if ((m.GetAtomWithIdx(idx1).GetAtomicNum() == 1)
                    #                         or (m.GetAtomWithIdx(idx4).GetAtomicNum() == 1)):
                    #                         continue
                    if False:  # m.GetAtomWithIdx(idx4).IsInRing():  和Equibind不同，有一些不太理解
                        torsionList.append(
                            (idx4 + atom_counter, idx3 + atom_counter, idx2 + atom_counter, idx1 + atom_counter))
                        break
                    else:
                        torsionList.append(
                            (idx1 + atom_counter, idx2 + atom_counter, idx3 + atom_counter, idx4 + atom_counter))
                        break
                break

        atom_counter += m.GetNumAtoms()  # atom_counter的作用？
    return torsionList


def mol_with_atom_index(mol):
    atoms = mol.GetNumAtoms()
    for idx in range(atoms):
        mol.GetAtomWithIdx(idx).SetProp('molAtomMapNumber', str(mol.GetAtomWithIdx(idx).GetIdx()))
    return mol


def SetDihedral(conf, atom_idx, new_vale):
    rdMolTransforms.SetDihedralDeg(conf, atom_idx[0], atom_idx[1], atom_idx[2], atom_idx[3], new_vale)


def GetDihedral(conf, atom_idx):
    return rdMolTransforms.GetDihedralDeg(conf, atom_idx[0], atom_idx[1], atom_idx[2], atom_idx[3])


def GetTransformationMatrix(transformations):
    x, y, z, disp_x, disp_y, disp_z = transformations
    transMat = np.array([[np.cos(z) * np.cos(y), (np.cos(z) * np.sin(y) * np.sin(x)) - (np.sin(z) * np.cos(x)),
                          (np.cos(z) * np.sin(y) * np.cos(x)) + (np.sin(z) * np.sin(x)), disp_x],
                         [np.sin(z) * np.cos(y), (np.sin(z) * np.sin(y) * np.sin(x)) + (np.cos(z) * np.cos(x)),
                          (np.sin(z) * np.sin(y) * np.cos(x)) - (np.cos(z) * np.sin(x)), disp_y],
                         [-np.sin(y), np.cos(y) * np.sin(x), np.cos(y) * np.cos(x), disp_z],
                         [0, 0, 0, 1]], dtype=np.double)
    return transMat


def apply_changes(mol, values, rotable_bonds):
    opt_mol = copy.deepcopy(mol)
    #     opt_mol = add_rdkit_conformer(opt_mol)
    # apply rotations
    if torch.is_tensor(values):
        [SetDihedral(opt_mol.GetConformer(), rotable_bonds[r], values[r].item()) for r in range(len(rotable_bonds))]
    else:
        [SetDihedral(opt_mol.GetConformer(), rotable_bonds[r], values[r]) for r in range(len(rotable_bonds))]

    # # apply transformation matrix
    # rdMolTransforms.TransformConformer(opt_mol.GetConformer(), GetTransformationMatrix(values[:6]))

    return opt_mol


def get_torsion(lig):  # 返回弧度
    rotable_bonds = get_Dihedral_atom([lig])
    dihedral_radians = torch.zeros(len(rotable_bonds), dtype=torch.float32)
    for idx, r in enumerate(rotable_bonds):
        dihedral_radians[idx] = GetDihedral(lig.GetConformer(), r)
    return rotable_bonds, dihedral_radians


def get_position(lig):  # 返回分子中第一个原子的坐标
    conf = lig.GetConformer()
    position = torch.as_tensor(conf.GetPositions()[0], dtype=torch.float32)
    # print(position)
    return position


def random_rotation_translation(translation_distance=5, device='cpu'):
    rotation = Rotation.random(num=1)  # 随机生成一个旋转
    rotation_matrix = rotation.as_matrix().squeeze()  # 由于根据num数量生成，实际上输出的是一个列表，只是列表的元素只有一个，因此需要squeeze
    # [[a,b,c,d]] -> [a,b,c,d]
    t = np.random.randn(1, 3)  # 从标准正态分布中返回一个样本
    t = t / np.sqrt(np.sum(t * t))
    length = np.random.uniform(low=0, high=translation_distance)  # 从均匀分布中抽取样本
    t = t * length
    return torch.from_numpy(rotation_matrix.astype(np.float32)).to(device), torch.from_numpy(t.astype(np.float32)).to(
        device)


def get_loss_mask(ligand_batch_num_nodes, receptor_batch_num_nodes):
    rows = ligand_batch_num_nodes.sum()
    cols = receptor_batch_num_nodes.sum()
    mask = torch.zeros(rows, cols)
    partial_l = 0
    partial_r = 0
    for l_n, r_n in zip(ligand_batch_num_nodes, receptor_batch_num_nodes):
        mask[partial_l: partial_l + l_n, partial_r: partial_r + r_n] = 1  # 同一个复合物的原子间设置为1，表示有意义
        partial_l = partial_l + l_n
        partial_r = partial_r + r_n
    return mask


def list_detach(element):
    '''
    takes arbitrarily nested list and detaches everyting from computation graph
    :param element: arbitrarily nested list
    :return:
    '''
    if isinstance(element, list):
        return [list_detach(x) for x in element]
    else:
        return element.detach()


def get_conformer(lig, new_angle, torsion_atoms, position, orientation, pocket_coords, device):
    ori = lig.GetConformer().GetPositions()
    new_lig = apply_changes(lig, new_angle, torsion_atoms)  # 将torsion应用于结构中
    new_conf = new_lig.GetConformer()
    new_coords = new_conf.GetPositions()
    new_coords = torch.as_tensor(torch.from_numpy(new_coords), dtype=torch.float64).to(device)  # 获取配体原子的坐标
    pocket_coords = torch.from_numpy(pocket_coords).to(torch.float64).to(device)
    # lig_rand_coords = (orientation.to(device) @ new_coords.T).T + (pocket_coords.to(device) + position) # 随机配体原子的坐标
    lig_rand_coords = (orientation.to(device).to(torch.float64) @ new_coords.T).T  # 随机配体原子的坐标
    dif_one_to_pocket = pocket_coords - lig_rand_coords[0]  # 令第一个原子平移到pocket的位置
    # lig_rand_coords += (pocket_coords.to(device) + position)
    lig_rand_coords += (dif_one_to_pocket + position)
    for i in range(new_conf.GetPositions().shape[0]):  # 位置参数
        new_conf.SetAtomPosition(i, lig_rand_coords[i].tolist())  # 将随机化后的坐标放入随机构像的对象中
    return lig_rand_coords, new_lig


def quat_to_mat(quat: torch.tensor):
    quat = torch.nn.functional.normalize(quat, p=2, dim=0)  # 归一化
    # print(quat)
    quat = quat.tolist()
    x, y, z, w = quat  # 注意顺序，cos项被放在了最后，sin项在前三处

    rot_matrix00 = 1 - 2 * y * y - 2 * z * z
    rot_matrix01 = 2 * x * y - 2 * w * z
    rot_matrix02 = 2 * x * z + 2 * w * y
    rot_matrix10 = 2 * x * y + 2 * w * z
    rot_matrix11 = 1 - 2 * x * x - 2 * z * z
    rot_matrix12 = 2 * y * z - 2 * w * x
    rot_matrix20 = 2 * x * z - 2 * w * y
    rot_matrix21 = 2 * y * z + 2 * w * x
    rot_matrix22 = 1 - 2 * x * x - 2 * y * y

    return torch.tensor([
        [rot_matrix00, rot_matrix01, rot_matrix02],
        [rot_matrix10, rot_matrix11, rot_matrix12],
        [rot_matrix20, rot_matrix21, rot_matrix22]
    ], dtype=torch.float32)


def softmax_cross_entropy(logits, labels):  # 计算两个分布的交叉熵

    loss = -1 * torch.sum(
        labels * torch.nn.functional.log_softmax(logits, dim=-1),
        dim=-1,
    )
    return loss


def move_to_device(element, device):
    '''
    takes arbitrarily nested list and moves everything in it to device if it is a dgl graph or a torch tensor
    获取任意嵌套的列表，并将其中的所有内容移动到设备
    :param element: arbitrarily nested list
    :param device:
    :return:
    '''
    if isinstance(element, list):  # isinstance()检查对象类型
        return [move_to_device(x, device) for x in element]
    else:
        return element.to(device) if isinstance(element, torch.Tensor) else element


def read_strings_from_txt(path):
    # every line will be one element of the returned list
    with open(path) as file:
        lines = file.readlines()
        return [line.rstrip() for line in lines]


def _prod(nums):
    out = 1
    for n in nums:
        out = out * n
    return out


def _calculate_fan(linear_weight_shape, fan="fan_in"):
    fan_out, fan_in = linear_weight_shape

    if fan == "fan_in":
        f = fan_in
    elif fan == "fan_out":
        f = fan_out
    elif fan == "fan_avg":
        f = (fan_in + fan_out) / 2
    else:
        raise ValueError("Invalid fan option")

    return f


# 生成截断正态分布
def trunc_normal_init_(weights, scale=1.0, fan="fan_in"):
    shape = weights.shape
    f = _calculate_fan(shape, fan)
    scale = scale / max(1, f)
    a = -2
    b = 2
    std = math.sqrt(scale) / truncnorm.std(a=a, b=b, loc=0, scale=1)
    size = _prod(shape)
    samples = truncnorm.rvs(a=a, b=b, loc=0, scale=std, size=size)
    samples = np.reshape(samples, shape)
    with torch.no_grad():
        weights.copy_(torch.tensor(samples, device=weights.device))


def lecun_normal_init_(weights):
    trunc_normal_init_(weights, scale=1.0)


def he_normal_init_(weights):
    trunc_normal_init_(weights, scale=2.0)


def glorot_uniform_init_(weights):
    nn.init.xavier_uniform_(weights, gain=1)


def final_init_(weights):
    with torch.no_grad():
        weights.fill_(0.0)


def gating_init_(weights):
    with torch.no_grad():
        weights.fill_(0.0)


def normal_init_(weights):
    torch.nn.init.kaiming_normal_(weights, nonlinearity="linear")


def ipa_point_weights_init_(weights):
    with torch.no_grad():
        softplus_inverse_1 = 0.541324854612918
        weights.fill_(softplus_inverse_1)


class Linear(nn.Linear):  # 继承了nn.Linear类
    """
    A Linear layer with built-in nonstandard initializations. Called just
    like torch.nn.Linear.

    Implements the initializers in 1.11.4, plus some additional ones found
    in the code.
    """

    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            bias: bool = True,
            init: str = "default",
            init_fn: Optional[Callable[[torch.Tensor, torch.Tensor], None]] = None,
    ):
        """
        Args:
            in_dim:
                The final dimension of inputs to the layer
            out_dim:
                The final dimension of layer outputs
            bias:
                Whether to learn an additive bias. True by default
            init:
                The initializer to use. Choose from:

                "default": LeCun fan-in truncated normal initialization
                "relu": He initialization w/ truncated normal distribution
                "glorot": Fan-average Glorot uniform initialization
                "gating": Weights=0, Bias=1
                "normal": Normal initialization with std=1/sqrt(fan_in)
                "final": Weights=0, Bias=0

                Overridden by init_fn if the latter is not None.
            init_fn:
                A custom initializer taking weight and bias as inputs.
                Overrides init if not None.
        """
        super(Linear, self).__init__(in_dim, out_dim, bias=bias)

        if bias:
            with torch.no_grad():
                self.bias.fill_(0)

        if init_fn is not None:
            init_fn(self.weight, self.bias)
        else:
            if init == "default":
                lecun_normal_init_(self.weight)
            elif init == "relu":
                he_normal_init_(self.weight)
            elif init == "glorot":
                glorot_uniform_init_(self.weight)
            elif init == "gating":
                gating_init_(self.weight)
                if bias:
                    with torch.no_grad():
                        self.bias.fill_(1.0)
            elif init == "normal":
                normal_init_(self.weight)
            elif init == "final":
                final_init_(self.weight)
            else:
                raise ValueError("Invalid init string.")


def concat_if_list(tensor_or_tensors):
    return torch.cat(tensor_or_tensors) if isinstance(tensor_or_tensors, list) else tensor_or_tensors


def adj_to_bias(adj, sizes, nhood=1):
    nb_graphs = adj.shape[0]  # 图的个数
    mt = np.empty(adj.shape)  # 新建一个空的邻接矩阵
    for g in range(nb_graphs):
        mt[g] = np.eye(adj.shape[1])
        for _ in range(nhood):
            mt[g] = np.matmul(mt[g], (adj[g] + np.eye(adj.shape[1])))
        for i in range(sizes[g]):
            for j in range(sizes[g]):
                if mt[g][i][j] > 0.0:
                    mt[g][i][j] = 1.0
    return -1e9 * (1.0 - mt)


def get_LAS_distance_constraint_mask(mol):
    # Get the adj
    adj = [Chem.GetAdjacencyMatrix(mol)]
    num_node = [adj[0].shape[0]]
    adj = adj_to_bias(adj)
    adj = torch.from_numpy(adj, num_node, 2)
    return adj


def single_conf_gen(tgt_mol, num_confs=1, seed=42, removeHs=True):
    mol = copy.deepcopy(tgt_mol)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    allconformers = AllChem.EmbedMultipleConfs(
        mol, numConfs=num_confs, randomSeed=seed, clearConfs=True
    )
    sz = len(allconformers)
    for i in range(sz):
        try:
            AllChem.MMFFOptimizeMolecule(mol, confId=i)
        except:
            continue
    if removeHs:
        mol = Chem.RemoveHs(mol)
    return mol


def clustering_coords(mol, M=1000, N=100, seed=42, removeHs=True):
    rdkit_coords_list = []
    rdkit_mol = single_conf_gen(mol, num_confs=M, seed=seed, removeHs=removeHs)
    noHsIds = [
        rdkit_mol.GetAtoms()[i].GetIdx()
        for i in range(len(rdkit_mol.GetAtoms()))
        if rdkit_mol.GetAtoms()[i].GetAtomicNum() != 1
    ]
    ### exclude hydrogens for aligning
    AlignMolConformers(rdkit_mol, atomIds=noHsIds)
    sz = len(rdkit_mol.GetConformers())
    for i in range(sz):
        _coords = rdkit_mol.GetConformers()[i].GetPositions().astype(np.float32)
        rdkit_coords_list.append(_coords)

    ### exclude hydrogens for clustering
    rdkit_coords_flatten = np.array(rdkit_coords_list)[:, noHsIds].reshape(sz, -1)
    ids = (
        KMeans(n_clusters=N, random_state=seed, n_init=10)
        .fit_predict(rdkit_coords_flatten)
        .tolist()
    )
    coords_list = [rdkit_coords_list[ids.index(i)] for i in range(N) if i in ids]
    return coords_list


def seed_all(seed):
    if not seed:
        seed = 0

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def Remove_h(atom_list, coord_list):
    masked = atom_list != 'H'
    atom_list = atom_list[masked]
    coord_list = coord_list[:, masked, :]
    return coord_list


def write_with_new_coords(mol, new_coords, toFile):
    # mol = Chem.RemoveHs(mol)
    conf = mol.GetConformer()
    # for i in range(mol.GetNumAtoms()):
    for i in range(new_coords.size(0)):
        x, y, z = new_coords[i]
        # print(x, y, z)
        x = float(x.item())
        y = float(y.item())
        z = float(z.item())
        # print(x, y, z)
        conf.SetAtomPosition(i, Point3D(x, y, z))
    Chem.MolToMolFile(mol, toFile)
