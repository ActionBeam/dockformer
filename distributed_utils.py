import os

import torch
import torch.distributed as dist


def init_distributed_mode(args):    # 初始化多GPU进程的函数
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:  # 再环境变量中自动找到可使用的GPU
        args.rank = int(os.environ["RANK"])   # 代表所有进程中的第几个进程
        args.world_size = int(os.environ['WORLD_SIZE'])  # 对应了进程数，每个GPU对应一个进程（可以是多机多卡的情况）
        args.gpu = int(os.environ['LOCAL_RANK'])   # 代表当前机器中的第几个进程
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.device = 'cuda'
    args.distributed = True
    args.dist_url = "env://"   # 默认方式
    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'  # 通信后端，nvidia GPU推荐使用NCCL
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)
    dist.barrier()


def cleanup():
    dist.destroy_process_group()


def is_dist_avail_and_initialized():
    """检查是否支持分布式环境"""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def reduce_value(value, average=True):
    world_size = get_world_size()
    if world_size < 2:  # 单GPU的情况
        return value

    with torch.no_grad():
        dist.all_reduce(value)   # 求得所有设备中value的总和
        if average:
            value /= world_size   # 求多个设备中value的均值

        return value
