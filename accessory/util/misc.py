# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import builtins
import datetime
import os
import time
from collections import defaultdict, deque
from pathlib import Path
import subprocess

import torch
import torch.distributed as dist
from torch import inf
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,
)

from fairscale.nn.model_parallel import initialize as fs_init

from .clip_grad import clip_grad_norm

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None, start_iter=0):
        i = start_iter
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        log_msg = [
            header,
            '[{0' + '}/{1}]',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0:
                try:
                    total_len = len(iterable)
                except:
                    total_len = "unknown"
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, total_len,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, total_len,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {}'.format(
            header, total_time_str))


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
#        force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def is_dist_avail_and_initialized():
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


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)

def init_distributed_mode(args):
    if args.dist_on_itp:
        args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        os.environ['MASTER_PORT'] = '8964'
        while 'MASTER_ADDR' not in os.environ or len(os.environ['MASTER_ADDR'].strip()) == 0:
            os.environ['MASTER_ADDR'] = subprocess.check_output('sinfo -Nh -n %s | head -n 1 | awk \'{print $1}\'' % os.environ['SLURM_NODELIST'], shell=True, ).decode().strip()
            time.sleep(1)
        print(os.environ['MASTER_ADDR'])
        args.world_size = int(os.environ['SLURM_NPROCS'])
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
        args.local_rank = args.gpu
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['WORLD_SIZE'] = str(args.world_size)
        os.environ['RANK'] = str(args.rank)
    else:
        print('Not using distributed mode')
        setup_for_distributed(is_master=True)  # hack
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def init_distributed_mode1(args):
    if args.dist_on_itp:
        args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        setup_for_distributed(is_master=True)  # hack
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self, args):
        self._scaler = ShardedGradScaler(enabled=args.precision in ["fp16"])

    def __call__(self, loss, optimizer, model: FSDP, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        if update_grad:
            self._scaler.scale(loss).backward(create_graph=create_graph)
            if clip_grad is None:
                clip_grad = 1e8 # large enough to disable clip grad
            self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
            norm = clip_grad_norm(model, clip_grad)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            with model.no_sync():
                self._scaler.scale(loss).backward(create_graph=create_graph)
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)



def save_checkpoint(output_dir, args, model, optimizer, loss_scaler, dataset_state, epoch=None, iteration=None):
    save_name = f"epoch{epoch}"
    if iteration is not None:
        save_name += f"-iter{iteration}"
    save_dir = os.path.join(output_dir, save_name)

    mp_rank = fs_init.get_model_parallel_rank()
    mp_world_size = fs_init.get_model_parallel_world_size()

    os.makedirs(save_dir, exist_ok=True)
    with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
    ):
        # run saving in separate functions to save memory
        def _save_model():
            save_dtype = {
                "fp16": torch.float16,
                "bf16": torch.bfloat16,
                "tf32": torch.float,
            }[args.precision]
            if getattr(args, "only_save_trainable", False):
                model_trainable_params = model.get_trainable_params()
                model_trainable_params = ['.'.join([_ for _ in key.split('.') if not _.startswith('_')])
                                          for key in model_trainable_params.keys()]
                consolidated_model_state_dict = {
                    "model": {key: val.to(save_dtype) for key, val in model.state_dict().items() if key in model_trainable_params},
                }
            else:
                consolidated_model_state_dict = {
                    "model": {key: val.to(save_dtype) for key, val in model.state_dict().items()},
                }

            save_path = os.path.join(
                save_dir,
                f"consolidated.{mp_rank:02d}-of-{mp_world_size:02d}.model.pth",
            )
            if fs_init.get_data_parallel_rank() == 0:
                torch.save(consolidated_model_state_dict, save_path)
        _save_model()
        print("model saved")

        def _save_optimizer():
            # torch.FSDP has a bug that passing dp_group to FSDP.full_optim_state_dict still calls dist.gather within the whole world
            _world = torch.distributed.GroupMember.WORLD
            torch.distributed.GroupMember.WORLD = fs_init.get_data_parallel_group()

            consolidated_optim_state_dict = {
                "optimizer": FSDP.full_optim_state_dict(model, optimizer)
            }
            save_path = os.path.join(
                save_dir,
                f"consolidated.{mp_rank:02d}-of-{mp_world_size:02d}.optimizer.pth",
            )
            if fs_init.get_data_parallel_rank() == 0:
                torch.save(consolidated_optim_state_dict , save_path)

            torch.distributed.GroupMember.WORLD = _world
        _save_optimizer()
        print("optimizer saved")

        def _save_other():
            consolidated_other_state_dict = {
                "epoch": epoch,
                "iter": iteration,
                "scaler": loss_scaler.state_dict() if hasattr(loss_scaler, 'state_dict') else loss_scaler,
                "args": args,
            }
            save_path = os.path.join(
                save_dir,
                f"consolidated.{mp_rank:02d}-of-{mp_world_size:02d}.other.pth",
            )
            if fs_init.get_data_parallel_rank() == 0:
                torch.save(consolidated_other_state_dict, save_path)
        _save_other()
        print("other rank-common saved")


    def _save_rank_specific():
        rank_specific_state_dict = {
            "dataset_state": dataset_state,
        }
        save_path = os.path.join(
                save_dir,
                f"rank-specific-{dist.get_rank():05d}-of-{dist.get_world_size():05d}.pth",
            )
        torch.save(rank_specific_state_dict, save_path)

    _save_rank_specific()
    print("rank-specific saved")


def resume_stage1(args, model_without_FSDP):
    """
    split resume into two separate stages since resuming from a full model state has to be done before FSDP model init
    :param args:
    :param model_without_FSDP:
    :return:
    """
    if args.resume:
        print("Resume checkpoint %s" % args.resume)

        mp_rank = fs_init.get_model_parallel_rank()
        mp_world_size = fs_init.get_model_parallel_world_size()
        dp_rank = fs_init.get_data_parallel_rank()

        if dp_rank == 0:
            consilidated_model_checkpoint_path = os.path.join(
                args.resume,
                f"consolidated.{mp_rank:02d}-of-{mp_world_size:02d}.model.pth",
            )

            consolidated_model_state_dict = torch.load(consilidated_model_checkpoint_path, map_location='cpu')
            model_without_FSDP.load_state_dict(consolidated_model_state_dict['model'], strict=False)
            print(f"load model from {consolidated_model_state_dict}")


def resume_stage2(args, model, optimizer, loss_scaler, dataset_train):
    if args.resume:
        print("Resume checkpoint %s" % args.resume)

        mp_rank = fs_init.get_model_parallel_rank()
        mp_world_size = fs_init.get_model_parallel_world_size()
        dp_rank = fs_init.get_data_parallel_rank()


        def _load_optimizer():
            # if dp_rank == 0:
            consilidated_optimizer_checkpoint_path = os.path.join(
                args.resume,
                f"consolidated.{mp_rank:02d}-of-{mp_world_size:02d}.optimizer.pth",
            )
            full_osd = torch.load(consilidated_optimizer_checkpoint_path)['optimizer']
            # else:
            #     full_osd = None

            sharded_osd = FSDP.shard_full_optim_state_dict(full_osd, model, optim=optimizer)
            optimizer.load_state_dict(sharded_osd)
            print(f"load optimizer from {consilidated_optimizer_checkpoint_path}")
        _load_optimizer()

        def _load_other():
            consilidated_other_checkpoint_path = os.path.join(
                args.resume,
                f"consolidated.{mp_rank:02d}-of-{mp_world_size:02d}.other.pth",
            )
            other_state_dict = torch.load(consilidated_other_checkpoint_path)
            loss_scaler.load_state_dict(other_state_dict['scaler'])

            _epoch_iter = [
                int(other_state_dict['epoch']) + 1 if other_state_dict.get('epoch', None) is not None else None,
                int(other_state_dict['iter']) + 1 if other_state_dict.get('iter', None) is not None else None
            ]
            print(f"load other from {consilidated_other_checkpoint_path}")
            print(f"loaded epoch & iter: {_epoch_iter}")
            return _epoch_iter
        epoch_iter = _load_other()


        def _load_rank_specific():
            rank_specific_checkpoint_path = os.path.join(
                args.resume,
                f"rank-specific-{dist.get_rank():05d}-of-{dist.get_world_size():05d}.pth",
            )
            try:
                rank_specific_state_dict = torch.load(rank_specific_checkpoint_path)
                dataset_train.load_state_dict(rank_specific_state_dict["dataset_state"])
            except Exception:
                print(f"dataset state loading failed, either because dataset has not attribute 'load_state_dict',"
                      f" or {rank_specific_checkpoint_path} does not exist")
                print(f"This is okay if the dataset has no state dict")



        _load_rank_specific()

        return epoch_iter


def load_pretrained(load_dir, pretrained_type, model):
    mp_rank = fs_init.get_model_parallel_rank()
    mp_world_size = fs_init.get_model_parallel_world_size()

    dp_rank = fs_init.get_data_parallel_rank()
    if dp_rank == 0: # later broadcast to all ranks through FSDP init
        if pretrained_type == "consolidated":
            candidate_names = [
                os.path.join(load_dir, f"consolidated.{mp_rank:02d}-of-{mp_world_size:02d}.pth"),
                os.path.join(load_dir, f"consolidated.{mp_rank:02d}-of-{mp_world_size:02d}.model.pth")
            ]
            state_dict_path = None
            for _ in candidate_names:
                if os.path.exists(_):
                    state_dict_path = _
                    break
            if state_dict_path is None:
                raise FileNotFoundError(f"none of {candidate_names} exist")
            state_dict = torch.load(state_dict_path, map_location='cpu')
            if 'model' in state_dict:
                state_dict = state_dict['model']
            load_result = model.load_state_dict(state_dict, strict=False)
        elif pretrained_type == "meta_ori":
            ckpt_mp_world_size = len([
                path for path in os.listdir(load_dir)
                if path.startswith("consolidated.") and path.endswith(".pth")
            ])
            assert ckpt_mp_world_size == mp_world_size, (
                "Loading from checkpoints of different mp_world_size is currently not supported."
            )
            state_dict_path = os.path.join(load_dir, f"consolidated.{mp_rank:02d}.pth")
            state_dict = torch.load(state_dict_path, map_location='cpu')
            model_state = {f"llma.{key}": val for key, val in state_dict.items()}
            load_result = model.load_state_dict(model_state, strict=False)
        else:
            raise ValueError(f"Unknown pretrained_type: {pretrained_type}")
        print('load pretrained result:\n', load_result)



def all_reduce_mean(x):
    world_size = get_world_size()
    if world_size > 1:
        if isinstance(x, torch.Tensor):
            x_reduce = x.clone().cuda()
        else:
            x_reduce = torch.tensor(x).cuda()
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    else:
        return x


def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        #if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
        if name.endswith(".bias") or name.endswith("norm.weight"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


def broadcast_nonmp_parameters(model):
    if fs_init.get_model_parallel_world_size() == 1:
        return
    print("starting broadcast non-model-parallel parameters within model parallel group")
    memo = set()
    modules = model.named_modules(prefix='', remove_duplicate=True)
    for module_prefix, module in modules:
        members = dict(module._parameters.items())
        for k, v in members.items():
            name = module_prefix + ('.' if module_prefix else '') + k
            if v is None or v in memo:
                continue
            if getattr(v, "is_model_parallel", False):
                print(f"ignore: {name}")
                continue
            memo.add(v)
            dist.broadcast(v, src=fs_init.get_model_parallel_src_rank(), group=fs_init.get_model_parallel_group())
    print("braodcast done")


def mark_mp_params(model: torch.nn.Module):
    from fairscale.nn.model_parallel.layers import (
        RowParallelLinear,
        ColumnParallelLinear,
        ParallelEmbedding,
    )
    for m in model.modules():
        if isinstance(m, ColumnParallelLinear):
            m.weight.is_model_parallel = True
            if m.bias is not None:
                m.bias.is_model_parallel = True

        if isinstance(m, RowParallelLinear):
            m.weight.is_model_parallel = True

        if isinstance(m, ParallelEmbedding):
            m.weight.is_model_parallel = True


def print_param_status(model: torch.nn.Module) -> None:
    for name, param in model.named_parameters():
        is_model_parallel = getattr(param, "is_model_parallel", False)
        print(f"Param {name}: requires_grad {param.requires_grad}, local_size {param.shape}, model_parallel {is_model_parallel}, dtype {param.dtype}")

