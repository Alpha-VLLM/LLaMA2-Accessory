import sys
import os
sys.path.append(os.path.abspath(__file__).rsplit('/', 2)[0])

import argparse
import datetime
import warnings

import numpy as np
import time
from pathlib import Path
import functools
from functools import partial

import torch
from torch.utils.data import Dataset
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    CPUOffload
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
)

from fairscale.nn.model_parallel import initialize as fs_init

try:
    from apex.optimizers import FusedAdam as AdamW
except ImportError:
    warnings.warn("cannot import FusedAdam from apex, use torch AdamW instead")
    from torch.optim import AdamW

import accessory.util.misc as misc
from accessory.util.misc import NativeScalerWithGradNormCount as NativeScaler
from accessory.util.tensor_type import default_tensor_type, promote_trainable_params_to_fp32
from accessory.util.tensor_parallel import load_tensor_parallel_model_list
from accessory.model.meta import MetaModel
from accessory.engine_pretrain import train_one_epoch
from accessory.data import falcon, falcon_packed


def get_args_parser():
    parser = argparse.ArgumentParser('LLaMA2-Accessory pretraining', add_help=False)
    parser.add_argument('--batch_size', default=4, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--accum_iter', default=4, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--llama_type', default='llama', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--llama_config', default=[], nargs="*",
                        help='Path to llama model config')
    parser.add_argument('--tokenizer_path', type=str, default="../tokenizer.model",
                        help='path to tokenizer.model')


    parser.add_argument('--pretrained_path', default=None, type=str, nargs="*",
                        help='(Optional) directory containing checkpoints to start from')
    parser.add_argument('--pretrained_type', type=str, default=None, choices=['consolidated', 'meta_ori'],
                        help='<Deprecated> pretrained checkpoint save format, will be automatically discerned now')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.02,
                        help='weight decay (default: 0.02)')

    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--min_lr', type=float, default=0.0001, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_iters', type=int, default=20000, metavar='N',
                        help='iterations to warmup LR')
    parser.add_argument('--lr_decay_iters', type=int, default=1800000, metavar='N',
                        help='iters before keeping minimal learning rate')

    parser.add_argument('--clip_grad', type=int, default=-1,
                        help='grad clipping norm')

    # Dataset parameters
    parser.add_argument('--data_meta_path', default='/path/to/data/meta/file', type=str,
                        help='path to data meta file')
    parser.add_argument('--data_root', default=None, type=str,
                        help='root path for data')
    parser.add_argument('--packed_data', action="store_true",
                        help='use packed dataset')
    parser.add_argument('--max_words', default=2048, type=int,
                        help='max token length')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--save_freq', type=int, default=5000,
                        help='number of iterations between model saving')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--num_workers', default=5, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training setting
    parser.add_argument('--dist_on_itp', action='store_true')

    parser.add_argument('--model_parallel_size', type=int, default=1)
    parser.add_argument('--data_parallel', type=str, choices=['sdp', 'fsdp'], default='sdp')
    parser.add_argument('--precision', type=str, choices=['fp16', 'bf16', 'tf32'], default='bf16')
    parser.add_argument('--checkpointing', action="store_true", default=False,
                        help="enable gradient checkopointing")

    return parser


def main(args):
    misc.init_distributed_mode(args)
    fs_init.initialize_model_parallel(args.model_parallel_size)
    if args.precision == "tf32":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    global_rank = misc.get_rank()
    mp_rank = fs_init.get_model_parallel_rank()
    dp_rank = fs_init.get_data_parallel_rank()
    dp_world_size = fs_init.get_data_parallel_world_size()
    dp_group = fs_init.get_data_parallel_group()

    # define the model
    mixed_precision_dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "tf32": torch.float32,
    }[args.precision]
    with default_tensor_type(dtype=mixed_precision_dtype, device="cuda"):
        model = MetaModel(args.llama_type, args.llama_config,
                          args.tokenizer_path, with_visual=False,
                          max_seq_len=args.max_words)
    promote_trainable_params_to_fp32(model)
    misc.print_param_status(model)
    if args.pretrained_path and fs_init.get_data_parallel_rank() == 0:
        print(f"load pretrained from {args.pretrained_path}")
        load_result = load_tensor_parallel_model_list(model, args.pretrained_path)
        print("load result: ", load_result)
        if args.pretrained_type is not None:
            warnings.warn(
                "The `--pretrained_type` argument has been deprecated and will be removed soon. "
                "The types of checkpoints are now automatically discerned by file names now"
            )
    print("Unwrapped Model = %s" % str(model))

    # resume stage1
    if args.resume:
        misc.resume_stage1(args, model_without_FSDP=model)

    TransformerBlock = type(model.llma.layers[0])

    model = FSDP(
        model,
        process_group=fs_init.get_data_parallel_group(),
        auto_wrap_policy=functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=[TransformerBlock],
        ),
        limit_all_gathers=True,
        use_orig_params=True,
        sync_module_states=True,
        mixed_precision=MixedPrecision(
            param_dtype=mixed_precision_dtype,
            reduce_dtype=mixed_precision_dtype,
            buffer_dtype=mixed_precision_dtype,
        ),
        sharding_strategy={
            "sdp": ShardingStrategy.SHARD_GRAD_OP,
            "ddp": ShardingStrategy.NO_SHARD,
            "fsdp": ShardingStrategy.FULL_SHARD,
        }[args.data_parallel],
        device_id=torch.cuda.current_device(),
    )

    # broadcast nonmp parameters within model parallel group
    misc.broadcast_nonmp_parameters(model)

    # gradient checkpointing
    if args.checkpointing:
        print("apply gradient checkpointing")
        non_reentrant_wrapper = partial(
            checkpoint_wrapper,
            offload_to_cpu=False,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )
        check_fn = lambda submodule: isinstance(submodule, TransformerBlock)
        apply_activation_checkpointing(model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn)

    print("Model = %s" % str(model))

    eff_batch_size = args.batch_size * args.accum_iter * fs_init.get_data_parallel_world_size()
    print("effective batch size: %d" % eff_batch_size)

    # following timm: set wd as 0 for bias and norm layers
    param_groups = misc.add_weight_decay(model, args.weight_decay)
    optimizer = AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler(args)

    # data
    if args.packed_data:
        DatasetTrainCls = falcon_packed.Falcon
        DatasetValCls = falcon_packed.FalconVal
    else:
        DatasetTrainCls = falcon.Falcon
        DatasetValCls = falcon.FalconVal
    dataset_train = DatasetTrainCls(
        args.data_meta_path, args.data_root, tokenizer_path=args.tokenizer_path,
        max_words=args.max_words, num_processes=dp_world_size, process_rank=dp_rank,
    )
    dataset_val = DatasetValCls(args.data_meta_path, args.data_root, tokenizer_path=args.tokenizer_path,
                                max_words=args.max_words)
    print(dataset_train)

    if global_rank == 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.output_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    sampler_val = torch.utils.data.DistributedSampler(
        dataset_val, num_replicas=dp_world_size, rank=dp_rank, shuffle=False
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        sampler=sampler_val,
        drop_last=True,
    )

    start_iter = 0
    if args.resume:
        _, start_iter = misc.resume_stage2(args=args, model=model, optimizer=optimizer,
                                           loss_scaler=loss_scaler, dataset_train=dataset_train)

    print(f"Start training")
    start_time = time.time()

    train_stats = train_one_epoch(
        model, data_loader_train, data_loader_val,
        optimizer, 0, start_iter, loss_scaler,
        log_writer=log_writer,
        args=args
    )

    if args.output_dir and misc.is_main_process():
        if log_writer is not None:
            log_writer.flush()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
