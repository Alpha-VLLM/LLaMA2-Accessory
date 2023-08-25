import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
import functools
from functools import partial

import torch
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

from apex.optimizers import FusedAdam

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.tensor_type import default_tensor_type, promote_trainable_params_to_fp32
from model.meta import MetaModel
from engine_finetune import train_one_epoch
from torch.utils.data import Dataset
from data.alpaca import FinetuneDataset, transform_train, FinetuneDistSampler
from data.conversation.dataset import FinetuneDialogDataset

from util.quant import quantize
from util.tensor_parallel import load_tensor_parallel_model


def get_args_parser():
    parser = argparse.ArgumentParser('LLaMA2-Accessory Finetuning', add_help=False)
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--accum_iter', default=4, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--llama_type', default='llama', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--llama_config', default=['params.json'], nargs="+",
                        help='Path to llama model config. If multiple jsons are given, their union will be used. '
                             'When the same key appears more than once, its last appearance is adopted.')

    parser.add_argument('--no_visual', action="store_true",
                        help='to not instantialize visual modules')
    parser.add_argument('--tokenizer_path', type=str, default="../tokenizer.model",
                        help='path to tokenizer.model')


    parser.add_argument('--pretrained_path', default='/path/to/pretrained', type=str,
                        help='path to checkpoint from pretrain stage')
    parser.add_argument('--pretrained_type', type=str, choices=['consolidated', 'meta_ori'],
                        help='pretrained checkpoint save format')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.02,
                        help='weight decay (default: 0.02)')

    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--min_lr', type=float, default=0.0001, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--warmup_epochs', type=float, default=1.0, metavar='N',
                        help='epoch to warmup LR')

    parser.add_argument('--clip_grad', type=int, default=-1,
                        help='grad clipping norm')

    # Dataset parameters
    parser.add_argument('--max_words', default=1024, type=int,
                        help='max token length')
    parser.add_argument('--dialog', action="store_true", default=False,
                        help='whether use dialog dataset')
    parser.add_argument('--data_config', default='/path/to/data/config/yaml', type=str,
                        help='data config path')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--save_interval', default=1, type=int,
                        help='number of epochs between model saving')
    parser.add_argument('--only_save_trainable', default=False, action="store_true",
                        help='only save trainable model parameters')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--num_workers', default=5, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    parser.add_argument('--model_parallel_size', type=int, default=1)
    parser.add_argument('--data_parallel', type=str, choices=['sdp', 'fsdp'], default='sdp')
    parser.add_argument('--precision', type=str, choices=['fp16', 'bf16', 'tf32'], default='bf16')
    parser.add_argument('--checkpointing', action="store_true", default=False,
                        help="enable gradient checkopointing")
    parser.add_argument('--quant', action="store_true", default=False,
                        help="enable quantization")

    return parser


def main(args):
    misc.init_distributed_mode(args)
    fs_init.initialize_model_parallel(args.model_parallel_size)
    if args.precision == "tf32":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

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
    print("Start initialization.")

    if args.quant:
        from transformers.utils.quantization_config import BitsAndBytesConfig
        for i in range(misc.get_world_size()):
            if i == misc.get_rank():
                print(f"## Processing on RANK {i}.", force=True)
                with default_tensor_type(dtype=mixed_precision_dtype, device="cpu"):
                    model = MetaModel(args.llama_type, args.llama_config,
                                    args.tokenizer_path, with_visual=not args.no_visual,
                                    max_seq_len=args.max_words)
                promote_trainable_params_to_fp32(model)
                misc.print_param_status(model)

                # load pre-trained weights
                print(f"## Load pretrained from {args.pretrained_path}", force=True)
                load_tensor_parallel_model(model, args.pretrained_path, args.pretrained_type)

                print("## Quantizing model to 4bit!", force=True)
                quantization_config = BitsAndBytesConfig.from_dict(
                    config_dict={
                        "load_in_8bit": False, 
                        "load_in_4bit": True, 
                        "bnb_4bit_quant_type": "nf4",
                    },
                    return_unused_kwargs=False,
                )
                quantize(model, quantization_config)
                
                # will (1) release CPU memory usage, and (2) occupy GPU memory.
                model.cuda() 
            torch.distributed.barrier()
    else:
        with default_tensor_type(dtype=mixed_precision_dtype, device="cpu"):
            model = MetaModel(args.llama_type, args.llama_config,
                            args.tokenizer_path, with_visual=not args.no_visual,
                            max_seq_len=args.max_words)
        print("Finish initialization.")
        promote_trainable_params_to_fp32(model)
        misc.print_param_status(model)
        print(f"load pretrained from {args.pretrained_path}")
        load_tensor_parallel_model(model, args.pretrained_path, args.pretrained_type)
    print("Unwrapped Model = %s" % str(model))

    # resume stage1
    if args.resume:
        misc.resume_stage1(args, model_without_FSDP=model)

    TransformerBlock = type(model.llma.layers[0])
    # ignored_named_parameters = {name: param for name, param in model.named_parameters() if not param.requires_grad}
    # print(ignored_named_parameters.keys())
    model = FSDP(
        model,
        process_group=fs_init.get_data_parallel_group(),
        auto_wrap_policy=functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=[] if model.is_peft else [TransformerBlock],
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
        device_id=device,
        ignored_parameters=[param for param in model.parameters() if not param.requires_grad]
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
    optimizer = FusedAdam(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler(args)

    # data
    if args.dialog:
        DatasetClass = FinetuneDialogDataset
    else:
        DatasetClass = FinetuneDataset
    dataset_train = DatasetClass(args.data_config, transform_train,
                                 max_words=args.max_words, image_words=model.get_image_words(),
                                 tokenizer_path=args.tokenizer_path)
    print(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    sampler_train = FinetuneDistSampler(
        dataset_train, num_replicas=dp_world_size, rank=dp_rank, shuffle=True, batch_size=args.batch_size,
        acc_grad=args.accum_iter
    )
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        sampler=sampler_train,
        drop_last=True,
    )

    start_epoch = 0
    if args.resume:
        start_epoch, _ = misc.resume_stage2(args=args, model=model, optimizer=optimizer, loss_scaler=loss_scaler,
                                            dataset_train=dataset_train)


    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )

        if args.output_dir and (epoch % args.save_interval == 0 or epoch + 1 == args.epochs):
            misc.save_checkpoint(
                output_dir=args.output_dir,
                args=args, epoch=epoch, iteration=None, model=model, optimizer=optimizer,
                loss_scaler=loss_scaler, dataset_state=None,
            )

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch,
                     **{f'val_{k}': v for k, v in train_stats.items()}}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
