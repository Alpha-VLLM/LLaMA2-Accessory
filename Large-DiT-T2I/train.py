# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch FSDP.
"""
import argparse
from collections import OrderedDict
import contextlib
from copy import deepcopy
from datetime import datetime
import functools
import json
import logging
import os
import random
import socket
from time import time

from PIL import Image
from diffusers.models import AutoencoderKL
import fairscale.nn.model_parallel.initialize as fs_init
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy, MixedPrecision, StateDictType, FullStateDictConfig
)
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import ItemProcessor, MyDataset, read_general
from diffusion import create_diffusion
from grad_norm import (
    get_model_parallel_dim_dict, calculate_l2_grad_norm, scale_grad,
)
from imgproc import var_center_crop, generate_crop_size_list
import models
from parallel import distributed_init, get_inter_node_process_group, get_local_world_size, get_intra_node_process_group


#############################################################################
#                            Data item Processor                            #
#############################################################################


class T2IItemProcessor(ItemProcessor):
    def __init__(self, transform, tokenizer_path, text_dropout_prob, max_words=128):
        self.image_transform = transform
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, add_bos_token=True, add_eos_token=True)
        self.text_dropout_prob = text_dropout_prob
        self.max_words = max_words

    def process_item(self, data_item, training_mode=False):
        if "conversations" in data_item:
            assert "image" in data_item and len(data_item['conversations']) == 2
            image = Image.open(read_general(data_item['image'])).convert("RGB")
            text = data_item['conversations'][1]['value']
        else:
            image_path = data_item['path']
            image = Image.open(read_general(image_path)).convert("RGB")
            text = data_item['prompt']

        image = self.image_transform(image)
        if random.uniform(0., 1.) < self.text_dropout_prob:
            text = ""
        tokenized_caption = self.tokenizer.encode(text, truncation=False)
        token_mask = torch.arange(self.max_words) < len(tokenized_caption)
        if len(tokenized_caption) < self.max_words:
            tokenized_caption = tokenized_caption + [0] * (self.max_words - len(tokenized_caption))
        else:
            tokenized_caption = tokenized_caption[:self.max_words]
        tokenized_caption = torch.tensor(tokenized_caption, dtype=torch.long)

        return image, tokenized_caption, token_mask


#############################################################################
#                           Training Helper Functions                       #
#############################################################################


def dataloader_collate_fn(samples):
    image = [x[0] for x in samples]
    tokenized_caption = torch.stack([x[1] for x in samples], dim=0)
    token_mask = torch.stack([x[2] for x in samples], dim=0)
    return image, tokenized_caption, token_mask


def get_train_sampler(dataset, rank, world_size, global_batch_size, max_steps,
                      resume_step, seed):
    sample_indices = torch.empty([max_steps * global_batch_size // world_size],
                                 dtype=torch.long)
    epoch_id, fill_ptr, offs = 0, 0, 0
    while fill_ptr < sample_indices.size(0):
        g = torch.Generator()
        g.manual_seed(seed + epoch_id)
        epoch_sample_indices = torch.randperm(len(dataset), generator=g)
        epoch_id += 1
        epoch_sample_indices = epoch_sample_indices[
            (rank + offs) % world_size::world_size
        ]
        offs = (offs + world_size - len(dataset) % world_size) % world_size
        epoch_sample_indices = epoch_sample_indices[
            :sample_indices.size(0) - fill_ptr
        ]
        sample_indices[fill_ptr: fill_ptr + epoch_sample_indices.size(0)] = \
            epoch_sample_indices
        fill_ptr += epoch_sample_indices.size(0)
    return sample_indices[resume_step * global_batch_size // world_size:].tolist()


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    assert set(ema_params.keys()) == set(model_params.keys())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(),
                      logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def setup_lm_fsdp_sync(model: nn.Module) -> FSDP:
    # LM FSDP always use FULL_SHARD among the node.
    model = FSDP(
        model,
        auto_wrap_policy=functools.partial(
            lambda_auto_wrap_policy,
            lambda_fn=lambda m: m in list(model.layers),
        ),
        process_group=get_intra_node_process_group(),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=MixedPrecision(
            param_dtype=next(model.parameters()).dtype,
        ),
        device_id=torch.cuda.current_device(),
        sync_module_states=True,
        limit_all_gathers=True,
        use_orig_params=True,
    )
    torch.cuda.synchronize()
    return model


def setup_fsdp_sync(model: nn.Module, args: argparse.Namespace) -> FSDP:
    mp_world_size = fs_init.get_model_parallel_world_size()
    pp_world_size = dist.get_world_size(fs_init.get_pipeline_parallel_group())
    dp_world_size = fs_init.get_data_parallel_world_size()
    mp_pp_world_size = mp_world_size * pp_world_size

    hsdp_hard_condition = (
        get_inter_node_process_group() is not None
        and get_local_world_size() % mp_pp_world_size == 0
    )
    hsdp_soft_condition = (
        hsdp_hard_condition
        and mp_pp_world_size <= get_local_world_size() // 4
    )

    if args.data_parallel is not None:
        if args.data_parallel == "hsdp" and not hsdp_hard_condition:
            raise NotImplementedError("Hard conditions for HSDP are not met.")
        fsdp_strategy = args.data_parallel
    else:
        fsdp_strategy = "hsdp" if hsdp_soft_condition else "sdp"
        args.data_parallel = fsdp_strategy
        logging.getLogger(__name__).info(
            "Using automatically decided data parallel strategy: "
            f"{fsdp_strategy}."
        )

    if fsdp_strategy == "hsdp":
        intra_node_dp_pg = None
        for i in range(dist.get_world_size() // get_local_world_size()):
            for j in range(mp_world_size * pp_world_size):
                ranks = list(range(j, get_local_world_size(), mp_world_size * pp_world_size))
                ranks = [x + i * get_local_world_size() for x in ranks]
                group = dist.new_group(ranks)
                if dist.get_rank() in ranks:
                    intra_node_dp_pg = group
        assert intra_node_dp_pg is not None
        process_group = (intra_node_dp_pg, get_inter_node_process_group())
    else:
        process_group = fs_init.get_data_parallel_group()

    model = FSDP(
        model,
        auto_wrap_policy=functools.partial(
            lambda_auto_wrap_policy,
            lambda_fn=lambda m: m in model.get_fsdp_wrap_module_list(),
        ),
        process_group=process_group,
        sharding_strategy={
            "fsdp": ShardingStrategy.FULL_SHARD,
            "hsdp": ShardingStrategy.HYBRID_SHARD,
            "sdp": ShardingStrategy.SHARD_GRAD_OP,
        }[fsdp_strategy],
        mixed_precision=MixedPrecision(
            param_dtype={
                "fp32": torch.float, "tf32": torch.float,
                "bf16": torch.bfloat16, "fp16": torch.float16,
            }[args.precision],
            reduce_dtype={
                "fp32": torch.float, "tf32": torch.float,
                "bf16": torch.bfloat16, "fp16": torch.float16,
            }[args.grad_precision or args.precision],
        ),
        device_id=torch.cuda.current_device(),
        sync_module_states=True,
        limit_all_gathers=True,
        use_orig_params=True,
    )
    torch.cuda.synchronize()

    return model


def setup_mixed_precision(args):
    if args.precision == "tf32":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    elif args.precision in ["bf16", "fp16", "fp32"]:
        pass
    else:
        raise NotImplementedError(f"Unknown precision: {args.precision}")


#############################################################################
#                                Training Loop                              #
#############################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), (
        "Training currently requires at least one GPU."
    )

    distributed_init(args)

    dp_world_size = fs_init.get_data_parallel_world_size()
    dp_rank = fs_init.get_data_parallel_rank()
    mp_world_size = fs_init.get_model_parallel_world_size()
    mp_rank = fs_init.get_model_parallel_rank()

    assert args.global_batch_size % dp_world_size == 0, (
        "Batch size must be divisible by data parrallel world size."
    )
    local_batch_size = args.global_batch_size // dp_world_size
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    setup_mixed_precision(args)
    # print(f"Starting rank={rank}, seed={seed}, "
    #       f"world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    os.makedirs(args.results_dir, exist_ok=True)
    checkpoint_dir = os.path.join(args.results_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    if rank == 0:
        logger = create_logger(args.results_dir)
        logger.info(f"Experiment directory: {args.results_dir}")
        tb_logger = SummaryWriter(os.path.join(
            args.results_dir, "tensorboard",
            datetime.now().strftime("%Y%m%d_%H%M%S_") + socket.gethostname()
        ))
    else:
        logger = create_logger(None)
        tb_logger = None

    logger.info("Training arguments: " + json.dumps(args.__dict__, indent=2))

    logger.info(f"Setting-up language model: {args.lm}")

    model_lm = AutoModelForCausalLM.from_pretrained(args.lm).get_decoder()  # e.g. meta-llama/Llama-2-7b-hf
    cap_feat_dim = model_lm.config.hidden_size
    model_lm = setup_lm_fsdp_sync(model_lm)

    # Create model:
    assert args.image_size % 8 == 0, (
        "Image size must be divisible by 8 (for the VAE encoder)."
    )
    model = models.__dict__[args.model](
        qk_norm=args.qk_norm,
        cap_feat_dim=cap_feat_dim,
        rope_scaling_factor=args.rope_scaling_factor,
    )
    logger.info(f"DiT Parameters: {model.parameter_count():,}")
    model_patch_size = model.patch_size

    model_parallel_dim_dict = get_model_parallel_dim_dict(model)

    if args.auto_resume and args.resume is None:
        try:
            existing_checkpoints = os.listdir(checkpoint_dir)
            if len(existing_checkpoints) > 0:
                existing_checkpoints.sort()
                args.resume = os.path.join(checkpoint_dir,
                                           existing_checkpoints[-1])
        except Exception:
            pass
        if args.resume is not None:
            logger.info(f"Auto resuming from: {args.resume}")

    # Note that parameter initialization is done within the DiT constructor
    model_ema = deepcopy(model)
    if args.resume:
        if dp_rank == 0:  # other ranks receive weights in setup_fsdp_sync
            logger.info(f"Resuming model weights from: {args.resume}")
            model.load_state_dict(torch.load(os.path.join(
                args.resume,
                f"consolidated.{mp_rank:02d}-of-{mp_world_size:02d}.pth",
            ), map_location="cpu"), strict=True)
            logger.info(f"Resuming ema weights from: {args.resume}")
            model_ema.load_state_dict(torch.load(os.path.join(
                args.resume,
                f"consolidated_ema.{mp_rank:02d}-of-{mp_world_size:02d}.pth",
            ), map_location="cpu"), strict=True)
    elif args.init_from:
        if dp_rank == 0:
            logger.info(f"Initializing model weights from: {args.init_from}")
            state_dict = torch.load(os.path.join(
                args.init_from,
                f"consolidated.{mp_rank:02d}-of-{mp_world_size:02d}.pth",
            ), map_location="cpu")
            missing_keys, unexpected_keys = \
                model.load_state_dict(state_dict, strict=False)
            missing_keys_ema, unexpected_keys_ema = \
                model_ema.load_state_dict(state_dict, strict=False)
            del state_dict
            assert set(missing_keys) == set(missing_keys_ema)
            assert set(unexpected_keys) == set(unexpected_keys_ema)
            logger.info("Model initialization result:")
            logger.info(f"  Missing keys: {missing_keys}")
            logger.info(f"  Unexpeected keys: {unexpected_keys}")
    dist.barrier()

    model = setup_fsdp_sync(model, args)
    model_ema = setup_fsdp_sync(model_ema, args)

    # default: 1000 steps, linear noise schedule
    diffusion = create_diffusion(timestep_respacing="")
    vae = AutoencoderKL.from_pretrained(
        f"stabilityai/sd-vae-ft-{args.vae}"
        if args.local_diffusers_model_root is None else
        os.path.join(args.local_diffusers_model_root,
                     f"stabilityai/sd-vae-ft-{args.vae}")
    ).to(device)

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant
    # learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(),
                            lr=args.lr, weight_decay=args.wd)
    if args.resume:
        opt_state_world_size = len([
            x for x in os.listdir(args.resume)
            if x.startswith("optimizer.") and x.endswith(".pth")
        ])
        assert opt_state_world_size == dist.get_world_size(), (
            f"Resuming from a checkpoint with unmatched world size "
            f"({dist.get_world_size()} vs. {opt_state_world_size}) "
            f"is currently not supported."
        )
        logger.info(f"Resuming optimizer states from: {args.resume}")
        opt.load_state_dict(torch.load(os.path.join(
            args.resume,
            f"optimizer.{dist.get_rank():05d}-of-"
            f"{dist.get_world_size():05d}.pth",
        ), map_location="cpu"))
        for param_group in opt.param_groups:
            param_group["lr"] = args.lr
            param_group["weight_decay"] = args.wd

        with open(os.path.join(args.resume, "resume_step.txt")) as f:
            resume_step = int(f.read().strip())
    else:
        resume_step = 0

    # Setup data:
    logger.info("Creating data transform...")
    patch_size = 8 * model_patch_size
    max_num_patches = round((args.image_size / patch_size) ** 2)
    logger.info(f"Limiting number of patches to {max_num_patches}.")
    crop_size_list = generate_crop_size_list(max_num_patches, patch_size)
    logger.info("List of crop sizes:")
    for i in range(0, len(crop_size_list), 6):
        logger.info(" " + "".join([f"{f'{w} x {h}':14s}" for w, h in crop_size_list[i: i + 6]]))
    image_transform = transforms.Compose([
        transforms.Lambda(
            functools.partial(var_center_crop, crop_size_list=crop_size_list)
        ),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5],
                             inplace=True),
    ])
    dataset = MyDataset(
        args.data_path,
        item_processor=T2IItemProcessor(
            image_transform,
            args.tokenizer_path,
            args.caption_dropout_prob,
            args.max_text_tokens
        ),
        cache_on_disk = False
    )
    num_samples = args.global_batch_size * args.max_steps
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")
    logger.info(f"Total # samples to consume: {num_samples:,} "
                f"({num_samples / len(dataset):.2f} epochs)")
    sampler = get_train_sampler(
        dataset, dp_rank, dp_world_size, args.global_batch_size,
        args.max_steps, resume_step, args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=local_batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=dataloader_collate_fn,
    )

    # Prepare models for training:
    # important! This enables embedding dropout for classifier-free guidance
    model.train()

    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    start_time = time()

    logger.info(f"Training for {args.max_steps:,} steps...")
    for step, (x, caps, cap_mask) in enumerate(loader, start=resume_step):
        x = [img.to(device, non_blocking=True) for img in x]
        caps = caps.to(device)
        cap_mask = cap_mask.to(device)

        with torch.no_grad():
            # Map input images to latent space + normalize latents:
            x = [vae.encode(img[None]).latent_dist.sample().mul_(0.18215)[0] for img in x]
        t = torch.randint(0, diffusion.num_timesteps, (len(x),),
                          device=device)

        if mp_world_size > 1:
            mp_src = fs_init.get_model_parallel_src_rank()
            mp_group = fs_init.get_model_parallel_group()
            for img in x:
                dist.broadcast(img, mp_src, mp_group)
            dist.broadcast(t, mp_src, mp_group)
            dist.broadcast(caps, mp_src, mp_group)
            dist.broadcast(cap_mask, mp_src, mp_group)
            assert caps.size(0) % mp_world_size == 0
            caps = caps[caps.size(0) // mp_world_size * mp_rank, caps.size(0) // mp_world_size * (mp_rank + 1)]

        with torch.no_grad():
            cap_feats = model_lm(input_ids=caps).last_hidden_state

        if mp_world_size > 1:
            local_cap_feats = cap_feats
            cap_feats = torch.zeros([mp_world_size, *local_cap_feats.size()],
                                    dtype=local_cap_feats.dtype, device=device)
            dist.all_gather_into_tensor(cap_feats, local_cap_feats, group=mp_group)
            cap_feats = cap_feats.flatten(0, 1)

        loss_item = 0.
        opt.zero_grad()
        for mb_idx in range(
            (local_batch_size - 1) // args.micro_batch_size + 1
        ):
            mb_st = mb_idx * args.micro_batch_size
            mb_ed = min((mb_idx + 1) * args.micro_batch_size,
                        local_batch_size)
            last_mb = (mb_ed == local_batch_size)

            x_mb = x[mb_st: mb_ed]
            cap_feats_mb = cap_feats[mb_st: mb_ed]
            cap_mask_mb = cap_mask[mb_st: mb_ed]
            t_mb = t[mb_st: mb_ed]

            model_kwargs = dict(cap_feats=cap_feats_mb, cap_mask=cap_mask_mb)
            with {
                "bf16": torch.cuda.amp.autocast(dtype=torch.bfloat16),
                "fp16": torch.cuda.amp.autocast(dtype=torch.float16),
                "fp32": contextlib.nullcontext(),
                "tf32": contextlib.nullcontext(),
            }[args.precision]:
                loss_dict = diffusion.training_losses(
                    model, x_mb, t_mb, model_kwargs
                )
            loss = loss_dict["loss"].sum() / local_batch_size
            loss_item += loss.item()
            with (
                model.no_sync()
                if args.data_parallel in ["sdp", "hsdp"] and not last_mb else
                contextlib.nullcontext()
            ):
                loss.backward()

        grad_norm = calculate_l2_grad_norm(model, model_parallel_dim_dict)
        if grad_norm > args.grad_clip:
            scale_grad(model, args.grad_clip / grad_norm)

        if tb_logger is not None:
            tb_logger.add_scalar("train/loss", loss_item, step)
            tb_logger.add_scalar("train/grad_norm", grad_norm, step)
            tb_logger.add_scalar("train/lr", opt.param_groups[0]["lr"], step)

        opt.step()
        update_ema(model_ema, model)

        # Log loss values:
        running_loss += loss_item
        log_steps += 1
        if (step + 1) % args.log_every == 0:
            # Measure training speed:
            torch.cuda.synchronize()
            end_time = time()
            secs_per_step = (end_time - start_time) / log_steps
            imgs_per_sec = args.global_batch_size * log_steps / (
                end_time - start_time
            )
            # Reduce loss history over all processes:
            avg_loss = torch.tensor(running_loss / log_steps,
                                    device=device)
            dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
            avg_loss = avg_loss.item() / dist.get_world_size()
            logger.info(f"(step={step + 1:07d}) "
                        f"Train Loss: {avg_loss:.4f}, "
                        f"Train Secs/Step: {secs_per_step:.2f}, "
                        f"Train Imgs/Sec: {imgs_per_sec:.2f}")
            # Reset monitoring variables:
            running_loss = 0
            log_steps = 0
            start_time = time()

        # Save DiT checkpoint:
        if (
            (step + 1) % args.ckpt_every == 0
            or (step + 1) == args.max_steps
        ):
            checkpoint_path = f"{checkpoint_dir}/{step + 1:07d}"
            os.makedirs(checkpoint_path, exist_ok=True)

            with FSDP.state_dict_type(
                model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
            ):
                consolidated_model_state_dict = model.state_dict()
                if fs_init.get_data_parallel_rank() == 0:
                    consolidated_fn = (
                        "consolidated."
                        f"{fs_init.get_model_parallel_rank():02d}-of-"
                        f"{fs_init.get_model_parallel_world_size():02d}"
                        ".pth"
                    )
                    torch.save(
                        consolidated_model_state_dict,
                        os.path.join(checkpoint_path, consolidated_fn),
                    )
            dist.barrier()
            del consolidated_model_state_dict
            logger.info(f"Saved consolidated to {checkpoint_path}.")

            with FSDP.state_dict_type(
                model_ema,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
            ):
                consolidated_ema_state_dict = model_ema.state_dict()
                if fs_init.get_data_parallel_rank() == 0:
                    consolidated_ema_fn = (
                        "consolidated_ema."
                        f"{fs_init.get_model_parallel_rank():02d}-of-"
                        f"{fs_init.get_model_parallel_world_size():02d}"
                        ".pth"
                    )
                    torch.save(
                        consolidated_ema_state_dict,
                        os.path.join(checkpoint_path, consolidated_ema_fn),
                    )
            dist.barrier()
            del consolidated_ema_state_dict
            logger.info(f"Saved consolidated_ema to {checkpoint_path}.")

            with FSDP.state_dict_type(
                model_ema,
                StateDictType.LOCAL_STATE_DICT,
            ):
                opt_state_fn = (
                    f"optimizer.{dist.get_rank():05d}-of-"
                    f"{dist.get_world_size():05d}.pth"
                )
                torch.save(opt.state_dict(),
                           os.path.join(checkpoint_path, opt_state_fn))
            dist.barrier()
            logger.info(f"Saved optimizer to {checkpoint_path}.")

            if dist.get_rank() == 0:
                torch.save(args,
                           os.path.join(checkpoint_path, "model_args.pth"))
                with open(
                    os.path.join(checkpoint_path, "resume_step.txt"), "w"
                ) as f:
                    print(step + 1, file=f)
            dist.barrier()
            logger.info(f"Saved training arguments to {checkpoint_path}.")

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train DiT_Llama2_7B_patch2 with the
    # hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--model", type=str,
                        default="DiT_Llama2_7B_patch2")
    parser.add_argument("--image_size", type=int,
                        default=256)
    parser.add_argument(
        "--max_steps", type=int, default=100_000,
        help="Number of training steps."
    )
    parser.add_argument("--global_batch_size", type=int, default=256)
    parser.add_argument("--micro_batch_size", type=int, default=1)
    parser.add_argument("--global_seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"],
                        default="ema")  # Choice doesn't affect training
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--ckpt_every", type=int, default=50_000)
    parser.add_argument("--master_port", type=int, default=8964)
    parser.add_argument("--model_parallel_size", type=int, default=1)
    parser.add_argument("--data_parallel", type=str,
                        choices=["sdp", "fsdp"], default="fsdp")
    parser.add_argument("--precision",
                        choices=["fp32", "tf32", "fp16", "bf16"],
                        default="bf16")
    parser.add_argument("--grad_precision",
                        choices=["fp32", "fp16", "bf16"])
    parser.add_argument(
        "--local_diffusers_model_root", type=str,
        help="Specify the root directory if diffusers models are to be loaded "
             "from the local filesystem (instead of being automatically "
             "downloaded from the Internet). Useful in environments without "
             "Internet access."
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4,
        help="Learning rate."
    )
    parser.add_argument(
        "--no_auto_resume", action="store_false", dest="auto_resume",
        help="Do NOT auto resume from the last checkpoint in --results_dir."
    )
    parser.add_argument(
        "--resume", type=str,
        help="Resume training from a checkpoint folder."
    )
    parser.add_argument(
        "--init_from", type=str,
        help="Initialize the model weights from a checkpoint folder. "
             "Compared to --resume, this loads neither the optimizer states "
             "nor the data loader states."
    )
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=2.0,
        help="Clip the L2 norm of the gradients to the given value."
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=0.,
        help="Weight decay for the optimizer.",
    )
    parser.add_argument(
        "--qk_norm",
        action="store_true",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="meta-llama/Llama-2-7b-hf"
    )
    parser.add_argument(
        "--lm",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
    )
    parser.add_argument(
        "--caption_dropout_prob",
        type=float,
        default=0.1,
        help="Randomly change the caption of a sample to a blank string with the given probability."
    )
    parser.add_argument(
        "--max_text_tokens",
        type=int,
        default=128,
        help="max number of text tokens"
    )
    parser.add_argument(
        "--rope_scaling_factor", type=float, default=1.0,
    )
    args = parser.parse_args()

    main(args)
