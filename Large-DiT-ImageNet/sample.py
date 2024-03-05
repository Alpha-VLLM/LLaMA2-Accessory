#!/usr/bin/env python

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import torch
import torch.distributed as dist
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from models import DiT_models
import argparse
import multiprocessing as mp
import socket
import os
import fairscale.nn.model_parallel.initialize as fs_init
import json


def main(args, rank, master_port):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)

    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(args.num_gpus)
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["MASTER_ADDR"] = "127.0.0.1"

    dist.init_process_group("nccl")
    fs_init.initialize_model_parallel(args.num_gpus)
    torch.cuda.set_device(rank)

    train_args = torch.load(os.path.join(args.ckpt, "model_args.pth"))

    if dist.get_rank() == 0:
        print("Model arguments used for inference:",
              json.dumps(train_args.__dict__, indent=2))

    # Load model:
    latent_size = train_args.image_size // 8
    model = DiT_models[train_args.model](
        input_size=latent_size,
        num_classes=train_args.num_classes,
        qk_norm=train_args.qk_norm,
    )

    torch_dtype = {
        "fp32": torch.float, "tf32": torch.float,
        "bf16": torch.bfloat16, "fp16": torch.float16,
    }[args.precision]
    model.to(torch_dtype).cuda()
    if args.precision == "tf32":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    assert train_args.model_parallel_size == args.num_gpus
    ckpt = torch.load(os.path.join(
        args.ckpt,
        f"consolidated{'_ema' if args.ema else ''}."
        f"{rank:02d}-of-{args.num_gpus:02d}.pth",
    ), map_location="cpu")
    model.load_state_dict(ckpt, strict=True)

    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(
        f"stabilityai/sd-vae-ft-{train_args.vae}"
        if args.local_diffusers_model_root is None else
        os.path.join(args.local_diffusers_model_root,
                     f"stabilityai/sd-vae-ft-{train_args.vae}")
    ).cuda()

    # Create sampling noise:
    n = len(args.class_labels)
    z = torch.randn(
        n, 4, latent_size, latent_size,
        dtype=torch_dtype, device="cuda",
    )
    y = torch.tensor(args.class_labels, device="cuda")

    # Setup classifier-free guidance:
    z = torch.cat([z, z], 0)
    y_null = torch.tensor([1000] * n, device="cuda")
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

    # Sample images:
    samples = diffusion.p_sample_loop(
        model.forward_with_cfg, z.shape, z, clip_denoised=False,
        model_kwargs=model_kwargs, progress=True, device="cuda",
    )

    if rank == 0:
        samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        samples = vae.decode(samples.float() / 0.18215).sample

        # Save and display images:
        save_image(
            samples,
            args.image_save_path or os.path.join(
                args.ckpt, f"sample{'_ema' if args.ema else ''}.png"
            ),
            nrow=4, normalize=True, value_range=(-1, 1)
        )

    dist.barrier()


def find_free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument("--num_sampling_steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument(
        "--class_labels", type=int, nargs="+",
        help="Class labels to generate the images for.",
        default=[207, 360, 387, 974, 88, 979, 417, 279],
    )
    parser.add_argument(
        "--precision", type=str, choices=["fp32", "tf32", "fp16", "bf16"],
        default="tf32",
    )
    parser.add_argument(
        "--local_diffusers_model_root", type=str,
        help="Specify the root directory if diffusers models are to be loaded "
             "from the local filesystem (instead of being automatically "
             "downloaded from the Internet). Useful in environments without "
             "Internet access."
    )
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--ema", action="store_true", help="Use EMA models.")
    parser.add_argument("--no_ema", action="store_false", dest="ema", help="Do not use EMA models.")
    parser.set_defaults(ema=True)
    parser.add_argument(
        "--image_save_path", type=str,
        help="If specified, overrides the default image save path "
             "(sample{_ema}.png in the model checkpoint directory)."
    )
    args = parser.parse_args()

    master_port = find_free_port()
    assert args.num_gpus == 1, "Multi-GPU sampling is currently not supported."
    main(args, 0, master_port)

