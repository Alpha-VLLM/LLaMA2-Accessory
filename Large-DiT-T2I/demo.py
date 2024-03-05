import argparse
import json
import multiprocessing as mp
import os
import socket

from diffusers.models import AutoencoderKL
import fairscale.nn.model_parallel.initialize as fs_init
import gradio as gr
import torch
import torch.distributed as dist
from torchvision.transforms.functional import to_pil_image
from transformers import AutoModelForCausalLM, AutoTokenizer

from diffusion import create_diffusion
import models


def find_free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


@torch.no_grad()
def model_main(args, master_port, rank, request_queue, response_queue, mp_barrier):
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(args.num_gpus)

    dist.init_process_group("nccl")
    fs_init.initialize_model_parallel(args.num_gpus)
    torch.cuda.set_device(rank)

    train_args = torch.load(os.path.join(args.ckpt, "model_args.pth"))
    if dist.get_rank() == 0:
        print("Loaded model arguments:",
              json.dumps(train_args.__dict__, indent=2))

    if dist.get_rank() == 0:
        print(f"Creating lm: {train_args.lm}")
    model_lm = AutoModelForCausalLM.from_pretrained(train_args.lm)  # e.g. meta-llama/Llama-2-7b-hf
    cap_feat_dim = model_lm.config.hidden_size
    if args.num_gpus > 1:
        raise NotImplementedError("Inference with >1 GPUs not yet supported")
    else:
        model_lm.cuda()
    model_lm.eval()

    tokenizer = AutoTokenizer.from_pretrained(train_args.tokenizer_path, add_bos_token=True, add_eos_token=True)

    # if dist.get_rank() == 0:
        # print(f"Creating vae: {train_args.vae}")
    vae = AutoencoderKL.from_pretrained(
        f"stabilityai/sd-vae-ft-{train_args.vae}"
    ).cuda()

    if dist.get_rank() == 0:
        print(f"Creating DiT: {train_args.model}")
    # latent_size = train_args.image_size // 8
    model = models.__dict__[train_args.model](
        max_seq_len=getattr(train_args, "max_seq_len", 288),
        qk_norm=train_args.qk_norm,
        cap_feat_dim=cap_feat_dim,
    )
    model.eval().cuda()

    assert train_args.model_parallel_size == args.num_gpus
    ckpt = torch.load(os.path.join(
        args.ckpt, f"consolidated{'_ema' if args.ema else ''}.{rank:02d}-of-{args.num_gpus:02d}.pth"
    ), map_location="cpu")
    model.load_state_dict(ckpt, strict=True)

    mp_barrier.wait()

    while True:
        caption, resolution, num_sampling_steps, cfg_scale = request_queue.get()

        diffusion = create_diffusion(str(num_sampling_steps))
        w, h = resolution.split("x")
        w, h = int(w), int(h)
        latent_w, latent_h = w // 8, h // 8
        z = torch.randn([1, 4, latent_w, latent_h], device="cuda")
        z = z.repeat(2, 1, 1, 1)

        cap_tok = tokenizer.encode(caption, truncation=False)
        null_cap_tok = tokenizer.encode("", truncation=False)
        tok = torch.zeros([2, max(len(cap_tok), len(null_cap_tok))], dtype=torch.long, device="cuda")
        tok_mask = torch.zeros_like(tok, dtype=torch.bool)
        tok[0, :len(cap_tok)] = torch.tensor(cap_tok)
        tok[1, :len(null_cap_tok)] = torch.tensor(null_cap_tok)
        tok_mask[0, :len(cap_tok)] = True
        tok_mask[1, :len(null_cap_tok)] = True

        with torch.no_grad():
            cap_feats = model_lm.get_decoder()(input_ids=tok).last_hidden_state.float()
        model_kwargs = dict(cap_feats=cap_feats, cap_mask=tok_mask, cfg_scale=cfg_scale)

        if dist.get_rank() == 0:
            print(f"caption: {caption}")
            print(f"num_sampling_steps: {num_sampling_steps}")
            print(f"cfg_scale: {cfg_scale}")
        samples = diffusion.p_sample_loop(
            model.forward_with_cfg, z.shape, z, clip_denoised=False,
            model_kwargs=model_kwargs, progress=dist.get_rank() == 0, device="cuda",
        )
        samples = samples[:1]

        samples = vae.decode(samples.float() / 0.18215).sample
        samples = (samples + 1.) / 2.
        samples.clamp_(0., 1.)
        img = to_pil_image(samples[0])

        if response_queue is not None:
            response_queue.put(img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--ema", action="store_true")
    args = parser.parse_args()

    master_port = find_free_port()

    processes = []
    request_queues = []
    response_queue = mp.Queue()
    mp_barrier = mp.Barrier(args.num_gpus + 1)
    for i in range(args.num_gpus):
        request_queues.append(mp.Queue())
        p = mp.Process(target=model_main,
                       args=(args, master_port, i, request_queues[i], response_queue if i == 0 else None, mp_barrier))
        p.start()
        processes.append(p)

    with gr.Blocks() as demo:
        with gr.Row():
            gr.Markdown(f"""# DiT image generation demo

**Model path:** {os.path.abspath(args.ckpt)}""")
        with gr.Row():
            with gr.Column():
                cap = gr.Textbox(lines=2, label="Caption", interactive=True)
                with gr.Row():
                    num_sampling_steps = gr.Slider(
                        minimum=1, maximum=1000, value=250, interactive=True,
                        label="Sampling steps"
                    )
                    cfg_scale = gr.Slider(
                        minimum=1., maximum=20., value=4., interactive=True,
                        label="CFG scale"
                    )
                with gr.Row():
                    resolution = gr.Dropdown(
                        value="256x256",
                        choices=[
                            "128x512", "144x432", "176x368", "192x336", "224x288", "256x256",
                            "288x224", "336x192", "368x176", "432x144", "512x128"
                        ]
                    )
                with gr.Row():
                    submit_btn = gr.Button("Submit", variant="primary")
                    reset_btn = gr.ClearButton([cap, num_sampling_steps, cfg_scale])
            with gr.Column():
                output_img = gr.Image(label="Generated image", interactive=False)

        def on_submit(caption, resolution, num_sampling_steps, cfg_scale):
            for q in request_queues:
                q.put((caption, resolution, num_sampling_steps, cfg_scale))
            return response_queue.get()

        submit_btn.click(on_submit, [cap, resolution, num_sampling_steps, cfg_scale], [output_img])

    mp_barrier.wait()
    demo.queue().launch(
        share=True, server_name="0.0.0.0",
    )

if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
