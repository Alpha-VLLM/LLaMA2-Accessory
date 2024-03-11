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

resolution2scale = {
    256: [
        "256x256", "128x512", "144x432", "176x368", "192x336", "224x288",
        "288x224", "336x192", "368x176", "432x144", "512x128"
    ],
    512: ['512 x 512', '1024 x 256', '1008 x 256', '992 x 256', '976 x 256', '960 x 256', '960 x 272',
          '944 x 272', '928 x 272', '912 x 272', '896 x 272', '896 x 288', '880 x 288',
          '864 x 288', '848 x 288', '848 x 304', '832 x 304', '816 x 304', '816 x 320',
          '800 x 320', '784 x 320', '768 x 320', '768 x 336', '752 x 336', '736 x 336',
          '736 x 352', '720 x 352', '704 x 352', '704 x 368', '688 x 368', '672 x 368',
          '672 x 384', '656 x 384', '640 x 384', '640 x 400', '624 x 400', '624 x 416',
          '608 x 416', '592 x 416', '592 x 432', '576 x 432', '576 x 448', '560 x 448',
          '560 x 464', '544 x 464', '544 x 480', '528 x 480', '528 x 496', '512 x 496',
          '496 x 512', '496 x 528', '480 x 528', '480 x 544', '464 x 544',
          '464 x 560', '448 x 560', '448 x 576', '432 x 576', '432 x 592', '416 x 592',
          '416 x 608', '416 x 624', '400 x 624', '400 x 640', '384 x 640', '384 x 656',
          '384 x 672', '368 x 672', '368 x 688', '368 x 704', '352 x 704', '352 x 720',
          '352 x 736', '336 x 736', '336 x 752', '336 x 768', '320 x 768', '320 x 784',
          '320 x 800', '320 x 816', '304 x 816', '304 x 832', '304 x 848', '288 x 848',
          '288 x 864', '288 x 880', '288 x 896', '272 x 896', '272 x 912', '272 x 928',
          '272 x 944', '272 x 960', '256 x 960', '256 x 976', '256 x 992', '256 x 1008',
          '256 x 1024'],
    1024: ['1024x1024', '2048x512', '2032x512', '2016x512', '2000x512', '1984x512', '1984x528', '1968x528',
           '1952x528', '1936x528', '1920x528', '1920x544', '1904x544', '1888x544', '1872x544',
           '1872x560', '1856x560', '1840x560', '1824x560', '1808x560', '1808x576', '1792x576',
           '1776x576', '1760x576', '1760x592', '1744x592', '1728x592', '1712x592', '1712x608',
           '1696x608', '1680x608', '1680x624', '1664x624', '1648x624', '1632x624', '1632x640',
           '1616x640', '1600x640', '1584x640', '1584x656', '1568x656', '1552x656', '1552x672',
           '1536x672', '1520x672', '1520x688', '1504x688', '1488x688', '1488x704', '1472x704',
           '1456x704', '1456x720', '1440x720', '1424x720', '1424x736', '1408x736', '1392x736',
           '1392x752', '1376x752', '1360x752', '1360x768', '1344x768', '1328x768', '1328x784',
           '1312x784', '1296x784', '1296x800', '1280x800', '1280x816', '1264x816', '1248x816',
           '1248x832', '1232x832', '1232x848', '1216x848', '1200x848', '1200x864', '1184x864',
           '1184x880', '1168x880', '1168x896', '1152x896', '1136x896', '1136x912', '1120x912',
           '1120x928', '1104x928', '1104x944', '1088x944', '1088x960', '1072x960', '1072x976',
           '1056x976', '1056x992', '1040x992', '1040x1008', '1024x1008', '1008x1024',
           '1008x1040', '992x1040', '992x1056', '976x1056', '976x1072', '960x1072', '960x1088',
           '944x1088', '944x1104', '928x1104', '928x1120', '912x1120', '912x1136', '896x1136',
           '896x1152', '896x1168', '880x1168', '880x1184', '864x1184', '864x1200', '848x1200',
           '848x1216', '848x1232', '832x1232', '832x1248', '816x1248', '816x1264', '816x1280',
           '800x1280', '800x1296', '784x1296', '784x1312', '784x1328', '768x1328', '768x1344',
           '768x1360', '752x1360', '752x1376', '752x1392', '736x1392', '736x1408', '736x1424',
           '720x1424', '720x1440', '720x1456', '704x1456', '704x1472', '704x1488', '688x1488',
           '688x1504', '688x1520', '672x1520', '672x1536', '672x1552', '656x1552', '656x1568',
           '656x1584', '640x1584', '640x1600', '640x1616', '640x1632', '624x1632', '624x1648',
           '624x1664', '624x1680', '608x1680', '608x1696', '608x1712', '592x1712', '592x1728',
           '592x1744', '592x1760', '576x1760', '576x1776', '576x1792', '576x1808', '560x1808',
           '560x1824', '560x1840', '560x1856', '560x1872', '544x1872', '544x1888', '544x1904',
           '544x1920', '528x1920', '528x1936', '528x1952', '528x1968', '528x1984', '512x1984',
           '512x2000', '512x2016', '512x2032', '512x2048']
}


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
    parser.add_argument("--resolution", type=int, default=256, choices=[256, 512, 1024])
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

        train_args = torch.load(os.path.join(args.ckpt, "model_args.pth"))
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
                        value=resolution2scale[int(train_args.image_size)][0],
                        choices=resolution2scale[int(train_args.image_size)]
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
