import functools
import sys
import os
sys.path.append(os.path.abspath(__file__).rsplit('/', 2)[0])

from model.meta import MetaModel

import argparse
import torch
import torch.distributed as dist
import gradio as gr

from util import misc
from fairscale.nn.model_parallel import initialize as fs_init

from data.conversation.lib import conv_templates, SeparatorStyle, Conversation

def get_args_parser():
    parser = argparse.ArgumentParser('Multi-turn (conversation) demo', add_help=False)
    # Model parameters
    parser.add_argument('--llama_type', default='llama', type=str, metavar='MODEL',
                        help='type of llama')
    parser.add_argument('--llama_config', default='/path/to/params.json', type=str, nargs="+",
                        help='Path to llama model config')
    parser.add_argument('--tokenizer_path', type=str, default="../tokenizer.model",
                        help='path to tokenizer.model')

    parser.add_argument('--pretrained_path', default='/path/to/pretrained', type=str,
                        help='directory containing pre-trained checkpoints')
    parser.add_argument('--pretrained_type', type=str, default="consolidated", choices=['consolidated', 'meta_ori'],
                        help='pretrained checkpoint save format')

    parser.add_argument('--device', default='cuda',
                        help='device for inference')
    parser.add_argument('--model_parallel_size', default=1, type=int)

    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    return parser

args = get_args_parser().parse_args()

# define the model
misc.init_distributed_mode(args)
fs_init.initialize_model_parallel(args.model_parallel_size)
model = MetaModel(args.llama_type, args.llama_config, args.tokenizer_path, with_visual=False)
print(f"load pretrained from {args.pretrained_path}")
misc.load_pretrained(args.pretrained_path, args.pretrained_type, model)
print("Model = %s" % str(model))
model.cuda().half()

@ torch.inference_mode()
def generate(
        prompt,
        max_gen_len,
        gen_t, top_p, reset,
        conv
):
    if conv is None:
        conv = conv_templates['v1'].copy()

    image = None

    if reset:
        conv.messages = []

    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    _prompt = conv.get_prompt()
    print(_prompt)

    dist.barrier()
    dist.broadcast_object_list([_prompt, image, max_gen_len, gen_t, top_p])
    with torch.cuda.amp.autocast():
        result = model.generate([_prompt], image, max_gen_len=max_gen_len, temperature=gen_t, top_p=top_p,)
        print(result)

    result = result[0]
    stop_str = conv.sep if conv.sep_style == SeparatorStyle.SINGLE else conv.sep2
    pos = result.find(stop_str)
    if pos!=-1:
        result = result[:pos]

    result = result.rstrip()+"\n"

    conv.messages[-1][-1] = result

    # print(conv.messages)

    return conv.get_prompt(), False, conv

def create_demo():
    with gr.Blocks() as demo:
        with gr.Row():
            prompt = gr.Textbox(lines=4, label="Input")
        with gr.Row():
            reset_box = gr.Checkbox(value=False, label="Reset")
        with gr.Row() as text_config_row:
            max_gen_len = gr.Slider(minimum=1, maximum=512, value=128, interactive=True, label="Single-turn max length")
            # with gr.Accordion(label='Advanced options', open=False):
            gen_t = gr.Slider(minimum=0, maximum=1, value=0.1, interactive=True, label="Temperature")
            top_p = gr.Slider(minimum=0, maximum=1, value=0.75, interactive=True, label="Top p")
        with gr.Row():
            # clear_botton = gr.Button("Clear")
            run_botton = gr.Button("Run", variant='primary')

        with gr.Row():
            gr.Markdown("Output")
        with gr.Row():
            text_output = gr.Textbox(lines=11, label='Text Out')
        conv = gr.State(value=None)

    inputs = [
        prompt,
        max_gen_len, gen_t, top_p, reset_box, conv
    ]
    outputs = [text_output, reset_box, conv]
    run_botton.click(fn=generate,
                     inputs=inputs, outputs=outputs)

    return demo


def worker_func():
    while True:
        dist.barrier()

        input_data = [None for _ in range(5)]
        dist.broadcast_object_list(input_data)
        _prompt, image, max_gen_len, gen_t, top_p = input_data
        with torch.cuda.amp.autocast():
            _ = model.generate([_prompt], image, max_gen_len=max_gen_len, temperature=gen_t, top_p=top_p, )


if dist.get_rank() == 0:
    description = """
    # Multi-turn demo🚀
    """

    with gr.Blocks(theme=gr.themes.Default(), css="#pointpath {height: 10em} .label {height: 3em}") as DEMO:
        gr.Markdown(description)
        create_demo()
    DEMO.queue(api_open=True, concurrency_count=1).launch(share=True)

else:
    worker_func()
