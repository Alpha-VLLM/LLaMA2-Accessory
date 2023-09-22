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

from data.alpaca import transform_train, format_prompt
from util.tensor_parallel import load_tensor_parallel_model_list
from util.tensor_type import default_tensor_type


def get_args_parser():
    parser = argparse.ArgumentParser('Single-turn (conversation) demo', add_help=False)
    # Model parameters
    parser.add_argument('--llama_type', default='llama', type=str, metavar='MODEL',
                        help='type of llama')
    parser.add_argument('--llama_config', default='/path/to/params.json', type=str, nargs="+",
                        help='Path to llama model config')
    parser.add_argument('--tokenizer_path', type=str, default="../tokenizer.model",
                        help='path to tokenizer.model')

    parser.add_argument('--pretrained_path', default='/path/to/pretrained', type=str, nargs="+",
                        help='directory containing pre-trained checkpoints')

    parser.add_argument('--device', default='cuda',
                        help='device for inference')
    parser.add_argument('--model_parallel_size', default=1, type=int)

    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument("--dtype", type=str, choices=["fp16", "bf16"], default="bf16",
                        help="The dtype used for model weights and inference.")
    parser.add_argument('--quant', action='store_true', help="enable quantization")
    return parser

args = get_args_parser().parse_args()

# define the model
misc.init_distributed_mode(args)
fs_init.initialize_model_parallel(args.model_parallel_size)
target_dtype = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}[args.dtype]
with default_tensor_type(dtype=target_dtype, device="cpu" if args.quant else "cuda"):
    model = MetaModel(args.llama_type, args.llama_config, args.tokenizer_path, with_visual=False)

print(f"load pretrained from {args.pretrained_path}")
load_result = load_tensor_parallel_model_list(model, args.pretrained_path)
print("load result: ", load_result)


if args.quant:
    print("Quantizing model to 4bit!")
    from util.quant import quantize
    from transformers.utils.quantization_config import BitsAndBytesConfig
    quantization_config = BitsAndBytesConfig.from_dict(
        config_dict={
            "load_in_8bit": False, 
            "load_in_4bit": True, 
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": torch.bfloat16
        },
        return_unused_kwargs=False,
    )
    quantize(model, quantization_config)

print("Model = %s" % str(model))
model.bfloat16().cuda()


@ torch.inference_mode()
def generate(
        prompt,
        question_input,
        system_prompt,
        max_gen_len,
        gen_t, top_p
):
    image = None

    # text output
    _prompt = format_prompt({"instruction":prompt, "input":question_input}, system_prompt)

    dist.barrier()
    dist.broadcast_object_list([_prompt, image, max_gen_len, gen_t, top_p])
    if args.quant:
        results = model.generate([_prompt], image, max_gen_len=max_gen_len, temperature=gen_t, top_p=top_p)
    else:
        with torch.cuda.amp.autocast(dtype=target_dtype):
            results = model.generate([_prompt], image, max_gen_len=max_gen_len, temperature=gen_t, top_p=top_p)
    text_output = results[0].strip()
    return text_output

def create_demo():
    with gr.Blocks() as demo:
        with gr.Row():
            prompt = gr.Textbox(lines=4, label="Question")
        with gr.Row():
            question_input = gr.Textbox(lines=4, label="Question Input (Optional)")
        with gr.Row():
            system_prompt = gr.Dropdown(choices=['alpaca', 'None'], value="alpaca", label="System Prompt")
        with gr.Row() as text_config_row:
            max_gen_len = gr.Slider(minimum=1, maximum=512, value=128, interactive=True, label="Max Length")
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

    inputs = [
        prompt, question_input, system_prompt,
        max_gen_len, gen_t, top_p,
    ]
    outputs = [text_output]
    run_botton.click(fn=generate, inputs=inputs, outputs=outputs)

    return demo


def worker_func():
    while True:
        dist.barrier()

        input_data = [None for _ in range(5)]
        dist.broadcast_object_list(input_data)
        _prompt, image, max_gen_len, gen_t, top_p = input_data
        with torch.cuda.amp.autocast(dtype=target_dtype):
            _ = model.generate([_prompt], image, max_gen_len=max_gen_len, temperature=gen_t, top_p=top_p, )

if dist.get_rank() == 0:
    description = f"""
    # Single-turn demo🚀
    """

    with gr.Blocks(theme=gr.themes.Default(), css="#pointpath {height: 10em} .label {height: 3em}") as DEMO:
        gr.Markdown(description)
        create_demo()
    DEMO.queue(api_open=True, concurrency_count=1).launch(share=True)

else:
    worker_func()
