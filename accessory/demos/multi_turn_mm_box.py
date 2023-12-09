import random
import sys
import os
sys.path.append(os.path.abspath(__file__).rsplit('/', 3)[0])

import argparse
import multiprocessing as mp
import numpy as np
from typing import List, Optional

import torch
import torch.distributed as dist

from fairscale.nn.model_parallel import initialize as fs_init

import gradio as gr

from accessory.util.misc import setup_for_distributed
from accessory.model.meta import MetaModel
from accessory.data.conversation import default_conversation
from PIL import Image, ImageDraw
from accessory.data.transform import get_transform
from segment_anything import sam_model_registry, SamPredictor

import regex as re

class Ready: pass
class ModelFailure: pass

def model_worker(
    rank: int, args: argparse.Namespace, barrier: mp.Barrier,
    request_queue: mp.Queue, response_queue: Optional[mp.Queue] = None,
) -> None:
    """
    The worker function that manipulates the GPU to run the inference.
    Exact n_gpu workers are started, with each one operating on a separate GPU.

    Args:
        rank (int): Distributed rank of the worker.
        args (argparse.Namespace): All command line arguments.
        barrier (multiprocessing.Barrier): A barrier used to delay the start
            of Web UI to be after the start of the model.
    """

    world_size = len(args.gpu_ids)
    gpu_id = args.gpu_ids[rank]
    dist.init_process_group(
        backend="nccl", rank=rank, world_size=world_size,
        init_method=f"tcp://{args.master_addr}:{args.master_port}",
    )
    print(f"| distributed init on worker {rank}/{world_size}. "
          f"using gpu: {gpu_id}")
    fs_init.initialize_model_parallel(world_size)
    torch.cuda.set_device(gpu_id)

    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)

    # set the print behavior.
    setup_for_distributed(rank == 0)

    target_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16
    }[args.dtype]
    model = MetaModel.from_pretrained(args.pretrained_path, args.llama_type, args.llama_config, args.tokenizer_path,
                                      with_visual=True, max_seq_len=args.max_seq_len,
                                      mp_group=fs_init.get_model_parallel_group(),
                                      dtype=target_dtype, device="cpu" if args.quant else "cuda", )
    if args.quant:
        from accessory.util.quant import quantize
        print("Quantizing model to 4bit!")
        from transformers.utils.quantization_config import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig.from_dict(
            config_dict={
                "load_in_8bit": False,
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
            },
            return_unused_kwargs=False,
        )
        quantize(model, quantization_config)
        model.cuda()
    model.eval()
    print(f"Model = {str(model)}")

    conv = default_conversation()
    conv_sep = conv.response_end_signal

    barrier.wait()

    while True:
        if response_queue is not None:
            response_queue.put(Ready())
        try:
            image, chatbot, max_gen_len, temperature, top_p, img_transform = request_queue.get()
            if image is not None:
                image = image.convert("RGB")
                transform = get_transform(img_transform, getattr(model.llma, 'image_size', 224))
                image = transform(image).unsqueeze(0).cuda().to(target_dtype)
            else:
                image = None
            conv.load_qas(chatbot)
            prompt = conv.get_prompt()

            with torch.cuda.amp.autocast(dtype=target_dtype, enabled=not args.quant):
                print(prompt)
                for stream_response in model.stream_generate(
                    prompt, image,
                    max_gen_len, temperature, top_p
                ):
                    end_pos = stream_response["text"].find(conv_sep)
                    if end_pos != -1:
                        stream_response["text"] = (
                            stream_response['text'][:end_pos].rstrip() + "\n"
                        )
                        stream_response["end_of_content"] = True

                    # keep a few characters if not end_of_content to avoid sending
                    # part of conv_sep before all of it is generated.
                    if not stream_response["end_of_content"]:
                        if len(stream_response["text"]) < len(conv_sep):
                            continue
                        stream_response["text"] = (
                            stream_response["text"][:-len(conv_sep)]
                        )

                    if response_queue is not None:
                        response_queue.put(stream_response)

                    if stream_response["end_of_content"]:
                        break
        except Exception:
            response_queue.put(ModelFailure())

def extract_and_color(input_string):
    """
    Extracts tuples of text and list from a given string and also generates a Markdown string
    with uniquely colored substrings wrapped with <p> </p>.

    Parameters:
        input_string (str): The string to extract and color information from

    Returns:
        Tuple: First element is a list of tuples where the first element is the text and the second is a list of floats,
               Second element is a Markdown-formatted string with uniquely colored substrings.
    """

    # Initialize result list
    result = []

    # Initialize Markdown string
    markdown_str = input_string

    # Predefined colors
    colors = ["red", "blue", "green", "purple", "orange"]

    # Counter for color index
    color_idx = 0

    # Regular expression to match '<p>...</p>[...]' pattern
    pattern = r'<p>(.*?)<\/p>\s*(\[[\d.,;\s]*\])'

    # Find all matches
    matches = list(re.finditer(pattern, markdown_str))
    matches = sorted(matches, key=lambda x: x.start(), reverse=True)

    # Parse each match
    for match in matches:
        # Extract text and list as string
        text, list_str = match.groups()
        # Convert the list string to an actual list of floats
        float_list = [[float(x) for x in re.findall(r'\d+\.\d+', _)] for _ in list_str.split(";")]

        # Assign color
        color = colors[color_idx]

        # Append to result list
        result.append((text, float_list, color))

        # Replace text in Markdown string with colored version
        # ||| as temporary mark, will be removed finally
        colored_text = f'<span style="color:{color}">{text}|||{list_str}</span>'
        markdown_str = markdown_str[:match.span()[0]] + colored_text + markdown_str[match.span()[1]:]

        # Move to next color
        color_idx = (color_idx + 1) % len(colors)  # Cycle through colors

    # Regular expression to match '[...]' pattern with no preceding </p>
    pattern2 = r'(?<!\|\|\|)(\[[\d.,;\s]*\])'
    matches2 = list(re.finditer(pattern2, markdown_str))
    matches2 = sorted(matches2, key = lambda x: x.start(), reverse=True)
    # Parse each match
    for match in matches2:
        # Extract text and list as string
        list_str = match.groups()[0]

        # Convert the list string to an actual list of floats
        float_list = [[float(x) for x in re.findall(r'\d+\.\d+', _)] for _ in list_str.split(";")]

        # Assign color
        color = colors[color_idx]

        # Append to result list
        result.append(("", float_list, color))

        # Replace text in Markdown string with colored version
        colored_text = f'<span style="color:{color}">{list_str}</span>'
        markdown_str = markdown_str[:match.span()[0]] + colored_text + markdown_str[match.span()[1]:]

    markdown_str = markdown_str.replace("|||[", "[")

    return result, markdown_str

def show_mask(img: Image, mask: torch.Tensor, color):
    alpha_value = int(0.5 * 255)
    alpha_mask = (mask * alpha_value).byte()
    mask_img = Image.new("RGB", img.size, color)

    blended_img = Image.composite(mask_img, img, Image.fromarray(alpha_mask.cpu().numpy(), "L"))
    return blended_img

def draw_box_mask_on_image(img: Image, l_name_box_color, predictor):
    max_edge = max((img.width, img.height))

    if img.width < img.height:
        x_origin = (img.height - img.width) // 2
        y_origin = 0
    else:
        x_origin = 0
        y_origin = (img.width - img.height) // 2

    img_box = img.copy()
    draw = ImageDraw.Draw(img_box)
    boxes = []
    box_colors = []

    key_point_cache = {} # todo support multi-object pose
    key_point_names = ["nose","left_eye","right_eye","left_ear","right_ear","left_shoulder","right_shoulder",
                       "left_elbow","right_elbow","left_wrist","right_wrist","left_hip","right_hip",
                       "left_knee","right_knee","left_ankle","right_ankle"]
    skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10],
                 [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
    for name, l_points_in_square, color in l_name_box_color:
        for points_in_square in l_points_in_square:
            if len(points_in_square) == 2:
                x1, y1 = points_in_square
                x1 = x1 * max_edge - x_origin
                y1 = y1 * max_edge - y_origin
                draw.ellipse([x1-5, y1-5, x1+5, y1+5], fill=color)
                key_point_cache[name.replace(" ", "_")] = (x1, y1)
            elif len(points_in_square) == 4:
                x1, y1, x2, y2 = points_in_square
                x1 = x1 * max_edge - x_origin
                y1 = y1 * max_edge - y_origin
                x2 = x2 * max_edge - x_origin
                y2 = y2 * max_edge - y_origin

                draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
                boxes.append([int(x1), int(y1), int(x2), int(y2)])
                box_colors.append(color)
            # draw.text((x1 + 3, y1 + 3), name, font=ImageFont.truetype("../asset/arial.ttf", 15), fill=color)

    # draw skeleton:
    for edge_s, edge_t in skeleton:
        edge_s_name, edge_t_name = key_point_names[edge_s-1], key_point_names[edge_t-1]
        if edge_s_name in key_point_cache and edge_t_name in key_point_cache:
            draw.line([key_point_cache[edge_s_name], key_point_cache[edge_t_name]], fill="green", width=3)

    if len(boxes) > 0:
        img_mask = img.copy()
        img_array = np.array(img)
        predictor.set_image(img_array)
        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=predictor.transform.apply_boxes_torch(torch.tensor(boxes).cuda(), img_array.shape[:2]),
            multimask_output=False,
        )
        for mask, color in zip(masks, box_colors):
            img_mask = show_mask(img_mask, mask[0], color)
    else:
        img_mask = img

    return img_box, img_mask


def gradio_worker(
    request_queues: List[mp.Queue], response_queue: mp.Queue,
    args: argparse.Namespace, barrier: mp.Barrier,
) -> None:
    """
    The gradio worker is responsible for displaying the WebUI and relay the
    requests to model workers. It should be launched only once.

    Args:
        request_queues (List[mp.Queue]): A list of request queues (one for
            each model worker).
        args (argparse.Namespace): All command line arguments.
        barrier (multiprocessing.Barrier): A barrier used to delay the start
            of Web UI to be after the start of the model.
    """

    sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth").cuda()
    sam_predictor = SamPredictor(sam)

    def show_user_input(msg, chatbot, chatbox_display):
        return "", chatbot + [[msg, None]], chatbox_display + [[msg, None]]

    def stream_model_output(img, chatbot, chatbot_display, max_gen_len, gen_t, top_p, img_transform):
        while True:
            content_piece = response_queue.get()
            if isinstance(content_piece, Ready):
                break
        for queue in request_queues:
            queue.put((img, chatbot, max_gen_len, gen_t, top_p, img_transform))
        while True:
            content_piece = response_queue.get()
            if isinstance(content_piece, ModelFailure):
                raise RuntimeError
            chatbot_display[-1][1] = content_piece['text'].replace("<", "&lt;").replace(">", "&gt;")
            if content_piece["end_of_content"]:
                chatbot[-1][1] = content_piece['text']
                boxed_objects, colored_piece = extract_and_color(content_piece['text'])
                chatbot_display[-1][1] = colored_piece
                if img is not None:
                    boxed_image, masked_image = draw_box_mask_on_image(img, boxed_objects, sam_predictor)
                else:
                    boxed_image, masked_image = None, None
                yield chatbot, chatbot_display, boxed_image, masked_image
                break
            # else:
            #     yield chatbot, chatbot_display, None, None

    def undo(chatbot, chatbot_display):
        if len(chatbot) > 0:
            chatbot = chatbot[:-1]
            chatbot_display = chatbot_display[:-1]
        return chatbot, chatbot_display

    def clear():
        chatbot = []
        chatbot_display = []
        msg = ""
        return chatbot, chatbot_display, msg

    with gr.Blocks(css="#image_input {height: 100% !important}") as demo:
        gr.Markdown("# SPHINX-MLLM Demo\n\n"
                    "**General Question Answering:** What's in the image?\n\n"
                    "**Detailed Caption:** Generate a detailed description about the image.\n\n"
                    "**Short Caption:** Provide a one-sentence caption for the provided image.\n\n"
                    "**Referring Expression Comprehension (REC):** Please provide the bounding box coordinate of the region this sentence describes: blue backpack.\n\n"
                    "**Relationship Grounding:** Please provide the bounding box coordinate of the region this sentence describes : people on car.\n\n"
                    "**Grounding Caption:** Describe the image concisely. Include the bounding box for each mentioned object.\n\n"
                    "**Object Detection:** Detect all people shown in the image.\n\n"
                    "**Human Keypoint Detection:** Detect the key points of the person in the region [x1, y1, x2, y2].\n\n"
                    "**Text Detection:** Please detect all texts and provide their bounding box coordinate.\n\n"
                    "**Chart Parsing:** Convert this chart to a table.\n\n"
                    )
        with gr.Row() as r:
            with gr.Column(scale=1):
                img_input = gr.Image(label='Image Input', type='pil', elem_id="image_input")
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(visible=False)
                chatbot_display = gr.Chatbot()
                msg = gr.Textbox()
        with gr.Row():
            submit_button = gr.Button("Submit", variant="primary")
            undo_button = gr.Button("Undo")
            clear_button = gr.ClearButton([chatbot, chatbot_display, msg, img_input])
        with gr.Row():
            max_gen_len = gr.Slider(
                minimum=1, maximum=args.max_seq_len // 4,
                value=args.max_seq_len // 4, interactive=True,
                label="Single-turn max response length",
            )
            gen_t = gr.Slider(
                minimum=0, maximum=1, value=0.1, interactive=True,
                label="Temperature",
            )
            top_p = gr.Slider(
                minimum=0, maximum=1, value=0.75, interactive=True,
                label="Top-p",
            )
            img_transform = gr.Dropdown(choices=["padded_resize", "resized_center_crop"],
                                          value="padded_resize", label="Image Transform", visible=False)
        with gr.Row(equal_height=True):
            image_box = gr.Image(show_label=False, interactive=False)
            image_mask = gr.Image(show_label=False, interactive=False)

        msg.submit(
            show_user_input, [msg, chatbot, chatbot_display], [msg, chatbot, chatbot_display],
        ).then(
            stream_model_output, [img_input, chatbot, chatbot_display, max_gen_len, gen_t, top_p, img_transform],
            [chatbot, chatbot_display, image_box, image_mask]
        )
        submit_button.click(
            show_user_input, [msg, chatbot, chatbot_display], [msg, chatbot, chatbot_display],
        ).then(
            stream_model_output, [img_input, chatbot, chatbot_display, max_gen_len, gen_t, top_p, img_transform],
            [chatbot, chatbot_display, image_box, image_mask]
        )
        undo_button.click(undo, [chatbot, chatbot_display], [chatbot, chatbot_display])
        img_input.change(clear, [], [chatbot, chatbot_display, msg])
    barrier.wait()
    demo.queue(api_open=True, concurrency_count=1).launch(
        share=True,
        server_name="0.0.0.0" if args.bind_all else "127.0.0.1",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("LLaMA2-Accessory Chat Demo")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--gpu_ids", type=int, nargs="+",
        help="A list of space-separated gpu ids to run the model on. "
             "The model will span across GPUs in tensor-parallel mode."
    )
    group.add_argument(
        "--n_gpus", type=int, default=1,
        help="Number of GPUs to run the model on. Equivalent to "
             "--gpu_ids 0 1 2 ... n-1"
    )
    parser.add_argument(
        "--pretrained_path", type=str, required=True, nargs="+",
        help="Path to the llama model checkpoints. A list of checkpoints is "
             "supported and will be merged from left to right.")
    parser.add_argument(
        "--llama_type", default=None, type=str, metavar="MODEL",
        help="LLaMA model type."
    )
    parser.add_argument(
        "--llama_config", type=str, default=None, nargs="*",
        help="Path to the llama model config json."
    )
    parser.add_argument(
        "--tokenizer_path", type=str, default=None,
        help="Path to the tokenizer.model file provided along with the LLaMA "
             "model."
    )
    parser.add_argument(
        "--max_seq_len", type=int, default=4096,
        help="Max sequence length accepted by the pretrained model."
    )
    parser.add_argument(
        "--master_port", type=int, default=23560,
        help="A port used by the PyTorch distributed module to initialize."
    )
    parser.add_argument(
        "--master_addr", type=str, default="127.0.0.1",
        help="An address used by the PyTorch distributed module to initialize."
    )
    parser.add_argument(
        "--dtype", type=str, choices=["fp16", "bf16"], default="bf16",
        help="The dtype used for model weights and inference."
    )
    parser.add_argument(
        "--quant", action="store_true", default=False,
        help="enable quantization"
    )
    parser.add_argument(
        "--bind_all", action="store_true",
        help="Listen to all addresses on the host."
    )
    args = parser.parse_args()

    # check and setup gpu_ids to use
    if args.gpu_ids is None:
        if args.n_gpus is None:
            args.n_gpus = 1
        assert args.n_gpus > 0, (
            "The demo currently must run on a positive number of GPUs."
        )
        args.gpu_ids = list(range(args.n_gpus))

    # using the default "fork" method messes up some imported libs (e.g.,
    # pandas)
    mp.set_start_method("spawn")

    # setup the queues and start the model workers
    request_queues = []
    response_queue = mp.Queue()
    worker_processes = []
    barrier = mp.Barrier(len(args.gpu_ids) + 1)
    for rank, gpu_id in enumerate(args.gpu_ids):
        request_queue = mp.Queue()
        rank_response_queue = response_queue if rank == 0 else None
        process = mp.Process(
            target=model_worker,
            args=(rank, args, barrier, request_queue, rank_response_queue),
        )
        process.start()
        worker_processes.append(process)
        request_queues.append(request_queue)

    gradio_worker(request_queues, response_queue, args, barrier)
