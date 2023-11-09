import sys
import os
sys.path.append(os.path.join(os.path.abspath(__file__).rsplit('/', 2)[0], 'accessory'))  # LLaMA2-Accessory/accessory

import numpy as np
import torch
import torch.distributed as dist
import multiprocessing as mp

from fairscale.nn.model_parallel import initialize as fs_init

from model.meta import MetaModel

from util.misc import setup_for_distributed
from util.tensor_parallel import load_tensor_parallel_model_list
from util.tensor_type import default_tensor_type
from data.transform import get_transform
from data.conversation.lib import conv_templates, SeparatorStyle
from PIL import Image



# *********************** Begin Configuration ***********************
# todo Please modify the following variables to fit your environment and needs

SPHINX_TYPE = "Long-SPHINX"  # "SPHINX" or "Long-SPHINX"
TOKENIZER_PATH = "PATH/TO/tokenizer.model"  # path to llama tokenizer
PRETRAINED_PATH = "/PATH/TO/PRETRAINED"

IMAGE_PATH = "examples/1.jpg"  # image for inference, can be None as SPHINX also supports text-only dialog
INSTRUCTION = "Please describe the image in detail."

N_GPU = 2  # number of GPUs to use for inference, current either 1 or 2 is supported

# ************************ End Configuration ************************


if SPHINX_TYPE == "Long-SPHINX":
    LLAMA_TYPE = "llama_ens5" # Long-SPHINX
elif SPHINX_TYPE == "SPHINX":
    LLAMA_TYPE = "llama_ens"
else:
    raise ValueError(f"unknown SPHINX type {SPHINX_TYPE}")


def main(world_size=1, rank=0) -> None:
    # ****************** Begin Environment Setup ******************

    # SPHINX model definition relies on fairscale model parallel
    # so distributed process groups need to be initialized even if when only one gpu is used
    dist.init_process_group(
        backend="nccl", rank=rank, world_size=world_size,
        init_method=f"tcp://127.0.0.1:23560",
    )
    fs_init.initialize_model_parallel(world_size)
    torch.cuda.set_device(rank)

    torch.manual_seed(1)
    np.random.seed(1)

    # block output from ranks other than rank 0
    setup_for_distributed(rank == 0)

    # ******************* End Environment Setup *******************


    # ******************** Begin Create Model *********************

    with default_tensor_type(dtype=torch.float16, device="cuda"):
        model = MetaModel(
            LLAMA_TYPE, llama_config=[], tokenizer_path=TOKENIZER_PATH,
            with_visual=True, max_seq_len=4096,
        )
    print("Loading pretrained weights ...")
    load_result = load_tensor_parallel_model_list(model, [PRETRAINED_PATH])
    print("load result:\n", load_result)
    assert load_result == {'missing_keys': [], 'unexpected_keys': []}, "checkpoint and model mismatch"
    model.eval()

    # ********************* End Create Model *********************


    # ****************** Begin Construct Input *******************

    if IMAGE_PATH is not None:
        image = Image.open(IMAGE_PATH)
        image = image.convert("RGB")
        target_size = getattr(model.llma, 'image_size', 224)  # 448 for Long-SPHINX, 224 for SPHINX
        image = get_transform("padded_resize", target_size)(image).unsqueeze(0).cuda().half()
    else:
        image = None  # SPHINX also supports text-only dialog

    conv = conv_templates["v1"].copy()

    conv.append_message("Human", INSTRUCTION)
    conv.append_message("Assistant", None)

    prompt = conv.get_prompt()
    print(prompt)

    # ******************* End Construct Input ********************


    # ********************* Begin Inference **********************

    conv_sep = (
        conv.sep
        if conv.sep_style == SeparatorStyle.SINGLE
        else conv.sep2
    )  # conv_sep marks the end of response

    with torch.cuda.amp.autocast(dtype=torch.float16):

        for stream_response in model.stream_generate(
            prompt, image, max_gen_len=512, temperature=0.1, top_p=0.75
        ):
            end_pos = stream_response["text"].find(conv_sep)
            if end_pos != -1:  # response ends
                stream_response["text"] = (
                    stream_response['text'][:end_pos].rstrip() + "\n"
                )
                break

    print(stream_response['text'])

    # ********************** End Inference ***********************


if __name__ == "__main__":
    if N_GPU == 1:
        main(world_size=1, rank=0)
    elif N_GPU == 2:
        mp.set_start_method("spawn")
        for rank in range(N_GPU):
            process = mp.Process(
                target=main,
                args=(N_GPU, rank),
            )
            process.start()
    else:
        raise ValueError("Currently only 1 or 2 is supported for N_GPU")
