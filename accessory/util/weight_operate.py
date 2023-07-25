import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from model import meta

from fairscale.nn.model_parallel import initialize as fs_init
from util import misc
import argparse

import torch

def get_args_parser():
    parser = argparse.ArgumentParser('Combine or separate the weights of the model.', add_help=False)
    # Model parameters
    parser.add_argument('--llama_type', default='llama', type=str, metavar='MODEL', choices=['llama'],
                        help='type of llama')
    parser.add_argument('--llama_config', default='/path/to/params.json', type=str,
                        help='Path to llama model config')
    parser.add_argument('--tokenizer_path', type=str, default="../tokenizer.model",
                        help='path to tokenizer.model')
    parser.add_argument('--pretrained_path', default='/path/to/pretrained', type=str,
                        help='directory containing pre-trained checkpoints')
    parser.add_argument('--pretrained_type', type=str, default="meta_ori", choices=['consolidated', 'meta_ori'],
                        help='pretrained checkpoint save format')
    parser.add_argument('--delta_path', default='/path/to/delta', type=str,
                        help='directory containing delta checkpoints')
    parser.add_argument('--delta_type', type=str, default="consolidated", choices=['consolidated', 'meta_ori'],
                        help='delta checkpoint save format')
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save')

    parser.add_argument('--device', default='cuda',
                        help='device for inference')
    parser.add_argument('--model_parallel_size', default=1, type=int)

    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    
    parser.add_argument('--operate_type', default='apply', choices=['extract', 'apply'])
    return parser

def calculate_weight_delta(original_model, fine_tuned_model):
    original_state_dict = original_model.state_dict()
    fine_tuned_state_dict = fine_tuned_model.state_dict()
    delta_state_dict = {}
    mp_rank = fs_init.get_model_parallel_rank()
    mp_world_size = fs_init.get_model_parallel_world_size()
    for key, val in original_state_dict.items():
        if key in fine_tuned_state_dict:
            delta_state_dict[key] = (fine_tuned_state_dict[key] - val)

    consolidated_model_state_dict = {
        "model": delta_state_dict
    }
    
    save_path = os.path.join(
        args.output_dir,
        f"consolidated.{mp_rank:02d}-of-{mp_world_size:02d}.model.pth",
    )
    
    if fs_init.get_data_parallel_rank() == 0:
        torch.save(consolidated_model_state_dict, save_path)


def merge_weights_and_save(original_model, delta_weights):
    original_state_dict = original_model.state_dict()
    delta_weights_dict = delta_weights.state_dict()
    new_state_dict = {}

    for key, val in original_state_dict.items():
        if key in delta_weights_dict:
            new_state_dict[key] = val + delta_weights_dict[key]
        else:
            new_state_dict[key] = val

    original_model.load_state_dict(new_state_dict)

    consolidated_model_state_dict = {
        "model": {key: val for key, val in original_model.state_dict().items()},
    }
    mp_rank = fs_init.get_model_parallel_rank()
    mp_world_size = fs_init.get_model_parallel_world_size()
    save_path = os.path.join(
        args.output_dir,
        f"consolidated.{mp_rank:02d}-of-{mp_world_size:02d}.model.pth",
    )

    if fs_init.get_data_parallel_rank() == 0:
        torch.save(consolidated_model_state_dict, save_path)

    return original_model


args = get_args_parser().parse_args()

misc.init_distributed_mode(args)
fs_init.initialize_model_parallel(args.model_parallel_size)
model_base = meta.MetaModel(args.llama_type, args.llama_config, args.tokenizer_path, with_visual=False)
misc.load_pretrained(args.pretrained_path, args.pretrained_type, model_base)
model_delta = meta.MetaModel(args.llama_type, args.llama_config, args.tokenizer_path, with_visual=False)
misc.load_pretrained(args.delta_path, args.delta_type, model_delta)

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

if args.operate_type == 'extract':
    calculate_weight_delta(model_base, model_delta)
elif args.operate_type == 'apply':
    merge_weights_and_save(model_base, model_delta)

