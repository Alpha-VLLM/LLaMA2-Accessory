import os
import argparse
import torch

def get_args_parser():
    parser = argparse.ArgumentParser('Combine or separate the weights of the model.', add_help=False)
    parser.add_argument('--pretrained_path', default='/path/to/pretrained', type=str,
                        help='directory containing pre-trained checkpoints')
    parser.add_argument('--delta_path', default='/path/to/delta', type=str,
                        help='directory containing delta checkpoints')
    parser.add_argument('--output_path', default='./output',
                        help='path where to save')
    
    parser.add_argument('--operate_type', default='apply', choices=['extract', 'apply'])
    return parser

def calculate_weight_delta(original_model, fine_tuned_model):
    original_state_dict = {key: val.float() for key, val in original_model.items()}
    fine_tuned_state_dict = {key: val.float() for key, val in fine_tuned_model['model'].items()}
    delta_state_dict = {}

    for key, val in fine_tuned_state_dict.items():
        if key in original_state_dict:
            delta_state_dict[key] = (val - original_state_dict[key])
        else:
            delta_state_dict[key] = val

    consolidated_model_state_dict = {
        "model": {key: val.half() for key, val in delta_state_dict.items()}
    }
    
    save_path = os.path.join(
        args.output_path,
        f"consolidated.00-of-01.pth",  # TODO fix multi GPU
    )
    
    torch.save(consolidated_model_state_dict, save_path)


def merge_weights_and_save(original_model, delta_weights):
    original_state_dict = {key: val.float() for key, val in original_model.items()}
    delta_weights_dict = {key: val.float() for key, val in delta_weights['model'].items()}
    new_state_dict = {}

    for key, val in original_state_dict.items():
        if key in delta_weights_dict:
            new_state_dict[key] = val + delta_weights_dict[key]
        else:
            new_state_dict[key] = val


    consolidated_model_state_dict = {
        "model": {key: val.half() for key, val in new_state_dict.items()},
    }

    save_path = os.path.join(
        args.output_path,
        f"consolidated.00-of-01.pth",  # TODO fix multi GPU
    )

    torch.save(consolidated_model_state_dict, save_path)



args = get_args_parser().parse_args()


if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)

model_base = torch.load(args.pretrained_path)
model_delta = torch.load(args.delta_path)

if args.operate_type == 'extract':
    calculate_weight_delta(model_base, model_delta)
elif args.operate_type == 'apply':
    merge_weights_and_save(model_base, model_delta)

