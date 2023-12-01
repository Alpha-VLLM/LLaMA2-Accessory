import os
import argparse
import torch

def get_args_parser():
    parser = argparse.ArgumentParser('Combine or separate the weights of the model.', add_help=False)
    parser.add_argument('--pretrained_path', default='/path/to/pretrained', type=str,
                        help='directory containing pretrained checkpoints')
    parser.add_argument('--delta_path', default='/path/to/delta', type=str,
                        help='directory containing delta checkpoints')
    parser.add_argument('--output_path', default='./output',
                        help='path where to save')
    
    parser.add_argument('--operate_type', default='apply', choices=['extract', 'apply'])
    return parser

def calculate_weight_delta(original_model, fine_tuned_model, num, max_num):
    original_state_dict = {key: val.float() for key, val in original_model.items()}
    fine_tuned_state_dict = {key: val.float() for key, val in fine_tuned_model['model'].items()}
    delta_state_dict = {}

    for key, val in fine_tuned_state_dict.items():
        delta_state_dict[key] = val - original_state_dict.get(key[5:], 0)


    consolidated_model_state_dict = {
        "model": {key: val.half() for key, val in delta_state_dict.items()}
    }
    
    save_path = os.path.join(
        args.output_path,
        f"consolidated.{num:02d}-of-{max_num:02d}.model-diff.pth", 
    )
    
    torch.save(consolidated_model_state_dict, save_path)


def merge_weights_and_save(original_model, delta_weights, num, max_num):
    original_state_dict = {key: val.float() for key, val in original_model.items()}
    delta_weights_dict = {key: val.float() for key, val in delta_weights['model'].items()}
    new_state_dict = {}

    for key, val in original_state_dict.items():
        new_state_dict['llma.'+key] = val
    for key, val in delta_weights_dict.items():
        new_state_dict[key] = val + new_state_dict.get(key, 0)


    consolidated_model_state_dict = {
        "model": {key: val.half() for key, val in new_state_dict.items()},
    }

    save_path = os.path.join(
        args.output_path,
        f"consolidated.{num:02d}-of-{max_num:02d}.model.pth",  
    )

    torch.save(consolidated_model_state_dict, save_path)



args = get_args_parser().parse_args()


if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)

pretrained_list = [path for path in os.listdir(args.pretrained_path) if path.startswith("consolidated.") and path.endswith(".pth")]
pretrained_list.sort()
delta_list = [path for path in os.listdir(args.delta_path) if path.startswith("consolidated.") and path.endswith(".pth")]
delta_list.sort()

assert len(pretrained_list) == len(delta_list)  
max_checkpoint = len(pretrained_list)
print(f"Found {max_checkpoint} checkpoints in {args.pretrained_path} and {args.delta_path}")

for i in range(max_checkpoint):
    print(f"Processing checkpoint {i+1}/{max_checkpoint}")
    model_base = torch.load(os.path.join(args.pretrained_path, pretrained_list[i]))
    model_delta = torch.load(os.path.join(args.delta_path, delta_list[i]))

    if args.operate_type == 'extract':
        calculate_weight_delta(model_base, model_delta, i , max_checkpoint)
    else:
        merge_weights_and_save(model_base, model_delta, i , max_checkpoint)

