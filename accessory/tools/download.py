import argparse
import os
import sys
from huggingface_hub import hf_hub_download

def download_file(repo_id, subfolder, filename, local_dir):
    try:
        hf_hub_download(repo_id=repo_id, repo_type="model", subfolder=subfolder, filename=filename, resume_download=True, local_dir=local_dir)
    except Exception as e:
        if args.down_diff:
            print(f"Error downloading {filename}: {str(e)}. Trying to download non-diff file.")
            download_file(repo_id, subfolder, filename.replace('-diff', ''), local_dir)
        print(f"Error downloading {filename}: {str(e)}. Please check your arguments.")
        exit(1)

def get_args_parser():
    parser = argparse.ArgumentParser('Download the weights of the model.', add_help=False)
    parser.add_argument('--model_name', type=str,
                        help='directory containing pre-trained checkpoints')
    parser.add_argument('--train_type', default='finetune', choices=['finetune', 'pretrain'])
    parser.add_argument('--output_path', default='./output',
                        help='path where to save')
    parser.add_argument('--input_type', default='sg', choices=['sg', 'mm'])
    parser.add_argument('--model_size', default='7B', choices=['7B', '13B', '70B'])
    parser.add_argument('--down_config', action="store_true" ,help='download config')
    parser.add_argument('--down_diff', action="store_true" ,help='download delta weights')
    return parser

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    subfolder = f"{args.train_type}/{args.input_type}/{args.model_name}"
    repo_id = f"Alpha-VLLM/LLaMA2-Accessory"

    if args.down_config:
        download_file(repo_id, 'config', 'tokenizer.model', args.output_path)
        param_file = f"{args.model_size}_params.json"
        download_file(repo_id, 'config', param_file, args.output_path)
        if args.model_name == None:
            sys.exit("Model name not specified, only configuration files were downloaded.")

    num_files_map = {'7B': 1, '13B': 2, '70B': 8}
    max_num = num_files_map[args.model_size]

    for num in range(max_num):
        if args.down_diff:
            file_name = f"consolidated.{num:02d}-of-{max_num:02d}.model-diff.pth"
        else:
            file_name = f"consolidated.{num:02d}-of-{max_num:02d}.model.pth"
        download_file(repo_id, subfolder, file_name, args.output_path)

    print(f"{args.model_name} model files downloaded successfully to {args.output_path}")
