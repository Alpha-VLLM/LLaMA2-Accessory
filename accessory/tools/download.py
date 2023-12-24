import argparse
import os
from huggingface_hub import hf_hub_download
from huggingface_hub import snapshot_download

def colored(text, color):
    color_map = {'yellow': "\033[93m", 'green': "\033[92m", 'red': "\033[91m"}
    return f"{color_map.get(color, '')}{text}\033[0m"
model_list = {
    'convert': {
        'sg': ['InternLM','Falcon','Falcon_180b','mixtral-8x7b-32kseqlen']
    },
    'finetune': {
        'mm': ['alpacaLlava_llamaQformerv2', 'alpacaLlava_llamaQformerv2_13b', 'alpacaLlava_llamaQformerv2Peft_13b', 'caption_llamaQformerv2', 'caption_llamaQformerv2_13b', 'SPHINX/SPHINX-1k','SPHINX/SPHINX','SPHINX/SPHINX-v2-1k'],
        'sg': ['alpaca', 'alpaca_internLM_en', 'alpaca_internLM_zh', 'alpaca_llamaPeft_normBias', 'dialog_flan', 'dialog_lima', 'dialog_moss', 'dialog_platypus', 'dialog_sharegpt', 'dialog_sharegpt_70b', 'dialog_ultra', 'dialog_wizardcode', 'dialog_wizardcode_codellama', 'dialog_wizardcode_loadcode220k', 'dialog_wizardLM', 'gorilla']
    }
}

def download_file(repo_id, subfolder, filename, local_dir):
    try:
        hf_hub_download(repo_id=repo_id, repo_type="model", subfolder=subfolder, filename=filename, resume_download=True, local_dir=local_dir,local_dir_use_symlinks=False)
        print(f"{filename} downloaded successfully.")
    except Exception as e:
        print(f"Error downloading {filename}: {str(e)}. Please check your arguments.")
        exit(1)

def download_files(repo_id, subfolder, file_names, output_path):
    for file_name in file_names:
        download_file(repo_id, subfolder, file_name, output_path)

def get_file_names(prefix, model_size):
    return [prefix + 'tokenizer.model', prefix + f"{model_size}_params.json"]

def ask_question(prompt, options, default_value=None):
    while True:
        print(colored(prompt, 'yellow'))
        for i, option in enumerate(options):
            print(f"{i+1}: {colored(option, 'green')}")
        choice = input(f"Please enter the option number (default is {default_value}): ")
        if not choice and default_value:
            return default_value
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice) - 1]
        else:
            print(colored("Invalid input, please try again.", 'red'))

def interactive_mode(args):
    args.train_type = args.train_type or ask_question("Choose a train type:", ['finetune', 'convert'])
    args.input_type = args.input_type or ask_question(f"\nChoose an input type for {args.train_type}:", ['sg', 'mm'])
    

    models = model_list.get(args.train_type, {}).get(args.input_type, [])
    if models:
        args.model_name = args.model_name or ask_question("\nChoose a model:", models)

    args.model_size = args.model_size or ask_question("\nChoose a model size:", ['7B', '13B', '34B', '70B', '180B'])
    config_choice = ask_question("\nDownload which version of params.json and tokenizer.model?", ['LLaMA2', 'InterLM', 'CodeLlama', 'SPHINX', 'no'])
    if config_choice != 'no':
        args.down_config = True
        args.down_internLM = (config_choice == 'InterLM')
        args.down_code = (config_choice == 'CodeLlama')
        args.down_sphinx = (config_choice == 'SPHINX')
    else:
        args.down_config = False

    args.down_diff = args.down_diff
    args.output_path = args.output_path or input("\nPlease enter the output path (default is ./output): ") or './output'

def get_args_parser():
    parser = argparse.ArgumentParser('Download the weights of the model.', add_help=False)
    parser.add_argument('--train_type', default=None, choices=['finetune', 'convert'])
    parser.add_argument('--input_type', default=None, choices=['sg', 'mm'])
    parser.add_argument('--model_size', default=None, choices=['7B', '13B', '34B', '70B','180B'])
    parser.add_argument('--model_name', default=None, type=str)
    parser.add_argument('--down_config', default=None, action="store_true")
    parser.add_argument('--down_diff', default=None, action="store_true")
    parser.add_argument('--down_internLM', default=None, action="store_true")
    parser.add_argument('--down_code', default=None, action="store_true")
    parser.add_argument('--output_path', default=None)
    return parser

def main():
    args = get_args_parser().parse_args()
    interactive_mode(args)
    
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    subfolder = f"{args.train_type}/{args.input_type}/{args.model_name}"
    repo_id = f"Alpha-VLLM/LLaMA2-Accessory"

    if args.down_config:
        files_to_download = []
        configfolder = 'config'
        prefix = ''

        if args.down_sphinx:
            files_to_download = ['config.json', 'tokenizer.model', 'meta.json']
            prefix = 'sphinx_'
            configfolder = subfolder
        elif args.down_internLM:
            prefix = 'internLM_'
        elif args.down_code:
            prefix = 'code_'

        if prefix != 'sphinx_':
            files_to_download.extend(get_file_names(prefix, args.model_size))

        download_files(repo_id, configfolder, files_to_download, args.output_path)


    snapshot_download(repo_id=repo_id, allow_patterns=f"{subfolder}/*", repo_type='model', local_dir=args.output_path, local_dir_use_symlinks=False, resume_download=True)
 

if __name__ == '__main__':
    main()
