import sys
import argparse
import os
sys.path.append(os.path.abspath(__file__).rsplit('/', 3)[0])
from accessory.model.meta import MetaModel
from accessory.data.system_prompt import format_prompt

BLUE = '\033[94m'
END = '\033[0m'

def main(pretrained_path):
    model = MetaModel.from_pretrained(pretrained_path, max_seq_len=2048)
    return model
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run single model CLI")
    parser.add_argument("--path", help="Path to the pretrained model or hf_repo id", required=False)
    parser.add_argument("--prompt", help="Instruction prompt", required=False)
    args = parser.parse_args()

    if args.path:
        model = main(args.path)
    else:
        pretrained_path = input(f"{BLUE}Enter the path to the pretrained model: {END}")
        model = main(pretrained_path)

    if args.prompt:
        instruction = args.prompt
    else:
        instruction = input(f"{BLUE}Enter your instruction: {END}")
    prompt = format_prompt({'instruction': instruction}, sys_name='alpaca')
    while True:
        response = model.generate([prompt], images=None, max_gen_len=512)[0]
        print("Response:", response)
        prompt = input(f"{BLUE}Enter your instruction (or type 'exit' to quit): {END}")
        if prompt == 'exit':
            break
