import sys
import os
sys.path.append(os.path.abspath(__file__).rsplit('/', 3)[0])
import argparse
from PIL import Image
from accessory.model.meta import MetaModel
from accessory.data.transform import get_transform
from accessory.data.system_prompt import format_prompt

BLUE = '\033[94m'
END = '\033[0m'

def load_image(image_path, model):
    transform_type = "padded_resize"  # or "resized_center_crop"
    transform = get_transform(transform_type, getattr(model.llma, 'image_size', 224))
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0).cuda().bfloat16()

def main(model, image_path, instruction):
    image = load_image(image_path, model)
    prompt = format_prompt({'instruction': instruction}, sys_name='alpaca')
    response = model.generate([prompt], images=image, max_gen_len=512)[0]
    print("Response:", response)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multi-modal model CLI")
    parser.add_argument("--path", help="Path to the pretrained model or hf_repo id")
    parser.add_argument("--image", help="Path to the image file")
    parser.add_argument("--prompt", help="Instruction prompt")
    args = parser.parse_args()

    if args.path:
        model = MetaModel.from_pretrained(args.path, with_visual=True, max_seq_len=2048)
    else:
        pretrained_path = input(f"{BLUE}Enter the path to the pretrained model: {END}")
        model = MetaModel.from_pretrained(pretrained_path, with_visual=True, max_seq_len=2048)

    while True:
        image_path = args.image if args.image else input(f"{BLUE}Enter the path to the image: {END}")
        if image_path.lower() == 'exit':
            break
        instruction = args.prompt if args.prompt else input(f"{BLUE}Enter your instruction: {END}")
        if instruction.lower() == 'exit':
            break
        main(model, image_path, instruction)
