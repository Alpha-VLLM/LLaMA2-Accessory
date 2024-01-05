from SPHINX import SPHINXModel
from PIL import Image
import os
import torch
import torch.distributed as dist


def main() -> None:
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ["RANK"])
    dist.init_process_group(
        world_size=world_size, rank=rank,
        backend="nccl", init_method=f"env://",
    )
    torch.cuda.set_device(rank)

    # mp_group tells the model which ranks will work together
    # through model parallel to compose a complete model.
    # When mp_group is None, a single-rank process group will
    # be created and used, which means model parallel size = 1 (not enabled)
    model = SPHINXModel.from_pretrained(
        pretrained_path="/path/to/pretrained", with_visual=True,
        mp_group=dist.new_group(ranks=list(range(world_size)))
    )
    # You may also, say, launch 4 processes and make [0,1] and [2,3] ranks to form mp groups, respectively.

    # it's important to make sure that ranks within the same 
    # model parallel group should always receive the same input simultaneously
    image = Image.open("examples/1.jpg")
    qas = [["What's in the image?", None]]

    response = model.generate_reponse(qas, image, max_gen_len=1024, temperature=0.9, top_p=0.5, seed=0)
    print(response)

    # if you wanna continue
    qas[-1][-1] = response
    qas.append(["Then how does it look like?", None])
    response2 = model.generate_reponse(qas, image, max_gen_len=1024, temperature=0.9, top_p=0.5, seed=0)
    print(response2)


if __name__ == "__main__":
    # launch this script with `torchrun --master_port=1112 --nproc_per_node=2 inference.py`
    main()