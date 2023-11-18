import sys
import os
sys.path.append(os.path.abspath(__file__).rsplit('/', 3)[0])

import glob
import os
import pandas as pd
import tqdm
from multiprocessing import Pool
from accessory.model.tokenizer import Tokenizer
import pickle


def pack_tokens(filename, save_dir, tokenizer):
    print(f"{filename} start")

    texts = pd.read_parquet(filename)['content'].tolist()

    l_packed_tokens = []
    _idx = 0
    _cache = [0 for _ in range(max_len)]

    for t in texts:
        token_split = tokenizer.encode(t, bos=True, eos=True)

        if token_split[0] == 1:
            token_split = token_split[1:]

        while _idx + len(token_split) > max_len:
            part_len = max_len - _idx
            _cache[_idx: _idx + part_len] = token_split[:part_len]
            assert _idx + part_len == max_len
            l_packed_tokens.append(_cache)
            _idx = 0
            _cache = [0 for _ in range(max_len)]
            token_split = token_split[part_len:]

        remaining_len = len(token_split)
        _cache[_idx:_idx + remaining_len] = token_split
        _idx += remaining_len
        assert _cache[_idx - 1] == 2

    l_packed_tokens.append(_cache)

    save_tokens_path = os.path.join(save_dir, os.path.basename(filename).split('.')[0] + '.pkl')
    with open(save_tokens_path, 'wb') as f:
        pickle.dump(l_packed_tokens, f)

    print(f"{save_tokens_path} finished")
    return



# Modify arguments here based on your needs
max_len = 2048
tokenizer = Tokenizer('./tokenizer.model')
files = glob.glob('data/*.parquet')
files.sort()
save_dir = "packed_tokens"
os.makedirs(save_dir, exist_ok=True)


pool = Pool(48)

for file in tqdm.tqdm(files):
    if not os.path.exists(os.path.join(save_dir, os.path.basename(file).split('.')[0] + '.pkl')):
        print(file)
        outs = pool.apply_async(pack_tokens, [file, save_dir, tokenizer])

pool.close()
pool.join()