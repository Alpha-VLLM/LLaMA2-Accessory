import copy
import json
import pickle
import random
from threading import Thread

import numpy as np
from torch.utils.data import IterableDataset, get_worker_info, Dataset
import torch

from accessory.model.tokenizer import Tokenizer

from multiprocessing import Manager


class Falcon(IterableDataset):
    def __init__(self, data_meta_path, data_root, tokenizer_path, max_words=None, seed=12345, shuffle=False, num_processes=1, process_rank=0):
        print("use packed dataset")
        with open(data_meta_path, 'r') as f:
            filenames = json.load(f)
            filenames = [f"{data_root}/{_}" for _ in filenames]

        filenames = [_.replace('.parquet', '.pkl') for _ in filenames]

        self._ori_filenames = filenames[:-1] # last used for validation
        self._filenames = self._ori_filenames.copy()

        self._seed = seed
        self._shuffle = shuffle
        self._epoch = 0
        if self._shuffle:
            random.seed(self._seed + self._epoch)
            random.shuffle(self._filenames)

        print("use packed dataset, max_words argument is not in use. Actual seq length is defined by data file")

        self.tokenizer = Tokenizer(model_path=tokenizer_path)

        self._num_processes = num_processes
        self._process_rank = process_rank

        manager = Manager()
        self.state_dict = manager.dict() # for resume

    def set_epoch(self, epoch):
        self._epoch = epoch
        if self._shuffle:
            self._filenames = self._ori_filenames.copy()
            random.seed(self._seed + self._epoch)
            random.shuffle(self._filenames)

    def load_state_dict(self, state_dict: dict):
        for key, val in state_dict.items():
            self.state_dict[key] = val

    def __iter__(self):
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0
        num_shards = num_workers * self._num_processes
        shard_id = self._process_rank * num_workers + worker_id

        max_num_files = len(self._filenames) // num_shards * num_shards
        filenames = self._filenames[shard_id : max_num_files : num_shards]

        print(f"[WORKER] R{self._process_rank:2d}W{worker_id:2d}: filenames first 3 {filenames[:min(3, len(filenames))]}")
        return FalconIterator(
            filenames=filenames,
            seed=self._seed,
            shuffle=self._shuffle,
            tokenizer=self.tokenizer,
            rank_id=self._process_rank,
            worker_id=worker_id,
            state_dict=self.state_dict
        )


class FalconIterator:
    def __init__(self, filenames, seed, shuffle, tokenizer, rank_id, worker_id, state_dict):
        self._seed = seed
        self._shuffle = shuffle
        self._rng = np.random.default_rng(seed) if shuffle else None

        self.print_head = f"[WORKER] R{rank_id:2d}W{worker_id:2d}: "
        self.worker_id = worker_id
        self.rank_id = rank_id

        self._filenames = filenames
        self._file_idx = -1

        self._curr_idx = 0 # current index of data item within current contents

        self.tokenizer = tokenizer

        self._curr_contents = None
        self._pre_cache = None
        self._pre_cache_thread = None

        if len(state_dict) != 0:
            self._file_idx = state_dict[self.worker_id]['_file_idx'] - 1
            self._pre_cache_thread = Thread(target=self._preload_cache)
            self._pre_cache_thread.start()
            self._load_new_file()
            assert self._file_idx == state_dict[self.worker_id]['_file_idx']
            self._curr_idx = state_dict[self.worker_id]['_curr_idx'] + 1
        else:
            self._pre_cache_thread = Thread(target=self._preload_cache)
            self._pre_cache_thread.start()
            self._load_new_file()

    def __iter__(self):
        return self

    def _preload_cache(self):
        if self._file_idx + 1 >= len(self._filenames):
            self._pre_cache = None
        else:
            print(f"{self.print_head} current {self._file_idx}, async load {self._file_idx + 1} {self._filenames[self._file_idx + 1]}")

            with open(self._filenames[self._file_idx + 1], 'rb') as f:
                ann = pickle.load(f)
            self._pre_cache = ann

        return

    def _load_new_file(self, pre_load=True):
        self._pre_cache_thread.join()

        if self._file_idx + 1 >= len(self._filenames):
            assert self._pre_cache is None
            raise StopIteration
        else:
            assert self._pre_cache is not None
            self._curr_contents = self._pre_cache

            self._pre_cache = None
            self._file_idx += 1
            self._curr_idx = 0
            print(
                f"{self.print_head} start to use {self._file_idx} {self._filenames[self._file_idx]}({len(self._curr_contents)})")


        if pre_load:
            self._pre_cache_thread = Thread(target=self._preload_cache)
            self._pre_cache_thread.start()

    def __next__(self):
        if self._curr_idx >= len(self._curr_contents):
            self._load_new_file()

        ann = self._curr_contents[self._curr_idx]
        input_data = torch.tensor(ann, dtype=torch.int64)
        output_data = copy.deepcopy(input_data)

        item_state = {"_curr_idx": self._curr_idx, "_file_idx": self._file_idx, "worker_id": self.worker_id}

        self._curr_idx = self._curr_idx + 1

        return input_data, output_data, item_state



class FalconVal(Dataset):
    def __init__(self, data_meta_path, data_root, tokenizer_path, max_words=None):

        with open(data_meta_path, 'r') as f:
            filenames = json.load(f)
            filenames = [f"{data_root}/{_}" for _ in filenames]

        filenames = [_.replace('.parquet', '.pkl') for _ in filenames]

        filename = filenames[-1]
        print(f"Falcon val filename: {filename}")
        with open(filename, 'rb') as f:
            ann = pickle.load(f)
        self.contents = ann

        self.tokenizer = Tokenizer(model_path=tokenizer_path)

    def __len__(self):
        return len(self.contents)

    def __getitem__(self, idx):
        ann = self.contents[idx]
        input_data = torch.tensor(ann, dtype=torch.int64)
        output_data = copy.deepcopy(input_data)

        return input_data, output_data
