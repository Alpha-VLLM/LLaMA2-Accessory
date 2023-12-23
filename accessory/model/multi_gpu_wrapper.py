import atexit
import sys
import os
import warnings
import random
import numpy as np
import builtins
import traceback
import subprocess
import torch
import torch.distributed as dist
from typing import List, Optional, Tuple

import accessory.util.misc
from accessory.model.meta import MetaModel


REQUESTS_WITH_STREAM_RESPONSE = [
    "stream_generate",
]

SPECIAL_REQUESTS = [
    "continue_yield", "stop_yield", "reset_status"
]

def init_print(is_master):
    builtin_print = builtins.print
    def _print(*args, **kwargs):
        force = kwargs.pop('force', False)
        kwargs['flush'] = True # flush should always be true
        if is_master or force:
            builtin_print('[model process] ', end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = _print

class ResetException(Exception):
    pass

class TerminateException(Exception):
    pass

def process_special_request(request: str):
    if request == "reset_status":
        raise ResetException
    elif request == "terminate":
        raise TerminateException

def model_worker(port: int, rank:int, world_size: int):

    def put_response(response):
        dist.gather_object(response, object_gather_list=None, dst=0)

    def get_request() -> List:
        _ = [[]]
        dist.broadcast_object_list(_, src=0)
        assert len(_) == 1
        _ = _[0]
        return _

    # specify random seed to ensure consistent token sampling among model parallel ranks
    random.seed(0)
    torch.random.manual_seed(0)
    np.random.seed(0)

    store = dist.TCPStore("127.0.0.1", port, world_size, False)

    dist.init_process_group(
        backend="gloo", rank=rank, world_size=world_size,
        # init_method=f"tcp://127.0.0.1:{port}",
        store=store
    )

    size = dist.get_world_size()
    rank = dist.get_rank()

    gpu_ids, from_pretrained_args, from_pretrained_kwargs = get_request()
    torch.cuda.set_device(gpu_ids[rank-1])

    init_print(rank==1)
    from_pretrained_kwargs['mp_group'] = dist.new_group(ranks=list(range(1, size)), backend="nccl")
    # mp_group identifies which ranks will work collaboratively through model parallelism
    model = MetaModel.from_pretrained(*from_pretrained_args, **from_pretrained_kwargs)

    dist.barrier()


    while True:
        try:
            request_type, (request_args, request_kwargs) = get_request()
            process_special_request(request_type)

            if request_type not in REQUESTS_WITH_STREAM_RESPONSE:
                result = getattr(model, request_type)(*request_args, **request_kwargs)
                put_response(("SUCCESS", result))
            else:
                for stream_response in getattr(model, request_type)(*request_args, **request_kwargs):
                    put_response(("YIELDING", stream_response))
                    yield_request, _ = get_request()
                    process_special_request(request_type)
                    if yield_request == "continue_yield":
                        continue
                    elif yield_request == "stop_yield":
                        raise ResetException
                    else:
                        raise ValueError(f"Unexpected request type during yield: {yield_request}")
                put_response(("YIELD_END", None))
        except ResetException:
            continue
        except TerminateException:
            exit(0)
        except Exception as e:
            if isinstance(e, RuntimeError) and "connection closed by peer" in str(e).lower():
                exit(0)
            else:
                put_response(("FAIL", traceback.format_exc()))


def _reset_world():
    for key in [
        "_pg_map", "_pg_names", "_pg_group_ranks", "_pg_backend_config", "_pg_to_tag", "_tags_to_pg"
    ]:
        if hasattr(dist.distributed_c10d, key):
            setattr(dist.distributed_c10d, key, {})
    setattr(dist.distributed_c10d, "_group_count", 0)
    dist.distributed_c10d._world = dist.distributed_c10d._World()

def _save_world():
    save_dict = {}
    for key in [
        "_pg_map", "_pg_names", "_pg_group_ranks", "_pg_backend_config", "_pg_to_tag", "_tags_to_pg"
    ]:
        if hasattr(dist.distributed_c10d, key):
            save_dict[key] = getattr(dist.distributed_c10d, key)
    save_world = dist.distributed_c10d._world
    return save_dict, save_world

def _load_world(load_dict, load_world):
    for key, val in load_dict.items():
        setattr(dist.distributed_c10d, key, val)
    dist.distributed_c10d._world = load_world

class MultiGpuWrapper:
    """
    A wrapper class for mimicking the single-process-multi-gpu model hosting behavior.
    Background: LLaMA2-Accessory is designed following the single-process-single-gpu principle. To split
    the load of a complete model onto different GPUs, the normal practice is to launch `n` processes,
    instantiate a model on each of them (each model instance only contains a part of the complete model),
    and introduce cross-process communication to the inner computations of the (incomplete) models
    for presenting the same behavior as one complete model.
    The aforementioned approach requires the manual launching and manipulating of multiple processes.
    In contrast, in most inference and evaluation cases, users may want to focus on the single main
    process only, and may want to operate with a model as if it is complete and non-distributed.
    This wrapper implements the aforementioned abstract, so that users can operate with the wrapper
    as if they are operating with the complete model, even if in fact the complete model is split
    and distributed on multi GPUs. This is achieved by internally creating separate processes
    for each GPU, instantiating in-complete models, and managing communication between them.

    :param from_pretrained_args: Positional arguments to be passed to the model's 'from_pretrained' method.
    :param gpus: Optional; the number of GPUs to be used. If None, `gpu_ids` must be provided.
    :param gpu_ids: Optional; specific GPU IDs to be used. If None, `gpus` must be specified.
    :param from_pretrained_kwargs: Keyword arguments to be passed to the model's 'from_pretrained' method.

    :Examples:
    >>> from accessory.model.multi_gpu_wrapper import MultiGpuWrapper
    >>> # split the model with model parallel size = 4, distribute the 4 parts to 4 processes, each with one GPU
    >>> wrapper = MultiGpuWrapper(
    >>>     pretrained_path="/path/to/pretrained", gpu_ids=[4,5,6,7], max_seq_len=2048, with_visual=True, quant=False
    >>> )
    >>> wrapper.generate(["The best programming language is"], max_gen_len=10, temperature=0)
    """
    def __init__(self,
                 *from_pretrained_args,
                 gpus:Optional[int]=None, gpu_ids:Optional[List[int]]=None,
                 **from_pretrained_kwargs):
        if gpus is None and gpu_ids is None:
            raise ValueError('You must specify either gpus or gpu_ids')

        if gpu_ids is None:
            gpu_ids = list(range(gpus))

        self.outer_world = _save_world()
        _reset_world()

        # launch model worker subprocesses and create inner world
        port = accessory.util.misc.find_free_port(10000+int(os.getpid())%100*100, 10100+int(os.getpid())%100*100)
        print(f"Launching {len(gpu_ids)} processes for hosting model with model parallel size {len(gpu_ids)}")
        print("Note that only the output from the FIRST model process will be printed")
        self.model_workers = []
        for rank in range(1, len(gpu_ids) + 1):
            p = subprocess.Popen([sys.executable, __file__, str(port), str(rank), str(len(gpu_ids)+1)])
            self.model_workers.append(p)

        atexit.register(self.on_exit)

        store = dist.TCPStore("127.0.0.1", port, len(gpu_ids) + 1, True)
        dist.init_process_group(
            backend="gloo", rank=0, world_size=len(gpu_ids) + 1,
            # init_method=f"tcp://127.0.0.1:{port}",
            store=store
        )
        self.inner_world = _save_world()


        dist.broadcast_object_list([[gpu_ids, from_pretrained_args, from_pretrained_kwargs]], src=0)
        dist.new_group(ranks=list(range(1, len(gpu_ids) + 1)), backend="nccl")
        dist.barrier()

        _load_world(*self.outer_world)

    def compute_logits(self, *args, **kwargs):
        """
        Wraps `accessory.model.MetaModel.compute_logits`

        :param args: Positional arguments for 'MetaModel.compute_logits'.
        :param kwargs: Keyword arguments for 'MetaModel.compute_logits'.
        :return: The result returned by 'MetaModel.compute_logits' method.
        """
        return self.call_model_func("compute_logits", *args, **kwargs)

    def evaluate_examples(self, *args, **kwargs):
        """
        Wraps `accessory.model.MetaModel.evaluate_examples`

        :param args: Positional arguments for 'MetaModel.evaluate_examples'.
        :param kwargs: Keyword arguments for 'MetaModel.evaluate_examples'.
        :return: The result returned by 'MetaModel.evaluate_examples' method.
        """
        return self.call_model_func("evaluate_examples", *args, **kwargs)

    def generate(self, *args, **kwargs):
        """
        Wraps `accessory.model.MetaModel.generate`

        :param args: Positional arguments for 'MetaModel.generate'.
        :param kwargs: Keyword arguments for 'MetaModel.generate'.
        :return: The result returned by 'MetaModel.generate' method.
        """
        return self.call_model_func("generate", *args, **kwargs)

    def stream_generate(self, *args, **kwargs):
        """
        Wraps `accessory.model.MetaModel.stream_generate`

        :param args: Positional arguments for 'MetaModel.stream_generate'.
        :param kwargs: Keyword arguments for 'MetaModel.stream_generate'.
        :return: A streaming generator that can be used in the same way as that
         returned by 'MetaModel.stream_generate' method.
        """
        return self.call_model_stream_func("stream_generate", *args, **kwargs)

    @property
    def tokenizer(self):
        """
        Wraps `accessory.model.MetaModel.tokenizer`
        return: The tokenizer of the model
        """
        return self.call_model_func('__getattribute__', "tokenizer")

    def call_model_func(self, request_name:str, *args, **kwargs):
        self.reset_status()
        self._emit_request(request_name, *args, **kwargs)
        response_signal, contents = self._get_response()
        if response_signal == "SUCCESS":
            return contents
        elif response_signal == "FAIL":
            trace = contents
            raise Exception(trace)
        else:
            raise ValueError("Unexpected response signal {}".format(response_signal))

    def call_model_stream_func(self, request_name:str, *args, **kwargs):
        self.reset_status()
        assert request_name in REQUESTS_WITH_STREAM_RESPONSE
        self._emit_request(request_name, *args, **kwargs)
        while True:
            response_signal, contents = self._get_response()
            if response_signal == "YIELDING":
                try:
                    yield contents
                except GeneratorExit:
                    self._emit_request("stop_yield")
                    raise
                else:
                    self._emit_request("continue_yield")
            elif response_signal == "YIELD_END":
                return
            elif response_signal == "FAIL":
                trace = contents
                raise Exception(trace)
            else:
                raise ValueError("Unexpected response signal {}".format(response_signal))

    def _emit_request(self, request_name:str, *args, **kwargs):
        self.outer_world = _save_world()
        _load_world(*self.inner_world)

        dist.broadcast_object_list([[request_name, (args, kwargs)]], src=0)

        self.inner_world = _save_world()
        _load_world(*self.outer_world)

    def _get_response(self):
        self.outer_world = _save_world()
        _load_world(*self.inner_world)

        response = [[] for _ in range(dist.get_world_size())]
        dist.gather_object(None, response, dst=0)

        self.inner_world = _save_world()
        _load_world(*self.outer_world)
        return response[1]

    def reset_status(self):
        self._emit_request("reset_status")

    def on_exit(self):
        self._emit_request("terminate")
        for p in self.model_workers:
            p.kill()


if __name__ == '__main__':
    model_worker(port=int(sys.argv[1]), rank=int(sys.argv[2]), world_size=int(sys.argv[3]))