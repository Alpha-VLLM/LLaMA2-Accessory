from enum import Enum, auto
import random
import numpy as np
import builtins
import multiprocessing as mp
mp = mp.get_context("spawn")
import traceback
import torch
import torch.distributed as dist
from typing import List, Optional

import accessory.util.misc
from .meta import MetaModel


class ResponseSignal(Enum):
    READY = auto()
    SUCCESS = auto()
    YIELDING = auto()
    YIELD_END = auto()
    FAIL = auto()

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
            builtin_print('[model process]', end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = _print


def model_worker(
        from_pretrained_args, from_pretrained_kwargs,
        world_size, rank, gpu_id, port,
        barrier: mp.Barrier, request_queue: mp.Queue, response_queue: Optional[mp.Queue],
):
    # specify random seed to ensure consistent token sampling among model parallel ranks
    random.seed(0)
    torch.random.manual_seed(0)
    np.random.seed(0)
    torch.cuda.set_device(gpu_id)
    dist.init_process_group(
        backend="nccl", rank=rank, world_size=world_size,
        init_method=f"tcp://127.0.0.1:{port}",
    )
    init_print(dist.get_rank()==0)

    from_pretrained_kwargs['mp_group'] = dist.GroupMember.WORLD
    # mp_group identifies which ranks will work collaboratively through model parallelism
    model = MetaModel.from_pretrained(*from_pretrained_args, **from_pretrained_kwargs)

    barrier.wait()

    def put_response(response):
        if response_queue is not None:
            response_queue.put(response)
        else:
            pass

    with torch.inference_mode():
        while True:
            try:
                request_type, (request_args, request_kwargs) = request_queue.get()
                if request_type == "reset_status":
                    continue
                if request_type not in REQUESTS_WITH_STREAM_RESPONSE:
                    result = getattr(model, request_type)(*request_args, **request_kwargs)
                    put_response((ResponseSignal.SUCCESS, result))
                else:
                    need_yield_end = True
                    for stream_response in getattr(model, request_type)(*request_args, **request_kwargs):
                        put_response((ResponseSignal.YIELDING, stream_response))
                        yield_request, _ = request_queue.get()
                        if yield_request == "continue_yield":
                            continue
                        elif yield_request in ["stop_yield", "reset_status"]:
                            need_yield_end = False
                            break
                        else:
                            raise ValueError(f"Unexpected request type during yield: {yield_request}")
                    if need_yield_end:
                        put_response((ResponseSignal.YIELD_END, None))  # todo if not rank0, dont put
            except Exception as e:
                put_response((ResponseSignal.FAIL, traceback.format_exc()))


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

        self.response_queue = mp.Queue()
        self.request_queues = []
        self.worker_processes = []

        if gpu_ids is None:
            gpu_ids = list(range(gpus))
        world_size = len(gpu_ids)
        barrier = mp.Barrier(world_size + 1)
        port = accessory.util.misc.find_free_port(10000 + random.randint(-100, 100), 11000)

        print(f"Launching {world_size} processes for hosting model with model parallel size {world_size}")
        print("Note that only the output from the FIRST model process will be printed")
        for i, gpu_id in enumerate(gpu_ids):
            request_queue_this_rank = mp.Queue()
            process = mp.Process(
                target=model_worker,
                args=(from_pretrained_args, from_pretrained_kwargs, world_size, i, gpu_id, port,
                      barrier, request_queue_this_rank, self.response_queue if i == 0 else None)
            )
            process.start()
            self.request_queues.append(request_queue_this_rank)
            self.worker_processes.append(process)

        barrier.wait()


    @torch.inference_mode()
    def compute_logits(self, *args, **kwargs):
        """
        Wraps `accessory.model.MetaModel.compute_logits`

        :param args: Positional arguments for 'MetaModel.compute_logits'.
        :param kwargs: Keyword arguments for 'MetaModel.compute_logits'.
        :return: The result returned by 'MetaModel.compute_logits' method.
        """
        return self.call_model_func("compute_logits", *args, **kwargs)

    @torch.inference_mode()
    def evaluate_examples(self, *args, **kwargs):
        """
        Wraps `accessory.model.MetaModel.evaluate_examples`

        :param args: Positional arguments for 'MetaModel.evaluate_examples'.
        :param kwargs: Keyword arguments for 'MetaModel.evaluate_examples'.
        :return: The result returned by 'MetaModel.evaluate_examples' method.
        """
        return self.call_model_func("evaluate_examples", *args, **kwargs)

    @torch.inference_mode()
    def generate(self, *args, **kwargs):
        """
        Wraps `accessory.model.MetaModel.generate`

        :param args: Positional arguments for 'MetaModel.generate'.
        :param kwargs: Keyword arguments for 'MetaModel.generate'.
        :return: The result returned by 'MetaModel.generate' method.
        """
        return self.call_model_func("generate", *args, **kwargs)

    @torch.inference_mode()
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
        self.emit_request(request_name, *args, **kwargs)
        response_signal, contents = self.response_queue.get()
        if response_signal == ResponseSignal.SUCCESS:
            return contents
        elif response_signal == ResponseSignal.FAIL:
            trace = contents
            raise Exception(trace)
        else:
            raise ValueError("Unexpected response signal {}".format(response_signal))

    def call_model_stream_func(self, request_name:str, *args, **kwargs):
        self.reset_status()
        assert request_name in REQUESTS_WITH_STREAM_RESPONSE
        self.emit_request(request_name, *args, **kwargs)
        while True:
            response_signal, contents = self.response_queue.get()
            if response_signal == ResponseSignal.YIELDING:
                try:
                    yield contents  # todo test here
                except GeneratorExit:
                    self.emit_request("stop_yield")
                    raise
                else:
                    self.emit_request("continue_yield")
            elif response_signal == ResponseSignal.YIELD_END:
                return
            elif response_signal == ResponseSignal.FAIL:
                trace = contents
                raise Exception(trace)
            else:
                raise ValueError("Unexpected response signal {}".format(response_signal))

    def emit_request(self, request_name:str, *args, **kwargs):
        for q in self.request_queues:
            q.put((request_name, (args, kwargs)))

    def reset_status(self):
        # clear response
        while not self.response_queue.empty():
            self.response_queue.get()

        self.emit_request("reset_status")