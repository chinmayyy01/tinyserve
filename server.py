import argparse
import asyncio
import glob
import json
import os
import queue
import random
import threading
import time
import uuid
from collections import OrderedDict, deque
from dataclasses import dataclass
import traceback
import warnings
import gc

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn

from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoConfig
from transformers import AutoTokenizer, GenerationConfig
from transformers import AutoModelForCausalLM
from datasets import load_dataset

from fastapi import Request as HTTPRequest
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse

try:
    from flash_attn import flash_attn_varlen_func
    FLASH_ATTN_AVAILABLE = True
except:
    FLASH_ATTN_AVAILABLE = False


@dataclass
class Config:

    model: str
    dtype: torch.dtype = torch.bfloat16
    max_model_len: str = 4096 # max tokens in prompt + output we accept
    attention_backend: str = "flash_attn" # "flash_attn" (fast) | "torch" (reference)

    # tokens per kv cache page. 0 = pick for backend, flash attn only supports
    # pages that are multiples of 256, vllm actually has its own fork for
    # smaller pages (more efficient if we use shorter prompts on average)
    # the torch reference code can use anything
    block_size: int = 0 
    max_num_seqs: int = 128 # max requests we can have in a batch

    # token budget per step, 
    max_num_batched_tokens: int = 2048
    gpu_memory_utilization: float = 0.85
    enable_prefix_caching: bool = True
    use_cuda_graphs: bool = True
    seed: int = 0


    def __post_init__(self):
        # if flash_attn not available dont use
        if self.attention_backend == "flash_attn" and not FLASH_ATTN_AVAILABLE:
            warnings.warn("flash attention not available, default to torch")
            self.attention_backend = "torch"

        # set page size as small as possible based on backend
        if self.block_size == 0:
            self.block_size = 256 if self.attention_backend == "flash_attn" else 16

        # our manually looped torch implementation cant be cuda graphed
        if self.attention_backend == "torch":
            self.use_cuda_graphs = False  # its Python loop can't be captured

@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_p: float = 1.0       # selects the smallest set of tokens whose cumulative probability is at least p
    top_k: int = -1          # -1 means off
    max_tokens: int = 64     # max tokens to gen
    ignore_eos: bool = False # normally we stop gen on eos token

    ### for testing ###
    forced_token_ids: list[int] | None = None
    return_logprobs: bool = False

class Request:
    """
    This is the metadata that is needed by the scheduler about a single request

    The main field we care about is num_computed_tokens, it is how
    many token_ids already have K/V in the cache, each step the scheduler
    picks how many new tokens to process for this request:

    for example, if we have a prompt of 1000 tokens but a budget of 512, 
    then we will first do 512 tokens, then the remaining 488 (this is
    chunked prefill), afterwards during generation it is one token per step
    """

    def __init__(self, 
                 request_id: str, 
                 prompt_ids: list[int], 
                 params: SamplingParams):

        self.request_id = request_id
        self.prompt_len = len(prompt_ids)

        # prompt + generated so far
        self.token_ids = list(prompt_ids) 
        self.params = params

        # how many of those tokens already have K/V written
        self.num_computed_tokens = 0

        # this requests page table (what blocks contain the KV cache for this specific request?)
        self.block_ids: list[int] = []

        # One has per full block, for prefix reuse
        self.block_hashes: list[int] = [] 
        self.num_cached_tokens = 0 # prefix cache hits at admission
        self.logprobs: list[torch.Tensor] = []
        self.finished = False
        self.finish_reason: str | None = None

    @property
    def num_tokens(self):
        return len(self.token_ids)

    @property
    def output_ids(self):
        # everything after the prompt is what was generated
        return self.token_ids[self.prompt_len:]

class KVCacheManager:
    """
    This manages which phyisical  KV cache blocks each request owns

    The block manager only manages block ids, it doesnt touch GPU memory
    the model runner later converts those ids to actual kv addresses. 

    Each request sees its kv cache as a sequence of logical blocks, while
    the blocks themselves can live anywhere in GPU memory. For example
    with block_size = 4 and a request containing:
    
    [A B C D E F G H I]

    the request needs three logical blocks::

        logical block 0 -> tokens [A B C D]
        logical block 1 -> tokens [E F G H]
        logical block 2 -> tokens [I]

    The block manager might assign physical blocks:
    
        req.block_ids = [17, 42, 81]

    so tokens A-D live in physical block 17, E-H in block 42, and I in block 81.
    The physical blocks do not need to be contiguous.

    For token i:

        logical_block = i // block_size
        offset        = i % block_size
        block_id      = req.block_ids[logical_block]

    if the cache is viewed as one flat array the corresponding slot is:
    
        slot = block_id * block_size + offset

    For example, token F has i = 5:

        logical_block = 5 // 4 = 1
        offset = 5 % 4 = 1
        block_id = req.block_ids[1] = 42
        slot = 42 * 4 + 1 = 169

    Prefix Caching: full blocks can be reused by later requests (i.e. things
    like the system prompt can be reused for each request). Each full block is
    identified by a hash of its own parent blocks hash and its own tokens:

        block_hash = hash(parent_hash, tokens)

    The parent hash matters beacuse the same tokens can have different KV values
    when they occur after different prefixes. For example:

    Request A: 
        [system prompt] -> block 17 -> [A B C D] -> block 42 -> [E F G H] 
    
    Request B: 
        [system prompt] -> block 17 -> [A B C D] -> block 42 -> [E F G H] -> [X Y]


    Request B can reuse block 17 and 42 instead of recomputing their KV values
    it only needs a new block for [X Y]. 

    Only complete blocks are entered into the prefix cache, a partial final block
    cannot be reused until it is full. When an active request finishes, its
    cached blocks are not immediately erased, they become available for reuse
    and are placed on a least recently used (lru) free list while their hash-to-block
    mapping stays searchable. If a later request asks for the same prefix, the block can
    be taken back without recomputing its KV values. 

    Eventually, if GPU is memory is needed for a new block, the least recently used
    cached block is evicted and its physical storage is reused. At that point
    its old prefix KV is lost and must be recomputed if requested again.  

    """
    def __init__(self, 
                 num_blocks: int, 
                 block_size: int, 
                 enable_caching: bool):

        self.block_size = block_size
        self.enable_caching = enable_caching

        # therefore block 2 cannot be evicted while other requests are still using it
        self.ref_cnt = [0] * num_blocks

        # if a physical block contains a cached prefix, what prefxi does it represent
        self.block_hash : list[int | None] = [None] * num_blocks

        # ordered dict is double linked list, allows for O(1) pop front
        # and O(1) removal from the middle. 
        # first we start at index 1, as block 0 is the null block used for padding tokens
        # their value doesnt matter as we mask it during attention anyway

        # This is effectively our LRU queue, it is a list of our unused blocks 
        self.free : OrderedDict[int, None] = OrderedDict((i, None) for i in range(1, num_blocks))

        # given this prefix hash, where is its kv data (hash -> physical block) ?
        self.hash_to_block: dict[int, int] = {}
