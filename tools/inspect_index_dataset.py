import argparse
import math
import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
import time
from tqdm import tqdm
import numpy as np
import multiprocessing
import torch

from megatron.training.tokenizer import build_tokenizer
from megatron.core.datasets import indexed_dataset


# prefix = sys.argv[1]

def validate_index_dataset(prefix):
    ds = indexed_dataset.IndexedDataset(prefix)
    
    # print(ds)
    assert len(ds.sequence_lengths) == ds.document_indices[-1]
    
    bin_file_size = os.path.getsize(prefix + '.bin')
    total_tokens = sum(ds.sequence_lengths)
    assert bin_file_size == total_tokens * 4
    
if __name__ == '__main__':
    dir = '/weka/oe-training-default/tianhua/ws-megatron/data/ai2-llm/megatron/dclm/text_openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train/allenai/dolma2-tokenizer'
    all_prefixes = set()
    for file in os.listdir(dir):
        all_prefixes.add(os.path.join(dir, file[:-4]))
        
    for prefix in sorted(all_prefixes):
        try:
            validate_index_dataset(prefix)
        except Exception as e:
            print(f"Error validating {prefix}: {e}")
            continue
        print(f"Validated {prefix}")