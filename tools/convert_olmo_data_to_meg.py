"""
This script is modified from megatron/tools/preprocess_data.py to process OLMoE data.
It's going to read OLMo data (numpy arrays) and convert it to Megatron-LM format.

"""

"""Processing large data for pretraining."""
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



from transformers import AutoTokenizer

import numpy as np
def process_olmoe_file(args, input_file_name, output_prefix):
    eod_token = args.eod_id
    print("Opening", input_file_name)
    with open(input_file_name, 'rb') as file:
        data = file.read()
    print("Read", len(data), "bytes")
    # Interpret binary data as an array of uint32
    
    print("Unpacking", len(data) // 4, "tokens")
    # tokens = list(struct.unpack(f'{len(data) // 4}I', data))
    tokens = np.frombuffer(data, dtype=np.uint32)
    del data
    print("Unpacked", len(tokens), "tokens")
    
        
    # Find all positions of eod_token
    print("Searching for EOD tokens")
    eod_positions = np.where(tokens == eod_token)[0]
    print("Found", len(eod_positions), "EOD tokens")
    
    # Extract documents based on boundaries
    start = 0
    documents = []
    for pos in eod_positions:
        if start != pos:
            documents.append(tokens[start:pos+1])
        start = pos + 1
    
    if start < len(tokens):  # Add last document if no trailing EOD
        documents.append(tokens[start:])
    
    tokenizer = AutoTokenizer.from_pretrained("allenai/dolma2-tokenizer")
    




    output_bin_files = {}
    output_idx_files = {}
    builders = {}

    key = 'text'
    level = "document"
    output_bin_files = "{}_{}_{}.bin".format(output_prefix,
                                                    key, level)
    output_idx_files = "{}_{}_{}.idx".format(output_prefix,
                                                    key, level)
    
    # create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_bin_files), exist_ok=True)
    
    builders = indexed_dataset.IndexedDatasetBuilder(
        output_bin_files,
        dtype=indexed_dataset.DType.optimal_dtype(tokenizer.vocab_size),
    )


    for ids in (tqdm(documents)):
        ids_len = len(ids)
        # ids need to be tensor
        ids = torch.tensor(ids, dtype=torch.int32)
        builders.add_document(ids, [ids_len])

    print("Finalizing dataset...")
    builders.finalize(output_idx_files)


def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', type=str, required=True,
                       help='Path to input JSON')



    group = parser.add_argument_group(title='tokenizer')


    group.add_argument('--eod-id', type=int, default=None,
                       help='End of Document token id')

    group = parser.add_argument_group(title='output data')
    group.add_argument('--output-prefix', type=str, required=True,
                       help='Path to binary output file without suffix')

    group = parser.add_argument_group(title='runtime')
    group.add_argument('--workers', type=int, default=1,
                       help=('Number of worker processes to launch.'
                             'A good default for fast pre-processing '
                             'is: (workers * partitions) = available CPU cores.'))

    group.add_argument('--log-interval', type=int, default=1000,
                       help='Interval between progress updates')

    args = parser.parse_args()
    args.keep_empty = False


    # some default/dummy values for the tokenizer
    args.rank = 1
    args.make_vocab_size_divisible_by = 128
    args.tensor_model_parallel_size = 1
    args.vocab_extra_ids = 0
    args.tokenizer_type = 'NullTokenizer' # used in build_tokenizer()
    return args




def main():
    args = get_args()

    process_olmoe_file(args, args.input, args.output_prefix)




if __name__ == '__main__':

    main()

