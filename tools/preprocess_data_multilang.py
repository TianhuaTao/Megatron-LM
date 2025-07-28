"""
This script is modified from megatron/tools/preprocess_data.py to process protein data.
For protein data, each document is just one protein sequence. For example, 'MIVLSHRGPFRFTREDDGTFTTTRGAGGVVSALTPLLLEPVHNSTWVAAAMTADDTEAQAEG' is a protein sequence."

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

import numpy as np
import multiprocessing


from megatron.training.tokenizer import build_tokenizer
from megatron.core.datasets import indexed_dataset




class Encoder(object):
    def __init__(self, args):
        self.args = args
        self.input_format = args.input_format

    def initializer(self):
        # Use Encoder class as a container for global data
        Encoder.tokenizer = build_tokenizer(self.args)
        Encoder.tokenizer.current_namespace = self.args.namespace


    def encode(self, text):
        text_len = len(text)
        if self.input_format == 'lines':
            text = text.strip()
            if self.args.namespace == 'P': # additional processing for protein data
                text = '<protein>' + text + '</protein>'
        elif self.input_format == 'jsonl':
            text = json.loads(text.strip())['text']
            
        ids = Encoder.tokenizer.tokenize(text).tolist()
        if self.args.add_bos:
            ids.insert(0, Encoder.tokenizer.bos)
        if self.args.add_eos:
            ids.append(Encoder.tokenizer.eos)
        lens = len(ids)
        return ids, lens, text_len

def print_processing_stats(args, count, proc_start, total_bytes_processed, total_size):
    if count % args.log_interval == 0:
        current = time.time()
        elapsed = current - proc_start
        mbs = total_bytes_processed/elapsed/1024/1024
        percent = 100 * total_bytes_processed / total_size
        print(f"Processed {count} documents ({percent:.3f} %), ",
                f"({count/elapsed} docs/s, {mbs} MB/s).",
                file=sys.stderr)




def process_txt_file(args, input_file_name, output_prefix):

    print("Opening", input_file_name)
    fin = open(input_file_name, 'r', encoding='utf-8')
    file_total_size_bytes = os.path.getsize(input_file_name)
    startup_start = time.time()
    encoder = Encoder(args)
    tokenizer = build_tokenizer(args)
    pool = multiprocessing.Pool(args.workers, initializer=encoder.initializer)
    encoded_docs = pool.imap_unordered(encoder.encode, fin, 100)

    level = "document"


    output_bin_files = {}
    output_idx_files = {}
    builders = {}

    key = 'text'
    output_bin_files[key] = "{}.bin".format(output_prefix)
    output_idx_files[key] = "{}.idx".format(output_prefix)
    builders[key] = indexed_dataset.IndexedDatasetBuilder(
        output_bin_files[key],
        dtype=indexed_dataset.DType.optimal_dtype(tokenizer.vocab_size),
    )

    startup_end = time.time()
    proc_start = time.time()
    total_bytes_processed = 0
    print("Time to startup:", startup_end - startup_start)
    for i, (ids, ids_len, bytes_processed) in enumerate(encoded_docs, start=1):
        total_bytes_processed += bytes_processed
        key = 'text'
        builders[key].add_document(ids, [ids_len])
        print_processing_stats(args, i, proc_start, total_bytes_processed, total_size=file_total_size_bytes)

    fin.close()
    builders[key].finalize(output_idx_files[key])


def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', type=str, required=True,
                       help='Path to input JSON')
    group.add_argument('--namespace', type=str, required=True,
                       help='Namespace for dataseet, "P" for protein, "D" for DNA')


    group = parser.add_argument_group(title='tokenizer')


    group.add_argument('--vocab-file', type=str, default=None,
                       help='Path to the vocab file')
    group.add_argument('--input-format', type=str, default='lines',
                       help='Data format of the input file. '
                            'Options: lines, jsonl')
    group.add_argument('--add-eos', action='store_true',
                       help='Append an <eos> token to the end of a document.')
    group.add_argument('--add-bos', action='store_true',
                       help='Append an <bos> token to the start of a document.')
    group = parser.add_argument_group(title='output data')
    group.add_argument('--output-prefix', type=str, required=True,
                       help='Path to binary output file without suffix')

    group = parser.add_argument_group(title='runtime')
    group.add_argument('--workers', type=int, required=True,
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
    args.tokenizer_type = 'MultiLangBioTokenizer' # used in build_tokenizer()
    return args




def main():
    args = get_args()

    process_txt_file(args, args.input, args.output_prefix)




if __name__ == '__main__':

    main()

