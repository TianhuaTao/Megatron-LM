"""
It's like a GPT dataset, where sequences are concatenated together, but with a masked language modeling twist, kind of like BERT.
"""




import logging
import os
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy
import numpy as np
import torch
import copy
import random

import torch.distributed

from megatron.core.datasets.blended_megatron_dataset_config import BlendedMegatronDatasetConfig
from megatron.core.datasets.indexed_dataset import IndexedDataset
from megatron.core.datasets.megatron_dataset import MegatronDataset
from megatron.core.datasets.megatron_tokenizer import MegatronTokenizer
from megatron.core.datasets.utils import Split
# from megatron.core.datasets.utils_s3 import S3Config, is_s3_path
from megatron.core.utils import log_single_rank

logger = logging.getLogger(__name__)

_PAD_TOKEN_ID = -1

from typing import Any, Callable, Dict, MutableMapping, Optional, Tuple, Union

def wait_for(condition: Callable[[], bool], description: str, timeout: float = 10.0):
    """Wait for the condition function to return True."""
    start_time = time.monotonic()
    while not condition():
        time.sleep(0.5)
        if time.monotonic() - start_time > timeout:
            raise TimeoutError(f"{description} timed out")
        
@dataclass
class ConcatMaskedDatasetConfig(BlendedMegatronDatasetConfig):
    """Configuration object for Megatron Core GPT datasets"""

    reset_position_ids: bool = None
    """Option to reset the position IDs in the dataset at an interval"""

    reset_attention_mask: bool = None
    """Option to reset the attention mask from the dataset"""

    eod_mask_loss: bool = None
    """Option to enable the EOD mask loss"""

    create_attention_mask: bool = True
    """Option to enable the attention masks generation. Can be disabled if attention kernel
       generates masks by itself.
    """

    drop_last_partial_validation_sequence: bool = True
    """Option to drop the last partial validation sequence"""

    add_extra_token_to_sequence: bool = False
    """Option to draw sequences with one extra token to ensure the sample input tokens and sample
       output tokens are both of the desired sequence length
    """

    s3_cache_path: str = None
    """Path for caching indices for s3 dataloading."""

    # mask related
    
    masking_probability: float = None
    """The probability we mask a candidate N-gram"""

    short_sequence_probability: float = None
    """The probability we return a sequence shorter than the target sequence length"""


    def __post_init__(self) -> None:
        """Do asserts and set fields post init"""
        super().__post_init__()

        assert self.tokenizer is not None

        assert self.reset_position_ids is not None
        assert self.reset_attention_mask is not None
        assert self.eod_mask_loss is not None


class ConcatMaskedDataset(MegatronDataset):
    """The base GPT dataset

    Args:
        indexed_dataset (IndexedDataset): The IndexedDataset around which to build the GPTDataset

        dataset_path (Optional[str]): The real path on disk to the dataset, for bookkeeping

        indexed_indices (numpy.ndarray): The set of the documents indices to expose

        num_samples (Optional[int]): The number of samples to draw from the indexed dataset. When
            None, build as many samples as correspond to one epoch.

        index_split (Split): The indexed_indices Split

        config (GPTDatasetConfig): The config
    """

    def __init__(
        self,
        indexed_dataset: IndexedDataset,
        dataset_path: Optional[str],
        indexed_indices: numpy.ndarray,
        num_samples: Optional[int],
        index_split: Split,
        config: ConcatMaskedDatasetConfig,
    ) -> None:
        super().__init__(
            indexed_dataset, dataset_path, indexed_indices, num_samples, index_split, config
        )
        # self.masks_and_position_ids_are_cacheable = not any(
        #     [
        #         self.config.reset_position_ids,
        #         self.config.reset_attention_mask,
        #         self.config.eod_mask_loss,
        #     ]
        # )
        self.masks_and_position_ids_are_cacheable = False # it cannot be cached since the loss mask changes
        self.masks_and_position_ids_are_cached = False
        self.cached_attention_mask = None
        self.cached_loss_mask = None
        self.cached_position_ids = None

        try:
            self._pad_token_id = self.config.tokenizer.pad
        except Exception:
            self._pad_token_id = _PAD_TOKEN_ID

        (self.document_index, self.sample_index, self.shuffle_index) = (
            self._build_document_sample_shuffle_indices()
        )

    @staticmethod
    def numel_low_level_dataset(low_level_dataset: IndexedDataset) -> int:
        """Abstract method implementation

        For GPT, the underlying IndexedDataset should be split by sequence, as opposed to, say,
        BERT, which should be split by document

        Args:
            low_level_dataset (IndexedDataset): The underlying IndexedDataset

        Returns:
            int: The number of unique elements in the underlying IndexedDataset
        """
        return low_level_dataset.sequence_lengths.shape[0]

    @staticmethod
    def build_low_level_dataset(dataset_path: str, config: ConcatMaskedDatasetConfig) -> IndexedDataset:
        """Abstract method implementation

        Args:
            dataset_path (str): The real path prefix to the IndexedDataset .bin and .idx files

            config (GPTDatasetConfig): The config

        Returns:
            IndexedDataset: The underlying IndexedDataset
        """
        # if is_s3_path(dataset_path):
            # return IndexedDataset(
            #     dataset_path,
            #     multimodal=False,
            #     mmap=config.mmap_bin_files,
            #     s3_config=S3Config(path_to_idx_cache=config.s3_cache_path),
            # )
            # raise NotImplementedError(
            #     "S3 dataset loading is not yet implemented for ConcatMaskedDataset"
            # )
        return IndexedDataset(dataset_path, multimodal=False, mmap=config.mmap_bin_files)

    def __len__(self) -> int:
        """Abstract method implementation

        Returns:
            int: The length of the dataset
        """
        return self.sample_index.shape[0] - 1
    
    def truncate_input(self, input_ids, rng):
        target_length = rng.randrange(32, len(input_ids))
        return input_ids[:target_length]
    
    def __getitem__(self, idx: Optional[int]) -> Dict[str, torch.Tensor]:
        """Abstract method implementation

        Args:
            idx (Optioal[int]): The index into the dataset

        Returns:
            Dict[str, torch.Tensor]: The sample information wrapped in a dictionary
        """
        if idx is None:
            # Batch padding sequence so the index does not matter
            text, _ = self._query_document_sample_shuffle_indices(0)
            rng = random.Random(random.Random(0).randint(0, 2 ** 32 - 1))
        else:
            text, _ = self._query_document_sample_shuffle_indices(idx)
            rng = random.Random(random.Random(int(idx)).randint(0, 2 ** 32 - 1))

        
        current_input_ids = text
        
        sequences = []
        if rng.random() < self.config.short_sequence_probability:
            current_input_ids = self.truncate_input(current_input_ids, rng)
        
        
        masked_lengths = int(len(current_input_ids) * self.config.masking_probability)
        cand_maked_pos = [idx for idx, token in enumerate(current_input_ids)]
        rng.shuffle(cand_maked_pos)

        target_tokens = copy.deepcopy(current_input_ids)
        source_tokens = copy.deepcopy(current_input_ids)
        loss_masks = numpy.zeros(len(target_tokens), dtype=int)
        position_ids = numpy.arange(len(target_tokens), dtype=int)
        for pos in cand_maked_pos[:masked_lengths]:
            loss_masks[pos] = 1
            if rng.random() < 0.8:  # 80%
                source_tokens[pos] = self.config.tokenizer.mask # make token mask
            elif rng.random() < 0.5:  # 10%
                index = rng.randint(1, self.config.tokenizer.vocab_size) # random index in vocabulary
                # print(pos, index)
                source_tokens[pos] = index # replace
        block_position_ids = numpy.concatenate(
            [numpy.zeros(len(target_tokens), dtype=int)]
        )
        
        position_ids = numpy.stack([position_ids, block_position_ids], axis=0)
        tokens, targets, loss_masks, position_ids = self.pad_batch(
            source_tokens, target_tokens, loss_masks, position_ids,
            max_seq_length=self.config.sequence_length
        )
        
        division = len(target_tokens)


        position_ids = position_ids[0]  # no_2d_encoding
        division = numpy.array([division], dtype=int)
        sequences.append((tokens, targets, loss_masks, position_ids, division))
        
        packed_tokens, packed_targets, packed_loss_masks, packed_position_ids, packed_division = self._pack_samples(sequences)
        packed_tokens = torch.from_numpy(packed_tokens).long()
        packed_targets = torch.from_numpy(packed_targets).long()
        packed_loss_masks = torch.from_numpy(packed_loss_masks).float()
        packed_position_ids = torch.from_numpy(packed_position_ids).long()
        
        # build the attention mask
        """
        attention_mask: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]],
             default = `None`. Boolean tensor(s) used to mask out attention softmax input.
             It should be 'None' for 'causal' and 'no_mask' types. For 'padding' masks, it should be
             a single tensor of [batch_size, 1, 1, seqlen_q] for self-attention, and a tuple of
             two tensors in shapes [batch_size, 1, 1, seqlen_q] and [batch_size, 1, 1, seqlen_kv]
             for cross-attention. For the 'arbitrary' mask type, it should be in a shape that is
             broadcastable to [batch_size, num_heads, max_seqlen_q, max_seqlen_kv]. A `True` value
             means the corresponding position is masked out and a `False` means that position is
             allowed to participate in attention.
             
        """

        # 2D attention mask
        # attn_mask = torch.zeros((self.config.sequence_length, self.config.sequence_length), dtype=torch.bool) # zero means no attention
        # attn_mask[:len(target_tokens), :len(target_tokens)] = True # allow attention within the sequence (ignore padding)
        # attn_mask = attn_mask.logical_not_() # False means attention
        # attn_mask.unsqueeze_(0) 
        
        # 1D attention mask
        attn_mask = torch.zeros((1, self.config.sequence_length), dtype=torch.bool) # zero means no attention
        attn_mask[:, :len(target_tokens)] = True # allow attention within the sequence (does not include padding)
        attn_mask = attn_mask.logical_not_() # False means should attend to
        attn_mask.unsqueeze_(0) 
        
        if self.config.create_attention_mask:
            ret = {
                "tokens": packed_tokens,
                "labels": packed_targets,
                "attention_mask": attn_mask,
                "loss_mask": packed_loss_masks,
                "position_ids": packed_position_ids,
            }
        else:
            ret = {
                "tokens": packed_tokens,
                "labels": packed_targets,
                "loss_mask": packed_loss_masks,
                "position_ids": packed_position_ids,
            }
        return ret
        #############################
        # text = torch.from_numpy(text).long()
        
        # if self.config.add_extra_token_to_sequence:
        #     tokens = text[:-1].contiguous()
        #     labels = text[1:].contiguous()
        # else:
        #     tokens = text
        #     labels = torch.roll(text, shifts=-1, dims=0)
        #     labels[-1] = self._pad_token_id

        # if (
        #     not self.masks_and_position_ids_are_cacheable
        #     or not self.masks_and_position_ids_are_cached
        # ):
        #     attention_mask, loss_mask, position_ids = _get_ltor_masks_and_position_ids(
        #         tokens,
        #         self.config.tokenizer.eod,
        #         self.config.reset_position_ids,
        #         self.config.reset_attention_mask,
        #         self.config.eod_mask_loss,
        #         self.config.create_attention_mask,
        #     )
        #     if self.masks_and_position_ids_are_cacheable:
        #         self.cached_attention_mask = attention_mask
        #         self.cached_loss_mask = loss_mask
        #         self.cached_position_ids = position_ids
        #         self.masks_and_position_ids_are_cached = True
        # else:
        #     attention_mask = self.cached_attention_mask
        #     loss_mask = self.cached_loss_mask
        #     position_ids = self.cached_position_ids

        # # For padded sequences, mask the loss
        # loss_mask[labels == self._pad_token_id] = 0.0

        # # For padded sequences, ensure the embedding layer can map the token ID
        # tokens[tokens == self._pad_token_id] = 0
        # labels[labels == self._pad_token_id] = 0

        # # Batch padding sequence so we mask the loss
        # if idx is None:
        #     loss_mask = torch.zeros_like(loss_mask)

        # if self.config.create_attention_mask:
        #     return {
        #         "tokens": tokens,
        #         "labels": labels,
        #         "attention_mask": attention_mask,
        #         "loss_mask": loss_mask,
        #         "position_ids": position_ids,
        #     }
        # else:
        #     return {
        #         "tokens": tokens,
        #         "labels": labels,
        #         "loss_mask": loss_mask,
        #         "position_ids": position_ids,
        #     }

    def pad_batch(self, tokens, targets, loss_masks, position_ids, max_seq_length=None):
        if max_seq_length is None:
            max_seq_length = self.config.sequence_length
        if len(tokens) >= max_seq_length:
            tokens = tokens[: max_seq_length]
            targets = targets[: max_seq_length]
            loss_masks = loss_masks[: max_seq_length]
            position_ids = position_ids[:, : max_seq_length]
        else:
            tokens = np.concatenate(
                (
                    tokens,
                    np.zeros(max_seq_length - len(tokens), dtype=int),
                )
            )
            targets = np.concatenate(
                (
                    targets,
                    np.zeros(max_seq_length - len(targets), dtype=int),
                )
            )
            loss_masks = np.concatenate(
                (
                    loss_masks,
                    np.zeros(
                        max_seq_length - len(loss_masks), dtype=int
                    ),
                )
            )
            position_ids = np.concatenate(
                (
                    position_ids,
                    np.zeros(
                        (2, max_seq_length - position_ids.shape[1]),
                        dtype=int,
                    ),
                ),
                axis=1,
            )
        return tokens, targets, loss_masks, position_ids
    
    def _query_document_sample_shuffle_indices(
        self, idx: int
    ) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """Get the text (token ids) and document ids for a given index

        Args:
            idx (int): The index into the dataset

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]: The text ids and document ids
        """
        # Do the shuffle mapping
        idx = self.shuffle_index[idx]

        # Get the beginning and end documents and offsets
        doc_index_beg, doc_index_beg_offset = self.sample_index[idx]
        doc_index_end, doc_index_end_offset = self.sample_index[idx + 1]

        document_ids = []
        sample_parts = []

        # Sample spans a single document
        if doc_index_beg == doc_index_end:
            # Add the document id
            document_ids.append(self.document_index[doc_index_beg])

            # Add the entire sample
            sample_parts.append(
                self.dataset.get(
                    self.document_index[doc_index_beg],
                    offset=doc_index_beg_offset,
                    length=doc_index_end_offset
                    - doc_index_beg_offset
                    + self.config.add_extra_token_to_sequence,
                )
            )

        # Sample spans multiple documents
        else:
            for i in range(doc_index_beg, doc_index_end + 1):
                # Add the document id
                document_ids.append(self.document_index[i])

                # Add the sample part
                offset = 0 if i > doc_index_beg else doc_index_beg_offset
                length = (
                    None
                    if i < doc_index_end
                    else doc_index_end_offset + self.config.add_extra_token_to_sequence
                )
                sample_parts.append(
                    self.dataset.get(self.document_index[i], offset=offset, length=length)
                )
        assert len(document_ids) == len(
            sample_parts
        ), f"len(document_ids) ({len(document_ids)}) != len(sample_parts) ({len(sample_parts)})"

        length = sum(map(len, sample_parts))

        # Pad the sample if necessary
        if length < (self.config.sequence_length + self.config.add_extra_token_to_sequence):
            sample_parts.append(
                [self._pad_token_id]
                * (self.config.sequence_length + self.config.add_extra_token_to_sequence - length)
            )

        return (
            numpy.concatenate(sample_parts, dtype=numpy.int64),
            numpy.array(document_ids, dtype=numpy.int64),
        )

    def _pack_samples(self, sequences):
        tokens, targets, loss_masks, position_ids, division = zip(*sequences)
        tokens = numpy.concatenate(tokens, axis=-1)
        targets = numpy.concatenate(targets, axis=-1)
        loss_masks = numpy.concatenate(loss_masks, axis=-1)
        position_ids = numpy.concatenate(position_ids, axis=-1)
        division = list(division)
        division = numpy.concatenate(division, axis=-1)
        return tokens, targets, loss_masks, position_ids, division
    
    def _build_document_sample_shuffle_indices(
        self,
    ) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
        """Build the document index, the sample index, and the shuffle index

        The document index:
            -- 1-D
            -- An ordered array of document ids

        The sample index:
            -- 2-D
            -- The document indices and offsets which mark the start of every sample

        The shuffle index:
            -- 1-D
            -- A random permutation of index range of the sample index

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]: The document index, the sample
            index, and the shuffle index
        """
        path_to_cache = self.config.path_to_cache
        if path_to_cache is None and not self.config.mock:
            path_to_cache = os.path.join(
                self.dataset.path_prefix, "cache", f"{type(self).__name__}_indices"
            )

        if path_to_cache:
            base = f"{self.unique_description_hash}-{type(self).__name__}-{self.index_split.name}"
            get_path_to = lambda affix: os.path.join(path_to_cache, f"{base}-{affix}")
            path_to_description = get_path_to("description.txt")
            path_to_document_index = get_path_to("document_index.npy")
            path_to_sample_index = get_path_to("sample_index.npy")
            path_to_shuffle_index = get_path_to("shuffle_index.npy")
            cache_hit = all(
                map(
                    os.path.isfile,
                    [
                        path_to_description,
                        path_to_document_index,
                        path_to_sample_index,
                        path_to_shuffle_index,
                    ],
                )
            )
        else:
            cache_hit = False

        if not path_to_cache or (
            not cache_hit
            and (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0)
        ):

            log_single_rank(
                logger,
                logging.INFO,
                f"Build and save the {type(self).__name__} {self.index_split.name} indices",
            )
            self.built_anew_on_cache_miss = True
            t_beg = time.time()

            sequence_length = self.config.sequence_length
            num_tokens_per_epoch = self._get_num_tokens_per_epoch()
            num_epochs = self._get_num_epochs(num_tokens_per_epoch)

            if num_epochs == 1:
                separate_final_epoch = False
            else:
                # Get the number of samples for the last epoch
                num_samples_sans_final_epoch = (
                    (num_epochs - 1) * num_tokens_per_epoch
                    - self.config.add_extra_token_to_sequence
                ) // sequence_length
                num_samples_from_final_epoch = self.num_samples - num_samples_sans_final_epoch
                num_samples_per_epoch = (
                    num_tokens_per_epoch - self.config.add_extra_token_to_sequence
                ) // sequence_length

                # num_samples_from_final_epoch should be non-negative
                assert num_samples_from_final_epoch >= 0

                # num_samples_from_final_epoch should not exceed max value
                assert num_samples_from_final_epoch <= num_samples_per_epoch + 1

                # Separate the final epoch if it falls below the threshold
                threshold = 0.80
                separate_final_epoch = num_samples_from_final_epoch < int(
                    threshold * num_samples_per_epoch
                )

                log_single_rank(
                    logger,
                    logging.INFO,
                    f"> num_samples_from_final_epoch: {num_samples_from_final_epoch}",
                )
                log_single_rank(logger, logging.INFO, f"> threshold: {threshold}")
                log_single_rank(
                    logger, logging.INFO, f"> num_samples_per_epoch: {num_samples_per_epoch}"
                )

            log_single_rank(
                logger, logging.INFO, f"> separate_final_epoch: {separate_final_epoch}"
            )

            numpy_random_state = numpy.random.RandomState(self.config.random_seed)

            # Build the document index
            log_single_rank(logger, logging.INFO, f"Building the {type(self).__name__} {self.dataset_path} document index")
            document_index = _build_document_index(
                self.indices, num_epochs, numpy_random_state, separate_final_epoch
            )
            log_single_rank(logger, logging.INFO, f"DONE building the {type(self).__name__} document index")

            drop_last_partial_sequence = True
            if self.index_split == Split.valid:
                drop_last_partial_sequence = self.config.drop_last_partial_validation_sequence

            # Build the sample index
            from megatron.core.datasets import helpers

            if self.index_split == Split.valid:
                drop_last_partial_sequence = self.config.drop_last_partial_validation_sequence
            else:
                drop_last_partial_sequence = True

            assert document_index.dtype == numpy.int32
            assert self.dataset.sequence_lengths.dtype == numpy.int32
            if len(document_index) * 2 > len(self.dataset.sequence_lengths):
                # If "access density" of sequence_lengths is high, force load the mmap-ed array
                # into memory by making a copy.
                #
                # System performance benefits come from two aspects:
                #   1. We sequentially pre-load the whole file, most of which we expect to read
                #   2. The GIL is held when entering the c++ program, improving the speed of which
                #      improves parallelism
                sequence_lengths_for_cpp = self.dataset.sequence_lengths.copy()
            else:
                sequence_lengths_for_cpp = self.dataset.sequence_lengths
                
            log_single_rank(
                logger, logging.INFO, f"Building the {type(self).__name__} {self.dataset_path} sample index"
            )
            sample_index = helpers.build_sample_idx(
                sequence_lengths_for_cpp,
                document_index,
                sequence_length,
                num_epochs,
                num_tokens_per_epoch,
                drop_last_partial_sequence,
                self.config.add_extra_token_to_sequence,
            )
            log_single_rank(logger, logging.INFO, f"DONE building the {type(self).__name__} sample index")

            # Build the shuffle index
            if separate_final_epoch:
                shuffle_index = _build_shuffle_index(
                    num_samples_sans_final_epoch, sample_index.shape[0] - 1, numpy_random_state
                )
            else:
                shuffle_index = _build_shuffle_index(
                    sample_index.shape[0] - 1, sample_index.shape[0] - 1, numpy_random_state
                )

            if path_to_cache:
                os.makedirs(path_to_cache, exist_ok=True)

                log_single_rank(
                    logger,
                    logging.INFO,
                    f"Saving to {os.path.basename(path_to_document_index)}, {os.path.basename(path_to_sample_index)}, {os.path.basename(path_to_shuffle_index)}",
                )
                numpy.save(path_to_document_index, document_index, allow_pickle=True)
                numpy.save(path_to_sample_index, sample_index, allow_pickle=True)
                numpy.save(path_to_shuffle_index, shuffle_index, allow_pickle=True)
                
                # Write the description
                with open(path_to_description, "wt") as writer:
                    writer.write(self.unique_description)
                    
                log_single_rank(
                    logger,
                    logging.INFO,
                    f"Saved to {os.path.basename(path_to_document_index)}, {os.path.basename(path_to_sample_index)}, {os.path.basename(path_to_shuffle_index)}",
                )
            else:
                log_single_rank(
                    logger,
                    logging.WARNING,
                    f"Unable to save {type(self).__name__} indexes because path_to_cache is None",
                )
                assert False, "Unable to save indexes because path_to_cache is None"

            t_end = time.time()
            log_single_rank(logger, logging.INFO, f"\t> time elapsed: {t_end - t_beg:4f} seconds")

            log_single_rank(
                logger, logging.INFO, f"> total number of samples: {sample_index.shape[0] - 1}"
            )
            log_single_rank(logger, logging.INFO, f"> total number of epochs: {num_epochs}")

            return document_index, sample_index, shuffle_index
            

        log_single_rank(
            logger, logging.INFO, f"Loading the {type(self).__name__} {self.index_split.name} indices from {os.path.basename(path_to_document_index)}"
        )

        
        # In the cases where we're using a shared NFS drive between ranks to save checkpoints,
        # writing the file from rank 0 might not be immediately
        # realized in the file systems of the other ranks.
        # So we wait here across all ranks until that final checkpoint directory is visible.
        wait_for(lambda: os.path.exists(path_to_document_index), f"Waiting for doc index {path_to_document_index}", timeout=20.0)
        
        t_beg = time.time()
        document_index = numpy.load(path_to_document_index, allow_pickle=True, mmap_mode='r')
        t_end = time.time()
        log_single_rank(logger, logging.INFO, f"\t> time elapsed: {t_end - t_beg:4f} seconds")

        log_single_rank(
            logger,
            logging.INFO,
            f"\tLoad the sample index from {os.path.basename(path_to_sample_index)}",
        )
        t_beg = time.time()
        sample_index = numpy.load(path_to_sample_index, allow_pickle=True, mmap_mode='r')
        t_end = time.time()
        log_single_rank(logger, logging.INFO, f"\t> time elapsed: {t_end - t_beg:4f} seconds")

        log_single_rank(
            logger,
            logging.INFO,
            f"\tLoad the shuffle index from {os.path.basename(path_to_shuffle_index)}",
        )
        t_beg = time.time()
        shuffle_index = numpy.load(path_to_shuffle_index, allow_pickle=True, mmap_mode='r')
        t_end = time.time()
        log_single_rank(logger, logging.INFO, f"\t> time elapsed: {t_end - t_beg:4f} seconds")

        log_single_rank(
            logger, logging.INFO, f"> total number of samples: {sample_index.shape[0] - 1}"
        )

        return document_index, sample_index, shuffle_index

    def _get_num_tokens_per_epoch(self) -> int:
        """Calculate the number of tokens in a single epoch

        Returns:
            int: The number of tokens in a single epoch
        """
        return int(numpy.sum(self.dataset.sequence_lengths[self.indices]))

    def _get_num_epochs(self, num_tokens_per_epoch: int) -> int:
        """Calculate the number of epochs

        Args:
            num_tokens_per_epoch (int): The number of tokens in a single epoch

        Returns:
            int: The number of epochs
        """
        num_epochs = 1
        num_tokens = num_tokens_per_epoch
        if self.num_samples is None:
            return num_epochs
        else:
            num_tokens_requested = (
                self.num_samples * self.config.sequence_length
            ) + self.config.add_extra_token_to_sequence
            while num_tokens < num_tokens_requested:
                num_epochs += 1
                num_tokens += num_tokens_per_epoch
        return num_epochs


def _build_document_index(
    documents: numpy.ndarray,
    num_epochs: int,
    numpy_random_state: numpy.random.RandomState,
    separate_final_epoch: bool,
) -> numpy.ndarray:
    """Build an array with length = num epochs * num documents

    Args:
        documents (numpy.ndarray): the subset of exposed document indices

        num_epochs (int): The number of epochs

        numpy_random_state (numpy.random.RandomState): The NumPy random state

        separate_final_epoch (bool): Whether to exclude the last epoch from the global shuffle

    Returns:
        numpy.ndarray: The document index
    """
    if not separate_final_epoch or num_epochs == 1:
        document_index = numpy.mgrid[0:num_epochs, 0 : len(documents)][1]
        document_index[:] = documents
        document_index = document_index.reshape(-1)
        document_index = document_index.astype(numpy.int32)
        log_single_rank(
            logger,
            logging.INFO,
            f"> Start to shuffle the document index with {len(document_index)} documents",
        )
        numpy_random_state.shuffle(document_index)
        log_single_rank(logger, logging.INFO, f"> DONE shuffling the document index")
        
        return document_index

    doc_idx_first = _build_document_index(documents, num_epochs - 1, numpy_random_state, False)
    doc_idx_last = _build_document_index(documents, 1, numpy_random_state, False)
    return numpy.concatenate((doc_idx_first, doc_idx_last))


def _build_shuffle_index(
    num_samples: int, total_size: int, numpy_random_state: numpy.random.RandomState
) -> numpy.ndarray:
    """Build the range [0, size) and shuffle

    Args:
        num_samples (int): The size of the first shuffle range [0, num_samples)

        total_size (int): The size of the entire index. If larger than 'num_samples', it defines
            the second shuffle range [num_samples, total_size)

        numpy_random_state (numpy.random.RandomState): The NumPy random state

    Returns:
        numpy.ndarray: The shuffle index
    """
    dtype_ = numpy.uint32
    if total_size >= (numpy.iinfo(numpy.uint32).max - 1):
        dtype_ = numpy.int64

    shuffle_idx_first = numpy.arange(start=0, stop=num_samples, step=1, dtype=dtype_)
    numpy_random_state.shuffle(shuffle_idx_first)
    if num_samples == total_size:
        return shuffle_idx_first

    shuffle_idx_last = numpy.arange(start=num_samples, stop=total_size, step=1, dtype=dtype_)
    numpy_random_state.shuffle(shuffle_idx_last)

    return numpy.concatenate((shuffle_idx_first, shuffle_idx_last))


def _get_ltor_masks_and_position_ids(
    data: torch.Tensor,
    eod_token: int,
    reset_position_ids: bool,
    reset_attention_mask: bool,
    eod_mask_loss: bool,
    create_attention_mask: bool,
):
    """Build masks and position id for left to right model.

    Args:
        data (torch.Tensor): The data tenor that holds the tokens from the dataset

        eod_token (int): ID of the token to that is considered the EOD

        reset_position_ids (bool): Switch to reset the document position ID's

        reset_attention_mask (bool): Switch to reset the attention mask

        eod_mask_loss (bool): Switch to enable the EOD mask loss

        create_attention_mask (bool): Switch to enable the attention masks generation. Can be
            disabled if attention kernel generates masks by itself.

    Returns:
        torch.Tensor: Attention mask needed to be used for Attention

        torch.Tensor: The mask used for loss value during training

        torch.Tensor: The position ID's of the token
    """
    seq_length = data.numel()

    if create_attention_mask:
        attention_mask = torch.tril(
            torch.ones((seq_length, seq_length), device=data.device)
        ).unsqueeze(0)
    else:
        attention_mask = None

    # Loss mask.
    loss_mask = torch.ones(seq_length, dtype=torch.float, device=data.device)
    if eod_mask_loss:
        loss_mask[data == eod_token] = 0.0

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long, device=data.device)
    # We need to clone as the ids will be modifed based on batch index.
    if reset_position_ids:
        position_ids = position_ids.clone()

    if reset_position_ids or reset_attention_mask:
        # Find indices where EOD token is.
        eod_index = position_ids[data == eod_token]
        # Detach indices from positions if going to modify positions.
        if reset_position_ids:
            eod_index = eod_index.clone()

        # Loop through EOD indices:
        prev_index = 0
        for j in range(eod_index.numel()):
            i = eod_index[j]
            # Mask attention loss.
            if reset_attention_mask and attention_mask is not None:
                attention_mask[0, (i + 1) :, : (i + 1)] = 0
            # Reset positions.
            if reset_position_ids:
                position_ids[(i + 1) :] -= i + 1 - prev_index
                prev_index = i + 1

    if attention_mask is not None:
        # Convert attention mask to binary:
        attention_mask = attention_mask < 0.5

    return attention_mask, loss_mask, position_ids

