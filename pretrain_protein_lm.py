# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain GPT."""

import os
import torch
from functools import partial
from contextlib import nullcontext
import inspect

from typing import List, Optional, Tuple, Union
from megatron.training import get_args
from megatron.training import print_rank_0
from megatron.training import get_timers
from megatron.training import get_tokenizer
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.utils import get_blend_from_list
# from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
# from megatron.core.datasets.gpt_dataset import MockGPTDataset, GPTDataset
from megatron.core.datasets.concat_masked_dataset import ConcatMaskedDatasetConfig, ConcatMaskedDataset
from megatron.core.rerun_state_machine import get_rerun_state_machine
import megatron.legacy.model
from megatron.core.models.gpt import GPTModel
from megatron.training import pretrain
from megatron.core.utils import StragglerDetector
from megatron.core.transformer.spec_utils import import_module
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
    get_blend_and_blend_per_split,
)
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.yaml_arguments import core_transformer_config_from_yaml
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_mtp_block_spec,
    # get_gpt_layer_with_transformer_engine_spec_custom_mask
)
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules
from copy import deepcopy
from megatron.core.jit import jit_fuser
from megatron.core.transformer.enums import AttnMaskType
from model_provider import model_provider


stimer = StragglerDetector()

def add_extra_override_config(config):
    config.layer_override_args_dict = {}
    
    # get string from env "LAYER_FULL_RECOMPUTE"
    layer_full_recompute = os.environ.get("LAYER_FULL_RECOMPUTE", None) # example "[1,1,0,0]+[0,0,0,0]"
    # convert to python list
    if layer_full_recompute is not None:
        assert False, "Layer full recompute override is not supported currently."
        layer_full_recompute = eval(layer_full_recompute) # 1 means full recompute, 0 means no recompute


        config_x = deepcopy(config)
        config_x.recompute_granularity = 'full'
        config_x.recompute_method = 'uniform'
        config_x.recompute_num_layers = 1
        config_x.moe_layer_recompute = False
        
        for i in range(len(layer_full_recompute)):
            if layer_full_recompute[i] == 1:
                config.layer_override_args_dict[i] = config_x
                print_rank_0(f"Layer {i} full recompute")
    
    return config

def protein_model_provider(args, pre_process, post_process, vp_stage=None, config=None, pg_collection=None) -> Union[GPTModel, megatron.legacy.model.GPTModel]:
    """Builds the model.

    The Protein-MoE model is a masked language model, but it takes the backbone from the GPT model. We need to chagne the causal attention to bidirectional attention. The GPT model seems to be better implemented than the BERT model, so we will use the GPT model as the backbone for the Protein-MoE model.
    
    It's also how it's done in the Megatron-DeepSpeed repo, where we migrated from.

    --------------------- Original docstring ---------------------
    If you set the use_legacy_models to True, it will return the legacy GPT model and if not the mcore GPT model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


    Returns:
        Union[GPTModel, megatron.legacy.model.GPTModel]: The returned model
    """

    print_rank_0('building GPT model ...')

    if config is None:
        if args.yaml_cfg is not None:
            config = core_transformer_config_from_yaml(args, "language_model")
        else:
            config = core_transformer_config_from_args(args)

    if args.use_legacy_models:
        assert False, "Protein-MoE is not implemented for the legacy GPT model"
        # model = megatron.legacy.model.GPTModel(
        #     config,
        #     num_tokentypes=0,
        #     parallel_output=True,
        #     pre_process=pre_process,
        #     post_process=post_process,
        # )
    else: # using core models
        if args.spec is not None:
            assert False, "Protein-MoE is only implemented with the default spec"
            transformer_layer_spec = import_module(args.spec)
        else:
            use_te = args.transformer_impl == "transformer_engine"

            if args.num_experts:
                # TODO: attention -> bidirectional attention
                # Define the decoder block spec
                transformer_layer_spec = get_gpt_decoder_block_spec(
                    config, 
                    use_transformer_engine=use_te, 
                    normalization=args.normalization, 
                    qk_l2_norm=args.qk_l2_norm,
                    vp_stage=vp_stage
                )
            else:
                # Define the decoder layer spec (same)
                transformer_layer_spec = get_gpt_decoder_block_spec(
                    config, 
                    use_transformer_engine=use_te, 
                    normalization=args.normalization, 
                    qk_l2_norm=args.qk_l2_norm,
                    vp_stage=vp_stage
                )
        mtp_block_spec = None
        if args.mtp_num_layers is not None:
            mtp_block_spec = get_gpt_mtp_block_spec(config, transformer_layer_spec, use_transformer_engine=use_te)


        print_rank_0(f"config:")
        print_rank_0(config)
        
        config = add_extra_override_config(config)
        for lspec in transformer_layer_spec.layer_specs:
            lspec.submodules.self_attention.params['attn_mask_type'] = AttnMaskType.no_mask
            

        model = GPTModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
            rotary_base=args.rotary_base,
            rope_scaling=args.use_rope_scaling,
            mtp_block_spec=mtp_block_spec,
            vp_stage=vp_stage,
            pg_collection=pg_collection,
        )

    return model


# def get_batch(data_iterator):
#     """Generate a batch."""

#     # # TODO: this is pretty hacky, find a better way
#     # if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):
#     #     return None, None, None, None, None

#     # get batches based on the TP rank you are on
#     batch = get_batch_on_this_tp_rank(data_iterator)

#     # slice batch along sequence dimension for context parallelism
#     batch = get_batch_on_this_cp_rank(batch)

#     return batch.values()

def get_batch(data_iterator):
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
        return None, None, None, None, None

    batch = {
           'tokens': data["tokens"].cuda(non_blocking = True),
           'labels': data["labels"].cuda(non_blocking = True),
           'loss_mask': data["loss_mask"].cuda(non_blocking = True),
           'attention_mask': None if "attention_mask" not in data else data["attention_mask"].cuda(non_blocking = True),
           'position_ids': data["position_ids"].cuda(non_blocking = True)
       }
    
    return batch.values()


# define spiky loss as a loss that's 10x the max loss observed
SPIKY_LOSS_FACTOR = 10

import nvtx
@jit_fuser
@nvtx.annotate('loss_func')
def loss_func(
    loss_mask: torch.Tensor, output_tensor: torch.Tensor, model: Optional[GPTModel] = None
):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses
        model (GPTModel, optional): The model (can be wrapped)

    Returns:
        the loss scalar for this micro-batch
        the number of non-padded tokens in this microbatch
        a dict containing reporting metrics on the loss and number of tokens across
            the data parallel ranks
    """
    args = get_args()

    losses = output_tensor.view(-1).float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses * loss_mask)

    # Check individual rank losses are not NaN prior to DP all-reduce.
    rerun_state_machine = get_rerun_state_machine()
    if args.check_for_nan_in_loss_and_grad:
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=torch.isnan,
            message="found NaN in local forward loss calculation",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=True,
        )
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=torch.isinf,
            message="found Inf in local forward loss calculation",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=True,
        )
    # Check for spiky loss
    if args.check_for_spiky_loss:
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=partial(
                rerun_state_machine.is_unexpectedly_large,
                threshold=SPIKY_LOSS_FACTOR,
                context="loss",
            ),
            message="Spiky loss",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=False,
        )

    num_tokens = loss_mask.sum().clone().detach().to(torch.int)
    reporting_loss = torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])

    return (loss, num_tokens, {'lm loss': reporting_loss})



def forward_step(data_iterator, model: GPTModel, return_schedule_plan: bool = False):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
    """
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    global stimer
    with stimer(bdata=True):
        tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
            data_iterator)
    timers('batch-generator').stop()

    with stimer:
        if args.use_legacy_models:
            output_tensor = model(tokens, position_ids, attention_mask,
                                labels=labels)
        else:
            if return_schedule_plan:
                assert args.overlap_moe_expert_parallel_comm, \
                    "overlap_moe_expert_parallel_comm must be enabled to return the schedule plan"
                schedule_plan = model.build_schedule_plan(
                    tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask
                )
                return schedule_plan, partial(loss_func, loss_mask, model=model)
            else:
                output_tensor = model(
                    tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask
                )

    return output_tensor, partial(loss_func, loss_mask)


def is_dataset_built_on_rank():
    # return (
    #     mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage()
    # ) and mpu.get_tensor_model_parallel_rank() == 0
    # return  mpu.get_tensor_model_parallel_rank() == 0
    return True



def core_gpt_dataset_config_from_args(args):
    tokenizer = get_tokenizer()

    # Sometimes --data-path is too long, instead we parse it from a file.
    blend: Optional[Tuple[List[str], Optional[List[float]]]]
    blend_per_split: Optional[List[Optional[Tuple[List[str], Optional[List[float]]]]]]
    blend, blend_per_split = get_blend_and_blend_per_split(args)

    return ConcatMaskedDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=blend,
        blend_per_split=blend_per_split,
        split=args.split,
        num_dataset_builder_threads=args.num_dataset_builder_threads,
        path_to_cache=args.data_cache_path,
        mmap_bin_files=args.mmap_bin_files,
        tokenizer=tokenizer,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        create_attention_mask=args.create_attention_mask_in_dataloader,
        s3_cache_path=args.object_storage_cache_path,
        masking_probability=args.mask_prob,
        short_sequence_probability=args.short_seq_prob,
    )


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    args = get_args()

    config = core_gpt_dataset_config_from_args(args)

    if args.mock_data:
        assert False, "Mock data is not implemented for the Protein-MoE model"
    else:
        dataset_type = ConcatMaskedDataset

    print_rank_0("> building train, validation, and test datasets for GPT ...")

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type,
        train_val_test_num_samples,
        is_dataset_built_on_rank,
        config
    ).build()

    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":
    
    import resource, os
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    print(f'PID {os.getpid()}  RLIMIT_NOFILE  soft={soft}  hard={hard}')
    
    # Temporary for transition to core datasets
    train_valid_test_datasets_provider.is_distributed = True

    pretrain(
        train_valid_test_datasets_provider,
        partial(model_provider, protein_model_provider),
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={'tokenizer_type': 'ProteinTokenizer'},
    )
