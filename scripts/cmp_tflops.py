

def num_floating_point_operations_old(args, batch_size):
    # Attention projection size.
    query_projection_size = args.kv_channels * args.num_attention_heads
    query_projection_to_hidden_size_ratio = query_projection_size / args.hidden_size
    # Group Query Attention.
    if not args.group_query_attention:
        args.num_query_groups = args.num_attention_heads
    # MoE.
    num_experts_routed_to = 1 if args.num_experts is None else args.moe_router_topk
    gated_linear_multiplier = 3 / 2 if args.swiglu else 1
    shared_expert_ffn_hidden_size = (
        0
        if args.moe_shared_expert_intermediate_size is None
        else args.moe_shared_expert_intermediate_size
    )
    if args.num_experts is None:
        ffn_hidden_size = args.ffn_hidden_size
    else:
        ffn_hidden_size = args.moe_ffn_hidden_size

    # The 12x term below comes from the following factors; for more details, see
    # "APPENDIX: FLOATING-POINT OPERATIONS" in https://arxiv.org/abs/2104.04473.
    # - 3x: Each GEMM in the model needs to be performed 3 times (forward pass,
    #       backward wgrad [weight gradient], backward dgrad [data gradient]).
    # - 2x: GEMMs of a particular size are stacked twice in the standard Transformer model
    #       architectures implemented in this codebase (e.g., h->ffn_h GEMM and ffn_h->h GEMM
    #       in MLP layer).
    # - 2x: A GEMM of a m*n tensor with a n*k tensor requires 2mnk floating-point operations.
    expansion_factor = 3 * 2 * 2
    attention = (
                (
                    1
                    + (args.num_query_groups / args.num_attention_heads)
                    + (args.seq_length / args.hidden_size)
                ) * query_projection_to_hidden_size_ratio
            )
    mlp = (
                (ffn_hidden_size / args.hidden_size)
                * num_experts_routed_to
                * gated_linear_multiplier
            )+ ((shared_expert_ffn_hidden_size / args.hidden_size) * gated_linear_multiplier)
    logit = (args.padded_vocab_size / (2 * args.num_layers * args.hidden_size))
    print( attention, mlp, logit)
    return (
        expansion_factor
        * batch_size
        * args.seq_length
        * args.num_layers
        * args.hidden_size
        * args.hidden_size
        * (
            # Attention.
            attention
            # MLP.
            + mlp
            # Shared Experts.
            
            # Logit.
            + logit
        )
    )


def num_floating_point_operations_new(args, batch_size):
    # Attention projection size.
    query_projection_size = args.kv_channels * args.num_attention_heads
    query_projection_to_hidden_size_ratio = query_projection_size / args.hidden_size
    # Group Query Attention.
    if not args.group_query_attention:
        args.num_query_groups = args.num_attention_heads
    # MoE.
    if args.num_experts is None:
        # Every Transformer MLP is dense.
        num_dense_layers = args.num_layers
        num_moe_layers = 0
        num_experts_routed_to = 0
    else:
        # Calculate number of dense and MoE Transformer MLPs.
        if isinstance(args.moe_layer_freq, int):
            moe_layer_pattern = [
                1 if (i % args.moe_layer_freq == 0) else 0 for i in range(args.num_layers)
            ]
        elif isinstance(args.moe_layer_freq, list):
            moe_layer_pattern = args.moe_layer_freq
        else:
            raise RuntimeError("Illegal --moe-layer-freq argument provided!")
        assert len(moe_layer_pattern) == args.num_layers
        num_moe_layers = sum(moe_layer_pattern)  # Number of 1s in `moe_layer_pattern`.
        num_dense_layers = args.num_layers - num_moe_layers
        num_experts_routed_to = args.moe_router_topk

    moe_ffn_hidden_size = args.moe_ffn_hidden_size if args.moe_ffn_hidden_size is not None else args.ffn_hidden_size
    shared_expert_ffn_hidden_size = (
        0
        if args.moe_shared_expert_intermediate_size is None
        else args.moe_shared_expert_intermediate_size
    )
    # SwiGLU.
    gated_linear_multiplier = 3 / 2 if args.swiglu else 1

    # The 12x term below comes from the following factors; for more details, see
    # "APPENDIX: FLOATING-POINT OPERATIONS" in https://arxiv.org/abs/2104.04473.
    # - 3x: Each GEMM in the model needs to be performed 3 times (forward pass,
    #       backward wgrad [weight gradient], backward dgrad [data gradient]).
    # - 2x: GEMMs of a particular size are stacked twice in the standard Transformer model
    #       architectures implemented in this codebase (e.g., h->ffn_h GEMM and ffn_h->h GEMM
    #       in MLP layer).
    # - 2x: A GEMM of a m*n tensor with a n*k tensor requires 2mnk floating-point operations.
    expansion_factor = 3 * 2 * 2
    attention = (
                (
                    1
                    + (args.num_query_groups / args.num_attention_heads)
                    # Only half of the attention matrix is non-zero and needs to be multiplied with V.
                    + (args.seq_length / args.hidden_size / 2)
                ) * query_projection_to_hidden_size_ratio
            )
    mlp= (
                (
                    # Dense.
                    (args.ffn_hidden_size * num_dense_layers) +
                    # MoE.
                    (
                        (
                            # Routed experts.
                            moe_ffn_hidden_size * num_experts_routed_to +
                            # Shared experts.
                            shared_expert_ffn_hidden_size
                        )
                        * num_moe_layers
                    )
                ) * gated_linear_multiplier / (args.num_layers * args.hidden_size)
            )
    logit = (args.padded_vocab_size / (2 * args.num_layers * args.hidden_size))
    print( attention, mlp, logit)
    
    return (
        expansion_factor
        * batch_size
        * args.seq_length
        * args.num_layers
        * args.hidden_size
        * args.hidden_size
        * (
            # Attention.
            attention
            # MLP.
            + mlp
            # Logit.
            + logit
        )
    )
    
    
if __name__ == "__main__":
    args = lambda: None
    args.batch_size = 4096
    args.num_layers = 32
    args.hidden_size = 4096
    args.seq_length = 4096
    args.ffn_hidden_size = 2048
    args.num_attention_heads = 32
    args.kv_channels = args.hidden_size/args.num_attention_heads
    args.group_query_attention = False
    args.num_experts = 64
    args.padded_vocab_size = 4096
    args.swiglu = True
    args.moe_router_topk = 2
    args.moe_shared_expert_intermediate_size = 8192
    args.moe_ffn_hidden_size = args.ffn_hidden_size
    args.moe_layer_freq = 1
    
    t1 = num_floating_point_operations_old(args, args.batch_size)
    t2 = num_floating_point_operations_new(args, args.batch_size)
    print(t1, t2)