#! /bin/bash
# usage: ./example_node_cmd.sh [SLURM_NODEID] [HOST_FILE_PATH]
SLURM_NODEID=${1:-0} # default to 0
HOST_FILE_PATH=${2:-"${WORKSPACE_DIR}/hostfiles/hostfile-1x8xdebug1"} 
TIMESTAMP=${3:-"latest"}

NUM_NODES=$(wc -l < ${HOST_FILE_PATH})

echo $SLURM_NODEID $HOST_FILE_PATH $NUM_NODES


NUM_GPUS_PER_WORKER=8
# WORKER_HOST_GROUP="A" # sometimes we devide 32 workers into multiple groups, A and B, etc.
WORKSPACE_DIR='/workspace'

# enable ssh
# service ssh start


# setup tcpxo 
NCCL_LIB_DIR="/var/lib/tcpxo/lib64" source /var/lib/tcpxo/lib64/nccl-env-profile.sh
NCCL_NET=FasTrak # optional, it should find this automatically if everything is set correctly
# I don't know why this is needed
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export NCCL_FASTRAK_DATA_TRANSFER_TIMEOUT_MS=600000 # 10 min 
export WANDB_API_KEY=61753d825c2bec08062290674ce9e3585bf31db3

cd ${WORKSPACE_DIR}/Megatron-LM

# OPTIONS_NCCL="NCCL_DEBUG=VERSION NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2"
OPTIONS_NCCL="CUDA_LAUNCH_BLOCKING=0"
# OPTIONS_NCCL="$OPTIONS_NCCL TORCH_CPP_LOG_LEVEL=INFO"
# OPTIONS_NCCL="$OPTIONS_NCCL TORCH_DISTRIBUTED_DEBUG=INFO"
OPTIONS_NCCL="$OPTIONS_NCCL NCCL_DEBUG=INFO"
OPTIONS_NCCL="$OPTIONS_NCCL NCCL_DEBUG_SUBSYS=INIT,NET"
OPTIONS_NCCL="$OPTIONS_NCCL CUDA_DEVICE_MAX_CONNECTIONS=1"

OTHER_OPTIONS="OMP_NUM_THREADS=8 NVTE_FLASH_ATTN=1 NVTE_FUSED_ATTN=1 NVTE_ALLOW_NONDETERMINISTIC_ALGO=1"
# NVTE_FLASH_ATTN=1
# NVTE_FUSED_ATTN=1
# NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=0 
# NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2

 


# new data TODO: check weights and path
# DATA_PATH=" \
#         0.252 ${WORKSPACE_DIR}/data/core/megatron/Uniref_90_Rep_text_document \
#         0.150 ${WORKSPACE_DIR}/data/core/megatron/colab_c_v2.1.sample_text_document \
#         0.412 ${WORKSPACE_DIR}/data/core/megatron/colabfold_v2.0_dupur90v2cfv1_text_document \
#         0.186 ${WORKSPACE_DIR}/data/core/megatron/colab_m_v2.1_text_document
#         "

TRAIN_DATA=" \
        0.25 ${WORKSPACE_DIR}/data/ai2-llm/megatron/dclm/text_openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train/allenai/dolma2-tokenizer/part-000-00000.npy_text_document \
        0.25 ${WORKSPACE_DIR}/data/ai2-llm/megatron/dclm/text_openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train/allenai/dolma2-tokenizer/part-003-00004.npy_text_document \
        0.25 ${WORKSPACE_DIR}/data/ai2-llm/megatron/dclm/text_openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train/allenai/dolma2-tokenizer/part-132-00002.npy_text_document \
        0.25 ${WORKSPACE_DIR}/data/ai2-llm/megatron/dclm/text_openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train/allenai/dolma2-tokenizer/part-179-00004.npy_text_document
        "

VALID_DATA=" \
        1.0 ${WORKSPACE_DIR}/data/ai2-llm/megatron/dclm/text_openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train/allenai/dolma2-tokenizer/part-186-00001.npy_text_document 
        "

TEST_DATA=" \
        1.0 ${WORKSPACE_DIR}/data/ai2-llm/megatron/dclm/text_openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train/allenai/dolma2-tokenizer/part-187-00004.npy_text_document 
        "




# VOCAB=/workspace/data/core/vocab.json
DATA_CACHE_DIR="${WORKSPACE_DIR}/data/ai2-llm/megatron/cache"


MICRO_BATCH_SIZE=8
GLOBAL_BATCH_SIZE=1024

TP_SIZE=1
PP_SIZE=4
EP_PARALLEL_SIZE=4
NUM_EXPERT=64
TOPK=8
TOTAL_LEN=4096

NHIDDEN=2048
FFN_HIDDEN=1024
NLAYERS=16
NHEADS=16
LRMAX=2e-4
LRMIN=2e-6

CLIP_GRAD=1.0
LR_WARMUP_STEPS=2000

# layer * (4 * Hidden * Hidden + 3 * topK * Hidden *FFN)/10^9
# truncate to 1 decimal places
VOCAB_SIZE=100278
EMBED_PARAM=$(echo "scale=3; 2 * ${NHIDDEN} * ${VOCAB_SIZE} / 1000000000" | bc | awk '{printf "%.3f\n", $0}')
echo "EMBED_PARAM: ${EMBED_PARAM}B"

SPARSE_PARAM_NO_EMBED=$(echo "scale=1; ${NLAYERS} * (4 * ${NHIDDEN} * ${NHIDDEN} + 3 * ${NUM_EXPERT} * ${NHIDDEN} * ${FFN_HIDDEN}) / 1000000000" | bc)
SPARSE_PARAM=$(echo "scale=1; (${SPARSE_PARAM_NO_EMBED} + ${EMBED_PARAM})/1" | bc)
echo "SPARSE_PARAM: ${SPARSE_PARAM}B"

ACTIVE_PARAM_NO_EMBED=$(echo "scale=1; ${NLAYERS} * (4 * ${NHIDDEN} * ${NHIDDEN} + 3 * ${TOPK} * ${NHIDDEN} * ${FFN_HIDDEN}) / 1000000000" | bc)
ACTIVE_PARAM=$(echo "scale=1; (${ACTIVE_PARAM_NO_EMBED} + ${EMBED_PARAM})/1" | bc)
echo "ACTIVE_PARAM: ${ACTIVE_PARAM}B"


EXP_NAME="meg_${ACTIVE_PARAM}B-${SPARSE_PARAM}B_${NLAYERS}L_${NUM_EXPERT}N_${TOPK}K_${NHIDDEN}DIM_${FFN_HIDDEN}FFN_${NHEADS}H_${TOTAL_LEN}L_${GLOBAL_BATCH_SIZE}GBS_${MICRO_BATCH_SIZE}MBS_${LR_WARMUP_STEPS}WA_${TP_SIZE}TP_${PP_SIZE}PP_${EP_PARALLEL_SIZE}EP"

# _${NUM_NODES}NODE



TAG="debug"
# 

# if "TAG" is set and not empty, add it to the EXP_NAME
if [ -n "${TAG}" ]; then
    EXP_NAME="${EXP_NAME}-${TAG}"
fi

INIT_DIR="" # for finetune over a pre-trained model


CHECKPOINT_PATH="${WORKSPACE_DIR}/checkpoint/${EXP_NAME}"

RESUME_ARGS="" # empty string

# if CHECKPOINT_PATH/latest exists, resume from CHECKPOINT_PATH, otherwise, resume from INIT_DIR

if [ -f "${CHECKPOINT_PATH}/latest_checkpointed_iteration.txt" ]; then
        # resume from the latest checkpoint
        echo "Resume from the latest checkpoint"
        LOAD_DIR="${CHECKPOINT_PATH}"
else
        if [ -f "${INIT_DIR}/latest_checkpointed_iteration.txt" ]; then
                # resume from the init checkpoint, discard the optimizer states
                echo "Start from init checkpoint, reset optimizer states"
                LOAD_DIR="${INIT_DIR}"
                RESUME_ARGS=""
        else
                # the init checkpoint does not exist, start from scratch, the LOAD_DIR is set, but not used
                echo "Start from scratch"
                LOAD_DIR="${CHECKPOINT_PATH}"
        fi
fi

# RESUME_ARGS=" --no-load-optim "
# LOAD_DIR="${CHECKPOINT_PATH}"
# RESUME_ARGS=" "

ARTIFACTS_BASE_DIR="${WORKSPACE_DIR}/artifacts"
ARTIFACTS_EXP_DIR="${ARTIFACTS_BASE_DIR}/${EXP_NAME}" # for experiment artifacts, which includes multiple runs
ARTIFACTS_RUN_DIR="${ARTIFACTS_EXP_DIR}/${TIMESTAMP}"
TENSORBOARD_PATH="${ARTIFACTS_EXP_DIR}" # in the root exp dir



LENGTH_PER_SAMPLE=${TOTAL_LEN} # sequence length per sample from BinaryDataset
SEQ_LEN=${TOTAL_LEN} # actual length during training (pad to this)

NUM_GPUS=$((NUM_NODES * NUM_GPUS_PER_WORKER))

### MoE configs

EXPERT_INTERVAL=1


MLC=0.01



SAVE_INTERVAL=50
# SAVE_TMP_INTERVAL=10
EVAL_INTERVAL=50


TRAIN_TOKENS=$(echo '120_000_000_000' | sed 's/_//g') # 120T



TRAIN_SAMPLES=$((TRAIN_TOKENS / SEQ_LEN))
LR_DECAY_SAMPLES=$((TRAIN_SAMPLES * 100 / 100))  # Decay for the first 10% tokens then continue at fixed --min-lr
# LR_WARMUP_SAMPLES=$((TRAIN_SAMPLES * 25 / 1000))  # 2.5% warmup
LR_WARMUP_SAMPLES=$(($LR_WARMUP_STEPS * $GLOBAL_BATCH_SIZE)) # based on steps



# LR_DECAY_SAMPLES=48828125
# LR_WARMUP_SAMPLES=1220703
# BATCH_WARMUP_SAMPLES=976562
# # --lr-decay-samples 48828125 --lr-warmup-samples 1220703 --rampup-batch-size 64 64 976562




OPTIMIZER_ARGS=" \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-8 \
    --lr ${LRMAX} \
    --min-lr ${LRMIN} \
    --lr-decay-style cosine \
    --lr-decay-samples $LR_DECAY_SAMPLES \
    --lr-warmup-samples $LR_WARMUP_SAMPLES \
    --clip-grad $CLIP_GRAD \
    --weight-decay 1e-1 \
    --seed 2024 \
    "


OUTPUT_ARGS=" \
    --log-interval 1 \
    --save-interval $SAVE_INTERVAL \
    --eval-interval $EVAL_INTERVAL \
    --eval-iters 5 \
    --tensorboard-dir $TENSORBOARD_PATH \
    --tensorboard-queue-size 5 \
    --log-throughput \
    --log-params-norm \
    --log-timers-to-tensorboard \
    --log-validation-ppl-to-tensorboard \
    --wandb-project tianhua-moe \
    --wandb-exp-name ${EXP_NAME} \
    --wandb-save-dir ${ARTIFACTS_RUN_DIR} \
    "

# there are problems with this
#     --non-persistent-save-interval $SAVE_TMP_INTERVAL \
#     --non-persistent-ckpt-type global \
#     --non-persistent-global-ckpt-dir $CHECKPOINT_PATH/non-persistent \

LM_ARGS="
       --tensor-model-parallel-size $TP_SIZE \
       --pipeline-model-parallel-size $PP_SIZE \
       --distributed-timeout-minutes 20 \
       --num-layers $NLAYERS \
       --hidden-size $NHIDDEN \
       --num-attention-heads $NHEADS \
       --make-vocab-size-divisible-by 128 \
       --short-seq-prob 0.00 \
       --mask-prob 0.15 \
       --position-embedding-type rope \
       --ffn-hidden-size $FFN_HIDDEN \
       --swiglu \
       --disable-bias-linear \
       --no-bias-gelu-fusion \
       --tokenizer-type HuggingFaceTokenizer \
       --tokenizer-model allenai/dolma2-tokenizer
       --untie-embeddings-and-output-weights \
       --overlap-grad-reduce \
       --num-layers-per-virtual-pipeline-stage 2 \
       --overlap-p2p-communication-warmup-flush \
       --overlap-param-gather \
       --use-distributed-optimizer \
       --normalization RMSNorm \
       --attention-dropout 0  \
       --hidden-dropout 0 \
       --apply-layernorm-1p "

#        --overlap-param-gather-with-optimizer-step \
    #    --vocab-file $VOCAB \

# enable --sequence-parallel if TP_SIZE > 1
if [ $TP_SIZE -gt 1 ]; then
        LM_ARGS="$LM_ARGS --sequence-parallel"
fi


# LM_ARGS="$LM_ARGS \
#         --rampup-batch-size 512 512 1000000 
#         "

#        --overlap-param-gather-with-optimizer-step \
#        --use-distributed-optimizer \
#        --overlap-param-gather \




#        --checkpoint-activations \
#        --recompute-activations \



MoE_ARGS=" \
        --num-experts ${NUM_EXPERT} \
        --moe-aux-loss-coeff ${MLC} \
        --moe-layer-freq 1 \
        --moe-router-topk ${TOPK} \
        --expert-model-parallel-size ${EP_PARALLEL_SIZE} \
        --moe-z-loss-coeff 1e-3 \
        --moe-per-layer-logging \
        --moe-token-dispatcher-type alltoall \
        --moe-layer-recompute \
        --moe-router-load-balancing-type aux_loss "

        # --recompute-num-layers 1 \
        # --recompute-method uniform \
        # --recompute-granularity full \
        # --moe-grouped-gemm \

gpt_options=" \
       $LM_ARGS \
       --micro-batch-size $MICRO_BATCH_SIZE \
       --global-batch-size $GLOBAL_BATCH_SIZE \
       --train-samples $TRAIN_SAMPLES \
       --seq-length $SEQ_LEN \
       --max-position-embeddings $SEQ_LEN \
       --num-workers 1 \
       --train-data-path $TRAIN_DATA \
       --valid-data-path $VALID_DATA \
       --test-data-path $TEST_DATA \
       --num-dataset-builder-threads 32 \
       --data-cache-path $DATA_CACHE_DIR \
       --save $CHECKPOINT_PATH \
       --load $LOAD_DIR \
       --distributed-backend nccl \
       --init-method-std 0.01 \
       --bf16 \
       --accumulate-allreduce-grads-in-fp32 \
       --use-flash-attn \
       $OPTIMIZER_ARGS \
       $OUTPUT_ARGS \
       $RESUME_ARGS \
       $MoE_ARGS 
"
# --data-path $DATA_PATH \
#        --split 978,20,2 \

#        --attention-softmax-in-fp32 \


### ds_config for FP16
mkdir -p ${ARTIFACTS_RUN_DIR}


port=4759

NODE0=$(head -n 1 "$HOST_FILE_PATH" | awk '{print $1}')


script_path="${WORKSPACE_DIR}/Megatron-LM/pretrain_gpt.py"


# export PYTHONPATH="${WORKSPACE_DIR}/Megatron-LM:$PYTHONPATH"
# report_mem_cmd="${OPTIONS_NCCL} ${OTHER_OPTIONS} torchrun --rdzv_endpoint $NODE0:$port --rdzv_id 10086 --rdzv_backend c10d --nnodes ${NUM_NODES} --nproc-per-node ${NUM_GPUS_PER_WORKER} --node_rank "${SLURM_NODEID}" ${report_mem_path} ${gpt_options}"
# run_cmd=${report_mem_cmd}
run_cmd="${OPTIONS_NCCL} ${OTHER_OPTIONS} torchrun --rdzv_endpoint $NODE0:$port --rdzv_id 10086 --rdzv_backend c10d --nnodes ${NUM_NODES} --nproc-per-node ${NUM_GPUS_PER_WORKER} --node_rank "${SLURM_NODEID}" ${script_path} ${gpt_options}"


if [[ $SLURM_NODEID -eq 0 ]]; then
        cp $0 ${ARTIFACTS_RUN_DIR}/ # save the script
        cp ${HOST_FILE_PATH} ${ARTIFACTS_RUN_DIR}/ # save the hostfile
fi
echo ${run_cmd} | tee ${ARTIFACTS_RUN_DIR}/run_cmd.${SLURM_NODEID}.txt
mkdir -p ${ARTIFACTS_RUN_DIR}/nvidia-smi
nvidia-smi > ${ARTIFACTS_RUN_DIR}/nvidia-smi/${SLURM_NODEID}.${HOSTNAME}.txt
eval ${run_cmd} 2>&1 | tee ${ARTIFACTS_RUN_DIR}/output.${SLURM_NODEID}.log
