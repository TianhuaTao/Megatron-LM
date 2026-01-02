#! /bin/bash
# usage: ./example_node_cmd.sh --dryrun [NODE_RANK_ID] [HOST_FILE_PATH] [TIMESTAMP]

WORKSPACE_DIR='/jfs/tianhua-tao/ws-ds-protein'

set -eo pipefail

# Defaults
WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"   # keep yours if already set elsewhere
NODE_RANK_ID=0
HOST_FILE_PATH="${WORKSPACE_DIR}/hostfile1"
TIMESTAMP="latest"
DRYRUN=0
DEBUG=0

usage() {
  cat <<'EOF'
Usage:
  script.sh [--dryrun] [--debug] [NODE_RANK_ID] [HOST_FILE_PATH] [TIMESTAMP]

Options:
  --dryrun    Print commands instead of running them
  -h, --help  Show help
EOF
}

# Parse options; keep leftover positionals
positional=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dryrun) DRYRUN=1; shift ;;
    --debug) DEBUG=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; positional+=("$@"); break ;;
    -*) echo "Unknown option: $1" >&2; usage; exit 2 ;;
    *) positional+=("$1"); shift ;;
  esac
done

# Map leftover positionals to your vars (with defaults)
if [[ ${#positional[@]} -ge 1 ]]; then NODE_RANK_ID="${positional[0]}"; fi
if [[ ${#positional[@]} -ge 2 ]]; then HOST_FILE_PATH="${positional[1]}"; fi
if [[ ${#positional[@]} -ge 3 ]]; then TIMESTAMP="${positional[2]}"; fi
if [[ ${#positional[@]} -gt 3 ]]; then
  echo "Too many positional arguments: ${positional[*]}" >&2
  usage
  exit 2
fi

run() {
  if (( DRYRUN )); then
    echo "[dryrun] $*"
  else
    "$@"
  fi
}

echo "NODE_RANK_ID=$NODE_RANK_ID"
echo "HOST_FILE_PATH=$HOST_FILE_PATH"
echo "TIMESTAMP=$TIMESTAMP"
echo "DRYRUN=$DRYRUN"

NUM_NODES=$(wc -l < ${HOST_FILE_PATH})

echo $NODE_RANK_ID $HOST_FILE_PATH $NUM_NODES


############## High-level configs ############## BEGIN

USE_FP8=0
USE_PROFILE=0
DROP_TOKENS=1
CAPACITY_FACTOR=1.2
NODE_NETWORK_TYPE="tcpxo" # ib/tcpxo
USE_RECOMPUTE=0
# TIMESTAMP="DEBUG"
PAD_MOE_TO_CAP=0
AUX_FREE=0
############## High-level configs ############## END


NUM_GPUS_PER_WORKER=8
# WORKER_HOST_GROUP="A" # sometimes we devide 32 workers into multiple groups, A and B, etc.

# enable ssh
# service ssh start


if [ $NODE_NETWORK_TYPE == "ib" ]; then
        echo "Using Infiniband"

        # Use all interfaces starting with `ib`. This selects the IB cards and avoids 
        # interfaces with names like bond0 and enp0, which are usually ethernet devices.
        # Ethernet networks are not robust/fast enough for most distributed training workloads.
        # https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-socket-ifname
        export NCCL_SOCKET_IFNAME=ib

        # Don't use the IB bond (which uses the attached ethernet cards) for the same reason.
        # https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-ib-hca
        export NCCL_IB_HCA=^=mlx5_bond_0
elif [ $NODE_NETWORK_TYPE == "tcpxo" ]; then
        echo "Using TCPXO"
        # setup tcpxo 
        # export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-} # make sure LD_LIBRARY_PATH is exported, otherwise, sourcing nccl-env-profile.sh might error on unbound variable
        NCCL_LIB_DIR="/usr/local/nvidia/lib64" source /usr/local/nvidia/lib64/nccl-env-profile.sh
        # export NCCL_TUNER_CONFIG_PATH=/usr/local/nvidia/lib64/a3plus_tuner_config.textproto
        # export NCCL_NET=FasTrak # optional, it should find this automatically if everything is set correctly
else
        echo "Unknown network type"
        exit 1
fi


export WANDB_API_KEY=4fe4dcd1bce264ca548c8676c4472a16165c512d
export PATH=/usr/local/nvidia/bin:$PATH # nvida-smi

cd ${WORKSPACE_DIR}/Megatron-LM

OPTIONS_NCCL="CUDA_LAUNCH_BLOCKING=0"
OPTIONS_NCCL="$OPTIONS_NCCL NCCL_DEBUG=INFO"
OPTIONS_NCCL="$OPTIONS_NCCL NCCL_DEBUG_SUBSYS=NET"
OPTIONS_NCCL="$OPTIONS_NCCL CUDA_DEVICE_MAX_CONNECTIONS=1"
# OPTIONS_NCCL="$OPTIONS_NCCL NCCL_NET_GDR_LEVEL=PIX"

OTHER_OPTIONS="OMP_NUM_THREADS=8 NVTE_FLASH_ATTN=1 NVTE_FUSED_ATTN=1 NVTE_ALLOW_NONDETERMINISTIC_ALGO=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

# Options
        # NVTE_FLASH_ATTN=1
        # NVTE_FUSED_ATTN=1
        # NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=0 
        # NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2



TRAIN_DATA_PATH="${WORKSPACE_DIR}/data/core/splited/splited/megatron/Uniref_90_Rep.fasta.train_text_document"
VALID_DATA_PATH="${WORKSPACE_DIR}/data/core/splited/splited/megatron/Uniref_90_Rep.fasta.valid_text_document"
TEST_DATA_PATH="${WORKSPACE_DIR}/data/core/splited/splited/megatron/Uniref_90_Rep.fasta.test_text_document"
# S3_CACHE_PATH="${WORKSPACE_DIR}/data/s3_cache"


# https://huggingface.co/genbio-ai/AIDO.Protein-16B/blob/main/vocab_protein.txt
# VOCAB=${WORKSPACE_DIR}/data/vocab_protein.txt
VOCAB=${WORKSPACE_DIR}/data/vocab_protein.json
DATA_CACHE_DIR="${WORKSPACE_DIR}/data/megatron/cache"


MICRO_BATCH_SIZE=8
GLOBAL_BATCH_SIZE=2048

TP_SIZE=1
PP_SIZE=1
EP_PARALLEL_SIZE=8
NUM_EXPERT=64
TOPK=16
TOTAL_LEN=2048

SHARED_EXP=0
SHARED_EXP_FFN=1024

NHIDDEN=2048
# FFN_HIDDEN=3584
FFN_HIDDEN=512
NLAYERS=16
NHEADS=32
LRMAX=3e-4
LRMIN=3e-6

CLIP_GRAD=1.0
# LR_WARMUP_STEPS=2000
LR_WARMUP_STEPS=2000

# layer * (4 * Hidden * Hidden + 3 * topK * Hidden *FFN)/10^9
# truncate to 1 decimal places
VOCAB_SIZE=128
EMBED_PARAM=$(echo "scale=3; 2 * ${NHIDDEN} * ${VOCAB_SIZE} / 1000000000" | bc | awk '{printf "%.3f\n", $0}')
echo "EMBED_PARAM: ${EMBED_PARAM}B"

SPARSE_PARAM_NO_EMBED=$(echo "scale=3; ${NLAYERS} * (4 * ${NHIDDEN} * ${NHIDDEN} + 3 * (${NUM_EXPERT} * ${NHIDDEN} * ${FFN_HIDDEN} + ${SHARED_EXP} * ${NHIDDEN} * ${SHARED_EXP_FFN}) ) / 1000000000" | bc)
SPARSE_PARAM=$(echo "scale=3; (${SPARSE_PARAM_NO_EMBED} + ${EMBED_PARAM})/1" | bc | awk '{printf "%.2f\n", $0}')
echo "SPARSE_PARAM: ${SPARSE_PARAM}B"

ACTIVE_PARAM_NO_EMBED=$(echo "scale=3; ${NLAYERS} * (4 * ${NHIDDEN} * ${NHIDDEN} + 3 * (${TOPK} * ${NHIDDEN} * ${FFN_HIDDEN} + ${SHARED_EXP} * ${NHIDDEN} * ${SHARED_EXP_FFN} ) ) / 1000000000" | bc)
ACTIVE_PARAM=$(echo "scale=3; (${ACTIVE_PARAM_NO_EMBED} + ${EMBED_PARAM})/1" | bc | awk '{printf "%.2f\n", $0}')
echo "ACTIVE_PARAM: ${ACTIVE_PARAM}B"


EXP_NAME="meg_${ACTIVE_PARAM}B-${SPARSE_PARAM}B_${NLAYERS}L"

if [ $SHARED_EXP -eq 0 ]; then
        EXP_NAME="${EXP_NAME}_${NUM_EXPERT}N${TOPK}K_${NHIDDEN}H-${FFN_HIDDEN}F"
else
        EXP_NAME="${EXP_NAME}_${NUM_EXPERT}N${SHARED_EXP}S${TOPK}K_${NHIDDEN}H-${FFN_HIDDEN}F-${SHARED_EXP_FFN}S"
fi
EXP_NAME="${EXP_NAME}_${NHEADS}H_${TOTAL_LEN}L_${GLOBAL_BATCH_SIZE}B${MICRO_BATCH_SIZE}M_${LR_WARMUP_STEPS}WA_${TP_SIZE}TP${PP_SIZE}PP${EP_PARALLEL_SIZE}EP"
# _${NUM_NODES}NODE

# A - standard
# B += --qk-layernorm --rotary-base 500000
TAG="staged-${NODE_NETWORK_TYPE}"

if [ $AUX_FREE -eq 1 ]; then
        TAG="$TAG-auxfree"
fi

if [ $DROP_TOKENS -eq 1 ]; then
        TAG="$TAG-cap${CAPACITY_FACTOR}"
        if [ $PAD_MOE_TO_CAP -eq 1 ]; then
                TAG="$TAG-pad"
        fi
fi


if [ $USE_FP8 -eq 1 ]; then
        TAG="$TAG-fp8"
fi

# if [ $USE_PROFILE -eq 1 ]; then
#         TAG="$TAG-profile"
# fi

# 

# if "TAG" is set and not empty, add it to the EXP_NAME
if [ -n "${TAG}" ]; then
    EXP_NAME="${EXP_NAME}-${TAG}"
fi
echo "EXP_NAME: ${EXP_NAME}"

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


### MoE configs

EXPERT_INTERVAL=1


MLC=0.01



SAVE_INTERVAL=1000
# SAVE_TMP_INTERVAL=10
EVAL_INTERVAL=500


# TRAIN_TOKENS=$(echo '120_000_000_000' | sed 's/_//g') # 120B
TRAIN_TOKENS=$(echo '1_000_000_000_000' | sed 's/_//g') # 120B
# TRAIN_TOKENS=$(echo '10_000_000_000' | sed 's/_//g') # 10B



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
    --tensorboard-queue-size 10 \
    --tensorboard-log-interval 1 \
    --log-throughput \
    --log-timers-to-tensorboard \
    --log-validation-ppl-to-tensorboard \
    --wandb-project protein-moe \
    --wandb-exp-name ${EXP_NAME} \
    --wandb-save-dir ${ARTIFACTS_RUN_DIR} \
    "
#     --log-params-norm \


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
       --position-embedding-type rope \
       --ffn-hidden-size $FFN_HIDDEN \
       --swiglu \
       --short-seq-prob 0.02 \
       --mask-prob 0.15 \
       --disable-bias-linear \
       --no-bias-gelu-fusion \
       --tokenizer-type ProteinTokenizer \
       --vocab-file $VOCAB \
       --untie-embeddings-and-output-weights \
       --normalization RMSNorm \
       --attention-dropout 0.1  \
       --hidden-dropout 0.1 \
       --use-distributed-optimizer \
       --apply-layernorm-1p "

#        --tokenizer-model allenai/dolma2-tokenizer \

# LM_ARGS="$LM_ARGS --overlap-grad-reduce --overlap-param-gather" # TE>=2.0 bug? does not work with overlap?
# LM_ARGS="$LM_ARGS --no-gradient-accumulation-fusion" # container 25.02 bug too

if [ $USE_FP8 -eq 1 ]; then
        LM_ARGS="$LM_ARGS --fp8-format hybrid --fp8-margin 0 --fp8-amax-history-len 1024 --fp8-amax-compute-algo max"
fi

if [ $USE_PROFILE -eq 1 ]; then
        LM_ARGS="$LM_ARGS --profile --profile-step-start 2011 --profile-step-end 2014 --profile-ranks 0 8 16 24"
fi


# --num-layers-per-virtual-pipeline-stage 2
if [ $PP_SIZE -gt 1 ]; then
        LM_ARGS="$LM_ARGS --num-virtual-stages-per-pipeline-rank 2 --overlap-p2p-communication-warmup-flush"
fi
#  --account-for-embedding-in-pipeline-split



# enable --sequence-parallel if TP_SIZE > 1
if [ $TP_SIZE -gt 1 ]; then
        LM_ARGS="$LM_ARGS --sequence-parallel"
        LM_ARGS="$LM_ARGS --tp-comm-overlap --tp-comm-overlap-rs-dgrad"
fi




# LM_ARGS="$LM_ARGS \
#         --rampup-batch-size 512 512 1000000 
#         "

#        --overlap-param-gather-with-optimizer-step \




if [ $NUM_EXPERT -eq 1 ]; then
        MoE_ARGS=" "
else
        MoE_ARGS=" \
        --num-experts ${NUM_EXPERT} \
        --moe-layer-freq 1 \
        --moe-router-topk ${TOPK} \
        --expert-model-parallel-size ${EP_PARALLEL_SIZE} \
        --moe-z-loss-coeff 1e-3 \
        --moe-token-dispatcher-type alltoall \
        --moe-grouped-gemm "
        # --moe-per-layer-logging \
        # --overlap-moe-expert-parallel-comm \

        MoE_ARGS="$MoE_ARGS --moe-permute-fusion --moe-router-dtype fp32"
        if [ $DROP_TOKENS -eq 1 ]; then
                MoE_ARGS="$MoE_ARGS --moe-expert-capacity-factor ${CAPACITY_FACTOR}"
                if [ $PAD_MOE_TO_CAP -eq 1 ]; then
                        MoE_ARGS="$MoE_ARGS --moe-pad-expert-input-to-capacity"
                fi
        fi


        # MoE_ARGS="$MoE_ARGS --expert-tensor-parallel-size 1" # disable tensor parallelism for experts

        if [ $SHARED_EXP -gt 0 ]; then  # using shared experts
                MoE_ARGS="$MoE_ARGS --moe-shared-expert-intermediate-size ${SHARED_EXP_FFN} --moe-shared-expert-overlap"
        fi
        
        if [ $AUX_FREE -eq 1 ]; then
                # aux-loss-free load balancing
                MoE_ARGS="$MoE_ARGS --moe-router-load-balancing-type seq_aux_loss --moe-aux-loss-coeff 1e-4 --moe-router-enable-expert-bias --moe-router-bias-update-rate 1e-3 --moe-router-score-function sigmoid "
        else
                # aux loss
                MoE_ARGS="$MoE_ARGS --moe-router-load-balancing-type aux_loss --moe-aux-loss-coeff ${MLC}"
                # tmp for staged model
                MoE_ARGS="$MoE_ARGS --moe-router-enable-expert-bias --moe-router-bias-update-rate 5e-4 --moe-router-score-function sigmoid "
        fi

fi

        #  \
        # --moe-enable-deepep \
        # --moe-token-dispatcher-type flex \
       

if [ $USE_RECOMPUTE -eq 1 ]; then
      LM_ARGS="$LM_ARGS \
        --recompute-num-layers 1 \
        --recompute-method uniform \
        --recompute-granularity full "
fi


gpt_options=" \
       $LM_ARGS \
       --micro-batch-size $MICRO_BATCH_SIZE \
       --global-batch-size $GLOBAL_BATCH_SIZE \
       --train-samples $TRAIN_SAMPLES \
       --seq-length $SEQ_LEN \
       --max-position-embeddings $SEQ_LEN \
       --num-workers 2 \
       --train-data-path $TRAIN_DATA_PATH \
       --valid-data-path $VALID_DATA_PATH \
       --test-data-path $TEST_DATA_PATH \
       --num-dataset-builder-threads 2 \
       --data-cache-path $DATA_CACHE_DIR \
       --save $CHECKPOINT_PATH \
       --load $LOAD_DIR \
       --distributed-backend nccl \
       --init-method-std 0.01 \
       --bf16 \
       --use-flash-attn \
       $OPTIMIZER_ARGS \
       $OUTPUT_ARGS \
       $RESUME_ARGS \
       $MoE_ARGS 
"
#        --s3-cache-path $S3_CACHE_PATH \

#        --per-split-data-args-path $DATA_ARGS_PATH \

# pete baseline
gpt_options="$gpt_options --qk-layernorm --rotary-base 500000"



### ds_config for FP16
mkdir -p ${ARTIFACTS_RUN_DIR}


port=24759

NODE0=$(head -n 1 "$HOST_FILE_PATH" | awk '{print $1}')


script_path="${WORKSPACE_DIR}/Megatron-LM/pretrain_protein_lm.py"
# script_path="${WORKSPACE_DIR}/Megatron-LM/scripts/min_torchrun.py"

if [ $DEBUG -eq 1 ]; then
    TORCHRUN="${WORKSPACE_DIR}/Megatron-LM/scripts/torchrun_debug.py"
else
    TORCHRUN="torchrun"
fi


# export PYTHONPATH="${WORKSPACE_DIR}/Megatron-LM:$PYTHONPATH"
# report_mem_cmd="${OPTIONS_NCCL} ${OTHER_OPTIONS} torchrun --rdzv_endpoint $NODE0:$port --rdzv_id 10086 --rdzv_backend c10d --nnodes ${NUM_NODES} --nproc-per-node ${NUM_GPUS_PER_WORKER} --node_rank "${NODE_RANK_ID}" ${report_mem_path} ${gpt_options}"
# run_cmd=${report_mem_cmd}
if [ $USE_PROFILE -eq 1 ]; then
        run_cmd="${OPTIONS_NCCL} ${OTHER_OPTIONS} NSYS_ENABLE_PYTHON_SOURCE_CORRELATION=1 nsys profile \
        -t nvtx,cuda,osrt \
        --sample=cpu \
        --capture-range=cudaProfilerApi \
        --capture-range-end=stop \
        --force-overwrite true \
        --trace-fork-before-exec='true' \
        -o ${ARTIFACTS_RUN_DIR}/megatron_profile_${NODE_RANK_ID} \
        $TORCHRUN --rdzv_endpoint $NODE0:$port --rdzv_id 20086 --rdzv_backend c10d --nnodes ${NUM_NODES} --nproc-per-node ${NUM_GPUS_PER_WORKER} --node_rank "${NODE_RANK_ID}" ${script_path} ${gpt_options}"
else
        run_cmd="${OPTIONS_NCCL} ${OTHER_OPTIONS} $TORCHRUN --rdzv_endpoint $NODE0:$port --rdzv_id 20086 --rdzv_backend c10d --nnodes ${NUM_NODES} --nproc-per-node ${NUM_GPUS_PER_WORKER} --node_rank "${NODE_RANK_ID}" ${script_path} ${gpt_options}"
fi



if [[ $NODE_RANK_ID -eq 0 ]]; then
        cp $0 ${ARTIFACTS_RUN_DIR}/ # save the script
        cp ${HOST_FILE_PATH} ${ARTIFACTS_RUN_DIR}/ # save the hostfile
fi
echo ${run_cmd} | tee ${ARTIFACTS_RUN_DIR}/run_cmd.${NODE_RANK_ID}.txt
mkdir -p ${ARTIFACTS_RUN_DIR}/nvidia-smi
nvidia-smi > ${ARTIFACTS_RUN_DIR}/nvidia-smi/${NODE_RANK_ID}.${HOSTNAME}.txt

if (( DRYRUN )); then
    echo "[dryrun] complete"
  else
    eval ${run_cmd} 2>&1 | tee ${ARTIFACTS_RUN_DIR}/output.${NODE_RANK_ID}.log
  fi

