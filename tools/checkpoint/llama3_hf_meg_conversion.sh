# hf --> meg

### NOTE: Please first download llama3 checkpoint from Hugging Face model hub and place it in the models directory, (e.g., models/Meta-Llama-3-8B)

CONVERT_ARGS="--model-type GPT --loader llama_mistral --saver megatron --load-dir models/Meta-Llama-3-8B --save-dir tmp/meg/Meta-Llama-3-8B-bf16"
SAVER_ARGS="--target-tensor-parallel-size 1 --target-pipeline-parallel-size 1"
LOADER_ARGS="--model-size llama3-8B --checkpoint-type hf --bf16 --tokenizer-model models/Meta-Llama-3-8B/original/tokenizer.model --megatron-path ./Megatron-LM --true-vocab-size 128256"


python Megatron-LM/tools/checkpoint/convert.py \
    $CONVERT_ARGS \
    $SAVER_ARGS \
    $LOADER_ARGS

##########################################

# meg --> hf
CONVERT_ARGS="--model-type GPT --loader megatron --saver llama2_hf --load-dir tmp/meg/Meta-Llama-3-8B-bf16 --save-dir tmp/hf/Meta-Llama-3-8B-bf16"
SAVER_ARGS="--hf-config-path models/Meta-Llama-3-8B"
LOADER_ARGS="--megatron-path ./Megatron-LM --true-vocab-size 128256 --position-embedding-type rope"


python Megatron-LM/tools/checkpoint/convert.py \
    $CONVERT_ARGS \
    $SAVER_ARGS \
    $LOADER_ARGS
