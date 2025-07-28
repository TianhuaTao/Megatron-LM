
#### Protein
BASE_DIR=/workspace/data/bio/multilang/raw/protein/splited
ls ${BASE_DIR}/*.train
ls ${BASE_DIR}/*.valid
ls ${BASE_DIR}/*.test

# train
ALLFILES=$(ls ${BASE_DIR}/*.train)
# add valid
ALLFILES="$ALLFILES $(ls ${BASE_DIR}/*.valid)"

ALLFILES="$ALLFILES $(ls ${BASE_DIR}g*.test)"

for file in $ALLFILES; do
    output_path=/workspace/data/bio/multilang/megatron-processed/protein/$(basename $file)
    echo python tools/preprocess_data_multilang.py --input $file --output-prefix $output_path --vocab-file /workspace/data/bio/multilang/raw/vocab_multilang.json --workers 64 --add-bos --add-eos --namespace P --input-format lines
done


#####





##################### DNA
BASE_DIR=/workspace/data/bio/multilang/raw/dna/long-sequence/pretraining_jsonl/gtdb_220/with_ssu/with_ssu
OUT_BASE=/workspace/data/bio/multilang/megatron-processed/dna/long-sequence/pretraining_jsonl/gtdb_220/with_ssu/with_ssu
mkdir -p ${OUT_BASE}
ls ${BASE_DIR}/train
ls ${BASE_DIR}/test

# train
ALLFILES=$(ls ${BASE_DIR}/train/train*.jsonl)

# add test
ALLFILES="$ALLFILES $(ls ${BASE_DIR}/test/test*.jsonl)"

for file in $ALLFILES; do
    output_path=$OUT_BASE/$(basename $file)
    echo python tools/preprocess_data_multilang.py --input $file --output-prefix $output_path --vocab-file /workspace/data/bio/multilang/raw/vocab_multilang.json --workers 64 --add-bos --add-eos --namespace D --input-format jsonl
done


### 
