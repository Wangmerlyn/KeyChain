#!/bin/bash
# wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json

# download musique
# ===============================================

set -e
set -x

# If gdown doesn't work, you can download files from mentioned URLs manually
# and put them at appropriate locations.
pip install gdown

ZIP_NAME="musique_v1.0.zip"

# URL: https://drive.google.com/file/d/1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h/view?usp=sharing
gdown --id 1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h --output $ZIP_NAME
unzip $(basename $ZIP_NAME)
rm $ZIP_NAME

# TODO: prevent these from zipping in.
rm -rf __MACOSX

# SAVE_DIR="./"
# SAVE_NAME="hotpotqa"
# TOKENIZER_PATH="/mnt/longcontext/models/siyuan/llama3/llama-3.1-8B-instruct"
# TOKENIZER_TYPE="hf"
# MAX_SEQ_LENGTH=131072
# TOKENS_TO_GENERATE=128
# NUM_SAMPLES=100
# TEMPLATE="{context}"
# DATASET="hotpot_train_v1.1.json"

# python qa.py \
#     --save_dir=${SAVE_DIR} \
#     --save_name=${SAVE_NAME} \
#     --tokenizer_path=${TOKENIZER_PATH} \
#     --tokenizer_type=${TOKENIZER_TYPE} \
#     --max_seq_length=${MAX_SEQ_LENGTH} \
#     --tokens_to_generate=${TOKENS_TO_GENERATE} \
#     --num_samples=${NUM_SAMPLES} \
#     --template="${TEMPLATE}" \
#     --dataset=${DATASET}
