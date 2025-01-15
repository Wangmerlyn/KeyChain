#!/bin/bash
# Toggle variables to control whether to download specific datasets
DOWNLOAD_HOTPOTQA=false
DOWNLOAD_MUSIQUE=false
DOWNLOAD_2WIKIMQA=false

# Dataset and save name selection
DATASETS=("hotpotqa" "musique" "2wikimqa")  # Options: "hotpotqa", "musique", "2wikimqa"
MAX_SEQ_LENGTHS=(4096 8192 16384 32768 65536 131072)

# Dataset download section
if [ "$DOWNLOAD_HOTPOTQA" = true ]; then
    echo "Downloading HotpotQA dataset..."
    wget -q http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json
fi

if [ "$DOWNLOAD_MUSIQUE" = true ]; then
    echo "Downloading Musique dataset..."
    pip install -q gdown
    ZIP_NAME="musique_v1.0.zip"
    # Use gdown to download the dataset from Google Drive
    gdown --id 1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h --output $ZIP_NAME
    unzip -q $(basename $ZIP_NAME)  # Unzip the downloaded file
    rm $ZIP_NAME                   # Remove the zip file after extraction
    rm -rf __MACOSX                # Clean up unwanted directories
fi

if [ "$DOWNLOAD_2WIKIMQA" = true ]; then
    echo "Downloading 2WikiMQA dataset..."
    wget -q -O 2wikimqa.zip https://www.dropbox.com/scl/fi/32t7pv1dyf3o2pp0dl25u/data_ids_april7.zip?rlkey=u868q6h0jojw4djjg7ea65j46&e=1
    unzip -q 2wikimqa.zip -d 2wikimqa  # Extract the dataset to the specified folder
    rm 2wikimqa.zip                   # Remove the zip file after extraction
fi

# Set parameters and loop through datasets and max_seq_lengths
SAVE_DIR="./"
TOKENIZER_PATH="/mnt/longcontext/models/siyuan/llama3/llama-3.1-8B-instruct"
TOKENIZER_TYPE="hf"
# MAX_SEQ_LENGTH=131072
TOKENS_TO_GENERATE=128
NUM_SAMPLES=100
TEMPLATE="{context}"

for DATASET_CHOICE in "${DATASETS[@]}"; do
    case "$DATASET_CHOICE" in
        "hotpotqa")
            SAVE_NAME="hotpotqa"
            DATASET="hotpot_train_v1.1.json"
            ;;
        "musique")
            SAVE_NAME="musique"
            DATASET="data/musique_full_v1.0_train.jsonl"
            ;;
        "2wikimqa")
            SAVE_NAME="2wikimqa"
            DATASET="2wikimqa/train.json"
            ;;
        *)
            # Handle invalid dataset selection
            echo "Invalid DATASET_CHOICE. Skipping..."
            continue
            ;;
    esac

    for MAX_SEQ_LENGTH in "${MAX_SEQ_LENGTHS[@]}"; do
        echo "Running Dataset synthesis with dataset: $DATASET and max_seq_length: $MAX_SEQ_LENGTH"

        # Execute the Python script with the specified parameters
        python qa.py \
            --save_dir=${SAVE_DIR} \
            --save_name=${SAVE_NAME} \
            --tokenizer_path=${TOKENIZER_PATH} \
            --tokenizer_type=${TOKENIZER_TYPE} \
            --max_seq_length=${MAX_SEQ_LENGTH} \
            --tokens_to_generate=${TOKENS_TO_GENERATE} \
            --num_samples=${NUM_SAMPLES} \
            --template="${TEMPLATE}" \
            --dataset=${DATASET}
    done
done