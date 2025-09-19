#!/bin/bash
# Toggle variables to control whether to download specific datasets
DOWNLOAD_HOTPOTQA=true
DOWNLOAD_MUSIQUE=true
DOWNLOAD_2WIKIMQA=true

# Dataset and save name selection
DATASETS=("hotpotqa" "musique" "2wikimqa")  # Options: "hotpotqa", "musique", "2wikimqa"
# DATASETS=("hotpotqa")
MAX_SEQ_LENGTHS=(4096 8192 16384 32768 65536 131072)
# MAX_SEQ_LENGTHS=(4096)

# Dataset download section
if [ "$DOWNLOAD_HOTPOTQA" = true ]; then
    echo "Downloading HotpotQA dataset..."
    wget -q http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json
    mv hotpot_train_v1.1.json hotpotqa/hotpot_train_v1.1.json
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
    mkdir musique
    mv data/musique_full_v1.0_train.jsonl musique/musique_full_v1.0_train.jsonl  # Move the dataset to the specified folder
fi

if [ "$DOWNLOAD_2WIKIMQA" = true ]; then
    echo "Downloading 2WikiMQA dataset..."
    wget -q -O 2wikimqa.zip "https://www.dropbox.com/scl/fi/32t7pv1dyf3o2pp0dl25u/data_ids_april7.zip?rlkey=u868q6h0jojw4djjg7ea65j46&e=1"
    unzip -q 2wikimqa.zip -d 2wikimqa  # Extract the dataset to the specified folder
    rm 2wikimqa.zip                   # Remove the zip file after extraction
fi

# Set parameters
SAVE_DIR="./"
TOKENIZER_PATH="llama-3.1-8B-instruct"
TOKENIZER_TYPE="hf"
TOKENS_TO_GENERATE=128
NUM_SAMPLES=-1
TEMPLATE="{context}"

# Function to process each combination of dataset and max_seq_length
process_combination() {
    DATASET_CHOICE=$1
    MAX_SEQ_LENGTH=$2

    case "$DATASET_CHOICE" in
        "hotpotqa")
            SAVE_NAME="hotpotqa"
            DATASET="hotpotqa/hotpot_train_v1.1.json"
            ;;
        "musique")
            SAVE_NAME="musique"
            DATASET="musique/musique_full_v1.0_train.jsonl"
            ;;
        "2wikimqa")
            SAVE_NAME="2wikimqa"
            DATASET="2wikimqa/train.json"
            ;;
        *)
            echo "Invalid DATASET_CHOICE. Skipping..."
            return
            ;;
    esac

    echo "Running Dataset synthesis with dataset: $DATASET and max_seq_length: $MAX_SEQ_LENGTH"

    # Execute the Python script with the specified parameters
    python qa_qwen_filtered_add_distractor.py \
        --save_dir=${SAVE_DIR} \
        --save_name=${SAVE_NAME} \
        --tokenizer_path=${TOKENIZER_PATH} \
        --tokenizer_type=${TOKENIZER_TYPE} \
        --max_seq_length=${MAX_SEQ_LENGTH} \
        --tokens_to_generate=${TOKENS_TO_GENERATE} \
        --num_samples=${NUM_SAMPLES} \
        --template="${TEMPLATE}" \
        --dataset=${DATASET} \
        --subset="qwen_filtered"
}

# Launch each combination in parallel
for DATASET_CHOICE in "${DATASETS[@]}"; do
    for MAX_SEQ_LENGTH in "${MAX_SEQ_LENGTHS[@]}"; do
        process_combination "$DATASET_CHOICE" "$MAX_SEQ_LENGTH" &
        # exit 0  
    done
done

# Wait for all parallel processes to finish
wait

echo "All combinations processed."
