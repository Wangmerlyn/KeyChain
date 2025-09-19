#!/bin/bash

# dataset_name="2wikimqa"
note="dist_run"
dataset_path_prefix="$HOME/filter_question/data"
install_env=true
tp_size=8
# Total number of nodes and the current node ID (0-indexed)
num_total_nodes=10
curr_node_id=0

# Total lengths of each dataset
hotpotqa_len=90447
musique_len=19938
wikimqa_len=167454  # This corresponds to '2wikimqa'
model_path="Qwen/Qwen2.5-7B-Instruct"

# === Function to compute start_idx and end_idx for a dataset ===
compute_range() {
    local total_len=$1
    local chunk_size=$(( (total_len + num_total_nodes - 1) / num_total_nodes ))  # ceil division
    local start=$(( curr_node_id * chunk_size ))
    local end=$(( start + chunk_size ))
    if [ "$end" -gt "$total_len" ]; then
        end=$total_len
    fi
    echo "$start $end"
}

# Compute start/end indices for each dataset
read hotpotqa_start_idx hotpotqa_end_idx < <(compute_range $hotpotqa_len)
read musique_start_idx musique_end_idx < <(compute_range $musique_len)
read wikimqa_start_idx wikimqa_end_idx < <(compute_range $wikimqa_len)

# === Print results ===
echo "[hotpotqa] Node $curr_node_id: $hotpotqa_start_idx to $hotpotqa_end_idx"
echo "[musique ] Node $curr_node_id: $musique_start_idx to $musique_end_idx"
echo "[2wikimqa] Node $curr_node_id: $wikimqa_start_idx to $wikimqa_end_idx"



wget https://azcopyvnext-awgzd8g7aagqhzhe.b02.azurefd.net/releases/release-10.27.1-20241113/azcopy_linux_amd64_10.27.1.tar.gz
tar -xzf azcopy_linux_amd64_10.27.1.tar.gz
export PATH=azcopy_linux_amd64_10.27.1:$PATH
echo 'export PATH=azcopy_linux_amd64_10.27.1:$PATH' >> ~/.bashrc
source ~/.bashrc

azcopy copy --recursive $blob_url ~


# curl -Lk 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64' --output vscode_cli.tar.gz
 
# tar -xzf vscode_cli.tar.gz
# ./code tunnel

source /opt/conda/etc/profile.d/conda.sh

if [ "$install_env" = true ]; then
    conda create --name filter python==3.10 -y
    conda activate filter
    pip install vllm -U
else 
    conda activate filter
fi


cd ~/filter_question
echo "========================================"
echo "Current directory: $(pwd)"
echo "processing dataset: hotpotqa"
echo "start_idx: $hotpotqa_start_idx"
echo "end_idx: $hotpotqa_end_idx"
echo "dataset_path_prefix: $dataset_path_prefix"
echo "note: $note"
echo "======================================="
python filter_infer.py \
    --dataset_name hotpotqa \
    --start_idx ${hotpotqa_start_idx} \
    --end_idx ${hotpotqa_end_idx} \
    --dataset_path_prefix ${dataset_path_prefix} \
    --note ${note} \
    --tp_size ${tp_size} \
    --model_path ${model_path}

output_file_name=hotpotqa_train_merged_pred_${hotpotqa_start_idx}_${hotpotqa_end_idx}_${note}.jsonl
echo "output_file_name: $output_file_name"
azcopy copy data/$output_file_name $upload_url

echo "hotpotqa done"

echo "========================================"
echo "processing dataset: musique"
echo "start_idx: $musique_start_idx"
echo "end_idx: $musique_end_idx"
echo "dataset_path_prefix: $dataset_path_prefix"
echo "note: $note"
echo "======================================="
python filter_infer.py \
    --dataset_name musique \
    --start_idx ${musique_start_idx} \
    --end_idx ${musique_end_idx} \
    --dataset_path_prefix ${dataset_path_prefix} \
    --note ${note} \
    --tp_size ${tp_size} \
    --model_path ${model_path}

output_file_name=musique_train_merged_pred_${musique_start_idx}_${musique_end_idx}_${note}.jsonl
echo "output_file_name: $output_file_name"
azcopy copy data/$output_file_name $upload_url
echo "musique done"
echo "========================================"
echo "processing dataset: 2wikimqa"
echo "start_idx: $wikimqa_start_idx"
echo "end_idx: $wikimqa_end_idx"
echo "dataset_path_prefix: $dataset_path_prefix"
echo "note: $note"
echo "======================================="

python filter_infer.py \
    --dataset_name 2wikimqa \
    --start_idx ${wikimqa_start_idx} \
    --end_idx ${wikimqa_end_idx} \
    --dataset_path_prefix ${dataset_path_prefix} \
    --note ${note} \
    --tp_size ${tp_size} \
    --model_path ${model_path}

output_file_name=2wikimqa_train_merged_pred_${wikimqa_start_idx}_${wikimqa_end_idx}_${note}.jsonl
echo "output_file_name: $output_file_name"
azcopy copy data/$output_file_name $upload_url
echo "2wikimqa done"
echo "========================================"
echo "All datasets processed successfully!"
echo "========================================"
