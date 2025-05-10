start_idx=0
end_idx=2500
dataset_name="2wikimqa"
note="dist_run"
dataset_path_prefix="~/filter_question/data"

sas="sp=racwdl&st=2025-05-10T18:26:16Z&se=2025-05-17T02:26:16Z&skoid=7b3a9ac3-4eaa-434a-8801-b2b90159bf0b&sktid=72f988bf-86f1-41af-91ab-2d7cd011db47&skt=2025-05-10T18:26:16Z&ske=2025-05-17T02:26:16Z&sks=b&skv=2024-11-04&spr=https&sv=2024-11-04&sr=c&sig=IvptF6%2FBlKQz1ng9jSwtmhYqdhQWZLtasnAfn9iOzmk%3D"


cd /scratch
wget https://azcopyvnext-awgzd8g7aagqhzhe.b02.azurefd.net/releases/release-10.27.1-20241113/azcopy_linux_amd64_10.27.1.tar.gz
tar -xzf azcopy_linux_amd64_10.27.1.tar.gz
echo 'export PATH=/scratch/azcopy_linux_amd64_10.27.1:$PATH' >> ~/.bashrc
source ~/.bashrc

azcopy copy --recursive "https://sanbpx4p3idss6q.blob.core.windows.net/longcontext/models/siyuan/test_code/longcontext_syth/filter_question?${sas}" ~


# curl -Lk 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64' --output vscode_cli.tar.gz
 
# tar -xzf vscode_cli.tar.gz
# ./code tunnel

source /opt/conda/etc/profile.d/conda.sh
conda create --name filter python==3.10 -y

conda activate filter
pip install vllm -U

cd filter_question
python filter_infer.py \
    --dataset_name ${dataset_name} \
    --start_idx ${start_idx} \
    --end_idx ${end_idx} \
    --dataset_path_prefix ${dataset_path_prefix} \
    --note ${note} \

output_file_name=${dataset_name}_train_merged_pred_${start_idx}_${end_idx}_${note}.jsonl
azcopy copy data/$output_file_name "https://sanbpx4p3idss6q.blob.core.windows.net/longcontext/models/siyuan/test_code/longcontext_syth/filter_question/data/${output_file_name}?${sas}" 