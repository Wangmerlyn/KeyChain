dataset_list=("hotpotqa" "musique" "2wikimqa")

# Loop through each dataset in the array
for dataset in "${dataset_list[@]}"; do
    # Create the directory if it doesn't exist
    # cp the file to current directory
    # curr_folder=$(pwd)
    # cp -r /mnt/longcontext/models/siyuan/test_code/longcontext_syth/${dataset} ${curr_folder}/
    # ==================================
    # copy the result file back to remote directory
    cp -rn ${dataset} /mnt/longcontext/models/siyuan/test_code/longcontext_syth/
done