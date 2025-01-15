import json

def compare_inputs(file1_path, file2_path):
    """
    Compare the `input` values of each element in two JSONL files.

    Parameters:
        file1_path (str): Path to the first JSONL file.
        file2_path (str): Path to the second JSONL file.

    Returns:
        bool: True if all `input` values are identical, False otherwise.
        list: List of indices where the `input` values differ.
    """
    with open(file1_path, 'r') as f1, open(file2_path, 'r') as f2:
        data1 = [json.loads(line) for line in f1]
        data2 = [json.loads(line) for line in f2]

    # Check if the lengths of the files are the same
    if len(data1) != len(data2):
        raise ValueError("Files contain a different number of elements.")

    all_equal = True
    differing_indices = []

    for i, (item1, item2) in enumerate(zip(data1, data2)):
        if item1.get("input") != item2.get("input"):
            all_equal = False
            differing_indices.append(i)

    return all_equal, differing_indices

# Example usage
if __name__ == "__main__":
    file1 = "hotpotqa/validation-num_sample_100-max_seq_4096.jsonl"
    file2 = "hotpotqa/validation-num_sample_100-max_seq_8192.jsonl"

    are_equal, differences = compare_inputs(file1, file2)

    if are_equal:
        print("All `input` values are identical.")
    else:
        print(f"`input` values differ at the following indices: {differences}")
