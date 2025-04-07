import random
import numpy as np

RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def generate_uuid(format="default"):
    """
    Generate a UUID string.
    """
    if format == "default":
        import uuid
        uuid = uuid.UUID(int=random.getrandbits(128), version=4)
        return str(uuid)
    elif format.startswith("nano"):
        num_digits=int(format.split("_")[-1]) if "_" in format else 21
        # Generate a UUID and return a nano-style representation
        # TODO this is not reproducible, since the random bits generator is os.urandom
        import nanoid
        return nanoid.generate(size=num_digits)
    else:
        raise ValueError(
            f"Unsupported format '{format}'. Use 'default' for standard UUID or 'nano' for nano-style."
        )

def generate_uuid_chain(num_uuids=4, format="default"):
    """
    Generate a chain of UUIDs.
    """
    uuids = []
    for _ in range(num_uuids):
        uuids.append(generate_uuid(format=format))
    return uuids

def generate_uuid_tree(num_levels=4, num_children=2, format="default"):
    """
    Generate a tree of UUIDs.
    """
    if num_levels < 1:
        return []

    def generate_tree(level, max_level):
        if level > max_level:
            return []
        uuids = [generate_uuid(format=format)]
        for _ in range(num_children):
            uuids.extend(generate_tree(level + 1, max_level))
        return uuids

    return generate_tree(1, num_levels)

def generate_uuid_string_from_chain(uuids, end_with="uuid"):
    """
    Generate a string representation of a chain of UUIDs.
    input: ['uuid1', 'uuid2', 'uuid3', 'uuid4']
    output: [{'uuid1'->'uuid2'}, {'uuid2'->'uuid3'}, {'uuid3'->'uuid4'}]
    """
    result_list = []
    for i in range(len(uuids) - 1):
        # Generate a simple string representation of the chain
        result_list.append(f'{{"{uuids[i]}": "{uuids[i + 1]}"}}')
    if end_with is None or end_with == "None":
        pass
    elif end_with == "uuid":
        # Add the last UUID in the chain and target it to a new UUID
        result = f'{{"{uuids[-1]}": "{generate_uuid()}"}}'
        result_list.append(result)
    elif isinstance(end_with, str):
        """
        Allow custom end with string, e.g. 'end' to indicate the end of the chain.
        """
        result = f'{{"{uuids[-1]}": "{end_with}"}}'
        result_list.append(result)
    return result_list


def generate_uuid_string_from_tree(uuid_tree, question="question"):
    """
    Generate a string representation of a tree of UUIDs from a complete binary tree.
    The tree is represented as a list where the node at index i has children at
    indices 2i + 1 and 2i + 2.

    Args:
    - uuid_tree (list): A list representing the complete binary tree of UUIDs.
    - question (str): The value to be assigned to one of the leaf nodes' successor.

    Returns:
    - list of strings: A list of strings representing parent-child relationships.
    """
    result_list = []
    
    # Helper function to generate parent-child relationships recursively
    def generate_parent_child_relation(index):
        if index >= len(uuid_tree):
            return
        # Get the left and right children indices
        left_index = 2 * index + 1
        right_index = 2 * index + 2

        # If the left child exists, add the relationship
        if left_index < len(uuid_tree):
            result_list.append(f'{{"{uuid_tree[index]}": "{uuid_tree[left_index]}"}}')

        # If the right child exists, add the relationship
        if right_index < len(uuid_tree):
            result_list.append(f'{{"{uuid_tree[index]}": "{uuid_tree[right_index]}"}}')

        # Recursively call for left and right children
        generate_parent_child_relation(left_index)
        generate_parent_child_relation(right_index)

    # Start the recursion from the root node (index 0)
    generate_parent_child_relation(0)

    # Identify the leaf nodes in the tree (they start from index len(uuid_tree)//2)
    leaf_nodes_start_index = len(uuid_tree) // 2
    leaf_nodes = uuid_tree[leaf_nodes_start_index:]

    # Randomly choose one leaf node to have the `question` as the successor
    chosen_leaf = random.choice(leaf_nodes)
    
    # Generate the relationships for the leaf nodes
    for leaf in leaf_nodes:
        if leaf == chosen_leaf:
            result_list.append(f'{{"{leaf}": "{question}"}}')
        else:
            result_list.append(f'{{"{leaf}": "{generate_uuid()}"}}')

    return result_list



chain = generate_uuid_chain(num_uuids=4)
print("Generated UUID Chain:")
print(chain)
print("String representation of UUID chain:")
print(generate_uuid_string_from_chain(chain, end_with="Given question"))

tree = generate_uuid_tree(num_levels=4, num_children=2)
print("\nGenerated UUID Tree:")
print(tree)
print("String representation of UUID tree:")
print(generate_uuid_string_from_tree(tree, question="Given question"))
print(len(tree))  # Check the number of UUIDs in the tree
print(f"Total uuid relationships in the tree: {len(generate_uuid_string_from_tree(tree, question='Given question'))}")