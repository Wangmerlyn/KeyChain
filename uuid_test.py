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



chain = generate_uuid_chain(num_uuids=4)
print("Generated UUID Chain:")
print(chain)
print("String representation of UUID chain:")
print(generate_uuid_string_from_chain(chain, end_with="Given question"))