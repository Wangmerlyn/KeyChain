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



print(generate_uuid_chain())
print("tree")
print(generate_uuid_tree(num_levels=4, num_children=2))