import argparse
import json
import random
import string
from typing import Optional

import uuid_test


def generate_letter_chain(num_identifiers: int, identifier_length: int, seed: Optional[int]):
    """Generate a list of random alphabetic identifiers."""
    rng = random.Random(seed)
    return [
        "".join(rng.choices(string.ascii_letters, k=identifier_length))
        for _ in range(num_identifiers)
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Preview the random letter chains used by the distractor script."
    )
    parser.add_argument("--num-identifiers", type=int, default=4, help="Length of each chain.")
    parser.add_argument("--identifier-length", type=int, default=8, help="Characters per identifier.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility.")
    parser.add_argument(
        "--preview-format",
        choices=["list", "chain"],
        default="chain",
        help="Show either the raw list or the formatted chain representation.",
    )
    parser.add_argument(
        "--end-with",
        type=str,
        default="question",
        help="Terminal value used when preview-format=chain.",
    )
    args = parser.parse_args()

    chain = generate_letter_chain(args.num_identifiers, args.identifier_length, args.seed)
    print("Raw identifiers:", chain)

    if args.preview_format == "chain":
        chain_strings = uuid_test.generate_uuid_string_from_chain(chain, end_with=args.end_with)
        print("Formatted chain:")
        print(json.dumps(chain_strings, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
