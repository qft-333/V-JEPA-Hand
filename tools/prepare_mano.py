"""Convert a trusted legacy MANO pickle to NumPy arrays in a Python 3.10 environment."""

import argparse
import os
import pickle
import sys
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Licensed original MANO_RIGHT.pkl")
    parser.add_argument("--output", type=Path, required=True, help="New NumPy-only MANO_RIGHT.pkl")
    parser.add_argument("--trust-model", action="store_true", help="Allow unpickling your trusted MANO asset")
    args = parser.parse_args()
    if not args.trust_model:
        parser.error("Pickle can execute code. Use --trust-model only for your own official MANO asset.")
    if sys.version_info[:2] != (3, 10):
        parser.error("Legacy Chumpy conversion requires Python 3.10; see README.md.")
    if args.output.name != "MANO_RIGHT.pkl":
        parser.error("Output filename must be MANO_RIGHT.pkl")
    if args.output.exists():
        parser.error("Output already exists; choose a new path")
    import chumpy
    import numpy as np

    def convert(value):
        if isinstance(value, chumpy.Ch):
            return np.asarray(value.r)
        if isinstance(value, dict):
            return {key: convert(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(convert(item) for item in value)
        return value

    with args.input.open("rb") as source:
        model = convert(pickle.load(source, encoding="latin1"))
    required = {"v_template", "shapedirs", "posedirs", "J_regressor", "weights", "f"}
    if not isinstance(model, dict) or not required.issubset(model):
        raise ValueError("Input is not a MANO model dictionary")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=args.output.parent, prefix=".mano-")
    try:
        with os.fdopen(descriptor, "wb") as output:
            pickle.dump(model, output, protocol=4)
        os.link(temporary, args.output)
    finally:
        Path(temporary).unlink(missing_ok=True)
    print(f"Prepared {args.output}. Keep this licensed asset private.")


if __name__ == "__main__":
    main()
