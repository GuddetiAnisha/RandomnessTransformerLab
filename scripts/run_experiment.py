from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from randomness_lab.experiment import ExperimentConfig, run_experiment
from randomness_lab.sources import GENERATORS, generate, load_bits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a compact Transformer on a binary sequence")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source", choices=sorted(GENERATORS))
    source.add_argument("--input")
    parser.add_argument("--format", choices=["raw", "ascii", "hex", "npy", "csv"], default="raw")
    parser.add_argument("--num-bits", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--target-bits", type=int, choices=range(1, 9), default=1)
    parser.add_argument("--d-model", type=int, default=48)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--output", default="reports/experiment")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bits = load_bits(args.input, args.format) if args.input else generate(args.source, args.num_bits, args.seed)
    config = ExperimentConfig(seed=args.seed, context_length=args.context_length,
                              target_bits=args.target_bits, d_model=args.d_model,
                              nhead=args.nhead, num_layers=args.num_layers,
                              epochs=args.epochs, batch_size=args.batch_size, stride=args.stride)
    result = run_experiment(bits, config, args.output)
    print(json.dumps({"source": args.source or args.input, "test": result["test"],
                      "baseline": result["baseline"],
                      "failed_tests": result["sequence_analysis"]["failed_tests"]}, indent=2))


if __name__ == "__main__":
    main()
