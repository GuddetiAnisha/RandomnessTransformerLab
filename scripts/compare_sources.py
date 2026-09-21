from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from randomness_lab.sources import GENERATORS, generate
from randomness_lab.statistics import analyze


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare non-ML indicators across generated sources")
    parser.add_argument("--num-bits", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="reports/source_comparison.csv")
    args = parser.parse_args()
    rows = []
    for source in GENERATORS:
        bits = generate(source, args.num_bits, args.seed)
        result = analyze(bits)
        rows.append({"source": source, "ones": result["proportion_ones"],
                     "shannon_entropy": result["shannon_entropy_per_bit"],
                     "mcv_min_entropy": result["mcv_min_entropy_per_bit"],
                     "markov_min_entropy": result["markov_min_entropy_per_bit"],
                     "compression_ratio": result["compression_ratio"],
                     "longest_run": result["longest_run"],
                     "failed_tests": len(result["failed_tests"])})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows).sort_values(["failed_tests", "markov_min_entropy"], ascending=[True, False])
    frame.to_csv(output, index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
