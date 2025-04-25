#!/usr/bin/env python3
# strip_hybrid_from_jsonl.py
#
# This script reads an existing JSONL file produced by arima_lstm_residual_no_vwap.py,
# removes the "hybrid" field from each line, and saves the cleaned lines
# to a new file called "base_arima.jsonl".
#
# Example usage:
#   python3 strip_hybrid_from_jsonl.py --input results/arima_lstm_novwap.jsonl --output results/base_arima.jsonl

import argparse
import json
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser("Strip 'hybrid' field from JSONL file")
    p.add_argument("--input",  required=True, help="Input JSONL file (with hybrid)")
    p.add_argument("--output", required=True, help="Output JSONL file (without hybrid)")
    return p.parse_args()

def main():
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    with input_path.open("r") as fin, output_path.open("w") as fout:
        for line in fin:
            record = json.loads(line)
            if "hybrid" in record:
                del record["hybrid"]
            fout.write(json.dumps(record) + "\n")

    print(f"Done. Wrote cleaned data to {output_path.resolve()}.")

if __name__ == "__main__":
    main()
