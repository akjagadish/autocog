#!/usr/bin/env bash
# Concatenate every judge_results.csv under <domain-dir> into
# <domain-dir>/all_judge_results.csv. Header taken from the first file;
# subsequent headers stripped. The output file itself is excluded so the
# script is safe to re-run.

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <domain-dir>" >&2
  exit 2
fi

domain_dir="$1"
if [[ ! -d "$domain_dir" ]]; then
  echo "not a directory: $domain_dir" >&2
  exit 2
fi

out="$domain_dir/all_judge_results.csv"

# Find every judge_results.csv except the aggregated output.
# Use a while-read loop instead of `mapfile` for compatibility with the
# bash 3.2 that ships on macOS.
inputs=()
while IFS= read -r line; do
  inputs+=("$line")
done < <(find "$domain_dir" -type f -name "judge_results.csv" | sort)

if [[ ${#inputs[@]} -eq 0 ]]; then
  echo "no judge_results.csv files found under $domain_dir" >&2
  exit 1
fi

# Header from the first file.
head -n 1 "${inputs[0]}" > "$out"

# Body rows from every input.
for f in "${inputs[@]}"; do
  tail -n +2 "$f" >> "$out"
done

echo "wrote $out  (${#inputs[@]} input files)"
