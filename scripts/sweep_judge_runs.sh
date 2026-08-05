#!/usr/bin/env bash
# Walk <source-dir> and run scripts/judge_runs.py on every run group found.
#
# Two modes:
#   1. Plain sweep: omit --ground-truths. <source-dir> is treated as the
#      parent of run groups and every leaf group under it is judged.
#   2. Per-ground-truth sweep: pass --ground-truths gt1,gt2,gt3. For each
#      <gt>, the script sweeps <source-dir>/<gt>/ with --ground-truth <gt>
#      automatically appended. Missing subtrees are logged and skipped.
#
# A "run group" is a directory whose immediate child directories all contain
# a rounds/ folder (i.e. they're all autopi runs). Mixed dirs (some runs,
# some containers) are skipped — sweeping them would conflate runs at this
# level with runs deeper in the tree. All other args are forwarded verbatim
# to judge_runs.py.
#
# Usage:
#   bash scripts/sweep_judge_runs.sh <source-dir> --domain <name> \
#       [--ground-truths gt1,gt2,gt3] [--skip-existing] [--dry-run] \
#       [--model <m>] [--provider <p>]
#
# Sweeper-only flags (consumed before forwarding):
#   --dry-run             Print the commands that would be run.
#   --ground-truths LIST  Comma-separated GTs; switches to per-GT mode.

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <source-dir> --domain <name> [--ground-truths gt1,gt2,gt3] [judge_runs.py args...]" >&2
  exit 2
fi

source_dir="$1"; shift

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
judge_script="$script_dir/judge_runs.py"

dry_run=0
ground_truths=""
forwarded=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      dry_run=1; shift ;;
    --ground-truths)
      ground_truths="$2"; shift 2 ;;
    --ground-truths=*)
      ground_truths="${1#--ground-truths=}"; shift ;;
    *)
      forwarded+=("$1"); shift ;;
  esac
done

if [[ ! -d "$source_dir" ]]; then
  echo "not a directory: $source_dir" >&2
  exit 2
fi

# Discover leaf run groups under $1 and dispatch judge_runs.py on each.
# Remaining args are passed verbatim to judge_runs.py. Returns 0 if all
# groups succeed, non-zero otherwise.
sweep_dir() {
  local dir="$1"; shift
  local extra=("$@")

  local groups=()
  local d
  while IFS= read -r d; do
    local any_run=0 all_runs=1
    local child
    for child in "$d"/*/; do
      [[ -d "$child" ]] || continue
      if [[ -d "${child}rounds" ]]; then
        any_run=1
      else
        all_runs=0
        break
      fi
    done
    [[ $any_run -eq 1 && $all_runs -eq 1 ]] && groups+=("$d")
  done < <(find "$dir" -type d | sort)

  if [[ ${#groups[@]} -eq 0 ]]; then
    echo "no run groups found under $dir" >&2
    return 1
  fi

  echo "found ${#groups[@]} run group(s) under $dir"
  local g
  for g in "${groups[@]}"; do
    echo "  - $g"
  done

  local failures=0 group cmd
  for group in "${groups[@]}"; do
    echo
    echo "=== $group ==="
    cmd=(python "$judge_script" --runs-dir "$group" ${extra[@]+"${extra[@]}"})
    if [[ $dry_run -eq 1 ]]; then
      echo "DRY RUN: ${cmd[*]}"
      continue
    fi
    if "${cmd[@]}"; then
      echo "OK"
    else
      echo "FAILED" >&2
      failures=$((failures + 1))
    fi
  done

  [[ $failures -eq 0 ]]
}

if [[ -z "$ground_truths" ]]; then
  # Plain mode.
  sweep_dir "$source_dir" ${forwarded[@]+"${forwarded[@]}"}
  exit $?
fi

# Per-GT mode.
IFS=',' read -ra gts <<< "$ground_truths"
total_failures=0
for gt in "${gts[@]}"; do
  subtree="$source_dir/$gt"
  if [[ ! -d "$subtree" ]]; then
    echo "skipping ground-truth=$gt: subtree not found at $subtree" >&2
    continue
  fi
  echo
  echo "######## ground-truth=$gt @ $subtree ########"
  if ! sweep_dir "$subtree" --ground-truth "$gt" ${forwarded[@]+"${forwarded[@]}"}; then
    echo "FAILED ground-truth=$gt" >&2
    total_failures=$((total_failures + 1))
  fi
done

if [[ $total_failures -gt 0 ]]; then
  echo "$total_failures ground-truth sweep(s) failed" >&2
  exit 1
fi
