#!/bin/bash
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
export LC_CTYPE=en_US.UTF-8

set -e

# === Config ===
BPE_MODEL_DIR="data/BPE_models"
BPE_SCRIPT="../OpenNMT-py/tools/apply_bpe.py"
DATA_DIR="data"
LOG_FILE="log/apply_bpe.log"
mkdir -p log

# === Domains and splits ===
DOMAINS=("wmt" "medical" "medical_combined")
SPLITS=("train" "dev" "test")

# === File extensions ===
declare -A EXT
EXT["src"]="src.tok"
EXT["tgt"]="tgt.tok"

TIMESTAMP=$(date "+%Y-%m-%d %H:%M")

for bpe_source in "${DOMAINS[@]}"; do
  echo "Using BPE model from: $bpe_source"

  bpe_src="${BPE_MODEL_DIR}/${bpe_source}.bpe.src"
  bpe_tgt="${BPE_MODEL_DIR}/${bpe_source}.bpe.tgt"

  for dataset in "${DOMAINS[@]}"; do
    echo "  Applying to dataset: $dataset"

    for split in "${SPLITS[@]}"; do
      src_file="${DATA_DIR}/${dataset}/${split}.${EXT[src]}"
      tgt_file="${DATA_DIR}/${dataset}/${split}.${EXT[tgt]}"
      out_src="${src_file}.bpe_${bpe_source}"
      out_tgt="${tgt_file}.bpe_${bpe_source}"

      if [[ -f "$src_file" ]]; then
        echo "    Applying BPE to source: $src_file → $out_src"
        python3 "$BPE_SCRIPT" -c "$bpe_src" < "$src_file" > "$out_src"
        echo "[$TIMESTAMP] $bpe_source BPE: $src_file → $out_src" >> "$LOG_FILE"
      else
        echo "    Skipping source (missing): $src_file"
        echo "[$TIMESTAMP] Skipped source: $src_file" >> "$LOG_FILE"
      fi

      if [[ -f "$tgt_file" ]]; then
        echo "    Applying BPE to target: $tgt_file → $out_tgt"
        python3 "$BPE_SCRIPT" -c "$bpe_tgt" < "$tgt_file" > "$out_tgt"
        echo "[$TIMESTAMP] $bpe_source BPE: $tgt_file → $out_tgt" >> "$LOG_FILE"
      else
        echo "    Skipping target (missing): $tgt_file"
        echo "[$TIMESTAMP] Skipped target: $tgt_file" >> "$LOG_FILE"
      fi
    done
  done
done

echo "All BPE applications complete and logged to $LOG_FILE"
