#!/bin/bash
set -e

# === Config ===
BPE_SCRIPT="../OpenNMT-py/tools/learn_bpe.py"
WITH_VOCAB=0
LOG_DIR="log"
LOG_FILE="$LOG_DIR/learn_bpe.log"
DATA_DIR="data"
OUT_DIR="${DATA_DIR}/BPE_models"

mkdir -p "$OUT_DIR" "$LOG_DIR"
echo "=== BPE Learning Log $(date) ===" > "$LOG_FILE"
echo "# Inspired by Ding et al. (2019) and Adlaon & Marcos (2024)" >> "$LOG_FILE"
echo "# Ding et al.: smaller vocab favored for Transformers; large vocab can reduce BLEU by ~3–4 pts ($$<arxiv.org/abs/1905.10453$$)" >> "$LOG_FILE"
echo "# Adlaon & Marcos: 2–8k merges perform best for low-resource datasets ($$aclanthology.org/2024.findings-emnlp.860$$)" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

DOMAINS=("wmt" "medical" "medical_combined")
SIDES=("src" "tgt")

declare -A EXT=( ["src"]="src.tok" ["tgt"]="tgt.tok" )

VOCAB_FLAG=""
[ "$WITH_VOCAB" -eq 1 ] && VOCAB_FLAG="-v"

for domain in "${DOMAINS[@]}"; do
  for side in "${SIDES[@]}"; do
    input="${DATA_DIR}/${domain}/train.${EXT[$side]}"
    output="${OUT_DIR}/${domain}.bpe.${side}"
    N=$(wc -l < "$input")

    if [ "$N" -lt 100000 ]; then
      MERGE_OPS=8000
    elif [ "$N" -lt 1000000 ]; then
      MERGE_OPS=30000
    else
      MERGE_OPS=50000
    fi

    {
      echo "## Domain: $domain | Side: $side"
      echo "Dataset size (lines): $N"
      echo "Merge operations chosen: $MERGE_OPS"
      echo "Input file: $input"
      echo "Output file: $output"
      echo "-------------------------------"
      echo "🧠 Learning BPE: $input → $output (merges=$MERGE_OPS)"
    } | tee -a "$LOG_FILE"

    python "$BPE_SCRIPT" -s "$MERGE_OPS" -i "$input" -o "$output" $VOCAB_FLAG \
      2>&1 | tee -a "$LOG_FILE"

    echo "✅ Completed $domain $side with $MERGE_OPS merges" | tee -a "$LOG_FILE"
    echo "" >> "$LOG_FILE"
  done
done

echo "✅ BPE models trained. Log: $LOG_FILE"
