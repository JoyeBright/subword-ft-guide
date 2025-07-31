#!/bin/bash
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
export LC_CTYPE=en_US.UTF-8

set -e  # Exit on error

# === Configuration ===
TOKENIZER="../OpenNMT-py/tools/tokenizer.perl"
THREADS=32
SRC_LANG="en"
TGT_LANG="de"
LOG_FILE="log/tokenization.log"
DATE=$(date '+%Y-%m-%d %H:%M:%S')


echo "📝 Starting tokenization at $DATE" > "$LOG_FILE"

# === Temporary Python script for CSV extraction + length check ===
EXTRACT_SCRIPT="extract_columns.py"

cat <<EOF > "$EXTRACT_SCRIPT"
import pandas as pd
import sys

csv_path, src_out, tgt_out, name = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
df = pd.read_csv(csv_path)
src_series = df["source"]
tgt_series = df["target"]

if len(src_series) != len(tgt_series):
    print(f"❌ ERROR: {name} - source and target lengths do not match!")
    print(f"    source: {len(src_series)} rows, target: {len(tgt_series)} rows")
    exit(1)

src_series.to_csv(src_out, index=False, header=False)
tgt_series.to_csv(tgt_out, index=False, header=False)
print(f"✅ Extracted {name}: {len(src_series)} sentence pairs")
EOF

# === Extract and tokenize CSV-based dataset ===
function extract_and_tokenize_csv() {
    dataset="$1"  # wmt or medical
    split="$2"    # train, dev, test

    CSV="data/$dataset/$split.csv"
    SRC="data/$dataset/$split.src"
    TGT="data/$dataset/$split.tgt"
    SRC_TOK="$SRC.tok"
    TGT_TOK="$TGT.tok"

    echo "📥 Extracting $CSV..." | tee -a "$LOG_FILE"
    python "$EXTRACT_SCRIPT" "$CSV" "$SRC" "$TGT" "$dataset/$split" >> "$LOG_FILE"

    echo "🪄 Tokenizing $dataset $split (src → $SRC_TOK)" | tee -a "$LOG_FILE"
    perl "$TOKENIZER" -l $SRC_LANG -no-escape -threads $THREADS < "$SRC" > "$SRC_TOK"
    wc -l "$SRC_TOK" >> "$LOG_FILE"

    echo "🪄 Tokenizing $dataset $split (tgt → $TGT_TOK)" | tee -a "$LOG_FILE"
    perl "$TOKENIZER" -l $TGT_LANG -no-escape -threads $THREADS < "$TGT" > "$TGT_TOK"
    wc -l "$TGT_TOK" >> "$LOG_FILE"
}

# === Tokenize pre-existing .src and .tgt files ===
function tokenize_raw_file_pair() {
    dataset="medical_combined"
    split="$1"

    SRC="data/$dataset/$split.src"
    TGT="data/$dataset/$split.tgt"
    SRC_TOK="$SRC.tok"
    TGT_TOK="$TGT.tok"

    echo "🪄 Tokenizing $dataset $split (src → $SRC_TOK)" | tee -a "$LOG_FILE"
    perl "$TOKENIZER" -l $SRC_LANG -no-escape -threads $THREADS < "$SRC" > "$SRC_TOK"
    wc -l "$SRC_TOK" >> "$LOG_FILE"

    echo "🪄 Tokenizing $dataset $split (tgt → $TGT_TOK)" | tee -a "$LOG_FILE"
    perl "$TOKENIZER" -l $TGT_LANG -no-escape -threads $THREADS < "$TGT" > "$TGT_TOK"
    wc -l "$TGT_TOK" >> "$LOG_FILE"
}

# === Run all ===
for dataset in wmt medical; do
    for split in train dev test; do
        extract_and_tokenize_csv "$dataset" "$split"
    done
done

for split in train dev test; do
    tokenize_raw_file_pair "$split"
done

# === Cleanup and Finish ===
rm "$EXTRACT_SCRIPT"
echo "✅ All tokenization complete. Log saved to $LOG_FILE"
