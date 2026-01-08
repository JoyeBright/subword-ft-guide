#!/bin/bash
set -e

# === Root Directory ===
ROOT="/home/pourmost/Redo/RANLP"

# === Paths ===
VOCAB_DIR="$ROOT/Vocabs"
BPE_DATA_DIR="$ROOT/data"
CONFIG_DIR="$ROOT/vocab_configs"
WRITTEN="$ROOT/written"
LOG_FILE="$ROOT/log/build_vocab.log"
mkdir -p "$VOCAB_DIR" "$CONFIG_DIR" "$WRITTEN" "$(dirname $LOG_FILE)"

# === Timestamp ===
TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")

# === Function to create and run vocab config ===
build_vocab () {
  bpe_model=$1
  data_source=$2
  config_name="vocab_from-${bpe_model}-on-${data_source}"
  src_file="${BPE_DATA_DIR}/${data_source}/train.src.tok.bpe_${bpe_model}"
  tgt_file="${BPE_DATA_DIR}/${data_source}/train.tgt.tok.bpe_${bpe_model}"
  config_path="${CONFIG_DIR}/${config_name}.yaml"

  echo "Creating vocab config: $config_name"
  cat > "$config_path" <<EOF
save_data: "${WRITTEN}/${config_name}"

src_vocab: "${VOCAB_DIR}/${config_name}.en.src"
tgt_vocab: "${VOCAB_DIR}/${config_name}.de.tgt"
share_vocab: false
overwrite: true

data:
    corpus_0:
        path_src: "${src_file}"
        path_tgt: "${tgt_file}"

src_seq_length: 150
tgt_seq_length: 150
skip_empty_level: silent
n_sample: -1
EOF

  echo "Building vocab: $config_name"
  {
    echo "[$TIMESTAMP] Building vocab: $config_name"
    if onmt_build_vocab -config "$config_path"; then
      echo "Finished: $config_name"
    else
      echo "Failed: $config_name"
    fi
    echo ""
  } >> "$LOG_FILE" 2>&1
}


# === Build all 7 vocab configs ===
build_vocab "wmt" "wmt"
build_vocab "wmt" "medical_combined"
build_vocab "wmt" "medical"
build_vocab "medical" "wmt"
build_vocab "medical" "medical_combined"
build_vocab "medical" "medical"
build_vocab "medical_combined" "medical_combined"

echo "All vocabulary configs created in: $CONFIG_DIR"
echo "Vocabularies saved to: $VOCAB_DIR"
echo "Log written to: $LOG_FILE"
