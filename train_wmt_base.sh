#!/bin/bash
set -e

# === Root paths ===
RANLP="/home/pourmost/Redo/RANLP"
DATA="${RANLP}/data/wmt"
VOCABS="${RANLP}/Vocabs"
MODELS="${RANLP}/Models"
LOG_DIR="${RANLP}/log/training_logs"
TB="${RANLP}/Tensorboard"
CO2="${RANLP}/CO2"
CONFIGS="${RANLP}/train_configs"

# === Model name and paths ===
MODEL_NAME="wmt_base"
YAML="${CONFIGS}/${MODEL_NAME}.yaml"
LOG_FILE="${LOG_DIR}/${MODEL_NAME}.log"
TB_PATH="${TB}/${MODEL_NAME}"
CO2_PATH="${CO2}/${MODEL_NAME}"
MODEL_PATH="${MODELS}/${MODEL_NAME}/model"

mkdir -p "$CONFIGS" "$LOG_DIR" "$TB_PATH" "$CO2_PATH" "$(dirname "$MODEL_PATH")"

SRC_VOCAB_SIZE=$(wc -l < "${VOCABS}/vocab_from-wmt-on-wmt.en.src")
TGT_VOCAB_SIZE=$(wc -l < "${VOCABS}/vocab_from-wmt-on-wmt.de.tgt")



# === Generate YAML config ===
cat > "$YAML" <<EOF
# === Data configuration ===
data:
  corpus_0:
    path_src: "${DATA}/train.src.tok.bpe_wmt"
    path_tgt: "${DATA}/train.tgt.tok.bpe_wmt"
    weight: 1
  valid:
    path_src: "${DATA}/dev.src.tok.bpe_wmt"
    path_tgt: "${DATA}/dev.tgt.tok.bpe_wmt"

# === Vocabulary paths ===
src_vocab: "${VOCABS}/vocab_from-wmt-on-wmt.en.src"
tgt_vocab: "${VOCABS}/vocab_from-wmt-on-wmt.de.tgt"

src_vocab_size: ${SRC_VOCAB_SIZE}
tgt_vocab_size: ${TGT_VOCAB_SIZE}

# === Model output ===
save_model: "${MODEL_PATH}"
save_checkpoint_steps: 2000
keep_checkpoint: 3

# === Logging ===
log_file: "${LOG_FILE}"
log_file_level: "INFO"
tensorboard: true
tensorboard_log_dir: "${TB_PATH}"

# === Training steps ===
train_steps: 200000
valid_steps: 1000
report_every: 100
early_stopping: 3
early_stopping_criteria: [accuracy]
reset_optim: "states"

# === Hardware setup ===
world_size: 1
gpu_ranks: [0]
num_workers: 10

# === Batching ===
batch_type: "tokens"
batch_size: 10240
valid_batch_size: 2048
accum_count: [4]
accum_steps: [0]
normalization: "tokens"

# === Optimization ===
model_dtype: "fp32"
optim: "adam"
learning_rate: 2.0
warmup_steps: 8000
decay_method: "noam"
adam_beta1: 0.9
adam_beta2: 0.998
max_grad_norm: 0.0
label_smoothing: 0.1

# === Model architecture ===
model_type: text
model_task: seq2seq
encoder_type: transformer
decoder_type: transformer
position_encoding: false
max_relative_positions: -1
param_init: 0
param_init_glorot: true
word_vec_size: 512
hidden_size: 512
enc_layers: 6
dec_layers: 6
heads: 8
transformer_ff: 2048
dropout: [0.1]
attention_dropout: [0.1]

# === Evaluation ===
valid_metrics: [BLEU]
scoring_debug: true
dump_preds: true

EOF

# === Run with CodeCarbon ===
echo "🚀 Starting base WMT training..."
python3 - <<END
from codecarbon import OfflineEmissionsTracker as EmissionsTracker
import subprocess
import datetime

tracker = EmissionsTracker(
    project_name="$MODEL_NAME",
    output_dir="$CO2_PATH",
    save_to_file=True,
    country_iso_code="NLD",          # <- Add this (Netherlands ISO code)
    region="NL"                      # <- Optional: fallback region code
)
tracker.start()

ret = subprocess.run(["onmt_train", "-config", "$YAML"], capture_output=True, text=True)

tracker.stop()

with open("$LOG_FILE", "a") as f:
    f.write("\n===== $(date "+%Y-%m-%d %H:%M:%S") =====\n")
    f.write("Training base WMT model\n")
    f.write("Return code: {}\n".format(ret.returncode))
    f.write("--- STDOUT ---\n")
    f.write(ret.stdout + "\n")
    f.write("--- STDERR ---\n")
    f.write(ret.stderr + "\n")
    f.write("✅ Training completed: $MODEL_NAME\n" if ret.returncode == 0 else "❌ Training failed\n")
END
