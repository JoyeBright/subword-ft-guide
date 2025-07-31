#!/bin/bash
set -e

# === Fixed path to pretrained model ===
PRETRAINED="/home/pourmost/Redo/RANLP/Models/wmt_base/model_step_48000.pt"

# === Root directory (everything else under RANLP) ===
RANLP="/home/pourmost/Redo/RANLP"
DATA="${RANLP}/data"
MODELS="${RANLP}/Models"
VOCABS="${RANLP}/Vocabs"
LOG_DIR="${RANLP}/log/training_logs"
TB_DIR="${RANLP}/Tensorboard"
CO2_DIR="${RANLP}/CO2"
CONFIGS="${RANLP}/train_configs"
mkdir -p "$LOG_DIR" "$TB_DIR" "$CO2_DIR" "$CONFIGS"

# === Training setup ===
TRAIN_STEPS=200000
VALID_STEPS=1000

# === Config pairs: (BPE_MODEL, VOCAB_SOURCE, CONFIG_ID) ===
CONFIGS_LIST=(
  "wmt|wmt|C1"
  "wmt|medical_combined|C2"
  "wmt|medical|C3"
  "medical|wmt|C4"
  "medical|medical_combined|C5"
  "medical|medical|C6"
  "medical_combined|medical_combined|C7"
)

# === Loop over configs ===
for idx in "${!CONFIGS_LIST[@]}"; do
  TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")
  IFS='|' read -r BPE VOCAB_SRC CID <<< "${CONFIGS_LIST[$idx]}"
  CONFIG_NAME="${CID}_vocab_from-${VOCAB_SRC}-on-${BPE}"
  MODEL_DIR="${MODELS}/${CONFIG_NAME}"
  LOG_FILE="${LOG_DIR}/${CONFIG_NAME}.log"
  TB_PATH="${TB_DIR}/${CONFIG_NAME}"
  CO2_PATH="${CO2_DIR}/${CONFIG_NAME}"
  YAML="${CONFIGS}/${CONFIG_NAME}.yaml"

  echo -e "\n🚧 Preparing config: $CONFIG_NAME"
  mkdir -p "$MODEL_DIR" "$TB_PATH" "$CO2_PATH"

  TRAIN_SRC="${DATA}/medical/train.src.tok.bpe_${BPE}"
  TRAIN_TGT="${DATA}/medical/train.tgt.tok.bpe_${BPE}"
  DEV_SRC="${DATA}/medical/dev.src.tok.bpe_${BPE}"
  DEV_TGT="${DATA}/medical/dev.tgt.tok.bpe_${BPE}"

  VOCAB_SRC_PATH="${VOCABS}/vocab_from-${BPE}-on-${VOCAB_SRC}.en.src"
  VOCAB_TGT_PATH="${VOCABS}/vocab_from-${BPE}-on-${VOCAB_SRC}.de.tgt"

  # === Count vocab sizes ===
  SRC_VOCAB_SIZE=$(wc -l < "$VOCAB_SRC_PATH")
  TGT_VOCAB_SIZE=$(wc -l < "$VOCAB_TGT_PATH")

  echo "[INFO] SRC vocab size: $SRC_VOCAB_SIZE | TGT vocab size: $TGT_VOCAB_SIZE"

  cat > "$YAML" <<EOF
# === Data configuration ===
data:
  corpus_0:
    path_src: "$TRAIN_SRC"
    path_tgt: "$TRAIN_TGT"
    weight: 1
  valid:
    path_src: "$DEV_SRC"
    path_tgt: "$DEV_TGT"

update_vocab: true
share_vocab: false
train_from: "$PRETRAINED"

# === Vocabulary paths ===
src_vocab: "$VOCAB_SRC_PATH"
tgt_vocab: "$VOCAB_TGT_PATH"
src_vocab_size: ${SRC_VOCAB_SIZE}
tgt_vocab_size: ${TGT_VOCAB_SIZE}

# === Model output ===
save_model: "${MODEL_DIR}/model"
save_checkpoint_steps: 1000
keep_checkpoint: 6

# === Logging ===
log_file: "$LOG_FILE"
log_file_level: "INFO"
tensorboard: true
tensorboard_log_dir: "$TB_PATH"

# === Training steps ===
train_steps: $TRAIN_STEPS
valid_steps: $VALID_STEPS
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

  echo "🧪 Launching training for $CONFIG_NAME with CodeCarbon..."

  python3 - <<END
from codecarbon import OfflineEmissionsTracker as EmissionsTracker
import subprocess
import datetime

tracker = EmissionsTracker(
    project_name="$CONFIG_NAME",
    output_dir="$CO2_PATH",
    save_to_file=True,
    country_iso_code="NLD",
    region="NL"
)
tracker.start()

ret = subprocess.run(["onmt_train", "-config", "$YAML"], capture_output=True, text=True)

tracker.stop()

with open("$LOG_FILE", "a") as f:
    f.write("\n===== $TIMESTAMP =====\n")
    f.write("Config name: $CONFIG_NAME\n")
    f.write("BPE model: $BPE\n")
    f.write("Vocab source: $VOCAB_SRC\n")
    f.write("SRC vocab size: $SRC_VOCAB_SIZE\n")
    f.write("TGT vocab size: $TGT_VOCAB_SIZE\n")
    f.write("Return code: {}\n".format(ret.returncode))
    f.write("--- STDOUT ---\n")
    f.write(ret.stdout + "\n")
    f.write("--- STDERR ---\n")
    f.write(ret.stderr + "\n")
    if ret.returncode != 0:
        f.write("❌ Training failed for $CONFIG_NAME\n")
    else:
        f.write("✅ Training completed for $CONFIG_NAME\n")
END
done

echo "✅ All trainings completed and logged."
