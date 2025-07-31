#!/bin/bash
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
export LC_CTYPE=en_US.UTF-8

# set -e

ROOT="/home/pourmost/Redo/RANLP"
MODELS="$ROOT/Models"
DATA="$ROOT/data/medical"
LOG_DIR="$ROOT/log/evaluation_logs"
EVAL_CONFIGS="$ROOT/eval_configs"
OUTPUT_CSV="$ROOT/evaluation_scores.csv"
EVAL_LOG_DIR="$LOG_DIR"
mkdir -p "$LOG_DIR" "$EVAL_CONFIGS"

DETOK_PERL="../OpenNMT-py/tools/detokenize.perl"

echo "Config ID,Config Name,BPE Model Used,Vocabulary Source,Fine-tune Dataset + BPE,Test Set Used,BLEU,COMET,CO2 (g),Time (s)" > "$OUTPUT_CSV"

declare -A TEST_SRC_MAP=(
  [wmt]="$DATA/test.src.tok.bpe_wmt"
  [medical]="$DATA/test.src.tok.bpe_medical"
  [medical_combined]="$DATA/test.src.tok.bpe_medical_combined"
)

declare -A TEST_TGT_MAP=(
  [wmt]="$DATA/test.tgt.tok.bpe_wmt"
  [medical]="$DATA/test.tgt.tok.bpe_medical"
  [medical_combined]="$DATA/test.tgt.tok.bpe_medical_combined"
)

declare -a CONFIGS=(
"C1|vocab_from-wmt-on-wmt|wmt|wmt|medical (BPE: wmt)"
"C2|vocab_from-medical_combined-on-wmt|wmt|medical_combined|medical (BPE: wmt)"
"C3|vocab_from-medical-on-wmt|wmt|medical|medical (BPE: wmt)"
"C4|vocab_from-wmt-on-medical|medical|wmt|medical (BPE: medical)"
"C5|vocab_from-medical_combined-on-medical|medical|medical_combined|medical (BPE: medical)"
"C6|vocab_from-medical-on-medical|medical|medical|medical (BPE: medical)"
"C7|vocab_from-medical_combined-on-medical_combined|medical_combined|medical_combined|medical (BPE: medical_combined)"
)

for CONFIG in "${CONFIGS[@]}"; do
  IFS='|' read -r ID NAME BPE_MODEL VOC_SRC FT_DESC <<< "$CONFIG"
  MODEL_PATH=$(find "$MODELS/$ID"* -name "*.pt" | sort | tail -n 1)
  LOG_FILE="$LOG_DIR/${ID}.log"
  YAML_PATH="$EVAL_CONFIGS/${ID}_eval.yaml"
  PRED_DIR="$MODELS/${ID}_pred"
  mkdir -p "$PRED_DIR"

  SRC_FILE="${TEST_SRC_MAP[$BPE_MODEL]}"
  TGT_FILE="${TEST_TGT_MAP[$BPE_MODEL]}"
  PRED_FILE="$PRED_DIR/pred_${ID}"

  echo "🔍 $ID: Using model $MODEL_PATH" | tee "$LOG_FILE"
  echo "🧪 Test set: $SRC_FILE" | tee -a "$LOG_FILE"

  # Create YAML config
  cat <<EOF > "$YAML_PATH"
model: $MODEL_PATH
src: $SRC_FILE
output: $PRED_FILE
n_best: 1
min_length: 0
max_length: 150
beam_size: 6
verbose: true
report_time: true
EOF

  echo "🚀 Translating $ID..." | tee -a "$LOG_FILE"
  onmt_translate -config "$YAML_PATH" 2>&1 | tee -a "$LOG_FILE"

  DETOK="$PRED_FILE.detok"
  REF_DETOK="$TGT_FILE.detok"
  SYS_PP="${DETOK}_pp"
  REF_PP="${REF_DETOK}_pp"
  SRC_PP="${SRC_FILE/.tok.bpe_/.tok.detok_pp}"

  perl "$DETOK_PERL" -no-escape -l de -threads 8 < "$PRED_FILE" > "$DETOK"
  perl "$DETOK_PERL" -no-escape -l de -threads 8 < "$TGT_FILE" > "$REF_DETOK"
  sed -E 's/(@@ )|(@@ ?$)|(@@)//g' "$DETOK" > "$SYS_PP"
  sed -E 's/(@@ )|(@@ ?$)|(@@)//g' "$REF_DETOK" > "$REF_PP"
  perl "$DETOK_PERL" -no-escape -l de -threads 8 < "$SRC_FILE" | \
    sed -E 's/(@@ )|(@@ ?$)|(@@)//g' > "$SRC_PP"

  echo "🔹 Running sacreBLEU for $ID..." | tee -a "$LOG_FILE"
  sacrebleu "$REF_PP" -i "$SYS_PP" --metrics bleu ter chrf \
      > "$EVAL_LOG_DIR/${ID}_sacrebleu.txt" 2>&1
  
  sacrebleu "$REF_PP" -i "$SYS_PP" --metrics bleu --sentence-level \
    --quiet > "$EVAL_LOG_DIR/${ID}_bleu_sentences.txt"

  sacrebleu "$REF_PP" -i "$SYS_PP" --metrics ter --sentence-level \
  --quiet > "$EVAL_LOG_DIR/${ID}_ter_sentences.txt"

  sacrebleu "$REF_PP" -i "$SYS_PP" --metrics chrf --sentence-level \
  --quiet > "$EVAL_LOG_DIR/${ID}_chrf_sentences.txt"

  if [[ $? -ne 0 ]]; then
      echo "❌ sacreBLEU failed for $ID. See $EVAL_LOG_DIR/${ID}_sacrebleu.txt" | tee -a "$LOG_FILE"
      BLEU="N/A"
  else
      BLEU=$(jq -r '.[] | select(.name=="BLEU") | .score' "$EVAL_LOG_DIR/${ID}_sacrebleu.txt")
      echo "✅ sacreBLEU done. BLEU: $BLEU" | tee -a "$LOG_FILE"
  fi

  echo "🔹 Running COMET for $ID..." | tee -a "$LOG_FILE"

  comet-score \
    -s "$SRC_PP" \
    -t "$SYS_PP" \
    -r "$REF_PP" \
    --model Unbabel/wmt22-comet-da \
    > "$EVAL_LOG_DIR/${ID}_comet.txt" 2>&1

  if [[ $? -ne 0 ]]; then
    echo "❌ COMET CLI failed for $ID." | tee -a "$LOG_FILE"
    COMET="N/A"
  else
    COMET=$(tail -n 1 "$EVAL_LOG_DIR/${ID}_comet.txt" | grep -Eo 'score: [0-9.]+' | awk '{print $2}')
    echo "📈 COMET: $COMET" | tee -a "$LOG_FILE"
  fi

  CSV_PATH="$ROOT/CO2/${ID}_${NAME}/emissions.csv"
  CO2=$(awk -F',' 'NR==2 { printf "%.2f", $5 * 1000 }' "$CSV_PATH")
  TIME=$(awk -F',' 'NR==2 { print int($4) }' "$CSV_PATH")
  echo "🌱 CO2: ${CO2}g, ⏱️ Time: ${TIME}s" | tee -a "$LOG_FILE"

  echo "$ID,$NAME,$BPE_MODEL,$VOC_SRC,\"$FT_DESC\",$(basename "$SRC_FILE"),$BLEU,$COMET,$CO2,$TIME" >> "$OUTPUT_CSV"
done

echo "✅ All evaluations complete. Results saved to $OUTPUT_CSV"
