#!/bin/bash

# Name: run_all_significance.sh
# Description: Bootstrap significance testing for BLEU, chrF2, and TER
# Author: Joye
# Usage: ./run_all_significance.sh

# Set paths
LOG_DIR="/home/pourmost/Redo/RANLP/log/evaluation_logs"

echo "📊 Starting bootstrap significance tests for BLEU, chrF2, and TER..."

python3 - <<EOF
import os
import re
import numpy as np
import pandas as pd

log_dir = "$LOG_DIR"
config_ids = [f"C{i}" for i in range(1, 8)]
metrics = {
    "bleu": {"higher_is_better": True, "label": "BLEU"},
    "chrf": {"higher_is_better": True, "label": "chrF2"},
    "ter":  {"higher_is_better": False, "label": "TER"}
}
bootstrap_samples = 1000
np.random.seed(42)

def load_scores(metric):
    scores = {}
    for cid in config_ids:
        path = os.path.join(log_dir, f"{cid}_{metric}_sentences.txt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")
        vals = []
        with open(path) as f:
            for line in f:
                # All metrics follow the same format: <metric>|... = SCORE
                match = re.search(r'=\s*([0-9.]+)', line)
                if match:
                    vals.append(float(match.group(1)))
        if not vals:
            raise ValueError(f"No scores found in {path}")
        scores[cid] = np.array(vals)
    return scores

def bootstrap_test(scores, higher_is_better=True):
    pval_matrix = pd.DataFrame(index=config_ids, columns=config_ids)
    n = len(next(iter(scores.values())))

    for i in config_ids:
        for j in config_ids:
            if i == j:
                pval_matrix.loc[i, j] = "-"
            else:
                diffs = []
                for _ in range(bootstrap_samples):
                    idx = np.random.choice(n, n, replace=True)
                    diff = scores[i][idx].mean() - scores[j][idx].mean()
                    diffs.append(diff)
                diffs = np.array(diffs)
                p = np.mean(diffs <= 0) if higher_is_better else np.mean(diffs >= 0)
                flag = "✔️" if p <= 0.05 else "✖️"
                pval_matrix.loc[i, j] = f"{p:.4f} {flag}"
    return pval_matrix

for metric, meta in metrics.items():
    print(f"\n📋 Pairwise bootstrap p-value matrix ({meta['label']}):")
    scores = load_scores(metric)
    matrix = bootstrap_test(scores, meta["higher_is_better"])
    print(matrix.to_string())
    output_path = os.path.join(log_dir, f"{metric}_bootstrap_pvals.csv")
    matrix.to_csv(output_path)
    print(f"✅ Saved to: {output_path}")
EOF
