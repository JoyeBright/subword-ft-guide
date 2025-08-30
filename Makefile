# Makefile for my project
# Instruction: Just run 'make all' to execute the complete pipeline

.PHONY: all clean help bpe vocab apply-bpe train evaluate
.DEFAULT_GOAL := help

# Colors for output
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
NC := \033[0m # No Color

# Main target - runs everything in order
all: bpe vocab apply-bpe train evaluate
	@echo "$(GREEN)Complete pipeline finished!$(NC)"
	@echo "$(YELLOW)Check results in:$(NC)"
	@echo "   - Evaluation scores: evaluation_scores.csv"
	@echo "   - CO2 emissions: CO2/ directory"
	@echo "   - Training logs: log/training_logs/"
	@echo "   - Evaluation logs: log/evaluation_logs/"

# Learn BPE models
bpe:
	@echo "$(YELLOW)Learning BPE models...$(NC)"
	@chmod +x learn_bpe.sh
	@./learn_bpe.sh

# Build vocabularies
vocab: bpe
	@echo "$(YELLOW)Building vocabularies...$(NC)"
	@chmod +x build_vocab.sh
	@./build_vocab.sh

# Apply BPE to datasets
apply-bpe: vocab
	@echo "$(YELLOW)Applying BPE to datasets...$(NC)"
	@chmod +x apply_bpe.sh
	@./apply_bpe.sh

# Train all models
train: apply-bpe
	@echo "$(YELLOW)Training all models...$(NC)"
	@chmod +x train_all.sh
	@./train_all.sh

# Evaluate all models
evaluate: train
	@echo "$(YELLOW)Evaluating all models...$(NC)"
	@chmod +x evaluate_all.sh
	@./evaluate_all.sh

# Individual targets for specific steps
bpe-only: bpe
	@echo "$(GREEN)BPE learning complete$(NC)"

vocab-only: vocab
	@echo "$(GREEN)Vocabulary building complete$(NC)"

apply-bpe-only: apply-bpe
	@echo "$(GREEN)BPE application complete$(NC)"

train-only: train
	@echo "$(GREEN)Training complete$(NC)"

evaluate-only: evaluate
	@echo "$(GREEN)Evaluation complete$(NC)"

# Clean up generated files
clean:
	@echo "$(YELLOW)Cleaning up generated files...$(NC)"
	@rm -rf data/BPE_models/*
	@rm -rf Vocabs/*
	@rm -rf Models/*
	@rm -rf log/*
	@rm -rf CO2/*
	@rm -f evaluation_scores.csv
	@rm -f evaluation_scores_wmt.csv
	@echo "$(GREEN)Cleanup complete$(NC)"

# Show help
help:
	@echo "$(YELLOW)Subword Fine-tuning Guide - Makefile$(NC)"
	@echo ""
	@echo "$(GREEN)Available targets:$(NC)"
	@echo "  $(YELLOW)all$(NC)        - Run complete pipeline (BPE → Vocab → Apply BPE → Train → Evaluate)"
	@echo "  $(YELLOW)bpe$(NC)        - Learn BPE models only"
	@echo "  $(YELLOW)vocab$(NC)      - Build vocabularies only (requires BPE)"
	@echo "  $(YELLOW)apply-bpe$(NC)  - Apply BPE to datasets only (requires vocab)"
	@echo "  $(YELLOW)train$(NC)      - Train all models only (requires BPE data)"
	@echo "  $(YELLOW)evaluate$(NC)   - Evaluate all models only (requires trained models)"
	@echo "  $(YELLOW)clean$(NC)      - Remove all generated files and start fresh"
	@echo "  $(YELLOW)help$(NC)       - Show this help message"
	@echo ""
	@echo "$(GREEN)Quick start:$(NC)"
	@echo "  make all    # Run everything"
	@echo "  make clean  # Start fresh"
	@echo ""
	@echo "$(GREEN)Individual steps:$(NC)"
	@echo "  make bpe-only       # Just BPE learning"
	@echo "  make vocab-only     # Just vocabulary building"
	@echo "  make apply-bpe-only # Just BPE application"
	@echo "  make train-only     # Just training"
	@echo "  make evaluate-only  # Just evaluation"
