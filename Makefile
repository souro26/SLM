.PHONY: help env lint test tokenizer data train-pilot train-full eval clean

help:
	@echo "SLM - Python Specialized Small Language Model"
	@echo ""
	@echo "  make env          Create conda environment"
	@echo "  make lint         Run ruff + black"
	@echo "  make test         Run all tests"
	@echo "  make tokenizer    Train BPE tokenizer"
	@echo "  make data         Run data pipeline"
	@echo "  make train-pilot  Pilot run (2B tokens)"
	@echo "  make train-full   Full run (10B tokens)"
	@echo "  make eval         Run HumanEval + MBPP"
	@echo "  make clean        Remove cache and logs"

env:
	conda env create -f environment.yml

lint:
	ruff check .
	black --check .

test:
	pytest tests/ -v --tb=short

tokenizer:
	python -m tokenizer.train --config configs/tokenizer.yaml

data:
	python -m data.pipeline --config configs/data.yaml

train-pilot:
	python -m train.run --config configs/train_pilot.yaml

train-full:
	python -m train.run --config configs/train_full.yaml

eval:
	python -m eval.run --config configs/eval.yaml

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -rf logs/*.csv logs/*.log