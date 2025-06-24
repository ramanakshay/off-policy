
.PHONY: install
install:
	uv sync

.PHONY: train-ddpg
train-ddpg:
	uv run src/train_ddpg.py

.PHONY: train-td3
train-td3:
	uv run src/train_td3.py

.PHONY: train-sac
train-sac:
	uv run src/train_sac.py

.PHONY: clean
clean:
	@echo "Cleaning up project artifacts.."
	find . -name '*.pyc' -delete
	find . -name '__pycache__' -delete
	rm -rf build dist src/*.egg-info
