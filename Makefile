include $(ENV_FILE)
export

PYTHONPATH=./

miner_dev:
	uv run python miner.py \
		--netuid $(NETUID) \
		--hf_repo_id $(HF_REPO_ID) \
		--wallet.name $(WALLET_NAME) \
		--wallet.hotkey $(WALLET_HOTKEY) \
		--subtensor.network $(SUBTENSOR_NETWORK) \
		--logging.debug \
		--eval-data-dir $(EVAL_DATA_DIR) \
		--submission-dir $(SUBMISSION_DIR) \