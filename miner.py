import argparse
import asyncio
from datetime import datetime
import json
import os
import random
import shutil
import time
import bittensor as bt

from huggingface_hub.hf_api import api
from flockoff import constants
from flockoff.constants import Competition
from dotenv import load_dotenv

from flockoff.miners import chain, model
from flockoff.miners.data import ModelId
from flockoff.utils.chain import assert_registered
from flockoff.validator.trainer import get_hg_revision
from flockoff.validator.validator_utils import load_jsonl

def download_dataset(
        namespace: str, revision: str, local_dir: str = "data", force: bool = False
):
    if not os.path.isabs(local_dir):
        local_dir = os.path.abspath(local_dir)

    if force:
        print(f"[HF] Force dataset download {namespace}@{revision} → {local_dir}")
        shutil.rmtree(local_dir, ignore_errors=True)
        
    os.makedirs(local_dir, exist_ok=True)

    print(f"[HF] Downloading dataset {namespace}@{revision} → {local_dir}")
    try:
        api.snapshot_download(
            repo_id=namespace, local_dir=local_dir, revision=revision, repo_type="dataset"
        )
    except Exception as e:
        print(f"api.snapshot_download error:{e}")

load_dotenv()

def load_dataset(eval_data_dir: str = "data/eval_data", eval_file: str = "data.jsonl"):
    competition = Competition.from_defaults()
    eval_namespace = competition.repo
    main_commit_id = get_hg_revision(eval_namespace, constants.eval_commit)
    download_dataset(
        eval_namespace,
        main_commit_id,
        local_dir=eval_data_dir,
        force=True
    )
    os.rename(os.path.join(eval_data_dir, "data.jsonl"), os.path.join(eval_data_dir, eval_file))

def make_submission(eval_data_dir: str = "data/eval_data", eval_file: str = "data.jsonl", submission_dir: str = "data/submissions", submission_size: int|None = None):
    eval_data = load_jsonl(os.path.join(eval_data_dir, eval_file))
    random.shuffle(eval_data)
    if submission_size is not None:
        eval_data = eval_data[:submission_size]
    os.makedirs(submission_dir, exist_ok=True)
    date = datetime.now().strftime("%Y%m%d%H")
    submission_file = f"submission_{date}_{submission_size}.jsonl" if submission_size is not None else f"submission_{date}.jsonl"
    print(f"📂 Saving submission to {os.path.join(submission_dir, submission_file)}")
    with open(os.path.join(submission_dir, submission_file), "w") as f:
        for item in eval_data:
            f.write(json.dumps(item, sort_keys=True) + "\n")

    print(f"✅ Submission file saved to {os.path.join(submission_dir, submission_file)}")
    return os.path.join(submission_dir, submission_file)

def wait_until(run_at: str) -> None:
    """Sleep until the given time today (HH:MM or HH:MM:SS). If already past, run tomorrow."""
    from datetime import time as dt_time
    parts = run_at.strip().split(":")
    if len(parts) == 2:
        target = dt_time(int(parts[0]), int(parts[1]))
    elif len(parts) == 3:
        target = dt_time(int(parts[0]), int(parts[1]), int(parts[2]))
    else:
        raise ValueError("--run-at must be HH:MM or HH:MM:SS")
    now = datetime.now()
    run_at_dt = datetime.combine(now.date(), target)
    if run_at_dt <= now:
        from datetime import timedelta
        run_at_dt += timedelta(days=1)
    delta = (run_at_dt - now).total_seconds()
    print(f"⏰ Scheduled run at {run_at_dt}. Sleeping {delta:.0f}s ({delta/3600:.1f}h)...")
    time.sleep(delta)


def get_config():
    # Initialize an argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-data-dir", type=str, default="data/eval_data")
    parser.add_argument("--eval-file", type=str, default="data.jsonl")
    parser.add_argument("--submission-dir", type=str, default="data/submissions")
    parser.add_argument("--submission-size", type=int, default=None)
    parser.add_argument(
        "--run-at",
        type=str,
        default=None,
        metavar="HH:MM",
        help="Run main every day at this time (24h). E.g. 14:30. Loops forever.",
    )
    parser.add_argument(
        "--hf_repo_id",
        type=str,
        help="The hugging face repo id, which should include the org or user and repo name",
    )
    parser.add_argument(
        "--netuid",
        type=str,
        help="The subnet UID.",
    )

    # Include wallet and logging arguments from bittensor
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    # Parse the arguments and create a configuration namespace
    return bt.config(parser)

async def main(config: bt.config):
    bt.logging(config=config)
    
    wallet = bt.wallet(config=config)
    subtensor = bt.subtensor(config=config)
    metagraph: bt.metagraph = subtensor.metagraph(int(config.netuid))

    bt.logging.info(f"Starting miner with config: {config}")

    # Make sure we're registered and have a HuggingFace token.
    assert_registered(wallet, metagraph)

    now = datetime.now().strftime("%Y%m%d%H")
    eval_file = f"eval_data_{now}.jsonl"
    if not os.path.exists(os.path.join(config.eval_data_dir, eval_file)):
        print(f"❌ Eval file does not exist: {os.path.join(config.eval_data_dir, eval_file)}")
        print("📑 Download the eval data first...")
        load_dataset(config.eval_data_dir, eval_file)
    else:
        print(f"✅ Eval file exists: {os.path.join(config.eval_data_dir, eval_file)}")
    
    print("⏭️ Go to the submission mode to make a submission.")
    submission_file_path =make_submission(config.eval_data_dir, eval_file, config.submission_dir)

    commit_id = model.upload_data(config.hf_repo_id, submission_file_path)

    print(f"✅ Uploaded submission to {commit_id}")

    competition = Competition.from_defaults()

    model_id_with_commit = ModelId(
        namespace=config.hf_repo_id, commit=commit_id, competition_id=competition.id
    )
    bt.logging.success(
        f"Now committing to the chain with model_id: {model_id_with_commit}"
    )

    # We can only commit to the chain every 20 minutes, so run this in a loop, until successful.
    while True:
        try:
            await chain.store_model_metadata(
                subtensor=subtensor,
                wallet=wallet,
                subnet_uid=config.netuid,
                data=model_id_with_commit.to_compressed_str(),
            )

            bt.logging.success("Committed dataset to the chain.")
            break
        except Exception as e:
            bt.logging.error(f"Failed to advertise model on the chain: {e}")
            bt.logging.error("Retrying in 120 seconds...")
            time.sleep(120)


if __name__ == "__main__":
    config = get_config()
    if config.run_at:
        while True:
            wait_until(config.run_at)
            asyncio.run(main(config))
            print(f"✅ Run finished. Next run at {config.run_at} tomorrow.")
    else:
        asyncio.run(main(config))