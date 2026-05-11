import argparse
from huggingface_hub import snapshot_download

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Download a model snapshot from Hugging Face Hub.")
    parser.add_argument("--repo_id", type=str, default="onnx-community/whisper-small",
                        help="Hugging Face repository ID (e.g., 'onnx-community/whisper-small')")
    parser.add_argument("--local_dir", type=str, default="ckpts/whisper-small",
                        help="Local directory to save the downloaded files")
    parser.add_argument("--no_symlinks", action="store_true",
                        help="If set, use actual files instead of symlinks (local_dir_use_symlinks=False)")
    return parser.parse_args()

def download_model(repo_id, local_dir, no_symlinks):
    """Download a model snapshot from Hugging Face Hub."""
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=not no_symlinks   # False if --no_symlinks is given
    )
    print("Done.")

def main():
    args = parse_args()
    download_model(args.repo_id, args.local_dir, args.no_symlinks)

if __name__ == "__main__":
    main()