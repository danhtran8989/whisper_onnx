# src/whisper_onnx/utils.py

import sys
from pathlib import Path
from huggingface_hub import snapshot_download

# Add the src directory to Python path so we can import whisper_onnx
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

def download_whisper_onnx_checkpoint(repo_id: str, local_dir: str, no_symlinks: bool = False) -> None:
    """
    Download a model snapshot from Hugging Face Hub.

    Args:
        repo_id: Hugging Face repository ID (e.g., 'onnx-community/whisper-small')
        local_dir: Local directory to save the downloaded files
        no_symlinks: If True, use actual files instead of symlinks
                     (sets local_dir_use_symlinks=False)
    """
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=not no_symlinks
    )
    print("Done.")
