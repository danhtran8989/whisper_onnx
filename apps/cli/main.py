#!/usr/bin/env python3
"""
Command-line interface for Whisper ONNX transcriber.
"""

#!/usr/bin/env python3
"""
Command-line interface for Whisper ONNX transcriber.
"""

import sys
import argparse
from pathlib import Path

# Add the src directory to Python path so we can import whisper_onnx
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.whisper_onnx import WhisperONNXTranscriber

def main():
    parser = argparse.ArgumentParser(description="Transcribe audio using Whisper ONNX models")
    parser.add_argument("--input", "-i", help="Path to the input audio file")
    parser.add_argument("--model_dir", default="ckpts/whisper-small",
                        help="Directory containing the Whisper model (with onnx subdirectory)")
    parser.add_argument("--language", default="vi", help="Language code (e.g., 'vi', 'en')")
    parser.add_argument("--task", default="transcribe", choices=["transcribe", "translate"],
                        help="Task to perform")
    parser.add_argument("--max_len", type=int, default=448,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--cpu", action="store_true", default=True,
                        help="Use CPU (default). GPU can be specified via providers.")
    parser.add_argument("--providers", nargs="+", default=["CPUExecutionProvider"],
                        help="ONNX Runtime providers, e.g., 'CUDAExecutionProvider'")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Audio file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    try:
        transcriber = WhisperONNXTranscriber(
            model_dir=args.model_dir,
            language=args.language,
            task=args.task,
            max_len=args.max_len,
            providers=args.providers
        )
        text = transcriber.transcribe(args.audio_file)
        print("\nTranscription:")
        print(text)
    except FileNotFoundError as e:
        print(f"Model error: {e}", file=sys.stderr)
        print("Please ensure the ONNX models exist in the specified model_dir/onnx/", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()

def main():
    parser = argparse.ArgumentParser(description="Transcribe audio using Whisper ONNX models")
    parser.add_argument("--input", "-i", help="Path to the input audio file")
    parser.add_argument("--model_dir", default="ckpts/whisper-small",
                        help="Directory containing the Whisper model (with onnx subdirectory)")
    parser.add_argument("--language", default="vi", help="Language code (e.g., 'vi', 'en')")
    parser.add_argument("--task", default="transcribe", choices=["transcribe", "translate"],
                        help="Task to perform")
    parser.add_argument("--max_len", type=int, default=448,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--cpu", action="store_true", default=True,
                        help="Use CPU (default). GPU can be specified via providers.")
    parser.add_argument("--providers", nargs="+", default=["CPUExecutionProvider"],
                        help="ONNX Runtime providers, e.g., 'CUDAExecutionProvider'")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Audio file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    try:
        transcriber = WhisperONNXTranscriber(
            model_dir=args.model_dir,
            language=args.language,
            task=args.task,
            max_len=args.max_len,
            providers=args.providers
        )
        text = transcriber.transcribe(args.audio_file)
        print("\nTranscription:")
        print(text)
    except FileNotFoundError as e:
        print(f"Model error: {e}", file=sys.stderr)
        print("Please ensure the ONNX models exist in the specified model_dir/onnx/", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
