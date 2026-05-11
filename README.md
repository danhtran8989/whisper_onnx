# Whisper ONNX Transcriber

Transcribe Vietnamese (or other languages) audio using ONNX-exported Whisper models.

The package provides a reusable Python class and a command-line interface.

## Requirements

- Python 3.10+
- Dependencies: `onnxruntime`, `librosa`, `transformers`, `numpy`

Install them with:

```bash
pip install onnxruntime librosa transformers numpy
```

## Model Preparation

You need the ONNX-exported Whisper model (encoder + decoder).  
Place them in a folder like `ckpts/whisper-small/onnx/`:

```text
ckpts/whisper-small/
└── onnx/
    ├── encoder_model.onnx
    └── decoder_model.onnx
```

> The ONNX models can be exported from a Hugging Face Whisper model using [`optimum-cli`](https://huggingface.co/docs/optimum/en/onnxruntime/usage_guides/export) or the [transformers.onnx](https://huggingface.co/docs/transformers/en/serialization#exporting-a-model-to-onnx) export.

## Usage

### Command-Line Interface

Run the CLI script from the project root:

```bash
python apps/cli/main.py --input <audio_file> [options]
```

**Example** (Vietnamese audio, default model):

```bash
python apps/cli/main.py --input 16.wav
```

or using the short flag:

```bash
python apps/cli/main.py -i 16.wav
```

**Options**:

| Argument          | Default                    | Description |
|-------------------|----------------------------|-----------|
| `--input`, `-i`   | **required**               | Path to the input audio file (WAV, MP3, etc.) |
| `--model_dir`     | `ckpts/whisper-small`      | Directory containing the `onnx/` subdirectory |
| `--language`      | `vi`                       | Language code (e.g., `vi`, `en`, `fr`) |
| `--task`          | `transcribe`               | `transcribe` or `translate` |
| `--max_len`       | `448`                      | Maximum number of tokens to generate |
| `--providers`     | `CPUExecutionProvider`     | ONNX Runtime execution providers (can specify multiple) |
| `--cpu`           | `True`                     | Force CPU usage (default) |

### Python Class

```python
from whisper_onnx import WhisperONNXTranscriber

transcriber = WhisperONNXTranscriber(
    model_dir="ckpts/whisper-small",
    language="vi",
    task="transcribe",
    max_len=448,
    providers=["CPUExecutionProvider"]
)

text = transcriber.transcribe("16.wav")
print(text)
```

## Project Structure

```text
.
├── apps/
│   └── cli/
│       └── main.py          # CLI entry point
├── src/
│   └── whisper_onnx/
│       ├── __init__.py
│       └── transcriber.py   # WhisperONNXTranscriber class
└── README.md
```

## Notes

- Audio is automatically resampled to **16 kHz mono**.
- The decoder currently uses **greedy search** (no beam search).
- Make sure the ONNX files are named exactly `encoder_model.onnx` and `decoder_model.onnx` inside the `onnx/` folder.
- If you encounter a `FileNotFoundError`, verify the model directory structure.
