import os
import numpy as np
import onnxruntime as ort
import librosa
from transformers import WhisperTokenizer, WhisperProcessor
from .utils import download_whisper_onnx_checkpoint

class WhisperONNXTranscriber:
    """
    ONNX-based Whisper transcriber for Vietnamese (or any language/task).
    Uses separate encoder and decoder ONNX models.
    """

    def __init__(
        self,
        model_dir="ckpts/whisper-small",
        language="vi",
        task="transcribe",
        sample_rate=16000,
        max_len=448,
        providers=["CPUExecutionProvider"]
    ):
        """
        Args:
            model_dir: Path to the HuggingFace Whisper model directory containing 'onnx/'
                       subdirectory with encoder_model.onnx and decoder_model.onnx.
            language: Language code for transcription (e.g., "vi").
            task: "transcribe" or "translate".
            sample_rate: Audio resampling rate (Whisper expects 16kHz).
            max_len: Maximum number of tokens to generate (excluding prefix tokens).
            providers: ONNX Runtime execution providers.
        """
        self.model_dir = model_dir
        self.encoder_path = os.path.join(model_dir, "onnx", "encoder_model.onnx")
        self.decoder_path = os.path.join(model_dir, "onnx", "decoder_model.onnx")
        self.language = language
        self.task = task
        self.sample_rate = sample_rate
        self.max_len = max_len

        # Validate that ONNX files exist; download once if any are missing
        if not os.path.exists(self.encoder_path) or not os.path.exists(self.decoder_path):
            download_whisper_onnx_checkpoint("onnx-community/whisper-small", model_dir)

        # After download (or if they already existed), verify both files are present
        if not os.path.exists(self.encoder_path):
            raise FileNotFoundError(f"Encoder ONNX file not found: {self.encoder_path}")
        if not os.path.exists(self.decoder_path):
            raise FileNotFoundError(f"Decoder ONNX file not found: {self.decoder_path}")

        # Load tokenizer and processor (used for prefix tokens and mel features)
        self.tokenizer = WhisperTokenizer.from_pretrained(
            model_dir, language=language, task=task
        )
        self.processor = WhisperProcessor.from_pretrained(
            model_dir, language=language, task=task
        )

        # Load ONNX sessions
        print("Loading ONNX models...")
        self.encoder_session = ort.InferenceSession(self.encoder_path, providers=providers)
        self.decoder_session = ort.InferenceSession(self.decoder_path, providers=providers)

        # Cache input names for later use
        self.encoder_input_names = [inp.name for inp in self.encoder_session.get_inputs()]
        self.decoder_input_names = [inp.name for inp in self.decoder_session.get_inputs()]

        print("Encoder inputs:", self.encoder_input_names)
        print("Decoder inputs:", self.decoder_input_names)
        print("Models loaded.")

    def load_audio(self, file_path):
        """Load and resample audio to the expected sample rate."""
        audio, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
        return audio

    def get_mel_features(self, audio):
        """Extract log-mel spectrogram features compatible with Whisper."""
        inputs = self.processor(audio, sampling_rate=self.sample_rate, return_tensors="np")
        return inputs.input_features  # shape: (1, 80, T)

    def greedy_decode(self, encoder_hidden_states):
        """
        Autoregressive greedy decode using the ONNX decoder.
        Args:
            encoder_hidden_states: numpy array of shape (1, enc_seq_len, hidden_size)
        Returns:
            Decoded text string.
        """
        # Prefix tokens: [SOT, language token, task token, no_timestamps]
        prefix_tokens = self.tokenizer.prefix_tokens
        input_ids = np.array([prefix_tokens], dtype=np.int64)
        eot_token = self.tokenizer.eos_token_id
        max_length = self.max_len + len(prefix_tokens)

        # Determine how to feed encoder hidden states
        if "encoder_hidden_states" in self.decoder_input_names:
            enc_key = "encoder_hidden_states"
        elif "encoder_outputs" in self.decoder_input_names:
            enc_key = "encoder_outputs"
        else:
            raise KeyError(
                f"Decoder expects neither 'encoder_hidden_states' nor 'encoder_outputs'. "
                f"Available: {self.decoder_input_names}"
            )

        # Base feed dictionary with static entries (encoder hidden states)
        feed_dict = {enc_key: encoder_hidden_states}

        # Add optional attention masks if required by the model
        if "encoder_attention_mask" in self.decoder_input_names:
            enc_seq_len = encoder_hidden_states.shape[1]
            feed_dict["encoder_attention_mask"] = np.ones((1, enc_seq_len), dtype=np.int64)
        elif "cross_attention_mask" in self.decoder_input_names:
            enc_seq_len = encoder_hidden_states.shape[1]
            feed_dict["cross_attention_mask"] = np.ones((1, enc_seq_len), dtype=np.int64)

        # Autoregressive generation loop
        for _ in range(max_length):
            # Update sequence-dependent inputs
            feed_dict["input_ids"] = input_ids
            if "attention_mask" in self.decoder_input_names:
                feed_dict["attention_mask"] = np.ones_like(input_ids, dtype=np.int64)

            outputs = self.decoder_session.run(None, feed_dict)
            logits = outputs[0]          # (1, seq_len, vocab_size)
            next_token_logits = logits[0, -1, :]
            next_token = np.argmax(next_token_logits).astype(np.int64)

            if next_token == eot_token:
                break

            input_ids = np.concatenate(
                [input_ids, np.array([[next_token]], dtype=np.int64)], axis=1
            )

        generated_tokens = input_ids[0, len(prefix_tokens):]
        text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return text

    def transcribe(self, audio_path):
        """
        Full pipeline: load audio -> mel features -> encoder -> decoder.
        Returns transcribed text.
        """
        # 1. Load audio
        audio = self.load_audio(audio_path)

        # 2. Compute mel features
        mel_features = self.get_mel_features(audio).astype(np.float32)  # (1, 80, T)

        # 3. Run encoder
        # The encoder expects input named "input_features" (typical for Whisper ONNX exports)
        encoder_outputs = self.encoder_session.run(
            None, {"input_features": mel_features}
        )
        encoder_hidden_states = encoder_outputs[0]  # (1, enc_seq_len, hidden_size)

        # 4. Decode
        transcription = self.greedy_decode(encoder_hidden_states)
        return transcription
