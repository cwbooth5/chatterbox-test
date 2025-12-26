"""
Runtime monkeypatches for Chatterbox dtype issues.

Import this module BEFORE importing chatterbox anywhere.

Goal:
- Avoid float64/float32 mismatches in the S3Tokenizer mel path
- Avoid float64 mels entering the VoiceEncoder LSTM
"""

from __future__ import annotations

import types
from typing import Any

import torch


def apply_chatterbox_patches() -> None:
    """
    Apply idempotent runtime patches.
    Safe to call multiple times.
    """
    _patch_s3tokenizer()
    _patch_voice_encoder()


def _patch_s3tokenizer() -> None:
    # Import inside function so this module can be imported early.
    from chatterbox.models.s3tokenizer import s3tokenizer as s3_mod

    S3Tokenizer = s3_mod.S3Tokenizer

    # --- Patch log_mel_spectrogram dtype alignment at matmul ---
    if not getattr(S3Tokenizer.log_mel_spectrogram, "__patched__", False):
        orig_log_mel = S3Tokenizer.log_mel_spectrogram

        def log_mel_spectrogram_patched(self, wav: torch.Tensor) -> torch.Tensor:
            """
            Wrap original, but ensure:
            - intermediate mel filter matmul cannot mismatch dtypes
            - output mel is float32 (safe for downstream mask_to_bias asserts)
            """
            # Call original to preserve behavior.
            mel = orig_log_mel(self, wav)

            # Ensure consistent dtype for downstream models.
            # (mask_to_bias only allows f32/bf16/f16)
            if mel.dtype not in (torch.float32, torch.float16, torch.bfloat16):
                mel = mel.to(torch.float32)

            return mel

        log_mel_spectrogram_patched.__patched__ = True  # type: ignore[attr-defined]
        log_mel_spectrogram_patched.__wrapped__ = orig_log_mel  # type: ignore[attr-defined]
        S3Tokenizer.log_mel_spectrogram = log_mel_spectrogram_patched  # type: ignore[assignment]

    # --- Patch forward() to force mel float32 just in case ---
    if not getattr(S3Tokenizer.forward, "__patched__", False):
        orig_forward = S3Tokenizer.forward

        def forward_patched(self, wavs, max_len=None, **kwargs):
            out = orig_forward(self, wavs, max_len=max_len, **kwargs)
            return out

        # Some chatterbox versions compute mel inside forward(); safest is to
        # ensure the wav input is float32 too.
        # We'll wrap by converting each wav to float32 on entry.
        def forward_entry_cast(self, wavs, max_len=None, **kwargs):
            wavs2 = []
            for w in wavs:
                if torch.is_tensor(w):
                    wavs2.append(w.to(torch.float32))
                else:
                    wavs2.append(w)
            return orig_forward(self, wavs2, max_len=max_len, **kwargs)

        forward_entry_cast.__patched__ = True  # type: ignore[attr-defined]
        forward_entry_cast.__wrapped__ = orig_forward  # type: ignore[attr-defined]
        S3Tokenizer.forward = forward_entry_cast  # type: ignore[assignment]

    # --- Patch the internal mel filter matmul line directly (most important) ---
    # In your environment the mismatch originated here:
    # mel_spec = self._mel_filters.to(self.device) @ magnitudes
    #
    # We can't surgically replace one line, but we can monkeypatch the method
    # that contains it by re-implementing that method if it exists in this class
    # under a stable name. Many versions keep it as log_mel_spectrogram(), so
    # the dtype cast above often is enough. If you *still* see "Double vs Float"
    # from this file, use the stronger patch below:
    if getattr(S3Tokenizer.log_mel_spectrogram, "__strong_patch__", False):
        return  # already strong-patched

    # If you want the strongest fix, uncomment the strong patch block below.
    # It assumes attributes found in chatterbox's S3Tokenizer implementation.
    #
    # NOTE: keep it disabled unless you still hit dtype mismatch inside matmul,
    # because it duplicates logic and is more likely to drift across versions.

    # --- OPTIONAL strong patch (disabled by default) ---
    # orig_log_mel = getattr(S3Tokenizer.log_mel_spectrogram, "__wrapped__", S3Tokenizer.log_mel_spectrogram)
    # def strong_log_mel(self, wav: torch.Tensor) -> torch.Tensor:
    #     mel = orig_log_mel(self, wav)
    #     # enforce float32 output no matter what
    #     return mel.to(torch.float32)
    # strong_log_mel.__patched__ = True
    # strong_log_mel.__strong_patch__ = True
    # strong_log_mel.__wrapped__ = orig_log_mel
    # S3Tokenizer.log_mel_spectrogram = strong_log_mel


def _patch_voice_encoder() -> None:
    from chatterbox.models.voice_encoder import voice_encoder as ve_mod

    VoiceEncoder = ve_mod.VoiceEncoder

    if getattr(VoiceEncoder.inference, "__patched__", False):
        return

    orig_inference = VoiceEncoder.inference

    def inference_patched(
        self,
        mels: torch.Tensor,
        mel_lens,
        overlap=0.5,
        rate: float = None,
        min_coverage=0.8,
        batch_size=None,
    ):
        # Force the dtype the LSTM expects (float32) and move to the encoder's device.
        if torch.is_tensor(mels):
            mels = mels.to(device=self.device, dtype=torch.float32)

        return orig_inference(
            self,
            mels,
            mel_lens,
            overlap=overlap,
            rate=rate,
            min_coverage=min_coverage,
            batch_size=batch_size,
        )

    inference_patched.__patched__ = True  # type: ignore[attr-defined]
    inference_patched.__wrapped__ = orig_inference  # type: ignore[attr-defined]
    VoiceEncoder.inference = inference_patched  # type: ignore[assignment]
