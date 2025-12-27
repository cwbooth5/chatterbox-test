"""
Runtime monkeypatches for Chatterbox dtype issues and watermarking control.

Import this module BEFORE importing chatterbox anywhere.

Goal:
- Avoid float64/float32 mismatches in the S3Tokenizer mel path
- Avoid float64 mels entering the VoiceEncoder LSTM
- Optionally disable Perth watermarking

Usage:
    To disable watermarking, set DISABLE_WATERMARKING before importing Chatterbox:

    import monkeypatches
    monkeypatches.DISABLE_WATERMARKING = True
    monkeypatches.apply_chatterbox_patches()

    from chatterbox.tts_turbo import ChatterboxTurboTTS

    Default behavior (DISABLE_WATERMARKING = False) keeps watermarking enabled.
"""

from __future__ import annotations


import torch

# Flag to control watermarking (default: True = watermark enabled)
DISABLE_WATERMARKING = False


# Monkey patch torch.Tensor.to() to prevent float64 on MPS
_original_tensor_to = torch.Tensor.to


def _patched_tensor_to(self, *args, **kwargs):
    """
    Patch Tensor.to() to convert float64 to float32 on MPS.
    MPS doesn't support float64, so we intercept the call.

    This is specifically here so we can run the turbo model on MPS.
    """
    # Check if we're trying to convert to MPS
    device_arg = None
    dtype_arg = None

    # Parse positional arguments
    if args:
        first_arg = args[0]
        if isinstance(first_arg, (str, torch.device)):
            device_arg = first_arg
        elif isinstance(first_arg, torch.dtype):
            dtype_arg = first_arg
        elif isinstance(first_arg, torch.Tensor):
            device_arg = first_arg.device
            dtype_arg = first_arg.dtype

    # Parse keyword arguments
    if 'device' in kwargs:
        device_arg = kwargs['device']
    if 'dtype' in kwargs:
        dtype_arg = kwargs['dtype']

    # Convert device to string for checking
    device_str = str(device_arg) if device_arg else ""

    # If target is MPS and dtype is float64, change to float32
    if 'mps' in device_str and dtype_arg == torch.float64:
        dtype_arg = torch.float32
        kwargs['dtype'] = torch.float32

    # If device is MPS and no dtype specified, ensure we don't get float64
    if 'mps' in device_str and dtype_arg is None and self.dtype == torch.float64:
        kwargs['dtype'] = torch.float32

    # Call original with potentially modified arguments
    return _original_tensor_to(self, *args, **kwargs)


torch.Tensor.to = _patched_tensor_to


def apply_chatterbox_patches() -> None:
    """
    Apply idempotent runtime patches.
    Safe to call multiple times.
    """
    _patch_s3tokenizer()
    _patch_voice_encoder()
    _patch_watermarking()


def _patch_s3tokenizer() -> None:
    # Import inside function so this module can be imported early.
    from chatterbox.models.s3tokenizer import s3tokenizer as s3_mod

    S3Tokenizer = s3_mod.S3Tokenizer

    # --- Patch __init__ to convert float64 to float32 on MPS ---
    if not getattr(S3Tokenizer.__init__, "__patched_init__", False):
        orig_init = S3Tokenizer.__init__

        def init_patched(self, *args, **kwargs):
            orig_init(self, *args, **kwargs)
            # After initialization, convert all float64 to float32 if on MPS
            if hasattr(torch.backends, "mps") and hasattr(self, 'device') and "mps" in str(self.device):
                for param in self.parameters() if hasattr(self, 'parameters') else []:
                    if param.dtype == torch.float64:
                        param.data = param.data.to(torch.float32)
                if hasattr(self, '__dict__'):
                    for attr_name, attr_val in self.__dict__.items():
                        if isinstance(attr_val, torch.Tensor) and attr_val.dtype == torch.float64:
                            setattr(self, attr_name, attr_val.to(torch.float32))

        init_patched.__patched_init__ = True  # type: ignore[attr-defined]
        init_patched.__wrapped__ = orig_init  # type: ignore[attr-defined]
        S3Tokenizer.__init__ = init_patched  # type: ignore[assignment]

    # --- Patch log_mel_spectrogram dtype alignment at matmul ---
    if not getattr(S3Tokenizer.log_mel_spectrogram, "__patched__", False):
        orig_log_mel = S3Tokenizer.log_mel_spectrogram

        def log_mel_spectrogram_patched(self, wav: torch.Tensor) -> torch.Tensor:
            """
            Patch for MPS/device dtype consistency.
            Ensures both operands of mel_filter @ magnitudes matmul are float32.
            """
            # Ensure input wav is float32 on the right device
            wav = wav.to(device=self.device, dtype=torch.float32)

            # Ensure _mel_filters is float32 on the device (critical for MPS matmul)
            if hasattr(self, '_mel_filters') and self._mel_filters is not None:
                self._mel_filters = self._mel_filters.to(device=self.device, dtype=torch.float32)

            # Call the original method which should now have consistent dtypes
            mel = orig_log_mel(self, wav)

            # Final safety: ensure output is in acceptable dtype for downstream
            if mel.dtype not in (torch.float32, torch.float16, torch.bfloat16):
                mel = mel.to(torch.float32)

            return mel

        log_mel_spectrogram_patched.__patched__ = True  # type: ignore[attr-defined]
        log_mel_spectrogram_patched.__wrapped__ = orig_log_mel  # type: ignore[attr-defined]
        S3Tokenizer.log_mel_spectrogram = log_mel_spectrogram_patched  # type: ignore[assignment]

    # --- Patch forward() to force mel float32 just in case ---
    if not getattr(S3Tokenizer.forward, "__patched__", False):
        orig_forward = S3Tokenizer.forward

        # Some chatterbox versions compute mel inside forward(); safest is to
        # ensure the wav input is float32 too, and on the correct device.
        # For MPS, we must also ensure float32 is used throughout (no float64 support).
        def forward_entry_cast(self, wavs, max_len=None, **kwargs):
            # Convert inputs to float32 on the correct device
            # Create fresh tensors to clear any dtype metadata
            wavs2 = []
            for w in wavs:
                if torch.is_tensor(w):
                    # Create a new float32 tensor on the device to ensure no float64 references
                    w_float32 = w.to(dtype=torch.float32, device=self.device)
                    # Force data() to create a fresh tensor without any metadata
                    w_fresh = w_float32.clone().detach()
                    wavs2.append(w_fresh)
                else:
                    wavs2.append(w)

            # For MPS compatibility, ensure all parameters and buffers are float32
            # (MPS doesn't support float64)
            if hasattr(torch.backends, "mps") and "mps" in str(self.device):
                # Convert all model parameters to float32 (permanent conversion)
                for param in self.parameters():
                    if param.dtype in (torch.float64, torch.double):
                        param.data = param.to(dtype=torch.float32).data

                # Convert all buffers to float32 (permanent conversion)
                for name, buf in self.named_buffers():
                    if buf.dtype in (torch.float64, torch.double):
                        setattr(self, name, buf.to(dtype=torch.float32))

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


def _patch_watermarking() -> None:
    """
    Optionally disable Perth watermarking by patching the apply_watermark method.
    When DISABLE_WATERMARKING is True, watermark application becomes a no-op.
    """
    if not DISABLE_WATERMARKING:
        return  # Watermarking enabled, no patching needed

    # Patch all TTS/VC modules that use watermarking
    module_names = ['tts_turbo', 'tts', 'vc', 'mtl_tts']

    for module_name in module_names:
        try:
            from importlib import import_module
            mod = import_module(f'chatterbox.{module_name}')

            # Find classes that have a watermarker
            for attr_name in dir(mod):
                if attr_name.startswith('_'):
                    continue
                attr = getattr(mod, attr_name)
                if not isinstance(attr, type):
                    continue

                # Check if this class has a watermarker attribute
                if hasattr(attr, '__init__'):
                    orig_init = attr.__init__

                    def make_patched_init(original_init, class_ref):
                        def init_with_watermark_disable(self, *args, **kwargs):
                            original_init(self, *args, **kwargs)
                            # If watermarker exists, replace its apply_watermark with no-op
                            if hasattr(self, 'watermarker') and self.watermarker is not None:
                                # Store original for potential re-enabling
                                self.watermarker._original_apply_watermark = self.watermarker.apply_watermark
                                # Replace with pass-through function
                                self.watermarker.apply_watermark = lambda wav, sample_rate: wav
                        return init_with_watermark_disable

                    attr.__init__ = make_patched_init(orig_init, attr)

        except (ImportError, AttributeError):
            pass
