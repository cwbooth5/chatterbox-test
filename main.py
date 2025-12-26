import torchaudio as ta
import torch

# total frickin hack. not proud.
import monkeypatches
monkeypatches.apply_chatterbox_patches()

from chatterbox.tts_turbo import ChatterboxTurboTTS

def pick_device() -> str:
    # Apple Silicon
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    # Linux ROCm/NVIDIA both show up here
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

DEVICE = pick_device()

# Load the Turbo model (CPU for now)
model = ChatterboxTurboTTS.from_pretrained(device=DEVICE)

# For MPS compatibility, convert all float64 to float32 (MPS doesn't support float64)
if DEVICE == "mps":
    def convert_to_float32(obj):
        """Recursively convert all float64 tensors to float32"""
        if isinstance(obj, torch.nn.Module):
            for param in obj.parameters():
                if param.dtype == torch.float64:
                    param.data = param.data.to(torch.float32)
            for buf in obj.buffers():
                if buf.dtype == torch.float64:
                    buf.copy_(buf.to(torch.float32))
            for child in obj.children():
                convert_to_float32(child)
        elif isinstance(obj, torch.Tensor):
            if obj.dtype == torch.float64:
                return obj.to(torch.float32)
        else:
            # For objects with __dict__, convert tensor attributes
            for attr_name in dir(obj):
                if attr_name.startswith('_'):
                    continue
                try:
                    attr = getattr(obj, attr_name)
                    if isinstance(attr, torch.nn.Module):
                        convert_to_float32(attr)
                    elif isinstance(attr, torch.Tensor) and attr.dtype == torch.float64:
                        setattr(obj, attr_name, attr.to(torch.float32))
                except (AttributeError, TypeError):
                    continue

    convert_to_float32(model)

# Text to synthesize
text = (
    "Everything's going to have a common theme, that it plays into the worst of our biases and beliefs."
    "It's like we used to spend our days outside, with real people, or alone at first. Newspapers and news networks had the information,"
    "they had professional editors. Then we got the phones and became chronically online. We made the content for one another."
    "You were the only editor required and could make up claims or invent information. Now we're firmly into the world where the GPUs make content for us."
    "They are fundamentally designed to hallucinate, with a slight basis in reality."
)

# this would have to be the sample of a voice to clone
ref_wav, ref_sr = ta.load("reading-text-voice.wav")  # shape: [C, T]

# Convert to mono if needed
if ref_wav.shape[0] > 1:
    ref_wav = ref_wav.mean(dim=0, keepdim=True)

# Resample to 16 kHz (Chatterbox expects this)
if ref_sr != 16000:
    ref_wav = ta.functional.resample(ref_wav, ref_sr, 16000)

# Force float32 (this fixes the Double vs Float crash)
ref_wav = ref_wav.to(dtype=torch.float32)

# Save a normalized reference clip
ref_path = "reading-text-voice_16k_f32.wav"
ta.save(ref_path, ref_wav, 16000)

# --- GENERATE ---
wav = model.generate(text, audio_prompt_path=ref_path)

# Save output
ta.save("test-turbo.wav", wav, model.sr)
