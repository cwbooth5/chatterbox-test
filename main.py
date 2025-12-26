import torchaudio as ta
import torch

# total frickin hack. not proud.
import monkeypatches
monkeypatches.apply_chatterbox_patches()

from chatterbox.tts_turbo import ChatterboxTurboTTS

# Load the Turbo model (CPU for now)
model = ChatterboxTurboTTS.from_pretrained(device="cpu")

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
