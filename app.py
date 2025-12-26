import os
import uuid
import time
import threading
import logging
from pathlib import Path
from typing import Dict, Optional
import subprocess
from pathlib import Path

import torch
import torchaudio as ta
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="perth")
warnings.filterwarnings("ignore", category=FutureWarning, module="diffusers.models.lora")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Reduce verbosity from uvicorn and other noisy libraries
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("chatterbox").setLevel(logging.WARNING)

# done to hack our way out of weird inconsistencies in these dependencies, YMMV
import monkeypatches
monkeypatches.apply_chatterbox_patches()

from chatterbox.tts_turbo import ChatterboxTurboTTS

BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
OUT_DIR = DATA_DIR / "out"
TEMPLATE_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

for d in (UPLOAD_DIR, OUT_DIR, TEMPLATE_DIR, STATIC_DIR):
    d.mkdir(parents=True, exist_ok=True)

app = FastAPI()
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

# ---- Model (load once) ----

def pick_device() -> str:
    # Apple Silicon
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    # Linux ROCm/NVIDIA both show up here
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

DEVICE = pick_device()
logger.info(f"Using device: {DEVICE}")
MODEL: Optional[ChatterboxTurboTTS] = None
MODEL_LOCK = threading.Lock()  # protect lazy init

def get_model() -> ChatterboxTurboTTS:
    global MODEL
    with MODEL_LOCK:
        if MODEL is None:
            MODEL = ChatterboxTurboTTS.from_pretrained(device=DEVICE)
        return MODEL

# ---- Job store (simple in-memory) ----
JOBS: Dict[str, Dict] = {}
JOBS_LOCK = threading.Lock()

def set_job(job_id: str, **kwargs):
    with JOBS_LOCK:
        JOBS[job_id].update(kwargs)

def get_job(job_id: str) -> Dict:
    with JOBS_LOCK:
        return JOBS.get(job_id, {})

def ensure_wav_16k_mono(in_path: Path) -> Path:
    """
    This is here because we can piss off the audio processing libs
    by feeding them audio in unexpected formats.
    """
    ext = in_path.suffix.lower()
    out_path = in_path.with_suffix(".wav")

    if ext == ".wav":
        return in_path

    # Convert anything else (m4a, mp3, etc) to wav 16k mono
    cmd = [
        "ffmpeg", "-y",
        "-i", str(in_path),
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return out_path

def preprocess_reference(in_path: Path) -> Path:
    """
    Normalize reference wav: mono, 16kHz, float32.
    This avoids dtype issues and keeps the prompt consistent.
    """
    wav, sr = ta.load(str(in_path))  # [C, T]
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = ta.functional.resample(wav, sr, 16000)
    wav = wav.to(dtype=torch.float32)

    out_path = in_path.with_name(in_path.stem + "_16k_f32.wav")
    ta.save(str(out_path), wav, 16000)
    return out_path

def run_job(job_id: str, text: str, upload_path: Path):
    try:
        logger.debug(f"[{job_id}] Starting preprocessing")
        set_job(job_id, status="preprocessing", progress=0.1)
        upload_path = ensure_wav_16k_mono(upload_path)
        ref_path = preprocess_reference(upload_path)

        logger.info(f"[{job_id}] Starting generation")
        logger.info(f"Using phrase: {text}")
        set_job(job_id, status="generating", progress=0.4)

        logger.info("setting up model")
        model = get_model()

        logger.info("starting generation")
        wav = model.generate(text, audio_prompt_path=str(ref_path), cfg_weight=0.3, exaggeration=0.8)

        logger.debug(f"[{job_id}] Generation complete, saving audio")
        set_job(job_id, status="saving", progress=0.8)
        out_path = OUT_DIR / f"{job_id}.wav"
        ta.save(str(out_path), wav, model.sr)

        logger.info(f"[{job_id}] Job complete: {out_path.name}")
        set_job(job_id, status="done", progress=1.0, output=str(out_path.name))
    except Exception as e:
        logger.error(f"[{job_id}] Job failed: {e}")
        set_job(job_id, status="error", error=str(e), progress=1.0)

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/start")
async def start(
    text: str = Form(...),
    voice_wav: UploadFile = File(...),
):
    job_id = uuid.uuid4().hex

    # Save upload
    upload_path = UPLOAD_DIR / f"{job_id}_{voice_wav.filename}"
    content = await voice_wav.read()
    upload_path.write_bytes(content)

    with JOBS_LOCK:
        JOBS[job_id] = {
            "id": job_id,
            "status": "queued",
            "progress": 0.0,
            "created_at": time.time(),
            "output": None,
            "error": None,
        }

    # Background thread
    t = threading.Thread(target=run_job, args=(job_id, text, upload_path), daemon=True)
    t.start()

    return JSONResponse({"job_id": job_id})

@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    job = get_job(job_id)
    if not job:
        return JSONResponse({"error": "job not found"}, status_code=404)
    return JSONResponse(job)

@app.get("/audio/{filename}")
def audio_file(filename: str):
    path = OUT_DIR / filename
    if not path.exists():
        return JSONResponse({"error": "file not found"}, status_code=404)
    # Serves as playable audio + downloadable file
    return FileResponse(str(path), media_type="audio/wav", filename=filename)
