import uuid
import time
import threading
import logging
import subprocess
from pathlib import Path
from typing import Dict, Optional

import torch
import torchaudio as ta
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="perth")
warnings.filterwarnings("ignore", category=FutureWarning, module="diffusers.models.lora")
warnings.filterwarnings("ignore", category=FutureWarning, module="contextlib")  # torch.backends.cuda.sdp_kernel deprecation
warnings.filterwarnings("ignore", message=".*LlamaSdpaAttention.*")  # transformers attention implementation warning

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Reduce verbosity from uvicorn and other noisy libraries
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("chatterbox").setLevel(logging.WARNING)

# done to hack our way out of weird inconsistencies in these dependencies, YMMV
import monkeypatches  # noqa: E402

# Optional: disable watermarking by setting this to True before applying patches
# monkeypatches.DISABLE_WATERMARKING = True
monkeypatches.apply_chatterbox_patches()

from chatterbox.tts import ChatterboxTTS  # noqa: E402
from chatterbox.tts_turbo import ChatterboxTurboTTS  # noqa: E402

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
MODEL: Optional[object] = None  # Can be ChatterboxTTS or ChatterboxTurboTTS
CURRENT_MODEL_TYPE: Optional[str] = None
MODEL_LOCK = threading.Lock()  # protect lazy init

def get_model(model_type: str = "turbo") -> object:
    """
    Lazy load model. Unloads previous model if switching types.

    Args:
        model_type: Either "turbo" or "standard"

    Returns:
        ChatterboxTurboTTS or ChatterboxTTS instance
    """
    global MODEL, CURRENT_MODEL_TYPE

    with MODEL_LOCK:
        # If switching models, unload old one
        if MODEL is not None and CURRENT_MODEL_TYPE != model_type:
            logger.info(f"Switching from {CURRENT_MODEL_TYPE} to {model_type}, unloading old model")
            del MODEL
            MODEL = None
            # Clear GPU/NPU cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()

        # Load new model if needed
        if MODEL is None:
            logger.info(f"Loading {model_type} model on {DEVICE}")
            if model_type == "turbo":
                MODEL = ChatterboxTurboTTS.from_pretrained(device=DEVICE)
            elif model_type == "standard":
                MODEL = ChatterboxTTS.from_pretrained(device=DEVICE)
            else:
                raise ValueError(f"Invalid model_type: {model_type}. Must be 'turbo' or 'standard'")

            CURRENT_MODEL_TYPE = model_type

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

def run_job(job_id: str, text: str, upload_path: Path, model_type: str, params: dict):
    try:
        logger.debug(f"[{job_id}] Starting preprocessing")
        set_job(job_id, status="preprocessing", progress=0.1)
        upload_path = ensure_wav_16k_mono(upload_path)
        ref_path = preprocess_reference(upload_path)

        logger.info(f"[{job_id}] Starting generation with {model_type} model")
        logger.info(f"Using phrase: {text}")
        set_job(job_id, status="generating", progress=0.4)

        logger.info(f"Loading {model_type} model")
        model = get_model(model_type)

        logger.info("Starting generation")

        # Build generate() kwargs based on model type
        generate_kwargs = {
            'text': text,
            'audio_prompt_path': str(ref_path),
            'temperature': params['temperature'],
            'top_p': params['top_p'],
            'repetition_penalty': params['repetition_penalty']
        }

        # Add model-specific parameters
        if model_type == 'turbo':
            generate_kwargs['top_k'] = params['top_k']
            generate_kwargs['norm_loudness'] = params['norm_loudness']
        elif model_type == 'standard':
            generate_kwargs['min_p'] = params['min_p']
            generate_kwargs['cfg_weight'] = params['cfg_weight']
            generate_kwargs['exaggeration'] = params['exaggeration']

        wav = model.generate(**generate_kwargs)

        logger.debug(f"[{job_id}] Generation complete, saving audio")
        set_job(job_id, status="saving", progress=0.8)
        out_path = OUT_DIR / f"{job_id}.wav"
        ta.save(str(out_path), wav, model.sr)

        logger.info(f"[{job_id}] Job complete: {out_path.name}")
        set_job(job_id, status="done", progress=1.0, output=str(out_path.name))
    except Exception as e:
        logger.error(f"[{job_id}] Job failed: {e}")
        set_job(job_id, status="error", error=str(e), progress=1.0)

def validate_and_clamp_params(model_type: str, **params) -> dict:
    """
    Validate and clamp parameters based on model type.

    Shared params: temperature, top_p, repetition_penalty
    Turbo-only: top_k, norm_loudness
    Standard-only: min_p, cfg_weight, exaggeration
    """
    validated = {}

    # Common parameters (both models)
    validated['temperature'] = max(0.5, min(1.2, float(params.get('temperature', 0.8))))
    validated['repetition_penalty'] = max(1.0, min(1.5, float(params.get('repetition_penalty', 1.2))))

    if model_type == "turbo":
        # Turbo-specific parameters
        validated['top_p'] = max(0.85, min(0.98, float(params.get('top_p', 0.95))))
        validated['top_k'] = max(100, min(1000, int(params.get('top_k', 1000))))
        validated['norm_loudness'] = bool(params.get('norm_loudness', True))
    elif model_type == "standard":
        # Standard-specific parameters
        validated['top_p'] = max(0.0, min(1.0, float(params.get('top_p', 1.0))))
        validated['min_p'] = max(0.0, min(0.1, float(params.get('min_p', 0.05))))
        validated['cfg_weight'] = max(0.0, min(1.0, float(params.get('cfg_weight', 0.5))))
        validated['exaggeration'] = max(0.0, min(1.0, float(params.get('exaggeration', 0.5))))
    else:
        raise ValueError(f"Invalid model_type: {model_type}")

    return validated

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/start")
async def start(
    text: str = Form(...),
    voice_wav: UploadFile = File(...),
    model_type: str = Form("turbo"),
    # Common parameters
    temperature: float = Form(0.8),
    top_p: float = Form(0.95),
    repetition_penalty: float = Form(1.2),
    # Turbo-only parameters
    top_k: int = Form(1000),
    norm_loudness: bool = Form(True),
    # Standard-only parameters
    min_p: float = Form(0.05),
    cfg_weight: float = Form(0.5),
    exaggeration: float = Form(0.5),
):
    # Validate model_type
    if model_type not in ("turbo", "standard"):
        return JSONResponse(
            {"error": f"Invalid model_type: {model_type}. Must be 'turbo' or 'standard'"},
            status_code=400
        )

    job_id = uuid.uuid4().hex

    # Save upload
    upload_path = UPLOAD_DIR / f"{job_id}_{voice_wav.filename}"
    content = await voice_wav.read()
    upload_path.write_bytes(content)

    # Collect and validate parameters
    raw_params = {
        'temperature': temperature,
        'top_p': top_p,
        'repetition_penalty': repetition_penalty,
        'top_k': top_k,
        'norm_loudness': norm_loudness,
        'min_p': min_p,
        'cfg_weight': cfg_weight,
        'exaggeration': exaggeration,
    }
    params = validate_and_clamp_params(model_type, **raw_params)

    with JOBS_LOCK:
        JOBS[job_id] = {
            "id": job_id,
            "status": "queued",
            "progress": 0.0,
            "created_at": time.time(),
            "output": None,
            "error": None,
        }

    # Background thread with model_type
    t = threading.Thread(target=run_job, args=(job_id, text, upload_path, model_type, params), daemon=True)
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
