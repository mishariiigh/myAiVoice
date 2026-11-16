"""Subprocess bridge between the Streamlit app and the Audar inference env.

Why a subprocess: llama.cpp, torch and neucodec live in the isolated
`.venv-local` environment (they cannot be installed into the lightweight
main `.venv`, and mixing them would bloat or break the app). This module
launches `app/engines/audar_engine.py` with that environment's interpreter
and parses its `##{json}` progress lines.

Security/robustness rules:
  - No shell interpolation: the argv list is passed directly.
  - Temporary output WAVs are created in a private directory.
  - Timeouts kill the child; errors raise LocalEngineError with a
    user-safe message.
"""
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

from .. import config

ENGINE_TIMEOUT_SECONDS = 60 * 45      # long CPU generation on an old i7
DEFAULT_VENV_LOCAL = Path(__file__).resolve().parents[2] / ".venv-local"
DEFAULT_GGUF = (Path(__file__).resolve().parents[2] / "models" / "audar" /
                "Audar-TTS-V1-Flash-Q4_K_M.gguf")


class LocalEngineError(Exception):
    """Raised when the local engine cannot run or fails mid-generation.

    Messages are safe to surface in the UI: they never include absolute
    paths or stack traces.
    """


def _venv_python() -> Path:
    override = os.environ.get("AUDAR_PYTHON", "").strip()
    if override:
        return Path(override)
    return DEFAULT_VENV_LOCAL / "bin" / "python"


def _gguf_path() -> Path:
    override = os.environ.get("AUDAR_GGUF", "").strip()
    if override:
        return Path(override)
    env_dir = getattr(config, "LOCAL_VOICE_MODEL_DIR", None)
    if env_dir:
        candidate = Path(env_dir) / "Audar-TTS-V1-Flash-Q4_K_M.gguf"
        if candidate.is_file():
            return candidate
    return DEFAULT_GGUF


def engine_ready() -> bool:
    """True when the isolated interpreter, engine script and GGUF all exist."""
    return not missing_pieces()


def missing_pieces() -> List[str]:
    """Human-readable list of what is missing for local inference."""
    pieces = []
    py = _venv_python()
    if not py.is_file():
        pieces.append("the isolated inference environment (.venv-local) is missing")
    engine = Path(__file__).resolve().parent / "audar_engine.py"
    if not engine.is_file():
        pieces.append("the engine script is missing")
    gguf = _gguf_path()
    if not gguf.is_file():
        pieces.append("the Audar GGUF model file is missing")
    return pieces


def run_local_generation(text: str, ref_wav_path: str, ref_text: str,
                         temperature: float = 1.0,
                         timeout: int = ENGINE_TIMEOUT_SECONDS) -> Dict:
    """Run one full Audar generation in .venv-local and return its metadata.

    Returns: {wav_path, duration_seconds, sample_rate, ref_code_count,
              token_count}. The caller owns the returned WAV file.
    """
    if not engine_ready():
        raise LocalEngineError(
            "The local voice engine is not installed correctly. "
            "Check that .venv-local and the Audar GGUF model exist.")

    ref = Path(ref_wav_path)
    if not ref.is_file():
        raise LocalEngineError("The reference audio file could not be found.")
    if not (ref_text or "").strip():
        raise LocalEngineError(
            "The reference transcript is missing. Enter the exact words "
            "spoken in the reference recording before generating.")

    out_dir = Path(tempfile.mkdtemp(prefix="mv_audar_"))
    out_wav = out_dir / "generated.wav"
    cmd = [
        str(_venv_python()),
        str(Path(__file__).resolve().parent / "audar_engine.py"),
        "generate",
        "--text", text,
        "--ref", str(ref),
        "--ref-text", ref_text,
        "--out", str(out_wav),
        "--gguf", str(_gguf_path()),
        "--temperature", str(temperature),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout,
                              text=True)
    except subprocess.TimeoutExpired:
        raise LocalEngineError(
            "Local generation took too long and was stopped. Try a shorter "
            "script.")
    except OSError as exc:
        raise LocalEngineError(
            "Could not start the local inference environment.") from exc

    events = []
    for line in (proc.stdout or "").splitlines():
        line = line.strip()
        if line.startswith("##"):
            try:
                events.append(json.loads(line[2:]))
            except json.JSONDecodeError:
                continue

    if proc.returncode != 0 or not out_wav.is_file():
        error = next((e.get("message") for e in reversed(events)
                      if e.get("stage") == "error"), None)
        raise LocalEngineError(
            error or "The local voice engine failed. Check the engine logs "
            "in the terminal running the app.")

    done = next((e for e in reversed(events) if e.get("stage") == "done"), None)
    ref_codes = next((e.get("count") for e in events
                      if e.get("stage") == "ref_codes"), None)
    tokens = next((e.get("count") for e in events
                   if e.get("stage") == "tokens"), None)
    if not done:
        raise LocalEngineError("The local voice engine finished without "
                               "reporting a result.")
    return {
        "wav_path": done.get("wav_path") or str(out_wav),
        "duration_seconds": done.get("duration_s"),
        "sample_rate": done.get("sample_rate"),
        "ref_code_count": ref_codes,
        "token_count": tokens,
    }
