"""Audar-TTS-V1-Flash local inference bridge — runs inside .venv-local.

This script is launched by app.engines.audar_local as a subprocess so the
main Streamlit .venv never needs torch/neucodec/llama-cpp.

Protocol (JSON on stdout, one line per stage, prefixed with '##'):
    {"stage": "ref_codes", "count": N}
    {"stage": "done", "wav_path": ..., "duration_s": ..., "sample_rate": ...}
    {"stage": "error", "message": "..."}
All other prints are considered noise and go to stderr.

Usage:
    python audar_engine.py encode --ref REF.wav --out codes.json
    python audar_engine.py generate --text T --ref REF.wav --ref-text TT \
        --out OUT.wav [--temperature 1.0] [--seed N]
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

MODEL_GGUF_DEFAULT = "models/audar/Audar-TTS-V1-Flash-Q4_K_M.gguf"
REF_SR = 16000
OUT_SR = 24000
GEN_DEFAULTS = dict(temp=1.0, top_k=40, top_p=0.9, repeat_penalty=1.1)

_SPEECH_RE = re.compile(r"<\|speech_(\d+)\|>")


def emit(payload: dict) -> None:
    print("##" + json.dumps(payload), flush=True)


def _load_codec():
    import torch
    from neucodec import NeuCodec
    codec = NeuCodec.from_pretrained("neuphonic/neucodec").eval()
    return codec


def _encode_reference(codec, ref_path: str):
    """Encode a WAV reference to FSQ codes. Resamples to 16 kHz mono if needed."""
    import torch
    import torchaudio
    y, sr = torchaudio.load(ref_path)
    if y.shape[0] > 1:
        y = y.mean(dim=0, keepdim=True)
    if sr != REF_SR:
        y = torchaudio.transforms.Resample(sr, REF_SR)(y)
    y = y[None, :]  # [B, 1, T]
    with torch.no_grad():
        codes = codec.encode_code(y)
    return [int(x) for x in codes.squeeze().tolist()]


def _build_prompt(target_text: str, ref_text: str, ref_codes) -> str:
    ref = "".join(f"<|speech_{c}|>" for c in ref_codes)
    return (
        "user: Convert the text to speech:"
        f"<|REF_TEXT_START|>{ref_text}<|REF_TEXT_END|>"
        f"<|REF_SPEECH_START|>{ref}<|REF_SPEECH_END|>"
        f"<|TARGET_TEXT_START|>{target_text}<|TARGET_TEXT_END|>"
        "\nassistant:<|TARGET_CODES_START|>"
    )


def cmd_encode(args) -> int:
    codec = _load_codec()
    codes = _encode_reference(codec, args.ref)
    Path(args.out).write_text(json.dumps(codes))
    emit({"stage": "ref_codes", "count": len(codes)})
    return 0


def cmd_generate(args) -> int:
    import torch
    import soundfile as sf
    from llama_cpp import Llama

    gguf = args.gguf or MODEL_GGUF_DEFAULT
    if not Path(gguf).is_file():
        emit({"stage": "error", "message": f"Model GGUF not found at {gguf}"})
        return 1

    t0 = time.time()
    codec = _load_codec()
    ref_codes = _encode_reference(codec, args.ref)
    emit({"stage": "ref_codes", "count": len(ref_codes)})

    llm = Llama(model_path=gguf, n_ctx=4096, n_gpu_layers=0, verbose=False)
    emit({"stage": "model_loaded", "seconds": round(time.time() - t0, 1)})

    prompt = _build_prompt(args.text, args.ref_text, ref_codes)
    tce = llm.tokenize(b"<|TARGET_CODES_END|>", add_bos=False, special=True)[0]
    toks = llm.tokenize(prompt.encode("utf-8"), add_bos=False, special=True)

    if args.seed is not None:
        llm.set_seed(args.seed)
    ids = []
    t1 = time.time()
    for tid in llm.generate(toks, temp=args.temperature, top_k=GEN_DEFAULTS["top_k"],
                            top_p=GEN_DEFAULTS["top_p"],
                            repeat_penalty=GEN_DEFAULTS["repeat_penalty"]):
        if tid == tce or len(ids) >= 2048:
            break
        ids.append(tid)
    emit({"stage": "tokens", "count": len(ids), "seconds": round(time.time() - t1, 1)})
    if not ids:
        emit({"stage": "error", "message": "Model emitted no speech tokens"})
        return 1

    text = "".join(llm.detokenize([t], special=True).decode("utf-8", "ignore") for t in ids)
    codes = [int(x) for x in _SPEECH_RE.findall(text)]
    if not codes:
        emit({"stage": "error", "message": "No recognizable speech codes in model output"})
        return 1
    emit({"stage": "speech_codes", "count": len(codes)})

    with torch.no_grad():
        wav = codec.decode_code(torch.tensor(codes)[None, None, :])
    audio = wav[0, 0, :].detach().cpu().numpy()
    sf.write(args.out, audio, OUT_SR)
    info = sf.info(args.out)
    emit({"stage": "done", "wav_path": args.out,
          "duration_s": round(info.duration, 2), "sample_rate": info.samplerate})
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("encode")
    e.add_argument("--ref", required=True)
    e.add_argument("--out", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--text", required=True)
    g.add_argument("--ref", required=True)
    g.add_argument("--ref-text", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--gguf", default=None)
    g.add_argument("--temperature", type=float, default=1.0)
    g.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    return cmd_encode(args) if args.cmd == "encode" else cmd_generate(args)


if __name__ == "__main__":
    sys.exit(main())
