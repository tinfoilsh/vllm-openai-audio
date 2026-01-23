#!/usr/bin/env python3
"""
Audio preprocessing proxy for vLLM.
Converts any audio format to WAV before forwarding to vLLM.
This works around vLLM's M4A handling bug (#26808).

Supported formats (via FFmpeg):
- Common: MP3, M4A, AAC, OGG, Opus, FLAC, WAV, WebM, WMA
- Apple: AIFF, CAF, M4A, M4B, M4R (voice memos, ringtones)
- Mobile: AMR, 3GP, 3G2
- Lossless: FLAC, ALAC, APE, WavPack
- Legacy: AU, RA, SND

All audio is normalized to 16kHz mono WAV for optimal Voxtral processing.

Security:
- Stateless: No user data persisted between requests
- Temp files: Created with restricted permissions, cleaned up immediately
- No logging of user content or filenames
"""

import os
import secrets
import shutil
import subprocess
import tempfile
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import httpx

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)  # Disable docs endpoints

VLLM_URL = os.environ.get("VLLM_URL", "http://127.0.0.1:8001")
MAX_AUDIO_SIZE = int(os.environ.get("MAX_AUDIO_SIZE_MB", "1024")) * 1024 * 1024  # Default 1024MB
CHUNK_SIZE = 64 * 1024  # 64KB chunks for streaming reads


async def read_upload_with_limit(file: UploadFile, max_size: int) -> bytes:
    """
    Read an uploaded file with size limit enforcement during streaming.
    Prevents memory exhaustion by checking size as we read.
    """
    chunks = []
    total_size = 0

    while True:
        chunk = await file.read(CHUNK_SIZE)
        if not chunk:
            break
        total_size += len(chunk)
        if total_size > max_size:
            raise HTTPException(
                status_code=413,
                detail=f"Audio file too large. Maximum size: {max_size // (1024*1024)}MB"
            )
        chunks.append(chunk)

    return b"".join(chunks)

# All audio formats FFmpeg can typically decode
SUPPORTED_FORMATS = {
    # Common formats
    ".wav", ".mp3", ".m4a", ".mp4", ".aac",
    ".ogg", ".oga", ".opus", ".flac", ".webm",
    ".wma", ".wmv",
    # Apple formats
    ".aiff", ".aif", ".aifc", ".caf",
    # Mobile/telephony formats
    ".amr", ".3gp", ".3gpp", ".3g2",
    # Other compressed formats
    ".spx", ".ape", ".wv", ".mka",
    # Raw/other
    ".au", ".snd", ".ra", ".ram",
    ".mpeg", ".mpga", ".mpg",
    # Voice memo formats
    ".m4b", ".m4p", ".m4r",
}


def convert_to_wav(input_bytes: bytes, input_format: str) -> bytes:
    """
    Convert any audio format to 16kHz mono WAV using FFmpeg.
    
    Security: Uses a private temp directory (0o700) so all files inside are
    protected by directory permissions - no race condition window.
    """
    # Create a private temp directory
    tmp_dir = tempfile.mkdtemp(prefix="audio_")
    random_id = secrets.token_hex(16)
    inp_path = os.path.join(tmp_dir, f"in_{random_id}{input_format}")
    out_path = os.path.join(tmp_dir, f"out_{random_id}.wav")

    try:
        # Write input file (directory permissions protect it)
        with open(inp_path, "wb") as f:
            f.write(input_bytes)

        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", inp_path,
                "-vn",              # no video
                "-ac", "1",         # mono
                "-ar", "16000",     # 16kHz (Voxtral's expected rate)
                "-c:a", "pcm_s16le",  # 16-bit PCM
                out_path
            ],
            capture_output=True,
            check=True
        )

        with open(out_path, "rb") as f:
            wav_bytes = f.read()
        
        print(f"[audio_proxy] Converted {len(input_bytes)} bytes ({input_format}) -> {len(wav_bytes)} bytes (.wav)")
        return wav_bytes
        
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode() if e.stderr else "Unknown error"
        print(f"[audio_proxy] FFmpeg conversion failed for {input_format}: {error_msg}")
        raise HTTPException(
            status_code=400,
            detail=f"Audio conversion failed for format '{input_format}': {error_msg}"
        )
    finally:
        # Clean up entire temp directory
        try:
            shutil.rmtree(tmp_dir)
        except OSError:
            pass


# Map content-types to file extensions for format detection
CONTENT_TYPE_TO_EXT = {
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/mp4": ".m4a",
    "audio/x-m4a": ".m4a",
    "audio/aac": ".aac",
    "audio/ogg": ".ogg",
    "audio/opus": ".opus",
    "audio/flac": ".flac",
    "audio/x-flac": ".flac",
    "audio/wav": ".wav",
    "audio/wave": ".wav",
    "audio/x-wav": ".wav",
    "audio/webm": ".webm",
    "video/webm": ".webm",
    "audio/x-ms-wma": ".wma",
    "audio/aiff": ".aiff",
    "audio/x-aiff": ".aiff",
    "audio/x-caf": ".caf",
    "audio/amr": ".amr",
    "audio/3gpp": ".3gp",
    "audio/3gpp2": ".3g2",
    "audio/basic": ".au",
    "audio/x-realaudio": ".ra",
}


def get_extension(filename: str) -> str:
    """Get lowercase file extension."""
    if filename and "." in filename:
        return "." + filename.rsplit(".", 1)[-1].lower()
    return ""


def detect_format(filename: str, content_type: str) -> str:
    """
    Detect audio format from filename extension or content-type.
    Returns extension like '.mp3', '.m4a', etc.
    Falls back to '.bin' if unknown (FFmpeg will try to auto-detect).
    """
    # Try filename extension first
    ext = get_extension(filename)
    if ext and ext in SUPPORTED_FORMATS:
        return ext
    
    # Try content-type mapping
    if content_type:
        ct = content_type.lower().split(";")[0].strip()
        if ct in CONTENT_TYPE_TO_EXT:
            return CONTENT_TYPE_TO_EXT[ct]
    
    # Fall back - FFmpeg can often auto-detect from file contents
    return ext if ext else ".bin"


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile = File(...),
    model: str = Form(...),
    language: str = Form(None),
    prompt: str = Form(None),
    response_format: str = Form(None),
    temperature: float = Form(None),
):
    """
    Proxy endpoint that converts audio to WAV before forwarding to vLLM.
    Supports all audio formats that FFmpeg can decode.
    """
    content = await read_upload_with_limit(file, MAX_AUDIO_SIZE)
    ext = detect_format(file.filename or "", file.content_type or "")
    print(f"[audio_proxy] Transcription: {len(content)} bytes, format={ext}")

    # Convert all audio to normalized 16kHz mono WAV
    content = convert_to_wav(content, ext)

    # Build form data for vLLM
    files = {"file": ("audio.wav", content, "audio/wav")}
    data = {"model": model}
    
    if language and language != "auto":
        data["language"] = language
    if prompt:
        data["prompt"] = prompt
    if response_format:
        data["response_format"] = response_format
    if temperature is not None:
        data["temperature"] = str(temperature)

    # Forward to vLLM
    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            resp = await client.post(
                f"{VLLM_URL}/v1/audio/transcriptions",
                files=files,
                data=data
            )
        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=f"vLLM connection error: {str(e)}")

    # Return vLLM's response directly
    return JSONResponse(
        status_code=resp.status_code,
        content=resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {"text": resp.text}
    )


@app.post("/v1/audio/translations")
async def translations(
    file: UploadFile = File(...),
    model: str = Form(...),
    prompt: str = Form(None),
    response_format: str = Form(None),
    temperature: float = Form(None),
):
    """
    Proxy endpoint for audio translations (translate audio to English text).
    Converts audio to WAV before forwarding to vLLM.
    """
    content = await read_upload_with_limit(file, MAX_AUDIO_SIZE)
    ext = detect_format(file.filename or "", file.content_type or "")
    print(f"[audio_proxy] Translation: {len(content)} bytes, format={ext}")

    # Convert all audio to normalized 16kHz mono WAV
    content = convert_to_wav(content, ext)

    # Build form data for vLLM
    files = {"file": ("audio.wav", content, "audio/wav")}
    data = {"model": model}
    
    if prompt:
        data["prompt"] = prompt
    if response_format:
        data["response_format"] = response_format
    if temperature is not None:
        data["temperature"] = str(temperature)

    # Forward to vLLM
    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            resp = await client.post(
                f"{VLLM_URL}/v1/audio/translations",
                files=files,
                data=data
            )
        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=f"vLLM connection error: {str(e)}")

    # Return vLLM's response directly
    return JSONResponse(
        status_code=resp.status_code,
        content=resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {"text": resp.text}
    )


@app.get("/health")
async def health():
    """Forward health check to vLLM."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get(f"{VLLM_URL}/health")
            return JSONResponse(status_code=resp.status_code, content=resp.json() if resp.text else {})
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"vLLM health check failed: {str(e)}")


@app.get("/metrics")
async def metrics():
    """Forward metrics to vLLM."""
    from fastapi.responses import PlainTextResponse
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.get(f"{VLLM_URL}/metrics")
            return PlainTextResponse(content=resp.text, status_code=resp.status_code)
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"vLLM metrics failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PROXY_PORT", "8082"))
    uvicorn.run(app, host="0.0.0.0", port=port)
