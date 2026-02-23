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

import base64
import json
import os
import re
import secrets
import shutil
import subprocess
import tempfile
from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import httpx

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)  # Disable docs endpoints

VLLM_URL = os.environ.get("VLLM_URL", "http://127.0.0.1:8001")
MAX_AUDIO_SIZE = int(os.environ.get("MAX_AUDIO_SIZE_MB", "1024")) * 1024 * 1024  # Default 1024MB
MAX_BODY_SIZE = MAX_AUDIO_SIZE  # JSON body limit (base64 audio can be large)
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


# ---------------------------------------------------------------------------
# Chat completions with audio support (Q&A, summarization, function calling)
# ---------------------------------------------------------------------------

DATA_URL_PATTERN = re.compile(r"^data:([^;]+);base64,(.+)$", re.DOTALL)

WAV_MIMES = ("audio/wav", "audio/wave", "audio/x-wav")


def _decode_and_convert(audio_bytes: bytes, ext: str, label: str) -> bytes:
    """Base64-decode is caller's job; this converts raw audio bytes to WAV."""
    print(f"[audio_proxy] Converting chat audio ({label}): {len(audio_bytes)} bytes -> WAV")
    return convert_to_wav(audio_bytes, ext)


def _convert_data_url(data_url: str) -> str:
    """Convert a ``data:mime;base64,…`` audio URL to WAV. No-op if already WAV."""
    match = DATA_URL_PATTERN.match(data_url)
    if not match:
        return data_url
    mime_type = match.group(1).lower()
    if mime_type in WAV_MIMES:
        return data_url
    audio_bytes = base64.b64decode(match.group(2))
    ext = CONTENT_TYPE_TO_EXT.get(mime_type, ".bin")
    wav_bytes = _decode_and_convert(audio_bytes, ext, mime_type)
    wav_b64 = base64.b64encode(wav_bytes).decode("ascii")
    return f"data:audio/wav;base64,{wav_b64}"


def process_messages_audio(messages: list) -> list:
    """
    Walk through chat messages and convert any embedded audio to WAV.
    Handles both Mistral (audio_url) and OpenAI (input_audio) content formats.
    """
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue

        for part in content:
            part_type = part.get("type", "")

            # Mistral / vLLM audio_url format
            if part_type == "audio_url":
                audio_url = part.get("audio_url")
                if isinstance(audio_url, str) and audio_url.startswith("data:"):
                    part["audio_url"] = _convert_data_url(audio_url)
                elif isinstance(audio_url, dict):
                    url = audio_url.get("url", "")
                    if url.startswith("data:"):
                        audio_url["url"] = _convert_data_url(url)

            # OpenAI input_audio format
            elif part_type == "input_audio":
                input_audio = part.get("input_audio", {})
                fmt = input_audio.get("format", "")
                data = input_audio.get("data", "")
                if data and fmt and fmt != "wav":
                    ext = f".{fmt}"
                    try:
                        audio_bytes = base64.b64decode(data)
                        wav_bytes = _decode_and_convert(audio_bytes, ext, f".{fmt}")
                        input_audio["data"] = base64.b64encode(wav_bytes).decode("ascii")
                        input_audio["format"] = "wav"
                    except Exception as e:
                        print(f"[audio_proxy] Failed to convert input_audio ({fmt}): {e}")

    return messages


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """
    Proxy chat completions with audio support.
    Converts any embedded audio in messages to WAV format.
    Supports both streaming (SSE) and non-streaming responses.
    """
    body_bytes = await request.body()
    if len(body_bytes) > MAX_BODY_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"Request body too large. Maximum size: {MAX_BODY_SIZE // (1024*1024)}MB"
        )
    body = json.loads(body_bytes)

    # Convert any audio in messages to WAV
    if "messages" in body:
        body["messages"] = process_messages_audio(body["messages"])

    is_streaming = body.get("stream", False)
    print(f"[audio_proxy] Chat completion: model={body.get('model')}, streaming={is_streaming}")

    if is_streaming:
        # Streaming: client lifecycle managed by the generator
        client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0))
        try:
            req = client.build_request(
                "POST",
                f"{VLLM_URL}/v1/chat/completions",
                json=body,
            )
            resp = await client.send(req, stream=True)
        except httpx.RequestError as e:
            await client.aclose()
            raise HTTPException(status_code=502, detail=f"vLLM connection error: {str(e)}")

        async def event_stream():
            try:
                async for chunk in resp.aiter_bytes():
                    yield chunk
            finally:
                await resp.aclose()
                await client.aclose()

        return StreamingResponse(
            event_stream(),
            status_code=resp.status_code,
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )
    else:
        # Non-streaming: simple request/response
        async with httpx.AsyncClient(timeout=300.0) as client:
            try:
                resp = await client.post(
                    f"{VLLM_URL}/v1/chat/completions",
                    json=body,
                )
            except httpx.RequestError as e:
                raise HTTPException(status_code=502, detail=f"vLLM connection error: {str(e)}")

        return JSONResponse(
            status_code=resp.status_code,
            content=resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {"error": resp.text}
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
