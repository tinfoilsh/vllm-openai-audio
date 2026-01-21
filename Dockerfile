FROM vllm/vllm-openai:v0.14.0@sha256:1d6866b87630d94f5e0cdae55ab5abb4ce0b03fcb84d9d10612f9d518d19d4fd

# Add FFmpeg for MP3/M4A audio format support (required by audioread/librosa)
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN uv pip install --system --require-hashes -r requirements.txt
