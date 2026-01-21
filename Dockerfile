FROM vllm/vllm-openai:v0.14.0@sha256:1d6866b87630d94f5e0cdae55ab5abb4ce0b03fcb84d9d10612f9d518d19d4fd

# Add FFmpeg for MP3/M4A audio format support (required by audioread/librosa)
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN uv pip install --system --require-hashes -r requirements.txt

# Copy audio preprocessing proxy
COPY audio_proxy.py /app/audio_proxy.py
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Proxy listens on 8082, vLLM on 8001
ENV VLLM_URL=http://127.0.0.1:8001
ENV PROXY_PORT=8082

ENTRYPOINT ["/app/entrypoint.sh"]
