FROM vllm/vllm-openai:v0.18.0-cu130@sha256:9951d6e8f54921a5c80b1b106aa67d495bdaf19d139bd1a8d34d42da747df2d5

# Add FFmpeg for MP3/M4A audio format support (required by audioread/librosa)
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN uv pip install --system --require-hashes -r requirements.txt

# Copy audio preprocessing proxy
COPY audio_proxy.py /app/audio_proxy.py
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Proxy listens on 8082 by default; backend port is inferred from --port.
ENV PROXY_PORT=8082

ENTRYPOINT ["/app/entrypoint.sh"]
