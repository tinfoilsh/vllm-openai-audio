FROM vllm/vllm-openai:v0.17.0@sha256:2296a2a7e1ce1dc59c6577ba5900f4e9910b76c4a0cb134833a8137f92404dfa

# Add FFmpeg for MP3/M4A audio format support (required by audioread/librosa)
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

COPY requirements.in .
RUN uv pip install --system -r requirements.in

# Copy audio preprocessing proxy
COPY audio_proxy.py /app/audio_proxy.py
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Proxy listens on 8082 by default; backend port is inferred from --port.
ENV PROXY_PORT=8082

ENTRYPOINT ["/app/entrypoint.sh"]
