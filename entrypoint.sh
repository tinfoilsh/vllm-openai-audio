#!/bin/bash
# Entrypoint that runs both the audio proxy and vLLM

set -e

# Start the audio preprocessing proxy in the background
# Use setsid to create new session, preventing terminal signals from stopping FFmpeg
echo "Starting audio preprocessing proxy on port ${PROXY_PORT:-8082}..."
setsid python3 /app/audio_proxy.py &
PROXY_PID=$!

# Give the proxy a moment to start
sleep 2

# Execute vLLM with all passed arguments (mimics vLLM's ENTRYPOINT ["vllm" "serve"])
echo "Starting vLLM..."
exec vllm serve "$@"
