#!/bin/bash
# Entrypoint that runs both the audio proxy and vLLM

set -e

# Infer the backend vLLM port from CLI args unless explicitly provided.
VLLM_INTERNAL_PORT="${VLLM_PORT:-}"
if [ -z "${VLLM_INTERNAL_PORT}" ]; then
  prev=""
  for arg in "$@"; do
    if [ "${prev}" = "--port" ]; then
      VLLM_INTERNAL_PORT="${arg}"
      break
    fi
    prev="${arg}"
  done
fi
VLLM_INTERNAL_PORT="${VLLM_INTERNAL_PORT:-8001}"
export VLLM_URL="${VLLM_URL:-http://127.0.0.1:${VLLM_INTERNAL_PORT}}"

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
