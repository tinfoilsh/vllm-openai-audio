#!/bin/bash
# Entrypoint that runs both the audio proxy and vLLM

set -e

# Start the audio preprocessing proxy in the background
echo "Starting audio preprocessing proxy on port ${PROXY_PORT:-8082}..."
python3 /app/audio_proxy.py &
PROXY_PID=$!

# Give the proxy a moment to start
sleep 2

# Execute vLLM with all passed arguments
# vLLM will listen on port 8001 by default (set via args)
echo "Starting vLLM..."
exec "$@"
