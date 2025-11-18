FROM vllm/vllm-openai:v0.11.0
RUN uv pip install --system "vllm[audio]==0.11.0"
