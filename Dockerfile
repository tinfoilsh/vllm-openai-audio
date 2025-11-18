FROM vllm/vllm-openai:v0.11.0@sha256:014a95f21c9edf6abe0aea6b07353f96baa4ec291c427bb1176dc7c93a85845c
RUN uv pip install --system "vllm[audio]==0.11.0"
