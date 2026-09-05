# The base ships a matched torch + vLLM + transformers stack. Installing vLLM
# on top of an arbitrary CUDA image is what caused the dependency failures in
# development; pinning an image where the stack is already aligned avoids it.
ARG BASE_IMAGE=vllm/vllm-openai:v0.28.0
FROM ${BASE_IMAGE}

WORKDIR /app

COPY requirements-serve.txt ./
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY src/ ./src/

# The model is mounted at runtime rather than baked in: weights are large,
# change independently of the code, and would make every image rebuild carry a
# gigabyte of unchanged tensors.
ENV MODEL_PATH=/models/lora_r32_seed0 \
    BACKEND=vllm \
    SHOTS=0 \
    MAX_TOKENS=256 \
    MAX_MODEL_LEN=2048 \
    GPU_MEMORY_UTILIZATION=0.85 \
    PYTHONUNBUFFERED=1

EXPOSE 8000

# Start-period is generous: vLLM spends tens of seconds on engine init, CUDA
# graph capture and KV cache allocation before it can serve anything.
HEALTHCHECK --interval=15s --timeout=5s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

# The base image's entrypoint launches vLLM's own OpenAI server; ours is a
# different application, so it is cleared explicitly.
ENTRYPOINT []
CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]
