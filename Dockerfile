# CPU inference image for the /detect API.
# The model is NOT baked in: mount one seed directory (best/, config.json, threshold.json) at /model.
#
#   docker build -t medhallu-api .
#   docker run --rm -p 8000:8000 -v /path/to/seed44:/model:ro medhallu-api

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=1 \
    MODEL_DIR=/model

WORKDIR /app

# CPU-only torch is much smaller than the default CUDA build.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

COPY src/__init__.py src/predict.py src/api.py src/

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
