FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TORCH_HOME=/opt/torch

WORKDIR /app

# CPU-only torch -- the default wheel drags in several GB of CUDA libraries
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir streamlit numpy pillow

COPY localization ./localization

# Bake the DINOv2 code + weights into the image so the first request isn't a download
RUN cd localization && python -c "import dino; dino.get_model('dinov2_vits14', 'cpu')"

EXPOSE 8501
CMD ["streamlit", "run", "localization/app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]
