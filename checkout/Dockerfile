FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
RUN pip install --no-cache-dir streamlit
COPY checkout ./checkout

EXPOSE 8501
CMD ["streamlit", "run", "checkout/app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", \
     "--server.headless=true", "--server.baseUrlPath=checkout", \
     "--browser.gatherUsageStats=false"]
