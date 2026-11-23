FROM python:3.12-slim

WORKDIR /app

# No third-party dependencies: the project runs on the standard library alone.
COPY src/ ./src/
COPY evals/ ./evals/
COPY scripts/ ./scripts/
COPY corpus/ ./corpus/

ENV PYTHONUNBUFFERED=1
ENV PORT=8000
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["python", "scripts/serve.py"]
