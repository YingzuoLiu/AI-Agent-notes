FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY server.py store.py retrieval.py trust_graph.py audit.py ./
COPY data ./data
COPY tests ./tests

RUN pip install --no-cache-dir -e .

CMD ["python", "server.py"]
